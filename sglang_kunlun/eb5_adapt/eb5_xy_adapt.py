"""xy fork 专属的 EB 适配（从 c11d9c8 / d86dfb7 迁入）。

与另外几个 ``eb5_*_adapt.py`` 同级、扁平存放：

* ERNIE 5.5 Flash 的模型级算子（原先注册在 ``kernels/kernel_ops.py``）
* DP 空转 / 短 prompt 的 zero-length kernel 守卫（原先是两个 ``@plugin_hook``）

``eb5_kunlun_adapt.apply()`` 不直接引用本模块；由
``sglang_kunlun.eb5_adapt.eb5_kunlun_patch`` 在通用适配之后调用 ``apply()``，
这样本文件可以独立增删，不影响上游同步。
"""

from __future__ import annotations

import torch
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

try:  # 包内导入
    from .eb5_extra_adapt import _patch_everywhere
except ImportError:  # EB5_KUNLUN_ADAPT_DIR 平铺目录
    from eb5_extra_adapt import _patch_everywhere

# --------------------------------------------------------------------------- #
# xy fork：ERNIE 5.5 Flash 模型级算子（原先散在 kernels/kernel_ops.py）
# --------------------------------------------------------------------------- #
def sigmoid_gate_mul_torch(x, gate, *, round_sigmoid: bool = False):
    """``x * sigmoid(gate)``，可选先对 sigmoid 结果做 dtype 舍入。"""
    s = torch.sigmoid(gate.to(torch.float32))
    if round_sigmoid:
        s = s.to(gate.dtype).to(torch.float32)
    return (x.to(torch.float32) * s).to(x.dtype)


def sigmoid_gate_mul_broadcast_torch(x, gate, *, round_sigmoid: bool = False):
    """与 :func:`sigmoid_gate_mul_torch` 同语义的 broadcast 变体。"""
    s = torch.sigmoid(gate.to(torch.float32))
    if round_sigmoid:
        s = s.to(gate.dtype).to(torch.float32)
    return (x.to(torch.float32) * s).to(x.dtype)


def fused_sigmoid_mul_torch(
    attn_output,
    gate,
    inplace: bool = False,
    quant: bool = False,
    round_sigmoid: bool = False,
):
    """``sglang.kernels.ops.elementwise.elementwise.fused_sigmoid_mul`` 的 torch 版。

    开源树把 EB5 的 attention 输出门控换成了这个带 ``inplace`` / ``quant`` /
    3D strided gate 的融合 triton kernel；triton 在 XPU 上被 noop 掉，
    ``_fused_sigmoid_mul_kernel[grid]`` 直接报 "function object is not
    subscriptable"。语义照抄 kernel：
    ``out = x * round?(sigmoid(gate))``，``quant`` 时返回按行 amax/448 的
    fp8_e4m3 与 ``[T, 1]`` scale。
    """
    g = gate
    if g.ndim == 3 and attn_output.ndim == 2:
        g = g.reshape(g.shape[0], -1) if g.is_contiguous() else g.flatten(1)
    elif g.shape != attn_output.shape:
        g = g.reshape(attn_output.shape)

    s = torch.sigmoid(g.to(torch.float32))
    if round_sigmoid:
        s = s.to(g.dtype).to(torch.float32)
    result = attn_output.to(torch.float32) * s

    if quant:
        num_tokens = attn_output.numel() // attn_output.shape[-1]
        flat = result.reshape(num_tokens, -1)
        scale = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10) / 448.0
        q = (flat / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return q.reshape(attn_output.shape), scale

    out = result.to(attn_output.dtype)
    if inplace:
        attn_output.copy_(out)
        return attn_output
    return out


def multi_router_topk_torch(
    logits,
    correction_bias,
    per_expert_scale,
    topk: int,
    renormalize: bool,
    *,
    num_routers: int,
):
    """``sglang.kernels.ops.moe.multi_router.multi_router_topk`` 的 torch 版。

    照抄 ``_multi_router_topk_kernel``：R 个 router 对同一批 expert 打分、
    sigmoid 相加；``scores + bias`` 取 TopK（greedy argmax，并列取小 id）；
    权重取**不含 bias** 的 scores，renormalize 后乘 per-expert scale。
    """
    tokens, width = logits.shape
    num_experts = width // num_routers
    scores = torch.sigmoid(logits.float().view(tokens, num_routers, num_experts)).sum(1)
    biased = scores + correction_bias.float()
    ids = torch.topk(biased, topk, dim=-1).indices
    weights = scores.gather(1, ids)
    if renormalize:
        denom = weights.sum(-1, keepdim=True)
        weights = weights / torch.where(denom != 0, denom, torch.ones_like(denom))
    weights = weights * per_expert_scale.float()[ids]
    return weights.contiguous(), ids.to(torch.int32).contiguous()


def postmix_fp8_wo_a_quant_torch(x, mix):
    """``postmix_fp8_wo_a.postmix_fp8_wo_a_quant`` 的 torch 版。

    与适配层已有的 ``postmix_quant`` 同一实现，只是开源树的入口不再传
    ``num_heads``（从 ``mix`` 的行数推）。
    """
    try:
        from sglang_kunlun.eb5_adapt.eb5_extra_adapt import postmix_quant
    except ImportError:  # EB5_KUNLUN_ADAPT_DIR 平铺目录
        from eb5_extra_adapt import postmix_quant

    return postmix_quant(x, mix, mix.shape[0])


def multimax_torch(x, ranges, ts):
    """ERNIE 5.5 Flash multimax：中间量按 bf16 舍入（与 kernel 逐位一致）。"""

    def rb(v):
        return v.to(torch.bfloat16).float()

    x32 = x.to(torch.float32)
    r0, r1, r2, r3 = [ranges[i].float() for i in range(4)]
    t0, t1, t2, t3 = [ts[i].float() for i in range(4)]
    lower = rb(torch.clamp(r0 - x32, min=0))
    upper = rb(torch.clamp(x32 - r1, min=0))
    lower_sq = rb(torch.clamp(r2 - x32, min=0))
    upper_sq = rb(torch.clamp(x32 - r3, min=0))
    out = rb(x32 + rb(t0 * lower))
    out = rb(out + rb(t1 * upper))
    out = rb(out + rb(t2 * rb(lower_sq * lower_sq)))
    out = rb(out + rb(t3 * rb(upper_sq * upper_sq)))
    return out.to(torch.bfloat16)


def _install_ernie5_ops() -> list:
    """把 ERNIE 5.5 的模型级算子显式 patch 到 fork 的 JIT 辅助函数上。

    这些实现原先由 ``kernels/kernel_ops.py`` 的 ``register_jit_op`` 注册、
    再由 ``kernel_ops.install()`` 统一落地。迁到适配层后不再依赖插件 hook 的
    装载顺序，EB5 插件一加载就生效。
    """
    patched = []
    for module_name, attr, impl in (
        (
            "sglang.kernels.ops.moe.triton_sigmoid_gate_mul",
            "sigmoid_gate_mul",
            sigmoid_gate_mul_torch,
        ),
        (
            "sglang.kernels.ops.moe.triton_sigmoid_gate_mul",
            "sigmoid_gate_mul_broadcast",
            sigmoid_gate_mul_broadcast_torch,
        ),
        ("sglang.kernels.ops.activation.multimax", "multimax", multimax_torch),
        (
            "sglang.kernels.ops.elementwise.elementwise",
            "fused_sigmoid_mul",
            fused_sigmoid_mul_torch,
        ),
        (
            "sglang.kernels.ops.moe.multi_router",
            "multi_router_topk",
            multi_router_topk_torch,
        ),
        (
            "sglang.kernels.ops.attention.dsv4.postmix_fp8_wo_a",
            "postmix_fp8_wo_a_quant",
            postmix_fp8_wo_a_quant_torch,
        ),
    ):
        try:
            patched.extend(_patch_everywhere(module_name, attr, impl))
        except Exception as exc:  # noqa: BLE001
            print(f"[eb5-extra-adapt] {module_name}.{attr} failed: {exc}")
    return patched


# --------------------------------------------------------------------------- #
# xy fork：DP 空转 / 短 prompt 的 zero-length kernel 守卫
# （原先是 hooks/layers 下的两个 @plugin_hook）
# --------------------------------------------------------------------------- #
def _make_embed_zero_guard(original):
    def _guard(self, input_):
        if input_.numel() == 0:
            return self.weight.new_empty(
                tuple(input_.shape) + (self.weight.shape[1],)
            )
        return original(self, input_)

    return _guard


def _make_compress_zero_guard(original):
    def _guard(self, compress_ratio, page_table, c_page_size):
        if compress_ratio in self.compressed:
            return original(self, compress_ratio, page_table, c_page_size)

        seq_lens_cpu = self.seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = self.seq_lens.tolist()
        if any(int(length) // compress_ratio for length in seq_lens_cpu):
            return original(self, compress_ratio, page_table, c_page_size)

        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            CompressedGather,
        )

        device = self.seq_lens.device
        zeros = torch.zeros(self.num_reqs, dtype=torch.int32, device=device)
        gather = CompressedGather(
            flat_token_ids=torch.empty(0, dtype=torch.int32, device=device),
            compressed_base=zeros,
            swa_base=0 + self.swa_offsets[:-1],
        )
        self.compressed[compress_ratio] = gather
        return gather

    return _guard


def _install_dp_idle_guards() -> list:
    """XPU 内核拒绝 zero-element launch，给两处空 batch 路径加短路。"""
    import importlib
    import sys

    patched = []
    targets = (
        (
            "sglang.srt.layers.vocab_parallel_embedding",
            "VocabParallelEmbedding",
            "_eb5_embed_zero_guard",
            _make_embed_zero_guard,
        ),
        (
            "sglang.srt.layers.attention.dsv4.sparse_prefill_utils",
            "SparsePrefillChunkCache",
            "_eb5_compress_zero_guard",
            _make_compress_zero_guard,
        ),
    )
    for module_path, cls_name, flag, factory in targets:
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[eb5-extra-adapt] import {module_path} failed: {exc}")
            continue
        # compat_aliases 可能留下同一文件的多个模块副本，逐个按类对象打。
        for mod_name, module in list(sys.modules.items()):
            if module is None or not mod_name.startswith("sglang."):
                continue
            cls = getattr(module, cls_name, None)
            if not isinstance(cls, type) or getattr(cls, flag, False):
                continue
            original = getattr(cls, "_embed_local_shard", None) if cls_name == (
                "VocabParallelEmbedding"
            ) else getattr(cls, "ensure_compressed", None)
            if original is None:
                continue
            target_attr = (
                "_embed_local_shard"
                if cls_name == "VocabParallelEmbedding"
                else "ensure_compressed"
            )
            setattr(cls, target_attr, factory(original))
            setattr(cls, flag, True)
            patched.append(f"{mod_name}.{cls_name}.{target_attr}")

    return patched


def apply() -> list:
    """应用 xy fork 专属适配，返回已 patch 的符号列表。"""
    patched = []
    for hook in (_install_ernie5_ops, _install_dp_idle_guards):
        try:
            patched.extend(hook())
        except Exception as exc:  # noqa: BLE001
            print(f"[eb5-xy-adapt] {hook.__name__} failed: {exc}")
    return patched


@plugin_hook(
    "sglang.srt.model_executor.model_runner.ModelRunner.__init__",
    type=HookType.AROUND,
)
def _reinstall_kernel_ops_around_model_init(original_fn, self, *args, **kwargs):
    """Re-apply the kernel replacements around ``ModelRunner.__init__``.

    ``ModelRunner.__init__`` imports the model module (binding the upstream
    ``fused_q_norm_rope`` & co. into its globals) **and** captures the decode
    CUDA graphs at the end.  The plugin's first ``install()`` therefore ran
    before those bindings existed, and repairing only afterwards is already
    too late for the capture.  Re-install on both sides: the sweep rewrites
    every already-imported alias, and the second pass catches modules imported
    while the runner was being built.
    """
    _reinstall_kernel_ops("before ModelRunner.__init__")
    ret = original_fn(self, *args, **kwargs)
    _reinstall_kernel_ops("after ModelRunner.__init__")
    return ret


_MODEL_MODULES = ("sglang.srt.models.deepseek_v4",)


def _reinstall_kernel_ops(stage: str) -> None:
    try:
        import importlib
        import sys

        from sglang_kunlun.eb5_adapt.kernels import kernel_ops as _kops

        _kops.install()
        # The model module binds the upstream ``fused_q_norm_rope`` & co. with
        # ``from ... import f`` at import time; if it is not imported yet the
        # sweep above cannot see those aliases, so pull it in and sweep again.
        newly_imported = []
        for name in _MODEL_MODULES:
            if name in sys.modules:
                continue
            try:
                importlib.import_module(name)
                newly_imported.append(name)
            except Exception as exc:  # noqa: BLE001
                print(f"[eb5-xy-adapt] import {name} failed: {exc}")
        if newly_imported:
            _kops.install()
        bad = getattr(_kops, "_FAILED_PATCHES", [])
        if bad:
            print(
                f"[kernel-ops/{stage}] {len(bad)} failed patches: {bad[:5]}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[eb5-xy-adapt] kernel_ops re-install ({stage}) failed: {exc}")
