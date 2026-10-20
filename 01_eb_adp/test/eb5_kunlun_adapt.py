"""EB5-lite 算子在昆仑上的适配层 —— 参考 dpsk-v4 的「融合算子拆解」做法。

EB5 单测**直接调用**的这批 kernel 在昆仑上跑不了（tilelang / deep_gemm / fork 自带
nvcc-JIT / triton）。dpsk-v4 的做法是：把融合 kernel 换成**纯 torch 拆解实现**（或
KLX 原生算子）。本模块照此把 EB5 的融合 kernel 换成 torch 实现，语义与 kernel 源码
逐行对应，供 `code/eb5_layer` 的 kernel 级单测和后续组网使用。

被替换的 kernel 与拆解：

* ``mhc_pre`` / ``mhc_post``
      整体换成与 DSV4 同构的散算子实现（`mhc_pre_torch_ref` / `mhc_post_torch_ref`）：
      GEMM → split-sinkhorn → combine → layernorm，不经过任何融合 kernel。
      2026-10-20 实测 EB5 的融合路径（`mhc_post_tilelang` / `mhc_pre_big_fuse_*` /
      split-K / tf32）在真实多 rank 前向挂死，这些替代实现已删除，只保留这一条通路。
"""

from __future__ import annotations

import os

import torch

_LOG2E = 1.4426950408889634


def _mhc_trace(tag: str) -> None:
    """``EB5_MHC_TRACE=1`` 时在每个子步骤后同步并打点，用来把挂死定位到具体 op。

    只用于定位：同步会打断异步流水，正常跑不要开。
    """
    if os.environ.get("EB5_MHC_TRACE", "0") == "1":
        torch.cuda.synchronize()
        rank = -1
        try:
            import torch.distributed as _dist

            if _dist.is_initialized():
                rank = _dist.get_rank()
        except Exception:
            pass
        print(f"[eb5-mhc-trace][r{rank}] {tag}", flush=True)


def _dsv4_torch_shim_path() -> str:
    """定位插件的 ``dsv4_torch_shim.py``（不依赖工作区落在哪个绝对路径）。

    优先按已安装的 ``sglang_kunlun`` 包位置解析；解析不到再退回旧工作区的默认路径。
    """
    import importlib.util
    import os

    try:
        spec = importlib.util.find_spec("sglang_kunlun")
    except Exception:  # noqa: BLE001 - 解析失败就退回默认路径
        spec = None
    if spec is not None and spec.origin:
        candidate = os.path.join(
            os.path.dirname(spec.origin), "kernels", "dsv4_torch_shim.py"
        )
        if os.path.exists(candidate):
            return candidate
    return "/root/paddlejob/bingoo/code/sglang-kunlun/sglang_kunlun/kernels/dsv4_torch_shim.py"





def mhc_post_torch_ref(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
):
    """``sglang.kernels.ops.layernorm.mhc.mhc_post`` 的**散装 torch 实现**。

    与 DSV4 在昆仑上跑的 ``hc_post_torch_impl``（`deepseek_v4.py:2739`）逐行同式：

        out[n,hco,h] = post[n,hco] * x[n,h] + Σ_hci comb[n,hci,hco] * residual[n,hci,h]

    ``post_layer_mix`` 允许 ``(n, hc, 1)``（EB5 的调用形状）或 ``(n, hc)``。
    由 `apply()` 无条件 patch，与 `mhc_pre_torch_ref` 配套，完全绕开融合 kernel。
    """
    if post_layer_mix.dim() == residual.dim():   # (n, hc, 1)
        post = post_layer_mix
    else:                                        # (n, hc)
        post = post_layer_mix.unsqueeze(-1)
    # 与 DSV4 的 hc_post_torch_impl 完全一致。
    out = post * x.unsqueeze(1) + (
        comb_res_mix.unsqueeze(-1) * residual.unsqueeze(2)
    ).sum(dim=1)
    _mhc_trace("post:out")
    return out.type_as(x)


def _rmsnorm_torch(x, weight, eps):
    """``input_layernorm`` 的显式写法：fp32 累加 + fp32 权重 + 输出回原 dtype。

    与 ``EB5LayerRMSNorm.forward``（底层 ``F.rms_norm``）和 mHC 融合 kernel 的
    norm 段一致。
    """
    xf = x.float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (xf * rsqrt * weight.float()).to(x.dtype)


def _hc_split_sinkhorn_torch(
    mixes,
    hc_scale,
    hc_base,
    hc_mult,
    sinkhorn_iters,
    eps,
    mhc_align_fleet=False,
    post_mult_value=2.0,
):
    """``hc_split_sinkhorn`` 的 torch 版（DSV4 在昆仑上由插件 hook 到同语义实现）。

    与 `mhc.py::hc_split_sinkhorn_kernel` / `_hc_split_sinkhorn_torch` 的 sinkhorn
    段逐行对应：第一轮 ``comb/row_sum + eps``，之后每轮 row/col 都是 ``/(sum + eps)``。
    """
    flat = mixes.float()
    hc = int(hc_mult)

    pre = torch.sigmoid(flat[:, :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = post_mult_value * torch.sigmoid(
        flat[:, hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    comb = (flat[:, 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).reshape(-1, hc, hc)
    _mhc_trace("sinkhorn:mixes")

    row_max = comb.amax(dim=2, keepdim=True)
    if mhc_align_fleet:
        comb = torch.exp2((comb - row_max) * _LOG2E)
    else:
        comb = torch.exp(comb - row_max)
    _mhc_trace("sinkhorn:softmax")
    comb = comb / comb.sum(dim=2, keepdim=True) + eps
    _mhc_trace("sinkhorn:norm0-dim2")
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    _mhc_trace("sinkhorn:norm0-dim1")
    for _it in range(max(int(sinkhorn_iters) - 1, 0)):
        comb = comb / (comb.sum(dim=2, keepdim=True) + eps)
        _mhc_trace(f"sinkhorn:iter{_it}-dim2")
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
        _mhc_trace(f"sinkhorn:iter{_it}-dim1")

    return pre, post, comb


def mhc_pre_torch_ref(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits: int = 1,
    n_splits_pre: int = 32,
    *,
    norm_weight=None,
    norm_eps=None,
    mhc_align_fleet: bool = False,
    mhc_align_accum_order: bool = False,
):
    """``sglang.kernels.ops.layernorm.mhc.mhc_pre`` 的**散装 torch 实现**。

    刻意照抄 DSV4 在昆仑上已经跑通的那条路（`deepseek_v4.py::hc_pre` 的 torch 分支），
    四步一一对应，不调用任何融合 kernel：

        1. hc_pre_torch_impl : mixes = F.linear(x_flat, fn) * rsqrt(mean(x²)+eps)
        2. hc_split_sinkhorn: pre / post / comb
        3. hc_combine       : y = Σ_hc pre * residual
        4. input_layernorm  : y = rmsnorm(y, norm_weight, norm_eps)

    DSV4 第 4 步在模型层单独调，EB5 的 ``mhc_pre`` 契约是把 norm 折进来，所以这里
    只是把同一公式放在同一函数里（与融合 kernel 的 norm 段一致：先 round 到 bf16，
    再 fp32 平方 / 乘 fp32 权重）。

    与融合实现的差别只有 ``mhc_align_fleet`` 的 eps 位置（PaddleFleet 权重要求
    ``1/(sqrt(mean)+eps)``，DSV4 是 ``rsqrt(mean+eps)``），其余完全同 DSV4。

    返回形状与 ``mhc.mhc_pre`` 一致：``(*outer, hc, 1)`` / ``(*outer, hc, hc)`` /
    ``(*outer, hidden)``。
    """
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    residual_flat = residual.reshape(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=residual.dtype, device=residual.device
    )
    if num_tokens == 0:
        return (
            post_mix.view(*outer_shape, hc_mult, 1),
            comb_mix.view(*outer_shape, hc_mult, hc_mult),
            layer_input.view(*outer_shape, hidden_size),
        )

    assert fn.shape[0] == hc_mult3, (
        f"fn.shape[0]={fn.shape[0]} != hc_mult3={hc_mult3}"
    )
    assert fn.shape[1] == hc_mult * hidden_size, (
        f"fn.shape[1]={fn.shape[1]} != hc_mult*hidden_size={hc_mult * hidden_size}"
    )

    # 1) 一次连续 GEMM：与 DSV4 `hc_pre_torch_impl` 同式。
    x_flat = residual_flat.reshape(num_tokens, hc_mult * hidden_size).float()
    _mhc_trace("pre:flatten")
    if mhc_align_fleet:
        rms = 1.0 / (
            torch.sqrt(x_flat.square().mean(-1, keepdim=True)) + rms_eps
        )
    else:
        rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.nn.functional.linear(x_flat, fn.float()) * rms
    _mhc_trace("pre:linear")

    # 2) split-sinkhorn：与 DSV4 `hc_split_sinkhorn` 同式。
    pre, post, comb = _hc_split_sinkhorn_torch(
        mixes,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_repeat,
        hc_sinkhorn_eps,
        bool(mhc_align_fleet),
        hc_post_mult_value,
    )
    _mhc_trace("pre:sinkhorn-done")
    post_mix.copy_(post)
    comb_mix.copy_(comb.reshape(num_tokens, hc_mult * hc_mult))

    # 3) combine：与 DSV4 `hc_combine` 同式（fp32 累加后回 bf16）。
    y = (pre.unsqueeze(-1) * x_flat.view(num_tokens, hc_mult, hidden_size)).sum(dim=1)
    y = y.to(residual.dtype)
    _mhc_trace("pre:combine")

    # 4) 模型层 RMSNorm：DSV4 在 `hc_pre` 之后单独调 input_layernorm。
    if norm_weight is None:
        layer_input.copy_(y)
    else:
        assert norm_eps is not None, "norm_eps required when norm_weight is provided"
        layer_input.copy_(_rmsnorm_torch(y, norm_weight, norm_eps))
    _mhc_trace("pre:norm")

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


# --------------------------------------------------------------------------- #
# 融合 rope（按 fork kernel 源码 + 单测参考实现，而非复用插件版本）
# --------------------------------------------------------------------------- #
def _rope_interleaved(x_rope: torch.Tensor, freqs_real: torch.Tensor, positions: torch.Tensor):
    """对 x 的最后 rope_dim 维做交错 RoPE。x_rope: (tokens, heads, rope_dim)。"""
    rope_dim = x_rope.shape[-1]
    f = freqs_real[positions.long()].float()                  # (tokens, rope_dim)
    f = f.reshape(*f.shape[:-1], rope_dim // 2, 2).unsqueeze(-3)   # (tokens,1,rope_dim//2,2)
    rp = x_rope.reshape(*x_rope.shape[:-1], rope_dim // 2, 2)
    re = rp[..., 0] * f[..., 0] - rp[..., 1] * f[..., 1]
    im = rp[..., 0] * f[..., 1] + rp[..., 1] * f[..., 0]
    return torch.stack([re, im], -1).reshape(*x_rope.shape)


def fused_q_norm_rope(q_input, q_output, eps, freqs_cis, positions):
    """RMSNorm(no weight) + 末尾 rope_dim 的交错 RoPE。"""
    x = q_input.float()
    normed = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    rope_dim = freqs_real.shape[-1]
    nope = q_input.shape[-1] - rope_dim
    out = normed.clone()
    if rope_dim > 0:
        out[..., nope:] = _rope_interleaved(normed[..., nope:], freqs_real, positions)
    q_output.copy_(out.to(q_output.dtype))
    return None


# --------------------------------------------------------------------------- #
# 注入
# --------------------------------------------------------------------------- #
def apply() -> list[str]:
    """把上面的 torch 拆解实现 patch 到 fork 的调用点上。返回已 patch 的符号列表。"""
    import importlib

    patched = []

    def _patch(module_name, attr, impl):
        # 用 MLA 模块里的通用注入器：它会把**改名/多处 import 的旧绑定**一并修掉
        # （fork 的 `sglang.srt.models.deepseek_v4` 等模块在 import 期就绑死了这些
        # 融合算子，只改定义模块改不到它们）。
        try:
            import eb5_mla_adapt

            patched.extend(eb5_mla_adapt.patch_everywhere(module_name, attr, impl))
        except Exception:
            try:
                mod = importlib.import_module(module_name)
            except Exception:
                return
            if hasattr(mod, attr):
                setattr(mod, attr, impl)
                patched.append(f"{module_name}.{attr}")

    ew0 = "sglang.kernels.ops.attention.dsv4.elementwise"
    _patch(ew0, "fused_q_norm_rope", fused_q_norm_rope)

    mhc_mod = "sglang.kernels.ops.layernorm.mhc"
    # EB5 的融合 mHC（tilelang post / big_fuse pre / split-K / tf32）在真实多 rank
    # 前向会挂死；直接整体换成与 DSV4 同构的散算子实现，不再保留融合回退分支。
    _patch(mhc_mod, "mhc_pre", mhc_pre_torch_ref)
    _patch(mhc_mod, "mhc_post", mhc_post_torch_ref)
    # `ernie5.py` 在 import 期就 `from ...mhc import mhc_pre as _mhc_pre` /
    # `mhc_post as _mhc_post`，`patch_everywhere` 会按对象身份把这类旧绑定一并改掉。

    # --- 融合 rope 家族：复用插件 dsv4_torch_shim 的 torch 拆解（签名一致） ---
    # 注意：插件包 `sglang_kunlun.kernels` 的 __init__ 会拉起只对 sglang 0.5.14 成立的
    # import 链（`sglang.srt.layers.quantization.fp8_kernel` 在 fork 里已改名），
    # 所以这里**按文件直接加载** shim，并给它唯一缺的那个符号打桩。
    try:
        import importlib.util
        import sys
        import types

        _stub_name = "sglang_kunlun.hooks.layers.attention.nsa.triton_kernel"
        if _stub_name not in sys.modules:
            # DSV4 的 indexer 量化路径（fused_q_indexer_rope_hadamard_quant ->
            # dsv4_torch_shim.act_quant_kunlun）会**真的**调它，所以要给能用的实现
            # （与插件 nsa/triton_kernel.act_quant_kunlun 同实现，都是 kunlun_ops.quant2d）。
            # 不要改成 import 插件那个模块：会拉起 sglang_kunlun.hooks.layers.attention
            # 的 nsa redirect，提前 import fork 的 attention 模块，使本模块 _TARGETS 的补丁
            # 被已绑定的旧符号绕过（EB5 单测 19 passed -> 4 failed）。
            def _act_quant_kunlun(x, block_size=128, scale_fmt=None):
                import kunlun_ops

                assert x.is_contiguous(), "Input tensor must be contiguous"
                assert x.size(-1) % block_size == 0
                assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)
                x_shape = x.shape
                x_blocks = x.view(-1, block_size).contiguous()
                y_int8 = torch.empty(x_blocks.shape, dtype=torch.int8, device=x.device)
                y_max = torch.empty(x_blocks.shape[0], dtype=torch.float32, device=x.device)
                kunlun_ops.quant2d(x_blocks, y_int8, y_max, force_sdnn=True)
                return y_int8.view(x_shape), y_max.view(x_shape[0], -1)

            _m = types.ModuleType(_stub_name)
            _m.act_quant_kunlun = _act_quant_kunlun
            sys.modules[_stub_name] = _m

        _shim_path = _dsv4_torch_shim_path()
        _spec = importlib.util.spec_from_file_location("_eb5_dsv4_torch_shim", _shim_path)
        _shim = importlib.util.module_from_spec(_spec)
        sys.modules["_eb5_dsv4_torch_shim"] = _shim
        _spec.loader.exec_module(_shim)

        ew = "sglang.kernels.ops.attention.dsv4.elementwise"
        for attr, impl_name in (
            ("fused_k_norm_rope_flashmla", "_fused_k_norm_rope_flashmla_torch"),
            ("fused_rope_inplace", "_fused_rope_inplace_torch"),
            ("fused_q_indexer_rope_hadamard_quant", "_fused_q_indexer_rope_hadamard_quant_torch"),
        ):
            impl = getattr(_shim, impl_name, None)
            if impl is not None:
                _patch(ew, attr, impl)
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] rope shim load failed:", _e, file=_sys.stderr)

    # --- attn ② MLA 专用拆解（另见 eb5_mla_adapt.py） ---
    try:
        import eb5_mla_adapt

        patched.extend(eb5_mla_adapt.apply())
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] mla adapt failed:", _e, file=_sys.stderr)

    # --- HCA / MoE / 前后处理拆解（另见 eb5_extra_adapt.py） ---
    try:
        import eb5_extra_adapt

        patched.extend(eb5_extra_adapt.apply())
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] extra adapt failed:", _e, file=_sys.stderr)

    return patched
