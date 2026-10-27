"""EB5 融合算子在昆仑上的 torch 替代。

EB5 直接调用的 tilelang / deep_gemm / nvcc-JIT / Triton kernel 在 P800 上不可用；
这里逐个给出签名、shape、dtype 与填充值一致的纯 torch 实现，由 ``apply()``
patch 到定义模块。mHC 的 pre / post / sinkhorn / combine 与 DSV4 共用
``kernels/mhc_torch.py``。
"""

from __future__ import annotations

import torch

from sglang_kunlun.eb5_adapt.kernels.mhc_torch import (
    _FP32_GEMM_TOKEN_CHUNK,
    _LOG2E,
    _matmul_bf16_by_fp32,
    _sqrsum_bf16_rows,
    hc_combine as mhc_combine_torch_ref,
    mhc_post as mhc_post_torch_ref,
    mhc_pre as mhc_pre_torch_ref,
)


def _import_sibling(name: str):
    """导入同目录模块，兼容包内导入与 ``EB5_KUNLUN_ADAPT_DIR`` 平铺导入。"""
    import importlib

    if __package__:
        try:
            return importlib.import_module(f".{name}", __package__)
        except ModuleNotFoundError as exc:
            # 只吞相对路径找不到，别掩盖被导入模块自身的依赖缺失。
            if exc.name not in (f"{__package__}.{name}", name):
                raise
    return importlib.import_module(name)


# --------------------------------------------------------------------------- #
# 融合 kernel 的 kernel 级 torch 替代
#
# 约束：纯 torch（不做 token 级 Python 循环）、大张量分块、原地输出的
# shape / dtype / 填充值与 kernel 一致。
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# hc_post：mhc_post_tilelang
# --------------------------------------------------------------------------- #
def mhc_post_tilelang(
    a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
):
    """``mhc.mhc_post_tilelang`` 的 torch 版，原地写 ``x``。

    ``x[n,hco,h] = c[n,hco]*d[n,h] + Σ_hci a[n,hci,hco]*b[n,hci,h]``。
    ``n_thr`` / ``h_blk`` 仅为签名兼容保留。
    """
    n = b.shape[0]
    if n == 0:
        return x
    for lo in range(0, n, _FP32_GEMM_TOKEN_CHUNK):
        hi = min(lo + _FP32_GEMM_TOKEN_CHUNK, n)
        # a 的 hci 维（dim=1）与 b 的 hci 维收缩，hco 落到输出的 dim=1。
        acc = torch.einsum("nhc,nhd->ncd", a[lo:hi], b[lo:hi].float())
        acc += c[lo:hi].unsqueeze(-1) * d[lo:hi].unsqueeze(1)
        x[lo:hi].copy_(acc.to(x.dtype))
    return x


# --------------------------------------------------------------------------- #
# hc_pre ①：deep_gemm.tf32_hc_prenorm_gemm
# --------------------------------------------------------------------------- #
def tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_splits=None):
    """``deep_gemm.tf32_hc_prenorm_gemm`` 的 torch 版（split-K，原地写 out / sqrsum）。

    ``out[s,t,:] = x[t, 段_s] @ fn[:, 段_s].T``、``sqrsum[s,t] = Σ x[t,k]²``。
    空输入直接返回，与 kernel 一致。
    """
    n, k = x.shape
    if n == 0:
        return None
    ns = int(out.shape[0]) if num_splits is None else int(num_splits)
    assert ns == out.shape[0], f"num_splits={ns} != out.shape[0]={out.shape[0]}"
    f = fn.float()
    j = min(fn.shape[0], out.shape[-1])
    # kernel 按 `T.ceildiv(k, ns)` 分段（最后一段可能不满）。
    split = -(-k // ns) if ns > 0 else k
    for s in range(ns):
        lo, hi = s * split, min((s + 1) * split, k)
        # 覆盖写：整行先清零，尾部列（> j）与空 split 都不能留 empty() 的垃圾值。
        out[s].zero_()
        sqrsum[s].zero_()
        if hi <= lo:
            continue
        xs = x[:, lo:hi]
        out[s][:, :j].copy_(_matmul_bf16_by_fp32(xs, f[:, lo:hi]))
        sqrsum[s].copy_(_sqrsum_bf16_rows(xs))
    return None


# --------------------------------------------------------------------------- #
# hc_pre ②：mhc_pre_gemm_sqrsum_tilelang / splitk / dispatch
# --------------------------------------------------------------------------- #
def mhc_pre_gemm_sqrsum_tilelang(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
):
    """``mhc.mhc_pre_gemm_sqrsum_tilelang`` 的 torch 版（单链、无 split）。

    ``out[t,j] = Σ_k x[t,k]*fn[j,k]``（``j < hc_mult3``）、``sqrsum[t] = Σ_k x[t,k]²``。
    """
    if not 0 < hc_mult3 <= 32:
        # kernel: assert hc_mult3 <= 32（0 会让下面 j=0，静默什么都不写）
        raise ValueError("hc_mult3 must be in [1, 32]")
    if hc_hidden_size % hidden_block:
        raise ValueError(
            f"hc_hidden_size={hc_hidden_size} not divisible by hidden_block={hidden_block}"
        )
    n = x.shape[0]
    j = min(int(hc_mult3), fn.shape[0], out.shape[1])
    if n == 0:
        return None
    out[:, :j].copy_(_matmul_bf16_by_fp32(x, fn[:j]))
    if j < out.shape[1]:          # 尾部列 kernel 不写，但也不能留垃圾
        out[:, j:].zero_()
    sqrsum.copy_(_sqrsum_bf16_rows(x))
    return None


def mhc_pre_gemm_sqrsum_splitk_kernel(
    hc_mult3: int,
    hc_hidden_size: int,
    split_k: int,
    token_block: int = 32,
    hidden_block: int = 256,
    threads: int = 128,
):
    """``mhc.mhc_pre_gemm_sqrsum_splitk_kernel`` 的 torch 版工厂。

    返回 ``(stage_0, stage_1)``，与 kernel 的返回顺序一致：stage_0 写 split 的
    ``out_partial`` / ``sqrsum_partial``（尾部列清零），stage_1 沿 split 求和。
    K 必须能被 ``split_k`` 与 ``hidden_block`` 整除。
    """
    if not 0 < hc_mult3 <= 32:
        raise ValueError("hc_mult3 must be in [1, 32]")
    if hc_hidden_size <= 0 or split_k <= 0 or hc_hidden_size % split_k:
        raise ValueError(
            f"hc_hidden_size={hc_hidden_size} must be positive and divisible by "
            f"positive split_k={split_k}"
        )
    if hc_hidden_size % hidden_block:
        raise ValueError(
            f"hc_hidden_size={hc_hidden_size} not divisible by hidden_block={hidden_block}"
        )
    split_size = hc_hidden_size // split_k
    if split_size % hidden_block:
        raise ValueError(
            f"split_size={split_size} not divisible by hidden_block={hidden_block}"
        )

    def stage_0(x, fn, out_partial, sqrsum_partial):
        n = x.shape[0]
        # 尾部列清零：big_fuse 按 gemm_last_dim(=32) 列求和，单测也是整块对照。
        out_partial[:, :n, :].zero_()
        if n == 0:
            return None
        f = fn.float()
        j = min(int(hc_mult3), f.shape[0])
        for s in range(split_k):
            lo, hi = s * split_size, (s + 1) * split_size
            xs = x[:, lo:hi]
            out_partial[s, :n, :j].copy_(_matmul_bf16_by_fp32(xs, f[:j, lo:hi]))
            sqrsum_partial[s, :n].copy_(_sqrsum_bf16_rows(xs))
        return None

    def stage_1(out_partial, sqrsum_partial, out, sqrsum):
        n = out.shape[0]
        if n == 0:
            return None
        j = min(int(hc_mult3), out.shape[1])
        out[:, :j].copy_(out_partial[:split_k, :n, :j].sum(0))
        if j < out.shape[1]:
            out[:, j:].zero_()
        sqrsum.copy_(sqrsum_partial[:split_k, :n].sum(0))
        return None

    return stage_0, stage_1


def _mhc_pre_gemm_sqrsum_dispatch():
    """``mhc._mhc_pre_gemm_sqrsum_dispatch`` 的 torch 版：统一返回单链实现。"""
    return mhc_pre_gemm_sqrsum_tilelang


# --------------------------------------------------------------------------- #
# hc_pre ③：mhc_pre_big_fuse_with_norm_tilelang
# --------------------------------------------------------------------------- #
def _mhc_pre_big_fuse_torch(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    hc_mult: int,
    gemm_last_dim: int,
    mhc_align_fleet: int,
    norm_weight=None,
    norm_eps=None,
):
    """``mhc_pre_big_fuse[_with_norm]_tilelang`` 的公共 torch 实现。

    ``norm_weight is None`` 走不带 norm 的分支；否则先 round 到 bf16，再用
    round 后的值计算 RMSNorm。
    """
    n = residual.shape[0]
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    if n == 0:
        return None
    assert residual.dim() == 3, f"residual must be (n, hc, hidden); got {residual.shape}"

    # 1) rms：沿 split 求和（kernel 是 T.serial(n_splits) 累加，n_splits 只有几十）。
    sq = gemm_out_sqrsum[:n_splits, :n].float().sum(0)                 # (n,)
    denom = hc_mult * hidden_size
    if mhc_align_fleet:
        # PaddleFleet：eps 在 sqrt **外**（`fused_mhc_kernels.py` 的 `norm/sqrt(K)+eps`）。
        rms = 1.0 / (torch.sqrt(sq / denom) + rms_eps)
    else:
        rms = torch.rsqrt(sq / denom + rms_eps)

    # 2) mixes：先按 split 求和，再乘 rms。gemm_out_mul 是 (n_splits, n, gemm_last_dim)。
    mixes = gemm_out_mul[:n_splits, :n, :gemm_last_dim].float().sum(0)
    mixes = mixes * rms.unsqueeze(-1)                                  # (n, hc_mult3)

    # 3) post_mix
    post_mix.copy_(
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )

    # 4) comb_mix：sinkhorn。dim=2 是 k（kernel 的 row），dim=1 是 j（kernel 的 col）。
    cm = (
        mixes[:, 2 * hc_mult : 2 * hc_mult + hc_mult * hc_mult] * hc_scale[2]
        + hc_base[2 * hc_mult : 2 * hc_mult + hc_mult * hc_mult]
    ).reshape(n, hc_mult, hc_mult)
    cm = cm - cm.amax(dim=2, keepdim=True)
    cm = torch.exp2(cm * _LOG2E) if mhc_align_fleet else torch.exp(cm)
    cm = cm / cm.sum(dim=2, keepdim=True) + hc_sinkhorn_eps
    cm = cm / (cm.sum(dim=1, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(max(int(sinkhorn_repeat) - 1, 0)):
        cm = cm / (cm.sum(dim=2, keepdim=True) + hc_sinkhorn_eps)
        cm = cm / (cm.sum(dim=1, keepdim=True) + hc_sinkhorn_eps)
    comb_mix.copy_(cm.reshape(n, hc_mult * hc_mult))

    # 5) layer_input = Σ_hc pre_mix*residual（[可选] RMSNorm）
    pre_mix = (
        torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
        + hc_pre_eps
    )                                                                  # (n, hc)
    ol = torch.zeros(n, hidden_size, dtype=torch.float32, device=residual.device)
    for i_hc in range(hc_mult):        # hc 维（4 次），不是 token / 元素维
        ol += pre_mix[:, i_hc : i_hc + 1] * residual[:, i_hc, :]
    # 与 kernel 一致：先 round 到 bf16（lizhenyun 那条注释要求与 torch 实现对齐）。
    ol_bf = ol.to(residual.dtype)
    if norm_weight is None:
        # 不带 norm 的 kernel：`T.copy(ol, layer_input[...])`，ol 就是 fp32 加权和。
        layer_input.copy_(ol_bf)
        return None
    # 带 norm：用 **round 之后** 的值算 sumsq，再乘 rsqrt 与 bf16 权重。
    sumsq = _sqrsum_bf16_rows(ol_bf)                                   # 沿 hidden 分块
    rsqrt_norm = torch.rsqrt(sumsq / hidden_size + norm_eps)
    ol_scaled = ol_bf.float() * rsqrt_norm.unsqueeze(-1)
    ol_scaled *= norm_weight.float()
    layer_input.copy_(ol_scaled.to(layer_input.dtype))
    return None


def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
    mhc_align_fleet: int = 0,
):
    """``mhc.mhc_pre_big_fuse_tilelang`` 的 torch 版：``norm_weight is None`` 分支。"""
    return _mhc_pre_big_fuse_torch(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
        gemm_last_dim,
        mhc_align_fleet,
    )


def mhc_pre_big_fuse_with_norm_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
    mhc_align_fleet: int = 0,
):
    """``mhc.mhc_pre_big_fuse_with_norm_tilelang`` 的 torch 版（原地写三个输出）。"""
    return _mhc_pre_big_fuse_torch(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        hc_mult,
        gemm_last_dim,
        mhc_align_fleet,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )


# --------------------------------------------------------------------------- #
# 融合 rope（按 fork kernel 源码 + 单测参考实现，而非复用插件版本）
# --------------------------------------------------------------------------- #


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
    """RMSNorm(no weight) + 末尾 rope_dim 的 RoPE。

    走 ``dsv4_torch_shim._fused_q_norm_rope_torch``，即 community 0.5.14 插件用的
    同一条路径（vendor ``kunlun_ops.dpsk_v4_norm_rope_gptj``，异常才回落 torch），
    让两边的逐算子对比只反映真正的语义差异，而不是 "一边融合算子、一边 torch 拆解"。
    注意：本改动对数值**没有**影响 —— 实测 layer0 的 rope 差异（0.47%）来自
    ``freqs_cis`` 本身（fork 只给压缩层加 YaRN，见下），换实现后仍为 0.470%。
    """
    from sglang_kunlun.eb5_adapt.kernels.dsv4_torch_shim import (
        _fused_q_norm_rope_torch,
    )

    _fused_q_norm_rope_torch(q_input, q_output, eps, freqs_cis, positions)
    return None


# --------------------------------------------------------------------------- #
# 注入
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# wo_a 低秩投影：替代 torch.einsum 的通用收缩
# --------------------------------------------------------------------------- #
def _apply_wo_a_bf16_matmul_torch(o, wo_a, is_decode=False, **_ignored):
    """``[T,G,D] x [G,R,D] -> [T,G,R]``：group 作 batch 的 bmm + 按 token 分块。

    fork 兜底的 ``einsum`` 会物化 4-D 中间量（整网首 forward 约 15 GiB）。
    ``is_decode`` 仅为签名兼容。
    """
    assert o.dim() == 3 and wo_a.dim() == 3, (o.shape, wo_a.shape)
    T, G, D = o.shape
    assert wo_a.shape[0] == G and wo_a.shape[-1] == D, (o.shape, wo_a.shape)
    R = wo_a.shape[1]
    if T == 0:
        return o.new_empty((0, G, R))
    w = wo_a.transpose(1, 2)
    if not w.is_contiguous():
        w = w.contiguous()                                   # [G,D,R]
    out = torch.empty((T, G, R), dtype=o.dtype, device=o.device)
    chunk = 1024
    for lo in range(0, T, chunk):
        hi = min(lo + chunk, T)
        x = o[lo:hi].permute(1, 0, 2).contiguous()           # [G,c,D]
        y = torch.bmm(x, w)                                  # [G,c,R]
        out[lo:hi] = y.permute(1, 0, 2)
    return out


def _install_kunlun_sampling_renorm() -> list[str]:
    """Point flashinfer's top-k/top-p renorm at the Kunlun implementations.

    ``eagle_utils.eagle_sample`` binds ``flashinfer.sampling.top_{k,p}_renorm_probs``
    whenever torch reports CUDA (which it does through xmlir). Those are JIT CUDA
    modules that need nvcc, so the first sampled request dies with
    ``Could not find nvcc``. ``sgl_kernel`` ships Kunlun implementations with the
    same signature -- the community 0.5.14 build calls them directly -- so alias
    them over the flashinfer entry points.
    """
    import importlib

    try:
        import flashinfer.sampling as _fs
        import sgl_kernel as _sk
    except Exception:  # noqa: BLE001
        return []
    patched = []
    # ``*_probs`` (plural) is a NotImplementedError stub in the Kunlun sgl_kernel;
    # the implemented entry points are the singular ``*_prob`` ones that the
    # community 0.5.14 build calls.
    for attr in ("top_p_renorm_probs", "top_k_renorm_probs"):
        impl = getattr(_sk, attr[:-1], None) or getattr(_sk, attr, None)
        if impl is not None and hasattr(_fs, attr):
            setattr(_fs, attr, impl)
            patched.append(f"flashinfer.sampling.{attr}")

    # ``eagle_sample`` also resolves the tree-sampling kernel per call. On this
    # stack ``torch.cuda`` is available (xmlir), so it picks the CUDA JIT module
    # from ``sglang.kernels.ops.speculative.sampling`` and dies building
    # ``sgl_kernel_jit_speculative_sampling_*`` (no nvcc). Point both possible
    # import sites at the kunlun_ops wrapper this package ships.
    try:
        from sglang_kunlun.kernels.sgl_kernel_kunlun import sampling as _ksamp

        impl = _ksamp.tree_speculative_sampling_target_only
        for mod_name in (
            "sglang.kernels.ops.speculative.sampling",
            "sgl_kernel",
        ):
            try:
                mod = importlib.import_module(mod_name)
            except Exception:  # noqa: BLE001
                continue
            if hasattr(mod, "tree_speculative_sampling_target_only"):
                setattr(mod, "tree_speculative_sampling_target_only", impl)
                patched.append(f"{mod_name}.tree_speculative_sampling_target_only")
    except Exception:  # noqa: BLE001
        pass
    return patched


def _install_routed_scaling_fix() -> list[str]:
    """Apply ``routed_scaling_factor`` that the XPU MoE epilogue drops.

    Activation taps on the same prompt show layer-0 routed-expert output is
    exactly 1/1.5 of the community build (cosine 0.99999, norm ratio 0.666,
    shared expert identical): the checkpoint's ``routed_scaling_factor=1.5``
    never reaches the weights. Routing here keeps ``apply_routed_scaling_
    factor_on_output`` false, so the value is supposed to be applied by the MoE
    epilogue; the runner used on this stack does not do it.

    Fold the factor into the top-k weights instead (same place the fused path
    would put it), skipping it when the routing already applied it.
    """
    import inspect
    import os

    if os.environ.get("SGLANG_KUNLUN_APPLY_ROUTED_SCALE", "1") == "0":
        return []
    try:
        import sglang.srt.layers.moe.topk as _topk
    except Exception:  # noqa: BLE001
        return []

    patched = []
    for name in (
        "biased_topk_impl",
        "biased_topk_jit_kernel_impl",
        "hash_topk",
    ):
        fn = getattr(_topk, name, None)
        if fn is None or getattr(fn, "_kunlun_scaled", False):
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue

        def make(orig, signature, label):
            def wrapper(*args, **kwargs):
                out = orig(*args, **kwargs)
                try:
                    bound = signature.bind_partial(*args, **kwargs)
                    rsf = bound.arguments.get("routed_scaling_factor")
                    fused = bound.arguments.get(
                        "apply_routed_scaling_factor_on_output", False
                    )
                except Exception:  # noqa: BLE001
                    rsf, fused = None, True
                if rsf not in (None, 1.0) and not fused:
                    weights, ids = out
                    return weights * float(rsf), ids
                return out

            wrapper._kunlun_scaled = True
            wrapper.__name__ = getattr(orig, "__name__", label)
            return wrapper

        setattr(_topk, name, make(fn, sig, name))
        patched.append(f"sglang.srt.layers.moe.topk.{name}")

    return patched


def _install_dense_prefill_default() -> list[str]:
    """Default DeepSeek-V4 prefill to the dense path on Kunlun XPU.

    ``SGLANG_OPT_FLASHMLA_SPARSE_PREFILL`` is on by default in this fork; on
    Kunlun the sparse prefill kernel is emulated by the plugin
    (``_forward_prefill_sparse`` gathers SWA + compressed pages and runs a torch
    sparse MLA). Layer-by-layer comparison against community v0.5.14 with the
    same weights shows the emulation is *not* numerically equivalent: the layer2
    attention output differs by 70% relative L2 and the error compounds to 37%
    at the final norm, while the dense path (``_kunlun_compressed_attention``,
    which is also what v0.5.14 runs) lands at 1.4% / 6.6%. The v0.5.14-era
    plugin never exercised the sparse path, so it was never validated here.

    Mirrors ``apply_deepseek_v4_defaults``' ROCm branch, which disables the same
    flag for the same reason. An explicit ``SGLANG_OPT_FLASHMLA_SPARSE_PREFILL``
    still wins so the path stays debuggable.
    """
    from sglang.srt.environ import envs

    flag = envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL
    if flag.is_set() and flag.get():
        print(
            "[eb5-kunlun-adapt] keeping SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1 "
            "(explicitly requested; the torch sparse-prefill emulation is not "
            "numerically aligned with the dense path yet)",
            flush=True,
        )
        return []
    flag.set(False)
    return ["envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=False"]


def apply() -> list[str]:
    """把上面的 torch 拆解实现 patch 到 fork 的调用点上。返回已 patch 的符号列表。"""
    import importlib

    patched = []
    try:
        patched.extend(_install_dense_prefill_default())
    except Exception as _e:  # noqa: BLE001
        import sys as _sys

        print("[eb5-kunlun-adapt] dense prefill default failed:", _e, file=_sys.stderr)
    try:
        patched.extend(_install_routed_scaling_fix())
    except Exception as _e:  # noqa: BLE001
        import sys as _sys

        print("[eb5-kunlun-adapt] routed scaling fix failed:", _e, file=_sys.stderr)
    try:
        patched.extend(_install_kunlun_sampling_renorm())
    except Exception as _e:  # noqa: BLE001
        import sys as _sys

        print("[eb5-kunlun-adapt] sampling renorm patch failed:", _e, file=_sys.stderr)

    def _patch(module_name, attr, impl):
        # 用 MLA 模块里的通用注入器：它会把**改名/多处 import 的旧绑定**一并修掉
        # （fork 的 `sglang.srt.models.deepseek_v4` 等模块在 import 期就绑死了这些
        # 融合算子，只改定义模块改不到它们）。
        try:
            eb5_mla_adapt = _import_sibling("eb5_mla_adapt")

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
    # ``norm_rope.fused_q_norm_rope`` dispatches on a module-level `_is_xpu`;
    # on this stack torch.xpu is absent, so it fell through to the CUDA JIT
    # builder (`sgl_kernel_jit_dpsk_v4_main_q_norm_rope_*`) and failed there.
    _patch(
        "sglang.kernels.ops.attention.dsv4.norm_rope",
        "fused_q_norm_rope",
        fused_q_norm_rope,
    )

    mhc_mod = "sglang.kernels.ops.layernorm.mhc"
    # ① 调用层：EB5 的模型前向统一走与 DSV4 同构的散算子实现（融合分支在真实多 rank
    #    前向会挂死，不再保留回退）。`ernie5.py` 在 import 期就
    #    `from ...mhc import mhc_pre as _mhc_pre` / `mhc_post as _mhc_post`，
    #    `patch_everywhere` 会按对象身份把这类旧绑定一并改掉。
    _patch(mhc_mod, "mhc_pre", mhc_pre_torch_ref)
    # ``mhc_combine`` was dropped from the registration set together with the
    # other mHC kernels; DSV4's boundary path still calls it, and the Triton
    # version then dies in graph capture (``_CudaTarget.arch``).
    _patch(mhc_mod, "mhc_combine", mhc_combine_torch_ref)
    # ①-b `mhc_post` 例外（ERNI-7 D2）：DSV4 侧这一步是**一发融合算子**
    #    `kunlun_ops.hc_post_kunlun_impl`（见 sglang05124 的
    #    `hooks/models/deepseek_v4.py:32-64`；trace 实测
    #    `xpuops3::hc_post_kunlun_impl<bfloat16,float>` 0.586 ms/step x86），
    #    EB5 这里因为 `ernie5.py` 调用时不传 `use_fused`、取 `mhc_torch.py` 的默认
    #    False，退化成 `unsqueeze/mul/sum/add/type_as` 散算子。
    #    `mhc_post` 是纯代数（加权和 + 残差重组），不含 softmax/rsqrt，
    #    所以 `mhc_align_fleet` 的两处对齐差异（eps 位置、exp2 vs exp）与它无关。
    #    形状门槛（mhc_torch.py:316-324）EB5 全满足：hc_mult=4、x 2-D、residual 3-D。
    #    用 `EB5_MHC_POST_FUSED` 门控，默认关；置 1 打开，回退 = 不设该变量。
    import os as _os

    if _os.environ.get("EB5_MHC_POST_FUSED", "0") == "1":

        _seen = []

        def _mhc_post_fused(x, residual, post_layer_mix, comb_res_mix, **kwargs):
            # 直调融合算子（与 DSV4 的 hook `hooks/models/deepseek_v4.py:32-64` 同形），
            # 不走 `mhc_torch.mhc_post` 的 `use_fused` 门——那条路的
            # `except Exception: pass` 会静默回退，没法证明 kernel 真的跑了。
            try:
                import kunlun_ops

                post = (
                    post_layer_mix.squeeze(-1)
                    if post_layer_mix.dim() == residual.dim()
                    else post_layer_mix
                )
                if x.shape[0] == 0:
                    return torch.empty_like(residual)
                b, n, d = residual.shape
                out = torch.empty((b, n, d), dtype=x.dtype, device=x.device)
                kunlun_ops.hc_post_kunlun_impl(
                    post.contiguous(),
                    x.contiguous(),
                    comb_res_mix.contiguous(),
                    residual.contiguous(),
                    out,
                    b,
                    n,
                    d,
                )
                if not _seen:
                    _seen.append(1)
                    print(
                        f"[ERNI-7] mhc_post FUSED active: hc_post_kunlun_impl "
                        f"b={b} n={n} d={d} dtype={x.dtype}",
                        flush=True,
                    )
                return out
            except Exception as exc:  # noqa: BLE001
                if not _seen:
                    _seen.append(1)
                    print(f"[ERNI-7] mhc_post FUSED failed -> torch: {exc!r}", flush=True)
                return mhc_post_torch_ref(x, residual, post_layer_mix, comb_res_mix)

        _patch(mhc_mod, "mhc_post", _mhc_post_fused)
        patched.append("mhc_post[use_fused=True]")
    else:
        _patch(mhc_mod, "mhc_post", mhc_post_torch_ref)

    # ①-c split-Sinkhorn（ERNI-7 D2 第 2 项）：`mhc_pre` 里最贵的一段。
    #    `mhc_torch.mhc_pre:271` 直调 `hc_split_sinkhorn_flat`，绕过了 `:130` 的
    #    派发器，所以 `use_fused` 根本到不了这里；派发器那条路的门
    #    （`mhc_torch.py:145`）又要求 `not mhc_align_fleet`，而 EB5 的
    #    `mhc_align_fleet` 恒为 True（`"ernie" in model_type`）。
    #
    #    那个门对 sinkhorn **看起来**是过严的：`mhc_align_fleet` 只改两处，
    #      - `1/(sqrt(mean)+eps)` vs `rsqrt(mean+eps)` —— 在 `mhc_pre` 的 RMS 里，
    #        发生在 GEMM 之前，**不属于** `hc_split_sinkhorn_flat`；
    #      - `exp2((x-max)*log2e)` vs `exp(x-max)` —— 在 sinkhorn 里，数学恒等。
    #
    #    ⚠️ 但实测结论相反：**这个门是对的，不要打开本开关。**
    #    - T1 性能确实大涨：服务端 `gen throughput` 1.79 -> 2.55~2.64 tok/s（+44%）。
    #    - T2 精度闸门**挂了**：`aime26_first2` **0/2**（pred=5 exp=277、pred=1 exp=62，
    #      两题都跑满 8192 token 没收敛）。按 ERNI-7 的硬闸门已回退。
    #    - 原因（tools/t0_sinkhorn_diag.py 实测）：融合 kernel 与参考实现的误差
    #      **随输入幅度增长** —— comb 的 max_rel 在 |mixes|~1 时 1.085e-6、
    #      ~4 时 2.444e-6、~16 时 **2.065e-4**。最初的 T0 用 `randn`（幅度~1）
    #      只看到 1e-6，是**样本幅度不足**；真实网络里 GEMM 后的 mixes 幅度大得多，
    #      误差经 43 层 x 2 个 block x 8192 步放大后足以带偏推理链。
    #    => 这一项应归 **需新开发融合算子**（要一个与 fleet 对齐、在大幅度输入下
    #       仍精确的 split-Sinkhorn kernel）。它的价值已实测：**+44% decode**。
    #
    #    开关保留但**默认关**，仅作为复现失败与后续开发的入口。
    #    接法：替换 `mhc_torch` 模块里的 `hc_split_sinkhorn_flat` 全局名。
    #    `mhc_pre` 在调用时按模块全局解析该名字，所以这样能接到；派发器 `:173`
    #    的兜底分支同样受益。
    if _os.environ.get("EB5_MHC_SINKHORN_FUSED", "0") == "1":
        from sglang_kunlun.eb5_adapt.kernels import mhc_torch as _mhc_torch

        _flat_ref = _mhc_torch.hc_split_sinkhorn_flat
        _sk_seen = []

        def _sinkhorn_flat_fused(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps,
            *, mhc_align_fleet=False, post_mult_value=2.0,
        ):
            # 直调融合算子，不走派发器的 `use_fused` 门；那条路的
            # `except Exception: pass` 会静默回退，没法证明 kernel 真的跑了。
            try:
                import kunlun_ops

                hc = int(hc_mult)
                if float(post_mult_value) != 2.0:
                    raise ValueError(f"post_mult_value={post_mult_value} != 2.0")
                m = mixes.reshape(-1, (2 + hc) * hc).contiguous()
                n = m.shape[0]
                if n == 0:
                    raise ValueError("empty batch")
                pre = torch.empty((n, hc), dtype=torch.float32, device=mixes.device)
                post = torch.empty((n, hc), dtype=torch.float32, device=mixes.device)
                comb = torch.empty((n, hc * hc), dtype=torch.float32, device=mixes.device)
                kunlun_ops.mhc_split_sinkhorn(
                    m,
                    hc_scale.float().contiguous(),
                    hc_base.float().contiguous(),
                    pre,
                    post,
                    comb,
                    hc,
                    int(sinkhorn_iters),
                    float(eps),
                )
                if not _sk_seen:
                    _sk_seen.append(1)
                    print(
                        f"[ERNI-7] sinkhorn FUSED active: mhc_split_sinkhorn "
                        f"n={n} hc={hc} iters={int(sinkhorn_iters)} "
                        f"align_fleet={bool(mhc_align_fleet)}",
                        flush=True,
                    )
                return pre, post, comb.reshape(n, hc, hc)
            except Exception as exc:  # noqa: BLE001
                if not _sk_seen:
                    _sk_seen.append(1)
                    print(f"[ERNI-7] sinkhorn FUSED failed -> torch: {exc!r}", flush=True)
                return _flat_ref(
                    mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps,
                    mhc_align_fleet=mhc_align_fleet,
                    post_mult_value=post_mult_value,
                )

        _mhc_torch.hc_split_sinkhorn_flat = _sinkhorn_flat_fused
        patched.append("hc_split_sinkhorn_flat[fused]")

    # ② kernel 级：昆仑没有 tilelang / deep_gemm，这些融合 kernel 由 torch 散算子
    #    逐个替代（**不能**只在 ① 的调用层绕开 —— 直调 kernel 的代码同样要能跑）。
    _patch(mhc_mod, "mhc_post_tilelang", mhc_post_tilelang)
    _patch(mhc_mod, "mhc_pre_big_fuse_tilelang", mhc_pre_big_fuse_tilelang)
    _patch(mhc_mod, "mhc_pre_big_fuse_with_norm_tilelang", mhc_pre_big_fuse_with_norm_tilelang)
    _patch(mhc_mod, "mhc_pre_gemm_sqrsum_tilelang", mhc_pre_gemm_sqrsum_tilelang)
    _patch(mhc_mod, "mhc_pre_gemm_sqrsum_splitk_kernel", mhc_pre_gemm_sqrsum_splitk_kernel)
    _patch(mhc_mod, "_mhc_pre_gemm_sqrsum_dispatch", _mhc_pre_gemm_sqrsum_dispatch)
    # deep_gemm 的 tf32 prenorm GEMM（`mhc_pre` 的 `SGLANG_OPT_DEEPGEMM_HC_PRENORM=1`
    # 分支、以及 kernel 级单测都直接 import 它）。
    # ③ wo_a 低秩投影：fork 兜底的 torch.einsum 在昆仑上会物化 4-D 中间量
    #    （整网 43 层首 forward 峰值 ~15 GiB -> OOM）。改成 bmm + 分块。
    _patch(
        "sglang.srt.models.deepseek_v4",
        "_apply_wo_a_bf16_matmul",
        _apply_wo_a_bf16_matmul_torch,
    )

    dg_mod = "sglang.srt.layers.deep_gemm_wrapper.entrypoint"
    _patch(dg_mod, "tf32_hc_prenorm_gemm", tf32_hc_prenorm_gemm)

    # 融合 rope 家族：复用包内 dsv4_torch_shim 的 torch 拆解。
    try:
        from sglang_kunlun.eb5_adapt.kernels import dsv4_torch_shim as _shim

        ew = "sglang.kernels.ops.attention.dsv4.elementwise"
        for attr, impl_name in (
            ("fused_k_norm_rope_flashmla", "_fused_k_norm_rope_flashmla_torch"),
            ("fused_rope_inplace", "_fused_rope_inplace_torch"),
            (
                "fused_q_indexer_rope_hadamard_quant",
                "_fused_q_indexer_rope_hadamard_quant_torch",
            ),
        ):
            impl = getattr(_shim, impl_name, None)
            if impl is not None:
                _patch(ew, attr, impl)
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] rope shim load failed:", _e, file=_sys.stderr)

    # --- attn ② MLA 专用拆解（另见本包的 eb5_mla_adapt.py） ---
    try:
        eb5_mla_adapt = _import_sibling("eb5_mla_adapt")

        patched.extend(eb5_mla_adapt.apply())
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] mla adapt failed:", _e, file=_sys.stderr)

    # --- HCA / MoE / 前后处理拆解（另见本包的 eb5_extra_adapt.py） ---
    try:
        eb5_extra_adapt = _import_sibling("eb5_extra_adapt")

        patched.extend(eb5_extra_adapt.apply())
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] extra adapt failed:", _e, file=_sys.stderr)

    # --- 在线 INT8 量化方法注册（另见本包的 eb5_int8_adapt.py） ---
    # 必须发生在 `prepare_server_args()` 之前：`--quantization` 的 argparse
    # choices 在参数解析时才构建。插件入口点早于参数解析，满足该时序。
    try:
        eb5_int8_adapt = _import_sibling("eb5_int8_adapt")

        patched.extend(eb5_int8_adapt.apply())
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] int8 adapt failed:", _e, file=_sys.stderr)

    # --- 离线 INT8 checkpoint 的 scale 路由（另见本包的 eb5_offline_int8_adapt.py） ---
    # checkpoint 自带 per-channel INT8 routed experts 时，loader 需要认识
    # ``w{1,2,3}.weight_scale``（而不是 block-FP8 的 ``weight_scale_inv``）。
    try:
        eb5_offline_int8_adapt = _import_sibling("eb5_offline_int8_adapt")

        patched.extend(eb5_offline_int8_adapt.apply())
    except ImportError:
        pass
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] offline int8 adapt failed:", _e, file=_sys.stderr)

    # --- DSA transform_index 的 torch 拆解（另见本包的 eb5_transform_index_adapt.py） ---
    try:
        eb5_transform_index_adapt = _import_sibling("eb5_transform_index_adapt")

        patched.extend(eb5_transform_index_adapt.apply())
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] transform_index adapt failed:", _e, file=_sys.stderr)

    # --- xy fork 专属适配（ERNIE 5.5 算子 + DP 空转守卫，另见 eb5_xy_adapt.py） ---
    # 上游仓库没有这个文件，缺失时静默跳过，保证本模块可以随上游同步。
    try:
        eb5_xy_adapt = _import_sibling("eb5_xy_adapt")

        patched.extend(eb5_xy_adapt.apply())
    except ImportError:
        pass
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] xy adapt failed:", _e, file=_sys.stderr)

    # --- DSV4 模型级算子（hc_pre / hc_post 走 kunlun_ops 融合实现） ---
    # 这些是 @plugin_hook，注册即生效；模块 import 失败不阻断启动。
    try:
        from sglang_kunlun.eb5_adapt.hooks.models import deepseek_v4 as _dsv4_model_hooks

        _ = _dsv4_model_hooks
        patched.append("models.deepseek_v4(hc_pre,hc_post)")
    except ImportError:
        pass
    except Exception as _e:
        import sys as _sys

        print("[eb5-kunlun-adapt] model hooks failed:", _e, file=_sys.stderr)

    return patched
