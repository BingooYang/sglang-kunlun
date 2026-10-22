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


def apply() -> list[str]:
    """把上面的 torch 拆解实现 patch 到 fork 的调用点上。返回已 patch 的符号列表。"""
    import importlib

    patched = []

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
    _patch(mhc_mod, "mhc_post", mhc_post_torch_ref)
    # ``mhc_combine`` was dropped from the registration set together with the
    # other mHC kernels; DSV4's boundary path still calls it, and the Triton
    # version then dies in graph capture (``_CudaTarget.arch``).
    _patch(mhc_mod, "mhc_combine", mhc_combine_torch_ref)
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

    return patched
