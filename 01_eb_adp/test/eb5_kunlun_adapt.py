"""EB5-lite 算子在昆仑上的适配层 —— 参考 dpsk-v4 的「融合算子拆解」做法。

EB5 单测**直接调用**的这批 kernel 在昆仑上跑不了（tilelang / deep_gemm / fork 自带
nvcc-JIT / triton）。dpsk-v4 的做法是：把融合 kernel 换成**纯 torch 拆解实现**（或
KLX 原生算子）。本模块照此把 EB5 的融合 kernel 换成 torch 实现，语义与 kernel 源码
逐行对应，供 `code/eb5_layer` 的 kernel 级单测和后续组网使用。

被替换的 kernel 与拆解：

* ``mhc_post_tilelang``
      x[n,hco,h] = c[n,hco]*d[n,h] + Σ_hci a[n,hci,hco]*b[n,hci,h]
* ``mhc_pre_big_fuse_with_norm_tilelang``
      rms = rsqrt(Σ_splits sqrsum /(hc*hidden) + rms_eps)
      mixes = (Σ_splits mul) * rms
      post_mix = sigmoid(mixes[hc:2hc]*scale[1]+base[hc:2hc]) * post_mult
      comb     = sinkhorn( mixes[2hc:2hc+hc²]*scale[2]+base[...] )   # 列和为 1
      pre_mix  = sigmoid(mixes[:hc]*scale[0]+base[:hc]) + hc_pre_eps
      layer_input = rmsnorm(Σ_hc pre_mix*residual, norm_weight, norm_eps)
* ``tf32_hc_prenorm_gemm``
      out[s] = x_k[:, chunk_s] @ fn[:, chunk_s].T ; sqrsum[s] = Σ x_k²  (split-K)
"""

from __future__ import annotations

import math
import os

import torch

_LOG2E = 1.4426950408889634


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



# --------------------------------------------------------------------------- #
# hc_post
# --------------------------------------------------------------------------- #
def mhc_post_tilelang(a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024):
    """torch 拆解：x[n,hco,h] = c[n,hco]*d[n,h] + Σ_hci a[n,hci,hco]*b[n,hci,h]。"""
    out = c.unsqueeze(-1) * d.unsqueeze(1) + torch.einsum(
        "nhc,nhd->ncd", a.float(), b.float()
    )
    x.copy_(out.to(x.dtype))
    return x


# --------------------------------------------------------------------------- #
# hc_pre
# --------------------------------------------------------------------------- #
def tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_splits=None):
    """torch 拆解（split-K）：out[s] = x[:, chunk] @ fn[:, chunk].T；sqrsum[s] = Σ x[:,chunk]²。"""
    xf = x.float()
    ff = fn.float()
    k = xf.shape[-1]
    ns = out.shape[0]
    chunk = math.ceil(k / ns) if ns > 0 else k
    for s in range(ns):
        lo, hi = s * chunk, min((s + 1) * chunk, k)
        xs = xf[:, lo:hi]
        out[s].copy_(xs @ ff[:, lo:hi].t())
        sqrsum[s].copy_((xs * xs).sum(-1))
    return None


def mhc_pre_gemm_sqrsum_tilelang(
    x, fn, out, sqrsum, hc_mult3: int, hc_hidden_size: int,
    token_block: int = 32, hidden_block: int = 256,
):
    """torch 拆解（单链、无 split）：out[t,j] = Σ_k x[t,k]*fn[j,k]（j<hc_mult3）；
    sqrsum[t] = Σ_k x[t,k]²。对应 mhc.py 的 ``mhc_pre_gemm_sqrsum_tilelang``。
    """
    xf = x.float()
    n = xf.shape[0]
    j = min(int(hc_mult3), fn.shape[0])
    out[:n, :j].copy_(xf @ fn[:j].float().t())
    sqrsum[:n].copy_((xf * xf).sum(-1))
    return None


def mhc_pre_gemm_sqrsum_splitk_kernel(
    hc_mult3: int, hc_hidden_size: int, split_k: int,
    token_block: int = 32, hidden_block: int = 256, threads: int = 128,
):
    """torch 拆解（split-K）工厂：返回 ``(kernel_0, None)``。

    kernel_0 按 k 把 ``hc_hidden_size`` 切成 ``split_k`` 段，逐段写
    ``out_partial[s,t,:hc_mult3] = x[t, 段] @ fn[:hc_mult3, 段].T``、
    ``sqrsum_partial[s,t] = Σ x[t, 段]²``；后面的 ``*_big_fuse_*`` 再把各段求和。
    """
    # 与原 kernel 的 32 列输出和等长 split 契约一致，避免静默丢弃 K 尾部。
    if not 0 < hc_mult3 <= 32:
        raise ValueError("hc_mult3 must be in [1, 32]")
    if hc_hidden_size <= 0 or split_k <= 0 or hc_hidden_size % split_k:
        raise ValueError("hc_hidden_size must be positive and divisible by positive split_k")
    split_size = hc_hidden_size // split_k

    def kernel_0(x, fn, out_partial, sqrsum_partial):
        xf = x.float()
        n = xf.shape[0]
        f = fn.float()
        j = min(int(hc_mult3), f.shape[0])
        if os.environ.get("EB5_DEBUG_MHC_PRE"):
            print(
                f"[mhc-dbg] kernel_0 x={tuple(x.shape)}/{x.dtype} fn={tuple(fn.shape)} "
                f"out={tuple(out_partial.shape)} sqrsum={tuple(sqrsum_partial.shape)} "
                f"hc_mult3={hc_mult3} split_k={split_k} split_size={split_size} "
                f"n={n} j={j}",
                flush=True,
            )
        # kernel 里 out_frag 宽 32、只写前 hc_mult3 列；剩下的列在 big_fuse 里被
        # 求和但从不读取，这里显式清零避免 empty() 的垃圾值参与求和。
        out_partial[:, :n, :].zero_()
        for s in range(split_k):
            lo = s * split_size
            hi = lo + split_size
            # ⚠️ 这里两个 matmul 操作数原本都是**非连续切片**：
            #   xs            = xf[:, lo:hi]        → stride (H, 1)
            #   f[:j, lo:hi].t()                    → stride (1, H)
            # XPU 后端对非连续 matmul 会走另一条路径。2026-10-17 EB5 整网首次前向
            # 就在这一行报设备级 kernel 异常（`wait for noc idle timeout` /
            # `KL_XID_KERNEL_EXCEPTION`），把设备带坏。
            # 显式 `.contiguous()` **不改变数值**（CPU 上两种写法已逐元素对照过），
            # 只是把它拉回常规路径；仓库里 `dsv4_torch_shim.py` 也一贯这么写
            # （那里有 20+ 处 `.contiguous()`，本文件原先只有 2 处）。
            xs = xf[:, lo:hi].contiguous()
            fk = f[:j, lo:hi].t().contiguous()
            out_partial[s, :n, :j].copy_(xs @ fk)
            sqrsum_partial[s, :n].copy_((xs * xs).sum(-1))
        return None

    return kernel_0, None


def mhc_pre_gemm_sqrsum_dispatch():
    """``_mhc_pre_gemm_sqrsum_dispatch()`` 的 torch 替代（num_tokens > 2048 那条链）。"""
    return mhc_pre_gemm_sqrsum_tilelang


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
    n = residual.shape[0]
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3

    sq = gemm_out_sqrsum[:n_splits].float().sum(0)                     # (n,)
    rms = torch.rsqrt(sq / (hc_mult * hidden_size) + rms_eps)          # (n,)
    mixes = gemm_out_mul[:n_splits, :, :gemm_last_dim].float().sum(0) * rms.unsqueeze(-1)

    # post_mix: sigmoid(mixes[hc:2hc]*scale[1] + base[hc:2hc]) * post_mult
    post_mix.copy_(
        torch.sigmoid(mixes[:, hc_mult:2 * hc_mult] * hc_scale[1]
                      + hc_base[hc_mult:2 * hc_mult]) * hc_post_mult_value
    )

    # comb_mix: sinkhorn(mixes[2hc:2hc+hc²]*scale[2] + base[...])
    cm = (mixes[:, 2 * hc_mult:2 * hc_mult + hc_mult * hc_mult] * hc_scale[2]
          + hc_base[2 * hc_mult:2 * hc_mult + hc_mult * hc_mult]).reshape(n, hc_mult, hc_mult)
    cm = cm - cm.amax(dim=2, keepdim=True)                             # 行 max（dim=2 是 k）
    cm = torch.exp2(cm * _LOG2E) if mhc_align_fleet else torch.exp(cm)
    cm = cm / cm.sum(dim=2, keepdim=True) + hc_sinkhorn_eps
    cm = cm / (cm.sum(dim=1, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(max(int(sinkhorn_repeat) - 1, 0)):
        cm = cm / (cm.sum(dim=2, keepdim=True) + hc_sinkhorn_eps)
        cm = cm / (cm.sum(dim=1, keepdim=True) + hc_sinkhorn_eps)
    comb_mix.copy_(cm.reshape(n, hc_mult * hc_mult))

    # layer_input: rmsnorm(Σ_hc pre_mix*residual, norm_weight, norm_eps)
    pre_mix = torch.sigmoid(
        mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    ) + hc_pre_eps                                                    # (n, hc)
    ol = torch.einsum("nh,nhd->nd", pre_mix, residual.float())
    ol_bf = ol.to(residual.dtype)                                     # 与 kernel 一致：先 round 到 bf16 再平方
    sumsq = (ol_bf.float() ** 2).sum(-1)
    rsqrt_norm = torch.rsqrt(sumsq / hidden_size + norm_eps)
    layer_input.copy_(
        (ol_bf.float() * rsqrt_norm.unsqueeze(-1) * norm_weight.float()).to(layer_input.dtype)
    )
    return None


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
    _patch(mhc_mod, "mhc_post_tilelang", mhc_post_tilelang)
    _patch(mhc_mod, "mhc_pre_big_fuse_with_norm_tilelang", mhc_pre_big_fuse_with_norm_tilelang)
    # mhc_pre 的两级 tilelang GEMM+sqrsum：<=2048 token 走 splitk 工厂，
    # 否则走单链 dispatch。两者都被 mhc_pre 在调用点按模块全局名解析。
    _patch(mhc_mod, "mhc_pre_gemm_sqrsum_splitk_kernel", mhc_pre_gemm_sqrsum_splitk_kernel)
    _patch(mhc_mod, "mhc_pre_gemm_sqrsum_tilelang", mhc_pre_gemm_sqrsum_tilelang)
    _patch(mhc_mod, "_mhc_pre_gemm_sqrsum_dispatch", mhc_pre_gemm_sqrsum_dispatch)

    dg_mod = "sglang.srt.layers.deep_gemm_wrapper.entrypoint"
    _patch(dg_mod, "tf32_hc_prenorm_gemm", tf32_hc_prenorm_gemm)

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
