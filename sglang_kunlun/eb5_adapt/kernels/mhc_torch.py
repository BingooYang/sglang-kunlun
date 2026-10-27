# Copyright (c) 2026 Baidu, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSV4 / EB5 共用的 mHC 原语。

两个模型的 mHC 结构相同，只是入口名不同（DSV4 的 ``hc_pre`` / ``hc_post``
与 EB5 的 ``mhc.mhc_pre`` / ``mhc.mhc_post``）。这里保留唯一实现，避免两份
代码漂移。``use_fused=True`` 时优先使用 Kunlun 原生算子，失败回退 torch。
"""

from __future__ import annotations

import os

import torch

_LOG2E = 1.4426950408889634

#: ``x[lo:hi].float() @ f.T`` token block: 512 x 16384 fp32 ~= 33 MB.
_FP32_GEMM_TOKEN_CHUNK = 512
#: Chunk for the row-wise square sum along K: 4096 x 2048 fp32 ~= 33 MB.
_SQRSUM_K_CHUNK = 4096


def trace(tag: str) -> None:
    """Sync and log each sub-step when ``EB5_MHC_TRACE=1`` (debug only)."""
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


def _matmul_bf16_by_fp32(x, f, token_chunk: int = _FP32_GEMM_TOKEN_CHUNK):
    """``x[n,k] (bf16) @ f[j,k].T (fp32) -> out[n,j] (fp32)``, chunked by token."""
    n = x.shape[0]
    j = f.shape[0]
    out = torch.empty(n, j, dtype=torch.float32, device=x.device)
    if n == 0:
        return out
    ft = f.t().contiguous()
    for lo in range(0, n, token_chunk):
        hi = min(lo + token_chunk, n)
        xb = x[lo:hi].float().contiguous()
        out[lo:hi] = xb @ ft
    return out


def _sqrsum_bf16_rows(x, k_chunk: int = _SQRSUM_K_CHUNK):
    """``sqrsum[t] = sum_k x[t,k]**2`` in fp32, chunked along K."""
    n, k = x.shape
    acc = torch.zeros(n, dtype=torch.float32, device=x.device)
    for lo in range(0, k, k_chunk):
        hi = min(lo + k_chunk, k)
        xb = x[:, lo:hi].float()
        acc += (xb * xb).sum(-1)
    return acc


def rmsnorm(x, weight, eps):
    """Explicit ``input_layernorm``: fp32 accumulate, fp32 weight, cast back."""
    xf = x.float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (xf * rsqrt * weight.float()).to(x.dtype)


def hc_split_sinkhorn_flat(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    *,
    mhc_align_fleet: bool = False,
    post_mult_value: float = 2.0,
):
    """Core split-Sinkhorn on a flat ``[n, hc*(2+hc)]`` mix tensor.

    Returns ``pre [n, hc]``, ``post [n, hc]`` and ``comb [n, hc, hc]``.
    """
    flat = mixes.float()
    if flat.dim() > 2:
        flat = flat.reshape(-1, flat.shape[-1])
    hc = int(hc_mult)
    scale = hc_scale.float()
    base = hc_base.float()

    pre = torch.sigmoid(flat[:, :hc] * scale[0] + base[:hc]) + eps
    post = post_mult_value * torch.sigmoid(
        flat[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    )
    comb = (flat[:, 2 * hc :] * scale[2] + base[2 * hc :]).reshape(-1, hc, hc)
    trace("sinkhorn:mixes")

    row_max = comb.amax(dim=2, keepdim=True)
    if mhc_align_fleet:
        comb = torch.exp2((comb - row_max) * _LOG2E)
    else:
        comb = torch.exp(comb - row_max)
    trace("sinkhorn:softmax")
    comb = comb / comb.sum(dim=2, keepdim=True) + eps
    trace("sinkhorn:norm0-dim2")
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    trace("sinkhorn:norm0-dim1")
    for it in range(max(int(sinkhorn_iters) - 1, 0)):
        comb = comb / (comb.sum(dim=2, keepdim=True) + eps)
        trace(f"sinkhorn:iter{it}-dim2")
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
        trace(f"sinkhorn:iter{it}-dim1")
    return pre, post, comb


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    *,
    mhc_align_fleet: bool = False,
    post_mult_value: float = 2.0,
    use_fused: bool = False,
):
    """``mhc.hc_split_sinkhorn`` replacement: ``(b, s, *)`` in, same shape out."""
    b, s, _ = mixes.size()
    hc = int(hc_mult)
    if use_fused and not mhc_align_fleet and float(post_mult_value) == 2.0:
        try:
            import kunlun_ops

            m = mixes.reshape(-1, (2 + hc) * hc).contiguous()
            n = m.shape[0]
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
            return (
                pre.reshape(b, s, hc),
                post.reshape(b, s, hc),
                comb.reshape(b, s, hc, hc),
            )
        except Exception:
            pass

    pre, post, comb = hc_split_sinkhorn_flat(
        mixes,
        hc_scale,
        hc_base,
        hc,
        sinkhorn_iters,
        eps,
        mhc_align_fleet=mhc_align_fleet,
        post_mult_value=post_mult_value,
    )
    return (
        pre.reshape(b, s, hc),
        post.reshape(b, s, hc),
        comb.reshape(b, s, hc, hc),
    )


def hc_combine(
    x_flat: torch.Tensor, pre: torch.Tensor, hc: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """``y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h]``."""
    m = x_flat.shape[0]
    h = x_flat.shape[1] // hc
    y = torch.empty((m, h), dtype=out_dtype, device=x_flat.device)
    if m == 0:
        return y
    xv = x_flat.reshape(m, hc, h).to(torch.float32)
    acc = (pre.to(torch.float32).unsqueeze(-1) * xv).sum(dim=1)
    y.copy_(acc.to(out_dtype))
    return y


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    n_splits_pre: int = 32,
    *,
    norm_weight=None,
    norm_eps=None,
    mhc_align_fleet: bool = False,
    mhc_align_accum_order: bool = False,
    eps_outside_sqrt: bool = False,
):
    """``mhc.mhc_pre`` replacement (same return shapes and kernels).

    Steps: GEMM + RMS -> split-Sinkhorn -> combine -> optional RMSNorm.
    Everything is expressed on the shared primitives above, so the DSV4
    ``hc_pre`` path and the EB5 ``mhc_pre`` path cannot drift apart.
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

    x_flat = residual_flat.reshape(num_tokens, hc_mult * hidden_size).float()
    trace("pre:flatten")
    # ``eps_outside_sqrt`` is upstream's name for this same alternative: it only
    # changes the mapping's rms, unlike ``mhc_align_fleet`` which also switches
    # the Sinkhorn softmax to exp2. ERNIE 5.5 Flash always passes it True.
    if mhc_align_fleet or eps_outside_sqrt:
        rms = 1.0 / (
            torch.sqrt(x_flat.square().mean(-1, keepdim=True)) + rms_eps
        )
    else:
        rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.nn.functional.linear(x_flat, fn.float()) * rms
    trace("pre:linear")

    pre, post, comb = hc_split_sinkhorn_flat(
        mixes,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_repeat,
        hc_sinkhorn_eps,
        mhc_align_fleet=bool(mhc_align_fleet),
        post_mult_value=hc_post_mult_value,
    )
    trace("pre:sinkhorn-done")
    post_mix.copy_(post)
    comb_mix.copy_(comb.reshape(num_tokens, hc_mult * hc_mult))

    y = hc_combine(
        residual_flat.reshape(num_tokens, hc_mult * hidden_size),
        pre,
        hc_mult,
        residual.dtype,
    )
    trace("pre:combine")

    if norm_weight is None:
        layer_input.copy_(y)
    else:
        assert norm_eps is not None, "norm_eps required when norm_weight is provided"
        layer_input.copy_(rmsnorm(y, norm_weight, norm_eps))
    trace("pre:norm")

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    *,
    use_fused: bool = False,
) -> torch.Tensor:
    """``mhc.mhc_post`` replacement.

    ``out[n,hco,h] = post[n,hco] * x[n,h] + sum_hci comb[n,hci,hco] * residual[n,hci,h]``.
    ``post_layer_mix`` accepts ``(n, hc, 1)`` (EB5 call shape) or ``(n, hc)``
    (DSV4 call shape).
    """
    if post_layer_mix.dim() == residual.dim():  # (n, hc, 1)
        post = post_layer_mix.squeeze(-1)
    else:
        post = post_layer_mix

    if x.shape[0] == 0:
        return torch.empty_like(residual)

    if (
        use_fused
        and post.is_contiguous()
        and x.is_contiguous()
        and residual.is_contiguous()
        and comb_res_mix.is_contiguous()
        and x.dim() == 2
        and residual.dim() == 3
        and post.dim() == 2
        and comb_res_mix.dim() == 3
        and residual.shape[1] == 4
    ):
        try:
            import kunlun_ops

            batch, n, d = residual.shape
            out = torch.empty((batch, n, d), dtype=x.dtype, device=x.device)
            kunlun_ops.hc_post_kunlun_impl(
                post, x, comb_res_mix, residual, out, batch, n, d
            )
            trace("post:out")
            return out
        except Exception:
            pass

    out = post.unsqueeze(-1) * x.unsqueeze(1) + (
        comb_res_mix.unsqueeze(-1) * residual.unsqueeze(2)
    ).sum(dim=1)
    trace("post:out")
    return out.type_as(x)
