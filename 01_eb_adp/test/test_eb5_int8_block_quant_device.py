"""Device (Kunlun P800) verification of the INT8 GEMM contract used by EB5.

What this pins down:

1. ``kunlun_ops.quant2d`` stores the raw row abs-max (dequant = ``q*max/127``),
   which is the convention every scale in this feature uses.
2. ``xpu_kunlun_ops.matmul`` with ``x_pc_max [m,1]`` + ``w_pc_max [n,1]`` is the
   dense W8A8 INT8 GEMM, and it accepts an fp32 bias.
3. ``m_grouped_gemm_I8_I8_bf16_nt_contiguous_v3`` with
   ``w_scale [E, n, 1]`` is the MoE expert GEMM, and the per-token activation
   quant that DeepEP dispatch already produces feeds it directly (so the MoE
   path needs no plugin change and no activation re-quantization).
4. **Negative result, pinned on purpose**: the 128x128 block-scale layout
   documented in ``kunlun_ops._deepgemm`` is NOT implemented by this build
   (NaN / >20% error).  If these assertions ever start failing, the kernel
   gained block support and ``quantize_weight_blockwise_int8`` becomes usable
   -- that would buy ~0.3-0.5 points of accuracy.

Run on a single card::

    XPU_VISIBLE_DEVICES=4 CUDA_VISIBLE_DEVICES=4 \
        python -m pytest test_eb5_int8_block_quant_device.py -q -s
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.internal.layers.quantization import blockwise_int8_patch as bi8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs the Kunlun XPU runtime"
)

# (name, n, k) taken from the EB5 checkpoint's projection shapes.
EB5_SHAPES = [
    ("q_a_proj", 1024, 4096),
    ("q_b_proj", 16384, 1024),
    ("kv_b_proj", 28672, 512),
    ("o_proj", 4096, 16384),
    ("dense_mlp_gate", 14336, 4096),
    ("dense_mlp_down", 4096, 14336),
    ("fc1_latent", 2048, 4096),
    ("expert_w13", 4096, 2048),
]
M_LIST = [1, 8, 128]


def _stats(ref, out):
    ref = ref.float()
    out = out.float()
    d = (out - ref).abs()
    return {
        "rel": ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item(),
        "max": d.max().item(),
        "mean": d.mean().item(),
    }


def _case(n, k, m, seed=0, outlier=2.0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.02).bfloat16()
    x = torch.randn(m, k, generator=g).bfloat16()
    if outlier > 1:
        rows = torch.randperm(n, generator=g)[: max(1, n // 128)]
        w[rows] = (w[rows].float() * outlier).bfloat16()
        cols = torch.randperm(k, generator=g)[: max(1, k // 64)]
        x[:, cols] = (x[:, cols].float() * outlier).bfloat16()
    return w.to("cuda"), x.to("cuda")


# ---------------------------------------------------------------------------
# 1. quant2d scale convention
# ---------------------------------------------------------------------------


def test_quant2d_stores_raw_absmax():
    import kunlun_ops

    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 3.0
    q = torch.empty(64, 128, dtype=torch.int8, device="cuda")
    s = torch.empty(64, 1, dtype=torch.float32, device="cuda")
    kunlun_ops.quant2d(x, q, s, force_sdnn=True)
    ref_amax = x.float().abs().amax(dim=1, keepdim=True)
    assert torch.allclose(s, ref_amax, rtol=1e-3, atol=1e-6), (
        s[:4].tolist(),
        ref_amax[:4].tolist(),
    )
    ref_q = torch.round(x.float() * (127.0 / ref_amax)).clamp(-127, 127)
    assert (q.float() - ref_q).abs().max().item() <= 1.0


def test_activation_pertoken_quant_matches_torch_reference():
    x = torch.randn(8, 2048, device="cuda", dtype=torch.bfloat16) * 2.0
    q, s = bi8.quantize_act_pertoken_int8(x)
    qr, sr = bi8.quantize_act_pertoken_int8_torch(x)
    assert q.shape == qr.shape and s.shape == sr.shape == (8, 1)
    assert torch.allclose(s, sr, rtol=1e-3, atol=1e-6)
    assert (q.float() - qr.float()).abs().max().item() <= 1.0


# ---------------------------------------------------------------------------
# 2. Dense W8A8 INT8 GEMM (per-token x, per-channel w)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,n,k", EB5_SHAPES)
@pytest.mark.parametrize("m", M_LIST)
def test_dense_int8_linear(name, n, k, m):
    w, x = _case(n, k, m, seed=n + k)
    qw, sw = bi8.quantize_weight_perchannel_int8(w)
    qx, sx = bi8.quantize_act_pertoken_int8(x)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    bi8.int8_pertoken_linear(qx, sx, qw, sw, out)
    torch.cuda.synchronize()
    ref = bi8.perchannel_reference_matmul(x, qw, sw)
    st = _stats(ref, out)
    print(f"[dense] {name} n={n} k={k} m={m}: {st['rel'] * 100:.3f}%")
    assert st["rel"] < 0.005, (name, m, st)  # kernel must match the model exactly


@pytest.mark.parametrize("m", M_LIST)
def test_dense_int8_linear_with_fp32_bias(m):
    n, k = 2048, 2048
    w, x = _case(n, k, m, seed=1234)
    qw, sw = bi8.quantize_weight_perchannel_int8(w)
    qx, sx = bi8.quantize_act_pertoken_int8(x)
    bias = torch.randn(n, dtype=torch.float32, device="cuda")
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    bi8.int8_pertoken_linear(qx, sx, qw, sw, out, bias)
    torch.cuda.synchronize()
    ref = bi8.perchannel_reference_matmul(x, qw, sw) + bias
    st = _stats(ref, out)
    print(f"[dense+bias] m={m}: {st['rel'] * 100:.3f}%")
    assert st["rel"] < 0.005, st


def test_dense_int8_matches_bf16_within_quantization_error():
    """End-to-end (quantize + GEMM) vs the BF16 reference, real EB5 shape."""
    n, k, m = 4096, 32768, 8
    w, x = _case(n, k, m, seed=77)
    qw, sw = bi8.quantize_weight_perchannel_int8(w)
    qx, sx = bi8.quantize_act_pertoken_int8(x)
    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    bi8.int8_pertoken_linear(qx, sx, qw, sw, out)
    torch.cuda.synchronize()
    st = _stats(x.float() @ w.float().T, out)
    print(f"[dense vs bf16] o_proj-like: {st['rel'] * 100:.3f}%")
    assert st["rel"] < 0.02, st


# ---------------------------------------------------------------------------
# 3. Grouped (MoE) INT8 GEMM
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("m_per_expert", [1, 4, 32])
def test_grouped_moe_gemm_pertoken_perchannel(num_experts, m_per_expert):
    import kunlun_ops

    n, k = 4096, 2048  # w13 of one EB5 expert (latent MoE, inter 2048)
    g = torch.Generator().manual_seed(num_experts * 100 + m_per_expert)
    w = (torch.randn(num_experts, n, k, generator=g) * 0.02).bfloat16().cuda()
    counts = torch.full((num_experts,), m_per_expert, dtype=torch.int32, device="cuda")
    total = num_experts * m_per_expert
    x = torch.randn(total, k, generator=g).bfloat16().cuda()

    qw = torch.empty_like(w, dtype=torch.int8)
    sw = torch.empty(num_experts, n, 1, dtype=torch.float32, device="cuda")
    for e in range(num_experts):
        q, s = bi8.quantize_weight_perchannel_int8(w[e])
        qw[e].copy_(q)
        sw[e].copy_(s)
    qx, sx = bi8.quantize_act_pertoken_int8(x)

    out = torch.empty(total, n, dtype=torch.bfloat16, device="cuda")
    kunlun_ops.m_grouped_gemm_I8_I8_bf16_nt_contiguous_v3(
        (qx, sx), (qw, sw), out, counts
    )
    torch.cuda.synchronize()

    ref = torch.empty(total, n, dtype=torch.float32, device="cuda")
    for e in range(num_experts):
        sl = slice(e * m_per_expert, (e + 1) * m_per_expert)
        ref[sl] = bi8.perchannel_reference_matmul(x[sl], qw[e], sw[e])
    st = _stats(ref, out)
    print(f"[grouped] E={num_experts} m/E={m_per_expert}: {st['rel'] * 100:.3f}%")
    assert st["rel"] < 0.005, st


def test_grouped_moe_gemm_deepep_dispatcher_scales():
    """Exactly the DeepEP shape: dispatcher emits x_int8 + [m,1] scale on the
    token axis while the expert weights are per-channel."""
    import kunlun_ops

    num_experts, n, k, mpe = 8, 4096, 2048, 4
    g = torch.Generator().manual_seed(7)
    w = (torch.randn(num_experts, n, k, generator=g) * 0.02).bfloat16().cuda()
    counts = torch.full((num_experts,), mpe, dtype=torch.int32, device="cuda")
    total = num_experts * mpe
    x = torch.randn(total, k, generator=g).bfloat16().cuda()

    qw = torch.empty_like(w, dtype=torch.int8)
    sw = torch.empty(num_experts, n, 1, dtype=torch.float32, device="cuda")
    for e in range(num_experts):
        q, s = bi8.quantize_weight_perchannel_int8(w[e])
        qw[e].copy_(q)
        sw[e].copy_(s)

    qx = torch.empty(total, k, dtype=torch.int8, device="cuda")
    sx = torch.empty(total, 1, dtype=torch.float32, device="cuda")
    kunlun_ops.quant2d(x, qx, sx, force_sdnn=True)  # deepep.dispatch_a

    out = torch.empty(total, n, dtype=torch.bfloat16, device="cuda")
    kunlun_ops.m_grouped_gemm_I8_I8_bf16_nt_contiguous_v3(
        (qx, sx), (qw, sw), out, counts
    )
    torch.cuda.synchronize()
    xd = qx.float() * (sx / 127.0)
    ref = torch.empty(total, n, dtype=torch.float32, device="cuda")
    for e in range(num_experts):
        sl = slice(e * mpe, (e + 1) * mpe)
        ref[sl] = xd[sl] @ bi8.dequantize_weight_perchannel_int8(qw[e], sw[e]).T
    st = _stats(ref, out)
    print(f"[grouped/deepep] {st['rel'] * 100:.3f}%")
    assert st["rel"] < 0.005, st


# ---------------------------------------------------------------------------
# 4. Pinned negative result: no 128x128 block-scale INT8 GEMM
# ---------------------------------------------------------------------------


def test_kernel_has_no_block_scale_support():
    """Documents *why* the granularity is per-channel.

    ``kunlun_ops`` advertises ``w_scale [ceil(n/128), ceil(k/128)]`` for the
    ``I8_I8`` GEMMs, but this build silently mis-reads it.  If this test starts
    failing, a block-scale kernel landed: switch the quantizer to
    ``quantize_weight_blockwise_int8`` and re-measure (expect ~1.0-1.3%).
    """
    import kunlun_ops

    n, k, m = 512, 512, 8
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(n, k, generator=g) * 0.02).bfloat16().cuda()
    x = torch.randn(m, k, generator=g).bfloat16().cuda()
    qw, sw = bi8.quantize_weight_blockwise_int8(w)
    qx, sx = bi8.quantize_act_blockwise_int8_torch(x)
    qx = qx.cuda()
    sx = sx.cuda()
    assert sw.shape == (4, 4) and sx.shape == (8, 4)

    ideal = bi8.block_reference_matmul(x, qw, sw)
    out = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
    kunlun_ops.xpu_kunlun_ops.matmul(
        qx, qw, out, False, True, 1.0, 0.0, None, None, None, sx, sw
    )
    torch.cuda.synchronize()
    rel = _stats(ideal, out)["rel"]
    print(f"[block-scale gemm] rel={rel * 100:.1f}% (expected to be junk)")

    grouped_bad = False
    counts = torch.full((4,), 4, dtype=torch.int32, device="cuda")
    wg = torch.stack([qw] * 4)
    sg = torch.stack([sw] * 4)
    xg = torch.randn(16, k, generator=g).bfloat16().cuda()
    qxg, sxg = bi8.quantize_act_pertoken_int8(xg)
    outg = torch.zeros(16, n, dtype=torch.bfloat16, device="cuda")
    kunlun_ops.m_grouped_gemm_I8_I8_bf16_nt_contiguous_v3(
        (qxg, sxg), (wg, sg), outg, counts
    )
    torch.cuda.synchronize()
    grouped_bad = bool(torch.isnan(outg).any().item())

    assert rel > 0.05 or grouped_bad, (
        "the kernel appears to understand 128x128 block scales now "
        f"(dense rel={rel:.3f}, grouped_nan={grouped_bad}) -- revisit the "
        "granularity choice"
    )
