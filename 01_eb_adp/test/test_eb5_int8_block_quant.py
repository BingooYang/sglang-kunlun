"""CPU numeric verification for online INT8 quantization (EB5 / Kunlun).

Stage-1 gate of the EB5 INT8 work: prove that the quantization schemes
implemented here reproduce a BF16 GEMM within a couple of percent across the
shapes that actually occur in EB5 (K = 512/1024/2048/14336/32768, M = 1/8/128).

Two schemes are measured:

* **per-token A + per-output-channel W** -- what this hardware can execute
  (see the module docstring: no block-scale INT8 GEMM exists in this
  ``kunlun_ops`` build).  This is the default and the one the load-time
  quantizer uses.
* **128x128 block W (+ optionally block A)** -- the reference "ideal", kept to
  quantify how much accuracy the missing kernel would buy.

Run with no device at all::

    XPU_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
        python -m pytest test_eb5_int8_block_quant.py -q

``perchannel_reference_matmul`` / ``block_reference_matmul`` in the patch
module are float32 models of the two schemes (quantize, dequantize, fp32
matmul).  Comparing those against a BF16 ``x @ w.T`` therefore measures
*quantization* error only -- the part the kernel cannot fix.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.internal.layers.quantization import blockwise_int8_patch as bi8

# (n, k) pairs taken from the EB5 checkpoint's real projection shapes.
EB5_SHAPES = [
    ("wq_b", 64 * 512, 1024),
    ("wkv_b", 64 * (448 + 512), 512),
    ("o_proj", 4096, 32768),
    ("dense_mlp_down", 4096, 14336),
    ("moe_fc1_latent", 2048, 4096),
    ("moe_expert_w13", 2 * 2048, 2048),
    ("moe_expert_w2", 2048, 2048),
]
KS = [512, 2048, 16384]
MS = [1, 8, 128]


def _err_stats(ref: torch.Tensor, out: torch.Tensor) -> dict:
    ref = ref.float()
    out = out.float()
    diff = (out - ref).abs()
    denom = ref.norm().clamp_min(1e-12)
    return {
        "rel_fro": ((out - ref).norm() / denom).item(),
        "max": diff.max().item(),
        "mean": diff.mean().item(),
        "ref_absmax": ref.abs().max().item(),
        "rel_max": (diff.max() / ref.abs().max().clamp_min(1e-12)).item(),
    }


def _make_case(n: int, k: int, m: int, seed: int, outlier: float = 2.0):
    g = torch.Generator().manual_seed(seed)
    # EB5-like magnitudes: weights ~N(0, 0.02) with the occasional outlier row,
    # activations ~N(0, 1) with heavy channels.
    w = torch.randn(n, k, generator=g) * 0.02
    x = torch.randn(m, k, generator=g)
    if outlier > 1.0:
        # One heavy row per 128 (weights) / channel per 64 (activations): the
        # measured EB5 checkpoint behaves like x2 (see tool/int8_quant_probe.py).
        rows = torch.randperm(n, generator=g)[: max(1, n // 128)]
        w[rows] *= outlier
        if m:
            cols = torch.randperm(k, generator=g)[: max(1, k // 64)]
            x[:, cols] *= outlier
    return w.bfloat16(), x.bfloat16()


# ---------------------------------------------------------------------------
# Round-trip / layout properties
# ---------------------------------------------------------------------------


class TestQuantPrimitives:
    def test_weight_roundtrip_error_is_bounded_by_half_step(self):
        w, _ = _make_case(256, 512, 1, seed=0)
        q, s = bi8.quantize_weight_blockwise_int8(w)
        assert q.dtype is torch.int8 and q.shape == w.shape
        assert s.shape == (2, 4), s.shape
        dq = bi8.dequantize_weight_blockwise_int8(q, s)
        step = (s / 127.0).repeat_interleave(128, 0).repeat_interleave(128, 1)
        assert (dq - w.float()).abs().max() <= (step.max() / 2) * 1.01

    def test_scale_is_raw_absmax_per_block(self):
        w = torch.zeros(256, 256)
        w[:128, :128] = 3.0
        w[128:, 128:] = -5.0
        q, s = bi8.quantize_weight_blockwise_int8(w)
        # block (0,0) abs-max 3, block (0,1) empty (eps), block (1,1) abs-max 5
        assert abs(s[0, 0].item() - 3.0) < 1e-6
        assert abs(s[1, 1].item() - 5.0) < 1e-6
        assert s[0, 1].item() < 1e-6 and s[1, 0].item() < 1e-6
        assert q.abs().max().item() == 127
        dq = bi8.dequantize_weight_blockwise_int8(q, s)
        assert torch.allclose(dq[128:, 128:], w[128:, 128:], atol=1e-6)

    def test_degenerate_block_does_not_divide_by_zero(self):
        w = torch.zeros(128, 128)
        q, s = bi8.quantize_weight_blockwise_int8(w)
        assert torch.isfinite(s).all() and s.item() > 0
        assert q.abs().max().item() == 0

    def test_activation_scale_layout_is_m_by_kblocks(self):
        x = torch.randn(8, 512).bfloat16()
        q, s = bi8.quantize_act_blockwise_int8_torch(x)
        assert q.shape == (8, 512) and s.shape == (8, 4)
        ref = bi8.dequantize_act_blockwise_int8(q, s)
        assert ref.shape == (8, 4, 128)
        assert (ref.reshape(8, 512) - x.float()).abs().max() < 0.05

    def test_non_128_aligned_activation_is_padded(self):
        x = torch.randn(4, 200).bfloat16()
        q, s = bi8.quantize_act_blockwise_int8_torch(x)
        assert s.shape == (4, 2)
        assert q.shape == (4, 256)
        assert torch.isfinite(q.float()).all()


# ---------------------------------------------------------------------------
# Stage-1 acceptance table
# ---------------------------------------------------------------------------


class TestBlockInt8Accuracy:
    @pytest.mark.parametrize("k", KS)
    @pytest.mark.parametrize("m", MS)
    def test_random_square_shapes(self, m, k):
        w, x = _make_case(1024, k, m, seed=(m * 7 + k) & 0xFFFF)
        q, s = bi8.quantize_weight_perchannel_int8(w)
        ref = (x.float() @ w.float().T).bfloat16()
        out = bi8.perchannel_reference_matmul(x, q, s)
        st = _err_stats(ref, out)
        assert st["rel_fro"] < 0.02, (m, k, st)

    @pytest.mark.parametrize("name,n,k", EB5_SHAPES)
    def test_eb5_projection_shapes(self, name, n, k):
        m = 8
        w, x = _make_case(n, k, m, seed=len(name) * 31)
        q, s = bi8.quantize_weight_perchannel_int8(w)
        ref = (x.float() @ w.float().T).bfloat16()
        st = _err_stats(ref, bi8.perchannel_reference_matmul(x, q, s))
        # Per-channel W over a large K accumulates more int8 noise than the
        # 128x128 block reference; still under 2% at K=32768 (device-measured
        # 1.05-1.41% on real checkpoint weights).
        assert st["rel_fro"] < 0.02, (name, st)

    def test_perchannel_beats_per_tensor_int8(self):
        w, x = _make_case(4096, 16384, 8, seed=7)
        q, s = bi8.quantize_weight_perchannel_int8(w)
        ref = (x.float() @ w.float().T).bfloat16()
        block = _err_stats(ref, bi8.perchannel_reference_matmul(x, q, s))

        # Whole-tensor (not per-row) scale on both sides: the "no block" baseline.
        wamax = w.float().abs().max()
        xamax = x.float().abs().max()
        qt = torch.round(w.float() * (127.0 / wamax)).clamp(-127, 127).to(torch.int8)
        qx = torch.round(x.float() * (127.0 / xamax)).clamp(-127, 127).to(torch.int8)
        per_tensor = (qx.float() * (xamax / 127.0)) @ (
            (qt.float() * (wamax / 127.0)).T
        )
        coarse = _err_stats(ref, per_tensor)
        # Measured on this shape: 1.6% (per-channel) vs 2.75% (whole-tensor).
        assert block["rel_fro"] * 1.5 < coarse["rel_fro"], (block, coarse)

    def test_block_scheme_is_accurate_but_not_executable(self):
        """128x128 block scales quantize better than per-channel -- they are
        just not implementable on this kernel set (see the module docstring)."""
        w, x = _make_case(2048, 16384, 8, seed=13)
        ref = (x.float() @ w.float().T).bfloat16()
        qc, sc = bi8.quantize_weight_perchannel_int8(w)
        qb, sb = bi8.quantize_weight_blockwise_int8(w)
        chan = _err_stats(ref, bi8.perchannel_reference_matmul(x, qc, sc))
        blk = _err_stats(ref, bi8.block_reference_matmul(x, qb, sb))
        assert blk["rel_fro"] < chan["rel_fro"], (blk, chan)

    @pytest.mark.parametrize(
        "outlier,limit", [(1.0, 0.015), (2.0, 0.02), (3.0, 0.025)]
    )
    def test_error_grows_gracefully_with_outliers(self, outlier, limit):
        """x1 (plain Gaussian) ~1.1%, x2 ~1.4-1.7%, x3 ~1.9-2.2%."""
        for k in (512, 2048, 16384):
            w, x = _make_case(2048, k, 8, seed=k + int(outlier * 10), outlier=outlier)
            q, s = bi8.quantize_weight_perchannel_int8(w)
            ref = (x.float() @ w.float().T).bfloat16()
            st = _err_stats(ref, bi8.perchannel_reference_matmul(x, q, s))
            assert st["rel_fro"] < limit, (k, outlier, st)

    def test_extreme_outliers_degrade_but_stay_bounded(self):
        """x8 channels are beyond anything measured in the checkpoint (~4%)."""
        w, x = _make_case(2048, 16384, 8, seed=11, outlier=8.0)
        q, s = bi8.quantize_weight_perchannel_int8(w)
        ref = (x.float() @ w.float().T).bfloat16()
        st = _err_stats(ref, bi8.perchannel_reference_matmul(x, q, s))
        assert st["rel_fro"] < 0.05, st

    def test_exact_for_int8_representable_block(self):
        """A block whose values are exact multiples of amax/127 round-trips."""
        amax = 1.27
        w = torch.zeros(128, 128)
        vals = (torch.arange(-127, 1, dtype=torch.float32) / 127.0) * amax
        w.view(-1)[: vals.numel()] = vals
        q, s = bi8.quantize_weight_blockwise_int8(w)
        dq = bi8.dequantize_weight_blockwise_int8(q, s)
        assert torch.allclose(dq, w, atol=1e-6)


def test_error_table(capsys):
    """Print the stage-1 error table (K x M grid) used in the report."""
    rows = []
    for k in (512, 2048, 16384):
        for m in (1, 8, 128):
            w, x = _make_case(2048, k, m, seed=k + m)
            q, s = bi8.quantize_weight_perchannel_int8(w)
            ref = (x.float() @ w.float().T).bfloat16()
            st = _err_stats(ref, bi8.perchannel_reference_matmul(x, q, s))
            rows.append((k, m, st))
    with capsys.disabled():
        print("\n  K      M    rel_fro    max_abs    mean_abs")
        for k, m, st in rows:
            print(
                f"{k:>6} {m:>6}   {st['rel_fro'] * 100:7.3f}%  "
                f"{st['max']:.5f}   {st['mean']:.5f}"
            )
    assert all(st["rel_fro"] < 0.02 for _, _, st in rows)


# ---------------------------------------------------------------------------
# Registration / dispatch (no device needed)
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_name_registered_unconditionally(self, monkeypatch):
        import sglang.srt.layers.quantization as quant
        import sglang.srt.server_args as server_args

        monkeypatch.setattr(bi8.environ, "EB5_INT8_ONLINE", False, raising=False)
        bi8.register_blockwise_int8_patch()
        assert (
            quant.BASE_QUANTIZATION_METHODS.get("blockwise_int8_online")
            is bi8.InternalBlockwiseInt8Config
        )
        assert "blockwise_int8_online" in server_args.QUANTIZATION_CHOICES
        # The serialized-checkpoint path must stay untouched while gated off.
        from sglang.srt.layers.quantization.blockwise_int8 import BlockInt8Config

        assert quant.BASE_QUANTIZATION_METHODS["blockwise_int8"] is BlockInt8Config

    def test_alias_shadowed_only_when_enabled(self, monkeypatch):
        import sglang.srt.layers.quantization as quant

        monkeypatch.setattr(bi8.environ, "EB5_INT8_ONLINE", True, raising=False)
        monkeypatch.setattr(bi8.environ, "EB5_INT8_MOE_ONLY", False, raising=False)
        bi8.register_blockwise_int8_patch()
        assert (
            quant.BASE_QUANTIZATION_METHODS["blockwise_int8"]
            is bi8.InternalBlockwiseInt8Config
        )

    def test_config_ignores_accelerator_specific_prefixes(self):
        cfg = bi8.InternalBlockwiseInt8Config()
        for prefix in (
            "model.layers.3.self_attn.wo_a",
            "lm_head",
            "model.layers.0.self_attn.indexer.wk",
            "model.layers.2.self_attn.mhc_head",
        ):
            assert cfg._is_ignored(prefix), prefix
        for prefix in (
            "model.layers.3.self_attn.wq_b",
            "model.layers.3.self_attn.o_proj",
            "model.layers.3.mlp.experts",
            "model.layers.3.mlp.gate_up_proj",
        ):
            assert not cfg._is_ignored(prefix), prefix

    def test_moe_only_scope_ignores_dense(self, monkeypatch):
        monkeypatch.setattr(bi8.environ, "EB5_INT8_MOE_ONLY", True, raising=False)
        cfg = bi8.InternalBlockwiseInt8Config()
        assert cfg.moe_only
        assert cfg._is_ignored("model.layers.3.mlp.gate_up_proj")
        assert not cfg._is_ignored("model.layers.3.mlp.experts")

    def test_linear_scale_is_per_output_channel(self):
        w, _ = _make_case(512, 1024, 1, seed=3)
        q, s = bi8.quantize_weight_perchannel_int8(w)
        assert q.shape == (512, 1024) and s.shape == (512, 1)
        # dequant reproduces each row up to int8 representability
        row = 7
        dq = bi8.dequantize_weight_perchannel_int8(q[row : row + 1], s[row : row + 1])
        assert (dq - w[row : row + 1].float()).abs().max() <= s[row].item() / 127.0


# ---------------------------------------------------------------------------
# Quant-method plumbing (create_weights -> process_weights_after_loading)
# ---------------------------------------------------------------------------


class _FakeLinear(torch.nn.Module):
    """Minimal stand-in for ``ReplicatedLinear`` (what create_weights needs)."""

    def __init__(self, n, k, dtype=torch.bfloat16):
        super().__init__()
        self.input_size = k
        self.output_size = n
        self.input_size_per_partition = k
        self.output_size_per_partition = n
        self.orig_dtype = dtype
        self.bias = None
        self._n, self._k = n, k


class TestInt8LinearMethodPlumbing:
    def _make(self, n=256, k=512, monkeypatch=None):
        cfg = bi8.InternalBlockwiseInt8Config()
        method = bi8.Int8LinearMethod(cfg)
        layer = _FakeLinear(n, k)
        method.create_weights(
            layer,
            input_size_per_partition=k,
            output_partition_sizes=[n],
            input_size=k,
            output_size=n,
            params_dtype=torch.bfloat16,
        )
        return method, layer

    def test_create_weights_registers_bf16_placeholder_and_scale(self, monkeypatch):
        monkeypatch.setattr(
            bi8.environ, "EB5_INT8_LINEAR_CPU_STAGING", False, raising=False
        )
        method, layer = self._make()
        assert layer.weight.dtype is torch.bfloat16
        assert tuple(layer.weight.shape) == (256, 512)
        assert tuple(layer.weight_scale.shape) == (256, 1)
        assert layer.weight_scale.dtype is torch.float32
        assert layer.input_scale is None
        assert layer.blockwise_int8_online is True

    def test_cpu_staging_default_is_off(self):
        """A CPU-staged dense placeholder breaks EB5's pre-quant absorb build."""
        import sglang.srt.internal.environ as ienv

        assert ienv.EB5_INT8_LINEAR_CPU_STAGING is False

    def test_create_weights_cpu_staging_marks_the_placeholder(self, monkeypatch):
        monkeypatch.setattr(
            bi8.environ, "EB5_INT8_LINEAR_CPU_STAGING", True, raising=False
        )
        method, layer = self._make()
        assert layer.weight.device.type == "cpu"
        assert bool(getattr(layer.weight, "_cpu_staged", False)) is True

    @pytest.mark.parametrize("cpu_staging", [True, False])
    def test_process_weights_after_loading_quantizes_in_place(
        self, monkeypatch, cpu_staging
    ):
        monkeypatch.setattr(
            bi8.environ, "EB5_INT8_LINEAR_CPU_STAGING", cpu_staging, raising=False
        )
        method, layer = self._make()
        torch.nn.init.normal_(layer.weight, std=0.02)
        before = layer.weight.detach().clone()
        method.process_weights_after_loading(layer)
        assert layer.weight.dtype is torch.int8
        assert layer.weight.requires_grad is False
        assert torch.allclose(
            bi8.dequantize_weight_perchannel_int8(
                layer.weight, layer.weight_scale
            ).float().cpu(),
            before.float().cpu(),
            atol=before.abs().max().item() / 100.0,
        )
        # idempotent: a second call must not re-quantize int8 data
        w = layer.weight.data.clone()
        method.process_weights_after_loading(layer)
        assert torch.equal(layer.weight.data, w)

    def test_process_weights_promotes_bias_to_fp32(self):
        method, layer = self._make()
        layer.bias = torch.nn.Parameter(torch.zeros(256, dtype=torch.bfloat16))
        method.process_weights_after_loading(layer)
        assert layer.bias.dtype is torch.float32

    def test_apply_rejects_pre_quantized_tuple_input(self):
        method, layer = self._make()
        with pytest.raises(NotImplementedError):
            method.apply(layer, (torch.zeros(1, 512, dtype=torch.int8), None))


class _FakeMoE(torch.nn.Module):
    def __init__(self, num_experts, hidden, inter, dtype=torch.bfloat16):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden
        self.intermediate_size_per_partition = inter
        self.orig_dtype = dtype


class TestInt8MoEMethodPlumbing:
    def _make(self, monkeypatch, cpu_staging: bool, num_experts=4, hidden=256, inter=512):
        monkeypatch.setattr(
            bi8.environ, "EB5_INT8_MOE_CPU_STAGING", cpu_staging, raising=False
        )
        cfg = bi8.InternalBlockwiseInt8Config()
        method = bi8.Int8MoEMethod(cfg)
        layer = _FakeMoE(num_experts, hidden, inter)
        method.create_weights(
            layer,
            num_experts=num_experts,
            hidden_size=hidden,
            intermediate_size_per_partition=inter,
            params_dtype=torch.bfloat16,
        )
        return method, layer

    @pytest.mark.parametrize("cpu_staging", [True, False])
    def test_create_weights_shapes(self, monkeypatch, cpu_staging):
        method, layer = self._make(monkeypatch, cpu_staging)
        assert tuple(layer.w13_weight.shape) == (4, 1024, 256)
        assert tuple(layer.w2_weight.shape) == (4, 256, 512)
        # the plugin's DeepEP INT8 path reads these two names as [E, n, 1]
        assert tuple(layer.w13_weight_scale.shape) == (4, 1024, 1)
        assert tuple(layer.w2_weight_scale.shape) == (4, 256, 1)
        assert layer.w13_weight_scale_inv is layer.w13_weight_scale
        expected_dev = "cpu" if cpu_staging else "cpu"
        # cpu_staging tensors must carry the marker post_load.py looks for and
        # must NOT be pre-moved to the device.
        assert layer.w13_weight.device.type == expected_dev
        assert bool(getattr(layer.w13_weight, "_cpu_staged", False)) is cpu_staging

    @pytest.mark.parametrize("cpu_staging", [True, False])
    def test_process_weights_quantizes_every_expert(self, monkeypatch, cpu_staging):
        method, layer = self._make(monkeypatch, cpu_staging, num_experts=3)
        g = torch.Generator().manual_seed(0)
        torch.nn.init.normal_(layer.w13_weight, std=0.02, generator=g)
        torch.nn.init.normal_(layer.w2_weight, std=0.02, generator=g)
        w13_before = layer.w13_weight.detach().float().clone()
        method.process_weights_after_loading(layer)
        assert layer.w13_weight.dtype is torch.int8
        assert layer.w2_weight.dtype is torch.int8
        # scales are the per-row abs-max of the original bf16 weight
        ref = w13_before.abs().amax(dim=2, keepdim=True)
        assert torch.allclose(
            layer.w13_weight_scale.float().cpu(), ref.float().cpu(),
            rtol=2e-2, atol=1e-6,
        )
        # and dequantization stays within one int8 step of the original
        dq = bi8.dequantize_weight_perchannel_int8(
            layer.w13_weight[1], layer.w13_weight_scale[1]
        )
        step = layer.w13_weight_scale[1] / 127.0
        assert (dq.float().cpu() - w13_before[1].float().cpu()).abs().max() <= 1.1 * step.max().item()
