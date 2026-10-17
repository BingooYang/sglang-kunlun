"""EB5(Ernie5) decoder-layer 7 模块 **真实 kernel** 覆盖单测。

用途：适配新硬件时，按 kernel 名快速筛出"哪些 kernel 在本硬件上跑不起来"。
与旧版"打桩冒烟"版本的区别：**不再把 kernel 换成 stub / lambda**，每个测试类都
真实执行对应模块用到的 kernel，并对结果做数值/形状断言。

层内 7 个模块与真实执行的 kernel：

    前处理   -> EB5LayerRMSNorm.forward        rms_norm (ATen CUDA RMSNorm)
    hc_pre  -> EB5DecoderLayer._mhc_pre        deep_gemm tf32_hc_prenorm_gemm +
                                               tilelang mhc_pre_big_fuse_with_norm_tilelang
    HCA     -> EB5AttentionCSAHCA.forward      整模块真跑(MQALayer): 投影 GEMM + q/kv
                                               norm + rope + compressor(compress_norm_rope_store)
                                               + KV store + attention(dsv4 backend)
    MLA     -> EB5AttentionMLA.forward_prepare/forward_core
                                               整模块真跑: fused qkv_a GEMM + q/kv norm +
                                               rope + q_b_proj + w_kc absorb bmm +
                                               attention(triton MLA) + gate + o_proj
    hc_post -> EB5DecoderLayer._mhc_post        tilelang mhc_post_tilelang
    moe     -> EB5MoE.forward                  moe_fused_gate (router topk) + triton
                                               fused_moe (expert GEMM) + silu_and_mul；
                                               另直调 EB5 专属 eb5_fused_topk 融合路由 kernel
    后处理   -> EB5LogitsProcessor._multimax_selu
                                               triton _multimax_selu_kernel (fused_multimax_selu)

执行方式（"整模块 forward 真跑" vs "直调 kernel"）：

- **全部 7 个模块都是"整模块 forward 真跑"**，只 mock 权重（随机张量）与无法构造的
  外部资源。
- HCA / MLA 需要真实 `DeepSeekV4Config` + KV pool + `ForwardBatch` + 真实 attention
  backend；脚手架复用 sglang 自带的 attention 单测 kit
  (``sglang.test.kits.attention_unittest.attention_methods.dsv4_attention`` /
  ``mla_attention``) 构造真实的 ``DeepseekV4AttnBackend`` / ``TritonAttnBackend``、
  ``DeepSeekV4TokenToKVPool`` / ``MLATokenToKVPool`` 与 ``ForwardBatch``，再用
  ``set_forward_context(ForwardContext(attn_backend=...))`` 挂上，然后调用模块**真实的
  forward**。这样模块内部触发的所有 kernel（投影 / norm / rope / compressor / 量化 /
  attention / gate / o_proj）都会被真实执行，而不是手工只点几个 kernel。

并行上下文：mhc kernel 内部读 ``get_tp_group`` / ``use_symmetric_memory`` /
``is_allocation_symmetric`` / ``is_dsa_prefill_cp_interleave``。这里**不 patch** 它们，
而是在模块顶层建立一个真实的单进程 ``torch.distributed``（world_size=1）并发布
``ServerArgs`` + ``get_parallel().override(attn_tp_size=1)``，让它们真实可用。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[5]  # <repo>/test_internal/internal/models/eb5/layer/xxx.py
_PYTHON_DIR = _REPO_ROOT / "python"
for _p in (str(_PYTHON_DIR), str(_HERE.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _init_single_process_dist() -> None:
    """建立真实的单进程 distributed 环境（world_size=1）。

    必须在 ``patch_hf_strict_for_tests()`` 之前调用：后者会把 ``torch._dynamo``
    换成轻量 stub，而 ``init_process_group`` 会用到真实的 ``torch._dynamo``。
    """
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    # 按可见 GPU 派生端口，避免与其他单测/服务撞端口（与 e2e 端口隔离思路一致）。
    _first_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    _master_port = 29600 + (int(_first_gpu) if _first_gpu.isdigit() else 0) * 13
    os.environ["MASTER_PORT"] = str(_master_port)
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{_master_port}",
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)


_init_single_process_dist()

from _hf_strict_compat import patch_hf_strict_for_tests

patch_hf_strict_for_tests()

from sglang.kernels.ops.layernorm import mhc as mhc_ops
from sglang.srt.internal.models.eb5 import ernie5
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase
from sglang.test_internal.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, num_gpus=1)

# 发布 ServerArgs + 并行 override，让 mhc / 并行上下文读取真实生效（非 patch）。
_SERVER_ARGS = ServerArgs(model_path="dummy")
_SERVER_ARGS.enable_dp_attention = False
set_global_server_args_for_scheduler(_SERVER_ARGS)
_PARALLEL_OVERRIDE = get_parallel().override(
    attn_tp_size=1, attn_tp_rank=0, attn_cp_size=1, attn_cp_rank=0
)
_PARALLEL_OVERRIDE.__enter__()

# MoE 后端配置（EB5MoE.__init__ 读 get_moe_a2a_backend / get_moe_impl_class）。
from sglang.srt.layers.moe.utils import initialize_moe_config  # noqa: E402

initialize_moe_config()

_DEVICE = "cuda"
_DTYPE = torch.bfloat16
_TOKENS = 4

# ── HCA(``EB5AttentionCSAHCA`` / DeepSeek-V4 ``MQALayer``) 维度 ──
# 必须与 dsv4 attention backend / DeepSeekV4TokenToKVPool 的硬编码几何一致：
# head_dim = qk_nope(448) + rope(64) = 512，kv_lora_rank = v_head_dim = 512，
# page_size = 256（backend 断言），h_q = 64（FlashMLA sparse 支持集）。
_HCA_HIDDEN = 512
_HCA_HEADS = 64
_HCA_HEAD_DIM = 512
_HCA_QK_NOPE = 448
_HCA_QK_ROPE = 64
_HCA_KV_LORA = 512
_HCA_V_HEAD = 512
_HCA_Q_LORA = 1024
_HCA_O_LORA = 1024
_HCA_O_GROUPS = 8
_HCA_MAX_POS = 256
_HCA_PAGE = 256
_HCA_SWA_SIZE = 1024

# ── MLA(``EB5AttentionMLA``) 维度：MLA+HCA 混合里 full MLA 层的几何 ──
# qk_nope=192 + rope=64 = head_dim 256，v_head_dim = kv_lora_rank = 512。
_MLA_HIDDEN = 512
_MLA_HEADS = 8
_MLA_QK_NOPE = 192
_MLA_QK_ROPE = 64
_MLA_V_HEAD = 512
_MLA_KV_LORA = 512
_MLA_Q_LORA = 512
_MLA_MAX_POS = 256
_MLA_PAGE = 64

# ── mHC 维度 ──
_HIDDEN = 4096
_HC_MULT = 4
_HC_MULT3 = _HC_MULT * (2 + _HC_MULT)


def _rand(*shape, dtype=_DTYPE):
    return torch.randn(*shape, dtype=dtype, device=_DEVICE)


def _require_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")


def _randomize(module: nn.Module, std: float = 0.02) -> None:
    """把模块参数填成随机张量（mock 权重），不加载 checkpoint。"""
    for _, p in module.named_parameters():
        p.data.normal_(0, std)


# ═════════════════════════════════════════════════════════════════════════
# 前处理
# ═════════════════════════════════════════════════════════════════════════
class TestPreprocess(CustomTestCase):
    """前处理：``EB5LayerRMSNorm.forward``（FD 风格 add-then-rmsnorm）。

    真实覆盖 kernel：``torch.nn.functional.rms_norm``（ATen CUDA RMSNorm kernel）。
    整模块 forward 真跑，无任何 stub / patch。
    """

    def _norm(self):
        norm = ernie5.EB5LayerRMSNorm(_HIDDEN, eps=1e-6).to(_DEVICE)
        # 真实权重是 bf16；默认构造是 fp32，显式对齐避免 dtype 告警。
        norm.weight.data = norm.weight.data.to(_DTYPE)
        return norm

    def test_add_then_norm_matches_reference(self):
        """真实 rms_norm kernel：residual 回写 == x+residual，输出形状/有限性正确。"""
        norm = self._norm()
        x = _rand(_TOKENS, _HIDDEN)
        residual = _rand(_TOKENS, _HIDDEN)
        out, new_residual = norm(x, residual)

        ref_residual = (x.float() + residual.float()).to(_DTYPE)
        self.assertEqual(out.shape, (_TOKENS, _HIDDEN))
        self.assertEqual(out.dtype, _DTYPE)
        self.assertTrue(torch.equal(new_residual, ref_residual))
        self.assertTrue(torch.isfinite(out).all())

    def test_norm_without_residual_returns_tensor(self):
        out = self._norm()(_rand(_TOKENS, _HIDDEN))
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (_TOKENS, _HIDDEN))

    def test_empty_input_passthrough(self):
        out = self._norm()(torch.empty(0, _HIDDEN, dtype=_DTYPE, device=_DEVICE))
        self.assertEqual(out.shape, (0, _HIDDEN))


# ═════════════════════════════════════════════════════════════════════════
# hc_pre
# ═════════════════════════════════════════════════════════════════════════
def _make_decoder_layer() -> "ernie5.EB5DecoderLayer":
    """构造只带 mhc 所需属性的 EB5DecoderLayer（不跑重量级 __init__）。"""
    layer = object.__new__(ernie5.EB5DecoderLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(rms_norm_eps=1e-6)
    layer.hc_eps = 1e-6
    layer.hc_sinkhorn_iters = 2
    layer.mhc_align_fleet = False
    layer.mhc_align_accum_order = False
    return layer


class TestHcPre(CustomTestCase):
    """hc_pre：``EB5DecoderLayer._mhc_pre``。

    真实覆盖 kernel：
      - ``deep_gemm`` ``tf32_hc_prenorm_gemm``（``SGLANG_OPT_DEEPGEMM_HC_PRENORM=1``，默认）
      - tilelang ``mhc_pre_big_fuse_with_norm_tilelang``

    整模块 forward 真跑：``_mhc_pre -> mhc.mhc_pre``。仅 mock 权重（随机张量）；
    TP/CP 并行上下文由真实单进程 dist + ServerArgs 提供，未 patch。
    """

    def _inputs(self):
        mix_width = _HC_MULT3
        return (
            _rand(_TOKENS, _HC_MULT, _HIDDEN),
            torch.randn(
                mix_width, _HC_MULT * _HIDDEN, dtype=torch.float32, device=_DEVICE
            )
            * 0.02,
            torch.ones(3, dtype=torch.float32, device=_DEVICE),
            torch.zeros(mix_width, dtype=torch.float32, device=_DEVICE),
            ernie5.EB5LayerRMSNorm(_HIDDEN, eps=1e-6).to(_DEVICE),
        )

    def test_mhc_pre_shapes(self):
        residual, fn, hc_scale, hc_base, norm = self._inputs()
        norm.weight.data = norm.weight.data.to(_DTYPE)
        mixed, post, comb = _make_decoder_layer()._mhc_pre(
            residual, fn, hc_scale, hc_base, norm
        )
        self.assertEqual(mixed.shape, (_TOKENS, _HIDDEN))
        self.assertEqual(mixed.dtype, _DTYPE)
        self.assertEqual(post.shape, (_TOKENS, _HC_MULT))
        self.assertEqual(comb.shape, (_TOKENS, _HC_MULT, _HC_MULT))
        self.assertTrue(torch.isfinite(mixed).all())
        self.assertTrue(torch.isfinite(post).all())
        self.assertTrue(torch.isfinite(comb).all())
        # comb 是 sinkhorn 归一化后的列随机矩阵：最后一列归一化保证列和为 1。
        self.assertTrue(
            torch.allclose(
                comb.float().sum(-2),
                torch.ones(_TOKENS, _HC_MULT, device=_DEVICE),
                atol=1e-3,
            )
        )

    def test_mhc_pre_matches_torch_reference(self):
        """真实 kernel 与 ``_mhc_pre_torch`` 数值一致（证明 kernel 真跑）。"""
        residual, fn, hc_scale, hc_base, norm = self._inputs()
        norm.weight.data = norm.weight.data.to(_DTYPE)
        layer = _make_decoder_layer()
        mixed, post, comb = layer._mhc_pre(residual, fn, hc_scale, hc_base, norm)
        ref_post, ref_comb, ref_mixed = mhc_ops._mhc_pre_torch(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps=1e-6,
            hc_pre_eps=layer.hc_eps,
            hc_sinkhorn_eps=layer.hc_eps,
            hc_post_mult_value=layer._MHC_POST_MULTIPLIER,
            sinkhorn_repeat=layer.hc_sinkhorn_iters,
        )
        # ernie5 传了 norm_weight，big_fuse_with_norm 会把 layer_input 再过一次
        # RMSNorm；参考实现不含这一步，这里补上再比。
        ref_normed = torch.nn.functional.rms_norm(
            ref_mixed.to(_DTYPE), (_HIDDEN,), norm.weight.data, 1e-6
        )
        self.assertTrue(
            torch.allclose(mixed.float(), ref_normed.float(), atol=5e-2, rtol=5e-2),
            "mhc_pre layer_input 与 torch 参考偏差过大",
        )
        self.assertTrue(
            torch.allclose(
                post.float(), ref_post.squeeze(-1).float(), atol=3e-2, rtol=3e-2
            )
        )

    def test_mhc_pre_empty_batch(self):
        """0-token 早退：不进入 kernel，返回正确形状的空张量。"""
        residual, fn, hc_scale, hc_base, norm = self._inputs()
        mixed, post, comb = _make_decoder_layer()._mhc_pre(
            residual[:0], fn, hc_scale, hc_base, norm
        )
        self.assertEqual(mixed.shape, (0, _HIDDEN))
        self.assertEqual(post.shape, (0, _HC_MULT))
        self.assertEqual(comb.shape, (0, _HC_MULT, _HC_MULT))


# ═════════════════════════════════════════════════════════════════════════
# HCA
# ═════════════════════════════════════════════════════════════════════════
def _make_hca_layer(compress_ratio: int, *, gated: bool) -> "ernie5.EB5AttentionCSAHCA":
    """构造真实的 ``EB5AttentionCSAHCA``（继承 DeepSeek-V4 ``MQALayer``）。

    ``compress_ratio``：0 = 纯 SWA 层；128 = HCA 压缩层（带 ``Compressor``）。
    权重用随机张量 mock；构造在 ``torch.device("cuda")`` 下进行，保证
    ``precompute_freqs_cis`` 等 plain attribute 一开始就在 GPU 上。
    """
    from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config

    cfg = DeepSeekV4Config(
        architectures=["ErnieCsaHcaForCausalLM"],
        quantization_config={},
        rope_scaling={
            "rope_type": "default",
            "factor": 1.0,
            "original_max_position_embeddings": _HCA_MAX_POS,
        },
        hidden_size=_HCA_HIDDEN,
        num_attention_heads=_HCA_HEADS,
        num_key_value_heads=1,
        qk_nope_head_dim=_HCA_QK_NOPE,
        qk_rope_head_dim=_HCA_QK_ROPE,
        head_dim=_HCA_HEAD_DIM,
        kv_lora_rank=_HCA_KV_LORA,
        q_lora_rank=_HCA_Q_LORA,
        o_lora_rank=_HCA_O_LORA,
        o_groups=_HCA_O_GROUPS,
        v_head_dim=_HCA_V_HEAD,
        max_position_embeddings=_HCA_MAX_POS,
        compress_ratios=[compress_ratio],
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        n_routed_experts=8,
        num_hidden_layers=1,
        moe_intermediate_size=64,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        rope_theta=10000,
        compress_rope_theta=40000,
        use_gated_attn=gated,
        gated_attn_use_q_lora=gated,
        gated_attn_act="sigmoid",
        use_vha_attention=False,
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(_DTYPE)
    try:
        with torch.device(_DEVICE):
            attn = ernie5.EB5AttentionCSAHCA(config=cfg, layer_id=0)
        attn = attn.to(_DEVICE)
    finally:
        torch.set_default_dtype(prev_dtype)
    _randomize(attn)
    return attn


def _make_dsv4_runner(compress_ratio: int, prefix_lens, extend_lens, forward_mode=None):
    """真实的 dsv4 attention backend + KV pool + ForwardBatch（复用 sglang 自带 kit）。

    HCA 层走的就是 DeepSeek-V4 的 ``MQALayer``，生产里由 ``EB5MLAHCAAttnBackend``
    把 HCA 层转发给 dsv4 机制；这里直接用 ``DeepseekV4AttnBackend``（真实 kernel）。
    """
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.test.kits.attention_unittest.attention_methods.dsv4_attention import (
        DSV4AttentionCase,
        MockDSV4ModelRunner,
        TinyDSV4ModelConfig,
        _make_forward_batch,
    )

    if forward_mode is None:
        forward_mode = ForwardMode.EXTEND
    case = DSV4AttentionCase(
        name="eb5_hca",
        backend="dsv4",
        forward_mode=forward_mode,
        num_heads=_HCA_HEADS,
        page_size=_HCA_PAGE,
        prefix_lens=tuple(prefix_lens),
        extend_lens=tuple(extend_lens),
        compress_ratio=compress_ratio,
    )
    model_config = TinyDSV4ModelConfig(
        num_heads=_HCA_HEADS,
        context_len=_HCA_MAX_POS,
        compression_ratios=[compress_ratio],
    )
    model_config.attention_arch = AttentionArch.MLA
    model_config.v_head_dim = _HCA_V_HEAD
    model_config.swa_v_head_dim = None
    model_config.get_num_kv_heads = lambda tp_size, dcp_size=1: 1
    runner = MockDSV4ModelRunner(
        case=case,
        model_config=model_config,
        dtype=_DTYPE,
        device=_DEVICE,
        max_context_len=_HCA_MAX_POS,
        swa_size=_HCA_SWA_SIZE,
        compression_ratios=[compress_ratio],
    )
    backend = ATTENTION_BACKENDS["dsv4"](runner)
    forward_batch = _make_forward_batch(
        case, runner, max_context_len=_HCA_MAX_POS, device=_DEVICE
    )
    return runner, backend, forward_batch


class TestHca(CustomTestCase):
    """HCA：``EB5AttentionCSAHCA.forward``（整模块真跑，继承 DeepSeek-V4 ``MQALayer``）。

    真实覆盖 kernel（torch.profiler 实测）：
      - 投影 GEMM：``wq_a`` / ``wkv`` / ``wq_b`` / ``wo_a``（deep_gemm / cublas）
      - ``fused_q_norm_rope``（q norm + RoPE）、``rms_norm``（q/kv norm）
      - 压缩层：``compress_norm_rope_store``（compressor 的 norm+rope+KV 写入）
      - ``set_swa_key_buffer_radix_fused_norm_rope``（KV cache 写入）
      - attention：``dsv4`` backend 的 SWA/C128 路径（FlashMLA sparse /
        flash_mla_with_kvcache）
      - 若开 gated：``_eb5_sigmoid_gate_mul_kernel`` + ``gate_proj`` GEMM

    整模块 forward 真跑：真实 ``DeepSeekV4Config`` + ``DeepSeekV4TokenToKVPool`` +
    ``ForwardBatch`` + 真实 backend（``set_forward_context``），只 mock 权重。
    """

    @classmethod
    def setUpClass(cls):
        _require_cuda()
        try:
            import sgl_kernel.flash_mla  # noqa: F401
        except ImportError as e:  # pragma: no cover - env dependent
            raise unittest.SkipTest(f"sgl_kernel.flash_mla unavailable: {e}")

    def _run(
        self,
        compress_ratio: int,
        *,
        gated: bool,
        prefix_lens=(0,),
        extend_lens=(32,),
        forward_mode=None,
    ):
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            set_forward_context,
        )

        attn = _make_hca_layer(compress_ratio, gated=gated)
        runner, backend, forward_batch = _make_dsv4_runner(
            compress_ratio, prefix_lens, extend_lens, forward_mode=forward_mode
        )
        # runner 在构造时 install 了 server-args override；用完必须还原，
        # 否则会影响同进程里后续依赖 get_server_args() 的用例（如 TestMoe）。
        self.addCleanup(runner._server_args_override.restore)

        set_forward_context(ForwardContext(attn_backend=backend))
        backend.init_forward_metadata(forward_batch)
        hidden = torch.randn(
            forward_batch.positions.shape[0], _HCA_HIDDEN, dtype=_DTYPE, device=_DEVICE
        )
        with torch.no_grad():
            out = attn.forward(forward_batch.positions, hidden, forward_batch)
        self.addCleanup(set_forward_context, None)
        return out, hidden

    def test_forward_swa_layer(self):
        """compress_ratio=0：纯 SWA HCA 层整模块 forward 真跑。"""
        out, hidden = self._run(0, gated=False)
        self.assertEqual(out.shape, hidden.shape)
        self.assertEqual(out.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(out).all())
        # 非退化：投影 + attention 真的算了。
        self.assertGreater(out.abs().sum().item(), 0.0)

    def test_forward_compressed_layer(self):
        """compress_ratio=128：HCA 压缩层（compressor norm+rope+KV store）整模块真跑。"""
        out, hidden = self._run(128, gated=False)
        self.assertEqual(out.shape, hidden.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(out.abs().sum().item(), 0.0)

    def test_forward_gated_layer(self):
        """开 sigmoid gated attention：走 gate_proj + ``eb5_sigmoid_gate_mul``。"""
        out, hidden = self._run(128, gated=True)
        self.assertEqual(out.shape, hidden.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(out.abs().sum().item(), 0.0)

    def test_forward_chunked_prefill_prefix(self):
        """带 prefix 的 chunked prefill（复用已缓存 KV）整模块真跑。"""
        out, hidden = self._run(0, gated=False, prefix_lens=(48,), extend_lens=(16,))
        self.assertEqual(out.shape, hidden.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_forward_decode(self):
        """DECODE → SWA sparse decode 路径（flash_mla sparse decode）整模块真跑。"""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        out, hidden = self._run(
            0,
            gated=False,
            prefix_lens=(8, 12, 40),
            extend_lens=(),
            forward_mode=ForwardMode.DECODE,
        )
        self.assertEqual(out.shape, hidden.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(out.abs().sum().item(), 0.0)


# ═════════════════════════════════════════════════════════════════════════
# MLA
# ═════════════════════════════════════════════════════════════════════════
def _make_mla_layer(*, gated: bool) -> "ernie5.EB5AttentionMLA":
    """构造真实的 ``EB5AttentionMLA``（继承 ``DeepseekV2AttentionMLA``）。

    full MLA 层几何：qk_nope=192 / rope=64 / v_head=512 / kv_lora=512。
    权重用随机张量 mock；另按 loader 的做法从 ``kv_b_proj.weight`` 折叠出
    ``w_kc`` / ``w_vc``（absorb 权重），否则 absorbed-MLA 路径拿不到它们。
    """
    cfg = SimpleNamespace(
        architectures=["ErnieMlaHcaForCausalLM"],
        model_type="ernie",
        rms_norm_eps=1e-6,
        hidden_size=_MLA_HIDDEN,
        num_attention_heads=_MLA_HEADS,
        rope_interleave=True,
        indexer_rope_interleave=False,
        use_gated_attn=gated,
        gated_attn_use_q_lora=gated,
        gated_attn_act="sigmoid",
        use_vha_attention=False,
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(_DTYPE)
    try:
        with torch.device(_DEVICE):
            attn = ernie5.EB5AttentionMLA(
                config=cfg,
                hidden_size=_MLA_HIDDEN,
                num_heads=_MLA_HEADS,
                qk_nope_head_dim=_MLA_QK_NOPE,
                qk_rope_head_dim=_MLA_QK_ROPE,
                v_head_dim=_MLA_V_HEAD,
                q_lora_rank=_MLA_Q_LORA,
                kv_lora_rank=_MLA_KV_LORA,
                rope_theta=10000,
                rope_scaling=None,
                max_position_embeddings=_MLA_MAX_POS,
                quant_config=None,
                reduce_results=False,
                layer_id=0,
                prefix="",
            )
        attn = attn.to(_DEVICE)
    finally:
        torch.set_default_dtype(prev_dtype)
    _randomize(attn)
    # loader 的 absorb 权重折叠（deepseek_weight_loader.py）：kv_b_proj.weight
    # [nh*(qk_nope+v_head), kv_lora] -> w_kc [nh, qk_nope, kv_lora] / w_vc [nh, kv_lora, v_head]
    w = attn.kv_b_proj.weight
    w_kc, w_vc = w.unflatten(0, (-1, _MLA_QK_NOPE + _MLA_V_HEAD)).split(
        [_MLA_QK_NOPE, _MLA_V_HEAD], dim=1
    )
    attn.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
    attn.w_vc = w_vc.contiguous().transpose(1, 2)
    return attn


def _make_mla_runner(prefix_lens):
    """真实的 triton MLA backend + ``MLATokenToKVPool`` + ForwardBatch（复用 kit）。"""
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.test.kits.attention_unittest.attention_methods.mla_attention import (
        MLAAttentionCase,
        MockMLAModelRunner,
        TinyMLAModelConfig,
        _make_forward_batch,
    )

    case = MLAAttentionCase(
        name="eb5_mla",
        backend="triton",
        forward_mode=ForwardMode.DECODE,
        num_heads=_MLA_HEADS,
        page_size=_MLA_PAGE,
        prefix_lens=tuple(prefix_lens),
        extend_lens=(),
    )
    model_config = TinyMLAModelConfig(
        num_heads=_MLA_HEADS,
        kv_lora_rank=_MLA_KV_LORA,
        qk_rope_head_dim=_MLA_QK_ROPE,
        hidden_size=_MLA_HIDDEN,
        context_len=_MLA_MAX_POS,
    )
    runner = MockMLAModelRunner(
        case=case,
        model_config=model_config,
        dtype=_DTYPE,
        device=_DEVICE,
        max_context_len=_MLA_MAX_POS,
        kv_lora_rank=_MLA_KV_LORA,
        qk_rope_head_dim=_MLA_QK_ROPE,
    )
    backend = ATTENTION_BACKENDS["triton"](runner)
    forward_batch = _make_forward_batch(
        case, runner, max_context_len=_MLA_MAX_POS, device=_DEVICE
    )
    return runner, backend, forward_batch


class TestMla(CustomTestCase):
    """MLA：``EB5AttentionMLA.forward_prepare`` + ``forward_core``（整模块真跑）。

    真实覆盖 kernel（torch.profiler 实测）：
      - ``fused_qkv_a_proj_with_mqa`` GEMM + ``q_a_layernorm`` / ``kv_a_layernorm``
      - RoPE（``apply_rotary_emb`` / 融合 qk norm+rope）
      - ``q_b_proj`` GEMM；absorb 的 ``w_kc`` bmm（``bmm_fp8`` / ``torch.bmm``）
      - attention：triton MLA decode kernel（``decode_attention_fwd``）
      - 若开 gated：``_eb5_sigmoid_gate_mul_kernel`` + ``gate_proj`` GEMM
      - ``o_proj`` GEMM

    整模块真跑：真实 ``EB5AttentionMLA`` + ``MLATokenToKVPool`` + ``ForwardBatch`` +
    真实 ``TritonAttnBackend``（``set_forward_context``），只 mock 权重。
    """

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    def _run(self, *, gated: bool, prefix_lens=(8, 12)):
        from sglang.srt.layers.communicator import (
            AttentionInputs,
            get_attn_tp_context,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            set_forward_context,
        )

        attn = _make_mla_layer(gated=gated)
        runner, backend, forward_batch = _make_mla_runner(prefix_lens)
        # runner 构造时 install 了 server-args override；用完还原（同 TestHca）。
        self.addCleanup(runner._server_args_override.restore)

        set_forward_context(ForwardContext(attn_backend=backend))
        backend.init_forward_metadata(forward_batch)
        hidden = torch.randn(
            forward_batch.positions.shape[0], _MLA_HIDDEN, dtype=_DTYPE, device=_DEVICE
        )
        # mHC 层会把归一化后的输入通过 attn-tp 上下文发布给 MLA 的 lazy qkv 投影；
        # 单层直跑时补上这一步，让 ``fetch_qkv_latent`` 真实可用。
        get_attn_tp_context().init_context(_MLA_Q_LORA, False)
        get_attn_tp_context().set_attn_inputs(
            AttentionInputs(hidden, forward_batch, attn.prepare_qkv_latent)
        )
        try:
            with torch.no_grad():
                state = attn.forward_prepare(
                    positions=forward_batch.positions,
                    hidden_states=hidden,
                    forward_batch=forward_batch,
                    zero_allocator=None,
                )
                out = attn.forward_core(state)
        finally:
            get_attn_tp_context().clear_attn_inputs()
            set_forward_context(None)
        return out, hidden

    def test_forward_absorbed_mla_decode(self):
        """DECODE → absorbed-MLA 路径整模块 forward 真跑。"""
        out, hidden = self._run(gated=False)
        self.assertEqual(out.shape, hidden.shape)
        self.assertEqual(out.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(out.abs().sum().item(), 0.0)

    def test_forward_gated_mla(self):
        """开 sigmoid gated attention：走 gate_proj + ``eb5_sigmoid_gate_mul``。"""
        out, hidden = self._run(gated=True)
        self.assertEqual(out.shape, hidden.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(out.abs().sum().item(), 0.0)


# ═════════════════════════════════════════════════════════════════════════
# hc_post
# ═════════════════════════════════════════════════════════════════════════
class TestHcPost(CustomTestCase):
    """hc_post：``EB5DecoderLayer._mhc_post``。

    真实覆盖 kernel：tilelang ``mhc_post_tilelang``。
    整模块 forward 真跑：``_mhc_post -> mhc.mhc_post``。仅 mock 权重（随机张量）。
    """

    def test_mhc_post_matches_torch_reference(self):
        residual = _rand(_TOKENS, _HC_MULT, _HIDDEN)
        x = _rand(_TOKENS, _HIDDEN)
        post = torch.rand(_TOKENS, _HC_MULT, dtype=torch.float32, device=_DEVICE)
        comb = torch.rand(
            _TOKENS, _HC_MULT, _HC_MULT, dtype=torch.float32, device=_DEVICE
        )
        out = _make_decoder_layer()._mhc_post(x, residual, post, comb)

        ref = mhc_ops._mhc_post_torch(x, residual, post.unsqueeze(-1), comb)
        self.assertEqual(out.shape, residual.shape)
        self.assertEqual(out.dtype, residual.dtype)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(
            torch.allclose(out.float(), ref.float(), atol=3e-2, rtol=3e-2),
            "mhc_post 与 torch 参考偏差过大",
        )

    def test_mhc_post_empty_batch(self):
        residual = torch.empty(0, _HC_MULT, _HIDDEN, dtype=_DTYPE, device=_DEVICE)
        out = _make_decoder_layer()._mhc_post(
            torch.empty(0, _HIDDEN, dtype=_DTYPE, device=_DEVICE),
            residual,
            torch.empty(0, _HC_MULT, dtype=torch.float32, device=_DEVICE),
            torch.empty(0, _HC_MULT, _HC_MULT, dtype=torch.float32, device=_DEVICE),
        )
        self.assertEqual(out.shape, residual.shape)


# ═════════════════════════════════════════════════════════════════════════
# moe
# ═════════════════════════════════════════════════════════════════════════
_MOE_CFG = SimpleNamespace(
    hidden_size=256,
    moe_intermediate_size=128,
    n_routed_experts=16,
    n_shared_experts=1,
    num_experts_per_tok=4,
    n_group=1,
    topk_group=1,
    norm_topk_prob=True,
    scoring_func="sigmoid",
    # False 走标准 sigmoid 路由；EB5 专属 split-feature 融合路由 kernel 另做直调覆盖。
    moe_split_feature_routing=False,
    routed_scaling_factor=2.5,
    routed_scaling_factor_learnable=True,
    hidden_act="silu",
    rms_norm_eps=1e-5,
    use_bias=False,
    moe_latent_size=None,
    use_latent_moe=False,
    topk_method="noaux_tc",
)


class TestMoe(CustomTestCase):
    """moe：``EB5MoE.forward`` + EB5 专属融合路由 kernel。

    真实覆盖 kernel：
      - ``moe_fused_gate``（router TopK，``sglang.kernels.ops.moe.moe_fused_gate``）
      - triton ``fused_moe``（routed expert GEMM，``FusedMoE`` runner）
      - ``silu_and_mul``（shared expert 激活，``SiluAndMul``）
      - EB5 专属 ``eb5_fused_topk.fused_split_feature_topk``（split-feature 融合路由）
    整模块 forward 真跑：``EB5MoE.forward``；仅 mock 权重（随机张量）。
    """

    def setUp(self):
        self._prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(_DTYPE)
        self.addCleanup(torch.set_default_dtype, self._prev_dtype)
        torch.manual_seed(0)
        self.moe = ernie5.EB5MoE(
            config=_MOE_CFG,
            layer_id=1,
            quant_config=None,
            prefix="layers.1.mlp",
        ).to(_DEVICE)
        for _, p in self.moe.named_parameters():
            p.data.normal_(0, 0.02)

    def tearDown(self):
        del self.moe
        torch.cuda.empty_cache()

    def test_forward_runs_expert_kernel(self):
        x = _rand(8, _MOE_CFG.hidden_size)
        with torch.no_grad():
            out = self.moe(x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(out).all())
        # 输出非退化：说明 expert GEMM + shared expert 真的算了。
        self.assertGreater(out.abs().sum().item(), 0.0)

    def test_zero_token_early_exit(self):
        empty = torch.empty(0, _MOE_CFG.hidden_size, dtype=_DTYPE, device=_DEVICE)
        self.assertIs(self.moe.forward(empty), empty)

    def test_fused_split_feature_router_topk(self):
        """直调 EB5 专属 split-feature 融合路由 kernel，对齐 eager fp32 参考。"""
        from sglang.srt.internal.layers.moe.eb5_fused_topk import (
            fused_split_feature_topk,
        )

        num_experts = _MOE_CFG.n_routed_experts
        topk = _MOE_CFG.num_experts_per_tok
        tokens = 37
        fused_logits = _rand(tokens, 2 * num_experts, dtype=torch.float32)
        bias = torch.zeros(num_experts, dtype=torch.float32, device=_DEVICE)
        per_expert_scale = torch.ones(
            num_experts, dtype=torch.float32, device=_DEVICE
        )

        weights, ids, _ = fused_split_feature_topk(
            fused_logits,
            bias,
            topk,
            True,
            "sigmoid",
            per_expert_scale=per_expert_scale,
        )
        self.assertEqual(ids.shape, (tokens, topk))
        self.assertEqual(weights.shape, (tokens, topk))

        halves = fused_logits.float().view(tokens, 2, num_experts)
        scores = torch.sigmoid(halves).sum(dim=1)
        ref_ids = torch.topk(scores, topk, dim=-1).indices
        self.assertTrue(torch.equal(ids.sort(dim=-1).values, ref_ids.sort(dim=-1).values))


# ═════════════════════════════════════════════════════════════════════════
# 后处理
# ═════════════════════════════════════════════════════════════════════════
class TestPostprocess(CustomTestCase):
    """后处理：``EB5LogitsProcessor._multimax_selu``（multimax-SELU 分段激活）。

    真实覆盖 kernel：triton ``_multimax_selu_kernel``（``fused_multimax_selu``）。
    整模块 forward 真跑：``EB5LogitsProcessor._multimax_selu``；仅 mock 权重。
    """

    def _processor(self, *, ranges, ts):
        proc = object.__new__(ernie5.EB5LogitsProcessor)
        nn.Module.__init__(proc)
        proc.use_multimax_lmhead = True
        proc.multimax_ranges = nn.Parameter(
            torch.tensor(ranges, dtype=_DTYPE, device=_DEVICE)
        )
        proc.multimax_ts = nn.Parameter(torch.tensor(ts, dtype=_DTYPE, device=_DEVICE))
        return proc

    def _torch_reference(self, x, ranges, ts):
        out = x.clone()
        out += ts[0] * torch.relu(ranges[0] - x)
        out += ts[1] * torch.relu(x - ranges[1])
        out += ts[2] * torch.relu(ranges[2] - x) ** 2
        out += ts[3] * torch.relu(x - ranges[3]) ** 2
        return out

    def test_fused_multimax_selu_runs_and_is_finite(self):
        """真实 triton kernel：bf16 + contiguous + numel==4 时走 fused 路径。"""
        ranges = [1.0, 2.0, 3.0, 4.0]
        ts = [0.1, 0.1, 0.1, 0.1]
        logits = _rand(3, 32)
        proc = self._processor(ranges=ranges, ts=ts)
        out = proc._multimax_selu(logits)
        self.assertEqual(out.shape, logits.shape)
        self.assertEqual(out.dtype, logits.dtype)
        self.assertTrue(torch.isfinite(out).all())

        ref = self._torch_reference(
            logits,
            torch.tensor(ranges, dtype=_DTYPE, device=_DEVICE),
            torch.tensor(ts, dtype=_DTYPE, device=_DEVICE),
        )
        # 融合 kernel 每步都按 bf16 舍入，和 eager bf16 图逐位对齐。
        self.assertTrue(torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
