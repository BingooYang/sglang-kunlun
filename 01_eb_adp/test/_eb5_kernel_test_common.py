"""EB5-lite 逐模块 kernel 覆盖单测的**公共脚手架**（非测试文件）。

本文件不含任何 ``register_cuda_ci`` / ``unittest.TestCase``，只提供各
``test_eb5_<module>.py`` 共享的进程级 / 并行上下文初始化与通用 helper。按 ku
文档《EBlite 和 dpsk-v4 算子对比》把算子覆盖拆成"每个小模块一个文件"后，公共部分
集中在这里：

* :func:`_init_single_process_dist` —— 单进程 ``torch.distributed``（``world_size=1``），
  **幂等**（``torch.distributed.is_initialized()`` 时直接返回）；端口按
  ``CUDA_VISIBLE_DEVICES`` 派生（不同 GPU 互不冲突），并在 ``EADDRINUSE`` 时**顺序重试**。
* ``patch_hf_strict_for_tests`` —— 由 ``_hf_strict_compat`` 提供的 HF strict 兼容 patch，
  必须在 dist 初始化**之后**调用（后者会替换 ``torch._dynamo``）。
* ``ServerArgs`` 发布 + ``get_parallel().override(attn_tp_size=1)`` + ``initialize_moe_config()``
  —— 让 kernel 内部读的 TP / 对称内存 / MoE 后端上下文真实可用（非 patch）。

通用 helper（从 ``test_eb5_operator_coverage.py`` / ``test_eb5_layer_modules.py`` 按需搬）：
``_rand`` / ``_randomize`` / ``_make_decoder_layer`` / ``_make_hca_layer`` /
``_make_dsv4_runner`` / ``_make_mla_layer`` / ``_make_mla_runner`` / ``_MOE_CFG`` 等。
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
    """建立真实的单进程 distributed 环境（``world_size=1``），幂等。

    必须在 ``patch_hf_strict_for_tests()`` 之前调用：后者会把 ``torch._dynamo``
    换成轻量 stub，而 ``init_process_group`` 会用到真实的 ``torch._dynamo``。

    端口按 ``CUDA_VISIBLE_DEVICES`` 派生（``29600 + gpu*13``），使同机不同 GPU 上
    并发跑的单测不会撞端口；再叠加一个 0..19 的顺序偏移重试，避免撞上同机其它进程
    / 上一轮残留的 ``TIME_WAIT``（``EADDRINUSE``）。
    """
    if torch.distributed.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    _first_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    _base_port = 29600 + (int(_first_gpu) if _first_gpu.isdigit() else 0) * 13
    last_err = None
    for _attempt in range(20):
        _master_port = _base_port + _attempt
        os.environ["MASTER_PORT"] = str(_master_port)
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{_master_port}",
                backend="nccl",
            )
            break
        except Exception as e:  # noqa: BLE001 - 换端口重试
            last_err = e
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
    else:
        raise last_err
    initialize_model_parallel(tensor_model_parallel_size=1)


_init_single_process_dist()

from _hf_strict_compat import patch_hf_strict_for_tests  # noqa: E402

patch_hf_strict_for_tests()

from sglang.srt.runtime_context import get_parallel  # noqa: E402
from sglang.srt.server_args import (  # noqa: E402
    ServerArgs,
    set_global_server_args_for_scheduler,
)
from sglang.test.test_utils import CustomTestCase  # noqa: E402

# 发布 ServerArgs + 并行 override，让 kernel 内部的 TP / 对称内存上下文真实生效。
_SERVER_ARGS = ServerArgs(model_path="dummy")
_SERVER_ARGS.enable_dp_attention = False
set_global_server_args_for_scheduler(_SERVER_ARGS)
_PARALLEL_OVERRIDE = get_parallel().override(
    attn_tp_size=1, attn_tp_rank=0, attn_cp_size=1, attn_cp_rank=0
)
_PARALLEL_OVERRIDE.__enter__()

# MoE 后端配置（EB5MoE / mega_moe 读 get_moe_a2a_backend / get_moe_impl_class）。
from sglang.srt.layers.moe.utils import initialize_moe_config  # noqa: E402

initialize_moe_config()

_DEVICE = "cuda"
_DTYPE = torch.bfloat16

# ── mHC 维度（hc_pre / hc_post / 后处理共用） ──
_HC_HIDDEN = 4096
_HC_MULT = 4
_HC_MULT3 = _HC_MULT * (2 + _HC_MULT)
_HC_TOKENS = 8

# ── HCA(``EB5AttentionCSAHCA`` / DeepSeek-V4 ``MQALayer``) 维度 ──
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

# ── MoE 配置（EB5MoE.forward 用） ──
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


def _require_cuda():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")


def _rand(*shape, dtype=_DTYPE):
    return torch.randn(*shape, dtype=dtype, device=_DEVICE)


def _randomize(module: nn.Module, std: float = 0.02) -> None:
    """把模块参数填成随机张量（mock 权重），不加载 checkpoint。"""
    for _, p in module.named_parameters():
        p.data.normal_(0, std)


def _make_decoder_layer() -> "ernie5.EB5DecoderLayer":
    """构造只带 mhc 所需属性的 EB5DecoderLayer（不跑重量级 __init__）。"""
    from sglang.srt.internal.models.eb5 import ernie5

    layer = object.__new__(ernie5.EB5DecoderLayer)
    nn.Module.__init__(layer)
    layer.config = SimpleNamespace(rms_norm_eps=1e-6)
    layer.hc_eps = 1e-6
    layer.hc_sinkhorn_iters = 2
    layer.mhc_align_fleet = False
    layer.mhc_align_accum_order = False
    return layer


def _make_hca_layer(compress_ratio: int, *, gated: bool) -> "ernie5.EB5AttentionCSAHCA":
    """构造真实的 ``EB5AttentionCSAHCA``（继承 DeepSeek-V4 ``MQALayer``）。

    ``compress_ratio``：0 = 纯 SWA 层；128 = HCA 压缩层（带 ``Compressor``）。
    权重用随机张量 mock；构造在 ``torch.device("cuda")`` 下进行，保证
    ``precompute_freqs_cis`` 等 plain attribute 一开始就在 GPU 上。
    """
    from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
    from sglang.srt.internal.models.eb5 import ernie5

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
    """真实的 dsv4 attention backend + KV pool + ForwardBatch（复用 sglang 自带 kit）。"""
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


def _make_mla_layer(*, gated: bool) -> "ernie5.EB5AttentionMLA":
    """构造真实的 ``EB5AttentionMLA``（继承 ``DeepseekV2AttentionMLA``）。

    full MLA 层几何：qk_nope=192 / rope=64 / v_head=512 / kv_lora=512。
    权重用随机张量 mock；另按 loader 的做法从 ``kv_b_proj.weight`` 折叠出
    ``w_kc`` / ``w_vc``（absorb 权重），否则 absorbed-MLA 路径拿不到它们。
    """
    from sglang.srt.internal.models.eb5 import ernie5

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
    # loader 的 absorb 权重折叠：kv_b_proj.weight
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


__all__ = [
    "CustomTestCase",
    "_DEVICE",
    "_DTYPE",
    "_HCA_HIDDEN",
    "_HCA_HEADS",
    "_HCA_HEAD_DIM",
    "_HCA_KV_LORA",
    "_HCA_MAX_POS",
    "_HCA_PAGE",
    "_HCA_QK_NOPE",
    "_HCA_QK_ROPE",
    "_HCA_V_HEAD",
    "_HC_HIDDEN",
    "_HC_MULT",
    "_HC_MULT3",
    "_HC_TOKENS",
    "_MLA_HEADS",
    "_MLA_HIDDEN",
    "_MLA_KV_LORA",
    "_MLA_MAX_POS",
    "_MLA_PAGE",
    "_MLA_QK_NOPE",
    "_MLA_QK_ROPE",
    "_MLA_V_HEAD",
    "_MOE_CFG",
    "_make_decoder_layer",
    "_make_dsv4_runner",
    "_make_hca_layer",
    "_make_mla_layer",
    "_make_mla_runner",
    "_rand",
    "_randomize",
    "_require_cuda",
]
