# Adapted from sgl-project/sglang (https://github.com/sgl-project/sglang)
# Copyright 2023-2024 SGLang Team
#
# This file has been modified by Baidu, Inc. to support Kunlun XPU.
# Modifications Copyright (c) 2026 Baidu, Inc. All rights reserved.
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
"""Hooks for DeepseekV4DecoderLayer methods (hc_post, hc_pre)."""
from __future__ import annotations

import os
from typing import Literal

import torch

from sglang.srt.environ import envs
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.plugins.hook_registry import HookType, plugin_hook
import kunlun_ops

try:  # the constant lives next to the model
    from sglang.srt.models.deepseek_v4 import MHC_POST_MULT_VALUE as _MHC_POST_MULT_VALUE
except Exception:  # noqa: BLE001
    _MHC_POST_MULT_VALUE = 2.0


def _fp32_weight(owner, attr: str, tensor: torch.Tensor) -> torch.Tensor:
    """Return a contiguous fp32 view of ``tensor``, cached on ``owner``.

    ``hc_pre`` runs per layer per step; converting the 24x(hidden) routing
    matrix every call would cost more than the fused kernel saves.
    """
    cached = getattr(owner, attr, None)
    if cached is not None and cached[0] is tensor:
        return cached[1]
    out = tensor if tensor.dtype == torch.float32 else tensor.float()
    if not out.is_contiguous():
        out = out.contiguous()
    setattr(owner, attr, (tensor, out))
    return out


@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_pre",
    type=HookType.AROUND,
)
def hc_pre_kunlun(
    original_fn,
    self,
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm=None,
    forward_batch=None,
):
    """Kunlun-native hc_pre via ``kunlun_ops.hc_pre_kunlun_impl``.

    The fork only takes the fused mHC path when ``_is_xpu`` is true; on this
    stack (torch.xpu absent) it falls back to a torch decomposition
    (``x.float()`` + ``square().mean()`` + ``rsqrt`` + ``F.linear``), which is
    the single largest remaining per-step dispatch block in extend. The vendor
    op does fc+mean-square -> sinkhorn -> bmm in one call (n == 4 only) and is
    what the XPU path uses, so prefer it and keep the original as the fallback.

    Returns ``(y, post, comb, False)``: ``False`` means the caller still has to
    apply the post-attention norm, exactly like the torch branch.
    """
    num_tokens = x.shape[0]
    hc_mult = int(self.hc_mult)
    if (
        os.environ.get("SGLANG_KUNLUN_NATIVE_HC_PRE", "1") == "0"
        or num_tokens == 0
        or hc_mult != 4
        or x.dtype not in (torch.bfloat16, torch.float16)
        or not hasattr(kunlun_ops, "hc_pre_kunlun_impl")
    ):
        return original_fn(
            self, x, hc_fn, hc_scale, hc_base, norm, forward_batch
        )

    x_flat = x.flatten(1)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()
    hidden = x_flat.shape[1] // hc_mult

    fn = _fp32_weight(self, "_kunlun_hc_pre_fn", hc_fn)
    base = hc_base if hc_base.dtype == torch.float32 else hc_base.float()
    scale = hc_scale if hc_scale.dtype == torch.float32 else hc_scale.float()

    y = torch.empty((num_tokens, hidden), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=x.dtype, device=x.device)
    comb = torch.empty(
        (num_tokens, hc_mult * hc_mult), dtype=x.dtype, device=x.device
    )
    kunlun_ops.hc_pre_kunlun_impl(
        x_flat,
        fn,
        base.contiguous(),
        scale.contiguous(),
        y,
        post,
        comb,
        float(self.rms_norm_eps),
        float(self.hc_eps),
        float(self.hc_eps),
        float(_MHC_POST_MULT_VALUE),
        int(self.hc_sinkhorn_iters),
    )
    # The kernel writes post/comb in the activation dtype; the torch branch
    # hands them back as fp32 and hc_post assumes that.
    return (
        y,
        post.float(),
        comb.view(num_tokens, hc_mult, hc_mult).float(),
        False,
    )


@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer.hc_post",
    type=HookType.AROUND,
)
def hc_post_kunlun(
    original_fn,
    self,
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
):
    """Kunlun-native hc_post via ``kunlun_ops.hc_post_kunlun_impl``.

    Computes ``out = post * x + comb @ residual`` (fused).
    Only N == 4 (hc_mult) is supported on XPU.
    """
    if (
        x.shape[0] == 0
        or os.environ.get("SGLANG_KUNLUN_NATIVE_HC_POST", "1") == "0"
        or int(self.hc_mult) != 4
        or not hasattr(kunlun_ops, "hc_post_kunlun_impl")
    ):
        return original_fn(self, x, residual, post, comb)

    batch = x.shape[0]
    N = self.hc_mult
    D = x.shape[-1]
    out = torch.empty((batch, N, D), dtype=x.dtype, device=x.device)
    kunlun_ops.hc_post_kunlun_impl(
        post.contiguous(),
        x.contiguous(),
        comb.contiguous(),
        residual.contiguous(),
        out,
        batch, N, D,
    )
    return out


# --------------------------- INT4 checkpoint support ---------------------
# The compressed-tensors targets/ignore lists in the INT4 checkpoint use the
# original HF module names (`attn`/`ffn`/`w1..w3`), while the weights
# themselves are remapped by `remap_weight_name_to_dpsk_hf_format`. The mapper
# is picked up by the model loader to remap the quantization config. The keys
# look like regexes because the checkpoint targets are regex strings; 0.5.14
# WeightsMapper still matches them as plain substrings.
DeepseekV4ForCausalLM.packed_modules_mapping = {
    "wqkv_a": ["wq_a", "wkv"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

DeepseekV4ForCausalLM.hf_to_sglang_mapper = WeightsMapper(
    orig_to_new_substr={
        r"\.attn\.": r"\.self_attn\.",
        r"\.ffn\.": r"\.mlp\.",
        r"\.attn_norm": r"\.input_layernorm",
        r"\.ffn_norm": r"\.post_attention_layernorm",
    },
    orig_to_new_suffix={
        r"\.w1$": r"\.gate_proj$",
        r"\.w2$": r"\.down_proj$",
        r"\.w3$": r"\.up_proj$",
    },
)


def _fuse_wqkv_a_weight_scale(weights):
    """Concat the ``wq_a``/``wkv`` per-channel scales for the fused ``wqkv_a``.

    Upstream `load_weights` already fuses ``.weight`` and ``.weight_scale_inv``;
    the int4 checkpoint instead carries ``.weight_scale``, which is fused here
    so the generic loader path picks it up as a plain ``wqkv_a.weight_scale``.
    """
    pending = {}
    for name, tensor in weights:
        is_q = name.endswith(".wq_a.weight_scale")
        if not is_q and not name.endswith(".wkv.weight_scale"):
            yield name, tensor
            continue
        fused_name = name.replace(".wq_a." if is_q else ".wkv.", ".wqkv_a.")
        bucket = pending.setdefault(fused_name, {})
        shard: Literal["q", "kv"] = "q" if is_q else "kv"
        assert shard not in bucket, f"duplicate shard {shard} for {fused_name}"
        bucket[shard] = tensor
        if len(bucket) == 2:
            yield fused_name, torch.cat([bucket["q"], bucket["kv"]], dim=0)
            pending.pop(fused_name)
    assert not pending, f"unpaired wqkv_a weight scales: {sorted(pending)}"


@plugin_hook(
    target="sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights",
    type=HookType.AROUND,
)
def load_weights_kunlun(original_fn, self, weights, is_nextn=False):
    """Pre-process the int4/int8 weight stream, then defer to upstream."""
    if envs.SGLANG_OPT_FUSE_WQA_WKV.get():
        weights = _fuse_wqkv_a_weight_scale(weights)
    result = original_fn(self, weights, is_nextn)
    return result
