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
"""HookRegistry registration for the MHC split-Sinkhorn / combine kernels.

Both are tilelang/Triton kernels upstream and cannot build on Kunlun XPU.
The implementations live in :mod:`sglang_kunlun.eb5_adapt.kernels.mhc_torch` -- the same
functions EB5 uses for ``mhc_pre`` / ``mhc_post``, so DSV4 and EB5 share one
definition of the maths.

``sglang.srt.models.deepseek_v4`` imports the symbol lazily inside the call
site, so it resolves the patched module attribute at call time.
"""

from __future__ import annotations

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from sglang_kunlun.eb5_adapt.kernels.mhc_torch import hc_combine as _hc_combine
from sglang_kunlun.eb5_adapt.kernels.mhc_torch import hc_split_sinkhorn as _hc_split_sinkhorn


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.hc_split_sinkhorn",
    type=HookType.REPLACE,
)
def _hc_split_sinkhorn_torch(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """DSV4 entry point: use the fused Kunlun op, fall back to torch."""
    return _hc_split_sinkhorn(
        mixes,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_iters,
        eps,
        use_fused=True,
    )


@plugin_hook(
    "sglang.kernels.ops.layernorm.mhc.hc_combine",
    type=HookType.REPLACE,
)
def hc_combine_kunlun(
    x_flat: torch.Tensor, pre: torch.Tensor, hc: int, out_dtype: torch.dtype
):
    """DSV4 entry point for the weighted sum (Triton is CUDA-only)."""
    return _hc_combine(x_flat, pre, hc, out_dtype)
