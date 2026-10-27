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
"""Hooks for ``sglang.kernels.ops.kvcache.mla_buffer``.

The MLA latent KV scatter. Kunlun reports ``supports_triton=False``, so the
dispatcher picks ``set_mla_kv_buffer_naive``, whose
``rows[dst, :nope_dim] = src`` advanced-index assignment on a *viewed* pool
materialises the whole destination on XPU: measured 2026-10-27 on the ERNIE
MLA layer, it asked for 24.38 GiB and OOMed with 23.82 GiB free.

The replacement writes the same rows with ``index_copy_``, which touches only
the ``[num_written, row_width]`` source. Same semantics: nope then rope,
``reserved_skip_index`` dropped, DCP ownership honoured.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)


@plugin_hook(
    "sglang.kernels.ops.kvcache.mla_buffer.set_mla_kv_buffer_naive",
    type=HookType.REPLACE,
)
def set_mla_kv_buffer_naive_kunlun(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: Optional[torch.Tensor],
    *,
    reserved_skip_index: int,
    dcp_world_size: int,
    dcp_rank: int,
) -> None:
    """Allocation-free MLA latent scatter (see module docstring)."""
    rows = kv_buffer.view(kv_buffer.shape[0], -1)
    nope = cache_k_nope.reshape(cache_k_nope.shape[0], -1)
    nope_dim = nope.shape[-1]
    has_rope = cache_k_rope is not None and cache_k_rope.numel() > 0

    loc = loc.to(torch.int64)
    if has_rope:
        is_valid = loc != reserved_skip_index
    else:
        # Mirrors set_mla_kv_buffer_kernel_norope, which takes no skip index.
        is_valid = torch.ones_like(loc, dtype=torch.bool)
    if dcp_world_size > 1:
        is_valid &= loc % dcp_world_size == dcp_rank
    dst = loc[is_valid] // dcp_world_size
    if dst.numel() == 0:
        return

    src = nope[is_valid]
    if has_rope:
        rope = cache_k_rope.reshape(cache_k_rope.shape[0], -1)[is_valid]
        src = torch.cat([src, rope], dim=-1)

    row_width = rows.shape[-1]
    if src.shape[-1] == row_width:
        # Duplicate locs race here exactly as they do in the kernel: which write
        # lands last is undefined either way. The one difference is granularity --
        # nope and rope go in a single index_copy_ instead of two separate stores,
        # so a duplicated loc cannot end up with nope from one row and rope from
        # another, which the kernel does permit.
        rows.index_copy_(0, dst, src.to(rows.dtype).contiguous())
        return

    # Narrower payload than the row (e.g. an fp8 pool with trailing scales):
    # write the covered prefix only, still without touching the rest.
    head = rows.narrow(1, 0, src.shape[-1])
    if head.is_contiguous():
        head.index_copy_(0, dst, src.to(rows.dtype).contiguous())
        return
    # Last resort: per-row copy_, which allocates nothing either.
    payload = src.to(rows.dtype)
    for i in range(dst.shape[0]):
        rows[dst[i], : src.shape[-1]].copy_(payload[i])
