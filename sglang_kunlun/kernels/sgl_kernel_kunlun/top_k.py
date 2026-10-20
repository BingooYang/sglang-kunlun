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
"""Kunlun implementations of sgl_kernel top-k operations.

``fast_topk``, ``fast_topk_transform_fused`` and
``fast_topk_transform_ragged_fused`` are pure-torch fallbacks that mirror the
CUDA kernels in ``csrc/elementwise/topk.cu`` (NSA / DSA indexer path).

``fast_topk_v2`` still relies on the CUDA kernel
(``torch.ops.sgl_kernel.fast_topk``) and therefore remains a stub in
``_sgl_kernel_stub.py``; it is only reachable with ``SGLANG_OPT_USE_TOPK_V2``
enabled, which is off on Kunlun.

Semantics shared by both transforms (identical to the CUDA kernels)
-------------------------------------------------------------------
For row ``i`` the selection window is
``score[i, row_starts[i] : row_starts[i] + lengths[i]]`` (``row_starts`` may be
``None``, meaning the window starts at column 0) and the returned positions are
**relative to the window start**. Output columns are padded with ``-1`` when a
row has fewer than ``topk`` valid positions (``lengths[i] < topk``, or the row
is narrower than ``topk``).
"""

import torch


def fast_topk(values, topk, dim):
    if topk == 1:
        return torch.max(values, dim=dim, keepdim=True)
    else:
        return torch.topk(values, topk, dim=dim)


def _window_topk(score, lengths, topk, row_starts):
    """Vectorised per-row top-k restricted to each row's visible window.

    Returns ``(rel_idx, selected_valid, k_eff)``:
      * ``rel_idx`` -- int64 ``(B, k_eff)`` positions relative to the window
        start of that row;
      * ``selected_valid`` -- bool ``(B, k_eff)``, False for slots that only got
        filled because the window is shorter than ``k_eff`` (those must be
        emitted as ``-1``, exactly like the kernels' padding);
      * ``k_eff`` -- ``min(topk, score.shape[1])``.

    Columns outside ``[start, start + length)`` are set to ``-inf`` so
    ``torch.topk`` never prefers them over a real score; ``selected_valid`` is
    gathered from the mask itself instead of being recomputed from the values,
    so a genuine ``-inf`` logit is not mistaken for padding.
    """
    num_rows, width = score.shape
    k_eff = min(int(topk), width)
    cols = torch.arange(width, device=score.device, dtype=torch.int64).unsqueeze(0)
    lens = lengths.to(torch.int64).view(num_rows, 1)
    if row_starts is None:
        starts = torch.zeros((1, 1), dtype=torch.int64, device=score.device)
    else:
        starts = row_starts.to(torch.int64).view(num_rows, 1)

    in_window = (cols >= starts) & (cols < starts + lens)
    masked = torch.where(
        in_window,
        score,
        torch.full((), float("-inf"), dtype=score.dtype, device=score.device),
    )
    # ``sorted=False`` mirrors the kernels: the slot order is not part of the
    # contract, only the selected set is.
    _, idx = torch.topk(masked, k_eff, dim=-1, sorted=False)
    selected_valid = torch.gather(in_window, 1, idx)
    return idx - starts, selected_valid, k_eff


def _pad_columns(out, k_eff, topk):
    """Right-pad ``out`` from ``k_eff`` to ``topk`` columns with -1."""
    if k_eff >= int(topk):
        return out
    pad = out.new_full((out.shape[0], int(topk) - k_eff), -1)
    return torch.cat([out, pad], dim=1)


def _page_table_row_ids(cu_seqlens_q, num_rows, num_table_rows, device):
    """Map each score row to its page-table row.

    Mirrors ``topk_transform_prefill_kernel``: row ``bid`` uses page-table row
    ``i`` where ``cu_seqlens_q[i] <= bid < cu_seqlens_q[i + 1]``. When
    ``num_table_rows == num_rows`` every request owns exactly one row (decode,
    where ``cu_seqlens_q`` is ``arange``), so the mapping is the identity and
    no ``B x num_requests`` compare matrix is materialised.
    """
    if cu_seqlens_q is None or num_table_rows == num_rows:
        return torch.arange(num_rows, device=device, dtype=torch.int64)
    cu = cu_seqlens_q.to(torch.int64).flatten()
    num_reqs = min(int(cu.numel()) - 1, num_table_rows)
    row_ids = torch.arange(num_rows, device=device, dtype=torch.int64).unsqueeze(1)
    rows = (row_ids >= cu[:num_reqs].unsqueeze(0)).sum(dim=1) - 1
    return rows.clamp_(0, num_table_rows - 1)


def fast_topk_transform_fused(
    score,
    lengths,
    page_table_size_1,
    cu_seqlens_q,
    topk,
    row_starts=None,
):
    """Kunlun torch implementation of ``sgl_kernel.fast_topk_transform_fused``.

    Windowed per-row top-k followed by a page-table (page_size == 1) lookup:
    ``out[i, j] = page_table[row_of(i), rel_idx[i, j]]`` for the selected slots
    and ``-1`` for padding, matching
    ``topk_transform_{decode,prefill}_kernel`` in ``csrc/elementwise/topk.cu``.
    """
    rel_idx, selected_valid, k_eff = _window_topk(score, lengths, topk, row_starts)

    device = score.device
    num_rows = score.shape[0]
    num_table_rows = page_table_size_1.shape[0]
    table = page_table_size_1.index_select(
        0, _page_table_row_ids(cu_seqlens_q, num_rows, num_table_rows, device)
    )
    table_width = table.shape[1]

    # ``rel_idx`` is 0-based inside the window, hence also the page-table
    # column. Clamp before ``gather`` (it faults on out-of-range indices) and
    # drop the clamped-out entries through ``in_range``.
    in_range = selected_valid & (rel_idx >= 0) & (rel_idx < table_width)
    safe_idx = rel_idx.clamp_(0, max(table_width - 1, 0))
    mapped = torch.gather(table, 1, safe_idx)
    out = torch.where(
        in_range, mapped, torch.full((), -1, dtype=mapped.dtype, device=device)
    )
    return _pad_columns(out.to(torch.int32), k_eff, topk)


def fast_topk_transform_ragged_fused(
    score,
    lengths,
    topk_indices_offset,
    topk,
    row_starts=None,
):
    """Kunlun torch implementation of ``sgl_kernel.fast_topk_transform_ragged_fused``.

    Same windowed top-k, then the window-relative index is shifted into the
    flattened (ragged) KV space: ``out[i, j] = rel_idx[i, j] + offset[i]``, and
    ``-1`` for padding -- ``topk_transform_prefill_ragged_kernel``.
    """
    rel_idx, selected_valid, k_eff = _window_topk(score, lengths, topk, row_starts)

    offset = topk_indices_offset.to(torch.int64).view(score.shape[0], 1)
    out = torch.where(
        selected_valid,
        rel_idx + offset,
        torch.full((), -1, dtype=torch.int64, device=score.device),
    )
    return _pad_columns(out.to(torch.int32), k_eff, topk)


def moe_fused_gate(
    input_tensor,
    bias,
    num_expert_group,
    topk_group,
    topk,
    num_fused_shared_experts,
    routed_scaling_factor,
    apply_routed_scaling_factor_on_output,
):
    """Kunlun implementation of ``sgl_kernel.moe_fused_gate``."""

    import kunlun_ops

    if num_fused_shared_experts != 0:
        raise NotImplementedError("Kunlun moe_fused_gate does not support fused shared experts")

    num_tokens, num_experts = input_tensor.shape
    block_statistic = torch.empty(
        12, num_experts, dtype=torch.int32, device=input_tensor.device
    )
    topk_weights = torch.empty(
        num_tokens, topk, dtype=torch.float32, device=input_tensor.device
    )
    topk_ids = torch.empty(
        num_tokens, topk, dtype=torch.int32, device=input_tensor.device
    )
    kunlun_ops.moe_sigmoid_group_topk_norm(
        x=input_tensor,
        topk_index=topk_ids,
        norm_score=topk_weights,
        block_static=block_statistic,
        bias=bias.float(),
        scale=(routed_scaling_factor if apply_routed_scaling_factor_on_output else 1.0),
        n_group=num_expert_group,
        topk_group=topk_group,
    )
    return topk_weights, topk_ids
