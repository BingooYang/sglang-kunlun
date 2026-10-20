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
"""Kunlun overrides for ``sglang.srt.layers.attention.nsa.nsa_indexer``.

Source: kunlun-0.5.10-mimo: sgl-kernel/.../patch/layers/attention/nsa/nsa_indexer.py

This module replaces seven Indexer methods with Kunlun-specific implementations
that route through ``kunlun_ops`` and ``xspeedgate_ops``. Heavy quantization
math is expressed via ``torch.compile(dynamic=True)`` to mirror mimo.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.plugins.hook_registry import HookType, plugin_hook

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool

logger = logging.getLogger(__name__)


def _topk_all_visible(metadata, topk: int, device) -> torch.Tensor:
    """``kv_len <= topk`` 的免 logits 快路径：返回每个 query token 的全部可见 slot。

    上游用 ``metadata.topk_transform(zeros, topk)`` 走 sgl-kernel 的
    ``fast_topk_transform_fused``（昆仑 stub 未实现，2026-10-20 实测抛
    ``NotImplementedError``）。该函数的语义是"先 topk 再经 page_table_1 映射成
    slot id"，而在长度不超过 topk 时 topk 就是全部位置，所以直接用 torch 拼：
    ``slot = page_table_1[batch_of_token, :visible]``，右侧补 -1。

    形状/语义与 ``_get_topk_ragged`` 一致：``dsa_seqlens_expanded`` 是 per-token
    可见 KV 长度，``page_table_1`` 在常规 prefill 下是 per-request，需要按
    extend 长度展开行。
    """
    page_table_1 = metadata.get_page_table_1()
    visible = metadata.get_seqlens_expanded().to(torch.int64)
    rows = page_table_1
    if page_table_1.shape[0] != visible.shape[0]:
        extend = torch.tensor(
            metadata.get_dsa_extend_len_cpu(), device=device, dtype=torch.int64
        )
        batch_idx = torch.repeat_interleave(
            torch.arange(extend.numel(), device=device), extend
        )
        rows = page_table_1.index_select(0, batch_idx)

    width = min(int(topk), rows.shape[1])
    cols = torch.arange(width, device=device, dtype=torch.int64)
    out = torch.where(
        cols.unsqueeze(0) < visible.unsqueeze(1),
        rows[:, :width].to(torch.int32),
        torch.full((), -1, dtype=torch.int32, device=device),
    )
    if width < int(topk):
        out = torch.cat(
            [out, out.new_full((out.shape[0], int(topk) - width), -1)], dim=1
        )
    _indexer_trace(f"topk-all-visible:done shape={tuple(out.shape)}")
    return out


def _indexer_kv_pool():
    """indexer 侧真正要写的 KV pool（EB5 是 full latent 子池）。

    fork 的 EB5 运行时（``internal/models/eb5/runtime_patch.py`` →
    ``register_dsa_full_pool_patch``）只把
    ``sglang.srt.layers.attention.dsa.dsa_indexer.get_token_to_kv_pool`` 这个
    **模块属性**换成"若外层池有 ``full_kv_pool`` 就返回它"的包装。本文件用
    ``from ... import`` 绑定的是原函数，直接调用会拿到外层 256 页的 HCA 池，
    ``set_index_k_scale_buffer`` 按 ``layer_mapping`` 找到 ratio=128 的 HCA 层，
    直接断言 ``No indexer pool for compression ratio 128``（2026-10-20 实测）。

    所以这里优先走被 patch 过的模块属性，再按同一语义兜底解包
    ``full_kv_pool``（DSV4 的池没有该属性，行为不变）。
    """
    import sglang.srt.layers.attention.dsa.dsa_indexer as _dsa_indexer

    fn = getattr(_dsa_indexer, "get_token_to_kv_pool", get_token_to_kv_pool)
    pool = fn()
    full_pool = getattr(pool, "full_kv_pool", None)
    return full_pool if full_pool is not None else pool


def _install_forward_mode_compat() -> None:
    """fork 把 ``ForwardMode.is_draft_extend`` 删了，只留 ``is_draft_extend_v2``。

    插件（本文件与 ``_nsa_backend_impl.py``）有十几处旧调用
    ``forward_mode.is_draft_extend(include_v2=True)``，2026-10-20 21:56 实测
    ``AttributeError: 'ForwardMode' object has no attribute 'is_draft_extend'``。
    在类上补一个兼容方法，一处修复所有调用点。
    """
    try:
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        if hasattr(ForwardMode, "is_draft_extend"):
            return
        v2 = getattr(ForwardMode, "is_draft_extend_v2", None)
        if v2 is None:
            return

        def is_draft_extend(self, include_v2: bool = True):
            return bool(v2(self))

        setattr(ForwardMode, "is_draft_extend", is_draft_extend)
    except Exception:  # noqa: BLE001
        pass


_install_forward_mode_compat()


def _seqlens_cpu(metadata, forward_batch, *, expanded: bool):
    """host 侧 seq_lens（fork 的 ``DSAIndexerMetadata`` 没有 ``*_cpu`` 访问器）。

    2026-10-20 22:02 实测：``AttributeError: 'DSAIndexerMetadata' object has no
    attribute 'get_seqlens_int32_cpu'``（decode 第一步）。

    decode 是每层都会走的热路径：优先用 ``ForwardBatch.seq_lens_cpu``（host 侧
    现成副本，零 D2H），只有 target_verify / draft_extend 这种 expand 过的形状
    才退回一次 ``.cpu()``（EAGLE 关闭时不会走到）。
    """
    name = "get_seqlens_expanded_cpu" if expanded else "get_seqlens_int32_cpu"
    fn = getattr(metadata, name, None)
    if fn is not None:
        return fn()
    if not expanded:
        seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None:
            # vendor dsa_paged_mqa_logits 要求 host 侧 context_lens 是 int32；
            # ForwardBatch.seq_lens_cpu 是 int64（22:28 实测 ValueError）。
            return seq_lens_cpu.to(torch.int32)
    src = metadata.get_seqlens_expanded() if expanded else metadata.get_seqlens_int32()
    return src.detach().to("cpu", torch.int32)


def _extend_lens_cpu(metadata):
    """本 batch 每请求的 query 长度列表（fork 上叫 ``get_dsa_extend_len_cpu``）。

    旧 API 名 ``get_nsa_extend_len_cpu`` 已被 fork 删除；decode 时 fork 的
    ``dsa_extend_seq_lens_list`` 是 ``[1] * bs``，与旧语义一致。
    """
    fn = getattr(metadata, "get_nsa_extend_len_cpu", None)
    if fn is None:
        fn = metadata.get_dsa_extend_len_cpu
    return fn()


def _indexer_trace(tag: str) -> None:
    """``EB5_INDEXER_TRACE=1`` 时打点（只在需要逐层定位时开，正常服务零开销）。"""
    if os.environ.get("EB5_INDEXER_TRACE", "0") == "1":
        print(f"[eb5-indexer-trace] {tag}", flush=True)


def _enable_prefill_cp(module) -> bool:
    """fork 把 ``nsa_enable_prefill_cp`` 改名为 ``dsa_enable_prefill_cp``。"""
    for name in ("dsa_enable_prefill_cp", "nsa_enable_prefill_cp"):
        if hasattr(module, name):
            return bool(getattr(module, name))
    return False


def _linear_w(module, x: torch.Tensor) -> torch.Tensor:
    """直接拿模块自己的权重做 ``F.linear``。

    EB5 的 indexer 是 **fused** 形态（``use_dsa_indexer_fusion=True``）：只有
    ``wk_weights_proj``，没有 ``wk`` / ``weights_proj``。而 fork 基类的
    ``_get_k_bf16`` 只支持非 fused 分支（直接 ``self.wk``），插件原来又硬用了
    ``self.wk`` → ``AttributeError: 'EB5FullMLAIndexer' object has no attribute 'wk'``。
    走 ``module.weight`` 可以同时绕开 ``wk`` / ``_fused_k_weights`` / ``weights_proj``
    这三套 API 差异。
    """
    w = getattr(module, "weight", None)
    if w is None:  # 兜底：没有 weight 的模块（理论上不会走到）
        out = module(x)
        return out[0] if isinstance(out, tuple) else out
    return F.linear(x, w)


def _k_weights(self):
    """返回 ``(k_weight, gate_weight_or_None)``，同时覆盖 fused / 非 fused。"""
    if bool(getattr(self, "use_dsa_indexer_fusion", False)):
        w = self.wk_weights_proj.weight  # [head_dim + n_heads, hidden]
        return w[: self.head_dim], w[self.head_dim :]
    return self.wk.weight, None


def _qk_from_weights(self, q_lora, x, positions):
    """纯 torch 的 q/k 投影 + k_norm + rotary + （非 fused 的）Hadamard 旋转。"""
    fusion = bool(getattr(self, "use_dsa_indexer_fusion", False))
    q = _linear_w(self.wq_b, q_lora).view(-1, self.n_heads, self.head_dim)
    kw, gate_w = _k_weights(self)
    if fusion:
        kw_out = F.linear(x, self.wk_weights_proj.weight)
        key = kw_out[..., : self.head_dim]
        gate_w = kw_out[..., self.head_dim :]
    else:
        key = F.linear(x, kw)
    key = self.k_norm(key)

    rope = self.rope_head_dim
    q_rope = q[..., :rope].contiguous()
    k_rope = key[..., :rope].contiguous()
    q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
    q = torch.cat([q_rope, q[..., rope:]], dim=-1)
    key = torch.cat([k_rope, key[..., rope:]], dim=-1)
    if not fusion:
        # fused 路径在训练时就去掉了 Hadamard（基类 ``_maybe_rotate`` 的语义）。
        q = _rotate_activation(q)
        key = _rotate_activation(key)
    _indexer_trace(f"qk:done n={x.shape[0]} fusion={fusion} gate={'yes' if gate_w is not None else 'no'}")
    return q, key, gate_w


def _k_from_weights(self, x, positions):
    """K-only（``kv_len <= index_topk`` 的短上下文快路径）。"""
    fusion = bool(getattr(self, "use_dsa_indexer_fusion", False))
    if fusion:
        kw_out = F.linear(x, self.wk_weights_proj.weight)
        key = kw_out[..., : self.head_dim]
    else:
        key = F.linear(x, self.wk.weight)
    key = self.k_norm(key)
    rope = self.rope_head_dim
    k_rope = key[..., :rope].contiguous()
    dummy = torch.empty_like(k_rope)
    _, k_rope = self.rotary_emb(positions, dummy, k_rope)
    key = torch.cat([k_rope, key[..., rope:]], dim=-1)
    if not fusion:
        key = _rotate_activation(key)
    _indexer_trace(f"k-only:done n={x.shape[0]} fusion={fusion}")
    return key


def _get_k_bf16_compat(module, x, positions):
    """兼容层：调用上面的纯 torch K-only 实现。"""
    return _k_from_weights(module, x, positions)


def _k_proj(self, x: torch.Tensor) -> torch.Tensor:
    """融合 / 非融合都能用的 K 投影（非 fused 走 ``self.wk``）。"""
    if bool(getattr(self, "use_dsa_indexer_fusion", False)):
        return F.linear(x, self.wk_weights_proj.weight)[..., : self.head_dim]
    return F.linear(x, self.wk.weight)


def _maybe_rotate(self, t: torch.Tensor) -> torch.Tensor:
    """fused indexer 不做 Hadamard 旋转（基类 ``_maybe_rotate`` 的语义）。"""
    fn = getattr(self, "_maybe_rotate", None)
    return fn(t) if fn is not None else _rotate_activation(t)


_KERNEL_BINDINGS_FIXED = False


def _fix_kernel_bindings_once() -> None:
    """把晚导入的模块里的 Triton kernel 统一重装成 ``KernelLauncher``。

    根因：``kernel_ops`` 的安装/修复 pass 发生在这些模块被 import 之前。之后
    sglang 在第一次 prefill / decode 时才 import
    ``mem_cache.allocator.paged``、``attention.dsa.transform_index`` 等，
    这些模块里的 kernel 仍是原始函数，``kernel[grid](...)`` 直接抛
    ``TypeError: 'function' object is not subscriptable``（本次先后命中
    ``transform_index_page_table_prefill_kernel`` 与 ``alloc_decode_kernel``）。

    做法：在第一次 indexer 调用（prefill）时先把常见晚加载模块 import 进来，
    再执行一次 ``reapply_all()`` + ``repair_imported_bindings()``。只做一次。
    """
    global _KERNEL_BINDINGS_FIXED
    if _KERNEL_BINDINGS_FIXED:
        return
    _KERNEL_BINDINGS_FIXED = True
    for mod in (
        "sglang.kernels.ops.attention.dsa.transform_index",
        "sglang.srt.mem_cache.allocator.paged",
        "sglang.srt.mem_cache.allocator.swa",
        "sglang.srt.mem_cache.allocator.common",
        "sglang.srt.mem_cache.utils",
    ):
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            pass
    try:
        from sglang_kunlun.kernels.kernel_ops import (
            reapply_all,
            repair_imported_bindings,
        )

        reapply_all()
        n = repair_imported_bindings()
        _indexer_trace(f"kernel-bindings:reapplied (repaired={n})")
    except Exception as exc:  # noqa: BLE001
        _indexer_trace(f"kernel-bindings:skip ({exc})")


def _indexer_torch_ref(
    self,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
    return_indices: bool = True,
):
    """indexer 的**纯 torch 散算子**实现（默认启用）。

    与 mHC 的 ``mhc_pre_torch_ref`` 同一思路：只把"投影 / norm / rotary / 门控"
    用 torch 表达，logits 与 topk 复用插件里已经在昆仑上跑通的
    ``_get_topk_paged`` / ``_get_topk_ragged``（它们内部走 paged MQA logits 与
    ``metadata.topk_transform``），**不重新实现一套 paged logits**。

    这样 EB5（fused indexer）和 DSV3/DSV4（非 fused）走同一条 torch 通路，
    不再依赖 ``wk`` / ``_fused_k_weights`` / ``weights_proj`` 的 API 形态。
    ``EB5_INDEXER_TORCH_REF=0`` 可退回原插件实现做对照。
    """
    import kunlun_ops

    _fix_kernel_bindings_once()
    x_meta = x[0] if isinstance(x, tuple) else x
    metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
    if metadata is None:
        return None

    skip_logits = False
    if (
        forward_batch.forward_mode.is_extend_without_speculative()
        and forward_batch.seq_lens_cpu is not None
    ):
        skip_logits = int(forward_batch.seq_lens_cpu.max().item()) <= self.index_topk
    _indexer_trace(f"enter layer={layer_id} skip_logits={skip_logits}")

    if skip_logits and not _enable_prefill_cp(self):
        key = _k_from_weights(self, x_meta, positions)
        key_shape = key.shape
        key2 = key.view(-1, self.block_size)
        k_fp8 = torch.empty(key2.shape, device=key2.device, dtype=torch.int8)
        k_scale = torch.empty([key2.shape[0], 1], device=key2.device, dtype=torch.float32)
        kunlun_ops.quant2d(key2, k_fp8, k_scale, force_sdnn=True)
        k_fp8 = k_fp8.view(key_shape)
        k_scale = k_scale.view(key_shape[0], -1)
        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
        _indexer_kv_pool().set_index_k_scale_buffer(
            layer_id=layer_id,
            loc=forward_batch.out_cache_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )
        if not return_indices:
            return None
        return _topk_all_visible(metadata, self.index_topk, x_meta.device)

    query, key, gate_w = _qk_from_weights(self, q_lora, x_meta, positions)
    query_shape, key_shape = query.shape, key.shape
    query = query.view(-1, self.block_size)
    key = key.view(-1, self.block_size)
    q_fp8 = torch.empty(query.shape, device=query.device, dtype=torch.int8)
    q_scale = torch.empty([query.shape[0], 1], device=query.device, dtype=torch.float32)
    k_fp8 = torch.empty(key.shape, device=key.device, dtype=torch.int8)
    k_scale = torch.empty([key.shape[0], 1], device=key.device, dtype=torch.float32)
    kunlun_ops.quant2d(query, q_fp8, q_scale, force_sdnn=True)
    q_fp8 = q_fp8.view(query_shape)
    q_scale = q_scale.view(query_shape[0], -1)
    kunlun_ops.quant2d(key, k_fp8, k_scale, force_sdnn=True)
    k_fp8 = k_fp8.view(key_shape)
    k_scale = k_scale.view(key_shape[0], -1)
    _indexer_trace("quant:done")

    if not forward_batch.out_cache_loc.is_contiguous():
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
    _indexer_kv_pool().set_index_k_scale_buffer(
        layer_id=layer_id,
        loc=forward_batch.out_cache_loc,
        index_k=k_fp8,
        index_k_scale=k_scale,
    )

    if gate_w is not None:
        # fused 路径：head-gate 与 K 同一次投影出来（等价于基类 _get_logits_head_gate）
        weights = gate_w.float() * self.n_heads**-0.5
        weights = weights * q_scale * self.softmax_scale
    else:
        x_for_gate = x_meta
        if isinstance(x, tuple) and len(x) in (2, 3):
            x_q, x_s = x[0], x[1]
            if x_s is not None and x_q.dim() == 2 and x_s.dim() == 2 and x_q.shape[0] == x_s.shape[0]:
                m, n = x_q.shape
                ng = x_s.shape[1]
                if ng > 0 and n % ng == 0:
                    group = n // ng
                    x_for_gate = (
                        x_q.to(torch.float32).view(m, ng, group)
                        .mul_(x_s.to(torch.float32).unsqueeze(-1))
                        .view(m, n).to(torch.bfloat16)
                    )
                else:
                    x_for_gate = x_q.to(torch.bfloat16)
            else:
                x_for_gate = x_q.to(torch.bfloat16)
        weights = self._get_logits_head_gate(x_for_gate, q_scale)
    if q_fp8.dtype is torch.int8:
        weights /= 127

    assert forward_batch.seq_lens_cpu is not None
    if len(forward_batch.seq_lens_cpu) == 0:
        return torch.full(
            (x_meta.shape[0], self.index_topk), -1, dtype=torch.int, device=x_meta.device
        )

    if (
        forward_batch.forward_mode.is_decode_or_idle()
        or forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend(include_v2=True)
    ):
        out = self._get_topk_paged(forward_batch, layer_id, q_fp8, weights, metadata)
        _indexer_trace(f"topk-paged:done shape={tuple(out.shape)}")
        return out

    out = self._get_topk_ragged(forward_batch, layer_id, q_fp8, weights, metadata)
    _indexer_trace(f"topk-ragged:done shape={tuple(out.shape)}")
    return out


def _shape_meta(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "contiguous": tensor.is_contiguous(),
        }
    if isinstance(tensor, (list, tuple)):
        return [_shape_meta(item) for item in tensor]
    return tensor


def _small_cpu_list(tensor, limit: int = 16):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        flat = tensor.detach().flatten()
        return flat[:limit].cpu().tolist()
    return tensor


def _make_cu_seqlens_from_lens(seq_lens_cpu: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    cu_cpu = torch.nn.functional.pad(
        torch.cumsum(seq_lens_cpu.to(torch.int32), dim=0, dtype=torch.int32), (1, 0)
    )
    # NOTE: pageable-host -> XPU copies must be synchronous. The XPU runtime
    # enqueues an async DMA from the (temporary, unpinned) host buffer, so with
    # non_blocking=True the sparse-attention/indexer kernels can consume stale
    # cu_seqlens and index out of bounds (illegal memory access / NaN rows).
    cu_xpu = cu_cpu.to(device=device)
    return cu_cpu, cu_xpu


# --------------------------------------------------------------------------
# Helpers (free functions — used by the patched methods below).
# --------------------------------------------------------------------------


def chunk_cu_seqlens(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
):
    """topk 按照 chunk_size 切分原始 sequence 区间."""
    cu = cu_seqlens.squeeze(0)
    total_len = cu[-1].item()
    seg_starts = cu[:-1]
    seg_ends = cu[1:]
    results = []
    for chunk_start in range(0, total_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_len)
        inter_starts = torch.clamp(seg_starts, min=chunk_start)
        inter_ends = torch.clamp(seg_ends, max=chunk_end)
        valid_mask = inter_starts < inter_ends
        valid_inter_starts = inter_starts[valid_mask]
        valid_inter_ends = inter_ends[valid_mask]
        inter_lens = valid_inter_ends - valid_inter_starts
        new_cu_seqlens = torch.nn.functional.pad(
            torch.cumsum(inter_lens, dim=0, dtype=torch.int32), (1, 0)
        )
        interval_bounds = torch.cat([valid_inter_starts[:1], valid_inter_ends])
        chunk_range = torch.tensor(
            [valid_inter_starts[0].item(), valid_inter_ends[-1].item()],
            dtype=torch.int32,
        )
        results.append((new_cu_seqlens, interval_bounds, chunk_range))
    return results


def _hadamard_transform(x: torch.Tensor, scale: float = None) -> torch.Tensor:
    if scale is None:
        return torch.ops.xspeedgate_ops.hadamard_transform(x)
    return torch.ops.xspeedgate_ops.hadamard_transform(x, scale)


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return _hadamard_transform(x, hidden_size**-0.5)


# --------------------------------------------------------------------------
# Indexer method replacements (plugin_hook REPLACE).
# --------------------------------------------------------------------------


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._project_and_scale_head_gates",
    type=HookType.REPLACE,
)
@torch.compile(dynamic=True)
def _project_and_scale_head_gates(self, x: torch.Tensor):
    weights, _ = self.weights_proj(x.to(self.weights_proj.weight.dtype))
    weights = weights.float()
    weights = weights * self.n_heads**-0.5
    return weights


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_logits_head_gate",
    type=HookType.REPLACE,
)
@torch.compile(dynamic=True)
def _get_logits_head_gate(self, x: torch.Tensor, q_scale: torch.Tensor):
    weights, _ = self.weights_proj(x.to(self.weights_proj.weight.dtype))
    weights = weights.float()
    weights = weights * self.n_heads**-0.5
    weights = weights * q_scale * self.softmax_scale
    return weights


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_paged",
    type=HookType.REPLACE,
)
def _get_topk_paged(
    self,
    forward_batch: ForwardBatch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata,
) -> torch.Tensor:
    """_get_topk_paged Kunlun replacement."""
    import kunlun_ops

    if TYPE_CHECKING:
        assert isinstance(_indexer_kv_pool(), NSATokenToKVPool)

    page_size = _indexer_kv_pool().page_size
    assert page_size == 64, "only support page size 64"
    block_tables = metadata.get_page_table_64()
    max_seq_len = block_tables.shape[1] * page_size
    kv_cache_fp8 = _indexer_kv_pool().get_index_k_with_scale_buffer(
        layer_id=layer_id
    )
    block_kv = 64
    num_heads_kv = 1

    if (
        forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend(include_v2=True)
    ):
        seqlens_32 = metadata.get_seqlens_expanded()
        seqlens_32_cpu = _seqlens_cpu(metadata, forward_batch, expanded=True)
    else:
        seqlens_32 = metadata.get_seqlens_int32()
        seqlens_32_cpu = _seqlens_cpu(metadata, forward_batch, expanded=False)

    assert len(q_fp8.shape) == 3, f"q_fp8.shape: {q_fp8.shape}"
    q_fp8 = q_fp8.unsqueeze(1)

    if len(weights.shape) == 2:
        weights = weights.unsqueeze(1)

    q_offset = sum(_extend_lens_cpu(metadata))

    head_dim = _indexer_kv_pool().index_head_dim
    k_fp8 = (
        kv_cache_fp8[:, : page_size * head_dim]
        .view(kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim)
        .to(torch.int8)
    )
    k_scale = torch.ops.xspeedgate_ops.gather_paged_scale(
        # indexer 的 k 缓存按 fp8 存，但 buffer 的 dtype 是 uint8(Byte)；vendor 算子
        # 只收 Char(int8)。2026-10-20 22:24 实测：
        # ``RuntimeError: dtype of kv_cache (Byte) must be one of [Char]``。
        # ``view`` 同 itemsize，零拷贝。
        kv_cache=kv_cache_fp8.view(torch.int8),
        block_tables=block_tables,
        page_size=page_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        scale_offset=page_size * head_dim,
        context_lens=seqlens_32,
        q_is_int8=(q_fp8.dtype == torch.int8),
    )

    fused_kv_cache = [k_fp8.contiguous(), k_scale.contiguous()]
    logits = torch.empty(
        (q_offset, q_fp8.shape[1], max_seq_len),
        dtype=torch.float32,
        device=q_fp8.device,
    )

    kunlun_ops.I8_paged_mqa_logits(
        q=q_fp8[:q_offset],
        fused_kv_cache=fused_kv_cache,
        weights=weights[:q_offset],
        context_lens=[seqlens_32_cpu, seqlens_32],
        block_table=block_tables,
        max_context_len=max_seq_len,
        clean_logits=False,
        out=logits,
        use_xfa_boost=False,
    )

    logits = logits.reshape(-1, max_seq_len)
    # ``fast_topk_transform_fused`` 由 ``sgl_kernel_kunlun/top_k.py`` 的 torch 版
    # 实现在 stub 层兜住（pipeline10 已端到端验证，见 doc §11.13），调用点保持上游写法。
    topk_result = metadata.topk_transform(logits, self.index_topk)
    if q_offset < q_fp8.shape[0]:
        pad_len = q_fp8.shape[0] - q_offset
        padding = torch.full(
            (pad_len, topk_result.shape[1]),
            -1,
            dtype=topk_result.dtype,
            device=topk_result.device,
        )
        topk_result = torch.cat([topk_result, padding], dim=0)
    del logits
    return topk_result


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_topk_ragged",
    type=HookType.REPLACE,
)
def _get_topk_ragged(
    self,
    forward_batch: ForwardBatch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    metadata,
) -> torch.Tensor:
    """_get_topk_ragged Kunlun replacement."""
    import kunlun_ops

    if TYPE_CHECKING:
        assert isinstance(_indexer_kv_pool(), NSATokenToKVPool)
    assert forward_batch.forward_mode.is_extend_without_speculative()

    page_size = _indexer_kv_pool().page_size
    assert page_size == 64, "only support page size 64"
    if len(weights.shape) == 3:
        weights = weights.squeeze(-1)
    k_fp8_list = []
    k_scale_list = []
    block_tables = metadata.get_page_table_64()

    assert (
        forward_batch.seq_lens_cpu is not None
        and forward_batch.extend_seq_lens_cpu is not None
    )

    batch_size = len(block_tables)
    token_nums, _, _ = q_fp8.shape
    device = q_fp8.device
    topk_result = torch.full(
        (token_nums, self.index_topk), -1, device=device, dtype=torch.int32
    )
    if batch_size == 0:
        return topk_result

    indexer_seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
    assert len(indexer_seq_lens_cpu) == batch_size
    for i in range(batch_size):
        seq_len = indexer_seq_lens_cpu[i].item()
        assert isinstance(seq_len, int)
        k_fp8 = _indexer_kv_pool().get_index_k_continuous(
            layer_id, seq_len, block_tables[i]
        )
        k_scale = _indexer_kv_pool().get_index_k_scale_continuous(
            layer_id, seq_len, block_tables[i]
        )
        k_fp8_list.append(k_fp8)
        k_scale_list.append(k_scale)

    k_fp8 = torch.cat(k_fp8_list, dim=0)
    k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
    if q_fp8.dtype is torch.int8:
        k_scale /= 127
    ks, ke = metadata.get_indexer_kvcache_range()
    seq_lens_expanded = metadata.get_seqlens_expanded()
    token_to_batch_idx = metadata.get_token_to_batch_idx()
    q_offset = ks.shape[0]
    k_offset = k_fp8.shape[0]

    token_nums, _, _ = q_fp8.shape
    device = q_fp8.device

    need_chunk, free_mem = self._should_chunk_mqa_logits(q_offset, k_offset, device)

    context_q_lens_xpu = metadata.get_cu_seqlens_q()
    # fork 的 ``DSAIndexerMetadata`` 没有 ``get_cu_seqlens_q_cpu``（只有 xpu 版）。
    _cu_cpu_fn = getattr(metadata, "get_cu_seqlens_q_cpu", None)
    context_q_lens_cpu = (
        _cu_cpu_fn()
        if _cu_cpu_fn is not None
        else metadata.get_cu_seqlens_q().detach().to("cpu", torch.int32)
    )
    context_k_lens_cpu, context_k_lens_xpu = _make_cu_seqlens_from_lens(
        indexer_seq_lens_cpu, device
    )
    if not need_chunk:
        assert q_fp8[:q_offset].shape[0] != 0

        max_k = int(indexer_seq_lens_cpu.max().item())
        logits = torch.full(
            (q_offset, max_k), -1.0, dtype=torch.float32, device=q_fp8.device
        )
        kunlun_ops.I8_mqa_logits(
            q=q_fp8[:q_offset],
            fused_kv_cache=(k_fp8.contiguous(), k_scale.contiguous()),
            weights=weights[:q_offset],
            context_q_lens=(context_q_lens_cpu, context_q_lens_xpu),
            context_k_lens=(context_k_lens_cpu, context_k_lens_xpu),
            logits=logits,
            clean_logits=True,
            use_xfa_boost=False,
        )

        ks_local = torch.zeros(q_offset, dtype=ks.dtype, device=ks.device)
        ke_local = seq_lens_expanded.to(dtype=ks.dtype)
        torch.ops.xspeedgate_ops.mask_for_I8_mqa_logits(
            seq_len_kv=max_k,
            cu_seqlen_ks=ks_local.contiguous(),
            cu_seqlen_ke=ke_local.contiguous(),
            logits=logits,
        )

        assert logits.shape[0] == len(seq_lens_expanded)

        page_table_1 = metadata.get_page_table_1()
        cu_seqlens_q_topk = metadata.attn_metadata.cu_seqlens_q
        dst_page_table = logits.new_empty(
            (logits.size(0), self.index_topk), dtype=torch.int32
        )
        torch.ops.xspeedgate_ops.topk_transform(
            score=logits.contiguous(),
            lengths=seq_lens_expanded.contiguous(),
            src_page_table=page_table_1.contiguous(),
            dst_page_table=dst_page_table,
            topk=self.index_topk,
            cu_seqlens_q=cu_seqlens_q_topk.contiguous(),
        )
        del logits
        del ks, q_fp8, k_fp8, k_scale
        topk_result[:q_offset] = dst_page_table
        return topk_result

    bytes_per_elem = 4
    bytes_per_row = k_offset * bytes_per_elem
    max_rows = max(1, int((free_mem * 0.5) // max(bytes_per_row, 1)))
    max_rows = min(max_rows, q_offset)

    global_topk_offset = metadata.attn_metadata.topk_indices_offset
    assert seq_lens_expanded.shape[0] == q_offset
    if global_topk_offset is not None:
        assert global_topk_offset.shape[0] >= q_offset

    q_interval_cpu = chunk_cu_seqlens(context_q_lens_cpu, chunk_size=max_rows)
    q_interval_xpu = chunk_cu_seqlens(context_q_lens_xpu, chunk_size=max_rows)
    for i in range(0, len(q_interval_cpu)):
        chunk_context_q_lens_cpu, _, chunk_range = q_interval_cpu[i]
        start = chunk_range[0].item()
        end = chunk_range[1].item()
        chunk_context_q_lens_cpu = chunk_context_q_lens_cpu - chunk_context_q_lens_cpu[0]

        logits_chunk = torch.full(
            (end - start, k_fp8.shape[0]),
            -1.0,
            dtype=torch.float32,
            device=q_fp8.device,
        )
        chunk_context_q_lens_xpu, _, _ = q_interval_xpu[i]
        chunk_context_q_lens_xpu = chunk_context_q_lens_xpu - chunk_context_q_lens_xpu[0]
        kunlun_ops.I8_mqa_logits(
            q=q_fp8[start:end],
            fused_kv_cache=(k_fp8.contiguous(), k_scale.contiguous()),
            weights=weights[start:end],
            context_q_lens=(chunk_context_q_lens_cpu, chunk_context_q_lens_xpu),
            context_k_lens=(context_k_lens_cpu, context_k_lens_xpu),
            logits=logits_chunk,
            clean_logits=True,
            use_xfa_boost=False,
        )

        torch.ops.xspeedgate_ops.mask_for_I8_mqa_logits(
            seq_len_kv=k_fp8.shape[0],
            cu_seqlen_ks=ks[start:end],
            cu_seqlen_ke=ke[start:end],
            logits=logits_chunk,
        )

        if global_topk_offset is not None:
            topk_offset_chunk = global_topk_offset[start:end]
            cu_seqlens_q_chunk = None
            batch_idx_chunk = None
        else:
            topk_offset_chunk = None
            B_chunk = logits_chunk.shape[0]
            cu_seqlens_q_chunk = torch.ones(
                B_chunk, dtype=torch.int32, device=device
            )
            batch_idx_chunk = token_to_batch_idx[start:end]

        raw_topk_chunk = metadata.topk_transform(
            logits_chunk,
            self.index_topk,
            ks=ks[start:end],
            cu_seqlens_q=cu_seqlens_q_chunk,
            ke_offset=ke[start:end],
            batch_idx_list=batch_idx_chunk,
            topk_indices_offset_override=topk_offset_chunk,
        )
        del logits_chunk
        topk_result[start:end] = raw_topk_chunk
    del ks, q_fp8, k_fp8, k_scale
    del q_interval_cpu, q_interval_xpu
    return topk_result


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._get_q_k_bf16",
    type=HookType.REPLACE,
)
def _get_q_k_bf16(
    self,
    q_lora: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor,
    enable_dual_stream: bool,
    forward_batch: ForwardBatch,
):
    """_get_q_k_bf16 Kunlun replacement."""
    if enable_dual_stream:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
        q_rope, _ = torch.split(
            query,
            [self.rope_head_dim, self.head_dim - self.rope_head_dim],
            dim=-1,
        )
        with torch.cuda.stream(self.alt_stream):
            key = _k_proj(self, x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )
        current_stream.wait_stream(self.alt_stream)
    else:
        query, _ = self.wq_b(q_lora)
        query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
        q_rope, _ = torch.split(
            query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        key = _k_proj(self, x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

    q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)
    query[..., : self.rope_head_dim] = q_rope
    key[..., : self.rope_head_dim] = k_rope

    if enable_dual_stream:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        query = _maybe_rotate(self, query)
        with torch.cuda.stream(self.alt_stream):
            key = _maybe_rotate(self, key)
        current_stream.wait_stream(self.alt_stream)
    else:
        query = _maybe_rotate(self, query)
        key = _maybe_rotate(self, key)

    if forward_batch.attn_cp_metadata is not None and getattr(self, "dsa_enable_prefill_cp", False):
        from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output

        key = cp_all_gather_rerange_output(
            key.contiguous(),
            self.cp_size,
            forward_batch,
            torch.cuda.current_stream(),
        )
    return query, key


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer._forward_cuda_k_only",
    type=HookType.REPLACE,
)
def _forward_cuda_k_only(
    self,
    x: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
    act_quant,
    enable_dual_stream: bool,
    metadata,
    return_indices: bool = True,
):
    """_forward_cuda_k_only Kunlun replacement."""
    import kunlun_ops

    assert forward_batch.forward_mode.is_extend_without_speculative()
    x_meta = x[0] if isinstance(x, tuple) else x

    key = _get_k_bf16_compat(self, x, positions)
    key_shape = key.shape
    key = key.view(-1, self.block_size)
    k_fp8 = torch.empty(key.shape, device=key.device, dtype=torch.int8)
    k_scale = torch.empty(
        [key.shape[0], 1], device=key.device, dtype=torch.float32
    )
    kunlun_ops.quant2d(key, k_fp8, k_scale, force_sdnn=True)
    k_fp8 = k_fp8.view(key_shape)
    k_scale = k_scale.view(key_shape[0], -1)

    if not forward_batch.out_cache_loc.is_contiguous():
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()

    _indexer_kv_pool().set_index_k_scale_buffer(
        layer_id=layer_id,
        loc=forward_batch.out_cache_loc,
        index_k=k_fp8,
        index_k_scale=k_scale,
    )

    if not return_indices:
        return None

    return _topk_all_visible(metadata, self.index_topk, x_meta.device)


@plugin_hook(
    target="sglang.srt.layers.attention.dsa.dsa_indexer.Indexer.forward_cuda",
    type=HookType.REPLACE,
)
def forward_cuda(
    self,
    x: torch.Tensor,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    layer_id: int,
    return_indices: bool = True,
) -> Optional[torch.Tensor]:
    """forward_cuda Kunlun replacement.

    ``EB5_INDEXER_TORCH_REF=1``（默认）走纯 torch 散算子实现；EB5 的 fused indexer
    与 DSV4 的 non-fused indexer 共用一条通路，不再逐点适配 ``wk``。
    """
    if os.environ.get("EB5_INDEXER_TORCH_REF", "1") == "1":
        return _indexer_torch_ref(
            self, x, q_lora, positions, forward_batch, layer_id, return_indices
        )
    import kunlun_ops
    from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

    if TYPE_CHECKING:
        assert isinstance(_indexer_kv_pool(), NSATokenToKVPool)
    x_meta = x[0] if isinstance(x, tuple) else x
    enable_dual_stream = False

    metadata = get_attn_backend().get_indexer_metadata(layer_id, forward_batch)
    if metadata is None:
        return None

    skip_logits_computation = False
    if forward_batch.forward_mode.is_extend_without_speculative():
        if forward_batch.seq_lens_cpu is not None:
            max_kv_len = forward_batch.seq_lens_cpu.max().item()
            skip_logits_computation = max_kv_len <= self.index_topk

    if skip_logits_computation and (not getattr(self, "dsa_enable_prefill_cp", False)):
        return self._forward_cuda_k_only(
            x,
            positions,
            forward_batch,
            layer_id,
            act_quant,
            enable_dual_stream,
            metadata,
            return_indices,
        )

    if enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        weights = self._project_and_scale_head_gates(x)
        query, key = self._get_q_k_bf16(
            q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
        )
        q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
        with torch.cuda.stream(self.alt_stream):
            k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)
        current_stream.wait_stream(self.alt_stream)
        weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
    else:
        query, key = self._get_q_k_bf16(
            q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
        )
        query_shape = query.shape
        key_shape = key.shape
        query = query.view(-1, self.block_size)
        key = key.view(-1, self.block_size)
        q_fp8 = torch.empty(query.shape, device=query.device, dtype=torch.int8)
        q_scale = torch.empty(
            [query.shape[0], 1], device=query.device, dtype=torch.float32
        )
        k_fp8 = torch.empty(key.shape, device=key.device, dtype=torch.int8)
        k_scale = torch.empty(
            [key.shape[0], 1], device=key.device, dtype=torch.float32
        )

        kunlun_ops.quant2d(query, q_fp8, q_scale, force_sdnn=True)
        q_fp8 = q_fp8.view(query_shape)
        q_scale = q_scale.view(query_shape[0], -1)
        kunlun_ops.quant2d(key, k_fp8, k_scale, force_sdnn=True)
        k_fp8 = k_fp8.view(key_shape)
        k_scale = k_scale.view(key_shape[0], -1)

        if isinstance(x, tuple):
            assert len(x) in (2, 3)
            x_q, x_s = x[0], x[1]
            if (
                x_s is not None
                and x_q.dim() == 2
                and x_s.dim() == 2
                and x_q.shape[0] == x_s.shape[0]
            ):
                m, n = x_q.shape
                ng = x_s.shape[1]
                if ng > 0 and n % ng == 0:
                    group = n // ng
                    x_for_gate = (
                        x_q.to(torch.float32)
                        .view(m, ng, group)
                        .mul_(x_s.to(torch.float32).unsqueeze(-1))
                        .view(m, n)
                        .to(torch.bfloat16)
                    )
                else:
                    x_for_gate = x_q.to(torch.bfloat16)
            else:
                x_for_gate = x_q.to(torch.bfloat16)
        else:
            x_for_gate = x
        weights = self._get_logits_head_gate(x_for_gate, q_scale)

    if not forward_batch.out_cache_loc.is_contiguous():
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()
    _indexer_kv_pool().set_index_k_scale_buffer(
        layer_id=layer_id,
        loc=forward_batch.out_cache_loc,
        index_k=k_fp8,
        index_k_scale=k_scale,
    )
    if q_fp8.dtype is torch.int8:
        weights /= 127

    assert forward_batch.seq_lens_cpu is not None
    if len(forward_batch.seq_lens_cpu) == 0:
        return torch.full(
            (x_meta.shape[0], self.index_topk),
            -1,
            dtype=torch.int,
            device=x_meta.device,
        )

    if (
        forward_batch.forward_mode.is_decode_or_idle()
        or forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend(include_v2=True)
    ):
        return self._get_topk_paged(forward_batch, layer_id, q_fp8, weights, metadata)

    # CP-v1 "in-seq-split" went away with the fork's CP V1 deprecation, together
    # with ``Indexer._get_topk_ragged_with_cp`` (and ``ForwardBatch.nsa_cp_metadata``
    # itself). EB5 runs with prefill CP disabled, so this branch is unreachable;
    # keep an explicit guard so that turning CP on fails loudly at the first
    # forward instead of with an AttributeError from a missing method/attribute.
    from sglang.srt.layers.attention.dsa import utils as _dsa_utils

    _cp_metadata = getattr(forward_batch, "nsa_cp_metadata", None)
    _is_in_seq_split = getattr(_dsa_utils, "is_dsa_prefill_cp_in_seq_split", None)
    if (
        _cp_metadata is not None
        and _is_in_seq_split is not None
        and _is_in_seq_split()
    ):
        raise NotImplementedError(
            "NSA in-seq-split prefill CP is not supported by the Kunlun plugin: "
            "the fork removed the Indexer CP top-k helper and "
            "dsa.utils.is_dsa_prefill_cp_in_seq_split. Run with prefill CP "
            "disabled (attn_cp_size=1) or use the round-robin/interleave strategy."
        )

    return self._get_topk_ragged(forward_batch, layer_id, q_fp8, weights, metadata)

