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
"""Kunlun-specific KV pool subclasses.

These subclasses override methods that would otherwise need REPLACE hooks,
allowing the platform factory methods to return them directly so the main
sglang code never needs patching for these classes.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    unwrap_write_loc,
)


class KunlunMHATokenToKVPool(MHATokenToKVPool):
    """Kunlun MHA KV pool with custom buffer creation and reshape-and-cache support."""

    def _create_buffers(self):
        """Create Kunlun-compatible key and value cache buffers."""
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ):
                if self.head_dim == self.v_head_dim:
                    self.k_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(
                            (self.size + self.page_size, self.head_num, self.v_head_dim),
                            dtype=self.store_dtype,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                else:
                    num_pages = self.size // self.page_size
                    num_blocks_total = num_pages + 1
                    shape = (num_blocks_total, self.head_num, self.page_size, self.head_dim)
                    self.k_buffer = [
                        torch.zeros(shape, dtype=self.store_dtype, device=self.device)
                        for _ in range(self.layer_num)
                    ]
                    self.v_buffer = [
                        torch.zeros(shape, dtype=self.store_dtype, device=self.device)
                        for _ in range(self.layer_num)
                    ]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_buffer] + [x.data_ptr() for x in self.v_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [
                np.prod(x.shape[1:]) * x.dtype.itemsize
                for x in self.k_buffer + self.v_buffer
            ],
            device=self.device,
        )

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info, # KVWriteLoc
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        layer_id_override: Optional[int] = None,
    ):
        """Set KV cache buffer using the kunlun_ops reshape_and_cache kernel."""
        from kunlun_ops import reshape_and_cache

        loc, _ = unwrap_write_loc(loc_info)

        if self.head_dim != self.v_head_dim:
            cache_v = F.pad(cache_v, pad=(0, 64), mode="constant", value=0)
        if layer_id_override is not None:
            layer_id = layer_id_override
        else:
            layer_id = layer.layer_id
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            if not self.enable_int8_kv_cache:
                cache_k = cache_k.to(self.dtype)
                cache_v = cache_v.to(self.dtype)

        if self.store_dtype != self.dtype:
            cache_k = cache_k.view(self.store_dtype)
            cache_v = cache_v.view(self.store_dtype)

        if cache_k.dim() == 3 and cache_k.shape[0] == loc.numel():
            k_bhd = cache_k.contiguous()
            v_bhd = cache_v.contiguous()
        elif cache_k.dim() == 3 and cache_k.shape[1] == loc.numel():
            k_bhd = cache_k.permute(1, 0, 2).contiguous()
            v_bhd = cache_v.permute(1, 0, 2).contiguous()
        else:
            raise RuntimeError(
                f"Unexpected K/V shape {tuple(cache_k.shape)} vs loc {tuple(loc.shape)}"
            )

        k4 = self.k_buffer[layer_id - self.start_layer]
        v4 = self.v_buffer[layer_id - self.start_layer]
        reshape_and_cache(k_bhd, v_bhd, k4, v4, loc.to(torch.int32), None, None, 0)

    def get_contiguous_buf_infos(self):
        """Return contiguous buffer pointers, byte lengths, and item sizes."""
        kv_data_ptrs = [
            self._get_key_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).data_ptr()
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_data_lens = [
            self._get_key_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i).nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        kv_item_lens = [
            self._get_key_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ] + [
            self._get_value_buffer(i)[0].nbytes
            for i in range(self.start_layer, self.start_layer + self.layer_num)
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_cpu_copy(self, indices):
        """Copy KV cache pages from GPU to CPU in chunks."""
        torch.cuda.synchronize()
        is_4d = self.head_dim != self.v_head_dim
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            k_buf = self.k_buffer[layer_id]
            v_buf = self.v_buffer[layer_id]
            if is_4d:
                k_flat = k_buf.reshape(-1, self.head_num, self.head_dim)
                v_flat = v_buf.reshape(-1, self.head_num, self.head_dim)
            else:
                k_flat = k_buf
                v_flat = v_buf
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu = k_flat[chunk_indices].to("cpu", non_blocking=True)
                v_cpu = v_flat[chunk_indices].to("cpu", non_blocking=True)
                kv_cache_cpu[-1].append([k_cpu, v_cpu])
        torch.cuda.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(self, kv_cache_cpu, indices):
        """Load KV cache pages from CPU back to GPU in chunks."""
        torch.cuda.synchronize()
        is_4d = self.head_dim != self.v_head_dim
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            k_buf = self.k_buffer[layer_id]
            v_buf = self.v_buffer[layer_id]
            if is_4d:
                k_flat = k_buf.reshape(-1, self.head_num, self.head_dim)
                v_flat = v_buf.reshape(-1, self.head_num, self.head_dim)
            else:
                k_flat = k_buf
                v_flat = v_buf
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                k_cpu, v_cpu = (
                    kv_cache_cpu[layer_id][i // chunk_size][0],
                    kv_cache_cpu[layer_id][i // chunk_size][1],
                )
                assert k_cpu.shape[0] == v_cpu.shape[0] == len(chunk_indices)
                k_chunk = k_cpu.to(k_flat.device, non_blocking=True)
                v_chunk = v_cpu.to(v_flat.device, non_blocking=True)
                k_flat[chunk_indices] = k_chunk
                v_flat[chunk_indices] = v_chunk
        torch.cuda.synchronize()


class KunlunMLATokenToKVPool(MLATokenToKVPool):
    """Kunlun MLA KV pool with custom KV buffer set/get operations."""

    def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
        """Set MLA KV buffer using the kunlun_ops triton kernel."""
        from kunlun_ops import set_mla_kv_buffer_triton

        layer_id = layer.layer_id
        if cache_k_nope.dtype != self.dtype:
            cache_k_nope = cache_k_nope.to(self.dtype)
            cache_k_rope = cache_k_rope.to(self.dtype)
        if self.store_dtype != self.dtype:
            cache_k_nope = cache_k_nope.view(self.store_dtype)
            cache_k_rope = cache_k_rope.view(self.store_dtype)

        set_mla_kv_buffer_triton(
            self.kv_buffer[layer_id],
            loc,
            cache_k_nope.contiguous().view(cache_k_nope.shape[0], cache_k_nope.shape[-1]),
            cache_k_rope.contiguous().view(cache_k_rope.shape[0], cache_k_rope.shape[-1]),
        )
        _eb5_kvw_probe(layer_id, loc, cache_k_nope, cache_k_rope, self.kv_buffer[layer_id])

    def get_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        dst_dtype: Optional[torch.dtype] = None,
    ):
        """Get MLA KV buffer using the xspeedgate_ops kernel."""
        layer_id = layer.layer_id
        kv_buffer = self.get_key_buffer(layer_id)
        dst_dtype = dst_dtype or self.dtype
        cache_k_nope = torch.empty(
            (loc.shape[0], 1, self.kv_lora_rank),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        cache_k_rope = torch.empty(
            (loc.shape[0], 1, self.qk_rope_head_dim),
            dtype=dst_dtype,
            device=kv_buffer.device,
        )
        torch.ops.xspeedgate_ops.get_mla_kv_buffer(
            kv_buffer,
            loc,
            cache_k_nope,
            cache_k_rope,
        )
        return cache_k_nope, cache_k_rope


class KunlunNSATokenToKVPool(DSATokenToKVPool):
    """Kunlun DSA/NSA KV pool with int8 index buffer dtype."""
    index_k_with_scale_buffer_dtype = torch.int8


_EB5_KVW = {"hits": 0, "seen": set()}


def _eb5_kvw_probe(layer_id, loc, nope, rope, buf) -> None:
    """Account for the KV write that the reader says never happened.

    Gated by ``EB5_KVW=1``. The reader (``_mla_gather_kv``) finds slots 256..382
    populated and every slot from 383 on permanently NaN while attention treats
    them as live. ``set_mla_kv_buffer`` (the torch stand-in) is never called, so
    this is the real writer. Log which slots it is handed, whether the source is
    finite, and whether the buffer actually holds finite values right after the
    write -- which separates "never written", "written with NaN" and "written but
    not visible".
    """
    import os

    if os.environ.get("EB5_KVW", "0") != "1":
        return
    st = _EB5_KVW
    if st["hits"] >= 14:
        return
    try:
        import torch

        fl = loc.reshape(-1)
        lo, hi = int(fl.min().item()), int(fl.max().item())
        # 之前这里有 `if hi < 380: return`，而写侧的 loc 很可能本来就只在 0..200
        # 区间（序列只有 ~130 token），于是把答案自己过滤掉了。读侧要的槽位是
        # 恰好 `[page_size, page_size+len-1]` 的连续段（128 -> [256,383]，
        # 133 -> [256,388]），疑似整体偏移了一个 page。所以这里不再按值过滤。
        key = (int(layer_id), hi // 32)
        if key in st["seen"]:
            return
        st["seen"].add(key)
        st["hits"] += 1
        src_bad = int((~torch.isfinite(nope.float())).sum().item()) + int(
            (~torch.isfinite(rope.float())).sum().item()
        )
        row = buf.reshape(buf.shape[0], -1)[hi]
        post_bad = int((~torch.isfinite(row.float())).sum().item())
        print(
            f"[kvw] layer={layer_id} loc=[{lo},{hi}] n={fl.numel()} "
            f"src_nonfinite={src_bad} buf_slot{hi}_nonfinite={post_bad}/{row.numel()}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[kvw] failed: {exc!r}", flush=True)
