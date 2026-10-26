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
"""Hooks for ``sglang.srt.mem_cache.deepseek_v4_compress_state``.

2MB-aligned re-allocation of the CompressStatePool kv_score buffer for Kunlun RDMA
(see ``_align`` for why). Approach: let the ORIGINAL ``_alloc_kv_score_buffer`` run
(correct shape/dtype for every pool variant), then swap the freshly-zeroed buffer for
a 2MB-aligned buffer of the SAME shape/dtype/device. CRITICAL: free the original
buffer BEFORE allocating its aligned replacement (and empty the XPU cache) so peak
memory stays at +1 buffer, not 2x the whole pool (that OOMs at mem-fraction 0.8+).
"""

from __future__ import annotations

import logging

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

from ._align import alloc_2m_aligned, empty_cache, is_kunlun

logger = logging.getLogger(__name__)

_ZERO_FILL_LOGGED = False


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_compress_state.CompressStatePool."
    "_alloc_kv_score_buffer",
    type=HookType.AROUND,
)
def compress_state_alloc_kunlun(original_fn, self, **kwargs):
    """2MB-align the CompressStatePool kv_score buffer for Kunlun RDMA."""
    original_fn(self, **kwargs)
    if not is_kunlun():
        return
    from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore

    ks = getattr(self, "kv_score_buffer", None)
    if ks is None or not isinstance(getattr(ks, "kv_score", None), torch.Tensor):
        return
    t = ks.kv_score
    shape, dtype, device = tuple(t.shape), t.dtype, t.device
    self.kv_score_buffer = None  # free original before reallocating
    del ks, t
    empty_cache()
    self.kv_score_buffer = KVAndScore(alloc_2m_aligned(shape, dtype, device))


@plugin_hook(
    "sglang.srt.mem_cache.deepseek_v4_compress_state.CompressStatePool.__init__",
    type=HookType.AROUND,
)
def compress_state_init_no_neg_inf_sentinel(original_fn, self, *args, **kwargs):
    """c128 状态池不留 ``-inf`` 哨兵行 —— EB5 decode 塌缩成 PAD 的修复。

    上游 ``__init__`` 在分配完缓冲后对非 HIP 后端执行
    ``self.kv_score_buffer[-1].clear()``，而 ``KVAndScore.clear()`` 是
    ``kv.zero_(); score.fill_(-inf)``。``compress_forward`` 会对一页内
    ``ratio`` 行的 score 列做 ``softmax(dim=1)``，这一行 ``-inf`` 会进入
    softmax。本 hook 在 ``__init__`` 之后把整池重新填成**有限零**。

    **已验证的事实**（昆仑 P800，EB5 43L INT8，2 题 × cap 60 快回路，pad token 数）：

    * 不动（上游行为，池内仅最后一行 score = ``-inf``）→ 18/60、18/60，
      decode 第 42 步起 ``argmax`` 永久锁死在 ``pad_token_id=0``
    * 改成整池 ``clear()``（128 行 score 全 ``-inf``）→ 18/60、18/60，**更糟不改善**
    * 整池 ``zero_()`` → **0/60、0/60**；完整 AIME 8 题 ``1/8 → 6/8``，
      ``stop_rate`` 0.125 → 0.875，``pad_ratio`` 全程 0

    故障链（每环都有实测）：压缩器输出整行 NaN → 写进 c128 KV 缓冲 →
    HCA layer 40 读到后注意力输出 32768/32768 全 NaN（入参 q/k 干净）→
    logits 全 NaN → ``argmax`` 恒为 0，而 0 不在 ``eos_token_id=[2,8]`` 里 →
    解码成空串、不停机、撞 ``max_tokens``。

    **尚未解释的部分**：昆仑上这块缓冲**本来就是零**（``alloc_2m_aligned`` 用的是
    ``torch.zeros``），所以本 hook 实际改变的只有"最后一行的 ``-inf``"。但探针测到
    首个压缩边界步读的是 page 1（行 128~255），并不包含最后一行（行 511 在 page 3），
    按此推理这一行不该被读到 —— 推理与实测矛盾，说明对 plan 的页/槽位账目理解仍有
    误。修复的**效果**是三轮实测确认的，**机理**只确认到"去掉 ``-inf`` 即消失"。
    后续若要动这段，请先把 ``compress_forward`` 的读取范围用探针测清楚，不要照着
    上面的页号推理改。

    详见 ``doc/13. EB5 XPU 精度问题总表.md`` A 节。
    """
    out = original_fn(self, *args, **kwargs)
    # 属性缺失按**异常**处理，不按"条件不满足"静默跳过：这是唯一的生产修复，
    # 一旦被跳过服务仍会正常启动，但精度退回原状（AIME 1/8）。所以用
    # ``self.ratio`` 直取而不是 ``getattr(..., None)``——上游 ``__init__`` 必然
    # 先赋值；取不到说明上游结构变了，应当响亮地报出来而不是悄悄不修。
    if self.online or self.ratio != 128:
        return out
    ks = self.kv_score_buffer
    t = ks.kv_score
    if not isinstance(t, torch.Tensor):
        raise TypeError(
            "eb5 c128 state-pool zero-fill: kv_score_buffer.kv_score is "
            f"{type(t).__name__}, expected torch.Tensor"
        )
    t.zero_()
    # 让"修复已生效"在启动日志里可核验：排查时第一个要确认的就是这条。
    global _ZERO_FILL_LOGGED
    if not _ZERO_FILL_LOGGED:
        _ZERO_FILL_LOGGED = True
        logger.info(
            "eb5 c128 state-pool zero-filled (%s rows, ratio=%s) -- decode "
            "pad-collapse guard active",
            t.shape[0],
            self.ratio,
        )
    return out
