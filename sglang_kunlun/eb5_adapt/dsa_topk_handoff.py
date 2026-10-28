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
"""decode 侧 indexer top-k 与 ``transform_index`` 之间的一个标志位。

EB5 的 decode 走两步：

1. ``Indexer._get_topk_paged`` 用 ``metadata.topk_transform`` 选出 top-k
   —— ``SGLANG_DSA_FUSE_TOPK=0`` 下它返回的是**请求内的局部位置**；
2. ``DSAAttnBackend.forward_decode`` 再用 ``transform_index_page_table_decode``
   把位置经 ``page_table_1``（page_size=1 的宽表）gather 成**物理 slot**。

``kunlun_ops.fast_topk_transform_fused`` 一刀把两步做完，而且用
``block_size=page_size`` 直接吃 indexer 自己的紧凑页表，连宽表都不用。接上之后
第 1 步的产物就已经是 slot 了，第 2 步必须让开，否则会把 slot 当下标再 gather 一次。

上游表达"已经是 slot"的方式是 ``DSAAttnBackend.use_fused_topk``，但那个开关同时
改写另外六处分支（``dsa_backend.py`` 的 1019 / 1534 / 2292 / 3894 / 4123 等），
而且 ``SGLANG_DSA_FUSE_TOPK=1`` 在 ``dsa_topk_backend`` 是 torch 时会直接
``raise RuntimeError``（``dsa_topk_backend.py:269``）。所以这里只在插件内部用一个
标志位交接，上游行为一个字节都不动。

标志位是**按次**设置的（每次 ``_get_topk_paged`` 返回前写一次），所以融合路径因为
形状/元数据不匹配回退到 torch 时，第 2 步会自动恢复 gather。
"""

from __future__ import annotations

import os

_STATE = {"produced_slots": False}


def fused_transform_enabled() -> bool:
    """是否启用融合的 decode top-k + page-table transform。

    置 ``EB5_DSA_FUSED_TOPK_TRANSFORM=0`` 退回"torch topk + torch gather"，便于二分。
    """
    return os.environ.get("EB5_DSA_FUSED_TOPK_TRANSFORM", "1") == "1"


def set_produced_slots(value: bool) -> None:
    _STATE["produced_slots"] = bool(value)


def produced_slots() -> bool:
    return _STATE["produced_slots"]
