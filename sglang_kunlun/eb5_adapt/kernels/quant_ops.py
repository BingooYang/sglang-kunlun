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
"""叶子量化实现，供 kernels 与 hooks 共用。

放在 ``eb5_adapt.kernels`` 下，避免 ``dsv4_torch_shim`` 为了拿
``act_quant_kunlun`` 而 import 整个 hooks 包、在 sglang 初始化期间重入。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def act_quant_kunlun(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-wise INT8 quantisation backed by ``kunlun_ops.quant2d``.

    Kunlun has no first-class ``float8_e4m3fn``; the indexer / attention
    kernels consume ``int8`` plus a per-block float32 scale.
    """
    import kunlun_ops

    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size (block_size={block_size})"
    )
    assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)

    x_shape = x.shape
    x_blocks = x.view(-1, block_size).contiguous()
    y_int8 = torch.empty(x_blocks.shape, dtype=torch.int8, device=x.device)
    y_max = torch.empty(x_blocks.shape[0], dtype=torch.float32, device=x.device)
    kunlun_ops.quant2d(x_blocks, y_int8, y_max, force_sdnn=True)

    return y_int8.view(x_shape), y_max.view(x_shape[0], -1)
