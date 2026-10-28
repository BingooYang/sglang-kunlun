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
"""logits softcap 的纯 torch 实现（昆仑 P800 无 Triton）。

这几个符号只在 **CUDA Graph 口径**下才会被调到：开图时
``LogitsProcessor._copy_logits_to_buffer``（``logits_processor.py:1262-1294``）必须把
logits 写进一块**地址固定**的 fp32 buffer，好让 graph 的输出位置在 replay 之间不变；
eager 口径下那条分支走的是 ``logits.float()``，碰不到这些 kernel。所以前 9 轮关图
一直没暴露，一开图就在 capture 阶段炸：

    cast_to_float32_logits -> _softcap_logits -> softcap_inplace_logits_kernel[grid](...)
    TypeError: 'function' object is not subscriptable

（Triton 被 stub 成普通函数，``kernel[grid]`` 的下标语法自然不成立。）

替换的符号：

* ``sglang.kernels.ops.activation.softcap``：``softcap_inplace_logits`` /
  ``softcap_to_float32_logits`` / ``cast_to_float32_logits`` / ``gathered_softcap_logits``；
* ``sglang.kernels.ops.sampling.softcap_row_argmax``：``softcap_row_argmax``。

语义与原 kernel 逐元素一致（softcap = ``tanh(x / cap) * cap``，输出 fp32）。
**不要**在这里引入 D2H（``.item()`` / ``bool(tensor.any())``）或依赖数据的分支：
这些函数会在 graph capture 里执行，形状之外的任何动态性都会让 replay 结果失真。
"""

from __future__ import annotations

import importlib
from typing import Optional

import torch


def _import_sibling(name: str):
    """导入同目录模块，兼容包内（相对）与 ``EB5_KUNLUN_ADAPT_DIR`` 平铺两种方式。"""

    if __package__:
        try:
            return importlib.import_module(f".{name}", __package__)
        except ModuleNotFoundError as exc:
            if exc.name not in (f"{__package__}.{name}", name):
                raise
    return importlib.import_module(name)


def _softcap_into(src: torch.Tensor, cap: Optional[float], dst: torch.Tensor) -> None:
    """``dst = tanh(src / cap) * cap``（``cap`` 为 None 时只做 dtype 转换）。"""

    if cap is None:
        dst.copy_(src)
        return
    # 先转到 dst 的 dtype 再做 tanh：原 kernel 是先 CAST_TO_FP32 再 softcap。
    dst.copy_(src)
    dst.div_(cap)
    torch.tanh(dst, out=dst)
    dst.mul_(cap)


def _softcap_logits(
    full_logits: torch.Tensor,
    final_logit_softcapping: Optional[float],
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """原 ``_softcap_logits`` 的 torch 复刻。

    ``output is None`` 时就地改 ``full_logits``；否则写到 ``output``（必须 fp32、同形状）。
    ``final_logit_softcapping is None`` 对应原 kernel 的 ``APPLY_SOFTCAP=False``：
    只搬数、不做 tanh（``output is None`` 时整体退化为 no-op）。
    """

    if output is not None:
        assert output.shape == full_logits.shape and output.dtype == torch.float32
        _softcap_into(full_logits, final_logit_softcapping, output)
        return output

    if final_logit_softcapping is None:
        return full_logits
    full_logits.div_(final_logit_softcapping)
    torch.tanh(full_logits, out=full_logits)
    full_logits.mul_(final_logit_softcapping)
    return full_logits


def softcap_inplace_logits(
    full_logits: torch.Tensor, final_logit_softcapping: Optional[float]
) -> torch.Tensor:
    return _softcap_logits(full_logits, final_logit_softcapping)


def softcap_to_float32_logits(
    logits: torch.Tensor, final_logit_softcapping: Optional[float], output: torch.Tensor
) -> torch.Tensor:
    """把 softcap 后的 fp32 logits 写进同形状、按行连续的 ``output``。"""

    return _softcap_logits(logits, final_logit_softcapping, output)


def cast_to_float32_logits(logits: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """只做 fp32 转换，写进同形状、按行连续的 ``output``。"""

    return _softcap_logits(logits, None, output)


def gathered_softcap_logits(
    gathered: torch.Tensor, final_logit_softcapping: float, output: torch.Tensor
) -> torch.Tensor:
    """rank-major 的 ``gathered[world, tokens, shard]`` → vocab-major ``output[tokens, :vocab]``。

    原 kernel 按 vocab 列号反查 shard：``shard = col // shard_width``。这里按 shard
    逐段 ``copy_``（world 只有 8 段，循环长度是静态的，capture 安全），再整块 softcap。
    """

    world, num_tokens, shard_width = gathered.shape
    vocab_size = output.shape[1]
    assert gathered.is_contiguous() and output.stride(1) == 1
    assert output.shape[0] == num_tokens and vocab_size <= world * shard_width

    for shard in range(world):
        lo = shard * shard_width
        if lo >= vocab_size:
            break
        hi = min(lo + shard_width, vocab_size)
        output[:, lo:hi].copy_(gathered[shard, :, : hi - lo])

    view = output[:, :vocab_size]
    view.div_(final_logit_softcapping)
    torch.tanh(view, out=view)
    view.mul_(final_logit_softcapping)
    return output


def softcap_row_argmax(
    logits: torch.Tensor,
    final_logit_softcapping: float,
    valid_width: int,
    col_offset: int,
) -> torch.Tensor:
    """每行取 softcap 后前 ``valid_width`` 列的 ``(max, 首个 argmax + col_offset)``。

    返回 fp32 ``[rows, 2]``。softcap 是单调递增的，所以 argmax 在 softcap 前后一致，
    但 max **值**必须是 softcap 之后的；原 kernel 的 tie-break 取最左，
    ``torch.max`` 在 dim 上也返回第一个最大值下标，语义相同。
    """

    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    view = logits[:, :valid_width]
    best_val, best_idx = torch.max(view, dim=1)
    best_val = torch.tanh(best_val / final_logit_softcapping) * (
        final_logit_softcapping
    )
    out = torch.empty(logits.shape[0], 2, dtype=torch.float32, device=logits.device)
    out[:, 0] = best_val
    out[:, 1] = best_idx.to(torch.float32) + float(col_offset)
    return out


_TARGETS = (
    (
        "sglang.kernels.ops.activation.softcap",
        "softcap_inplace_logits",
        softcap_inplace_logits,
    ),
    (
        "sglang.kernels.ops.activation.softcap",
        "softcap_to_float32_logits",
        softcap_to_float32_logits,
    ),
    (
        "sglang.kernels.ops.activation.softcap",
        "cast_to_float32_logits",
        cast_to_float32_logits,
    ),
    (
        "sglang.kernels.ops.activation.softcap",
        "gathered_softcap_logits",
        gathered_softcap_logits,
    ),
    (
        "sglang.kernels.ops.activation.softcap",
        "_softcap_logits",
        _softcap_logits,
    ),
    (
        "sglang.kernels.ops.sampling.softcap_row_argmax",
        "softcap_row_argmax",
        softcap_row_argmax,
    ),
)


def apply() -> list:
    """把 torch 实现替换到调用点（含 import 期绑死的别名，如 ``fused_softcap``）。"""

    mla = _import_sibling("eb5_mla_adapt")
    patched = []
    for module_name, attr, impl in _TARGETS:
        patched.extend(mla.patch_everywhere(module_name, attr, impl))
    return patched
