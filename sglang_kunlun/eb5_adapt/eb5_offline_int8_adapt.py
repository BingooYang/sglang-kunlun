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
"""离线 INT8 checkpoint 的 EB5 适配。

在线 INT8（:mod:`eb5_int8_adapt`）在加载 BF16 权重后自己做 per-output-channel
量化；离线 checkpoint 则把量化结果和 ``weight_scale`` 一起存盘：

* 只有 routed experts 被量化：``...block_sparse_moe.experts.<e>.w{1,2,3}``
  是 ``int8[out, in]``，旁边是 ``.weight_scale`` ``fp32[out, 1]``；
* scale 的约定是 ``amax / 127``（``config.json`` 的
  ``scale_is_amax_over_127``），昆仑 GEMM 要的是 raw absmax，所以
  ``sglang_kunlun.hooks.layers.quantization.w8a8_int8`` 里已有的
  ``process_weights_after_loading`` 钩子会统一的 ``* 127``；
* 其余 dense 权重仍是 BF16，由同一个钩子的 config 分支跳过。

这里只补最后一块：fork 的 ``ernie5.py`` 权重加载器只认识 block-FP8 的
``.weight_scale_inv``，遇到 per-channel INT8 的 ``.weight_scale`` 会在
``expert_shards[...]`` 上抛 ``KeyError``。给 ``_split_scale_suffix`` /
``_as_scale_name`` 加上 channel-scale 分支即可，路由仍然复用原来的
``load_routed_expert``（scale 的 ``quant_method`` 已由
``W8A8Int8MoEMethod.create_weights`` 标成 ``CHANNEL``）。

不修改 fork 源码：两个 helper 都是模块级函数，``load_weights`` 里的闭包在调用时
才从模块 globals 解析，所以 import 后替换即可。
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

# ``_split_scale_suffix`` 和 ``_as_scale_name`` 是两次独立调用，用 thread-local
# 记住上一次看到的是哪种 scale，避免把 channel scale 误写成 ``_scale_inv``。
_STATE = threading.local()


def _patch_ernie5_scale_suffix() -> list[str]:
    import sglang.srt.internal.models.eb5.ernie5 as ernie5

    if getattr(ernie5, "_kunlun_channel_scale_patch", False):
        return []

    orig_split = ernie5._split_scale_suffix
    orig_as_scale_name = ernie5._as_scale_name
    orig_transpose_key = ernie5._transpose_base_key

    def split_scale_suffix(suffix: str) -> tuple[str, bool]:
        # ``foo.weight_scale`` -> ``(foo.weight, True)`` (per-channel INT8).
        # Keep ``_scale_inv`` on the original block-FP8 path.
        if suffix.endswith(".weight_scale") and not suffix.endswith(
            ".weight_scale_inv"
        ):
            _STATE.scale_kind = "channel"
            return suffix[: -len("_scale")], True
        _STATE.scale_kind = (
            "block" if suffix.endswith(".weight_scale_inv") else None
        )
        return orig_split(suffix)

    def as_scale_name(weight_name: str) -> str:
        if getattr(_STATE, "scale_kind", None) == "channel" and weight_name.endswith(
            "weight"
        ):
            return weight_name + "_scale"
        return orig_as_scale_name(weight_name)

    def transpose_base_key(target_name: str) -> str:
        if target_name.endswith(".weight_scale") and not target_name.endswith(
            ".weight_scale_inv"
        ):
            return target_name[: -len(".weight_scale")]
        return orig_transpose_key(target_name)

    ernie5._split_scale_suffix = split_scale_suffix
    ernie5._as_scale_name = as_scale_name
    ernie5._transpose_base_key = transpose_base_key
    ernie5._kunlun_channel_scale_patch = True

    logger.info(
        "eb5_offline_int8: ernie5 loader now routes .weight_scale to the "
        "per-channel INT8 scale params (w13_weight_scale / w2_weight_scale)."
    )
    return [
        "sglang.srt.internal.models.eb5.ernie5._split_scale_suffix",
        "sglang.srt.internal.models.eb5.ernie5._as_scale_name",
        "sglang.srt.internal.models.eb5.ernie5._transpose_base_key",
    ]


def apply() -> list[str]:
    """适配层入口：让 EB5 loader 认识离线 per-channel INT8 的 ``weight_scale``。"""
    try:
        return _patch_ernie5_scale_suffix()
    except Exception as exc:  # noqa: BLE001 - 适配层失败不能把服务带崩
        logger.warning("eb5_offline_int8: patch skipped: %s", exc)
        return []
