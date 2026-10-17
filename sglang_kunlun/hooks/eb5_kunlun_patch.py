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
"""EB5 / EB-Lite 在昆仑上的服务侧适配入口。

**为什么需要它**：`code/eb5_layer/` 里那三个适配模块（`eb5_kunlun_adapt.py` /
`eb5_mla_adapt.py` / `eb5_extra_adapt.py`）平时只由 pytest 的 `conftest.py` 加载，
服务进程不会碰。要让 EB5 在昆仑上起服务，得把它们挂进 sglang 的插件机制。

**怎么挂**：fork 的 `python/pyproject.toml` 已经注册了

    [project.entry-points."sglang.srt.plugins"]
    eb5_internal_runtime_patch = "sglang.srt.internal.models.eb5.runtime_patch:register_eb5_runtime_patch"

本模块在插件里再注册一个同名组的 entry point（见 `pyproject.toml`）：

    eb5_kunlun_patch = "sglang_kunlun.hooks.eb5_kunlun_patch:register_eb5_kunlun_patch"

启动时用 ``SGLANG_PLUGINS`` 白名单把它一起打开（**注意它是白名单，设了就只加载列出的**）：

    export SGLANG_PLUGINS=kunlun_hooks,eb5_internal_runtime_patch,eb5_kunlun_patch

适配模块所在目录用 ``EB5_KUNLUN_ADAPT_DIR`` 覆盖，默认
``/root/paddlejob/bingoo/code/eb5_layer``。
"""

from __future__ import annotations

import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)

DEFAULT_ADAPT_DIR = "/root/paddlejob/bingoo/code/eb5_layer"


def _load_fork_plugins() -> None:
    """加载 fork 自带的插件。

    fork（`baidu/wenxin/sglang`）的 `python/pyproject.toml` 注册了三个
    ``sglang.srt.plugins`` entry point（``eb5_internal_runtime_patch`` /
    ``internal_hooks`` / ``spec_adaptive``），但**本环境里 fork 只挂在 PYTHONPATH 上、
    没有 pip 安装**，`importlib.metadata.entry_points()` 发现不到它们
    （实测 `Available plugins` 只有插件的 `kunlun_hooks`）。

    EB5 依赖其中两个：`internal_hooks`（含 DSV4/EB5 的 `compute_kv_score_bf16`
    等）与 `eb5_internal_runtime_patch`（注册 `eb5_mla_hca` attention backend、
    把通用 arch 改写成 `ErnieMlaHcaForCausalLM`、补 server defaults）。
    所以在这里显式调用，替代 entry-point 发现。
    """
    for label, mod_path, fn_name in (
        (
            "internal_hooks",
            "sglang.srt.internal.plugins",
            "register_internal_plugins",
        ),
        (
            "eb5_internal_runtime_patch",
            "sglang.srt.internal.models.eb5.runtime_patch",
            "register_eb5_runtime_patch",
        ),
    ):
        try:
            mod = importlib.import_module(mod_path)
            getattr(mod, fn_name)()
            logger.warning("eb5_kunlun_patch: loaded fork plugin %s", label)
        except Exception as exc:  # noqa: BLE001 - 单个插件失败不阻断
            logger.warning("eb5_kunlun_patch: fork plugin %s failed: %s", label, exc)


def register_eb5_kunlun_patch() -> None:
    """Entry point used by SGLang's plugin loader.

    两步：
    1. 加载 fork 自带、但 entry-point 机制发现不到的插件（见 `_load_fork_plugins`）；
    2. 把 EB5 的昆仑适配层（融合算子 → torch 拆解 / KLX 原生算子）应用到当前进程。

    幂等；任何失败只告警，不阻断服务启动。
    """
    if os.environ.get("EB5_KUNLUN_PATCH", "1").lower() in {"0", "false", "no", "off"}:
        logger.info("eb5_kunlun_patch: disabled by EB5_KUNLUN_PATCH")
        return

    _load_fork_plugins()

    adapt_dir = os.environ.get("EB5_KUNLUN_ADAPT_DIR", DEFAULT_ADAPT_DIR)
    if not os.path.isdir(adapt_dir):
        logger.warning(
            "eb5_kunlun_patch: adapt dir %s not found; nothing applied", adapt_dir
        )
        return
    if adapt_dir not in sys.path:
        sys.path.insert(0, adapt_dir)

    try:
        mod = importlib.import_module("eb5_kunlun_adapt")
    except Exception as exc:  # noqa: BLE001 - 插件加载不能把服务带崩
        logger.warning("eb5_kunlun_patch: cannot import eb5_kunlun_adapt: %s", exc)
        return

    # `eb5_kunlun_adapt.apply()` 内部会依次调用 eb5_mla_adapt / eb5_extra_adapt。
    try:
        patched = mod.apply()
    except Exception as exc:  # noqa: BLE001
        logger.warning("eb5_kunlun_patch: apply() failed: %s", exc, exc_info=True)
        return

    logger.warning(
        "eb5_kunlun_patch: applied %d EB5 Kunlun patches from %s", len(patched), adapt_dir
    )
