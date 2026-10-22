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
"""Attention hooks for Kunlun XPU.

Side effects on import:
1. Triton driver pre-shim (backend="cuda") for fla platform detection.
2. Imports nsa, fla submodules for hook side effects.
"""

from __future__ import annotations

import logging

from sglang_kunlun.eb5_adapt.bootstrap import _kunlun_pre_shim

logger = logging.getLogger(__name__)


# 把 Triton active driver 的 get_current_target().backend 伪装成 "cuda"，避免crash
_kunlun_pre_shim()

# Submodules with their own hook / setattr-time side effects.
from . import attention_registry  # noqa: E402,F401
from . import nsa  # noqa: E402,F401
from . import fla  # noqa: E402,F401
from sglang_kunlun.eb5_adapt.hooks.layers.attention import deepseek_v4_backend  # noqa: E402,F401