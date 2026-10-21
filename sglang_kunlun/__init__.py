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
"""sglang-kunlun: OOT SGLang platform plugin for Kunlun (XPU) chips."""

# Bind ``_kunlun_pre_shim`` FIRST.  ``compat_aliases.install()`` below imports
# sglang submodules, and sglang's ``_resolve_platform()`` may then import
# ``sglang_kunlun.platform`` *while this module is still initialising*; that
# module does ``from sglang_kunlun import _kunlun_pre_shim``, so the name has to
# exist by then or the platform plugin fails to activate with a circular-import
# error.
from sglang_kunlun.bootstrap import _kunlun_pre_shim

# Install module-path aliases before any hook module is imported, so that
# imports written against the sglang v0.5.x layout keep resolving on
# newer sglang releases that relocated the same modules.
from sglang_kunlun.bootstrap import compat_aliases as _compat_aliases

_compat_aliases.install()

# Newer sglang imports tilelang at module level from its kernel modules;
# tilelang cannot run on Kunlun and those kernels are never selected, so
# a permissive stub is enough to let the import succeed.
from sglang_kunlun.bootstrap import optional_kernel_stubs as _optional_stubs

_optional_stubs.install_all()

__all__ = ["_kunlun_pre_shim"]
# Debug-only, env-gated: per-rank collective trace for capture-hang triage.
import os as _os

if _os.environ.get("EB5_COLLECTIVE_TRACE") == "1":
    from sglang_kunlun.debug import collective_trace as _collective_trace

    _collective_trace.install()

if _os.environ.get("EB5_COPY_TRACE") == "1":
    from sglang_kunlun.debug import collective_trace as _collective_trace_copy

    _collective_trace_copy.install_copy_trace()
