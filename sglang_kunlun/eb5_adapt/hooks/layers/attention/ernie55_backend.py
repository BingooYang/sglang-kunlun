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
"""Route ERNIE 5.5 Flash's HCA layers through the Kunlun fused attention op.

Upstream pins EB5 to ``attention_backend="ernie55"``
(``srt/arg_groups/model_overrides/ernie55.py``), which resolves to
:class:`sglang.srt.layers.attention.ernie55_backend.Ernie5_5AttnBackend`.
That class inherits ``DeepseekV4AttnBackend.forward`` unchanged for the HCA
layers (``compress_ratio == 128``), and on Kunlun that forward ends up in the
*pure-torch* ``flash_mla_with_kvcache`` that ``eb5_mla_adapt._install_flash_mla``
installs into the ``flash_mla`` wheel stub -- the fused
``xspeedgate_ops.compressed_attention`` that DSV4 runs in production is never
called, because the Kunlun override of that forward lives only on
:class:`KunlunDeepseekV4AttnBackend`, registered under the names ``kunlun`` /
``kunlun_dsv4`` that EB5 never selects.

The two class hierarchies meet cleanly: ``Ernie5_5HcaAttention`` is the same
``MQALayer`` DSV4 uses, and ``Ernie5_5TokenToKVPool`` subclasses
``DeepSeekV4TokenToKVPool`` (which is what the Kunlun forward asserts).  So a
plain co-operative subclass is enough -- the MRO

    KunlunErnie5_5AttnBackend
      -> Ernie5_5AttnBackend          (ERNIE's MLA sub-backend + index_topk)
        -> KunlunDeepseekV4AttnBackend (fused forward)
          -> DeepseekV4AttnBackend     (metadata, cache, graph state)

keeps ERNIE's ``__init__`` / metadata plumbing and only swaps in the fused
``forward``.  The MLA layers are untouched: they run their own
``Ernie5_5MlaAttnBackend`` (DSA) that ``Ernie5_5AttnBackend.__init__`` builds.

Measured working on 8xP800 (eager, MTP/EAGLE on): decode 4.3860 -> 4.9836
tok/s, ``aime26_first2`` 2/2, so it is **on by default**;
``EB5_HCA_FUSED_ATTENTION=0`` falls back to the untouched upstream backend for
bisecting.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

ENV_FLAG = "EB5_HCA_FUSED_ATTENTION"


def fused_hca_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "1") == "1"


def _build_class():
    from sglang.srt.layers.attention.ernie55_backend import Ernie5_5AttnBackend

    from sglang_kunlun.eb5_adapt.hooks.layers.attention.deepseek_v4_backend import (
        KunlunDeepseekV4AttnBackend,
    )

    class KunlunErnie5_5AttnBackend(Ernie5_5AttnBackend, KunlunDeepseekV4AttnBackend):
        """ERNIE 5.5 Flash with the Kunlun fused HCA attention."""

        # One line per (mode, compress_ratio) pair, so the first decode proves
        # the fused op really took the HCA calls instead of the torch fallback.
        _logged_routes: set = set()

        def forward(self, *args, **kwargs):
            if logger.isEnabledFor(logging.INFO):
                fb = kwargs.get("forward_batch")
                mode = getattr(getattr(fb, "forward_mode", None), "name", None)
                key = (mode, kwargs.get("compress_ratio"))
                if key not in self._logged_routes:
                    self._logged_routes.add(key)
                    q = kwargs.get("q")
                    logger.info(
                        "[eb5-hca-fused] mode=%s compress_ratio=%s q=%s -> "
                        "KunlunDeepseekV4AttnBackend.forward",
                        mode,
                        key[1],
                        tuple(q.shape) if q is not None else None,
                    )
            return KunlunDeepseekV4AttnBackend.forward(self, *args, **kwargs)

    return KunlunErnie5_5AttnBackend


def install() -> list:
    """Take over the ``ernie55`` attention-backend name.

    Always registers, so the flag can be flipped per launch; with the flag off
    the factory hands back the untouched upstream backend.
    """
    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )

    @register_attention_backend("ernie55")
    def create_ernie55_backend(runner):
        from sglang.srt.layers.attention.ernie55_backend import Ernie5_5AttnBackend

        if not fused_hca_enabled():
            logger.info(
                "Using upstream Ernie5_5AttnBackend (%s != 1); HCA attention stays "
                "on the torch flash_mla path.",
                ENV_FLAG,
            )
            return Ernie5_5AttnBackend(runner)

        cls = _build_class()
        logger.info(
            "Using %s for ernie55 attention backend (Kunlun fused HCA attention).",
            cls.__name__,
        )
        return cls(runner)

    return ["sglang.srt.layers.attention.attention_registry:ATTENTION_BACKENDS[ernie55]"]
