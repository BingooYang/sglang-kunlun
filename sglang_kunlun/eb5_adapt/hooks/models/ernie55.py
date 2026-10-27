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
"""Hooks for ``sglang.srt.models.ernie55`` (the open-source EB5 model).

Only exists in trees that ship ERNIE 5.5 Flash natively; the internal fork
serves EB5 from ``sglang.srt.internal.models.eb5`` instead, so the importer in
``hooks/models/__init__.py`` skips this module there.
"""

from __future__ import annotations

import logging

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

logger = logging.getLogger(__name__)

# Scale params the Kunlun online-INT8 quantizer registers as empty placeholders
# and fills in ``process_weights_after_loading``. No checkpoint tensor backs
# them, so ``Ernie5_5WeightLoader.finish`` counts them as never loaded.
_ONLINE_QUANT_SCALE_LEAVES = (
    "weight_scale",
    "weight_scale_inv",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
)


def _companion_weight(leaf: str) -> str:
    """``w13_weight_scale`` -> ``w13_weight``; ``weight_scale_inv`` -> ``weight``."""
    for suffix in ("_scale_inv", "_scale"):
        if leaf.endswith(suffix):
            return leaf[: -len(suffix)]
    return leaf


def _online_quant_placeholders(loader) -> list:
    """Unloaded scale params whose companion weight *did* load.

    Requiring the companion keeps a genuinely missing weight fatal: only the
    derived scale of an already-loaded weight is excused.
    """
    excused = []
    for name in set(loader.params) - loader.loaded:
        head, _, leaf = name.rpartition(".")
        if leaf not in _ONLINE_QUANT_SCALE_LEAVES:
            continue
        if f"{head}.{_companion_weight(leaf)}" in loader.loaded:
            excused.append(name)
    return excused


def _ernie55_present() -> bool:
    """Whether this sglang tree ships ``sglang.srt.models.ernie55``.

    Registering a hook on a target the tree does not have makes
    ``apply_hooks`` fail with ``AttributeError: module 'sglang.srt.models' has
    no attribute 'ernie55'`` and takes the whole start-up down -- measured
    2026-10-27, it broke the internal-fork baseline.
    """
    import importlib.util

    try:
        return importlib.util.find_spec("sglang.srt.models.ernie55") is not None
    except (ImportError, ValueError):
        return False


@plugin_hook(
    target="sglang.srt.models.ernie55.Ernie5_5WeightLoader.finish",
    type=HookType.AROUND,
) if _ernie55_present() else (lambda fn: fn)
def finish_kunlun(original_fn, self, *args, **kwargs):
    """Excuse the online-INT8 scale placeholders from the leftover check."""
    excused = _online_quant_placeholders(self)
    if excused:
        self.loaded.update(excused)
        logger.warning(
            "sglang-kunlun: ernie55 loader: excused %d online-quant scale params "
            "(first: %s)",
            len(excused),
            sorted(excused)[:3],
        )
    return original_fn(self, *args, **kwargs)


_ERNIE55_CKPT_ARCH = "Ernie5_5_MoeForCausalLM"
_ERNIE55_CKPT_MODEL_TYPE = "ernie5_5_moe"
# What the published EB5 checkpoints carry; the open-source model class only
# answers to the names above.
_LEGACY_ARCHS = ("ErnieForCausalLM", "ErnieMlaHcaForCausalLM")


def _retarget_eb5_checkpoint(hf_config) -> bool:
    """Point a legacy EB5 checkpoint at ``Ernie5_5_MoeForCausalLM``.

    ``hf_checkpoint-48800`` and friends declare ``architectures=
    ["ErnieForCausalLM"]`` / ``model_type="ernie"``, which the internal fork
    resolves through its own runtime patch. The open-source tree registers the
    model as ``Ernie5_5_MoeForCausalLM`` / ``ernie5_5_moe`` and rejects anything
    else ("serves ERNIE 5.5 Flash checkpoints only, got a ErnieConfig"), so
    rewrite the two fields here instead of shipping a patched copy of the
    checkpoint. Every other field in those configs is already what
    ``Ernie5_5_MoeConfig`` expects.
    """
    archs = list(getattr(hf_config, "architectures", None) or [])
    if not archs or archs[0] not in _LEGACY_ARCHS:
        return False
    hf_config.architectures = [_ERNIE55_CKPT_ARCH]
    if getattr(hf_config, "model_type", None) != _ERNIE55_CKPT_MODEL_TYPE:
        hf_config.model_type = _ERNIE55_CKPT_MODEL_TYPE
    logger.warning(
        "sglang-kunlun: retargeted EB5 checkpoint %s -> %s (model_type=%s)",
        archs[0],
        _ERNIE55_CKPT_ARCH,
        _ERNIE55_CKPT_MODEL_TYPE,
    )
    return True


@plugin_hook(
    target="sglang.srt.configs.model_config.ModelConfig.__init__",
    type=HookType.AFTER,
) if _ernie55_present() else (lambda fn: fn)
def model_config_init_kunlun(result, self, *args, **kwargs):
    """Rewrite the EB5 arch/model_type as soon as the config is built."""
    hf_config = getattr(self, "hf_config", None)
    if hf_config is not None:
        _retarget_eb5_checkpoint(hf_config)
