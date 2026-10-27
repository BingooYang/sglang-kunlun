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


def _retarget_eb5_checkpoint(hf_config):
    """Rebuild a legacy EB5 config as a genuine ``Ernie5_5_MoeConfig``.

    ``hf_checkpoint-48800`` / ``eblite-int8-moe`` and friends declare
    ``architectures=["ErnieForCausalLM"]`` / ``model_type="ernie"``, which the
    internal fork resolves through its own runtime patch. The open-source tree
    registers the model as ``Ernie5_5_MoeForCausalLM`` / ``ernie5_5_moe``, and
    the model class type-checks its config, so renaming the two fields on the
    existing object is not enough -- it stays an ``ErnieConfig`` and the model
    raises "serves ERNIE 5.5 Flash checkpoints only, got a ErnieConfig".
    Construct the real class from the same field dict instead, so no patched
    copy of the checkpoint is needed.

    Returns the replacement config, or ``None`` when nothing had to change.
    """
    from sglang.srt.configs.ernie55 import Ernie5_5_MoeConfig

    archs = list(getattr(hf_config, "architectures", None) or [])
    if not archs or archs[0] not in _LEGACY_ARCHS:
        return None
    if isinstance(hf_config, Ernie5_5_MoeConfig):
        return None
    fields = hf_config.to_dict()
    fields["architectures"] = [_ERNIE55_CKPT_ARCH]
    fields["model_type"] = _ERNIE55_CKPT_MODEL_TYPE
    # ``__init__`` derives both ratio lists from its ``compress_ratios``
    # argument, where -2 marks an MLA layer. ``compress_ratios`` on an already
    # built config may be the normalised copy (-2 -> 0), which would erase
    # every MLA layer, so prefer the verbatim list when it is present.
    ratios = fields.get("checkpoint_compress_ratios") or fields.get("compress_ratios")
    if ratios:
        fields["compress_ratios"] = ratios
    rebuilt = Ernie5_5_MoeConfig(**fields)
    logger.warning(
        "sglang-kunlun: retargeted EB5 checkpoint %s -> %s (model_type=%s, "
        "config %s -> Ernie5_5_MoeConfig)",
        archs[0],
        _ERNIE55_CKPT_ARCH,
        _ERNIE55_CKPT_MODEL_TYPE,
        type(hf_config).__name__,
    )
    return rebuilt


@plugin_hook(
    target="sglang.srt.configs.model_config.get_config",
    type=HookType.AROUND,
) if _ernie55_present() else (lambda fn: fn)
def get_config_kunlun(original_fn, *args, **kwargs):
    """Retarget the EB5 checkpoint before ``ModelConfig`` reads the config.

    ``ModelConfig.__init__`` derives the attention backend, the hybrid-SWA
    flags and the DSV4 pool layout from ``architectures``, so an AFTER hook on
    ``__init__`` is too late: those branches would already have been skipped.
    ``get_config`` applies ``--json-model-override-args`` last and returns, so
    this is the first point where the config is complete.
    """
    config = original_fn(*args, **kwargs)
    try:
        rebuilt = _retarget_eb5_checkpoint(config)
    except Exception:  # noqa: BLE001 - a non-EB5 config must still start up
        logger.exception("sglang-kunlun: EB5 config retarget failed; leaving as is")
        return config
    return config if rebuilt is None else rebuilt
