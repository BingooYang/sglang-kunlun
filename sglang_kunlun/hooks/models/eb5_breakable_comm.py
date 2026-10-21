# Copyright (c) 2026 Baidu, Inc. All rights reserved.
"""Run EB5's per-layer DP collectives *eagerly* between captured graph segments.

Why
---
EB5's decoder layer calls ``dp_gather_partial`` / ``dp_scatter`` inside every
layer (``ernie5.py`` ~3767/3776).  With the ``full`` decode-graph backend those
XCCL collectives end up *inside* the captured graph, and on Kunlun XPU the
capture hangs: all four ranks enter ``Capture target decode CUDA graph begin``,
every collective returns on the host, and then the device wedges
(``_ALLGATHER`` watchdog timeout; see ``doc/7`` §11).

``BreakableCudaGraphBackend`` already ships the mechanism we need:
``eager_on_graph`` ends the current graph segment, runs the decorated callable
eagerly, then starts a new segment — and does the same again on every replay.
Wrapping the two DP collectives with it therefore keeps the communication out
of the captured graphs while everything between them stays captured.

Enable with ``EB5_BREAKABLE_EAGER_COMM=1`` **and**
``--cuda-graph-backend-decode breakable``.  Default off: plain ``full`` graphs
(single rank) are untouched.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def install() -> bool:
    """Wrap ``dp_attention.dp_gather_partial`` / ``dp_scatter`` if enabled."""
    if os.environ.get("EB5_BREAKABLE_EAGER_COMM") != "1":
        return False
    try:
        from sglang.srt.layers import dp_attention
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
            eager_on_graph,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("eb5_breakable_eager_comm: import failed: %s", exc)
        return False

    patched = []
    for name in ("dp_gather_partial", "dp_gather_replicate", "dp_scatter"):
        original = getattr(dp_attention, name, None)
        if original is None or getattr(original, "_eb5_eager_comm", False):
            continue
        inner_wrapper = eager_on_graph(True)(original)

        def wrapper(*args, __inner=inner_wrapper, __name=name, **kwargs):
            if os.environ.get("EB5_EAGER_COMM_DEBUG") == "1":
                try:
                    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
                        _current_capture_var,
                    )

                    if _current_capture_var.get() is not None:
                        print(f"[eager_comm] break at {__name}", flush=True)
                except Exception:
                    pass
            return __inner(*args, **kwargs)

        wrapper._eb5_eager_comm = True
        wrapper._eb5_inner = inner_wrapper
        setattr(dp_attention, name, wrapper)
        patched.append(name)

    # ``ernie5`` uses ``from ... import dp_gather_partial``; if it was imported
    # before this hook ran, re-point its module-level aliases too.
    import sys

    # compat_aliases can leave duplicate module objects: rebind every module
    # that still holds the pre-patch callable.
    for mod in list(sys.modules.values()):
        ns = getattr(mod, "__dict__", None)
        if not ns:
            continue
        for name in patched:
            original = ns.get(name)
            if original is not None and getattr(original, "_eb5_eager_comm", False):
                continue
            if original is not None and getattr(original, "__module__", "").endswith("dp_attention"):
                setattr(mod, name, getattr(dp_attention, name))

    for mod_name in (
        "sglang.srt.internal.models.eb5.ernie5",
        "sglang.srt.internal.models.eb5.ernie5_dspark",
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for name in patched:
            if hasattr(mod, name):
                setattr(mod, name, getattr(dp_attention, name))

    logger.warning("eb5_breakable_eager_comm: wrapped %s", patched)
    return bool(patched)

def install_capture_barrier() -> bool:
    """Barrier all ranks right before/after decode-graph capture.

    Watchdog evidence (doc/7 §11): ranks 1/2/3 time out on
    ``_ALLGATHER_BASE(NumelIn=4096, NumelOut=16384)`` while rank 0 never reaches
    it, i.e. the ranks enter/leave capture at different points.  A host-side
    barrier around the capture call is the cheapest way to test that hypothesis.
    """
    if os.environ.get("EB5_GRAPH_CAPTURE_BARRIER") != "1":
        return False
    try:
        from sglang.srt.model_executor.runner import (
            decode_cuda_graph_runner as _runner,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("eb5_capture_barrier: import failed: %s", exc)
        return False

    cls = _runner.DecodeCudaGraphRunner
    original = cls.capture
    if getattr(original, "_eb5_barrier", False):
        return True

    import torch.distributed as dist

    def capture(self, *args, **kwargs):
        def _barrier(tag):
            try:
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
            except Exception as exc:  # noqa: BLE001
                print(f"[capture_barrier] {tag} failed: {exc}", flush=True)

        _barrier("pre")
        try:
            return original(self, *args, **kwargs)
        finally:
            _barrier("post")

    capture._eb5_barrier = True
    cls.capture = capture
    logger.warning("eb5_capture_barrier: installed around DecodeCudaGraphRunner.capture")
    return True

def install_capture_layer_limit() -> bool:
    """Diagnostic only: capture a graph with only the first N decoder layers.

    ``EB5_GRAPH_CAPTURE_LAYERS=K`` truncates the layer loop *while the stream is
    capturing*, so the resulting graph is incomplete (never replay it) — the
    point is to bisect which layer first makes the multi-rank capture hang.
    """
    raw = os.environ.get("EB5_GRAPH_CAPTURE_LAYERS")
    if not raw:
        return False
    layers = int(raw)
    try:
        import torch
        from sglang.srt.internal.models.eb5 import ernie5
    except Exception as exc:  # noqa: BLE001
        logger.warning("eb5_capture_layer_limit: import failed: %s", exc)
        return False

    cls = ernie5.EB5V2Model
    original = cls.forward
    if getattr(original, "_eb5_layer_limit", False):
        return True

    def forward(self, *args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            saved = self.end_layer
            self.end_layer = min(saved, self.start_layer + layers)
            try:
                return original(self, *args, **kwargs)
            finally:
                self.end_layer = saved
        return original(self, *args, **kwargs)

    forward._eb5_layer_limit = True
    cls.forward = forward
    logger.warning("eb5_capture_layer_limit: capture limited to %d layer(s)", layers)
    return True
