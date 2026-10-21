# Copyright (c) 2026 Baidu, Inc. All rights reserved.
"""Per-rank collective trace (debug, env-gated).

Why: the multi-rank decode-graph capture hangs with an XCCL watchdog report of
``WorkXCCL(SeqNum=1, OpType=_ALLGATHER)`` on only *some* ranks, which means the
ranks issue different collective sequences during capture.  This module records
every collective a rank issues, in order, so the two traces can be diffed and
the first divergence found.

Usage::

    EB5_COLLECTIVE_TRACE=1 EB5_COLLECTIVE_TRACE_DIR=/path/to/traces \\
        bash tool/run_eb5_kunlun.sh ...

Each scheduler process appends to ``<dir>/rank<rank>_pid<pid>.log``.  It costs
one ``open()`` per collective, so never leave it on in production.
"""

from __future__ import annotations

import os
import time

_INSTALLED = False
_COUNTERS: dict = {}


def _tensor_desc(value) -> str:
    try:
        import torch

        if isinstance(value, (list, tuple)):
            return ",".join(_tensor_desc(v) for v in value)
        if isinstance(value, torch.Tensor):
            return f"{tuple(value.shape)}/{value.dtype}"
    except Exception:
        pass
    return type(value).__name__


def _write(out_dir, rank, op_name, seq, phase, extra) -> None:
    """Append one trace line; never let tracing break the collective."""
    try:
        with open(
            os.path.join(out_dir, f"rank{rank}_pid{os.getpid()}.log"), "a"
        ) as fh:
            fh.write(
                "%.3f rank=%d op=%s seq=%d %s %s\n"
                % (time.time(), rank, op_name, seq, phase, extra)
            )
    except Exception:
        pass


def install() -> bool:
    """Wrap ``torch.distributed`` collectives with a per-rank logger."""
    global _INSTALLED
    if _INSTALLED:
        return True
    out_dir = os.environ.get("EB5_COLLECTIVE_TRACE_DIR", "/tmp/eb5_collective_trace")
    try:
        os.makedirs(out_dir, exist_ok=True)
        import torch.distributed as dist
    except Exception:
        return False

    ops = (
        "all_gather",
        "all_gather_into_tensor",
        "all_reduce",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "_all_gather_base",
        "_reduce_scatter_base",
        "broadcast",
        "barrier",
        "send",
        "recv",
        "batch_isend_irecv",
    )

    for name in ops:
        original = getattr(dist, name, None)
        if original is None or getattr(original, "_eb5_traced", False):
            continue

        def make_wrapper(op_name, fn):
            def wrapper(*args, **kwargs):
                try:
                    rank = dist.get_rank() if dist.is_initialized() else -1
                except Exception:
                    rank = -99
                seq = _COUNTERS.get(op_name, 0) + 1
                _COUNTERS[op_name] = seq
                _write(
                    out_dir,
                    rank,
                    op_name,
                    seq,
                    "enter",
                    "argc=%d %s %s"
                    % (
                        len(args),
                        ",".join(_tensor_desc(a) for a in args[:3]),
                        ",".join(f"{k}={_tensor_desc(v)}" for k, v in list(kwargs.items())[:3]),
                    ),
                )
                try:
                    result = fn(*args, **kwargs)
                except BaseException as exc:
                    _write(out_dir, rank, op_name, seq, "raise", type(exc).__name__)
                    raise
                _write(out_dir, rank, op_name, seq, "exit", "")
                return result

            wrapper._eb5_traced = True
            return wrapper

        setattr(dist, name, make_wrapper(name, original))

    _INSTALLED = True
    return True

def install_copy_trace() -> bool:
    """Log every ``Tensor.copy_`` with its call site (env: EB5_COPY_TRACE=1).

    Multi-rank capture was found (gdb) to spin inside ``xpu_launch_async`` on a
    device-to-device ``memcpy_1d`` issued by ``Tensor.copy_``, i.e. a plain copy
    that the XPU runtime refuses to record.  Which copy it is cannot be seen
    from the C++ stack, so record the Python call site instead.
    """
    out_dir = os.environ.get("EB5_COLLECTIVE_TRACE_DIR", "/tmp/eb5_collective_trace")
    try:
        os.makedirs(out_dir, exist_ok=True)
        import torch
        from torch import Tensor
    except Exception:
        return False
    original = Tensor.copy_
    if getattr(original, "_eb5_traced", False):
        return True

    import traceback

    def copy_wrapper(self, src, non_blocking: bool = False):
        try:
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else -1
        except Exception:
            rank = -99
        seq = _COUNTERS.get("copy_", 0) + 1
        _COUNTERS["copy_"] = seq
        frames = traceback.extract_stack(limit=5)[:-1]
        site = " < ".join(
            "%s:%d" % (os.path.basename(f.filename), f.lineno) for f in frames[-3:]
        )
        _write(
            out_dir,
            rank,
            "copy_",
            seq,
            "enter",
            "dst=%s src=%s site=%s"
            % (tuple(self.shape), tuple(getattr(src, "shape", ())), site),
        )
        result = original(self, src, non_blocking)
        _write(out_dir, rank, "copy_", seq, "exit", "")
        return result

    copy_wrapper._eb5_traced = True
    Tensor.copy_ = copy_wrapper
    return True
