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
                try:
                    with open(
                        os.path.join(out_dir, f"rank{rank}_pid{os.getpid()}.log"), "a"
                    ) as fh:
                        fh.write(
                            "%.3f rank=%d op=%s seq=%d argc=%d %s %s\n"
                            % (
                                time.time(),
                                rank,
                                op_name,
                                seq,
                                len(args),
                                ",".join(_tensor_desc(a) for a in args[:3]),
                                ",".join(
                                    f"{k}={_tensor_desc(v)}"
                                    for k, v in list(kwargs.items())[:3]
                                ),
                            )
                        )
                except Exception:
                    pass
                return fn(*args, **kwargs)

            wrapper._eb5_traced = True
            return wrapper

        setattr(dist, name, make_wrapper(name, original))

    _INSTALLED = True
    return True
