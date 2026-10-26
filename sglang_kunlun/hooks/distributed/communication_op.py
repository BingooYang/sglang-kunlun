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
"""Hooks for ``sglang.srt.distributed.communication_op``.

BKCL's all-reduce does not combine partial sums in a fixed order, so a bf16
payload is rounded differently from run to run. Measured 2026-10-25 on P800
(8L BF16, identical 4x1000-token fixtures, two runs, probe dumps compared
bitwise): every tensor produced *after* a RowParallel all-reduce differed by
exactly one bf16 ULP (``Attn_output`` 0.5-1.0, ``moe_or_dense_output`` 2.0),
while everything before the first all-reduce -- ``embedding_input``,
``embedding_output``, layer-0 ``layer_input`` -- was bit-identical. The error
compounds per layer (3.1e-2 at layer 1 to 5.0e-1 at layer 7 of eight) and at
temperature 0 eventually flips the greedy argmax: the first sampled token
alternates between 23406 (logprob -2.08) and 50941 (-2.35) for the same prompt.
``BKCL_DISABLE_QUANT=1`` and forcing a single flat ring
(``BKCL_MAX_CLUSTERS=1 BKCL_CLUSTERS_PER_RING=1 BKCL_ENABLE_TREE=0
BKCL_FLAT_RING=1``) do not change this.

Reducing in fp32 does not make BKCL's order fixed, but it moves the ordering
noise ~2^16 below the bf16 ULP, so the value the model actually consumes stops
depending on it. It is **off by default**: it halves the measured error in the
MLP output and makes the whole attention chain bit-identical, but a 2026-10-25
AIME run with it on still over-generated on 5 of 8 questions, so the accuracy
benefit is unproven and it doubles the all-reduce payload. Turn it on with
``SGLANG_KUNLUN_FP32_ALLREDUCE=1``.
"""

from __future__ import annotations

import os

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_ENV = "SGLANG_KUNLUN_FP32_ALLREDUCE"
_ENV_KEEP = "SGLANG_KUNLUN_FP32_ALLREDUCE_KEEP"
# Only the low-precision floating payloads are affected; fp32/fp64 already
# round once, and integer reductions are exact in any order.
_LOW_PRECISION = (torch.bfloat16, torch.float16)
_announced = False


def _enabled() -> bool:
    return os.environ.get(_ENV, "0").lower() in ("1", "true", "yes", "on")


def _keep_fp32() -> bool:
    """Hand the fp32 result straight to the caller instead of rounding back.

    EB5's consumers of a reduced tensor are ``mhc_post`` / ``mhc_pre``, whose
    torch implementations multiply by fp32 mix matrices and only round once at
    the end (`sglang_kunlun/eb5_adapt/kernels/mhc_torch.py:mhc_post`). Casting
    to bf16 here and letting them promote back is a pure round trip: it is the
    *only* remaining rounding between the fp32 reduction and the fp32
    accumulation, and it is what still seeds a 1-ULP difference in
    ``moe_or_dense_output``. Off by default because any consumer that assumes
    bf16 will now see fp32.
    """
    return os.environ.get(_ENV_KEEP, "0").lower() in ("1", "true", "yes", "on")


@plugin_hook(
    "sglang.srt.distributed.parallel_state.GroupCoordinator.all_reduce",
    type=HookType.AROUND,
)
def all_reduce_kunlun(original_fn, self, input_: torch.Tensor):
    """Reduce bf16/fp16 payloads in fp32 so the result stops depending on
    BKCL's partial-sum order.

    Hooked on the group coordinator rather than on
    ``tensor_model_parallel_all_reduce`` because EB5 reaches the collective by
    several routes: the module-level helper (MoE output, RowParallel linears)
    and ``parallel.attn_tp_group.all_reduce`` directly (attention output under
    DP attention, ernie5.py:3744). The method is the single choke point.
    """
    global _announced
    if not _enabled() or input_.dtype not in _LOW_PRECISION:
        return original_fn(self, input_)
    keep = _keep_fp32()
    if not _announced:
        _announced = True
        tail = "fp32 (kept)" if keep else f"{input_.dtype}"
        print(
            f"[kunlun-fp32-allreduce] active ({input_.dtype} -> fp32 -> {tail})",
            flush=True,
        )
    out = original_fn(self, input_.to(torch.float32))
    if keep:
        return out
    # ``all_reduce`` may return the storage it was handed; cast back regardless
    # so callers keep seeing the dtype they passed in.
    return out.to(input_.dtype)
