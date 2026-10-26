"""Native XPU sparse prefill for EB5 HCA SWA.

The fork's ``EB5AttnBackend`` calls ``sgl_kernel.flash_mla.flash_mla_sparse_fwd``
for the SWA prefill window.  ``eb5_mla_adapt`` installs a pure-torch fallback at
that symbol.  This hook wraps that fallback:

* when ``SGLANG_EB5_HCA_NATIVE_SPARSE_PREFILL`` is on and every query has a
  full window (no ``-1`` padding), call the native
  ``kunlun_ops.sparse_prefill_fwd_opt_v2`` and fold the attention sink in from
  its natural-log ``lse``;
* otherwise delegate to the torch fallback unchanged.

Why the full-window guard: the native op does not mask ``-1`` indices, so a
partial first-chunk window would read a bogus slot.  The fork's metadata
already pads those slots with ``-1``; we simply fall back for that chunk.

The wrapper is installed after ``eb5_mla_adapt.apply()`` so it wraps the torch
implementation rather than replacing it, keeping the fallback available.
"""

from __future__ import annotations

import os
import sys

import torch

_ENV = "SGLANG_EB5_HCA_NATIVE_SPARSE_PREFILL"
_ACTIVE_LOGGED = False


def _enabled() -> bool:
    return os.environ.get(_ENV, "0").lower() in ("1", "true", "yes", "on")


def _native_sparse_prefill(q, kv, indices, sm_scale, d_v, attn_sink):
    """Call the native sparse prefill and fold the natural-log sink.

    ``attn_sink`` is a natural-log bias (same convention as the native paged
    decode); the torch fallback works in log2, hence the explicit conversion
    only in that path.  Here ``lse`` is natural-log, so the virtual V=0 key is
    ``exp(lse) / (exp(lse) + exp(sink))``.
    """
    import kunlun_ops

    global _ACTIVE_LOGGED
    if not _ACTIVE_LOGGED:
        print("[eb5-native-prefill] native sparse_prefill_fwd_opt_v2 active", flush=True)
        _ACTIVE_LOGGED = True

    s_q, h_q, _ = q.shape
    d_v = int(d_v or 512)
    kv2d = kv[:, 0, :] if kv.dim() == 3 else kv

    out = torch.empty((s_q, h_q, d_v), dtype=q.dtype, device=q.device)
    max_logits = torch.empty((s_q, h_q), dtype=torch.float32, device=q.device)
    lse = torch.empty((s_q, h_q), dtype=torch.float32, device=q.device)
    ret = kunlun_ops.sparse_prefill_fwd_opt_v2(
        q.contiguous(),
        kv2d,
        indices.contiguous(),
        out,
        max_logits,
        lse,
        float(sm_scale),
        d_v=d_v,
        is_causal=False,
    )
    if ret != 0:
        raise RuntimeError(f"sparse_prefill_fwd_opt_v2 returned {ret}")

    if attn_sink is not None:
        sink = attn_sink.float().reshape(-1)[:h_q].reshape(1, h_q)
        m = torch.maximum(lse, sink)
        w = torch.exp(lse - m) / (torch.exp(lse - m) + torch.exp(sink - m))
        w = torch.where(torch.isfinite(m), w, torch.zeros_like(w))
        out = out * w.unsqueeze(-1)
        # Keep the returned lse/max_logits consistent with the sink-aware
        # torch fallback: the virtual V=0 key also contributes to both.
        lse = torch.logaddexp(lse, sink)
        max_logits = torch.maximum(max_logits, sink)

    return out, max_logits, lse


def _wrap(torch_impl):
    def flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v=512,
        attn_sink=None,
        topk_length=None,
    ):
        if _enabled() and topk_length is not None:
            topk = int(indices.shape[-1])
            if bool((topk_length.reshape(-1) == topk).all().item()):
                return _native_sparse_prefill(
                    q, kv, indices, sm_scale, d_v, attn_sink
                )
        return torch_impl(
            q, kv, indices, sm_scale, d_v, attn_sink, topk_length
        )

    return flash_mla_sparse_fwd


def install() -> list:
    if os.environ.get("SGLANG_EB5_DISABLE_NATIVE_HOOKS") == "1":
        return []
    """Wrap the installed torch ``flash_mla_sparse_fwd``; idempotent.

    ``sgl_kernel`` is a ``_StubModule`` whose ``__getattr__`` fabricates a
    child module for *any* missing name, so ``getattr(mod, "x", False)`` is
    always truthy.  Every presence check here must go through ``__dict__``.
    """
    mod = sys.modules.get("sgl_kernel.flash_mla")
    if mod is None:
        return []
    if mod.__dict__.get("_eb5_native_sparse_prefill"):
        return []
    impl = mod.__dict__.get("flash_mla_sparse_fwd")
    if impl is None:
        return []

    wrapped = _wrap(impl)
    mod.flash_mla_sparse_fwd = wrapped
    mod._eb5_native_sparse_prefill = True
    patched = ["sgl_kernel.flash_mla.flash_mla_sparse_fwd"]

    root = sys.modules.get("sgl_kernel")
    if root is not None:
        root.flash_mla_sparse_fwd = wrapped
        patched.append("sgl_kernel.flash_mla_sparse_fwd")
    return patched
