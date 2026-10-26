"""Native XPU sparse decode for EB5 HCA layers.

Two call shapes reach ``sgl_kernel.flash_mla.flash_mla_with_kvcache`` from the
EB5 backend:

* ``EB5AttnBackend._swa_decode_flashmla`` (window-only, no compressor cache);
* the DSV4 ``forward`` that ``EB5MLAHCAAttnBackend._forward_hca`` delegates to:
  a window of SWA slot ids **plus** the c128 compressor pool
  (``extra_k_cache`` / ``extra_indices_in_kvcache``).

``eb5_mla_adapt`` installs a pure-torch fallback at that symbol.  This hook
wraps it and calls the Kunlun native ops instead:

* both pools present -> ``compressed_attention`` / ``hybrid_attention`` (the
  same hybrid window+compressed sink-aware op DSV4 runs on XPU);
* window only -> ``speculative_attention``.

The compressor branch needs ``compress_ratio`` and the per-request sequence
lengths, which are not kernel arguments.  ``_forward_hca`` publishes them in
``_CTX`` for the duration of the call.

The EB5 SWA pool is addressed by flat slot ids (``page * page_size + offset``),
so flattening ``[num_pages, page_size, 1, dim]`` to ``[num_slots, dim]`` keeps
the same addressing and matches what the native ops consume.

Gated by ``SGLANG_EB5_HCA_NATIVE_SPARSE_DECODE`` (off by default).
"""

from __future__ import annotations

import math
import os
import sys

import torch

_ENV = "SGLANG_EB5_HCA_NATIVE_SPARSE_DECODE"
_CTX: dict = {}
_ACTIVE_LOGGED = False


def _enabled() -> bool:
    return os.environ.get(_ENV, "0").lower() in ("1", "true", "yes", "on")


def _flatten(cache: torch.Tensor) -> torch.Tensor:
    """[num_pages, page, heads, dim] -> [num_slots, dim] (flat slot ids)."""
    return cache.reshape(-1, cache.shape[-1]).contiguous()


def _normalise(indices: torch.Tensor, num_slots: int) -> torch.Tensor:
    """The native op only understands -1 as "no key"."""
    return torch.where(
        (indices >= 0) & (indices < num_slots),
        indices,
        torch.full_like(indices, -1),
    ).contiguous()


def _qfused(
    q,
    win_cache,
    win_indices,
    com_cache,
    com_indices,
    seq_lens,
    scale,
    win_size,
    com_ratio,
    com_topk,
    attn_sink,
    h_q,
    d_v,
):
    import kunlun_ops

    bs = win_indices.shape[0]
    q3 = q.squeeze(1).contiguous()
    out = torch.empty((bs, h_q, d_v), dtype=q.dtype, device=q.device)
    max_logits = torch.empty((bs, h_q), dtype=torch.float32, device=q.device)
    lse = torch.empty_like(max_logits)

    qlod_cpu = torch.arange(bs + 1, dtype=torch.int32)
    qlod_xpu = qlod_cpu.to(q.device)
    kvseq_cpu = seq_lens.to(torch.int32).cpu().contiguous()
    kvseq_xpu = seq_lens.to(torch.int32).contiguous()

    # ``hybrid_attention`` is the generic entry point; ``compressed_attention``
    # is the h_q=64 fast path the DSV4 XPU backend uses for decode.  Both take
    # the same math; pick with SGLANG_EB5_HCA_NATIVE_SPARSE_DECODE_OP.
    if os.environ.get("SGLANG_EB5_HCA_NATIVE_SPARSE_DECODE_OP", "fast") != "hybrid":
        torch.ops.xspeedgate_ops.compressed_attention(
            q3,
            win_cache,
            win_indices,
            com_cache,
            com_indices,
            out,
            max_logits,
            lse,
            qlod_cpu.contiguous(),
            qlod_xpu.contiguous(),
            kvseq_cpu,
            kvseq_xpu,
            float(scale),
            True,
            win_size,
            com_ratio,
            com_topk,
            attn_sink.contiguous(),
        )
        return out.unsqueeze(1)
    kunlun_ops.hybrid_attention(
        q=q3,
        win_kv_cache=win_cache,
        win_indices=win_indices,
        com_kv_cache=com_cache,
        com_indices=com_indices,
        o=out,
        max_logits=max_logits,
        lse=lse,
        qlod_cpu=qlod_cpu.contiguous(),
        qlod_xpu=qlod_xpu.contiguous(),
        kvseqlen_cpu=kvseq_cpu,
        kvseqlen_xpu=kvseq_xpu,
        sm_scale=float(scale),
        is_causal=True,
        max_window_size=win_size,
        compress_ratio=com_ratio,
        com_topk=com_topk,
        attn_sink=attn_sink.contiguous(),
    )
    return out.unsqueeze(1)


def _native_compressed_decode(
    q,
    k_cache,
    indices,
    extra_k_cache,
    extra_indices,
    head_dim_v,
    softmax_scale,
    attn_sink,
):
    global _ACTIVE_LOGGED
    if not _ACTIVE_LOGGED:
        print(
            "[eb5-native-decode] native compressed_attention (window+c128) active",
            flush=True,
        )
        _ACTIVE_LOGGED = True

    bs, qlen, h_q, d_qk = q.shape
    d_v = int(head_dim_v or d_qk)
    scale = (
        float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(d_qk)
    )

    win_cache = _flatten(k_cache)
    com_cache = _flatten(extra_k_cache)
    # EB5's window rows are most-recent-first with the pre-sequence tail set to
    # -1.  With is_causal=True the op keeps only entries t < kvseqlen, which is
    # exactly the valid prefix, so the -1 tail is never read.
    win_indices = _normalise(indices[:, 0, :].to(torch.int64), win_cache.shape[0]).to(
        torch.int32
    )
    com_indices = _normalise(
        extra_indices[:, 0, :].to(torch.int64), com_cache.shape[0]
    )

    seq_lens = _CTX["forward_batch"].seq_lens[:bs].to(torch.int32)
    com_ratio = int(_CTX.get("compress_ratio") or 128)
    # DSV4 slices the compressor window to the batch's longest live prefix; the
    # op asserts com_topk == com_indices.size(1) and masks the rest itself.
    com_topk = max(int(seq_lens.max().item()) // com_ratio, 1)
    com_topk = min(com_topk, com_indices.shape[1])
    com_indices = com_indices[:, :com_topk].contiguous().to(torch.int32)
    # The op asserts max_window_size == win_indices.size(1).
    win_size = win_indices.shape[1]

    return (
        _qfused(
            q,
            win_cache,
            win_indices,
            com_cache,
            com_indices,
            seq_lens,
            scale,
            win_size,
            com_ratio,
            com_topk,
            attn_sink,
            h_q,
            d_v,
        ),
        None,
    )


def _native_sparse_decode(
    q, k_cache, indices, topk_length, head_dim_v, softmax_scale, attn_sink
):
    global _ACTIVE_LOGGED
    if not _ACTIVE_LOGGED:
        print("[eb5-native-decode] native speculative_attention active", flush=True)
        _ACTIVE_LOGGED = True

    import kunlun_ops

    bs, qlen, h_q, d_qk = q.shape
    d_v = int(head_dim_v or d_qk)
    block_tables = indices[:, 0, :].to(torch.int32).contiguous()
    context_lens = topk_length.reshape(-1).to(torch.int32).contiguous()
    max_blocks = int(block_tables.shape[1])
    scale = (
        float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(d_qk)
    )

    out = torch.empty((bs, qlen, h_q, d_v), dtype=q.dtype, device=q.device)
    kunlun_ops.speculative_attention(
        out=out,
        q=q.contiguous(),
        k_cache=k_cache,
        v_cache=k_cache,
        context_lens_cpu=context_lens.cpu(),
        context_lens_xpu=context_lens,
        batch_num=bs,
        qlen=qlen,
        max_context_len=max_blocks,
        head_num=h_q,
        head_dim=d_qk,
        scale=scale,
        kv_head_num=1,
        block_size=1,
        max_num_blocks_per_seq=max_blocks,
        max_window_size=-1,
        block_tables=block_tables,
        sink=attn_sink.to(torch.float32) if attn_sink is not None else None,
    )
    return out, None


def _wrap(torch_impl):
    def flash_mla_with_kvcache(
        q,
        k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=None,
        tile_scheduler_metadata=None,
        num_splits=None,
        softmax_scale=None,
        causal=False,
        is_fp8_kvcache=False,
        indices=None,
        page_size=64,
        topk_length=None,
        attn_sink=None,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        extra_topk_length=None,
        **kwargs,
    ):
        base_ok = (
            _enabled()
            and indices is not None
            and attn_sink is not None
            and block_table is None
            and cache_seqlens is None
            and not causal
            and not is_fp8_kvcache
            and q.dim() == 4
            and q.shape[1] <= 32
        )
        if (
            base_ok
            and extra_k_cache is not None
            and extra_indices_in_kvcache is not None
            and "compress_ratio" in _CTX
        ):
            return _native_compressed_decode(
                q,
                k_cache,
                indices,
                extra_k_cache,
                extra_indices_in_kvcache,
                head_dim_v,
                softmax_scale,
                attn_sink,
            )
        if (
            base_ok
            and extra_k_cache is None
            and extra_indices_in_kvcache is None
            and topk_length is not None
        ):
            return _native_sparse_decode(
                q,
                k_cache,
                indices,
                topk_length,
                head_dim_v,
                softmax_scale,
                attn_sink,
            )
        return torch_impl(
            q=q,
            k_cache=k_cache,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=head_dim_v,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
            softmax_scale=softmax_scale,
            causal=causal,
            is_fp8_kvcache=is_fp8_kvcache,
            indices=indices,
            page_size=page_size,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_k_cache=extra_k_cache,
            extra_indices_in_kvcache=extra_indices_in_kvcache,
            extra_topk_length=extra_topk_length,
            **kwargs,
        )

    return flash_mla_with_kvcache


def _install_context_hook() -> list:
    """Publish compress_ratio / forward_batch around ``_forward_hca``."""
    try:
        from sglang.srt.internal.layers.attention.mla_hca.backend import (
            EB5MLAHCAAttnBackend,
        )
    except Exception:  # noqa: BLE001 - not an EB5 process
        return []
    if getattr(EB5MLAHCAAttnBackend, "_eb5_native_decode_ctx", False):
        return []

    original = EB5MLAHCAAttnBackend._forward_hca

    def _forward_hca(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        compress_ratio,
        *,
        save_kv_cache,
        attn_sink,
        **kwargs,
    ):
        _CTX["backend"] = self
        _CTX["forward_batch"] = forward_batch
        _CTX["compress_ratio"] = compress_ratio
        try:
            return original(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                compress_ratio,
                save_kv_cache=save_kv_cache,
                attn_sink=attn_sink,
                **kwargs,
            )
        finally:
            _CTX.clear()

    EB5MLAHCAAttnBackend._forward_hca = _forward_hca
    EB5MLAHCAAttnBackend._eb5_native_decode_ctx = True
    return ["EB5MLAHCAAttnBackend._forward_hca"]


def install() -> list:
    if os.environ.get("SGLANG_EB5_DISABLE_NATIVE_HOOKS") == "1":
        return []
    """Wrap the installed torch ``flash_mla_with_kvcache``; idempotent.

    ``sgl_kernel`` is a ``_StubModule`` whose ``__getattr__`` fabricates a
    child module for *any* missing name, so ``getattr(mod, "x", False)`` is
    always truthy.  Every presence check here must go through ``__dict__``.
    """
    patched = []
    try:
        patched.extend(_install_context_hook())
    except Exception as exc:  # noqa: BLE001
        print(f"[eb5-native-decode] context hook failed: {exc}")

    mod = sys.modules.get("sgl_kernel.flash_mla")
    if mod is None:
        return patched
    if mod.__dict__.get("_eb5_native_sparse_decode"):
        return patched
    impl = mod.__dict__.get("flash_mla_with_kvcache")
    if impl is None:
        return patched

    wrapped = _wrap(impl)
    mod.flash_mla_with_kvcache = wrapped
    mod._eb5_native_sparse_decode = True
    patched.append("sgl_kernel.flash_mla.flash_mla_with_kvcache")

    root = sys.modules.get("sgl_kernel")
    if root is not None:
        root.flash_mla_with_kvcache = wrapped
        patched.append("sgl_kernel.flash_mla_with_kvcache")
    return patched
