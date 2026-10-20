#!/usr/bin/env python3
"""Offline (CPU) accuracy probe for online 128x128 blockwise INT8.

Reads real EB5 weight tensors straight out of the safetensors checkpoint and
reports the GEMM error of the exact quantization scheme the Kunlun INT8 kernels
implement (per-128x128-block weight scale + per-128-wide-K-block activation
scale, both raw abs-max).  No device is needed.

    python tool/int8_quant_probe.py [--model DIR] [--m 8] [--ks 512,2048,16384]

The activation side has no checkpoint to sample from, so it is synthetic:
``randn`` scaled so |x| ~ 1 (post-RMSNorm magnitude), with an optional outlier
column pattern (``--outlier``, default 2.0 = mild, EB5-like heavy channels).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from sglang.srt.internal.layers.quantization import blockwise_int8_patch as bi8

DEFAULT_MODEL = "/root/paddlejob/workspace/bingoo/models/hf_checkpoint-48800"
TENSORS = [
    # layer 0: dense MLP (first_k_dense_replace=1)
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    # layer 3 (compress_ratio 128 / HCA flavour)
    "model.layers.3.self_attn.linear_q_up_proj.weight",
    "model.layers.3.self_attn.linear_q_down_proj.weight",
    "model.layers.3.self_attn.linear_kv_proj.weight",
    "model.layers.3.self_attn.o_proj.weight",
    # layer 14 (compress_ratio -2 / full-attention flavour)
    "model.layers.14.self_attn.q_b_proj.weight",
    "model.layers.14.self_attn.q_a_proj.weight",
    "model.layers.14.self_attn.kv_b_proj.weight",
    "model.layers.14.self_attn.o_proj.weight",
    # latent MoE glue + routed expert (expert_hidden = 2048)
    "model.layers.14.block_sparse_moe.fc1_latent_proj.weight",
    "model.layers.14.block_sparse_moe.fc2_latent_proj.weight",
    "model.layers.14.block_sparse_moe.experts.0.w1.weight",
    "model.layers.14.block_sparse_moe.experts.0.w2.weight",
    "model.layers.14.block_sparse_moe.experts.0.w3.weight",
]


def _load(model_dir: str, name: str) -> torch.Tensor:
    from safetensors import safe_open

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    shard = weight_map.get(name)
    if shard is None:
        raise KeyError(name)
    with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
        return f.get_tensor(name)


def _make_act(m: int, k: int, outlier: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator=g)
    if outlier > 1.0:
        cols = torch.randperm(k, generator=g)[: max(1, k // 64)]
        x[:, cols] *= outlier
    return x.bfloat16()


def _stats(ref, out):
    ref = ref.float()
    out = out.float()
    d = (out - ref).abs()
    return (
        ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item(),
        d.max().item(),
        d.mean().item(),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--outlier", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"model={args.model} m={args.m} outlier=x{args.outlier}")
    print(
        f"{'tensor':<58} {'n':>6} {'k':>6} {'rel_fro':>9} {'max':>9} {'mean':>9}"
    )
    worst = 0.0
    rows = []
    for name in TENSORS:
        try:
            w = _load(args.model, name)
        except Exception as e:  # noqa: BLE001 - a missing tensor must not stop the probe
            print(f"{name:<58} SKIP ({type(e).__name__}: {e})")
            continue
        if w.dim() != 2:
            print(f"{name:<58} SKIP (dim={w.dim()})")
            continue
        n, k = w.shape
        if k % 128 or n % 128:
            print(f"{name:<58} SKIP (not 128-aligned: {n}x{k})")
            continue
        x = _make_act(args.m, k, args.outlier, args.seed)
        q, s = bi8.quantize_weight_blockwise_int8(w)
        ref = x.float() @ w.float().T
        out = bi8.block_reference_matmul(x, q, s)
        rel, mx, mean = _stats(ref, out)
        worst = max(worst, rel)
        rows.append((name, rel))
        print(f"{name:<58} {n:>6} {k:>6} {rel * 100:8.3f}% {mx:9.5f} {mean:9.5f}")
        del w, q, s, x, ref, out
    print(f"\nworst rel_fro = {worst * 100:.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
