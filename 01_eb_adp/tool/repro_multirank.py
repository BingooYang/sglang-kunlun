#!/usr/bin/env python3
"""多 rank 挂死复现器：**不加载权重、不跑模型**，只把一个可疑算子放进真实多 rank 环境。

为什么需要
----------
单测是单进程单卡：`int(offset)`、布尔索引 `x[mask]` 这类**隐式 D2H** 在那种环境下
不会挂，只有 `init_process_group(nccl)` + 多 rank 才会稳定复现（doc/3 第四节）。
而服务验证要经过"加载权重 → warmup → watchdog 300 s"，一轮十几分钟，设备还会被拖坏。

本脚本把这两步都省掉：N 个 rank 起来后**直接调一个算子**，用 5 s 心跳标出每个 rank
走到了哪一步。挂死时能立刻知道是哪个 case、哪个 rank，不需要等模型的 watchdog。

用法（一般由 `repro_multirank.sh` 调用）：
    RANK=0 WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 MASTER_PORT=29601 \
      python -u repro_multirank.py --case memcpy-partial

case 列表：
    baseline         只做 init_process_group + all_reduce + `.cpu()`（设备健康基线）
    memcpy-full      memcpy_triton：整块拷贝（旧单测唯一的参数）
    memcpy-partial   memcpy_triton：非零 offset + 截断（2026-10-20 挂死点）
    memcpy-scatter   memcpy_triton：`offset_src=False`
    mhc-pre          直接调 fork 的 `mhc.mhc_pre`（3 token，最接近服务）
                     走适配层的散装 torch 实现（与 DSV4 同构）
    mhc-post         直接调 fork 的 `mhc.mhc_post`（同上）

**挂死时不要 kill**（doc/3 开头）：等它自己超时退出。
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import threading
import time
import traceback

_HELP = pathlib.Path(__file__).resolve().parent
_TEST_DIR = _HELP.parent / "test"

HC_MULT = 4
HIDDEN = 4096
HC_HIDDEN = HC_MULT * HIDDEN          # 16384
HC_MULT3 = HC_MULT * 2 + HC_MULT * HC_MULT  # 24
TOKENS = 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", required=True, choices=[
        "baseline", "memcpy-full", "memcpy-partial", "memcpy-scatter",
        "mhc-pre", "mhc-post",
    ])
    ap.add_argument("--splits", type=int, default=32, help="split-k 段数（默认 32，服务真实值）")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    ws = int(os.environ["WORLD_SIZE"])
    dev = torch_device()
    t0 = time.time()
    state = {"step": "import"}
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(5):
            print(
                f"[eb5-repro][r{rank}] alive step={state['step']} "
                f"t={time.time() - t0:.0f}s",
                flush=True,
            )

    def step(name: str) -> None:
        state["step"] = name
        print(f"[eb5-repro][r{rank}] STEP {name}", flush=True)

    threading.Thread(target=beat, daemon=True).start()

    import torch
    import torch.distributed as dist

    step("init_pg")
    torch.cuda.set_device(dev)
    dist.init_process_group(
        backend="nccl",
        init_method=(
            f"tcp://{os.environ.get('MASTER_ADDR', '127.0.0.1')}:"
            f"{os.environ.get('MASTER_PORT', '29601')}"
        ),
        world_size=ws,
        rank=rank,
    )

    # ---- 基线：doc/3 第四节的最小复现（int64 op + D2H）----
    step("baseline_allreduce_d2h")
    t = torch.tensor([float(rank + 1)], device=dev)
    dist.all_reduce(t)
    p8l = torch.arange(8, dtype=torch.int64, device=dev) * (rank + 1)
    _ = p8l.cpu()
    torch.cuda.synchronize()
    print(f"[eb5-repro][r{rank}] baseline sum={t.item()}", flush=True)

    if args.case == "baseline":
        return finish(rank, args.case, ws)

    step("apply_adapt")
    sys.path.insert(0, str(_TEST_DIR))
    import eb5_kunlun_adapt as A

    A.apply()

    if args.case.startswith("memcpy"):
        step(args.case)
        from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton

        dst = torch.zeros(8, 64, dtype=torch.bfloat16, device=dev)
        src = torch.randn(16, 64, dtype=torch.bfloat16, device=dev)
        if args.case == "memcpy-full":
            off = torch.tensor(0, dtype=torch.int32, device=dev)
            sz = torch.tensor(8, dtype=torch.int32, device=dev)
            offset_src = True
        else:
            off = torch.tensor(3, dtype=torch.int32, device=dev)
            sz = torch.tensor(4, dtype=torch.int32, device=dev)
            offset_src = args.case == "memcpy-partial"
        memcpy_triton(dst, src, 0, off, sz, offset_src)
        torch.cuda.synchronize()

    elif args.case == "mhc-pre":
        step(args.case)
        from sglang.kernels.ops.layernorm.mhc import mhc_pre

        residual = torch.randn(TOKENS, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=dev)
        fn = torch.randn(HC_MULT3, HC_HIDDEN, dtype=torch.float32, device=dev)
        hc_scale = torch.ones(3, dtype=torch.float32, device=dev)
        hc_base = torch.zeros(HC_MULT3, dtype=torch.float32, device=dev)
        norm_weight = torch.ones(HIDDEN, dtype=torch.bfloat16, device=dev)
        mhc_pre(
            residual, fn, hc_scale, hc_base,
            1e-6, 1e-6, 1e-6, 1.0, 4,          # eps*3 + post_mult + sinkhorn_repeat
            1, args.splits,                     # n_splits / n_splits_pre
            norm_weight=norm_weight,
            norm_eps=1e-6,
            mhc_align_fleet=True,               # EB5 权重（model_type=ernie）的真实取值
        )
        torch.cuda.synchronize()

    elif args.case == "mhc-post":
        step(args.case)
        from sglang.kernels.ops.layernorm.mhc import mhc_post

        x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device=dev)
        residual = torch.randn(
            TOKENS, HC_MULT, HIDDEN, dtype=torch.bfloat16, device=dev
        )
        post = torch.sigmoid(
            torch.randn(TOKENS, HC_MULT, dtype=torch.float32, device=dev)
        ).unsqueeze(-1)
        comb = torch.softmax(
            torch.randn(TOKENS, HC_MULT, HC_MULT, dtype=torch.float32, device=dev),
            dim=-1,
        )
        mhc_post(x, residual, post, comb)
        torch.cuda.synchronize()

    return finish(rank, args.case, ws, dist)


def torch_device():
    return int(os.environ.get("REPRO_DEVICE", "0"))


def finish(rank: int, case: str, ws: int, dist=None) -> int:
    print(f"[eb5-repro][r{rank}] CASE {case} OK", flush=True)
    if dist is not None:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print(f"[eb5-repro][r{os.environ.get('RANK')}] CASE ERROR", flush=True)
        traceback.print_exc()
        raise SystemExit(2)
