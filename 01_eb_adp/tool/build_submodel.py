#!/usr/bin/env python3
"""构建**减层子模型**目录（软链分片 + 过滤 index + 改写 config），用于小卡数适配。

为什么需要
----------
EB5 完整权重 573 GB：8 卡时每卡 ~72 GB（可用 96 GB），**4 卡时每卡 143 GB 装不下**。
调试时用减层子模型可以把加载降到 20 s 以内、每卡只占 20~30 GB。

减层不能随便减（2026-10-19 实测）
--------------------------------
1. `srt/internal/layers/attention/mla_hca/plan.py::build_attn_plan` 要求
   `len(compress_ratios)` ∈ {`num_hidden_layers`, `num_hidden_layers + num_nextn`}，
   所以 **`compress_ratios` 必须同步截断**，并在末尾补回 MTP 的 `-2`。
2. KV pool 要求 `0..N-1` 里**至少有一个 full-MLA 层**（`compress_ratios[i] == -2`）。
   本权重该层在 **index 7**，所以最小可用是 **8 层**；4 层会报
   `EB5 MLA+HCA: the full-token budget was not captured`。

注意：**MTP 层不在本脚本的产物里**
----------------------------------
过滤条件是 `layers.<n>`，所以整网的 MTP 层（`model.layers.<num_hidden_layers>.*`，
`compress_ratios` 末尾那个 `-2`）会被一起裁掉 —— 想开
`--speculative-algorithm EAGLE` / NextN 时 draft 一个权重都读不到。
MTP 用配套的 `tool/build_mtp_draft.py` 单独取成一个 draft 目录，
见 `doc/8. EB5 8L MTP 适配.md`。

用法
----
    python tool/build_submodel.py \
        --src  models/hf_checkpoint-48800 \
        --dst  models/hf_checkpoint-48800-8L \
        --layers 8 --gpus 4

`--gpus` 只用于估算"能不能装下"，不通过时默认报错（`--allow-oversize` 可强制）。
已存在的目标目录不会被覆盖（`--force` 除外）。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)
CARD_BYTES = 96 * 1024**3          # P800 单卡显存
SAFE_FRACTION = 0.90               # 权重要给 KV cache 留余量


def die(msg: str) -> None:
    print(f"[build_submodel] ❌ {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="原始模型目录（含 config.json + index）")
    ap.add_argument("--dst", required=True, help="输出目录")
    ap.add_argument("--layers", type=int, required=True, help="保留的 decoder 层数（从 0 开始）")
    ap.add_argument("--gpus", type=int, default=8, help="用于估算显存的卡数（默认 8）")
    ap.add_argument("--allow-oversize", action="store_true", help="估算超显存也继续")
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的 dst 目录")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    n = args.layers
    if not os.path.isdir(src):
        die(f"源目录不存在: {src}")
    if os.path.exists(dst) and not args.force:
        die(f"目标已存在（拒绝覆盖）: {dst}；需要重建请加 --force")
    if n <= 0:
        die("--layers 必须 > 0")

    cfg = load_json(os.path.join(src, "config.json"))
    ratios = [int(r) for r in cfg["compress_ratios"]]
    orig_layers = int(cfg["num_hidden_layers"])
    num_nextn = int(cfg.get("num_nextn_predict_layers", 1))
    if len(ratios) not in (orig_layers, orig_layers + num_nextn):
        die(f"源 compress_ratios 长度 {len(ratios)} 与 num_hidden_layers={orig_layers} 不匹配")
    if n > orig_layers:
        die(f"--layers={n} 超过源模型层数 {orig_layers}")

    full_mla = [i for i, r in enumerate(ratios[:orig_layers]) if r == -2]
    if not full_mla:
        die("源模型没有 full-MLA 层（compress_ratios 里没有 -2）")
    min_layers = full_mla[0] + 1
    if n < min_layers:
        die(
            f"至少需要 {min_layers} 层：前 {n} 层里没有 full-MLA 层"
            f"（第一个 -2 在 index {full_mla[0]}），KV pool 会断言失败"
        )

    # ---- 新 config：截断 compress_ratios，并把 MTP 的 -2 补回末尾 ----
    cfg["num_hidden_layers"] = n
    cfg["compress_ratios"] = ratios[:n] + [ratios[-1]]

    # ---- 过滤 index：只保留 <= n 层 + 所有非层张量 ----
    idx = load_json(os.path.join(src, "model.safetensors.index.json"))
    need = {
        k: s
        for k, s in idx["weight_map"].items()
        if not k.startswith("model.layers.") or int(k.split(".")[2]) < n
    }
    shards = sorted(set(need.values()))
    read_bytes = sum(os.path.getsize(os.path.join(src, s)) for s in shards)
    est_per_card = read_bytes / args.gpus
    print(
        f"[build_submodel] layers={n} full_mla={full_mla[:3]} ratios={cfg['compress_ratios'][:3]}"
        f"...{cfg['compress_ratios'][-2:]}"
    )
    print(
        f"[build_submodel] keys={len(need)} shards={len(shards)}/{len(set(idx['weight_map'].values()))}"
        f" 读取量={read_bytes / 1e9:.1f} GB  按 {args.gpus} 卡约 {est_per_card / 1e9:.1f} GB/卡"
    )
    if est_per_card > CARD_BYTES * SAFE_FRACTION and not args.allow_oversize:
        die(
            f"估算每卡 {est_per_card / 1e9:.1f} GB 超过单卡安全上限"
            f" {CARD_BYTES * SAFE_FRACTION / 1e9:.0f} GB；减小 --layers 或增加 --gpus"
        )

    # ---- 落盘 ----
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    with open(os.path.join(dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=4)
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": idx.get("metadata", {}), "weight_map": need}, fh)

    for name in TOKENIZER_FILES:
        p = os.path.join(src, name)
        if os.path.exists(p):
            os.symlink(p, os.path.join(dst, name))
    for s in shards:
        os.symlink(os.path.join(src, s), os.path.join(dst, s))

    print(f"[build_submodel] ✅ 已生成 {dst}（软链，不占额外空间）")
    print(
        "[build_submodel] 启动示例：\n"
        f"  EB5_GPUS=0,1,2,3 CONTEXT_LENGTH=8192 \\\n"
        f"    bash tool/run_eb5_kunlun.sh {dst} 30087"
    )


if __name__ == "__main__":
    main()
