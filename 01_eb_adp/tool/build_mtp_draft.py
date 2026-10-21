#!/usr/bin/env python3
"""从整网 checkpoint 里**只取 MTP（NextN）那一层**，做成一个独立的 draft 目录。

为什么需要
----------
`build_submodel.py` 生成的减层子模型（`models/hf_checkpoint-48800-8L`）只保留
`model.layers.0..n-1`，把 MTP 层（整网里的 `model.layers.<num_hidden_layers>.*`）
一起裁掉了 —— 于是 `layers.43.*`（MTP）全都不在 index 里，开
`--speculative-algorithm EAGLE` 时 draft 权重**一个都加载不到**。

EB5 的 MTP 权重命名（和 DeepSeek 的 `model.decoder.*` 不同）
-----------------------------------------------------------
`ernie5.py::build_eb5_weight_handlers` 里：

    nextn_layer_id = model.config.num_hidden_layers
    ...
    if is_nextn and layer_id != nextn_layer_id: continue
    target_prefix = "model.decoder" if is_nextn else f"model.layers.{layer_id}"

也就是说 draft **按 `model.layers.<num_hidden_layers>.*` 从 checkpoint 里读**，
然后映射到运行时的 `model.decoder.*`；lm_head 走 `mtp_shared_last_layer`
（`load_lm_head_from_target=True`，eagle_worker 用 target 的 embed/lm_head 覆盖）。

所以本脚本不改任何张量名：draft 目录的 `num_hidden_layers` 保留整网值（43），
index 里只留 `model.layers.43.*`（+ `model.embed_tokens.weight`），
`compress_ratios` 保留整网那份 44 长的列表（`plan[43] == -2` 是
`EB5 MLA+HCA NextN requires a full-MLA MTP entry` 的硬条件）。

用法
----
    python tool/build_mtp_draft.py \
        --src models/hf_checkpoint-48800 \
        --dst models/hf_checkpoint-48800-mtp

起服务时：

    EXTRA_ARGS="... --speculative-algorithm EAGLE \
      --speculative-draft-model-path models/hf_checkpoint-48800-mtp"
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
FULL_MLA = -2


def die(msg: str) -> None:
    print(f"[build_mtp_draft] ERROR {msg}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--src", required=True, help="整网模型目录（含 MTP 层）")
    ap.add_argument("--dst", required=True, help="输出目录（只放 MTP 层）")
    ap.add_argument(
        "--layer",
        type=int,
        default=None,
        help="MTP 层在 checkpoint 里的下标（默认 = src 的 num_hidden_layers）",
    )
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的 dst")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if not os.path.isdir(src):
        die(f"源目录不存在: {src}")
    if os.path.exists(dst) and not args.force:
        die(f"目标已存在（拒绝覆盖）: {dst}；需要重建请加 --force")

    cfg = load_json(os.path.join(src, "config.json"))
    ratios = [int(r) for r in cfg["compress_ratios"]]
    num_layers = int(cfg["num_hidden_layers"])
    num_nextn = int(cfg.get("num_nextn_predict_layers", 1))
    mtp_layer = num_layers if args.layer is None else int(args.layer)

    if num_nextn <= 0:
        die(f"源 config 的 num_nextn_predict_layers={num_nextn}，不是一个带 MTP 的模型")
    if len(ratios) not in (num_layers, num_layers + num_nextn):
        die(
            f"compress_ratios 长度 {len(ratios)} 与 num_hidden_layers={num_layers} "
            f"+num_nextn={num_nextn} 不匹配"
        )
    if len(ratios) <= mtp_layer or ratios[mtp_layer] != FULL_MLA:
        die(
            f"compress_ratios[{mtp_layer}] 必须是 {FULL_MLA}（full-MLA MTP 层），"
            f"实际 {ratios[mtp_layer] if len(ratios) > mtp_layer else '<越界>'}；"
            "EB5 MLA+HCA NextN 只支持 full-MLA 的 MTP 层"
        )

    idx = load_json(os.path.join(src, "model.safetensors.index.json"))
    wm = idx["weight_map"]
    prefix = f"model.layers.{mtp_layer}."
    need = {k: v for k, v in wm.items() if k.startswith(prefix)}
    # draft 自己的 embed_tokens（运行时会被 target 的覆盖，但先让 param 有值）
    for k in ("model.embed_tokens.weight",):
        if k in wm:
            need[k] = wm[k]
    if not need:
        die(f"{src} 的 index 里没有 {prefix}*；这个 checkpoint 没带 MTP 层")

    shards = sorted(set(need.values()))
    read_bytes = sum(os.path.getsize(os.path.join(src, s)) for s in shards)

    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    with open(os.path.join(dst, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=4)
    # 关键：draft config 的 num_hidden_layers 必须等于 MTP 层下标，
    # 因为 loader 拿它当 `nextn_layer_id`。
    if int(cfg["num_hidden_layers"]) != mtp_layer:
        die(
            f"config 的 num_hidden_layers={cfg['num_hidden_layers']} 与 MTP 层下标 "
            f"{mtp_layer} 不一致；loader 的 nextn_layer_id 就是 num_hidden_layers，"
            "这种情况需要重写张量名（本脚本不支持）"
        )
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": idx.get("metadata", {}), "weight_map": need}, fh)

    for name in TOKENIZER_FILES:
        p = os.path.join(src, name)
        if os.path.exists(p):
            os.symlink(p, os.path.join(dst, name))
    for s in shards:
        os.symlink(os.path.join(src, s), os.path.join(dst, s))

    n_experts = sum(1 for k in need if ".experts." in k)
    print(
        f"[build_mtp_draft] mtp_layer={mtp_layer} keys={len(need)} "
        f"(routed-expert={n_experts}) shards={len(shards)} "
        f"read={read_bytes / 1e9:.1f} GB"
    )
    print(
        "[build_mtp_draft] mHC="
        f"{bool(cfg.get('enable_hyper_connections'))} "
        f"mtp_shared_last_layer={cfg.get('mtp_shared_last_layer')} "
        f"num_nextn={num_nextn}"
    )
    print(f"[build_mtp_draft] OK -> {dst} (symlinks, no extra space)")
    print(
        "[build_mtp_draft] start example:\n"
        f"  EB5_GPUS=4,5,6,7 EXTRA_ARGS=\"--speculative-algorithm EAGLE "
        f"--speculative-num-steps 1 --speculative-num-draft-tokens 2 "
        f"--speculative-eagle-topk 1 --speculative-draft-model-path {dst}\" \\\n"
        "    bash tool/run_eb5_kunlun.sh models/hf_checkpoint-48800-8L 30120"
    )


if __name__ == "__main__":
    main()
