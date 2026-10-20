#!/usr/bin/env bash
# 多 rank 挂死复现器（不加载模型权重）。
#
# 背景：单测是单进程单卡，D2H 类挂死抓不到；而起服务要 "加载权重 + warmup +
# watchdog 300 s"，一轮十几分钟。本脚本用 N 个 rank 直接跑一个算子，5 s 心跳，
# 几十秒就能判断"这个算子在多 rank 下会不会挂"。
#
# 用法：
#   bash repro_multirank.sh <case> [gpus] [wait_seconds]
#   bash repro_multirank.sh memcpy-partial 0,1,2,3 90
#   bash repro_multirank.sh mhc-pre 0,1,2,3 90
#
# case：baseline | memcpy-full | memcpy-partial | memcpy-scatter | mhc-pre | mhc-post
#
# 🔴 挂死时**不要 kill**：进程会自己超时退出（doc/3 开头）。脚本只报告，不发信号。
set -uo pipefail

BINGOO=${BINGOO:-/root/paddlejob/workspace/bingoo}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CASE=${1:?用法: repro_multirank.sh <case> [gpus] [wait_seconds]}
GPUS=${2:-0,1,2,3}
WAIT=${3:-90}
NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
PORT=${REPRO_PORT:-29601}
LOG=${REPRO_LOG:-/tmp/eb5_repro_${CASE}_$(date +%Y%m%d_%H%M%S).log}

say() { echo "[repro] $*"; }

# ---- 设备必须空闲，否则测的是上一个残留 ----
free=0
for d in $(tr ',' ' ' <<<"$GPUS"); do
  # 不能写成 `xpu-smi -i "$d" | grep -q`：`set -o pipefail` 下 grep 命中后提前退出，
  # xpu-smi 收 SIGPIPE(141)，整条管道判失败 → 空闲卡被误判成占用（与 preflight §10.4 同型）。
  out=$(xpu-smi -i "$d" 2>/dev/null) || continue
  grep -qE '0MiB / 98304MiB' <<<"$out" && free=$((free + 1))
done
if [ "$free" -ne "$NGPU" ]; then
  say "❌ 只有 $free/$NGPU 张卡空闲，先按 doc/3 第五节恢复，别硬上"
  exit 1
fi

source "$BINGOO/code/sglang-kunlun/01_eb_adp/tool/env.sh"
export XPU_VISIBLE_DEVICES=$GPUS CUDA_VISIBLE_DEVICES=$GPUS
export MASTER_ADDR=127.0.0.1 MASTER_PORT=$PORT
: > "$LOG"

say "case=$CASE gpus=$GPUS ws=$NGPU port=$PORT log=$LOG"
say "参数：splits=${REPRO_SPLITS:-32}"
say "🔴 若挂死不要 kill，等它自行超时退出"

for r in $(seq 0 $((NGPU - 1))); do
  (
    env RANK=$r WORLD_SIZE=$NGPU REPRO_DEVICE=$r \
      python -u "$HERE/repro_multirank.py" --case "$CASE" \
      --splits "${REPRO_SPLITS:-32}" >> "$LOG" 2>&1
  ) &
done

deadline=$(( $(date +%s) + WAIT ))
while :; do
  ok=$(grep -c "CASE $CASE OK" "$LOG" 2>/dev/null || true)
  err=$(grep -c "CASE ERROR" "$LOG" 2>/dev/null || true)
  [ "${ok:-0}" -ge "$NGPU" ] && break
  [ "${err:-0}" -gt 0 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    say "❌ ${WAIT}s 内没出齐 $NGPU/$NGPU（OK=${ok:-0} ERROR=${err:-0}）"
    for r in $(seq 0 $((NGPU - 1))); do
      last=$(grep "\[eb5-repro\]\[r$r\]" "$LOG" | tail -1)
      say "  rank$r 最后一行: ${last:-<无输出>}"
    done
    say "⚠️ 挂住的进程**不要 kill**，等它自行超时；日志：$LOG"
    exit 1
  fi
  sleep 2
done

if [ "${err:-0}" -gt 0 ]; then
  say "❌ 有用例报错（不是挂死）："
  grep -A 15 "CASE ERROR" "$LOG" | tail -40
  exit 2
fi

say "✅ $CASE 在 $NGPU 个 rank 上全部通过"
grep "CASE $CASE OK" "$LOG"
