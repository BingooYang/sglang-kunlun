#!/usr/bin/env bash
# 起服务前的健康闸门。**任一检查不过就拒绝起服务**。
#
# 为什么需要（2026-10-17 的教训）：EB5 首次前向触发设备级 kernel 异常
# （`KL_XID_KERNEL_EXCEPTION`）之后，`xpu-smi` 仍然显示 8 卡 `0MiB / 0%`、
# 驱动登记 0 个进程 —— **看起来完全正常**，但设备其实已经坏了：
# 紧接着的一次起服务就在加载权重阶段挂死（rank 0/1/2 空转、3 卡 util 100%）。
#
# 所以判据不能只看显存，必须做**功能性探针**：
#   1) 设备 8 卡都回到 `0 MiB / 0%`
#   2) dmesg 的 Xid 计数没有比上次记录时增长（设备没受过新伤）
#   3) 8 进程 XCCL 探针 8/8 通过  ← 单卡探针抓不到 rank 0/1/2 挂死，必须 8 进程
#
# 用法：  bash preflight.sh
#   通过 exit 0；不过 exit 1。
#   首次运行会记录 Xid 基线。**基线会自己漂移**：即使没人主动跑任务，
#   "上次挂死/崩溃的收尾"也会让 Xid 计数涨 —— 此时第 2 项会拒绝启动，
#   这是**故意的**（宁可挡住，也别在受伤的设备上再上一次）。
#   确认设备已按 doc/3 第五节恢复后，用
#     PREFLIGHT_ACCEPT_XID=1 bash preflight.sh
#   接受新基线并继续。
#   跳过全部检查（**不推荐**，仅用于明确知道在干什么时）：
#     SKIP_PREFLIGHT=1 bash preflight.sh
set -uo pipefail

BINGOO=${BINGOO:-/root/paddlejob/workspace/bingoo}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASELINE="$BINGOO/logs/.preflight_xid_baseline"
PROBE_TIMEOUT=${PROBE_TIMEOUT:-240}

say() { echo "[preflight] $*"; }
fail() { say "❌ $*"; exit 1; }

if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
  say "⚠️ SKIP_PREFLIGHT=1，跳过全部检查（你自己担着）"
  exit 0
fi

# ---- 1) 设备空闲 ----
free=$(xpu-smi 2>/dev/null | grep -cE '0MiB / 98304MiB')
say "设备：$free/8 张卡空闲（0 MiB）"
[ "$free" -eq 8 ] || fail "有卡没回到 0 MiB —— 先按 doc/3 第五节恢复，别硬上"

# ---- 2) Xid 计数（设备是否受过新伤）----
mkdir -p "$(dirname "$BASELINE")"
cur=$(dmesg -T 2>/dev/null | grep -c 'KL_XID_KERNEL_EXCEPTION' || echo 0)
if [ ! -f "$BASELINE" ]; then
  echo "$cur" > "$BASELINE"
  say "首次记录 Xid 基线 = $cur"
else
  base=$(cat "$BASELINE" 2>/dev/null || echo 0)
  say "Xid 计数：当前 $cur，基线 $base"
  if [ "$cur" -gt "$base" ]; then
    if [ "${PREFLIGHT_ACCEPT_XID:-0}" = "1" ]; then
      say "⚠️ PREFLIGHT_ACCEPT_XID=1，接受新基线 $cur"
      echo "$cur" > "$BASELINE"
    else
      say "设备在基线之后又报过 $((cur - base)) 次 KL_XID_KERNEL_EXCEPTION。"
      say "这通常意味着设备已经受伤（哪怕 xpu-smi 看着是 0 MiB）。"
      fail "拒绝起服务。确认设备已恢复后，用 PREFLIGHT_ACCEPT_XID=1 重跑以接受新基线。"
    fi
  fi
fi

# ---- 3) 8 进程 XCCL 探针 ----
source "$BINGOO/code/sglang-kunlun/01_eb_adp/tool/env.sh"
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_ADDR=127.0.0.1 MASTER_PORT=${PROBE_PORT:-29511}
OUT=$(mktemp /tmp/preflight_xccl.XXXXXX)

say "跑 8 进程 XCCL 探针（最多等 ${PROBE_TIMEOUT}s）…"
for r in 0 1 2 3 4 5 6 7; do
  ( env RANK=$r WORLD_SIZE=8 python -u "$HERE/xccl_probe.py" >> "$OUT" 2>&1 ) &
done

deadline=$(( $(date +%s) + PROBE_TIMEOUT ))
while :; do
  if grep -q 'all_reduce ok' "$OUT" 2>/dev/null; then
    n=$(grep -o 'all_reduce ok' "$OUT" | wc -l)
    [ "$n" -ge 8 ] && break
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    say "❌ 探针在 ${PROBE_TIMEOUT}s 内没出齐 8/8，日志：$OUT"
    say "   当前出齐 $(grep -o 'all_reduce ok' "$OUT" 2>/dev/null | wc -l)/8"
    say "   ⚠️ **不要 kill 这些进程**（它们可能正卡在 XPU 调用里，见 doc/3 开头）。"
    say "      先等；确认设备已恢复后重启节点。"
    exit 1
  fi
  sleep 3
done

say "探针：$(grep -o 'all_reduce ok' "$OUT" | wc -l)/8 通过，sum=$(grep -o 'sum=[0-9.]*' "$OUT" | sort -u | tr '\n' ' ')"

# 探针进程应自行退出；给它们一点时间回收
for _ in $(seq 1 20); do
  sleep 1
  pgrep -f "xccl[_]probe" >/dev/null 2>&1 || break
done
pgrep -f "xccl[_]probe" >/dev/null 2>&1 && say "⚠️ 仍有探针进程未退出（见 $OUT），继续之前请先确认"

free=$(xpu-smi 2>/dev/null | grep -cE '0MiB / 98304MiB')
[ "$free" -eq 8 ] || fail "探针跑完后只剩 $free/8 张卡空闲 —— 不正常"
# ---- 4) 真实算力探针 ----
# 光有 XCCL 探针不够：2026-10-17 实测设备"半坏"时，8 卡 0MiB/0% 正常、
# 8 进程 XCCL 探针 8/8 通过，但**真实计算 kernel 照样挂死**
# （`mhc_pre_big_fuse_with_norm_tilelang` 卡在 isfinite().all()，单卡 util 100%、
#  CPU 101% 空转十几分钟，且不产生新 Xid）。所以再加一个"真算"探针。
say "跑真实算力探针（8 张卡各一次，最多等 ${PROBE_TIMEOUT}s）…"
COUT=$(mktemp /tmp/preflight_compute.XXXXXX)
for d in 0 1 2 3 4 5 6 7; do
  ( env PROBE_DEVICE=$d python -u "$HERE/compute_probe.py" >> "$COUT" 2>&1 ) &
done
deadline=$(( $(date +%s) + PROBE_TIMEOUT ))
while :; do
  n=$(grep -o 'COMPUTE_OK' "$COUT" 2>/dev/null | wc -l)
  [ "$n" -ge 8 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    say "❌ 算力探针在 ${PROBE_TIMEOUT}s 内只出齐 $n/8，日志：$COUT"
    say "   设备处于"通信能通、算力不能用"的半坏状态。"
    say "   ⚠️ **不要 kill 这些进程**（它们正卡在 XPU 调用里，见 doc/3 开头）。"
    say "      按 doc/3 第五节做节点级恢复。"
    exit 1
  fi
  sleep 3
done
say "算力探针：$(grep -o 'COMPUTE_OK' "$COUT" | wc -l)/8 通过"
for _ in $(seq 1 20); do
  sleep 1
  pgrep -f "compute[_]probe" >/dev/null 2>&1 || break
done
pgrep -f "compute[_]probe" >/dev/null 2>&1 && say "⚠️ 仍有算力探针进程未退出（见 $COUT）"

free=$(xpu-smi 2>/dev/null | grep -cE '0MiB / 98304MiB')
[ "$free" -eq 8 ] || fail "探针跑完后只剩 $free/8 张卡空闲 —— 不正常"
say "✅ preflight 全过（设备空闲 + Xid 未增 + XCCL 8/8 + 算力 8/8）"
