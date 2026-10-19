#!/usr/bin/env bash
# 服务"挂死"时的**只读**取证脚本 —— 不发任何信号，不 kill 任何进程。
#
# 为什么
# ------
# doc/3 开头那条：对卡在 XPU 调用里的进程 `kill -9` 会在设备上留 stuck kernel、
# 泄漏 "in use" 引用计数，最后只能节点级重启。挂死时唯一正确的动作是**取证 + 等**：
# XPU 调用自带的超时通常会在 8~20 分钟内让进程自行退出（doc/3 §9.1）。
#
# 本脚本收集：
#   1) 设备表 / Xid 计数（判断设备是否受伤）
#   2) sglang 进程状态与 CPU（区分"空转"和"在干活"）
#   3) 每个 scheduler 的 Python 栈（py-spy，只读）—— 直接指出卡在哪个算子
#   4) 最新服务日志尾部
#
# 用法：
#   bash tool/diag_hang.sh                 # 自动发现 sglang 进程
#   bash tool/diag_hang.sh 1234 5678       # 指定 pid
#   DIAG_TOP=12 bash tool/diag_hang.sh     # 每个栈只看前 12 帧
set -uo pipefail

BINGOO=${BINGOO:-/root/paddlejob/workspace/bingoo}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIAG_TOP=${DIAG_TOP:-14}
PY_SPY=${PY_SPY:-$BINGOO/env/bin/py-spy}

say() { echo "===== $* ====="; }

say "时间"
date

say "设备（0MiB/0% 只说明没占显存，不代表算力可用；见 doc/3 §9.2）"
xpu-smi 2>/dev/null | grep -E 'MiB / 98304MiB|XPU  Name' || echo "xpu-smi 不可用"

say "Xid（KL_XID_KERNEL_EXCEPTION 总数 / 最近 3 条）"
dmesg -T 2>/dev/null | grep -c 'KL_XID_KERNEL_EXCEPTION' || true
dmesg -T 2>/dev/null | grep 'KL_XID_KERNEL_EXCEPTION' | tail -3 || true

say "相关进程（R + 高 CPU 且日志不增长 = 空转挂死；S 在 futex 多为等待）"
ps -eo pid,ppid,stat,pcpu,etime,args \
  | grep -E '[s]glang[-]kunlun-launch|[s]glang[:][:]|[x]ccl_probe|[c]ompute_probe' \
  | cut -c1-160

if [ "$#" -gt 0 ]; then
  PIDS=("$@")
else
  mapfile -t PIDS < <(pgrep -f 'sglang[:][:]scheduler' 2>/dev/null)
fi

if [ "${#PIDS[@]}" -eq 0 ]; then
  say "未发现 scheduler 进程（服务可能已自行退出）"
else
  if [ ! -x "$PY_SPY" ]; then
    say "未找到 py-spy（$PY_SPY），跳过栈采样"
  else
    for p in "${PIDS[@]}"; do
      [ -d "/proc/$p" ] || continue
      say "py-spy dump pid=$p（只读；前 $DIAG_TOP 帧）"
      timeout 40 "$PY_SPY" dump --pid "$p" --nonblocking 2>&1 | head -n "$DIAG_TOP"
    done
  fi
fi

say "最新服务日志尾部"
LOG=$(ls -t "$BINGOO"/logs/eb5_kunlun_*.log 2>/dev/null | head -1)
if [ -n "${LOG:-}" ]; then
  echo "log: $LOG"
  tail -n 15 "$LOG"
else
  echo "未找到 logs/eb5_kunlun_*.log"
fi

say "提醒"
cat <<'EOF'
* 不要 kill 卡在 XPU 调用里的进程（doc/3 开头）。等它自行超时（约 8~20 分钟）。
* `timeout` / `timeout -k` 都不是护栏（doc/3 §9.4）。
* 若 Xid 增长或算力探针挂死 → 设备已受伤，按 doc/3 第五节做节点级恢复。
EOF
