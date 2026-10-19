#!/usr/bin/env bash
# 安全停掉昆仑上的 sglang 服务（launcher + 它的 worker），供 run_*.sh 起服务前调用。
#
# 用法：  bash stop_kunlun.sh      # 或 source 后调 stop_kunlun
#
# ── 为什么不能直接 `pkill -9 -f "sglang.launch_server|sglang::"` ──────────────
# 两个真实的坑（2026-10-17 各踩过一次）：
#
#  1) **会打到调用者自己**。`pkill -f` 匹配的是整条命令行，而调用它的那条命令
#     本身往往就含这些字符串，于是把自己（甚至父 shell）一起杀掉，脚本半途而废。
#     所以本脚本一律用字符类（`sglang[-]kunlun-launch`、`sglang[:][:]`）书写模式，
#     让它匹配不到含字面量的自身命令行。
#
#  2) **对 launcher 用 -9 是 doc/3「最重要的一条」明令禁止的**。所以这里分级处理：
#       a. 先给 launcher 发 TERM+INT（sglang 装了优雅退出 handler）；
#       b. 但 launcher 主线程通常卡在 `waitpid` 等子进程，而 `sglang::tokenizer_worker`
#          / `sglang::detokenizer` **不响应 TERM/INT**，handler 永远走不完 ——
#          这就是"杀了也退不掉"的真因；
#       c. 于是只对**这两类纯 CPU、且处于空闲等待**的 worker 用 -9，解开 waitpid 死锁
#          （它们按设计只做分词/反分词，不发任何 XPU 调用；实测状态为 S + ep_poll/
#          futex_wait，`xpu-smi -q` 也不登记它们 —— 杀它们不会留下 stuck kernel）；
#       d. **launcher 本身一次 -9 都不发**：子进程被回收后它会在几秒内自己走完 handler；
#       e. 最后校验 8 张卡都回到 `0 MiB / 0%`，不满足就报错退出交给人看。
set -uo pipefail

PAT_LAUNCH='sglang[-]kunlun-launch'
PAT_WORKER='sglang[:][:]'
# 允许 -9 的角色：纯 CPU 的分词/反分词 worker
PAT_CPU_WORKER='sglang[:][:]tokenizer_worker|sglang[:][:]detoken'

say() { echo "[stop_kunlun] $*"; }

alive() { pgrep -f "$PAT_LAUNCH" >/dev/null 2>&1 || pgrep -f "$PAT_WORKER" >/dev/null 2>&1; }

if ! alive; then
  say "没有在跑的 sglang 进程"
  exit 0
fi

say "发现残留："
pgrep -af "$PAT_LAUNCH|$PAT_WORKER" | sed 's/^/  /' | cut -c1-120

# a) 优雅信号
for p in $(pgrep -f "$PAT_LAUNCH" 2>/dev/null); do
  kill -TERM "$p" 2>/dev/null && say "TERM -> $p"
  kill -INT  "$p" 2>/dev/null
done

# b) 给优雅退出一点时间
for _ in $(seq 1 6); do
  sleep 5
  pgrep -f "$PAT_LAUNCH" >/dev/null 2>&1 || { say "launcher 已优雅退出"; break; }
done

# c) 还活着：只 -9 空闲的纯 CPU worker，解开 waitpid 死锁
if pgrep -f "$PAT_LAUNCH" >/dev/null 2>&1; then
  say "launcher 仍在（多半卡在 waitpid 等 worker），只对空闲的纯 CPU worker 用 -9"
  for p in $(pgrep -f "$PAT_CPU_WORKER" 2>/dev/null); do
    st=$(awk '{print $3}' /proc/"$p"/stat 2>/dev/null)
    wchan=$(cat /proc/"$p"/wchan 2>/dev/null)
    case "$st:$wchan" in
      Z:*) say "  $p 已是僵尸，跳过" ;;
      S:ep_poll|S:do_epoll_wait|S:futex_wait_queue_me|S:poll_schedule_timeout)
        kill -9 "$p" 2>/dev/null && say "  -9 -> $p（空闲 CPU worker，状态 $st/$wchan）" ;;
      *) say "  ⚠️ $p 状态 $st/$wchan 不像空闲 worker，**跳过**（不冒险）" ;;
    esac
  done
fi

# d) 等 launcher 自己退出
for _ in $(seq 1 24); do
  sleep 5
  pgrep -f "$PAT_LAUNCH" >/dev/null 2>&1 || break
done

if alive; then
  say "❌ 仍有进程未退出："
  pgrep -af "$PAT_LAUNCH|$PAT_WORKER" | sed 's/^/  /' | cut -c1-120
  say "不要对 launcher 用 -9（doc/3：会留 stuck kernel）。请人工确认后再处理。"
  exit 1
fi

free=$(xpu-smi 2>/dev/null | grep -cE '0MiB / 98304MiB')
say "设备：$free/8 张卡空闲（0 MiB）"
if [ "$free" -ne 8 ]; then
  say "❌ 有卡没回到 0 MiB —— 先别起下一个任务，按 doc/3 第五节处理"
  exit 1
fi
say "✅ 已清干净"
