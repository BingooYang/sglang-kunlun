#!/bin/bash
# DeepSeek-V4-Flash on Kunlun P800 (single node, 8 cards).
#   MODE=bringup|ll|llgraph  (llgraph = low_latency + EAGLE + CUDA graph)
#   CONDA_ENV=<conda env>    PORT=38080   MODEL_PATH=...
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-llgraph}"
PORT="${PORT:-38080}"
CONDA_ENV="${CONDA_ENV:-sglang_main}"
TAG="${TAG:-dsv4-$MODE}"
LOG_DIR="${LOG_DIR:-$HERE/logs}"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$TAG.log"; PIDFILE="$LOG_DIR/$TAG.pid"

source "$HERE/env.sh" "$CONDA_ENV"
[ "$MODE" != "bringup" ] && source "$HERE/env_ll.sh"

COMMON=(
  --model-path "$MODEL_PATH"
  --tp-size 8 --ep-size 8 --dp-size 8 --enable-dp-attention
  --port "$PORT" --host 0.0.0.0
  --nccl-port 9999
  --quantization w8a8_int8 --trust-remote-code
  --kv-cache-dtype bfloat16 --dtype bfloat16
  --disable-radix-cache
  --swa-full-tokens-ratio 0.06
  --max-running-requests 176
  --model-loader-extra-config '{"enable_multithread_load": "true", "num_threads": 64}'
  --attention-backend compressed
  --moe-a2a-backend deepep
  --served-model-name dsv4flash
)

case "$MODE" in
  bringup)
    EXTRA=(--mem-fraction-static 0.65 --deepep-mode normal --disable-cuda-graph)
    ;;
  ll)
    EXTRA=(--mem-fraction-static 0.72 --deepep-mode low_latency --disable-cuda-graph
           --speculative-algorithm EAGLE --speculative-num-steps 3
           --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    ;;
  llgraph)
    EXTRA=(--mem-fraction-static 0.72 --deepep-mode low_latency
           --cuda-graph-max-bs-decode 32
           --speculative-algorithm EAGLE --speculative-num-steps 3
           --speculative-eagle-topk 1 --speculative-num-draft-tokens 4)
    ;;
  *) echo "unknown MODE=$MODE"; exit 1;;
esac

echo "launch: MODE=$MODE PORT=$PORT CONDA_ENV=$CONDA_ENV TAG=$TAG"

# Clear leftovers safely (never kill -9 a launcher: it leaves stuck XPU kernels).
STOP="$HERE/../../01_eb/tool/stop_kunlun.sh"
if [ "${SKIP_STOP:-0}" != "1" ] && [ -f "$STOP" ]; then
    bash "$STOP" || { echo "leftover sglang processes; refusing to start"; exit 1; }
fi

nohup sglang-kunlun-launch "${COMMON[@]}" "${EXTRA[@]}" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "pid=$(cat $PIDFILE) log=$LOG"
