#!/bin/bash
# EB5 / EB-Lite (Ernie Lite) 在昆仑 P800 上起服务。
#
# 参考：
#   * 昆仑侧：`01_eb_adp/tool/env.sh`（本仓库自建 venv 的环境激活脚本）
#   * EB5 侧：fork 的 `ce/scripts/eb/gpu-b/eb5_mtp_tp4dp4ep4_128k_opt.sh`
#              + `scripts/internal/launch_demo_scripts/eb5/common_eb5_env.sh`
#
# 与原始 EB5 脚本的差异（昆仑没有的东西 / 本权重是 bf16）：
#   - 本权重 `hf_checkpoint-48800` 是**纯 BF16**（无 blockwise_fp8）→
#     去掉 `--quantization blockwise_fp8`
#   - 去掉 `--fp8-gemm-backend deep_gemm` / `--moe-runner-backend deep_gemm`
#     （deep_gemm 整包缺失）
#   - 去掉 `--enable-hierarchical-cache` / `--speculative-*`（先跑通主链路，后加）
#   - `SGLANG_PLUGINS` 白名单里加 `eb5_kunlun_patch`（把 EB5 融合算子换成
#     torch 拆解 / KLX 原生算子），见 `sglang_kunlun/hooks/eb5_kunlun_patch.py`
#
# 用法：
#   bash run_eb5_kunlun.sh [MODEL_PATH] [PORT]
set -o pipefail

BINGOO=${BINGOO:-/root/paddlejob/workspace/bingoo}

# 环境：venv + 动态库搜索路径 + 设备可见性 + 昆仑运行时开关 + KLX 集合通信变量。
# **不要**再单独设 `SGLANG_PLATFORM` / `SGLANG_USE_XPU`：标准 launcher 通过 entry
# points 发现 kunlun 平台，提前设会让 sitecustomize 过早 bootstrap XPU
# （见 `01_eb_adp/tool/env.sh` 第 3 段的注释）。
source "$BINGOO/code/sglang-kunlun/01_eb_adp/tool/env.sh"

MODEL_PATH=${1:-$BINGOO/models/hf_checkpoint-48800}
PORT=${2:-30082}
# BF16 权重（每卡 ~67 GB / 可用 96 GB）下 KV cache 只剩 ~25 GB，先跑通：
# 小 context、低并发、关 CUDA graph。
MEM_FRACTION=${MEM_FRACTION:-0.90}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-32768}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-8}
EXTRA_ARGS=${EXTRA_ARGS:-}
LOG=$BINGOO/logs/eb5_kunlun_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$BINGOO/logs"

# `code/sglang`（fork）与 `code/sglang-kunlun`（插件）都是 editable 安装，
# 不再需要手工拼 PYTHONPATH。
unset PYTHONPATH

# ---- 平台 + 插件（SGLANG_PLUGINS 是白名单，设了就只加载列出的） ----
# **`kunlun` 必须列进来**：白名单对 `sglang.srt.platforms` 与 `sglang.srt.plugins`
# 两个 group 一视同仁（`load_plugins_by_group` 里统一过滤），漏了它平台插件会被跳过，
# 自动回落到 `CudaSRTPlatform`（昆仑伪装成 CUDA），`_kunlun_pre_shim` 不跑、
# `sgl_kernel` 不被 stub，随后 `sglang.srt.layers.quantization` 一挂就是
# `<kunlun-alias-stub>` 连锁失败。
export SGLANG_PLUGINS=kunlun,kunlun_hooks,eb5_internal_runtime_patch,eb5_kunlun_patch
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang.srt.internal.models.eb5
# EB5 的融合算子拆解（torch / KLX 原生）就挂在 01_eb_adp/test/ 下那三个
# `eb5_*_adapt.py`，服务侧由 eb5_kunlun_patch 从这里 import。
export EB5_KUNLUN_ADAPT_DIR=$BINGOO/code/sglang-kunlun/01_eb_adp/test

# ---- 昆仑 torch-shim 基线（沿用 DSV4 已验证的那套） ----
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=True SGLANG_OPT_USE_JIT_NORM=True
export SGLANG_OPT_USE_TOPK_V2=False SGLANG_TOPK_TRANSFORM_512_TORCH=False
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=False SGLANG_OPT_USE_TILELANG_INDEXER=False SGLANG_OPT_USE_AITER_INDEXER=False
export SGLANG_OPT_USE_FUSED_STORE_CACHE=0 SGLANG_OPT_USE_COMPRESSOR_V2=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_HACK_FLASHMLA_BACKEND=kunlun SGLANG_FIX_DSV4_BASE_MODEL_LOAD=1 SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_IS_FLASHINFER_AVAILABLE=False SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_OPT_FP8_WO_A_GEMM=False SGLANG_OPT_BF16_FP32_GEMM_ALGO="torch"
export SGLANG_PREP_IN_CUDA_GRAPH=False SGLANG_OPT_CP_REARRANGE_TRITON=False
export SGLANG_KUNLUN_INDEXER_FP32=1

# ---- EB5 侧（来自 ce/scripts/eb/gpu-b/eb5_mtp_tp4dp4ep4_128k_opt.sh） ----
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-256}
export SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER=4
export SGLANG_EB5_MLA_HCA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
export NO_PROXY="$no_proxy"

echo "[run_eb5_kunlun] model=$MODEL_PATH port=$PORT log=$LOG"

# ⚠️⚠️ 下面这段 `pkill -9` 是**有代价**的，先读 `doc/3. eb5模型运行.md` 的
# 「最重要的一条」再动它：对"卡在 XPU 调用里"的进程强杀（CPU 100% 空转、GPU 0% util、
# 显存不涨）会在设备上留下杀不掉的 kernel，拖坏整个 XPU runtime，并让驱动泄漏
# "in use" 引用计数 —— 之后 `xpu-smi -r` 对全部卡报 "currently in use by another
# process"，只能节点级重启。
# 只在确认**没有 XPU 调用在飞**时用；杀完必须确认 8 张卡都回到 `0 MiB / 0%` 再起服务。
#
# 上一次失败的 launcher 常常还活着（scheduler 死了但 tokenizer/detokenizer
# worker 不是 daemon，主进程不退出），继续占着 port_base=port+233 等派生端口，
# 下一次启动就会 "port_base is not available in 30 seconds"。更坑的是
# `sglang::tokenizer_worker:*` 会被 reparent 到 pid 1 并继续占着 HTTP 端口，
# 让下一次 uvicorn 直接 "[Errno 98] Address already in use"。两类都清掉。
if pgrep -f "sglang.launch_server|sglang::" >/dev/null 2>&1; then
  echo "[run_eb5_kunlun] killing stale sglang launcher/workers"
  pkill -9 -f "sglang.launch_server" || true
  pkill -9 -f "sglang::" || true
  sleep 5
fi

# 入口用插件自带的 `sglang-kunlun-launch`，**不要**用 `python -m sglang.launch_server`：
# 标准入口在 `load_plugins()` 里先加载 general 插件，此时平台还没解析、
# `_kunlun_pre_shim` 没跑、`sgl_kernel` 没被 stub，fork 的 `eb5_internal_runtime_patch`
# 一 import `sglang.srt.internal.models.eb5`（→ quantization → `from sgl_kernel import ...`）
# 就炸，随后 compat_aliases 把 `sglang.srt.layers.quantization` 变成
# `<kunlun-alias-stub>`，连锁失败。`sglang-kunlun-launch` 会先 `import torch`、
# 再显式调 `_kunlun_pre_shim()`，然后才 runpy 进 launch_server。
nohup sglang-kunlun-launch \
  --model-path "$MODEL_PATH" \
  --served-model-name Ernie_Lite \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --dtype bfloat16 \
  --context-length "$CONTEXT_LENGTH" \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --disable-cuda-graph \
  --swa-full-tokens-ratio 0.1 \
  --tp-size 8 --dp-size 8 --ep-size 8 \
  --enable-dp-attention \
  --disable-custom-all-reduce \
  --page-size 256 \
  --nccl-port 9999 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --tool-call-parser ernie_lite \
  --reasoning-parser ernie_lite \
  --tokenizer-worker-num 4 \
  --enable-metrics \
  $EXTRA_ARGS \
  > "$LOG" 2>&1 &

echo "[run_eb5_kunlun] pid=$! log=$LOG"
echo "[run_eb5_kunlun] 探活: curl -s http://127.0.0.1:$PORT/health_generate"
