#!/bin/bash
# DeepSeek-V4-Flash (W8A8 INT8) on the Kunlun env built by `doc/2. 构建环境.md`.
# Config follows the validated recipe in doc (ku: DSv4 Flash Sglang-kunlun),
# standalone variant (no PD disaggregation).
# Usage: DSV4_GPUS=0,1,5,6 bash run_dsv4_newenv.sh [port]
#
# 设备列表可用 `DSV4_GPUS` 覆盖（默认 0-7 全用）。TP/EP/DP 都按卡的张数自动取。
set -o pipefail

BINGOO=${BINGOO:-/root/paddlejob/workspace/bingoo}
DSV4_GPUS=${DSV4_GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(awk -F, '{print NF}' <<<"$DSV4_GPUS")

# 环境：venv + 动态库搜索路径 + 设备可见性 + 昆仑运行时开关 + KLX 集合通信变量。
# **不要**再单独设 `SGLANG_PLATFORM` / `SGLANG_USE_XPU`：标准 launcher 通过 entry
# points 发现 kunlun 平台，提前设会让 sitecustomize 过早 bootstrap XPU
# （见 `01_eb_adp/tool/env.sh` 第 3 段的注释）。
source "$BINGOO/code/sglang-kunlun/01_eb_adp/tool/env.sh"

# 设备可见性：env.sh 是按 XPU_VISIBLE_DEVICES 派生的，这里按 DSV4_GPUS 覆盖。
export XPU_VISIBLE_DEVICES=$DSV4_GPUS
export CUDA_VISIBLE_DEVICES=$DSV4_GPUS

PORT=${1:-8806}
MODEL_DIR=$BINGOO/models/DeepSeek-V4-Flash-FP8-W8A8-INT8-Dynamic
LOG=$BINGOO/logs/dsv4_newenv_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$BINGOO/logs"

# `code/sglang`（fork）与 `code/sglang-kunlun`（插件）都是 editable 安装，
# 不再需要手工拼 PYTHONPATH。
unset PYTHONPATH

# ---- validated torch-shim base (verbatim from the doc) ----
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=True SGLANG_OPT_USE_JIT_NORM=True
export SGLANG_OPT_USE_TOPK_V2=False SGLANG_TOPK_TRANSFORM_512_TORCH=False
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=False SGLANG_OPT_USE_TILELANG_INDEXER=False SGLANG_OPT_USE_AITER_INDEXER=False
export SGLANG_OPT_USE_FUSED_STORE_CACHE=0 SGLANG_OPT_USE_COMPRESSOR_V2=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0 SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_HACK_FLASHMLA_BACKEND=kunlun SGLANG_FIX_DSV4_BASE_MODEL_LOAD=1 SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_TOOL_STRICT_LEVEL=2
export SGLANG_KUNLUN_INDEXER_FP32=1
export SGLANG_OPT_FP8_WO_A_GEMM=False SGLANG_OPT_BF16_FP32_GEMM_ALGO="torch"
export SGLANG_PREP_IN_CUDA_GRAPH=False SGLANG_OPT_CP_REARRANGE_TRITON=False
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=3600 OPS_DEBUG_CHECK=1
export PYTHONUNBUFFERED=1
# decode extras (EAGLE + spec v2)
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_ENABLE_SPEC_V2=True SGLANG_OPT_V4_DRAFT_EXTEND_CUDA_GRAPH=True SGLANG_ENABLE_OVERLAP_PLAN_STREAM=True

# ---- XPU profiler / CPU-affinity patch (the doc runs this before launch) ----
python "$BINGOO/code/sglang-kunlun/sglang_kunlun/hooks/utils/sglang_xpu_patch.py" >/dev/null 2>&1 || true

# 让插件的 `eb5_kunlun_patch` 找到适配层（torch 拆解 + 几个机器级 kernel 绕行）。
# 对 DSV4 也是必需的：`eplb/expert_location.py::_init_raw` 里那处 3 维 int64 常量
# `F.pad`（底层 `constant_pad1d_ncw<long>`）在本机 XPU 上会 task timeout，随后
# 整个 session 被 taint，紧接着的 `.cpu()` 永久挂住 —— 表现为所有 rank 卡死在
# `_init_raw`。适配层的 `_install_expert_location_pad` 用 `torch.cat` 绕开它。
export EB5_KUNLUN_ADAPT_DIR=$BINGOO/code/sglang-kunlun/01_eb_adp/test

# ⚠️⚠️ 下面这段 `pkill -9` 是**有代价**的，先读 `doc/3. eb5模型运行.md` 的
# 「最重要的一条」再动它：对"卡在 XPU 调用里"的进程强杀（CPU 100% 空转、GPU 0% util、
# 显存不涨）会在设备上留下杀不掉的 kernel，拖坏整个 XPU runtime，并让驱动泄漏
# "in use" 引用计数 —— 之后 `xpu-smi -r` 对全部卡报 "currently in use by another
# process"，只能节点级重启。
# 只在确认**没有 XPU 调用在飞**时用；杀完必须确认 8 张卡都回到 `0 MiB / 0%` 再起服务。
#
# 残留 launcher / worker 会占着 port_base 与 HTTP 端口，先清掉。
if pgrep -f "sglang.launch_server|sglang::" >/dev/null 2>&1; then
  echo "[run_dsv4_newenv] killing stale sglang launcher/workers"
  pkill -9 -f "sglang.launch_server" || true
  pkill -9 -f "sglang::" || true
  sleep 5
fi

echo "[run_dsv4_newenv] gpus=$DSV4_GPUS (n=$NGPU) port=$PORT log=$LOG"

# 入口用插件自带的 `sglang-kunlun-launch`，**不要**用 `python -m sglang.launch_server`：
# 标准入口在 `load_plugins()` 里先加载 general 插件，此时平台还没解析、
# `_kunlun_pre_shim` 没跑、`sgl_kernel` 没被 stub，fork 的插件一 import
# `sglang.srt.internal.models.*`（→ quantization → `from sgl_kernel import ...`）
# 就炸，随后 compat_aliases 把 `sglang.srt.layers.quantization` 变成
# `<kunlun-alias-stub>`，连锁失败。`sglang-kunlun-launch` 会先 `import torch`、
# 再显式调 `_kunlun_pre_shim()`，然后才 runpy 进 launch_server。
nohup sglang-kunlun-launch \
  --model-path "$MODEL_DIR" \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 --port "$PORT" \
  --nccl-port 9998 \
  --tp "$NGPU" --ep "$NGPU" --dp "$NGPU" --enable-dp-attention \
  --quantization w8a8_int8 --trust-remote-code \
  --kv-cache-dtype float16 --dtype float16 \
  --mem-fraction-static "${MEM_FRACTION:-0.85}" --max-running-requests "${MAX_REQ:-$((NGPU * 22))}" \
  --swa-full-tokens-ratio 0.06 \
  --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
  --enable-metrics \
  --attention-backend compressed \
  --allow-auto-truncate --moe-a2a-backend deepep --deepep-mode low_latency \
  --tool-call-parser deepseekv4 \
  --tokenizer-worker-num 4 \
  --speculative-algorithm EAGLE --speculative-num-draft-tokens 2 \
  --speculative-num-steps 1 --speculative-eagle-topk 1 --cuda-graph-max-bs-decode 22 \
  > "$LOG" 2>&1 &

echo "[run_dsv4_newenv] pid=$! log=$LOG"
echo "[run_dsv4_newenv] 探活: curl -s http://127.0.0.1:$PORT/health_generate"
