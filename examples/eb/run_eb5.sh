#!/bin/bash
# 一键在昆仑 P800 上跑 EB5 / ERNIE 5.5 Flash（离线 INT8 权重）。
#
# 环境变量和启动参数都在这一个文件里，不依赖 tool/ 下的其它脚本。
#
# 用法：
#   bash run_eb5.sh                                   # 全用默认值，8 卡
#   bash run_eb5.sh --gpus 0,1,2,3
#   bash run_eb5.sh --model <权重绝对路径> --port 30201
#   bash run_eb5.sh --sglang <sglang 的 python 目录> --kunlun <sglang-kunlun 仓库>
#   bash run_eb5.sh -- --speculative-algorithm EAGLE  # `--` 之后原样传给 launch
#
# 四条路径都有默认值（见下面的 ENV_DIR / MODEL / SGLANG_DIR / KUNLUN_DIR），
# 全部写成绝对路径，不依赖脚本位置和当前目录。
#
# 服务跑在前台，日志直接打到终端；要后台常驻就自己接管：
#   bash run_eb5.sh > eb5.log 2>&1 &
#
# 如果上一次运行残留了 XPU 上下文，本次会在初始化阶段失败；
# 先杀掉持有 /dev/xpu* 的进程，再 `xpu-smi -fr -i <卡号>`。
set -uo pipefail

# ---------------------------------------------------------------- 路径
# venv：装了 torch_xmlir / sglang / sglang_kunlun 的那个环境
ENV_DIR=/root/paddlejob/gpfsspace/bingoo/p800_eb/env
# 权重：自带 int8 routed experts + w{1,2,3}.weight_scale 的 checkpoint
MODEL=/root/paddlejob/gpfsspace/bingoo/p800_eb/models/eblite-int8-moe
# sglang：公版树 + ERNIE 5.5 Flash（EB5 做进了主干 models/ernie55.py）
SGLANG_DIR=/root/paddlejob/gpfsspace/bingoo/p800_eb/code/wenbin/sglang-ERNIE-5.5-Flash/python
# sglang-kunlun：昆仑平台插件 + EB5 适配层（仓库根目录，含 sglang_kunlun/ 包）
KUNLUN_DIR=/root/paddlejob/gpfsspace/bingoo/p800_eb/code/xy/sglang-kunlun

# ---------------------------------------------------------------- 其它默认值
GPUS=0,1,2,3,4,5,6,7
CONTEXT=8192
MEM=0.75
PORT=30200
# `--` 之后的参数原样追加到 sglang-kunlun-launch，用来做单变量实验
# （例如 -- --speculative-algorithm EAGLE ...），不用改这个脚本。
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --env)     ENV_DIR=$2; shift 2;;
    --model)   MODEL=$2; shift 2;;
    --sglang)  SGLANG_DIR=$2; shift 2;;
    --kunlun)  KUNLUN_DIR=$2; shift 2;;
    --gpus)    GPUS=$2; shift 2;;
    --context) CONTEXT=$2; shift 2;;
    --mem)     MEM=$2; shift 2;;
    --port)    PORT=$2; shift 2;;
    --)        shift; EXTRA+=("$@"); break;;
    -h|--help) sed -n '2,14p' "$0"; exit 0;;
    *) echo "未知参数：$1（-h 看用法）"; exit 1;;
  esac
done
for d in "$ENV_DIR/bin/python" "$MODEL/config.json" "$SGLANG_DIR/sglang" \
         "$KUNLUN_DIR/sglang_kunlun"; do
  [ -e "$d" ] || { echo "路径不存在：$d"; exit 1; }
done
NGPU=$(awk -F, '{print NF}' <<<"$GPUS")

# ---------------------------------------------------------------- 环境
export VIRTUAL_ENV="$ENV_DIR"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHONHOME
# 本机基础 CPython 是静态编译的，torch_plugin 的导入钩子却要 dlopen
# libpython3.so；klx_stub/ 里是空实现占位库。
export LD_LIBRARY_PATH="$VIRTUAL_ENV/klx_stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
unset XPU_DUMMY_EVENT
export XPU_VISIBLE_DEVICES=$GPUS CUDA_VISIBLE_DEVICES=$GPUS CUDA_DEVICE_ORDER=OAM_ID

# 昆仑运行时开关
export SGLANG_IS_FLASHINFER_AVAILABLE=False SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_SET_CPU_AFFINITY=1 XMLIR_FORCE_USE_XPU_GRAPH=1 XPU_USE_FAST_SWIGLU=1
export XPU_USE_DEFAULT_CTX=1 XMLIR_ENABLE_FAST_FC=1 XMLIR_CUDNN_ENABLED=1
export CUDA_GRAPH_OPTIMIZE_STREAM=1 HYDRAX_SKIP_PROTEUS_BOOTSTRAP=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

# 集合通信 / DeepEP（缺 XSHMEM_MODE 时对称缓冲区会落到主机内存）
export BKCL_RDMA_VERBS=1 XSHMEM_QP_NUM_PER_RANK=32 BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS=${BKCL_RDMA_NICS:-eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4}
export NCCL_IB_GID_INDEX=3 CUDA_ENABLE_P2P_NO_UVA=1
export XSHMEM_MODE=1 XSHMEM_SYMMETRIC_SIZE=4294967296

# 昆仑 torch-shim 基线：这些融合算子在 XPU 上没有实现，走 torch / KLX 原生拆解。
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=True SGLANG_OPT_USE_JIT_NORM=True
export SGLANG_OPT_USE_TOPK_V2=False SGLANG_TOPK_TRANSFORM_512_TORCH=False
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=False SGLANG_OPT_USE_TILELANG_INDEXER=False
export SGLANG_OPT_USE_AITER_INDEXER=False SGLANG_OPT_USE_FUSED_STORE_CACHE=0
export SGLANG_OPT_USE_COMPRESSOR_V2=False SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0 SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_HACK_FLASHMLA_BACKEND=kunlun SGLANG_FIX_DSV4_BASE_MODEL_LOAD=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_OPT_FP8_WO_A_GEMM=False SGLANG_OPT_BF16_FP32_GEMM_ALGO="torch"
export SGLANG_PREP_IN_CUDA_GRAPH=False SGLANG_OPT_CP_REARRANGE_TRITON=False
export SGLANG_KUNLUN_INDEXER_FP32=1 EB5_DSA_META_GRAPH_PATCH=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER=4 SGLANG_EB5_MLA_HCA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# 两条分开写：同一个 export 里的赋值会先全部展开再生效，set -u 下
# NO_PROXY="$no_proxy" 会读到还没赋值的 no_proxy。
export no_proxy="127.0.0.1,localhost${no_proxy:+,${no_proxy}}"
export NO_PROXY="$no_proxy"

# 平台插件白名单。kunlun 必须列进来：漏了它会回落到 CudaSRTPlatform，
# sgl_kernel 不被 stub，随后 quantization 连锁失败。
export SGLANG_PLUGINS=kunlun,kunlun_hooks,eb5_internal_runtime_patch,eb5_kunlun_patch
# 公版树没有 srt/internal/，外部模型包必须置空，否则 import 直接失败。
export SGLANG_EXTERNAL_MODEL_PACKAGE=""
# sglang 在前：它要顶掉 venv 里 editable 安装的那一份。
export PYTHONPATH="$SGLANG_DIR:$KUNLUN_DIR"
# checkpoint 自带 int8 routed experts + w{1,2,3}.weight_scale，不要再做在线量化。
export EB5_INT8_ONLINE=0

# ---------------------------------------------------------------- 起服务
# 入口必须用 sglang-kunlun-launch，不能用 python -m sglang.launch_server：
# 标准入口加载 general 插件时平台还没解析，sgl_kernel 没被 stub，一 import
# quantization 就连锁失败。这个入口会先 import torch 再跑 _kunlun_pre_shim()。
exec sglang-kunlun-launch \
  --model-path "$MODEL" \
  --served-model-name Ernie_Lite \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code \
  --dtype bfloat16 \
  --quantization w8a8_int8 \
  --context-length "$CONTEXT" \
  --kv-cache-dtype bfloat16 \
  --mem-fraction-static "$MEM" \
  --max-running-requests 8 \
  --disable-cuda-graph \
  --swa-full-tokens-ratio 0.1 \
  --tp-size "$NGPU" --dp-size "$NGPU" --ep-size "$NGPU" \
  --enable-dp-attention \
  --disable-custom-all-reduce \
  --page-size 256 \
  --nccl-port 9999 \
  --moe-a2a-backend deepep --deepep-mode normal \
  --dsa-topk-backend torch \
  --tool-call-parser ernie5 --reasoning-parser ernie5 \
  --tokenizer-worker-num 4 \
  --enable-metrics \
  "${EXTRA[@]}"
