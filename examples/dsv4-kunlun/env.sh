#!/bin/bash
# Kunlun P800 runtime environment for SGLang / DeepSeek-V4.
# Usage: source examples/dsv4-kunlun/env.sh [conda-env-name]
__ENV_NAME="${1:-sglang_main}"

# Prefer a conda env when installed; otherwise use the workspace venv ($W/env).
if [ -f "${CONDA_ROOT:-/root/miniconda3}/etc/profile.d/conda.sh" ]; then
    export CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$__ENV_NAME"
else
    _DSV4_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
    while [ "$_DSV4_DIR" != / ] && [ ! -x "$_DSV4_DIR/env/bin/python" ]; do
        _DSV4_DIR=$(dirname "$_DSV4_DIR")
    done
    export BINGOO=${BINGOO:-$_DSV4_DIR}
    unset _DSV4_DIR
    export VIRTUAL_ENV="$BINGOO/env"
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    export LD_LIBRARY_PATH="$VIRTUAL_ENV/klx_stub${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

unset PYTHONPATH
unset XPU_DUMMY_EVENT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XPU_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export CUDA_DEVICE_ORDER=OAM_ID

# Platform plugin
# Do not preset SGLANG_PLATFORM: sglang-kunlun-launch imports torch and runs
# _kunlun_pre_shim() before the platform is resolved.
export HYDRAX_SKIP_PROTEUS_BOOTSTRAP=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# eb5_adapt overlay: the DSV4/ERNIE adaptations (incl. the DP-idle / short-prompt
# zero-length guards in eb5_xy_adapt) are applied by the eb5_kunlun_patch entry
# point. SGLANG_PLUGINS is a whitelist, so every needed plugin must be listed;
# `kunlun` must appear or the platform silently falls back to CUDA.
export SGLANG_PLUGINS="${SGLANG_PLUGINS:-kunlun,kunlun_hooks,eb5_kunlun_patch}"

# KLX / XCCL runtime
export XSHMEM_MODE=1
export BKCL_ENABLE_IBGDA=1
export BKCL_RDMA_VERBS=1
export XSHMEM_QP_NUM_PER_RANK=32
export CUDA_ENABLE_P2P_NO_UVA=1
export BKCL_RDMA_NICS="${BKCL_RDMA_NICS:-eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4}"
export BKCL_ENABLE_XDR=1
export NCCL_IB_GID_INDEX=3

# Kernel-path switches validated on P800
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=True
export SGLANG_OPT_USE_JIT_NORM=True
export SGLANG_OPT_USE_TOPK_V2=False
export SGLANG_TOPK_TRANSFORM_512_TORCH=False
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=False
export SGLANG_OPT_USE_TILELANG_INDEXER=False
export SGLANG_OPT_USE_AITER_INDEXER=False
export SGLANG_OPT_USE_FUSED_STORE_CACHE=0
export SGLANG_OPT_USE_COMPRESSOR_V2=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_HACK_FLASHMLA_BACKEND=kunlun
export SGLANG_FIX_DSV4_BASE_MODEL_LOAD=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DSV4_FP4_EXPERTS=0
export SGLANG_TOOL_STRICT_LEVEL=2
export SGLANG_KUNLUN_INDEXER_FP32=1
export SGLANG_OPT_FP8_WO_A_GEMM=False
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_PREP_IN_CUDA_GRAPH=False
export SGLANG_OPT_CP_REARRANGE_TRITON=False

export LD_LIBRARY_PATH="${CONDA_PREFIX:-$VIRTUAL_ENV}/lib/python3.10/site-packages/torch/lib:${CONDA_PREFIX:-$VIRTUAL_ENV}/lib/python3.10/site-packages/torch_xmlir:${CONDA_PREFIX:-$VIRTUAL_ENV}/xcudart/lib:${CONDA_PREFIX:-$VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1

export MODEL_PATH="${MODEL_PATH:-${BINGOO:-/root/paddlejob/gpfsspace/bingoo/p800_eb}/models/DeepSeek-V4-Flash-FP8-W8A8-INT8-Dynamic}"
