"""在收集 EB5 kernel 单测前，先应用昆仑适配层（融合算子 -> torch 拆解）。"""
import os
import sys
import pathlib

# ── 配置 profile：让单测能跑在"服务的那套配置"下 ────────────────────────────
# 单测默认只 source `tool/env.sh`，而服务脚本 `run_eb5_kunlun.sh` 在此之外还设了
# 28 个变量，其中若干**直接决定 kernel 走哪个分支**。2026-10-17 EB5 整网首次前向
# 炸在 `mhc_pre` 的 splitk 拆解（`kernel_0`）上，而单测 82 个用例全绿，根因就是
# `SGLANG_OPT_DEEPGEMM_HC_PRENORM` 两边取值不同：
#     * 单测：没设 → 默认 True → 走 `tf32_hc_prenorm_gemm`
#     * 服务：显式 0    → 走 `kernel_0`（splitk）→ 崩
# 同一个分发器、**不同分支**，所以"单测全绿"证明不了服务那条路能跑。
# 2026-10-20 起该分发器下的融合 kernel 已由适配层在 **kernel 级**换成 torch 替代
# （`tf32_hc_prenorm_gemm` / `mhc_pre_gemm_sqrsum_*` / `mhc_pre_big_fuse*` 都被
# `eb5_kunlun_adapt.apply()` 逐个 patch），这个开关不再决定 mHC 走哪条分支；
# 保留它只为复现服务环境的其余行为。
#
# 用法：  EB5_TEST_PROFILE=service CUDA_VISIBLE_DEVICES=0 python -m pytest ... -q
# 建议**两套都跑**（默认 + service）。
#
# 这里只复现"运行期会被 kernel 读"的开关（`envs.SGLANG_*`），
# 故意**不复现** `SGLANG_PLUGINS` / `SGLANG_EXTERNAL_MODEL_PACKAGE` ——
# 那两个是启动期决定插件与模型包的，测试进程里早就 import 完了，改了反而误导。
_SERVICE_PROFILE = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    "SGLANG_DSV4_FP4_EXPERTS": "0",
    "SGLANG_EB5_MLA_HCA_DSA": "1",
    "SGLANG_FIX_DSV4_BASE_MODEL_LOAD": "1",
    "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "True",
    "SGLANG_HACK_FLASHMLA_BACKEND": "kunlun",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    "SGLANG_KUNLUN_INDEXER_FP32": "1",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_CP_REARRANGE_TRITON": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "0",   # 历史开关；kernel 已被 torch 替代，不再分叉
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    "SGLANG_OPT_USE_AITER_INDEXER": "False",
    "SGLANG_OPT_USE_COMPRESSOR_V2": "False",
    "SGLANG_OPT_USE_FUSED_STORE_CACHE": "0",
    "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "False",
    "SGLANG_OPT_USE_JIT_NORM": "True",
    "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "0",
    "SGLANG_OPT_USE_TILELANG_INDEXER": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "0",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "0",
    "SGLANG_OPT_USE_TOPK_V2": "False",
    "SGLANG_PREP_IN_CUDA_GRAPH": "False",
    "SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER": "4",
    "SGLANG_TOPK_TRANSFORM_512_TORCH": "False",
}
if os.environ.get("EB5_TEST_PROFILE") == "service":
    for _k, _v in _SERVICE_PROFILE.items():
        # 显式选择 service 时固定配置，避免继承的相反开关悄悄切换分支。
        os.environ[_k] = _v
    print(
        "[eb5-test-profile] service 配置已生效，关键开关 "
        f"SGLANG_OPT_DEEPGEMM_HC_PRENORM="
        f"{os.environ.get('SGLANG_OPT_DEEPGEMM_HC_PRENORM')}"
    )

# 用例会直接对 fp8 张量做 `.float()` / `.to(...)`，需要适配层给 `aten::_to_copy`
# 装的那层兜底。组网推理里这层是负担（见 eb5_mla_adapt._install_fp8_upcast），
# 所以默认关闭，只有单测把它打开。
os.environ.setdefault("EB5_KUNLUN_FP8_UPCAST", "1")

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    import eb5_kunlun_adapt

    print("[eb5-kunlun-adapt] patched:", eb5_kunlun_adapt.apply())
except Exception as _e:  # never block collection
    print("[eb5-kunlun-adapt] apply failed:", _e)
