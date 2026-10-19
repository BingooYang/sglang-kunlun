# EB Lite 算子单测支持报告（昆仑 P800）

> 🔴 **跑单测/起服务前必读**：单测卡死时**不要 `kill -9`**。对"卡在 XPU 调用里"的进程
> （CPU 100% 空转、GPU `0% util`、显存不涨）强杀，会在设备上留下**杀不掉的 kernel**
> （`util=100%` 但 `Processes: None`），拖坏整个 XPU runtime（之后连
> `torch.ones` + `sync` 都间歇性挂死），并让驱动**泄漏 "in use" 引用计数** ——
> `xpu-smi -r` 会对全部 8 张卡报 `is currently in use by another process`，
> **只能节点级重启**。
>
> 正确做法：① 先等它自己超时；② 必须杀时，杀完确认 8 张卡都回到 `0 MiB / 0%` 再跑下一个；
> ③ 用 `CUDA_VISIBLE_DEVICES=<单卡>` 跑单模块，避免整机 8 卡同时上。
> 完整时间线与最小复现见 **`doc/3. eb5模型运行.md`**。

日期：2026-10-13
环境：`env/kunlun_py`（公网构建的 torch29 昆仑环境；**该环境已废弃，第九节是在新环境下的复跑**）+ `code/sglang-kunlun`（OOT 插件）+ `code/sglang`（文心内部 fork，`d4a840ae2d`）
被测：`test_eb5_*.py`（8 个模块，staging 到 `<repo>/test_internal/internal/models/eb5/layer/`；源文件在 `01_eb_adp/test/`，用 `patches/sglang-eb5-layer.patch` 落地）
原始日志：`logs/eb5_sweep_*.log`（含 traceback）、`logs/eb5_outcomes_*.log`（逐用例结果）

---

## 一、结论摘要

**84 个用例 → 17 通过 / 65 失败 / 2 跳过。**

通过的 17 个全部是**不碰 triton、不碰 JIT、或走早退分支**的路径（ATen `RMSNorm`、cuBLASLt 选核的 GEMM、模块级 0-token/空输入早退）。

三个主阻塞（按影响面排序）：

| 类 | 影响 | 一句话根因 |
|---|---|---|
| **T1 triton kernel 编出来跑不了** | 24 用例 | 新环境用的是**公网 stock triton 3.5.0（只有 nvidia/amd 后端）**，缺 KLX 的 **XPU triton 后端**（镜像里那份是 `triton 3.0.0+51cb5a00`，带 `backends/xpu`）。插件把 target 强制成 `backend="cuda"` → triton 用 nvidia 后端编出 **sm80 cubin** → 交给 XPU 驱动加载报 `CUDA_ERROR_NOT_SUPPORTED` |
| **T2 fork 自带 JIT 链建不起来** | 16 用例 | fork 的 `sglang.kernels.jit` 用 ninja+nvcc，build file 里写死 `nvcc = /usr/local/cuda/bin/nvcc`，宿主**没有 `/usr/local/cuda`** → `Failed to build JIT module sgl_kernel_jit_*` |
| **T4 deep_gemm 缺失/不完整** | 5 用例 | `ENABLE_JIT_DEEPGEMM=False`（昆仑伪装 CUDA 且报 sm_86<90）→ `entrypoint` 从不 import `deep_gemm`，而 mhc 路径无条件调用 → `NameError`；另有 `module 'deep_gemm' has no attribute 'bf16_mqa_logits'` |

次要：T5 tilelang 缺失（2）、T6 flash_mla API 错位（1）。

> 构建过程中还修掉了插件自身的一个 bug（T0）：`pre_shim._CudaTarget` 只设了 `backend="cuda"`，而 triton 3.x 断言 `isinstance(target, GPUTarget)` 且读 `target.arch` → 修成返回 `GPUTarget("cuda", 80, 32)` 后，约 30 个用例从 `'arch'` 报错推进到 T1（暴露出更深的 XPU 问题）。

---

## 一·补：为什么跑通的 DSV4 不受影响？（**DSV4 完全没用 triton**）

一个自然的问题：同一套新环境，DSV4 为什么能跑？答案是**跑通的 DSV4 版本根本没走 triton**，证据：

1. **triton 编译缓存里没有 DSV4 的痕迹**：DSV4 那次运行（20:58）前后（20:50–21:10）新增缓存条目 **0 个**；缓存里只有 11:0x–15:5x（旧环境）和 21:16–21:25（本次 EB5）两批。
2. DSV4 日志里 `Failed to build JIT` **0 条**、`CUDA_ERROR_NOT_SUPPORTED` **0 条**。
3. **配置本身就在刻意关掉 triton/JIT 路径**（文档那套 env）：`SGLANG_OPT_USE_JIT_INDEXER_METADATA=False`、`USE_TOPK_V2=False`、`USE_FUSED_STORE_CACHE=0`、`USE_COMPRESSOR_V2=False`、`USE_TILELANG_INDEXER/AITER_INDEXER=False`、`SGLANG_OPT_CP_REARRANGE_TRITON=False`…；剩下的交给插件换成 **KLX 原生算子**（`xspeedgate_ops` / `kunlun_ops` / `xtorch_ops`）。
4. 注意力后端是插件提供的 **`kunlun_dsv4`**（KLX 原生），不是 triton 后端。

**这正好解释了 DSV4 能跑、EB5 不能的差别**：DSV4 靠"关掉 triton + 换 KLX 原生算子"绕过去；而 **EB5 是 triton 重度依赖**（HCA / MLA / moe / 前后处理几乎每个 kernel 都是 triton 或 fork 自带 JIT），没有等价的 KLX 原生替换。

对照 EB5：triton 内核**确实编译成功**（缓存里有 `_init_compressed_attn_metadata_kernel.cubin` 等 41 条，target = `sm80`），但**加载时被 XPU 驱动拒绝** → `CUDA_ERROR_NOT_SUPPORTED`。即 T1：公网 triton 没有昆仑后端，编出来的是 NVIDIA cubin。

> 注：`xpu_fla/python/triton_mock.py` 那个"real triton backend usable (cuda-compat)"探测，判据是 `driver.active.get_current_target() is not None`；而插件恰好把 `get_current_target` 强制成自己的 target，所以这个探测恒为"可用"，**不能**用来判断 triton 在 XPU 上真能用。

---

## 二、逐模块结果

| 模块 | 通过/总数 | 通过的是 |
|---|---|---|
| 前处理 `test_eb5_preprocess.py` | **0/8** | — |
| hc_pre `test_eb5_hc_pre.py` | **0/4** | — |
| hc_post `test_eb5_hc_post.py` | **0/2** | — |
| 后处理 `test_eb5_postprocess.py` | **1/4** | `nvjet_sm100_tst_192x8_...` |
| moe `test_eb5_moe.py` | **2/8**（+2 skip） | `RMSNorm_2048`、`nvjet_sm100_tst_64x8_...` |
| attn ② MLA `test_eb5_attn_mla.py` | **7/23** | `RMSNorm_1024/512`、4×`nvjet_sm100_tst_*`、`cublasLt_splitKreduce` |
| attn ① HCA `test_eb5_attn_hca.py` | **1/14** | `RMSNorm_1024` |
| 模块级 `test_eb5_layer_modules.py` | **6/19** | 3×前处理、`mhc_pre_empty_batch`、`mhc_post_empty_batch`、`zero_token_early_exit` |

---

## 三、逐算子支持矩阵

图例：✅通过 ❌失败 ｜ 阻塞类见第四节

### 前处理（0/8）

| kernel | 结果 | 类 |
|---|---|---|
| `compute_position_kernel` | ❌ | T1 |
| `_expand_prefill_causally_kernel` | ❌ | T1 |
| `_page_table_positions_kernel` | ❌ | T1 |
| `_causal_swa_page_indices_kernel` | ❌ | T1 |
| `_init_compressed_attn_metadata_kernel` | ❌ | T1 |
| `topk_plan` | ❌ | T2 |
| `plan_compress_prefill_kernel_1` | ❌ | T2 |
| `get_mla_metadata_kernel` | ❌ | T6 |

### hc_pre / hc_post（0/6）

| kernel | 结果 | 类 |
|---|---|---|
| `sm100_tf32_hc_prenorm_gemm_impl` | ❌ | T4 |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | ❌ | T4 |
| `mhc_post_tilelang_kernel`（attn / moe 各 1） | ❌ | T5 |

### attn ① HCA（1/14）

| kernel | 结果 | 类 |
|---|---|---|
| `RMSNorm<1024>` | ✅ | — |
| `per_token_group_quant_flat_kernel` | ❌ | T2 |
| `deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl` | ❌ | T2 |
| `fused_q_norm_rope` | ❌ | T2 |
| `fused_k_norm_rope_flashmla` | ❌ | T2 |
| `nvjet_sm100_tss_64x8_64x16_1x4_h_bz_TNT` | ❌ | T1 |
| `write_c128_prefill` | ❌ | T2 |
| `append_indexer_swa_kernel` | ❌ | T1 |
| `get_k_and_s_triton_kernel` | ❌ | T1 |
| `dequantize_k_cache_paged_kernel` | ❌ | T1 |
| `sparse_attn_fwd_kernel` | ❌ | T1 |
| `deepseek_rope_kernel` | ❌ | T2 |
| `postmix_quant_kernel` | ❌ | T1 |
| `eb5_sigmoid_gate_mul_kernel` | ❌ | T1 |

### attn ② MLA（7/23）

| kernel | 结果 | 类 |
|---|---|---|
| `RMSNorm<1024>` / `RMSNorm<512>` | ✅ | — |
| `nvjet_sm100_tst_*`（4 个几何） | ✅ | — |
| `cublasLt::splitKreduce_kernel` | ✅ | — |
| `deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl` | ❌ | T2 |
| `per_token_group_quant_flat_kernel` | ❌ | T2 |
| `sm100_mqa_logits` | ❌ | T4 |
| `topk_main_kernel` | ❌ | T2 |
| `fused_q_indexer_rope_first_quant` | ❌ | T2 |
| `fused_k_indexer_norm_rope_store` | ❌ | T2 |
| `concat_mla_absorb_q_kernel` | ❌ | T2 |
| `fused_rope_kernel` | ❌ | T2 |
| `set_mla_kv_buffer_kernel` | ❌ | T2 |
| `quantize_k_cache_fast_kernel` | ❌ | T1 |
| `append_indexer_swa_kernel` | ❌ | T1 |
| `get_k_and_s_triton_kernel` | ❌ | T1 |
| `flash_fwd_splitkv_mla_fp8_sparse_kernel` | ❌ | T1 |
| `flash_fwd_mla_combine_kernel` | ❌ | T1 |
| `eb5_sigmoid_gate_mul_kernel` | ❌ | T1 |
| `get_mla_metadata_kernel` | ❌ | T6 |

### moe（2/8 + 2 skip）

| kernel | 结果 | 类 |
|---|---|---|
| `RMSNorm<2048>` | ✅ | — |
| `nvjet_sm100_tst_64x8_64x16_1x4_h_bz_TNT` | ✅ | — |
| `MEGA_MOE` / `all_reduce_1shot_push_kernel` | SKIP | 需多 rank / symm buffer |
| `deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl` | ❌ | T2 |
| `per_token_group_quant_flat_kernel` | ❌ | T2 |
| `mega_moe_pre_dispatch_kernel` | ❌ | T2 |
| `situ_and_mul_kernel` | ❌ | T2 |
| `eb5_router_topk_bitonic_kernel` | ❌ | T1 |
| `memcpy_triton_kernel` | ❌ | T1 |

### 后处理（1/4）

| kernel | 结果 | 类 |
|---|---|---|
| `nvjet_sm100_tst_192x8_64x8_2x1_v_bz_TNT` | ✅ | — |
| `alloc_decode_kernel` | ❌ | T1 |
| `hc_head_kernel_opt` | ❌ | T1 |
| `multimax_selu_kernel` | ❌ | T1 |

### 模块级（6/19）

| 用例 | 结果 | 类 |
|---|---|---|
| `TestPreprocess::test_add_then_norm_matches_reference` / `test_norm_without_residual_returns_tensor` / `test_empty_input_passthrough` | ✅ | — |
| `TestHcPre::test_mhc_pre_empty_batch` / `TestHcPost::test_mhc_post_empty_batch` / `TestMoe::test_zero_token_early_exit` | ✅ | — |
| `TestHcPre::test_mhc_pre_shapes` / `test_mhc_pre_matches_torch_reference` | ❌ | T4 |
| `TestHcPost::test_mhc_post_matches_torch_reference` | ❌ | T5 |
| `TestHca::test_forward_swa_layer` / `_compressed_layer` / `_gated_layer` / `_chunked_prefill_prefix` / `_decode` | ❌ | T1 |
| `TestMla::test_forward_absorbed_mla_decode` / `test_forward_gated_mla` | ❌ | T1 |
| `TestMoe::test_forward_runs_expert_kernel` / `test_fused_split_feature_router_topk` | ❌ | T1 |
| `TestPostprocess::test_fused_multimax_selu_runs_and_is_finite` | ❌ | T1 |

---

## 四、阻塞分类

### T1 —— triton kernel 编出来跑不了（24 用例，**最硬**）

- 现象：`RuntimeError: Triton Error [CUDA]: CUDA_ERROR_NOT_SUPPORTED`（triton 编译**成功**，加载 cubin 时被 XPU 驱动拒绝）
- 根因：**新环境缺 KLX 的 XPU triton 后端**
  - 新环境 `triton 3.5.0`（公网 pip，`backends/` 只有 `amd`、`nvidia`）
  - 镜像里是 `triton 3.0.0+51cb5a00`（`backends/` 有 `xpu`、`xcn`）
  - 插件的 `_kunlun_pre_shim` 把 triton target 强制成 `backend="cuda"`（本意是让 triton 走 CUDA 兼容路径），于是 nvidia 后端编出 sm80 cubin，XPU 跑不了
- 性质：**环境缺口**（不是 EB5 代码问题）。EB5 是 triton 重度依赖，这一类不解决，绝大部分 kernel 跑不了
- 解决方向：拿到与 torch29 配套的 KLX XPU triton（内部渠道 / 镜像），或让 KLX 提供 torch29 版 `triton/backends/xpu`

### T2 —— fork 自带 `sglang.kernels.jit`（ninja+nvcc）建不起来（16 用例）

- 现象：`RuntimeError: Failed to build JIT module sgl_kernel_jit_*`；build file 里 `nvcc = /usr/local/cuda/bin/nvcc`
- 根因：宿主没有 `/usr/local/cuda`（无 CUDA toolkit）
- 注意：即使补上 nvcc，编出来仍是 **NVIDIA cubin（sm80/sm86）**，"给 XPU 跑"方向本身存疑 —— 这与旧报告（P5）结论一致
- 解决方向：需要 KLX 侧的编译方案（KLX 的 `xtdk` LLVM 工具链 / 昆仑 nvcc），而不是装个公网 CUDA

### T4 —— deep_gemm 缺失 / 不完整（5 用例）

- 现象 A：`NameError: name 'deep_gemm' is not defined`（hc_pre 的 prenorm GEMM）
  - `deep_gemm_wrapper/entrypoint.py` 只在 `if ENABLE_JIT_DEEPGEMM:` 里 import；`ENABLE_JIT_DEEPGEMM` 由 `configurer._compute_enable_deep_gemm()` 决定，昆仑报 **sm_86 < 90 → False**，于是 `deep_gemm` 从未绑定，而 mhc 路径无条件调用它
- 现象 B：`module 'deep_gemm' has no attribute 'bf16_mqa_logits'`
- 性质：依赖缺失 + 版本错位（KLX 的 deep_gemm ≠ fork 期望的）
- 已有绕法（DSV4 在用）：`SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`

### T5 —— tilelang 缺失（2 用例）

- 现象：`NotImplementedError: [tilelang-stub] tilelang kernel 'tilelang.jit(mhc_post_tilelang)' cannot run on Kunlun P800`
- 根因：`kernels/ops/layernorm/mhc.py` 的 `mhc_pre()/mhc_post()` 函数体内**没有非 tilelang 分支**
- 已有绕法：`SGLANG_OPT_USE_TILELANG_MHC_PRE=0` / `..._POST=0` → 走 `_mhc_pre_torch` / `_mhc_post_torch`（纯 torch 参考实现）；KLX 自己的 mHC 走 `xtorch_ops.mhc_post_fusion`

### T6 —— flash_mla API 错位（1 用例）

- 现象：`AttributeError: 'FlashMLASchedMeta' object has no attribute 'dim'`
- 根因：KLX 的 `sgl_kernel.flash_mla` 返回对象与 fork 期望的字段不一致（旧报告 P6 是参数个数不匹配，这里又前进了一步、变成字段名不匹配）

---

## 五、融合算子 → dpskv4 拆解方案

插件 `sglang_kunlun/kernels/dsv4_torch_shim.py` 里已经有一整套 **dpsk-v4 的融合算子拆解实现**（纯 torch 参考 + KLX-native 两条），可以直接复用到 EB5 的融合算子上：

| EB5 融合算子（失败） | dpskv4 拆解方案 | 实现位置 |
|---|---|---|
| `fused_q_norm_rope` | rmsnorm-self（无 weight）+ 末 `rope_dim` 维 RoPE | `_fused_q_norm_rope_torch` |
| `fused_k_norm_rope_flashmla` | rmsnorm（带 `kv_weight`）+ RoPE + 写 FlashMLA paged cache | `_fused_k_norm_rope_flashmla_torch`（torch）/ `_fused_k_norm_rope_flashmla_kunlun`（KLX `dpsk_v4_norm_rope_gptj`） |
| `fused_rope_kernel` / `deepseek_rope_kernel` | in-place GPT-J RoPE | `_fused_rope_inplace_torch` / `_fused_rope_inplace_kunlun`（`xspeedgate_ops.flashinfer_rotary_embedding`） |
| `fused_q_indexer_rope_first_quant` | 末 64 维 RoPE + 128 点 Hadamard + INT8 量化 | `_fused_q_indexer_rope_hadamard_quant_torch` / `_kunlun` |
| `write_c128_prefill` / `plan_compress_prefill_kernel_1` / `compress_plan` | softmax-pool compress + plan 生成（写/读 compress buffer） | `_compress_forward_c128_torch`、`_make_prefill_plan_torch`、`_compress_norm_rope_store_torch` |
| `mhc_pre_*` / `mhc_post_*` | 纯 torch 参考（或 KLX `xtorch_ops.mhc_post_fusion`） | `SGLANG_OPT_USE_TILELANG_MHC_PRE/POST=0` 走 `_mhc_pre_torch`/`_mhc_post_torch` |
| `silu_and_mul` / `situ_and_mul_kernel` | 纯 torch（`SiluAndMul.forward_native`） | 插件 `sglang.jit_kernel.activation::silu_and_mul` |

**关键提醒（这次实测发现的接不上问题）**：插件的这些 override 是用 `register_jit_op` 注册到 **`sglang.jit_kernel.*`**（上游 sglang 0.5.14 的 JIT API）；而**文心 fork 的调用点是 `sglang.kernels.ops.*`**（另一套 JIT 框架）。两者**模块路径不同**，所以插件的拆解实现目前**并没有被 fork 的 kernel 调用到**。要复用，需要把 `dsv4_torch_shim.py` 里那些 `_*_torch` / `xspeedgate` 实现**接到 fork 的 `sglang.kernels.ops.*` 调用点上**（这属于"把插件从上游 sglang 适配到 fork"的工作）。

---

## 六、建议与下一步

1. **T1 优先**：向 KLX 索取 torch29 版 XPU triton（`triton/backends/xpu`）。这是 EB5 能否跑通的决定性一项 —— 24 个用例、且模块级 forward 全挂在它上面。
2. **T2**：确认 KLX 对 torch29 的 JIT 编译方案（是否有昆仑 nvcc / 是否应改走 XPU triton 而不是 nvcc）。可先给宿主补 `/usr/local/cuda` 做一次对照实验，但预期只能推进、不能解决。
3. **T4/T5**：短期用开关绕（`SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`、`SGLANG_OPT_USE_TILELANG_MHC_PRE/POST=0`）；长期找 KLX/fork 对齐 deep_gemm 与 tilelang 版本。
4. **接拆解方案**：把 `dsv4_torch_shim.py` 的拆解实现从 `sglang.jit_kernel.*` 适配到 fork 的 `sglang.kernels.ops.*`（见第五节表）。
5. **端到端**：EB Lite 的模型权重（`eb_lite_dsa_hca` / `eb_lite_mla_hca`）本机仍没有，真正的组网验证需要先拿到权重。

---

## 七、昆仑适配进展（参考 dpsk-v4 的融合算子拆解）

按 dpsk-v4 的做法，把 EB5 单测**直接调用**、在昆仑上跑不了的融合 kernel 换成**纯 torch 拆解实现**：

- 适配层：`code/eb5_layer/eb5_kunlun_adapt.py`（`apply()` 把拆解实现 patch 到 fork 的调用点）
- 注入：`<repo>/test_internal/internal/models/eb5/layer/conftest.py`（pytest 收集前自动 `apply()`）

已实现的拆解：

| kernel | 拆解 | 依据 |
|---|---|---|
| `mhc_post_tilelang` | `c[...,None]*d[:,None,:] + einsum('nhc,nhd->ncd')` | kernel 源码 + 单测参考 |
| `mhc_pre_big_fuse_with_norm_tilelang` | rms → mixes → sigmoid post_mix → sinkhorn comb → rmsnorm layer_input | kernel 源码逐行 |
| `tf32_hc_prenorm_gemm` | split-K 的 `x@fn.T` + `sum(x²)` | 单测参考 |
| `fused_q_norm_rope` | RMSNorm(no weight) + 末尾 `rope_dim` 交错 RoPE | 单测参考 |
| `fused_k_norm_rope_flashmla` / `fused_rope_inplace` / `fused_q_indexer_rope_hadamard_quant` | 复用插件 `dsv4_torch_shim` 的 `_*_torch`（签名一致） | 插件 |

**结果：17 通过 → 29 通过**（失败 65 → 53）：

| 模块 | 适配前 | 适配后 |
|---|---|---|
| hc_pre | 0/4 | **4/4** |
| hc_post | 0/2 | **2/2** |
| attn ① HCA | 1/14 | 3/14 |
| attn ② MLA | 7/23 | 8/23 |
| 模块级 | 6/19 | 9/19 |
| **合计** | **17/84** | **29/84** |

踩到的坑（都已在适配层里处理）：
1. 插件 `sglang_kunlun.kernels` 包的 `__init__` 会拉起只对 sglang 0.5.14 成立的 import 链（`sglang.srt.layers.quantization.fp8_kernel` 在 fork 里已改名）→ 只能**按文件直接加载** `dsv4_torch_shim.py`，并给它唯一缺的 `act_quant_kunlun` 打桩。
2. 复用插件的 `_fused_q_norm_rope_torch` 数值对不上（RoPE 约定不同）→ 改为**按单测参考**自己实现。

剩余：T1（triton，32 个）需 KLX XPU triton 或逐个补拆解；T2 其余（quant / topk / compress / store / concat_mla / situ_and_mul 等）逐个补 torch 实现；T4/T5/T6 同前。

---

## 附：复现方式

第八节那套 `env/kunlun_py` 旧环境已随工作区搬迁而消失，现在按**第九节**跑：

```bash
# staging（用归档 patch，别手抄）
git -C /root/paddlejob/workspace/bingoo/code/sglang apply \
    /root/paddlejob/workspace/bingoo/patches/sglang-eb5-layer.patch

source /root/paddlejob/workspace/bingoo/code/sglang-kunlun/01_eb_adp/tool/env.sh
cd /root/paddlejob/workspace/bingoo/code/sglang
CUDA_VISIBLE_DEVICES=0 python -m pytest test_internal/internal/models/eb5/layer -q
# -> 82 passed, 2 skipped
```

---

## 八、续：**8 个文件 / 84 用例全部通过**（2026-10-15）

在第四节的基础上继续适配，最终 **82 passed / 2 skipped / 0 failed**（连跑 3 次稳定）。
2 个 skip 是需要多 rank 的对称内存用例，非失败。

| 文件 | 基线 | 现在 |
|---|---|---|
| `test_eb5_preprocess.py` | 0/8 | **8 passed** |
| `test_eb5_hc_pre.py` | 0/4 | **4 passed** |
| `test_eb5_hc_post.py` | 0/2 | **2 passed** |
| `test_eb5_postprocess.py` | 1/4 | **4 passed** |
| `test_eb5_moe.py` | 2/8 | **8 passed + 2 skipped** |
| `test_eb5_attn_mla.py` | 7/23 | **23 passed** |
| `test_eb5_attn_hca.py` | 1/14 | **14 passed** |
| `test_eb5_layer_modules.py` | 6/19 | **19 passed** |
| **合计** | **17/84** | **82 passed / 2 skipped / 0 failed** |

> ⚠️ 口径：所有"通过"都是**在 torch 拆解（或 KLX 原生算子）之上**的语义可跑通，
> **不等于原 CUDA/triton kernel 在昆仑上可跑**。真跑原生仍需 KLX 的 XPU triton 后端 /
> nvcc 工具链 / deep_gemm。

### 适配层（3 个文件，`code/eb5_layer/`，需 copy 到 staged 目录）

- `eb5_kunlun_adapt.py` —— hc_pre / hc_post / 融合 rope；`apply()` 依次调用下两个
- `eb5_mla_adapt.py` —— attn② MLA 全部 kernel + triton MLA 后端 + 运行时补丁
- `eb5_extra_adapt.py` —— HCA / MoE / 前后处理

### 逐轮推进记录

| 轮次 | 修掉 | 累计 |
|---|---|---|
| 1 | MLA kernel 级 16 个拆解 + `aten::_to_copy`(fp8 upcast) / `uint8.abs` / `torch.equal` 三个运行时补丁 | 8 → 23 |
| 2 | HCA / MoE / 前后处理 kernel 级 | 17 → 74 |
| 3 | `flash_mla_with_kvcache` 的 **SWA KV 布局**（656 vs 584 + 页跨距 576 对齐） | +1 |
| 4 | `flash_mla_sparse_fwd`、`compress_forward`、`moe_align_block_size` + `invoke_fused_moe_kernel` + `silu_and_mul` | +5 |
| 5 | **triton MLA 后端三件套**：`get_num_kv_splits_triton`、`apply_rope_inplace`/`rotary_embedding`、`decode_attention_fwd` | +2 → 82 |

### 三个关键教训

**① 内核单测 ≠ 模块路径。** 同一概念在 fork 里常有 2~3 个并行实现，内核单测挑一个、模块
forward 走另一个：

| 概念 | 内核单测调的 | 模块 forward 调的 |
|---|---|---|
| HCA 稀疏 prefill | `paged_prefill.py::sparse_attn_v4_paged_prefill` | `sgl_kernel.flash_mla.flash_mla_sparse_fwd` |
| MLA 注意力 | `sgl_kernel.flash_mla.flash_mla_with_kvcache` | `decode_attention.py::decode_attention_fwd` |
| MoE | `moe/moe_fused_gate.py::moe_fused_gate` | `fused_moe.py`（align + grouped GEMM + 激活） |

补模块路径要顺着 `test_eb5_layer_modules.py` 的 traceback **逐帧**找。

**② SWA KV 缓冲布局坑。** `get_swa_key_buffer_radix` 返回 `(num_pages, page_size, 1, 584)`
/ stride `(149760, 584, 584, 1)`，看着像每 token 584 B；但
`store.cuh::fused_store_flashmla_cache` 实际写的是页内 **`[page_size×576 的 value]
[page_size×8 的 scale]`**（token 跨距 **576**），页跨距 =
`ceil_div(584*page_size,576)*576`。按 584 线性读会每 token 漂 8 字节、跨页后完全错位。
正确做法：`as_strided` 取回真实字节流，再按 `page*page_stride + off*576` 寻址。

**③ `setattr` 定义模块不够。** fork 里同一符号常被多处 `from x import f` /
`from x import f as _g` 在 import 期绑死（`sglang.srt.models.deepseek_v4` 持有
`fused_q_norm_rope` 的旧引用，且在 conftest 之前就已 import）。必须再扫一遍所有
`sglang.*` 模块，按**对象身份**重指旧绑定。

### 昆仑 torch 运行时缺口（已打补丁绕过）

1. 设备侧 **fp8 → float 的 cast 完全没实现**（`xdnn cast.cpp: UnknownScalarType`）→ 给
   `aten::_to_copy` 注册 CUDA-key 实现，fp8 源走 256 项 LUT 查表。
2. `uint8.abs()` 不支持 → Python 层 `Tensor.abs`，uint8 直接 `clone()`。
3. `torch.equal(int32, int64)` 不提升 dtype → 先提升再比。
4. `index_select` 对 fp8 / uint8 都不支持 → 改用高级索引。

---

## 九、在新环境（`doc/2 构建环境.md` 那套 venv）复跑（2026-10-17）

工作区从 `/root/paddlejob/bingoo` 搬到 `/root/paddlejob/workspace/bingoo`，环境也从
`env/kunlun_py`（公网镜像）换成本仓库自建的 venv（用 `tool/env.sh` 激活）。
复跑结果**与第八节完全一致**：**82 passed / 2 skipped / 0 failed**（连跑 3 次稳定）。

| 文件 | 第八节 | 本次 |
|---|---|---|
| `test_eb5_preprocess.py` | 8 | 8 |
| `test_eb5_hc_pre.py` | 4 | 4 |
| `test_eb5_hc_post.py` | 2 | 2 |
| `test_eb5_postprocess.py` | 4 | 4 |
| `test_eb5_moe.py` | 8 + 2 skip | 8 + 2 skip |
| `test_eb5_attn_mla.py` | 23 | 23 |
| `test_eb5_attn_hca.py` | 14 | 14 |
| `test_eb5_layer_modules.py` | 19 | 19 |
| **合计** | **82 passed / 2 skipped** | **82 passed / 2 skipped** |

### 新环境比旧环境缺的三样东西

`doc/2` 的验证只到 import / 平台激活 / `ServerArgs` 构造，**没覆盖整条量化 import 链**；
跑这套单测才暴露（前两个已补，第三个未修）：

1. **缺 `httpx`** —— `sglang.test.test_utils` 经 `run_eval` 顶层 import 它，而
   `_eb5_kernel_test_common.py` 要 `CustomTestCase`。缺了 **8 个文件全部 collect 失败**。
   补：`pip install -c pkg/constraints.txt httpx`。
2. **缺 `flashinfer_python==0.6.18`**（fork `python/pyproject.toml` 的 pin，`--no-deps` 时被跳过）
   —— 缺了 `sglang.srt.layers.quantization.fp8_utils` 顶层 `from flashinfer import bmm_fp8`
   直接 ImportError、`initialize_model_parallel()` 一挂，同样 **8 个文件全 collect 失败**。
   - 内网源上的 `flashinfer` 是 1KB 假包，**必须装 `flashinfer_python`**。
   - 它会带出 `nvidia-cutlass-dsl`，后者在本机驱动上报 `driver supports up to CUDA 10.2`
     → `import flashinfer` 直接 RuntimeError，装完卸掉 `nvidia-cutlass-dsl*` / `cuda-tile` /
     `cuda-core` 即可。
3. **适配层写死了旧工作区路径（代码里的真 bug，未修）** —— `eb5_kunlun_adapt.py:265` /
   `eb5_extra_adapt.py:1234` 用绝对路径
   `/root/paddlejob/bingoo/code/sglang-kunlun/sglang_kunlun/kernels/dsv4_torch_shim.py`
   加载插件 shim。工作区搬家后加载失败 → `fused_k_norm_rope_flashmla` / `fused_rope_inplace` /
   `fused_q_indexer_rope_hadamard_quant` / `compress_forward` 这些拆解**不生效** →
   **8 个用例回落到 fork 的 nvcc JIT（宿主无 `/usr/local/cuda`）而失败**（未修时 74 passed / 8 failed）：

   | 失败用例 | 回落的 JIT module |
   |---|---|
   | `attn_hca::test_fused_k_norm_rope_flashmla`、`layer_modules::TestHca::test_forward_{swa,gated,compressed,chunked_prefill_prefix,decode}` | `sgl_kernel_jit_dpsk_v4_main_k_norm_rope_flashmla_bf16_*` |
   | `attn_hca::test_deepseek_rope_kernel`、`attn_mla::test_fused_rope_kernel` | `sgl_kernel_jit_dpsk_v4_fused_rope_false` |

   - 临时兜法：`ln -sfn /root/paddlejob/workspace/bingoo /root/paddlejob/bingoo`
     （顺带修好 `run_eb5_kunlun.sh` 与插件 `EB5_KUNLUN_ADAPT_DIR` 默认值里的旧路径）。
   - 根治：两处改成按 `importlib.util.find_spec("sglang_kunlun")` 的包位置解析。

### 注意：新环境**不要**设 `SGLANG_PLATFORM`

旧环境要 `export SGLANG_PLATFORM=kunlun SGLANG_USE_XPU=1`；新环境**不能设**：
会让 sitecustomize 过早 bootstrap XPU，且 `_kunlun_pre_shim` 与 `compat_aliases` 打架，
报 `cannot import name 'QUANTIZATION_METHODS' from ... <kunlun-alias-stub>`，8 个文件全 collect 失败。
`tool/env.sh` 的注释已写明。

---

## 十、复跑前置：staging 与两个注意点（2026-10-17 晚）

第九节说"用归档 patch，别手抄"：

```bash
git -C code/sglang apply patches/sglang-eb5-layer.patch
```

但**工作区里已经没有 `patches/` 目录**（`patches/sglang-eb5-layer.patch` 丢失）。
而 staging 位置不能随便放 —— `_eb5_kernel_test_common.py` 靠这两行定位：

```python
_REPO_ROOT = _HERE.parents[5]                 # <repo>/test_internal/internal/models/eb5/layer/xxx.py
_PYTHON_DIR = _REPO_ROOT / "python"
for _p in (str(_PYTHON_DIR), str(_HERE.parent.parent)):
    sys.path.insert(0, _p)
```

即 **repo 根**和 **`eb5/` 目录**（`_hf_strict_compat.py` 在那儿）。
所以文件**必须**落在 `code/sglang/test_internal/internal/models/eb5/layer/`；
放在 `01_eb_adp/test/` 下会报 `No module named '_hf_strict_compat'`
（**是位置问题，不是环境问题** —— 排查时别往环境上找）。

手工 staging（等价于那个 patch）：

```bash
SRC=code/sglang-kunlun/01_eb_adp/test
DST=code/sglang/test_internal/internal/models/eb5/layer
mkdir -p "$DST"
cp "$SRC"/_eb5_kernel_test_common.py "$SRC"/eb5_kunlun_adapt.py \
   "$SRC"/eb5_extra_adapt.py "$SRC"/eb5_mla_adapt.py "$SRC"/conftest.py \
   "$SRC"/test_eb5_*.py "$DST"/
```

跑法（**单卡**，逐模块，避免整机 8 卡同时上）：

```bash
source code/sglang-kunlun/01_eb_adp/tool/env.sh
cd code/sglang
CUDA_VISIBLE_DEVICES=0 python -m pytest test_internal/internal/models/eb5/layer -q
```

实测与第九节一致：

| 文件 | 用例 |
|---|---|
| `test_eb5_preprocess.py` | 8 |
| `test_eb5_hc_pre.py` | 4 |
| `test_eb5_hc_post.py` | 2 |
| `test_eb5_postprocess.py` | 4 |
| `test_eb5_moe.py` | 8 + 2 skip |
| `test_eb5_attn_mla.py` | 23 |
| `test_eb5_attn_hca.py` | 14 |
| `test_eb5_layer_modules.py` | 19 |
| **合计** | **82 passed / 2 skipped / 0 failed** |

跑之前建议先确认设备干净（`xpu-smi | grep -c '0MiB / 98304MiB'` = 8）；
卡死时**不要 `kill -9`**（见本文开头与 `doc/3`）。

### 两个注意点

1. **改了 `01_eb_adp/test/eb5_*_adapt.py` 一定要同步 staging 副本** —— 单测 import 的是
   staging 那份（`conftest.py` 里 `import eb5_kunlun_adapt`）。不同步会出现"改了源码但单测
   行为没变"的错觉，白排查一轮。
2. **`act_quant_kunlun` 那个桩是 DSV4 的雷**。`eb5_kunlun_adapt.py` 原来把它打成一个抛异常的
   桩，理由是"只有 `*_kunlun` 变体会用到"；但 **DSV4 的 indexer 量化路径会真的调到它**
   （`fused_q_indexer_rope_hadamard_quant → dsv4_torch_shim.py:642`），表现为第一次前向
   `NotImplementedError`。详见 `doc/3` 第七节 7.2①。

   改这里时**不要**用 `importlib.import_module` 去复用插件的真实现：会提前 import fork 的
   attention 模块，让本模块 `_TARGETS` 的补丁被已绑定的旧符号绕过，**单测 19 passed → 4 failed**
   （`dequant_k_cache.py:68 TypeError: 'function' object is not subscriptable`）。
   要用自包含实现（`kunlun_ops.quant2d`）。

---

## 十一、单测补强：为什么"82 全绿"挡不住整网炸（2026-10-17 深夜）

第十节记了复跑方法。这一节记**为什么那 82 个用例全绿、整网却炸**，以及为此补了什么。

### 11.1 根因：单测和整网**走了同一个分发器的不同分支**

`sglang/kernels/ops/layernorm/mhc.py:1119` 按环境变量分支：

```python
if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
    tf32_hc_prenorm_gemm(...)                              # ← 单测走这条
else:
    kernel_0, _ = mhc_pre_gemm_sqrsum_splitk_kernel(...)   # ← 服务走这条（崩在这）
```

而 `SGLANG_OPT_DEEPGEMM_HC_PRENORM` 的**默认值是 `True`**（`srt/environ.py:1485`）：

| 上下文 | 取值 | 走哪条 |
|---|---|---|
| 单测（只 source `tool/env.sh`） | 未设 → **默认 True** | `tf32_hc_prenorm_gemm` |
| 服务（`run_eb5_kunlun.sh:62`） | 显式 **0** | `kernel_0`（splitk） |

而且 `run_eb5_kunlun.sh` 比 `env.sh` 多设 **28 个**变量，而 kernel/model 热路径里有
**65 处** `envs.SGLANG_*.get()` 分支 —— 单测等于在**另一套配置**下验证。
"单测全绿"证明的是另一条路能跑。

### 11.2 补了什么

**(1) 配置 profile** —— `conftest.py`

```bash
EB5_TEST_PROFILE=service CUDA_VISIBLE_DEVICES=0 python -m pytest <layer_dir> -q
```

把服务那套**运行期**开关复现一遍（26 个；故意**不含** `SGLANG_PLUGINS` /
`SGLANG_EXTERNAL_MODEL_PACKAGE` —— 那两个是启动期决定插件与模型包的，测试进程里早就
import 完了，改了反而误导）。**建议两套都跑**：默认 + `service`。

**(2) sqrsum / splitk 家族用例** —— `test_eb5_hc_pre.py::TestHcPreSqrsumFamily`

这一族原来**零覆盖**：`kernel_0`、`mhc_pre_gemm_sqrsum_tilelang`、
`_mhc_pre_gemm_sqrsum_dispatch` 都**不在** ku 表的第三方 kernel 清单里，
而覆盖是按那张表枚举的。新增：

- `test_mhc_pre_gemm_sqrsum_splitk_kernel` —— 真实配置（hc_mult=4 / hidden=4096 →
  hc_hidden=16384、hc_mult3=24、split_k=16）+ token 数扫 `8 / 256 / 2048 / 2049`
  （覆盖 `mhc.py` 的 2048 分发阈值两侧），与 fp32 参考逐项对齐
- `test_mhc_pre_gemm_sqrsum_tilelang` —— 2-D 调用约定（`out=(N, hc_mult3)`）
- `test_mhc_pre_gemm_sqrsum_dispatch` —— 返回单链实现
- `test_kernel_0_matmul_contiguity_contrast` —— **对照实验**：`kernel_0` 里两个 matmul
  操作数原本**都是非连续切片**（`xs` stride `(H,1)`、`f[:j,lo:hi].t()` stride `(1,H)`）。
  先跑 `.contiguous()` 形式、再跑非连续形式 —— **顺序是刻意的**：非连续若触发设备级异常
  会把 XPU 上下文带坏，把干净的那条放前面，结果才可判定
  （都过 → 与连续性无关；连续过/非连续炸 → 坐实 strided matmul；连续就炸 → 不是（唯一）原因）

> 用例从 **`sglang.kernels.ops.layernorm.mhc`** 取符号（适配层补丁**替换后**的那个），
> 而不是直接 import `eb5_kunlun_adapt` —— 这样测的就是模型真正会调到的对象。

配套还改了 `eb5_kunlun_adapt.py::kernel_0`：显式 `.contiguous()` 后再做 matmul。
**数值不变**（CPU 上两种写法已逐元素对照过），只是把它拉回常规路径；
仓库里 `dsv4_torch_shim.py` 一贯这么写（那里 20+ 处 `.contiguous()`，这个文件原先只有 2 处）。
**注：这个改动在设备恢复前无法验证**（见 11.3）。

**(3) 覆盖自检** —— `test_eb5_adapt_coverage.py`

解析适配层三个文件里"被替换的符号"，要求**每个都有用例提到**，否则 fail ——
把"悄悄漏一个"变成"跑不过"。现状：28 个符号、**23 个被提到（82%）**，
剩下 5 个在 `_KNOWN_UNCOVERED` 里**带原因**登记（另有一条用例挡住清单腐烂）。

**这个文件刻意不依赖 XPU**（不 import 那个在 import 阶段就 `init_process_group(backend="nccl")`
的 `_eb5_kernel_test_common`），所以**设备不健康时也能单跑**，用来快速确认覆盖没退化。

### 11.3 一个重要的边界

单测保证的是"**拆解算得对** + **分支可达**"，**不保证**整网能跑：

- **规模/形状**：单测用 `_TOKENS = 4 / 8`，整网是真实 token 数；
- **多卡集合通信**：单测 `num_gpus=1`、`world_size=1`、不 init XCCL，
  而 rank 0/1/2 的 D2H 挂死在原理上就覆盖不到 —— 那个只能靠 8 进程 XCCL 探针兜底
  （`tool/xccl_probe.py` + `tool/preflight.sh`）；
- **设备健康**：设备"半坏"时**单测自己会挂死**，那不是用例的问题
  （`doc/3` 第九节：`0MiB/0%` + XCCL 8/8 都可能全绿，但真实计算 kernel 照样挂）。
  起服务/跑单测前先过 `tool/preflight.sh`。

---

## 十二、又三个盲区：split_k / align_fleet / 小 token（2026-10-19）

第十一节解决了"分支不同"（`SGLANG_OPT_DEEPGEMM_HC_PRENORM`），但 2026-10-19 在
4 卡 8 层子模型上复现整网挂死时，发现**同一分支内还有三个参数差异**，单测一个都没覆盖。

### 12.1 `split_k`：单测 16，服务是 32

`ernie5.EB5DecoderLayer._mhc_pre` 调 `mhc.mhc_pre` 时**不传 `n_splits_pre`**，
于是走签名默认值 **32**（`kernels/ops/layernorm/mhc.py`）：

```python
def mhc_pre(..., n_splits_pre: int = 32, ...):
    ...
    deterministic = _is_deterministic_inference_enabled()
    if deterministic:
        n_splits_pre = 16          # 只有确定性推理才改成 16
```

而 `test_eb5_hc_pre.py::TestHcPreSqrsumFamily.SPLIT_K` 原来写死 **16**。
16 与 32 是**两条不同的 kernel 形状**（`split_size = 16384/16 = 1024` vs `512`，
且 `partial_out` 形状不同），所以"16 全绿"证明不了 32 能跑。

**已改**：`SPLIT_K = 32`（与默认值一致），并新增
`test_mhc_pre_gemm_sqrsum_splitk_kernel_all_splits` 同时覆盖 **16 与 32**。

### 12.2 `mhc_align_fleet`：单测 False，服务是 True

服务端按 `model_type` 设置：
`self.mhc_align_fleet = "ernie" in model_type.lower()`，本权重 `model_type="ernie"`
→ **True**。它切换两处算法：

* mapping norm 用 `1/(sqrt(sum_sq/D)+eps)`（eps 在 sqrt **外**）；
* sinkhorn softmax 用 `exp2((x-max)*LOG2E)` 而非 `exp(x-max)`。

而适配层的层测试一律写 `layer.mhc_align_fleet = False`（HF/DSV4 口径）。

**已改**：新增 `test_mhc_pre_align_fleet_matches_hf_form`，直接按 `mhc_align_fleet=True`
跑 `mhc_pre_big_fuse_with_norm_tilelang`，校验有限性与 comb 列和。

### 12.3 token 数：单测从 8 起，warmup 是 3

服务 warmup 的请求是（`entrypoints/http_server.py::_execute_server_warmup`）：

```python
json_data["input_ids"] = [[10, 11, 12] for _ in range(get_parallel().dp_size)]
```

即 **每个 DP rank 3 token**。而 `TOKEN_SWEEP = (8, 256, 2048, 2049)` 的最小值是 8。
`n=1..7` 从未被测过。

### 12.4 单卡单测原理上抓不到的一类问题：D2H 同步

2026-10-19 定位到的两个真实 bug 都属于这一类：

* `eb5_extra_adapt.py::memcpy_triton` 把 **device 标量** `offset`/`sz` 用 `int()` 拉回主机；
* `fused_split_feature_topk` 同样把 `num_token_non_padded` 用 `int()` 拉回主机。

原始 triton kernel 都是把这两个值当 **device 指针**在 kernel 内 `tl.load`，**没有 D2H**。
在本机 XPU 上，这种标量 D2H 会让所有 rank 一起挂死（8~10 分钟后进程自行 `exit -3`）。

> **单测为什么抓不到**：单进程 `world_size=1`、不 init XCCL、`num_gpus=1`，
> 这些标量同步在单卡上并不必然挂；只有整网（多 rank + DP attention + 真实调度）
> 才会稳定复现。所以这条只能靠**整网复现**兜底，属于第十一节 11.3 那条边界的延伸。

**排查口径**：适配层里出现 `int(x)` / `.item()` / `.cpu()` / `.tolist()`，
先确认 `x` 是不是 device tensor。当前三个适配文件已逐个核对调用点，
其余都是 Python int（`extend_seq_lens_sum`、`num_tokens`、`num_seq` 等）。

### 12.5 当前单测状态

在 **4/5/6/7** 四张卡上、`EB5_TEST_PROFILE=service` 跑 `test_eb5_hc_pre.py`：

```
10 passed, 18 subtests passed     # 含新增的 all_splits 与 align_fleet
```

完整 `layer/` 目录四卡各一遍均为 `89 passed / 2 skipped / 0 failed`
（与第十一节基线一致；新增用例后为 91 个用例口径）。

> 复跑方式见第十节；`_hf_strict_compat` 只存在于 staging 目录
> （`code/sglang/test_internal/internal/models/eb5/`），所以 pytest 要在 staging 下跑。
