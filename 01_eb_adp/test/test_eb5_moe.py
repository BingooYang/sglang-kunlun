"""EB5-lite **moe** 小模块的算子级覆盖单测（按 ku《EBlite 和 dpsk-v4 算子对比》）。

文件内**每个 eb5_lite 侧 kernel 一个独立 test 方法**（方法名 = kernel 名，把
``::`` / ``<`` / ``>`` / ``[`` / ``]`` / ``-`` 等替换成 ``_``）。每个方法都**真跑该
kernel**（不打桩），并做形状 / 有限性 /（能对参考的）数值断言。脚手架见
``_eb5_kernel_test_common.py``。

moe（ku 表格 eb5_lite 侧 14 算子 / 602 次）——本文件覆盖的 kernel：

    MEGA_MOE                                  deep_gemm.fp8_fp4_mega_moe        (skip)
    deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl   deep_gemm.fp8_fp4_gemm_nt
    per_token_group_quant_flat_kernel         sglang_per_token_group_quant_fp8
    mega_moe_pre_dispatch_kernel              kernels.ops.attention.dsv4.moe
    nvjet_sm100_tst_64x8_64x16_1x4_h_bz_TNT   router GEMM（cuBLASLt 选核）
    _eb5_router_topk_bitonic_kernel           eb5_fused_topk.fused_split_feature_topk
    situ_and_mul_kernel                       kernels.ops.kimi_k3.activation.situ_and_mul
    RMSNorm<2048>                             torch.nn.functional.rms_norm (ATen/flashinfer)
    all_reduce_1shot_push_kernel              communication.all_reduce           (skip)
    memcpy_triton_kernel                      kernels.ops.memory.memcpy_triton

**本模块有意跳过的纯 torch / cub 算子**：

    at::el[FillFunctor<c10::BFloat16>] ×2   cub::DeviceScanKernel ×1
    at::el[FillFunctor<long>] ×1            cub::DeviceScanInitKernel ×1

**无法在单进程算子级构造、方法内 skipTest 的 kernel**：

* ``MEGA_MOE``（``deep_gemm.fp8_fp4_mega_moe``）：需要 DeepGEMM 的 symmetric-memory
  ``SymmBuffer`` + UTCCP 预排布权重 / packed 激活 SF 布局，单进程算子级用例无法在合理
  范围内构造（probe 已确认 symm buffer 需 multi-rank 布局约束）。由
  ``test_eb5_moe_mega_moe.py``（模块级）与 ``test_w4a8_megamoe.py`` 覆盖。
* ``all_reduce_1shot_push_kernel``（``communication.all_reduce``）：kernel 在
  ``world_size==1`` 时直接报 "Invalid world size: 1"，本单进程用例无法构造 push plane。
"""

from __future__ import annotations

import unittest

import torch

from _eb5_kernel_test_common import (
    CustomTestCase,
    _DEVICE,
    _DTYPE,
    _rand,
    _require_cuda,
)
from sglang.test_internal.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, num_gpus=1)


class TestMoeKernels(CustomTestCase):
    """moe 小模块的 kernel（每个 kernel 一个 test 方法）。"""

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    def test_MEGA_MOE(self):
        """``MEGA_MOE``（``deep_gemm.fp8_fp4_mega_moe``）：单进程算子级不可构造。"""
        self.skipTest(
            "MEGA_MOE 需要 DeepGEMM 的 symmetric-memory SymmBuffer + UTCCP 预排布权重/"
            "packed 激活 SF 布局，单进程算子级用例无法构造；由 test_eb5_moe_mega_moe.py "
            "（模块级）与 test_w4a8_megamoe.py 覆盖。"
        )

    def test_deep_gemm_sm100_fp8_fp4_gemm_1d1d_impl(self):
        """``deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl``：DeepGEMM fp8 NT GEMM。"""
        import deep_gemm
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8_ue8m0,
        )

        torch.manual_seed(0)
        M, N, K = 64, 128, 256
        a = _rand(M, K)
        b = _rand(N, K)
        aq, asc = sglang_per_token_group_quant_fp8_ue8m0(a, 32)
        bq, bsc = sglang_per_token_group_quant_fp8_ue8m0(b, 32)
        out = torch.empty(M, N, dtype=_DTYPE, device=_DEVICE)
        deep_gemm.fp8_fp4_gemm_nt(
            (aq, asc), (bq, bsc), out, recipe_a=(1, 32), recipe_b=(1, 32)
        )
        ref = a.float() @ b.float().t()
        self.assertEqual(out.shape, (M, N))
        self.assertTrue(torch.isfinite(out).all())
        # fp8 输入 -> bf16 输出，相对误差受 fp8 量化限制。
        self.assertTrue(
            torch.allclose(out.float(), ref, atol=6.0, rtol=0.35),
            "deep_gemm fp8 NT GEMM 与 fp32 参考偏差过大",
        )

    def test_per_token_group_quant_flat_kernel(self):
        """``per_token_group_quant_flat_kernel``：per-token-group fp8 量化。"""
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        tokens, hidden, group_size = 8, 256, 32
        x = _rand(tokens, hidden)
        x_q, x_s = sglang_per_token_group_quant_fp8(x, group_size)
        self.assertEqual(x_q.shape, (tokens, hidden))
        self.assertEqual(x_q.dtype, torch.float8_e4m3fn)
        self.assertEqual(x_s.shape, (tokens, hidden // group_size))
        # dequant 回 fp32：每个 group 内误差受 fp8 精度约束。
        deq = x_q.float() * x_s.float().repeat_interleave(group_size, dim=-1)
        self.assertTrue(torch.isfinite(deq).all())
        self.assertTrue(
            torch.allclose(deq, x.float(), atol=0.2, rtol=0.1),
            "per_token_group_quant_flat 量化误差过大",
        )

    def test_mega_moe_pre_dispatch_kernel(self):
        """``mega_moe_pre_dispatch_kernel``：量化 + topk 打包进 mega-moe 缓冲。"""
        from sglang.kernels.ops.attention.dsv4.moe import mega_moe_pre_dispatch

        num_tokens, top_k, hidden, group_size = 8, 2, 512, 32
        padded_max = num_tokens
        x = _rand(num_tokens, hidden)
        topk_idx = torch.randint(
            0, 4, (num_tokens, top_k), dtype=torch.int32, device=_DEVICE
        )
        topk_w = torch.rand(num_tokens, top_k, dtype=torch.float32, device=_DEVICE)

        buf_x = torch.empty(
            padded_max, hidden, dtype=torch.float8_e4m3fn, device=_DEVICE
        )
        num_groups = hidden // group_size
        buf_x_sf = torch.empty(
            padded_max, num_groups // 4, dtype=torch.int32, device=_DEVICE
        )
        buf_topk_idx = torch.empty(
            padded_max, top_k, dtype=torch.int64, device=_DEVICE
        )
        buf_topk_weights = torch.empty(
            padded_max, top_k, dtype=torch.float32, device=_DEVICE
        )

        mega_moe_pre_dispatch(
            x,
            topk_idx,
            topk_w,
            buf_x,
            buf_x_sf,
            buf_topk_idx,
            buf_topk_weights,
            quant_group_size=group_size,
        )
        self.assertTrue(torch.isfinite(buf_x.float()).all())
        # topk_idx 被拓宽成 int64 原样写入。
        self.assertTrue(torch.equal(buf_topk_idx, topk_idx.to(torch.int64)))
        self.assertTrue(torch.allclose(buf_topk_weights, topk_w))

    def test_nvjet_sm100_tst_64x8_64x16_1x4_h_bz_TNT(self):
        """router GEMM（``nvjet_sm100_tst_64x8_64x16_1x4_h_bz_TNT`` 的 cuBLASLt 选核）。

        以 router 的真实几何（[T, hidden] x [experts, hidden]^T）跑一次 bf16 GEMM，
        Blackwell 上由 cuBLASLt 落到 nvjet 后端；核对数值正确。
        """
        tokens, hidden, experts = 8, 256, 16
        x = _rand(tokens, hidden)
        w = _rand(experts, hidden)
        logits = torch.nn.functional.linear(x, w)
        self.assertEqual(logits.shape, (tokens, experts))
        self.assertEqual(logits.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(logits).all())
        ref = x.float() @ w.float().t()
        self.assertTrue(torch.allclose(logits.float(), ref, atol=5e-2, rtol=5e-2))

    def test_eb5_router_topk_bitonic_kernel(self):
        """``_eb5_router_topk_bitonic_kernel``：split-feature 融合路由 TopK。"""
        from sglang.srt.internal.layers.moe.eb5_fused_topk import (
            fused_split_feature_topk,
        )

        num_experts, topk, tokens = 16, 4, 37
        fused_logits = _rand(tokens, 2 * num_experts, dtype=torch.float32)
        bias = torch.zeros(num_experts, dtype=torch.float32, device=_DEVICE)
        per_expert_scale = torch.ones(num_experts, dtype=torch.float32, device=_DEVICE)

        weights, ids, _ = fused_split_feature_topk(
            fused_logits,
            bias,
            topk,
            True,
            "sigmoid",
            per_expert_scale=per_expert_scale,
        )
        self.assertEqual(ids.shape, (tokens, topk))
        self.assertEqual(weights.shape, (tokens, topk))
        halves = fused_logits.float().view(tokens, 2, num_experts)
        scores = torch.sigmoid(halves).sum(dim=1)
        ref_ids = torch.topk(scores, topk, dim=-1).indices
        self.assertTrue(
            torch.equal(ids.sort(dim=-1).values, ref_ids.sort(dim=-1).values)
        )

    def test_situ_and_mul_kernel(self):
        """``situ_and_mul_kernel``：SiTU(SoftCap-GLU) 激活，对齐 eager 参考。"""
        from sglang.kernels.ops.kimi_k3.activation import situ_and_mul

        tokens, hidden = 8, 256
        x = _rand(tokens, 2 * hidden)
        beta, linear_beta = 4.0, 25.0
        out = situ_and_mul(x, None, beta, linear_beta)
        self.assertEqual(out.shape, (tokens, hidden))
        self.assertEqual(out.dtype, _DTYPE)

        gate = x[..., :hidden].float()
        up = x[..., hidden:].float()
        ref = (beta * torch.tanh(gate / beta) * torch.sigmoid(gate)) * (
            linear_beta * torch.tanh(up / linear_beta)
        )
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(
            torch.allclose(out.float(), ref, atol=3e-2, rtol=3e-2),
            "situ_and_mul 与 eager 参考偏差过大",
        )

    def test_RMSNorm_2048(self):
        """``RMSNorm<2048>``：MoE latent / hidden 维的 RMSNorm（ATen/flashinfer）。"""
        hidden = 2048
        x = _rand(8, hidden)
        weight = _rand(hidden)
        out = torch.nn.functional.rms_norm(x, (hidden,), weight, 1e-5)
        self.assertEqual(out.shape, (8, hidden))
        self.assertEqual(out.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(out).all())
        ref = (
            x.float()
            * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
            * weight.float()
        )
        self.assertTrue(torch.allclose(out.float(), ref, atol=3e-2, rtol=3e-2))

    def test_all_reduce_1shot_push_kernel(self):
        """``all_reduce_1shot_push_kernel``：kernel 拒绝 world_size==1。"""
        self.skipTest(
            "custom_all_reduce 的 1shot_push kernel 在 world_size==1 时报 "
            "'Invalid world size: 1'；push plane 需要多 rank 的 symmetric IPC "
            "workspace，单进程用例无法构造，由多 rank e2e 覆盖。"
        )

    def test_memcpy_triton_kernel(self):
        """``memcpy_triton_kernel``：offset/size 驱动的 device 间拷贝。"""
        from sglang.kernels.ops.memory.memcpy_triton import memcpy_triton

        dst = torch.zeros(8, 32, dtype=_DTYPE, device=_DEVICE)
        src = _rand(8, 32)
        offset = torch.tensor(0, dtype=torch.int32, device=_DEVICE)
        # 注意：offset / sz 以「行」为单位，kernel 内部再乘 chunk_size（= 每行元素数）。
        # 传 numel 会让 mask 覆盖到 buffer 之外（OOB），所以这里传行数。
        sz = torch.tensor(dst.shape[0], dtype=torch.int32, device=_DEVICE)
        memcpy_triton(dst, src, 0, offset, sz, True)
        self.assertTrue(torch.equal(dst, src))

        # 只有「整块拷贝」是不够的：适配层 2026-10-19 的 D2H 修复（`int(offset)` →
        # device 侧）和 2026-10-20 的 mask 索引修复（`x[mask]` 隐式 D2H）都只在
        # 非零 offset / 部分拷贝 / 两个方向下才走得到。这里按 `memcpy_cpu` 的语义
        # （行窗口 + 截断 + 双向）逐个对照，offset/sz 一律用 **device 标量**。
        for offset_src in (True, False):
            for off, size in ((0, 8), (3, 4), (3, 99), (9, 4)):
                with self.subTest(offset_src=offset_src, offset=off, size=size):
                    dst = _rand(8, 32)
                    src = _rand(12, 32)
                    exp = dst.clone()
                    dst_start = 0 if offset_src else off
                    src_start = off if offset_src else 0
                    end = min(
                        min(dst_start + size, dst.shape[0]) - dst_start,
                        min(src_start + size, src.shape[0]) - src_start,
                    )
                    if end > 0:
                        exp[dst_start : dst_start + end] = src[
                            src_start : src_start + end
                        ]
                    memcpy_triton(
                        dst,
                        src,
                        0,
                        torch.tensor(off, dtype=torch.int32, device=_DEVICE),
                        torch.tensor(size, dtype=torch.int32, device=_DEVICE),
                        offset_src,
                    )
                    self.assertTrue(torch.equal(dst, exp))


if __name__ == "__main__":
    unittest.main()
