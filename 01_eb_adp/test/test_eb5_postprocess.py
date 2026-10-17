"""EB5-lite **后处理** 小模块的算子级覆盖单测（按 ku《EBlite 和 dpsk-v4 算子对比》）。

文件内**每个 eb5_lite 侧 kernel 一个独立 test 方法**（方法名 = kernel 名，把
``::`` / ``<`` / ``>`` / ``[`` / ``]`` / ``-`` 等替换成 ``_``）。每个方法都**真跑该
kernel**（不打桩），并做形状 / 有限性 /（能对参考的）数值断言。脚手架见
``_eb5_kernel_test_common.py``。

后处理（ku 表格 eb5_lite 侧 13 算子 / 30 次）——本文件覆盖的 kernel：

    _hc_head_kernel_opt                       srt.internal.kernels.deepseek_v4.mhc_head_opt
    nvjet_sm100_tst_192x8_64x8_2x1_v_bz_TNT   lm_head GEMM（cuBLASLt 选核）
    alloc_decode_kernel                       kernels.ops.memory.allocator
    _multimax_selu_kernel                     srt.internal.models.eb5.multimax

**本模块有意跳过的纯 torch / cub 算子**：

    at::index_elementwise_kernel ×8            at::el[CUDAFunctorOnSelf_add<long>] ×6
    at::unrolled_elementwise_kernel[direct_copy] ×4  at::el[CUDAFunctorOnSelf_add<int>] ×2
    at::reduce_kernel ×1                       at::vectorized_layer_norm_kernel ×1
    at::el[CUDAFunctor_add<float>] ×1          cub::DeviceScanKernel ×1
    cub::DeviceScanInitKernel ×1
"""

from __future__ import annotations

import unittest

import torch

from _eb5_kernel_test_common import (
    CustomTestCase,
    _DEVICE,
    _DTYPE,
    _HC_MULT,
    _rand,
    _require_cuda,
)
from sglang.test_internal.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, num_gpus=1)


class TestPostprocessKernels(CustomTestCase):
    """后处理小模块的 kernel（每个 kernel 一个 test 方法，均真跑）。"""

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    def test_hc_head_kernel_opt(self):
        """``_hc_head_kernel_opt``：RMSNorm + hc 线性 + sigmoid-gate 加权和。"""
        from sglang.srt.internal.kernels.deepseek_v4.mhc_head_opt import (
            fused_hc_head_opt,
        )

        tokens, hc_mult, hidden = 8, _HC_MULT, 1024
        norm_eps, hc_eps = 1e-6, 1e-6
        x = _rand(tokens, hc_mult, hidden)
        hc_fn = (
            torch.randn(hc_mult, hc_mult * hidden, dtype=torch.float32, device=_DEVICE)
            * 0.02
        )
        hc_scale = torch.ones(1, dtype=torch.float32, device=_DEVICE)
        hc_base = torch.zeros(hc_mult, dtype=torch.float32, device=_DEVICE)

        y = fused_hc_head_opt(x, hc_fn, hc_scale, hc_base, norm_eps, hc_eps)
        self.assertEqual(y.shape, (tokens, hidden))
        self.assertEqual(y.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(y).all())

        xf = x.float().view(tokens, -1)
        sumsq = (xf * xf).sum(-1, keepdim=True)
        rsqrt = torch.rsqrt(sumsq / (hc_mult * hidden) + norm_eps)
        mix = xf @ hc_fn.float().t()
        pre = torch.sigmoid(mix * rsqrt * hc_scale + hc_base) + hc_eps
        ref = torch.einsum("tm,tmd->td", pre, x.float())
        self.assertTrue(
            torch.allclose(y.float(), ref, atol=3e-2, rtol=3e-2),
            "_hc_head_kernel_opt 与 torch 参考偏差过大",
        )

    def test_nvjet_sm100_tst_192x8_64x8_2x1_v_bz_TNT(self):
        """lm_head GEMM（``nvjet_sm100_tst_192x8_64x8_2x1_v_bz_TNT`` 的 cuBLASLt 选核）。

        以 lm_head 的真实几何（[T, hidden] x [vocab, hidden]^T）跑一次 bf16 GEMM，
        Blackwell 上由 cuBLASLt 落到 nvjet 后端；核对数值正确。
        """
        tokens, hidden, vocab = 4, 4096, 512
        x = _rand(tokens, hidden)
        w = _rand(vocab, hidden)
        logits = torch.nn.functional.linear(x, w)
        self.assertEqual(logits.shape, (tokens, vocab))
        self.assertEqual(logits.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(logits).all())
        ref = x.float() @ w.float().t()
        self.assertTrue(torch.allclose(logits.float(), ref, atol=5e-2, rtol=5e-2))

    def test_alloc_decode_kernel(self):
        """``alloc_decode_kernel``：decode 每步分配新 KV slot 的索引。"""
        from sglang.kernels.ops.memory.allocator import alloc_decode_kernel

        page_size = 4
        bs = 3
        seq_lens = torch.tensor([4, 5, 8], dtype=torch.int32, device=_DEVICE)
        last_loc = torch.tensor([3, 7, 11], dtype=torch.int64, device=_DEVICE)
        free_pages = torch.tensor([2, 3, 4], dtype=torch.int32, device=_DEVICE)
        out_indices = torch.empty(bs, dtype=torch.int64, device=_DEVICE)
        alloc_decode_kernel[(bs,)](
            seq_lens, last_loc, free_pages, out_indices, 4, page_size
        )
        # row0: seq 4/3 同页 -> last_loc+1；row1: seq 5 跨页 -> free_pages[0]*page_size；
        # row2: seq 8/7 同页 -> last_loc+1。
        self.assertTrue(
            torch.equal(
                out_indices,
                torch.tensor([4, 8, 12], dtype=torch.int64, device=_DEVICE),
            )
        )

    def test_multimax_selu_kernel(self):
        """``_multimax_selu_kernel``：multimax-SELU 分段激活，对齐 eager 图。"""
        from sglang.srt.internal.models.eb5.multimax import fused_multimax_selu

        ranges = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=_DTYPE, device=_DEVICE)
        ts = torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=_DTYPE, device=_DEVICE)
        logits = _rand(3, 32)
        out = fused_multimax_selu(logits, ranges, ts)
        self.assertEqual(out.shape, logits.shape)
        self.assertEqual(out.dtype, logits.dtype)
        self.assertTrue(torch.isfinite(out).all())

        # 融合 kernel 每步都按 bf16 舍入，参考图也按 bf16 逐 op 算（对齐 eager 图）。
        ref = logits.clone()
        ref = ref + ts[0] * torch.relu(ranges[0] - logits)
        ref = ref + ts[1] * torch.relu(logits - ranges[1])
        ref = ref + ts[2] * torch.relu(ranges[2] - logits) ** 2
        ref = ref + ts[3] * torch.relu(logits - ranges[3]) ** 2
        self.assertTrue(torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    unittest.main()
