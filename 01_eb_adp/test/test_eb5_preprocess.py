"""EB5-lite **前处理** 小模块的算子级覆盖单测（按 ku《EBlite 和 dpsk-v4 算子对比》）。

本文件是 ``test_eb5_operator_coverage.py`` 拆分后的一部分：**一个小模块一个文件**，
文件内**每个 eb5_lite 侧 kernel 一个独立 test 方法**（方法名 = kernel 名，把
``::`` / ``<`` / ``>`` / ``[`` / ``]`` / ``-`` 等替换成 ``_``）。每个方法都**真跑该
kernel**（不打桩），并做形状 / 有限性 /（能对参考的）数值断言。脚手架见同目录
``_eb5_kernel_test_common.py``。

前处理（ku 表格 eb5_lite 侧 27 算子 / 56 次）——本文件覆盖的**自研 kernel**：

    compute_position_kernel              compute_position_triton
    _expand_prefill_causally_kernel      expand_prefill_causally_triton
    _page_table_positions_kernel         build_page_table_positions_triton
    _causal_swa_page_indices_kernel      build_causal_swa_page_indices_triton
    _init_compressed_attn_metadata_kernel init_compression_metadata
    topk_plan                            plan_topk_v2 -> TopKKernel::plan
    plan_compress_prefill_kernel_1       CompressorPrefillPlan.generate
    get_mla_metadata_kernel              sgl_kernel.flash_mla.get_mla_metadata

**本模块有意跳过的纯 torch / cub 算子**（ATen / cub 通用逐元素 / 扫描 kernel，非
eb5_lite 自研或第三方算子，跑它们只是在测 PyTorch 本身；ku 表格保留这些行只为说明
"两侧在纯 torch 侧完全同构"）：

    at::CatArrayBatchedCopy ×7            at::index_elementwise_kernel ×3
    at::el[FillFunctor<int>] ×6           at::unrolled_elementwise_kernel[direct_copy] ×4
    cub::DeviceScanKernel ×3              at::el[launch_clamp_scalar] ×3
    at::elementwise_kernel[direct_copy] ×2 at::el[FillFunctor<float>] ×3
    cub::DeviceScanInitKernel ×3          at::arange_cuda_out ×3
    at::el[CUDAFunctorOnSelf_add<int>] ×2 at::vectorized_gather_kernel ×1
    at::elementwise_kernel ×1             at::vectorized_elementwise_kernel ×1
    at::multi_tensor_apply_kernel ×1      at::el[div_floor] ×1
    at::el[CUDAFunctor_add<float>] ×1     at::el[CUDAFunctorOnSelf_add<long>] ×1
    at::el[CUDAFunctor_add<int>] ×1
"""

from __future__ import annotations

import unittest

import torch

from _eb5_kernel_test_common import CustomTestCase, _DEVICE, _DTYPE, _require_cuda
from sglang.test_internal.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, num_gpus=1)


class TestPreprocessKernels(CustomTestCase):
    """前处理小模块的自研 kernel（每个 kernel 一个 test 方法，均真跑）。"""

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    def test_compute_position_kernel(self):
        """``compute_position_kernel``：逐 batch 生成 positions + extend_start_loc。"""
        from sglang.kernels.ops.attention.position import compute_position_triton

        prefix = torch.tensor([0, 5, 11], dtype=torch.int32, device=_DEVICE)
        extend = torch.tensor([4, 3, 2], dtype=torch.int32, device=_DEVICE)
        total = int(extend.sum().item())
        positions, start_loc = compute_position_triton(prefix, extend, total)

        self.assertEqual(positions.shape, (total,))
        self.assertEqual(positions.dtype, torch.int64)
        self.assertEqual(start_loc.shape, (3,))
        # positions = prefix + [0..extend-1]，逐 batch 拼接；start_loc = 前缀和。
        ref_positions = torch.cat(
            [
                torch.arange(int(p), int(p) + int(e), device=_DEVICE)
                for p, e in zip(prefix.tolist(), extend.tolist())
            ]
        )
        ref_start = torch.tensor([0, 4, 7], dtype=torch.int32, device=_DEVICE)
        self.assertTrue(torch.equal(positions, ref_positions))
        self.assertTrue(torch.equal(start_loc.to(torch.int32), ref_start))

    def test_expand_prefill_causally_kernel(self):
        """``_expand_prefill_causally_kernel``：seq_lens → 逐 token 因果长度。"""
        from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
            expand_prefill_causally_triton,
        )

        req_pool = torch.tensor([7, 9], dtype=torch.int32, device=_DEVICE)
        seq_lens = torch.tensor([6, 4], dtype=torch.int32, device=_DEVICE)
        extend_lens = torch.tensor([3, 2], dtype=torch.int32, device=_DEVICE)
        num_tokens = int(extend_lens.sum().item())
        res = expand_prefill_causally_triton(
            req_pool_indices=req_pool,
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            num_tokens=num_tokens,
            padded_num_tokens=None,
        )
        self.assertEqual(res.seq_lens_casual.shape, (num_tokens,))
        self.assertEqual(res.req_pool_indices_repeated.shape, (num_tokens,))
        # batch0: kv_len 6 / qo_len 3 → 4,5,6；batch1: kv_len 4 / qo_len 2 → 3,4。
        self.assertTrue(
            torch.equal(
                res.seq_lens_casual.to(torch.int32),
                torch.tensor([4, 5, 6, 3, 4], dtype=torch.int32, device=_DEVICE),
            )
        )
        self.assertTrue(
            torch.equal(
                res.req_pool_indices_repeated.to(torch.int32),
                torch.tensor([7, 7, 7, 9, 9], dtype=torch.int32, device=_DEVICE),
            )
        )

    def test_page_table_positions_kernel(self):
        """``_page_table_positions_kernel``：positions / page_table / swa_topk_lengths。"""
        from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
            build_page_table_positions_triton,
        )

        page_size, max_seq_len, swa_window = 64, 256, 128
        num_q = 3
        # req_to_token: 每行是某请求的 token→slot 映射；slot = page*page_size + off。
        req_to_token = torch.arange(
            4 * max_seq_len, dtype=torch.int32, device=_DEVICE
        ).view(4, max_seq_len)
        req_pool_repeated = torch.tensor([0, 1, 2], dtype=torch.int32, device=_DEVICE)
        seq_lens_casual = torch.tensor([1, 65, 200], dtype=torch.int32, device=_DEVICE)
        res = build_page_table_positions_triton(
            req_to_token=req_to_token,
            req_pool_indices_repeated=req_pool_repeated,
            seq_lens_casual=seq_lens_casual,
            max_seq_len=max_seq_len,
            page_size=page_size,
            swa_window=swa_window,
        )
        self.assertEqual(res.positions_casual.shape, (num_q,))
        self.assertTrue(torch.equal(res.positions_casual, seq_lens_casual - 1))
        num_pages = max_seq_len // page_size
        self.assertEqual(res.page_table.shape, (num_q, num_pages))
        # 第 r 行的 page_table[p] = req_to_token[rp, p*page_size] // page_size。
        ref_pt = (
            req_to_token[req_pool_repeated.to(torch.int64), :max_seq_len:page_size]
            // page_size
        ).to(torch.int32)
        self.assertTrue(torch.equal(res.page_table, ref_pt))
        self.assertTrue(
            torch.equal(
                res.swa_topk_lengths,
                torch.clamp(seq_lens_casual, max=swa_window),
            )
        )

    def test_causal_swa_page_indices_kernel(self):
        """``_causal_swa_page_indices_kernel``：因果 SWA 窗口内的 slot 索引。"""
        from sglang.kernels.ops.attention.dsv4_attn_metadata_kernels import (
            build_causal_swa_page_indices_triton,
        )

        page_size, swa_window = 64, 8
        max_seq_len = 128
        req_to_token = torch.arange(
            max_seq_len, dtype=torch.int32, device=_DEVICE
        ).view(1, max_seq_len)
        full_to_swa = torch.arange(max_seq_len, dtype=torch.int32, device=_DEVICE)
        req_pool_repeated = torch.tensor([0, 0], dtype=torch.int32, device=_DEVICE)
        seq_lens_casual = torch.tensor([3, 10], dtype=torch.int32, device=_DEVICE)
        out = build_causal_swa_page_indices_triton(
            req_to_token=req_to_token,
            full_to_swa_mapping=full_to_swa,
            req_pool_indices_repeated=req_pool_repeated,
            seq_lens_casual=seq_lens_casual,
            swa_window=swa_window,
            page_index_aligned_size=8,
        )
        self.assertEqual(out.shape, (2, swa_window))
        # 第 0 行 pos=2：窗口 [2,1,0,-1...] → [2,1,0,-1,-1,-1,-1,-1]。
        self.assertTrue(
            torch.equal(
                out[0].to(torch.int32),
                torch.tensor(
                    [2, 1, 0, -1, -1, -1, -1, -1], dtype=torch.int32, device=_DEVICE
                ),
            )
        )
        # 第 1 行 pos=9：窗口 [9..2]。
        self.assertTrue(
            torch.equal(
                out[1].to(torch.int32),
                torch.tensor(
                    [9, 8, 7, 6, 5, 4, 3, 2], dtype=torch.int32, device=_DEVICE
                ),
            )
        )

    def test_init_compressed_attn_metadata_kernel(self):
        """``_init_compressed_attn_metadata_kernel``：c4 / c128 压缩元数据。"""
        from sglang.kernels.ops.attention.dsv4.metadata_kernel import (
            init_compression_metadata,
        )

        bs = 4
        seq_lens = torch.tensor([4, 128, 130, 256], dtype=torch.int32, device=_DEVICE)
        positions = seq_lens - 1
        raw_out_loc = torch.arange(bs, dtype=torch.int64, device=_DEVICE)
        page_size = 128
        page_table = torch.arange(bs * 2, dtype=torch.int32, device=_DEVICE).view(bs, 2)
        (
            c4_out_loc,
            c4_positions,
            c4_seq_lens_raw,
            c4_seq_lens_clamp1,
            c128_out_loc,
            c128_positions,
            c128_seq_lens_raw,
            c128_seq_lens_clamp1,
            c128_page_indices,
        ) = init_compression_metadata(
            seq_lens,
            positions,
            raw_out_loc,
            page_table,
            page_size,
            True,
        )
        self.assertEqual(c4_seq_lens_raw.shape, (bs,))
        self.assertEqual(c128_seq_lens_raw.shape, (bs,))
        self.assertTrue(torch.equal(c4_seq_lens_raw, (seq_lens // 4).to(torch.int32)))
        self.assertTrue(
            torch.equal(c128_seq_lens_raw, (seq_lens // 128).to(torch.int32))
        )
        # c128 clamp1 至少为 1。
        self.assertTrue(bool((c128_seq_lens_clamp1 >= 1).all()))
        self.assertIsNotNone(c128_page_indices)
        self.assertEqual(c128_page_indices.shape[0], bs)

    def test_topk_plan(self):
        """``topk_plan``：为 v2 top-k transform 预生成 routing plan。"""
        from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2

        seq_lens = torch.tensor([1, 64, 2048, 3000], dtype=torch.int32, device=_DEVICE)
        metadata = plan_topk_v2(seq_lens, static_threshold=0)
        # metadata 形状固定为 (bs + 1, 2) int32。
        self.assertEqual(metadata.shape, (seq_lens.numel() + 1, 2))
        self.assertEqual(metadata.dtype, torch.int32)
        # row 0 = {cluster_threshold, num_cluster_items}，二者都应有限。
        self.assertTrue(bool(torch.isfinite(metadata.float()).all()))

    def test_plan_compress_prefill_kernel_1(self):
        """``plan_compress_prefill_kernel_1``：compressor prefill plan（plan_c / plan_w）。"""
        from sglang.kernels.ops.attention.dsv4.compress import CompressorPrefillPlan

        num_q = 256
        req_pool = torch.tensor([0], dtype=torch.int64, device=_DEVICE)
        seq_lens = torch.tensor([num_q], dtype=torch.int64, device=_DEVICE)
        extend_lens = torch.tensor([num_q], dtype=torch.int64, device=_DEVICE)
        req_to_token = torch.arange(num_q, dtype=torch.int32, device=_DEVICE).view(
            1, num_q
        )
        full_to_state = torch.arange(num_q, dtype=torch.int64, device=_DEVICE)
        plan = CompressorPrefillPlan.generate(
            compress_ratio=128,
            req_pool_indices=req_pool,
            seq_lens=seq_lens,
            extend_lens=extend_lens,
            req_to_token=req_to_token,
            full_to_state=full_to_state,
            swa_page_size=128,
            ring_size=128,
            num_q_tokens=num_q,
        )
        self.assertEqual(plan.compress_ratio, 128)
        self.assertEqual(plan.plan_c.shape, (num_q, 16))
        self.assertEqual(plan.plan_w.shape, (num_q, 8))

    def test_get_mla_metadata_kernel(self):
        """``get_mla_metadata_kernel``：FlashMLA 的 sparse decode 调度元数据。"""
        import sgl_kernel.flash_mla as fm

        cache_seqlens = torch.tensor([128], dtype=torch.int32, device=_DEVICE)
        metadata, num_splits = fm.get_mla_metadata(
            cache_seqlens=cache_seqlens,
            num_q_tokens_per_head_k=1,
            num_heads_k=1,
            num_heads_q=64,
            is_fp8_kvcache=True,
            topk=512,
        )
        self.assertEqual(metadata.dim(), 2)
        self.assertGreaterEqual(int(num_splits.numel()), 1)
        self.assertTrue(torch.isfinite(metadata.float()).all())


if __name__ == "__main__":
    unittest.main()
