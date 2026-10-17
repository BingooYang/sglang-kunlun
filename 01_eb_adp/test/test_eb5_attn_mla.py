"""EB5-lite **attn ② MLA** 小模块的算子级覆盖单测（按 ku《EBlite 和 dpsk-v4 算子对比》）。

eb5_lite 的 MLA 层 = 全量 MLA + lightning indexer（6 层，每 7 层一个）。文件内
**每个 eb5_lite 侧 kernel 一个独立 test 方法**（方法名 = kernel 名，把 ``::`` / ``<`` /
``>`` / ``[`` / ``]`` / ``-`` 等替换成 ``_``）。每个方法都**真跑该 kernel**（不打桩），
并做形状 / 有限性 /（能对参考的）数值断言。脚手架见 ``_eb5_kernel_test_common.py``。

本文件覆盖的 kernel（ku 表格「attn ② MLA」eb5_lite 列）：

    deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl  deep_gemm.fp8_fp4_gemm_nt
    per_token_group_quant_flat_kernel        sglang_per_token_group_quant_fp8
    RMSNorm<1024> / RMSNorm<512>             torch.nn.functional.rms_norm
    sm100_mqa_logits                         deep_gemm.bf16_mqa_logits（lightning indexer）
    topk_main_kernel                         kernels.ops.attention.dsv4.topk.topk_transform_ragged_v2
    fused_q_indexer_rope_first_quant         jit_kernel.internal.dsa
    fused_k_indexer_norm_rope_store          jit_kernel.internal.dsa
    flash_fwd_splitkv_mla_fp8_sparse_kernel  sgl_kernel.flash_mla.flash_mla_with_kvcache (fp8 sparse)
    flash_fwd_mla_combine_kernel             sgl_kernel.flash_mla.flash_mla_with_kvcache (split/combine)
    concat_mla_absorb_q_kernel               kernels.ops.attention.concat_mla
    fused_rope_kernel                        kernels.ops.attention.dsv4.elementwise.fused_rope_inplace
    set_mla_kv_buffer_kernel                 kernels.ops.kvcache.set_mla_kv_buffer
    _quantize_k_cache_fast_kernel            kernels.ops.attention.dsa.quant_k_cache
    _append_indexer_swa_kernel               srt.internal.layers.attention.mla_hca.dsa_swa_index
    _get_k_and_s_triton_kernel               kernels.ops.attention.dsa.index_buf_accessor
    nvjet_sm100_tst_256x24_64x6_2x1_v_bz_TNT    absorb GEMM（cuBLASLt 选核）
    nvjet_sm100_tst_128x24_64x11_2x1_v_bz_TNT   absorb GEMM（cuBLASLt 选核）
    nvjet_sm100_tst_64x32_64x16_2x2_2cta_h_bz_NNT  o_proj GEMM（cuBLASLt 选核）
    nvjet_sm100_tst_64x8_64x16_1x4_h_bz_splitK_TNT splitK GEMM（cuBLASLt 选核）
    cublasLt::splitKreduce_kernel            splitK 归约（cuBLASLt 选核）
    _eb5_sigmoid_gate_mul_kernel             srt.internal.layers.attention.eb5_sigmoid_gate_mul
    get_mla_metadata_kernel                  sgl_kernel.flash_mla.get_mla_metadata

**本模块有意跳过的纯 torch / cub 算子**：

    at::el[FillFunctor<int>] ×12   at::unrolled_elementwise_kernel[direct_copy] ×6
    at::elementwise_kernel[direct_copy] ×6

**说明（cuBLASLt / FlashMLA 选核）**：``nvjet_sm100_tst_*`` / ``cublasLt::splitKreduce_kernel``
由 cuBLASLt 在 GEMM 内部选核，``flash_fwd_*`` 由 FlashMLA 内部选核，均无法在算子级
"点名"调用；这里以与 MLA 真实几何一致的 GEMM / FlashMLA sparse-decode 调用**真跑**该
后端，并核对数值。
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

register_cuda_ci(est_time=300, num_gpus=1)


class TestMlaKernels(CustomTestCase):
    """attn② MLA 小模块的 kernel（每个 kernel 一个 test 方法，均真跑）。"""

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    # ── 投影 / 量化 / norm ──
    def test_deep_gemm_sm100_fp8_fp4_gemm_1d1d_impl(self):
        """``deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl``：MLA 投影 GEMM。"""
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
        self.assertTrue(torch.allclose(out.float(), ref, atol=6.0, rtol=0.35))

    def test_per_token_group_quant_flat_kernel(self):
        """``per_token_group_quant_flat_kernel``：投影前的 per-token-group fp8 量化。"""
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        tokens, hidden, group_size = 8, 512, 32
        x = _rand(tokens, hidden)
        x_q, x_s = sglang_per_token_group_quant_fp8(x, group_size)
        self.assertEqual(x_q.shape, (tokens, hidden))
        self.assertEqual(x_q.dtype, torch.float8_e4m3fn)
        deq = x_q.float() * x_s.float().repeat_interleave(group_size, dim=-1)
        self.assertTrue(torch.allclose(deq, x.float(), atol=0.2, rtol=0.1))

    def test_RMSNorm_1024(self):
        """``RMSNorm<1024>``：MLA q/kv norm 的 1024 维（ATen/flashinfer）。"""
        hidden = 1024
        x = _rand(8, hidden)
        weight = _rand(hidden)
        out = torch.nn.functional.rms_norm(x, (hidden,), weight, 1e-6)
        self.assertEqual(out.shape, (8, hidden))
        self.assertTrue(torch.isfinite(out).all())
        ref = (
            x.float()
            * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        )
        self.assertTrue(torch.allclose(out.float(), ref, atol=3e-2, rtol=3e-2))

    def test_RMSNorm_512(self):
        """``RMSNorm<512>``：MLA q/kv norm（ATen/flashinfer）。"""
        hidden = 512
        x = _rand(8, hidden)
        weight = _rand(hidden)
        out = torch.nn.functional.rms_norm(x, (hidden,), weight, 1e-6)
        self.assertEqual(out.shape, (8, hidden))
        self.assertTrue(torch.isfinite(out).all())
        ref = (
            x.float()
            * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        )
        self.assertTrue(torch.allclose(out.float(), ref, atol=3e-2, rtol=3e-2))

    # ── lightning indexer ──
    def test_sm100_mqa_logits(self):
        """``sm100_mqa_logits``：lightning indexer 的 MQA logits（DeepGEMM SM100）。

        eb5 MLA 的 indexer 在 SM100 上走 ``deep_gemm.bf16_mqa_logits``（非 paged ragged
        MQA logits，即 trace 里的 ``sm100_mqa_logits``）。与 fp32 einsum 参考比对。
        """
        import deep_gemm

        T, H, D, KV = 4, 8, 128, 64
        torch.manual_seed(0)
        q = _rand(T, H, D)
        kv = _rand(KV, D)
        weights = torch.randn(T, H, dtype=torch.float32, device=_DEVICE)
        ks = torch.zeros(T, dtype=torch.int32, device=_DEVICE)
        ke = torch.full((T,), KV, dtype=torch.int32, device=_DEVICE)
        out = deep_gemm.bf16_mqa_logits(q, kv, weights, ks, ke, clean_logits=False)
        self.assertEqual(out.shape, (T, KV))
        ref = torch.einsum("thd,vd->thv", q.float(), kv.float())
        ref = (weights.float().unsqueeze(-1) * torch.relu(ref)).sum(1)
        self.assertTrue(
            torch.allclose(out.float(), ref, atol=5e-1, rtol=1e-2),
            "sm100_mqa_logits 与 fp32 参考偏差过大",
        )

    def test_topk_main_kernel(self):
        """``topk_main_kernel``：ragged prefill 的 fused top-k。"""
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

        rows, length, topk = 2, 64, 8
        scores = _rand(rows, length, dtype=torch.float32)
        seq_lens = torch.full((rows,), length, dtype=torch.int32, device=_DEVICE)
        out_offsets = torch.zeros(rows, dtype=torch.int32, device=_DEVICE)
        out_indices = torch.empty(rows, topk, dtype=torch.int32, device=_DEVICE)
        row_starts = torch.zeros(rows, dtype=torch.int32, device=_DEVICE)
        topk_transform_ragged_v2(
            scores,
            seq_lens,
            out_offsets=out_offsets,
            out_indices=out_indices,
            row_starts=row_starts,
        )
        ref = torch.topk(scores, topk, dim=-1).indices
        # 顺序不保证，比较集合。
        self.assertTrue(
            torch.equal(out_indices.sort(dim=-1).values, ref.sort(dim=-1).values)
        )

    def test_fused_q_indexer_rope_first_quant(self):
        """``fused_q_indexer_rope_first_quant``：indexer Q 的 RoPE + fp8 量化。"""
        from sglang.jit_kernel.internal.dsa import fused_q_indexer_rope_first_quant
        from sglang.srt.layers.rotary_embedding import get_rope_wrapper

        head_dim, rope_dim, max_pos = 128, 64, 4096
        rotary = get_rope_wrapper(
            rope_dim,
            rotary_dim=rope_dim,
            max_position=max_pos,
            base=10000.0,
            rope_scaling=None,
            is_neox_style=True,
            device=_DEVICE,
        ).to(_DEVICE)
        tokens, heads = 8, 4
        q = _rand(tokens, heads, head_dim)
        weight = _rand(tokens, heads)
        positions = torch.randint(
            1, max_pos, (tokens,), dtype=torch.int32, device=_DEVICE
        )
        q_fp8, weights_out = fused_q_indexer_rope_first_quant(
            q, weight, 1.0, rotary.cos_sin_cache, positions
        )
        self.assertEqual(q_fp8.shape, (tokens, heads, head_dim))
        self.assertEqual(q_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(weights_out.shape, (tokens, heads, 1))
        self.assertTrue(torch.isfinite(q_fp8.float()).all())

    def test_fused_k_indexer_norm_rope_store(self):
        """``fused_k_indexer_norm_rope_store``：indexer K 的 LayerNorm+RoPE+fp8+paged 写回。"""
        from sglang.jit_kernel.internal.dsa import fused_k_indexer_norm_rope_store
        from sglang.srt.layers.rotary_embedding import get_rope_wrapper

        head_dim, rope_dim, max_pos = 128, 64, 4096
        rotary = get_rope_wrapper(
            rope_dim,
            rotary_dim=rope_dim,
            max_position=max_pos,
            base=10000.0,
            rope_scaling=None,
            is_neox_style=True,
            device=_DEVICE,
        ).to(_DEVICE)
        tokens, page_size, num_pages = 8, 64, 4
        key_raw = _rand(tokens, head_dim)
        ln_weight = _rand(head_dim, dtype=torch.float32)
        ln_bias = _rand(head_dim, dtype=torch.float32)
        positions = torch.randint(
            1, max_pos, (tokens,), dtype=torch.int32, device=_DEVICE
        )
        loc = torch.randperm(num_pages * page_size, device=_DEVICE)[:tokens].to(
            torch.int64
        )
        cache = torch.zeros(num_pages, 132 * page_size, dtype=torch.uint8, device=_DEVICE)
        fused_k_indexer_norm_rope_store(
            key_raw,
            cache,
            loc,
            ln_weight,
            ln_bias,
            1e-6,
            rotary.cos_sin_cache,
            positions,
            page_size,
        )
        self.assertGreater(cache.abs().sum().item(), 0.0)

    # ── FlashMLA sparse decode ──
    def _fp8_sparse_pool(self, slots=512):
        from sglang.kernels.ops.attention.dsa.quant_k_cache import (
            quantize_k_cache_separate,
        )

        kv_bf16 = _rand(slots, 1, 576)
        nope_u8, rope_u8 = quantize_k_cache_separate(
            kv_bf16[:, :, :512], kv_bf16[:, :, 512:]
        )
        kv_fp8 = torch.empty(slots, 1, 656, dtype=torch.uint8, device=_DEVICE)
        kv_fp8[:, :, :528] = nope_u8
        kv_fp8[:, :, 528:] = rope_u8
        return kv_fp8.view(torch.float8_e4m3fn).view(-1, 1, 1, 656)

    def test_flash_fwd_splitkv_mla_fp8_sparse_kernel(self):
        """``flash_fwd_splitkv_mla_fp8_sparse_kernel``：FlashMLA fp8 稀疏 decode 主 kernel。

        trace 里的 ``flash_fwd_splitkv_mla_fp8_sparse_kernel`` 由 FlashMLA 内部选核，
        无法算子级点名；这里以 MLA 的真实 fp8 paged KV（656B/token）+ 稀疏 index 调用
        ``flash_mla_with_kvcache(is_fp8_kvcache=True)`` **真跑**该后端，并核对形状/有限性。
        """
        import sgl_kernel.flash_mla as fm

        bs, H, DV, topk = 2, 64, 512, 64
        q = _rand(bs, 1, H, 576)
        kv_fp8 = self._fp8_sparse_pool(slots=512)
        indices = torch.randint(
            0, 512, (bs, 1, topk), dtype=torch.int32, device=_DEVICE
        )
        o, _ = fm.flash_mla_with_kvcache(
            q=q,
            k_cache=kv_fp8,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=DV,
            tile_scheduler_metadata=fm.get_mla_metadata()[0],
            softmax_scale=1.0 / (256.0 ** 0.5),
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
        )
        self.assertEqual(o.shape, (bs, 1, H, DV))
        self.assertEqual(o.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(o).all())
        self.assertGreater(o.abs().sum().item(), 0.0)

    def test_flash_fwd_mla_combine_kernel(self):
        """``flash_fwd_mla_combine_kernel``：FlashMLA split-KV 的 LSE/输出归约。

        多请求 + 宽 top-k 触发 FlashMLA 的 split-KV 调度（``num_splits`` 由
        ``get_mla_metadata`` 给出），从而真实执行 combine kernel；核对形状/有限性。
        """
        import sgl_kernel.flash_mla as fm

        bs, H, DV, topk = 8, 64, 512, 512
        q = _rand(bs, 1, H, 576)
        kv_fp8 = self._fp8_sparse_pool(slots=4096)
        indices = torch.randint(
            0, 4096, (bs, 1, topk), dtype=torch.int32, device=_DEVICE
        )
        metadata, num_splits = fm.get_mla_metadata()
        o, lse = fm.flash_mla_with_kvcache(
            q=q,
            k_cache=kv_fp8,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=DV,
            tile_scheduler_metadata=metadata,
            num_splits=num_splits,
            softmax_scale=1.0 / (256.0 ** 0.5),
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
        )
        self.assertEqual(o.shape, (bs, 1, H, DV))
        self.assertTrue(torch.isfinite(o).all())

    def test_concat_mla_absorb_q_kernel(self):
        """``concat_mla_absorb_q_kernel``：把 q_nope 与 q_rope 拼成 absorbed q。"""
        from sglang.kernels.ops.attention.concat_mla import concat_mla_absorb_q

        tokens, heads = 4, 8
        a = _rand(tokens, heads, 512)
        b = _rand(tokens, heads, 64)
        out = concat_mla_absorb_q(a, b)
        self.assertEqual(out.shape, (tokens, heads, 576))
        self.assertEqual(out.dtype, _DTYPE)
        self.assertTrue(torch.equal(out[..., :512], a))
        self.assertTrue(torch.equal(out[..., 512:], b))

    def test_fused_rope_kernel(self):
        """``fused_rope_kernel``（``fused_rope_inplace``）：MLA Q/K 交错 RoPE。"""
        from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace

        tokens, heads, rope_dim, max_pos = 6, 2, 64, 64
        q = _rand(tokens, heads, rope_dim)
        k = _rand(tokens, heads, rope_dim)
        freqs = torch.randn(max_pos, rope_dim // 2, dtype=torch.float32, device=_DEVICE)
        freqs_cis = torch.complex(freqs, torch.randn_like(freqs))
        positions = torch.randint(
            0, max_pos, (tokens,), dtype=torch.int32, device=_DEVICE
        )
        q_ref = q.clone()
        fused_rope_inplace(q, k, freqs_cis, positions, inverse=False)
        self.assertEqual(q.shape, (tokens, heads, rope_dim))
        self.assertTrue(torch.isfinite(q).all())
        freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
        for t in range(tokens):
            f = freqs_real[int(positions[t])].float().reshape(rope_dim // 2, 2)
            fr, fi = f[:, 0], f[:, 1]
            xr = q_ref[t].float()[..., 0::2]
            xi = q_ref[t].float()[..., 1::2]
            ref_re = xr * fr - xi * fi
            ref_im = xr * fi + xi * fr
            got_re = q[t].float()[..., 0::2]
            got_im = q[t].float()[..., 1::2]
            self.assertTrue(torch.allclose(got_re, ref_re, atol=3e-2, rtol=3e-2))
            self.assertTrue(torch.allclose(got_im, ref_im, atol=3e-2, rtol=3e-2))
        self.assertFalse(torch.equal(q, q_ref))  # rope 真的改了 q

    def test_set_mla_kv_buffer_kernel(self):
        """``set_mla_kv_buffer_kernel``：把 [nope | rope] 行写入 paged KV buffer。"""
        from sglang.kernels.ops.kvcache.set_mla_kv_buffer import set_mla_kv_buffer

        nope_dim, rope_dim = 528, 128
        kv_buffer = torch.zeros(
            8, nope_dim + rope_dim, dtype=torch.uint8, device=_DEVICE
        )
        # slot 0 默认被保留给 CUDA-graph padding（reserved_skip_index=0），故写到 1/3。
        loc = torch.tensor([1, 3], dtype=torch.int64, device=_DEVICE)
        nope = torch.randint(1, 256, (2, nope_dim), dtype=torch.uint8, device=_DEVICE)
        rope = torch.randint(1, 256, (2, rope_dim), dtype=torch.uint8, device=_DEVICE)
        set_mla_kv_buffer(kv_buffer, loc, nope, rope)
        expected = torch.cat([nope, rope], dim=-1)
        self.assertTrue(torch.equal(kv_buffer[1], expected[0]))
        self.assertTrue(torch.equal(kv_buffer[3], expected[1]))

    def test_quantize_k_cache_fast_kernel(self):
        """``_quantize_k_cache_fast_kernel``：K 的 nope 量化 + rope 拷贝（打包）。"""
        from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache

        k = _rand(2, 4, 1, 576)
        packed = quantize_k_cache(k)
        # [num_blocks, block_size, 1, 512(fp8) + 4*4(scale fp32) + 128(rope bf16)]。
        self.assertEqual(packed.shape, (2, 4, 1, 656))
        self.assertEqual(packed.dtype, torch.float8_e4m3fn)
        # rope 部分（最后 128 字节）是 bf16 精确拷贝。
        rope_out = packed.view(torch.uint8)[..., 528:].view(torch.bfloat16)
        rope_ref = k[..., 512:].reshape(2, 4, 1, 64)
        self.assertTrue(torch.equal(rope_out, rope_ref))

    # ── 索引构建 / 收集 ──
    def test_append_indexer_swa_kernel(self):
        """``_append_indexer_swa_kernel``：拼出 top-k ∪ 最近窗口的稀疏索引行。"""
        from sglang.srt.internal.layers.attention.mla_hca.dsa_swa_index import (
            append_indexer_swa_triton,
        )

        rows, topk, window, page_size = 4, 8, 4, 64
        topk_indices = torch.randint(
            0, 4096, (rows, topk), dtype=torch.int32, device=_DEVICE
        )
        seq_lens = torch.tensor([10, 20, 30, 40], dtype=torch.int32, device=_DEVICE)
        page_table = torch.arange(
            rows * 8, dtype=torch.int32, device=_DEVICE
        ).view(rows, 8)
        out, lengths = append_indexer_swa_triton(
            topk_indices,
            seq_lens,
            num_rows=rows,
            topk=topk,
            window=window,
            alignment=8,
            page_table=page_table,
            page_size=page_size,
        )
        self.assertEqual(out.shape[0], rows)
        self.assertEqual(lengths.shape, (rows,))
        expected = torch.clamp(seq_lens - window, min=0).clamp(max=topk) + torch.clamp(
            seq_lens, max=window
        )
        self.assertTrue(torch.equal(lengths, expected.to(torch.int32)))

    def test_get_k_and_s_triton_kernel(self):
        """``_get_k_and_s_triton_kernel``：从 paged indexer 缓存收集 K 与 scale。"""
        from types import SimpleNamespace

        from sglang.kernels.ops.attention.dsa.index_buf_accessor import GetKAndS

        page_size, head_dim = 64, 128
        num_pages = 4
        buf = torch.randint(
            0,
            256,
            (num_pages, page_size * (head_dim + 4)),
            dtype=torch.uint8,
            device=_DEVICE,
        )
        pool = SimpleNamespace(
            page_size=page_size,
            index_head_dim=head_dim,
            quant_block_size=head_dim,
            device=_DEVICE,
        )
        page_indices = torch.arange(2, dtype=torch.int32, device=_DEVICE).view(1, 2)
        seq_len_tensor = torch.tensor([100], dtype=torch.int32, device=_DEVICE)
        k_out, s_out = GetKAndS.triton(
            pool, buf, page_indices, seq_len_tensor, 100, 100
        )
        self.assertEqual(k_out.shape, (100, head_dim))
        self.assertEqual(s_out.shape, (100, 4))
        self.assertEqual(k_out.dtype, torch.uint8)
        self.assertTrue(torch.equal(k_out[0], buf[0, :head_dim]))
        self.assertTrue(
            torch.equal(
                s_out[0], buf[0, page_size * head_dim : page_size * head_dim + 4]
            )
        )

    # ── cuBLASLt 选核（nvjet / splitK）──
    def _gemm_bf16(self, M, N, K, *, batch=None):
        if batch is None:
            a = _rand(M, K)
            b = _rand(N, K)
            out = torch.nn.functional.linear(a, b)
            ref = a.float() @ b.float().t()
        else:
            a = _rand(batch, M, K)
            b = _rand(batch, K, N)
            out = torch.bmm(a, b)
            ref = torch.bmm(a.float(), b.float())
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out.float(), ref, atol=5e-2, rtol=5e-2))

    def test_nvjet_sm100_tst_256x24_64x6_2x1_v_bz_TNT(self):
        """absorb ``w_kc`` bmm（``nvjet_sm100_tst_256x24_...`` 的 cuBLASLt 选核）。

        MLA absorb 几何：[T, nh, kv_lora=512] x w_kc[nh, 512, qk_nope=192]（batched）。
        """
        self._gemm_bf16(4, 192, 512, batch=8)

    def test_nvjet_sm100_tst_128x24_64x11_2x1_v_bz_TNT(self):
        """absorb ``w_vc`` bmm（``nvjet_sm100_tst_128x24_...`` 的 cuBLASLt 选核）。"""
        self._gemm_bf16(4, 512, 512, batch=8)

    def test_nvjet_sm100_tst_64x32_64x16_2x2_2cta_h_bz_NNT(self):
        """``o_proj`` GEMM（``nvjet_sm100_tst_64x32_..._NNT`` 的 cuBLASLt 选核）。"""
        self._gemm_bf16(4, 512, 512)

    def test_nvjet_sm100_tst_64x8_64x16_1x4_h_bz_splitK_TNT(self):
        """splitK GEMM（``nvjet_sm100_tst_64x8_..._splitK_TNT`` 的 cuBLASLt 选核）。"""
        self._gemm_bf16(4, 64, 4096)

    def test_cublasLt_splitKreduce_kernel(self):
        """``cublasLt::splitKreduce_kernel``：splitK GEMM 的归约阶段。

        小 M/N + 大 K 会触发 cuBLASLt 的 splitK 分解，其归约即
        ``cublasLt::splitKreduce_kernel``；这里真跑该 GEMM 并核对数值。
        """
        self._gemm_bf16(2, 8, 8192)

    # ── gated attention / 调度元数据 ──
    def test_eb5_sigmoid_gate_mul_kernel(self):
        """``_eb5_sigmoid_gate_mul_kernel``：gated MLA 输出的 sigmoid 门控乘。"""
        from sglang.srt.internal.layers.attention.eb5_sigmoid_gate_mul import (
            eb5_sigmoid_gate_mul,
            eb5_sigmoid_gate_mul_torch,
        )

        x = _rand(4, 512)
        gate = _rand(4, 512)
        self.assertTrue(
            torch.equal(
                eb5_sigmoid_gate_mul(x, gate), eb5_sigmoid_gate_mul_torch(x, gate)
            )
        )

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
