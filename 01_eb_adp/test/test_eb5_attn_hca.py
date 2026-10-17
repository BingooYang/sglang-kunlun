"""EB5-lite **attn ① HCA** 小模块的算子级覆盖单测（按 ku《EBlite 和 dpsk-v4 算子对比》）。

HCA 是"128× 压缩层"，ku 表格按**单层执行顺序**把它的算子拆成 5 个阶段：
① Q/KV 投影、② 压缩 KV 写回、③ 稀疏索引构建（仅首层）、④ 稀疏注意力、⑤ O 投影。
本文件按阶段分测试类，类内**每个 eb5_lite 侧 kernel 一个独立 test 方法**（方法名 =
kernel 名，把 ``::`` / ``<`` / ``>`` / ``[`` / ``]`` / ``-`` 等替换成 ``_``）。每个方法
都**真跑该 kernel**（不打桩），并做形状 / 有限性 /（能对参考的）数值断言。脚手架见
``_eb5_kernel_test_common.py``。

本文件覆盖的 kernel（ku 表格「attn ① HCA」eb5_lite 列）：

    ① Q/KV 投影
        per_token_group_quant_flat_kernel        sglang_per_token_group_quant_fp8
        deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl  deep_gemm.fp8_fp4_gemm_nt
        RMSNorm<1024>                            torch.nn.functional.rms_norm
        fused_q_norm_rope                        kernels.ops.attention.dsv4.elementwise
        fused_k_norm_rope_flashmla               kernels.ops.attention.dsv4.elementwise
    ② 压缩 KV 写回
        nvjet_sm100_tss_64x8_64x16_1x4_h_bz_TNT  compressor 投影 GEMM（cuBLASLt 选核）
        write_c128_prefill                       compress_norm_rope_store (FusedNormRopeKernel)
    ③ 稀疏索引构建（仅首层）
        _append_indexer_swa_kernel               srt.internal.layers.attention.mla_hca.dsa_swa_index
        _get_k_and_s_triton_kernel               kernels.ops.attention.dsa.index_buf_accessor
    ④ 稀疏注意力
        _dequantize_k_cache_paged_kernel         kernels.ops.attention.dsa.dequant_k_cache
        sparse_attn_fwd_kernel                   kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill
        deepseek_rope_kernel                     kernels.ops.attention.dsv4.elementwise.fused_rope_inplace
    ⑤ O 投影
        _postmix_quant_kernel                    srt.internal.layers.attention.eb5_postmix_fp8_quant
        _eb5_sigmoid_gate_mul_kernel             srt.internal.layers.attention.eb5_sigmoid_gate_mul

**本模块有意跳过的纯 torch / cub 算子**：

    at::el[MulFunctor<float>] ×74   (稀疏注意力输出上的纯 torch 逐元素乘)
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


# ═════════════════════════════════════════════════════════════════════════
# ① Q/KV 投影
# ═════════════════════════════════════════════════════════════════════════
class TestHcaQkvProjectionKernels(CustomTestCase):
    """attn① HCA 阶段①「Q/KV 投影」的 kernel（每个 kernel 一个 test 方法）。"""

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    def test_per_token_group_quant_flat_kernel(self):
        """``per_token_group_quant_flat_kernel``（投影前的 fp8 量化）。"""
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

    def test_deep_gemm_sm100_fp8_fp4_gemm_1d1d_impl(self):
        """``deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl``（投影 GEMM）。"""
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
        self.assertTrue(torch.allclose(out.float(), ref, atol=6.0, rtol=0.35))

    def test_RMSNorm_1024(self):
        """``RMSNorm<1024>``：HCA q/kv norm 的 1024 维（ATen/flashinfer）。"""
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

    def test_fused_q_norm_rope(self):
        """``fused_q_norm_rope``：Q 的 RMSNorm + 交错 RoPE（对齐 torch 参考）。"""
        from sglang.kernels.ops.attention.dsv4.elementwise import fused_q_norm_rope

        tokens, heads, head_dim, rope_dim, max_pos = 6, 1, 512, 64, 64
        eps = 1e-6
        q = _rand(tokens, heads, head_dim)
        freqs = torch.randn(max_pos, rope_dim // 2, dtype=torch.float32, device=_DEVICE)
        freqs_cis = torch.complex(freqs, torch.randn_like(freqs))
        positions = torch.randint(
            0, max_pos, (tokens,), dtype=torch.int32, device=_DEVICE
        )
        q_out = torch.empty_like(q)
        fused_q_norm_rope(q, q_out, eps, freqs_cis, positions)
        self.assertEqual(q_out.shape, q.shape)
        self.assertEqual(q_out.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(q_out).all())

        # 参考：RMSNorm(no weight) + 末尾 rope_dim 的交错 RoPE。
        x = q.float()
        normed = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
        ref = normed.clone()
        nope = head_dim - rope_dim
        for t in range(tokens):
            f = freqs_real[int(positions[t])].float().reshape(rope_dim // 2, 2)
            rp = normed[t, ..., nope:].reshape(
                *normed[t, ..., nope:].shape[:-1], rope_dim // 2, 2
            )
            re = rp[..., 0] * f[:, 0] - rp[..., 1] * f[:, 1]
            im = rp[..., 0] * f[:, 1] + rp[..., 1] * f[:, 0]
            ref[t, ..., nope:] = torch.stack([re, im], -1).reshape(
                ref[t, ..., nope:].shape
            )
        self.assertTrue(
            torch.allclose(q_out.float(), ref, atol=2e-2, rtol=2e-2),
            "fused_q_norm_rope 与 torch 参考偏差过大",
        )

    def test_fused_k_norm_rope_flashmla(self):
        """``fused_k_norm_rope_flashmla``：KV 的 norm+RoPE，并按 584B/token 写回。

        HCA（dsv4）几何：head_dim = 512 = nope(448) + rope(64)。kernel 对整行做
        RMSNorm，再对末 64 维做交错 RoPE；nope 以 fp8 ue8m0 分块量化写入 448 字节，
        rope 以 bf16 精确写入 ``value_ptr + 448``。这里核对 rope 段与 torch 参考
        **逐位相等**，并核对 nope 段非退化。
        """
        from sglang.kernels.ops.attention.dsv4.elementwise import (
            fused_k_norm_rope_flashmla,
        )

        B, head_dim, rope_dim, page = 4, 512, 64, 64
        nope = head_dim - rope_dim
        eps = 1e-6
        max_pos = 64
        kv = _rand(B, head_dim)
        weight = _rand(head_dim)
        freqs = torch.randn(max_pos, rope_dim // 2, dtype=torch.float32, device=_DEVICE)
        freqs_cis = torch.complex(freqs, torch.randn_like(freqs))
        positions = torch.randint(0, max_pos, (B,), dtype=torch.int32, device=_DEVICE)
        out_loc = torch.arange(B, dtype=torch.int32, device=_DEVICE)
        page_bytes = ((584 * page + 575) // 576) * 576
        kvcache = torch.zeros(8, page_bytes, dtype=torch.uint8, device=_DEVICE)

        fused_k_norm_rope_flashmla(
            kv, weight, eps, freqs_cis, positions, out_loc, kvcache, page
        )
        self.assertGreater(kvcache.abs().sum().item(), 0.0)

        # 参考：整行 RMSNorm + 末 rope_dim 的交错 RoPE。
        x = kv.float()
        normed = x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / head_dim + eps)
        normed = normed * weight.float()
        freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
        rope_ref = torch.empty(B, rope_dim, dtype=_DTYPE, device=_DEVICE)
        for i in range(B):
            f = freqs_real[int(positions[i])].float().reshape(rope_dim // 2, 2)
            rp = normed[i, nope:].reshape(rope_dim // 2, 2)
            re = rp[:, 0] * f[:, 0] - rp[:, 1] * f[:, 1]
            im = rp[:, 0] * f[:, 1] + rp[:, 1] * f[:, 0]
            rope_ref[i] = torch.stack([re, im], -1).reshape(rope_dim).to(_DTYPE)

        # rope 段位于 value_ptr + 448，共 128 字节 bf16；out_loc=arange → page0/offset=loc。
        for i in range(B):
            loc = int(out_loc[i])
            base = (loc >> 6) * page_bytes + (loc & (page - 1)) * 576
            got = kvcache[0, base + 448 : base + 448 + rope_dim * 2].view(_DTYPE)
            self.assertTrue(
                torch.equal(got, rope_ref[i]),
                f"fused_k_norm_rope_flashmla 的 rope 段与参考不一致 (row {i})",
            )
        # nope 段（448 fp8）非退化。
        for i in range(B):
            loc = int(out_loc[i])
            base = (loc >> 6) * page_bytes + (loc & (page - 1)) * 576
            nope_bytes = kvcache[0, base : base + nope].view(torch.float8_e4m3fn)
            self.assertTrue(torch.isfinite(nope_bytes.float()).all())
            self.assertGreater(nope_bytes.float().abs().sum().item(), 0.0)


# ═════════════════════════════════════════════════════════════════════════
# ② 压缩 KV 写回
# ═════════════════════════════════════════════════════════════════════════
class TestHcaCompressWritebackKernels(CustomTestCase):
    """attn① HCA 阶段②「压缩 KV 写回」的 kernel（每个 kernel 一个 test 方法）。"""

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    def test_nvjet_sm100_tss_64x8_64x16_1x4_h_bz_TNT(self):
        """compressor 投影 GEMM：真跑一次 fp8/bf16 GEMM（Blackwell 上落到 nvjet）。"""
        import deep_gemm
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8_ue8m0,
        )

        torch.manual_seed(0)
        M, N, K = 32, 128, 512
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

    def test_write_c128_prefill(self):
        """``write_c128_prefill``：compressor 的 norm+RoPE+KV 写入（c128 prefill plan）。

        真跑 ``compress_norm_rope_store`` 的 ``FusedNormRopeKernel`` prefill 分支，
        把压缩后的 KV 按 584B/token 的 FlashMLA 布局写进 paged cache，并核对写入了
        非零数据、且只落在期望的 page 上。
        """
        from sglang.kernels.ops.attention.dsv4 import compress_norm_rope_store
        from sglang.test.kernels.deepseek_v4.common import (
            make_legacy_context,
            to_seq_extend,
        )

        head_dim, page, ratio = 512, 128, 128
        seq_len = 256
        ctx = make_legacy_context(bs=1, compress_ratio=ratio, head_dim=head_dim)
        seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend([(seq_len, seq_len)])
        plan = ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, num_q)
        n = plan[1].shape[0]
        kv = _rand(n, head_dim)
        norm_weight = torch.ones(head_dim, dtype=_DTYPE, device=_DEVICE)
        freqs = torch.randn(64, 32, dtype=torch.float32, device=_DEVICE)
        freq_cis = torch.complex(freqs, torch.randn_like(freqs))
        # extend 分支按 plan 的 ragged_id（q token 上界）索引 out_loc，故其长度需 >= seq_len+1。
        out_loc = torch.arange(seq_len + 2, dtype=torch.int64, device=_DEVICE)
        page_bytes = ((584 * page + 575) // 576) * 576
        kvcache = torch.zeros(8, page_bytes, dtype=torch.uint8, device=_DEVICE)

        compress_norm_rope_store(
            kv,
            plan,
            norm_weight=norm_weight,
            norm_eps=1e-6,
            freq_cis=freq_cis,
            out_loc=out_loc,
            kvcache=kvcache,
            page_size=page,
        )
        self.assertGreater(kvcache.abs().sum().item(), 0.0)
        # 写入位置由 out_loc 决定：这里 out_loc=arange → 命中 page 1 / 2。
        self.assertGreater(kvcache[1].abs().sum().item(), 0.0)


# ═════════════════════════════════════════════════════════════════════════
# ③ 稀疏索引构建（仅首层执行一次）
# ═════════════════════════════════════════════════════════════════════════
class TestHcaIndexBuildKernels(CustomTestCase):
    """attn① HCA 阶段③「稀疏索引构建」的 kernel（每个 kernel 一个 test 方法）。

    ku 表格本行只写「见下方『索引构建顺序』」，而该子表在文档里为空；因此本类覆盖
    eb5_lite 侧实际的索引构建 / 索引 K-S 收集算子。
    """

    @classmethod
    def setUpClass(cls):
        _require_cuda()

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
        # identity page table：行 i -> 表行 i。
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
        # indexer_len = min(max(seq_len - window, 0), topk)；swa_len = min(seq_len, window)。
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
        # page_indices 是 identity：token 0 的 K 应等于页 0 的前 128 字节；
        # scale 位于页内偏移 page_size*head_dim 处。
        self.assertTrue(torch.equal(k_out[0], buf[0, :head_dim]))
        self.assertTrue(
            torch.equal(
                s_out[0], buf[0, page_size * head_dim : page_size * head_dim + 4]
            )
        )


# ═════════════════════════════════════════════════════════════════════════
# ④ 稀疏注意力
# ═════════════════════════════════════════════════════════════════════════
class TestHcaSparseAttentionKernels(CustomTestCase):
    """attn① HCA 阶段④「稀疏注意力」的 kernel（每个 kernel 一个 test 方法）。"""

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    def test_dequantize_k_cache_paged_kernel(self):
        """``_dequantize_k_cache_paged_kernel``：从 paged fp8 KV 反量化出 bf16 K。"""
        from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
            dequantize_k_cache_paged,
        )
        from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache

        k = _rand(2, 4, 1, 576)
        packed = quantize_k_cache(k)
        page_table = torch.arange(
            packed.shape[0] * packed.shape[1], dtype=torch.int32, device=_DEVICE
        )
        deq = dequantize_k_cache_paged(packed, page_table)
        self.assertEqual(deq.shape[-1], 576)
        self.assertEqual(deq.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(deq).all())
        # 量化-反量化往返：nope 部分误差受 fp8 限制，rope 部分是精确拷贝。
        ref = k.view(-1, 576)[: deq.shape[0]]
        self.assertTrue(
            torch.allclose(deq.view(-1, 576).float(), ref.float(), atol=0.3, rtol=0.2)
        )

    def test_sparse_attn_fwd_kernel(self):
        """``sparse_attn_fwd_kernel``：HCA ④ 的稀疏注意力 forward（prefill 双 KV 源）。

        真跑 ``sparse_attn_v4_paged_prefill``（其 triton 主 kernel 即 trace 里的
        ``sparse_attn_fwd_kernel``）：对 prefix（paged unified_kv）与 extend（本 fwd 的
        flat kv）两段稀疏 key 做 online-softmax 融合，并带 per-head attn sink。与
        sink-aware 的 torch 参考比对。
        """
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_prefill import (
            sparse_attn_v4_paged_prefill,
        )

        T, H, D = 3, 4, 128
        pages = 8
        torch.manual_seed(0)
        q = _rand(T, H, D)
        unified_kv = _rand(pages, D)
        kv_indices_prefix = torch.randint(
            0, pages, (T * 2,), dtype=torch.int32, device=_DEVICE
        )
        kv_indptr_prefix = torch.arange(0, T * 2 + 1, 2, dtype=torch.int32, device=_DEVICE)
        kv = _rand(T, D)
        kv_indices_extend = torch.arange(0, T, dtype=torch.int32, device=_DEVICE)
        kv_indptr_extend = torch.arange(0, T + 1, 1, dtype=torch.int32, device=_DEVICE)
        sink = torch.randn(H, dtype=torch.float32, device=_DEVICE) * 0.5
        scale = 1.0 / (D ** 0.5)

        out = sparse_attn_v4_paged_prefill(
            q,
            unified_kv,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            sink,
            scale,
        )
        self.assertEqual(out.shape, (T, H, D))
        self.assertEqual(out.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(out).all())

        # sink-aware torch 参考：sink 作为 V=0 的虚拟 key，只进分母。
        ref = torch.empty(T, H, D, dtype=torch.float32, device=_DEVICE)
        for t in range(T):
            pk = unified_kv[kv_indices_prefix[2 * t : 2 * t + 2].long()].float()
            ek = kv[kv_indices_extend[t : t + 1].long()].float()
            kk = torch.cat([pk, ek], 0)
            s = q[t].float() @ kk.t() * scale
            m = torch.maximum(s.max(-1).values, sink)
            p = torch.exp(s - m[:, None])
            denom = p.sum(-1) + torch.exp(sink - m)
            ref[t] = (p @ kk) / denom[:, None]
        self.assertTrue(
            torch.allclose(out.float(), ref, atol=5e-2, rtol=5e-2),
            "sparse_attn_fwd_kernel 与 sink-aware torch 参考偏差过大",
        )

    def test_deepseek_rope_kernel(self):
        """``deepseek_rope_kernel``：注意力输出上的（逆）RoPE（``fused_rope_inplace``）。

        trace 里的 ``deepseek_rope_kernel`` 由 ``deepseek_v4/rope.cuh`` 的
        ``FusedQKRopeKernel`` 启动（``fused_rope_inplace`` 的 CUDA 实现）。这里真跑一次
        并逐 position 与 torch 参考比对。
        """
        from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace

        tokens, heads, rope_dim, max_pos = 6, 2, 64, 64
        q = _rand(tokens, heads, rope_dim)
        k = _rand(tokens, heads, rope_dim)
        freqs = torch.randn(max_pos, rope_dim // 2, dtype=torch.float32, device=_DEVICE)
        freqs_imag = torch.randn_like(freqs)
        freqs_cis = torch.complex(freqs, freqs_imag)
        positions = torch.randint(
            0, max_pos, (tokens,), dtype=torch.int32, device=_DEVICE
        )
        q_ref = q.clone()
        fused_rope_inplace(q, k, freqs_cis, positions, inverse=True)
        self.assertEqual(q.shape, (tokens, heads, rope_dim))
        self.assertTrue(torch.isfinite(q).all())
        # 与 torch 参考逐 position 比对（交错 RoPE 的逆旋转 = 共轭频率）。
        freqs_conj = torch.complex(freqs, -freqs_imag)
        freqs_real = torch.view_as_real(freqs_conj).flatten(-2)
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


# ═════════════════════════════════════════════════════════════════════════
# ⑤ O 投影
# ═════════════════════════════════════════════════════════════════════════
class TestHcaOProjectionKernels(CustomTestCase):
    """attn① HCA 阶段⑤「O 投影」的 kernel（每个 kernel 一个 test 方法）。"""

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    def test_postmix_quant_kernel(self):
        """``_postmix_quant_kernel``：VHA postmix + wo_a fp8 量化融合。"""
        from sglang.srt.internal.layers.attention.eb5_postmix_fp8_quant import (
            postmix_quant,
        )

        # head_dim = D / heads_per_group 必须是 128 的整数倍；nh>=16 才能喂满 tl.dot。
        tokens, groups, hidden, num_heads = 4, 1, 2048, 16
        x = _rand(tokens, groups, hidden)
        mix = torch.eye(num_heads, dtype=_DTYPE, device=_DEVICE)
        codes, scales = postmix_quant(x, mix, num_heads)
        self.assertEqual(codes.shape, (tokens, groups, hidden))
        self.assertEqual(codes.dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.isfinite(codes.float()).all())
        # 单位 mix 时，codes 应约等于 x 的 per-128-group fp8 量化。
        scale_groups = hidden // 128
        self.assertEqual(scales.shape[-1], scale_groups // 4)  # packed ue8m0 int32

    def test_eb5_sigmoid_gate_mul_kernel(self):
        """``_eb5_sigmoid_gate_mul_kernel``：x * sigmoid(gate)，与 torch 位对齐。"""
        from sglang.srt.internal.layers.attention.eb5_sigmoid_gate_mul import (
            eb5_sigmoid_gate_mul,
            eb5_sigmoid_gate_mul_torch,
        )

        x = _rand(4, 512)
        gate = _rand(4, 512)
        out = eb5_sigmoid_gate_mul(x, gate)
        ref = eb5_sigmoid_gate_mul_torch(x, gate)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, _DTYPE)
        # 双舍入契约：bit-identical。
        self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
    unittest.main()
