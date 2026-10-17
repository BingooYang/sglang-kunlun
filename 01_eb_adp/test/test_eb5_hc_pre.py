"""EB5-lite **hc_pre_attn / hc_pre_moe** 小模块的算子级覆盖单测。

按 ku《EBlite 和 dpsk-v4 算子对比》，hc_pre_attn 与 hc_pre_moe 是**同一对 kernel**
（调用次数也相同，各 2 算子 / 86 次），因此两个小模块共用同一组用例，但各自成一个
``CustomTestCase`` 子类（按小模块名各跑一遍，便于按模块名定位失败）。

每个 eb5_lite 侧 kernel 一个独立 test 方法（方法名 = kernel 名，``<``/``>`` → ``_``），
真跑该 kernel（不打桩）并做形状 / 有限性 / 数值断言。脚手架见
``_eb5_kernel_test_common.py``。

hc_pre_attn / hc_pre_moe（ku 表格 eb5_lite 侧 2 算子 / 86 次）：

    sm100_tf32_hc_prenorm_gemm_impl            deep_gemm.tf32_hc_prenorm_gemm
    mhc_pre_big_fuse_with_norm_tilelang_kernel kernels.ops.layernorm.mhc.mhc_pre_big_fuse_with_norm_tilelang

**本模块有意跳过的纯 torch / cub 算子**：无（这两个小模块的 eb5_lite 列只有上面两个
第三方 kernel）。
"""

from __future__ import annotations

import unittest

import torch

from _eb5_kernel_test_common import (
    CustomTestCase,
    _DEVICE,
    _DTYPE,
    _HC_HIDDEN,
    _HC_MULT,
    _HC_MULT3,
    _HC_TOKENS,
    _rand,
    _require_cuda,
)
from sglang.test_internal.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, num_gpus=1)


class TestHcPreAttnKernels(CustomTestCase):
    """hc_pre_attn 小模块的 kernel（每个 kernel 一个 test 方法，均真跑）。"""

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    def _inputs(self):
        residual = _rand(_HC_TOKENS, _HC_MULT, _HC_HIDDEN)
        fn = (
            torch.randn(
                _HC_MULT3, _HC_MULT * _HC_HIDDEN, dtype=torch.float32, device=_DEVICE
            )
            * 0.02
        )
        hc_scale = torch.ones(3, dtype=torch.float32, device=_DEVICE)
        hc_base = torch.zeros(_HC_MULT3, dtype=torch.float32, device=_DEVICE)
        norm_weight = _rand(_HC_HIDDEN)
        return residual, fn, hc_scale, hc_base, norm_weight

    def test_sm100_tf32_hc_prenorm_gemm_impl(self):
        """``sm100_tf32_hc_prenorm_gemm_impl``：mHC 投影 GEMM + 平方和。"""
        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
            tf32_hc_prenorm_gemm,
        )

        residual, fn, _, _, _ = self._inputs()
        num_tokens = residual.shape[0]
        hc_hidden = _HC_MULT * _HC_HIDDEN
        n_splits = 1
        gemm_out_mul = torch.empty(
            n_splits, num_tokens, _HC_MULT3, dtype=torch.float32, device=_DEVICE
        )
        gemm_out_sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=_DEVICE
        )
        tf32_hc_prenorm_gemm(
            residual.view(num_tokens, hc_hidden),
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
        # gemm_out_mul[0] == x @ fn.T；sqrsum == sum(x^2)。
        ref_mul = residual.view(num_tokens, hc_hidden).float() @ fn.float().t()
        ref_sqr = (residual.view(num_tokens, hc_hidden).float() ** 2).sum(-1)
        self.assertEqual(gemm_out_mul.shape, (n_splits, num_tokens, _HC_MULT3))
        self.assertEqual(gemm_out_sqrsum.shape, (n_splits, num_tokens))
        self.assertTrue(
            torch.allclose(gemm_out_mul[0], ref_mul, atol=2e-1, rtol=2e-2),
            "tf32_hc_prenorm_gemm 的 GEMM 输出与 fp32 参考偏差过大",
        )
        self.assertTrue(
            torch.allclose(gemm_out_sqrsum[0], ref_sqr, atol=2e-1, rtol=2e-2),
            "tf32_hc_prenorm_gemm 的 sqrsum 与 fp32 参考偏差过大",
        )

    def test_mhc_pre_big_fuse_with_norm_tilelang_kernel(self):
        """tilelang ``mhc_pre_big_fuse_with_norm_tilelang``：sinkhorn + RMSNorm 融合。"""
        from sglang.kernels.ops.layernorm.mhc import (
            mhc_pre_big_fuse_with_norm_tilelang,
        )
        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
            tf32_hc_prenorm_gemm,
        )

        residual, fn, hc_scale, hc_base, norm_weight = self._inputs()
        num_tokens = residual.shape[0]
        hc_hidden = _HC_MULT * _HC_HIDDEN
        n_splits = 1
        gemm_out_mul = torch.empty(
            n_splits, num_tokens, _HC_MULT3, dtype=torch.float32, device=_DEVICE
        )
        gemm_out_sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=_DEVICE
        )
        tf32_hc_prenorm_gemm(
            residual.view(num_tokens, hc_hidden),
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
        post_mix = torch.empty(num_tokens, _HC_MULT, dtype=torch.float32, device=_DEVICE)
        comb_mix = torch.empty(
            num_tokens, _HC_MULT * _HC_MULT, dtype=torch.float32, device=_DEVICE
        )
        layer_input = torch.empty(num_tokens, _HC_HIDDEN, dtype=_DTYPE, device=_DEVICE)
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight,
            _HC_HIDDEN,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            2,
            1e-6,
            n_splits=n_splits,
            hc_mult=_HC_MULT,
        )
        self.assertEqual(post_mix.shape, (num_tokens, _HC_MULT))
        self.assertEqual(comb_mix.shape, (num_tokens, _HC_MULT * _HC_MULT))
        self.assertEqual(layer_input.shape, (num_tokens, _HC_HIDDEN))
        self.assertEqual(layer_input.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(post_mix).all())
        self.assertTrue(torch.isfinite(comb_mix).all())
        self.assertTrue(torch.isfinite(layer_input).all())
        # sinkhorn 归一化后，comb 的列和为 1。
        comb = comb_mix.view(num_tokens, _HC_MULT, _HC_MULT)
        self.assertTrue(
            torch.allclose(
                comb.sum(-2),
                torch.ones(num_tokens, _HC_MULT, device=_DEVICE),
                atol=1e-3,
            )
        )
        self.assertGreater(layer_input.abs().sum().item(), 0.0)


class TestHcPreMoeKernels(TestHcPreAttnKernels):
    """hc_pre_moe 小模块的 kernel。

    ku 表格里 hc_pre_moe 与 hc_pre_attn 是**同一对 kernel**
    （``sm100_tf32_hc_prenorm_gemm_impl`` + ``mhc_pre_big_fuse_with_norm_tilelang_kernel``，
    调用次数相同），故直接继承 :class:`TestHcPreAttnKernels` 的用例，按小模块名各跑
    一遍，以便按模块名定位失败。

    **本模块有意跳过的纯 torch / cub 算子**：无。
    """


if __name__ == "__main__":
    unittest.main()
