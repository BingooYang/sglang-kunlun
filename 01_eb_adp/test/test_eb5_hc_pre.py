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

# ═════════════════════════════════════════════════════════════════════════
# hc_pre 的 sqrsum / splitk 家族（适配层**自己新增**的 torch 拆解）
#
# 为什么单独立一组：`mhc_pre_gemm_sqrsum_splitk_kernel` / `_tilelang` / `_dispatch`
# 这三个**不在** ku《EBlite 和 dpsk-v4 算子对比》的第三方 kernel 清单里（那张表只有
# `sm100_tf32_hc_prenorm_gemm_impl` 和 `mhc_pre_big_fuse_with_norm_tilelang_kernel`），
# 于是按"清单枚举"的覆盖模型**天然漏掉了它们** —— 2026-10-17 EB5 整网首次前向
# 正是炸在 `kernel_0` 上（`KL_XID_KERNEL_EXCEPTION` / `wait for noc idle timeout`）。
#
# 调用约定对齐 `sglang/kernels/ops/layernorm/mhc.py::mhc_pre`，真实配置
# hc_mult=4 / hidden=4096 → hc_hidden = hc_mult*hidden = 16384、hc_mult3 = 24：
#   * splitk 路径（`SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`，**服务脚本用的就是这条**）：
#       partial_out=(split_k, N, 32)、partial_sqrsum=(split_k, N)、gemm_last_dim=32
#   * tilelang 路径（该 env 默认 1，**原来的单测只走了这条**）：
#       out=(N, hc_mult3)、sqrsum=(N,)、gemm_last_dim=hc_mult3
# 两条路径的 out 维度就不同（3-D vs 2-D）—— 这正是"单测全绿、整网炸"的一处结构性原因。
#
# 这里刻意从 **`sglang.kernels.ops.layernorm.mhc`** 取符号（适配层补丁替换后的那个），
# 而不是直接 import `eb5_kunlun_adapt` —— 这样测的就是模型真正会调到的对象。
# ═════════════════════════════════════════════════════════════════════════
class TestHcPreSqrsumFamily(CustomTestCase):
    HC_MULT = 4
    HIDDEN = 4096
    HC_HIDDEN = HC_MULT * HIDDEN          # 16384
    HC_MULT3 = HC_MULT * 2 + HC_MULT**2   # 24
    # 服务实际用的 split_k：`ernie5._mhc_pre` **不传** `n_splits_pre`，走
    # `mhc.mhc_pre` 签名默认值 **32**（`mhc.py:1055`）。原先这里写死 16，
    # 于是「单测全绿、整网炸」——16 和 32 是两条不同的 kernel 形状。
    SPLIT_K = 32
    # 覆盖 mhc.py 里 num_tokens <= 2048 的分发阈值两侧
    TOKEN_SWEEP = (8, 256, 2048, 2049)

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    def _make_inputs(self, num_tokens):
        x = _rand(num_tokens, self.HC_HIDDEN)
        fn = (
            torch.randn(
                self.HC_MULT3, self.HC_HIDDEN, dtype=torch.float32, device=_DEVICE
            )
            * 0.02
        )
        return x, fn

    def _ref_splitk(self, x, fn, split_k):
        """fp32 参考：逐段 GEMM + 逐段平方和。"""
        xf, f = x.float(), fn.float()
        n = xf.shape[0]
        j = min(self.HC_MULT3, f.shape[0])
        split_size = self.HC_HIDDEN // split_k
        out = torch.zeros(split_k, n, 32, dtype=torch.float32, device=x.device)
        sq = torch.zeros(split_k, n, dtype=torch.float32, device=x.device)
        for s in range(split_k):
            lo, hi = s * split_size, (s + 1) * split_size
            out[s, :, :j] = xf[:, lo:hi] @ f[:j, lo:hi].t()
            sq[s] = (xf[:, lo:hi] ** 2).sum(-1)
        return out, sq

    def test_mhc_pre_gemm_sqrsum_splitk_kernel(self):
        """`kernel_0`（split-K）：真实配置 + 真实 token 规模下数值正确。

        这条路径服务端必走（`run_eb5_kunlun.sh` 设 `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`），
        而原来的单测从不经过它。
        """
        from sglang.kernels.ops.layernorm.mhc import (
            mhc_pre_gemm_sqrsum_splitk_kernel,
        )

        kernel_0, _ = mhc_pre_gemm_sqrsum_splitk_kernel(
            self.HC_MULT3, self.HC_HIDDEN, split_k=self.SPLIT_K
        )
        for n in self.TOKEN_SWEEP:
            with self.subTest(num_tokens=n):
                x, fn = self._make_inputs(n)
                out_partial = torch.empty(
                    self.SPLIT_K, n, 32, dtype=torch.float32, device=_DEVICE
                )
                sqrsum_partial = torch.empty(
                    self.SPLIT_K, n, dtype=torch.float32, device=_DEVICE
                )
                kernel_0(x, fn, out_partial, sqrsum_partial)
                torch.cuda.synchronize()
                ref_out, ref_sq = self._ref_splitk(x, fn, self.SPLIT_K)
                self.assertTrue(
                    torch.allclose(out_partial, ref_out, atol=1e-3, rtol=1e-3),
                    f"num_tokens={n}: out_partial 与 fp32 参考不一致",
                )
                self.assertTrue(
                    torch.allclose(sqrsum_partial, ref_sq, atol=1e-3, rtol=1e-3),
                    f"num_tokens={n}: sqrsum_partial 与 fp32 参考不一致",
                )

    def test_mhc_pre_gemm_sqrsum_splitk_kernel_all_splits(self):
        """`kernel_0` 覆盖**所有可能被服务用到的** split_k：16 与 32。

        服务在同一条调用链上有两个取值来源，不能只测一个：
          * 普通路径：`mhc.mhc_pre` 的签名默认 `n_splits_pre=32`（`ernie5` 不传该参数）
          * 确定性推理：`_is_deterministic_inference_enabled()` 时显式改成 16
        """
        from sglang.kernels.ops.layernorm.mhc import (
            mhc_pre_gemm_sqrsum_splitk_kernel,
        )

        for split_k in (16, 32):
            kernel_0, _ = mhc_pre_gemm_sqrsum_splitk_kernel(
                self.HC_MULT3, self.HC_HIDDEN, split_k=split_k
            )
            for n in self.TOKEN_SWEEP:
                with self.subTest(split_k=split_k, num_tokens=n):
                    x, fn = self._make_inputs(n)
                    out_partial = torch.empty(
                        split_k, n, 32, dtype=torch.float32, device=_DEVICE
                    )
                    sqrsum_partial = torch.empty(
                        split_k, n, dtype=torch.float32, device=_DEVICE
                    )
                    kernel_0(x, fn, out_partial, sqrsum_partial)
                    torch.cuda.synchronize()
                    ref_out, ref_sq = self._ref_splitk(x, fn, split_k)
                    self.assertTrue(
                        torch.allclose(out_partial, ref_out, atol=1e-3, rtol=1e-3),
                        f"split_k={split_k} num_tokens={n}: out_partial 不一致",
                    )
                    self.assertTrue(
                        torch.allclose(sqrsum_partial, ref_sq, atol=1e-3, rtol=1e-3),
                        f"split_k={split_k} num_tokens={n}: sqrsum_partial 不一致",
                    )

    def test_kernel_0_matmul_contiguity_contrast(self):
        """对照实验：matmul 操作数先 `.contiguous()` vs 直接喂非连续切片。

        背景：`kernel_0` 里两个操作数**都是非连续的** ——
        `xs = xf[:, lo:hi]`（stride `(16384, 1)`）与 `f[:j, lo:hi].t()`（stride `(1, 16384)`）。
        CPU 上两种写法数值完全相同（已实测），所以这是**纯 XPU 后端行为**问题。

        **顺序是刻意的**：先跑"干净"的连续形式。若非连续形式会触发设备级异常，
        它会把 XPU 上下文带坏、后面全部连带失败；把连续形式放前面，结果就可判定：
          * 都过             → 与连续性无关
          * 连续过 / 非连续炸 → 坐实 strided matmul
          * 连续就炸          → 连续性问题不是（唯一）原因
        """
        n = 2048
        x, fn = self._make_inputs(n)
        xf, f = x.float(), fn.float()
        split_size = self.HC_HIDDEN // self.SPLIT_K
        xs = xf[:, 0:split_size]
        fs_t = f[: self.HC_MULT3, 0:split_size].t()

        with self.subTest(form="contiguous"):
            ref = xs.contiguous() @ fs_t.contiguous()
            torch.cuda.synchronize()
            ref = ref.clone()

        with self.subTest(form="strided"):
            got = xs @ fs_t
            torch.cuda.synchronize()
            self.assertTrue(
                torch.allclose(got, ref, atol=1e-3, rtol=1e-3),
                "连续与非连续两种 matmul 的数值必须一致",
            )

    def test_mhc_pre_gemm_sqrsum_tilelang(self):
        """`mhc_pre_gemm_sqrsum_tilelang`（单链、无 split）：2-D 调用约定。"""
        from sglang.kernels.ops.layernorm.mhc import (
            mhc_pre_gemm_sqrsum_tilelang,
        )

        for n in (8, 2048):
            with self.subTest(num_tokens=n):
                x, fn = self._make_inputs(n)
                out = torch.empty(n, self.HC_MULT3, dtype=torch.float32, device=_DEVICE)
                sqrsum = torch.empty(n, dtype=torch.float32, device=_DEVICE)
                mhc_pre_gemm_sqrsum_tilelang(
                    x, fn, out, sqrsum, self.HC_MULT3, self.HC_HIDDEN
                )
                torch.cuda.synchronize()
                xf, f = x.float(), fn.float()
                ref_out = xf @ f[: self.HC_MULT3].t()
                ref_sq = (xf * xf).sum(-1)
                self.assertTrue(
                    torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3),
                    f"num_tokens={n}: tilelang out 与参考不一致",
                )
                self.assertTrue(
                    torch.allclose(sqrsum, ref_sq, atol=1e-3, rtol=1e-3),
                    f"num_tokens={n}: tilelang sqrsum 与参考不一致",
                )

    def test_mhc_pre_align_fleet_matches_hf_form(self):
        """`mhc_align_fleet=True`（**服务真实取值**）下 big_fuse 数值正确。

        服务端 `EB5DecoderLayer` 按 `model_type` 设置：
        `self.mhc_align_fleet = "ernie" in model_type`，本权重 `model_type="ernie"`
        → **True**。它切换两处算法（`mhc.py` docstring）：
          * mapping norm 用 `1/(sqrt(sum_sq/D)+eps)`（eps 在 sqrt 外）
          * sinkhorn softmax 用 `exp2((x-max)*LOG2E)` 而非 `exp(x-max)`
        原适配层单测一律用 `False`（HF/DSV4 口径），**从未测过 EB5 实际走的那条**。
        """
        from sglang.kernels.ops.layernorm.mhc import (
            mhc_pre_big_fuse_with_norm_tilelang,
            mhc_pre_gemm_sqrsum_splitk_kernel,
        )

        for n in (256, 1024):
            with self.subTest(num_tokens=n):
                x, fn = self._make_inputs(n)
                split_k = self.SPLIT_K
                out_partial = torch.zeros(
                    split_k, n, 32, dtype=torch.float32, device=_DEVICE
                )
                sq_partial = torch.zeros(
                    split_k, n, dtype=torch.float32, device=_DEVICE
                )
                kernel_0, _ = mhc_pre_gemm_sqrsum_splitk_kernel(
                    self.HC_MULT3, self.HC_HIDDEN, split_k=split_k
                )
                kernel_0(x, fn, out_partial, sq_partial)

                residual = _rand(n, self.HC_MULT, self.HIDDEN)
                post_mix = torch.empty(n, self.HC_MULT, dtype=torch.float32, device=_DEVICE)
                comb_mix = torch.empty(
                    n, self.HC_MULT**2, dtype=torch.float32, device=_DEVICE
                )
                layer_input = torch.empty(n, self.HIDDEN, dtype=_DTYPE, device=_DEVICE)
                norm_weight = torch.randn(self.HIDDEN, dtype=_DTYPE, device=_DEVICE)
                hc_scale = torch.ones(3, dtype=torch.float32, device=_DEVICE)
                hc_base = torch.zeros(self.HC_MULT3, dtype=torch.float32, device=_DEVICE)

                mhc_pre_big_fuse_with_norm_tilelang(
                    out_partial, sq_partial, hc_scale, hc_base, residual,
                    post_mix, comb_mix, layer_input, norm_weight,
                    self.HIDDEN, 1e-6, 1e-6, 1e-6, 2.0, 20, 1e-6,
                    split_k, self.HC_MULT, 32, 1,   # mhc_align_fleet=True
                )
                torch.cuda.synchronize()

                self.assertTrue(torch.isfinite(post_mix).all(), f"n={n}: post_mix 非有限")
                self.assertTrue(torch.isfinite(comb_mix).all(), f"n={n}: comb_mix 非有限")
                self.assertTrue(torch.isfinite(layer_input).all(), f"n={n}: layer_input 非有限")
                # sinkhorn 归一化后列和为 1（与服务里 comb 的语义一致）
                self.assertTrue(
                    torch.allclose(
                        comb_mix.view(n, self.HC_MULT, self.HC_MULT).float().sum(-2),
                        torch.ones(n, self.HC_MULT, device=_DEVICE),
                        atol=1e-3,
                    ),
                    f"n={n}: comb 列和不为 1",
                )

    def test_mhc_pre_gemm_sqrsum_dispatch(self):
        """`_mhc_pre_gemm_sqrsum_dispatch()` 返回单链实现（mhc.py 会直接调用它）。"""
        from sglang.kernels.ops.layernorm.mhc import (
            _mhc_pre_gemm_sqrsum_dispatch,
            mhc_pre_gemm_sqrsum_tilelang,
        )

        fn = _mhc_pre_gemm_sqrsum_dispatch()
        self.assertTrue(callable(fn), "dispatch 必须返回可调用对象")
        self.assertIs(fn, mhc_pre_gemm_sqrsum_tilelang)
