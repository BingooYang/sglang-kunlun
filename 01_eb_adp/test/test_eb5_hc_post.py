"""EB5-lite **hc_post_attn / hc_post_moe** 小模块的算子级覆盖单测。

按 ku《EBlite 和 dpsk-v4 算子对比》，hc_post_attn 与 hc_post_moe 共用**同一个**
tilelang kernel ``mhc_post_tilelang_kernel``，因此两个小模块共用同一组用例，但各自成
一个 ``CustomTestCase`` 子类（按小模块名各跑一遍，便于按模块名定位失败）。

每个 eb5_lite 侧 kernel 一个独立 test 方法，真跑该 kernel（不打桩）并做形状 / 有限性 /
数值断言。脚手架见 ``_eb5_kernel_test_common.py``。

hc_post_attn / hc_post_moe（ku 表格 eb5_lite 侧）：

    mhc_post_tilelang_kernel    kernels.ops.layernorm.mhc.mhc_post_tilelang

**本模块有意跳过的纯 torch / cub 算子**：

    at::el[CUDAFunctor_add<c10::BFloat16>] ×42  (hc_post_moe 的纯 torch 逐元素加)
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


class TestHcPostAttnKernels(CustomTestCase):
    """hc_post_attn 小模块的 kernel（每个 kernel 一个 test 方法，均真跑）。"""

    @classmethod
    def setUpClass(cls):
        _require_cuda()

    def test_mhc_post_tilelang_kernel(self):
        """tilelang ``mhc_post_tilelang``：x = c*d + a^T @ b，与 torch 参考一致。"""
        from sglang.kernels.ops.layernorm.mhc import mhc_post_tilelang

        n, hc, hidden = 8, _HC_MULT, 1024
        a = torch.randn(n, hc, hc, dtype=torch.float32, device=_DEVICE) * 0.1
        b = _rand(n, hc, hidden)
        c = torch.rand(n, hc, dtype=torch.float32, device=_DEVICE)
        d = _rand(n, hidden)
        x = torch.empty(n, hc, hidden, dtype=_DTYPE, device=_DEVICE)
        mhc_post_tilelang(a, b, c, d, x, hc, hidden)

        # 参考：x[n, hco, h] = c[n, hco]*d[n, h] + sum_hci a[n, hci, hco]*b[n, hci, h]
        ref = (
            c[:, :, None] * d[:, None, :]
            + torch.einsum("nhc,nhd->ncd", a.float(), b.float())
        ).to(_DTYPE)
        self.assertEqual(x.shape, (n, hc, hidden))
        self.assertEqual(x.dtype, _DTYPE)
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue(
            torch.allclose(x.float(), ref.float(), atol=3e-2, rtol=3e-2),
            "mhc_post_tilelang 与 torch 参考偏差过大",
        )


class TestHcPostMoeKernels(TestHcPostAttnKernels):
    """hc_post_moe 小模块的 kernel。

    ku 表格里 hc_post_moe 与 hc_post_attn 用**同一个** tilelang
    ``mhc_post_tilelang_kernel``，故直接继承 :class:`TestHcPostAttnKernels` 的用例，
    按小模块名各跑一遍，以便按模块名定位失败。

    **本模块有意跳过的纯 torch / cub 算子**：``at::el[CUDAFunctor_add<c10::BFloat16>]``
    ×42（纯 torch 逐元素加）。
    """


if __name__ == "__main__":
    unittest.main()
