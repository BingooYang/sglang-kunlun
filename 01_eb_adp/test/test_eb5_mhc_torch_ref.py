"""EB5 mHC **散算子实现**的数值回归（纯 CPU，不需要 XPU / distributed）。

为什么单独一条：`mhc_pre_torch_ref` / `mhc_post_torch_ref` 是 EB5 在昆仑上的**默认**
路径（`EB5_MHC_TORCH_REF` 默认 1），但前者原先复用的是 `mhc_pre_big_fuse_*` 的
拆解函数，没有和“DSV4 的 `hc_pre_torch_impl` + `hc_split_sinkhorn` + `hc_combine`”
这条已验证路线做过逐元素对照；而 `test_eb5_hc_pre.py` 那套又必须在设备上跑，
服务 warmup 的 1–3 token（见 `problem/2026-10-20-mhc-hang.md` §2.5）完全没覆盖。

本文件把参考实现按 `mhc.py::mhc_pre_big_fuse_with_norm_tilelang` 的公式独立写一遍，
与适配层逐元素比对；只 import torch，可在设备不健康时单独跑：

    python -m unittest test_eb5_mhc_torch_ref -v
"""

from __future__ import annotations

import itertools
import unittest

import torch
import torch.nn.functional as F

import eb5_kunlun_adapt as adapt

_LOG2E = 1.4426950408889634


def _ref_pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    post_mult,
    sinkhorn_repeat,
    norm_weight=None,
    norm_eps=None,
    align_fleet=False,
):
    """`mhc_pre_big_fuse_with_norm_tilelang` 的纯 torch 参考（独立写，不调适配层）。"""
    s, n, h = residual.shape
    x_flat = residual.view(s, n * h).float()

    if align_fleet:
        rms = 1.0 / (torch.sqrt(x_flat.square().mean(-1, keepdim=True)) + rms_eps)
    else:
        rms = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x_flat, fn.float()) * rms

    pre = torch.sigmoid(mixes[:, :n] * hc_scale[0] + hc_base[:n]) + hc_pre_eps
    post = post_mult * torch.sigmoid(
        mixes[:, n : 2 * n] * hc_scale[1] + hc_base[n : 2 * n]
    )
    comb = (mixes[:, 2 * n :] * hc_scale[2] + hc_base[2 * n :]).view(s, n, n)

    row_max = comb.amax(dim=-1, keepdim=True)
    if align_fleet:
        comb = torch.exp2((comb - row_max) * _LOG2E)
    else:
        comb = torch.exp(comb - row_max)
    comb = comb / comb.sum(-1, keepdim=True) + hc_sinkhorn_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_sinkhorn_eps)

    # 与 kernel 的 norm 段一致：先 round 到 bf16，再 fp32 平方 / 乘 fp32 权重。
    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(1).to(residual.dtype)
    if norm_weight is not None:
        xf = layer_input.float()
        layer_input = (
            xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm_eps)
            * norm_weight.float()
        ).to(residual.dtype)

    return post.unsqueeze(-1), comb, layer_input


def _ref_post(x, residual, post, comb):
    return (
        post * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(1)
    ).type_as(x)


class TestMhcTorchRef(unittest.TestCase):
    """散算子实现 vs kernel 公式参考：逐元素相等（同序 fp32 运算）。"""

    def _case(self, hc, hidden, tokens, align, use_norm, dtype):
        gen = torch.Generator().manual_seed(
            hash((hc, hidden, tokens, align, use_norm, str(dtype))) & 0xFFFF
        )
        mix = (2 + hc) * hc
        residual = torch.randn(tokens, hc, hidden, generator=gen, dtype=torch.float32).to(
            dtype
        )
        fn = (
            torch.randn(mix, hc * hidden, generator=gen, dtype=torch.float32) * 0.05
        ).to(dtype)
        hc_scale = torch.randn(3, generator=gen) * 0.5 + 1.0
        hc_base = torch.randn(mix, generator=gen) * 0.1
        norm_weight = (torch.rand(hidden, generator=gen) + 0.5).to(torch.float32)
        norm_eps = 1e-6
        kwargs = dict(
            rms_eps=1e-6,
            hc_pre_eps=1e-6,
            hc_sinkhorn_eps=1e-6,
            post_mult=2.0,
            sinkhorn_repeat=4,
            norm_weight=norm_weight if use_norm else None,
            norm_eps=norm_eps if use_norm else None,
            align_fleet=align,
        )

        got = adapt.mhc_pre_torch_ref(
            residual,
            fn,
            hc_scale,
            hc_base,
            kwargs["rms_eps"],
            kwargs["hc_pre_eps"],
            kwargs["hc_sinkhorn_eps"],
            kwargs["post_mult"],
            kwargs["sinkhorn_repeat"],
            norm_weight=kwargs["norm_weight"],
            norm_eps=kwargs["norm_eps"],
            mhc_align_fleet=align,
        )
        want = _ref_pre(residual, fn, hc_scale, hc_base, **kwargs)

        self.assertEqual(got[0].shape, (tokens, hc, 1))
        self.assertEqual(got[1].shape, (tokens, hc, hc))
        self.assertEqual(got[2].shape, (tokens, hidden))
        for name, a, b in zip(("post", "comb", "layer_input"), got, want):
            self.assertTrue(
                torch.equal(a, b),
                f"{name} mismatch (hc={hc} hidden={hidden} tokens={tokens} "
                f"align={align} norm={use_norm} dtype={dtype}): "
                f"maxdiff={(a.float() - b.float()).abs().max().item():.3e}",
            )

        x = torch.randn(tokens, hidden, generator=gen, dtype=torch.float32).to(dtype)
        self.assertTrue(
            torch.equal(
                adapt.mhc_post_torch_ref(x, residual, got[0], got[1]),
                _ref_post(x, residual, want[0], want[1]),
            )
        )

    def test_pre_and_post_match_kernel_formula(self):
        # tokens 覆盖 1（DP warmup 常见的最小 shape）和 3（服务 warmup），
        # 这正是 2026-10-20 单测 8-token 盲区之外的部分。
        for hc, hidden, tokens, align, use_norm, dtype in itertools.product(
            (4, 2),
            (32, 64),
            (1, 3, 8),
            (False, True),
            (False, True),
            (torch.float32, torch.bfloat16),
        ):
            with self.subTest(
                hc=hc,
                hidden=hidden,
                tokens=tokens,
                align=align,
                norm=use_norm,
                dtype=dtype,
            ):
                self._case(hc, hidden, tokens, align, use_norm, dtype)

    def test_empty_tokens(self):
        hc, hidden, mix = 4, 32, 24
        residual = torch.empty(0, hc, hidden, dtype=torch.bfloat16)
        fn = torch.zeros(mix, hc * hidden, dtype=torch.bfloat16)
        post, comb, y = adapt.mhc_pre_torch_ref(
            residual,
            fn,
            torch.ones(3),
            torch.zeros(mix),
            1e-6,
            1e-6,
            1e-6,
            2.0,
            4,
            norm_weight=torch.ones(hidden),
            norm_eps=1e-6,
        )
        self.assertEqual(post.shape, (0, hc, 1))
        self.assertEqual(comb.shape, (0, hc, hc))
        self.assertEqual(y.shape, (0, hidden))


if __name__ == "__main__":
    unittest.main()
