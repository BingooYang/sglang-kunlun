"""适配层覆盖自检：**每个被替换的符号都必须有用例提到**。

为什么需要这条：原来的覆盖模型是"按 ku《EBlite 和 dpsk-v4 算子对比》里的第三方
kernel 清单枚举"，而适配层**自己新增**的 torch 拆解不在那张表里 —— 2026-10-17
EB5 整网首次前向炸在 `mhc_pre_gemm_sqrsum_splitk_kernel`（`kernel_0`）上时，
它当时是**零覆盖**（连同 `_mhc_pre_gemm_sqrsum_dispatch` / `mhc_pre_gemm_sqrsum_tilelang`，
整个 sqrsum/splitk 家族都漏了）。这条用例把"悄悄漏一个"变成"跑不过"。

**本文件刻意不依赖 XPU**：不 import `_eb5_kernel_test_common`（那个模块在 import
阶段就 `init_process_group(backend="nccl")`），所以它**可以在设备不健康时单独跑**，
用来快速确认覆盖有没有退化。
"""

from __future__ import annotations

import pathlib
import re
import unittest

_HERE = pathlib.Path(__file__).resolve().parent
_SELF = pathlib.Path(__file__).name

# 已知"暂时没有专门用例"的符号 —— 加进来必须写清原因，且不应长期留在这里。
# 留在这里的作用是：让**新增**的遗漏立刻 fail，而不是又一次悄悄溜过去。
_KNOWN_UNCOVERED = {
    "build_swa_token_ids": "DSV4/HCA 稀疏 prefill 的索引构造，当前只被间接覆盖",
    "combine_topk_swa_indices": "同上",
    "create_flashinfer_kv_indices_triton": "kv 索引构造，单测未单独覆盖",
    "get_paged_mqa_logits_metadata": "mqa logits 元数据，单测未单独覆盖",
    "linear_bf16_fp32": "bf16 GEMM 包装，单测未单独覆盖",
}


def _replaced_symbols() -> set[str]:
    """解析适配层三个文件，取出所有"被替换掉的符号名"。"""
    names: set[str] = set()
    # eb5_extra_adapt.py 的 _TARGETS 是 ("模块", "符号", 实现) 三元组
    src = (_HERE / "eb5_extra_adapt.py").read_text(encoding="utf-8")
    names |= {m.group(2) for m in re.finditer(r'\("([\w\.]+)",\s*"(\w+)",', src)}
    # eb5_kunlun_adapt.py / eb5_mla_adapt.py 用 _patch(模块变量, "符号", 实现)
    for fname in ("eb5_kunlun_adapt.py", "eb5_mla_adapt.py"):
        src = (_HERE / fname).read_text(encoding="utf-8")
        names |= {m.group(2) for m in re.finditer(r'_patch\((\w+),\s*"(\w+)"', src)}
    return names


def _test_sources() -> str:
    """所有测试文件的正文（**排除本文件**，否则清单里的名字会自我满足）。"""
    return "\n".join(
        p.read_text(encoding="utf-8")
        for p in sorted(_HERE.glob("test_eb5_*.py"))
        if p.name != _SELF
    )


class TestAdaptCoverage(unittest.TestCase):
    def test_replaced_symbol_set_is_non_trivial(self):
        """自检：解析本身没坏（不然下面那条会假绿）。"""
        syms = _replaced_symbols()
        # mHC 的融合 kernel（tilelang post / big_fuse pre / big_fuse pre + norm /
        # split-K / sqrsum dispatch / deep_gemm tf32）**在 kernel 级**恢复了 torch
        # 替代，被替换符号 24 -> 31（2026-10-20）。阈值跟着抬，防止再被悄悄删掉：
        # 一旦有人把某个 kernel 级替代实现删了，符号数会掉到 31 以下，这里立刻 fail。
        self.assertGreaterEqual(
            len(syms), 31, f"只解析出 {len(syms)} 个被替换符号，解析逻辑/实现可能被删"
        )

    def test_every_replaced_symbol_is_referenced_by_some_test(self):
        syms = _replaced_symbols()
        src = _test_sources()
        missing = sorted(
            s
            for s in syms
            if s not in src and s not in _KNOWN_UNCOVERED
        )
        self.assertFalse(
            missing,
            "适配层替换了、但没有任何用例提到的符号（新增遗漏会在这里 fail）：\n  "
            + "\n  ".join(missing)
            + "\n如果确实暂时不覆盖，请连同原因加进 _KNOWN_UNCOVERED。",
        )

    def test_known_uncovered_list_stays_honest(self):
        """`_KNOWN_UNCOVERED` 里的条目必须仍是真实存在的替换符号（防止清单腐烂）。"""
        syms = _replaced_symbols()
        stale = sorted(k for k in _KNOWN_UNCOVERED if k not in syms)
        self.assertFalse(stale, f"_KNOWN_UNCOVERED 里已不存在的符号，请删掉: {stale}")


if __name__ == "__main__":
    unittest.main()
