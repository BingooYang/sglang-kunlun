"""多 rank 复现的 pytest 入口（默认 skip，需显式开启）。

为什么单独放一个文件：单测是单进程单卡，`int(offset)` / 布尔索引这类**隐式 D2H**
在那种环境下不会挂；只有 `init_process_group(nccl)` + 多 rank 才复现（doc/3 第四节）。
这里把 `tool/repro_multirank.sh` 包成用例，**不加载模型权重**，几十秒出结果。

用法（会占用 4 张卡，且挂死时进程要等它自己超时，不要 kill）：

    EB5_MULTIRANK_REPRO=1 EB5_REPRO_GPUS=0,1,2,3 \
      python -m pytest test_eb5_multirank_repro.py -q

不设 `EB5_MULTIRANK_REPRO` 时整类跳过，不会被普通 `pytest layer/` 带到。
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import unittest

_TOOL = pathlib.Path(__file__).resolve().parent.parent / "tool" / "repro_multirank.sh"
_GPUS = os.environ.get("EB5_REPRO_GPUS", "0,1,2,3")
_WAIT = os.environ.get("EB5_REPRO_WAIT", "90")


@unittest.skipUnless(
    os.environ.get("EB5_MULTIRANK_REPRO") == "1",
    "多 rank 复现会占用多张卡，需 EB5_MULTIRANK_REPRO=1 显式开启",
)
class TestMultirankRepro(unittest.TestCase):
    """每个 case 一次独立的多 rank 运行；失败信息里带每个 rank 的最后一步。"""

    def _run(self, case: str) -> None:
        proc = subprocess.run(
            ["bash", str(_TOOL), case, _GPUS, _WAIT],
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"case={case} 失败：\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}",
        )

    def test_baseline_xccl_d2h(self):
        """设备基线：init_process_group + all_reduce + D2H（doc/3 §4 最小复现）。"""
        self._run("baseline")

    def test_memcpy_full(self):
        """整块拷贝：旧单测唯一覆盖的参数。"""
        self._run("memcpy-full")

    def test_memcpy_partial(self):
        """非零 offset + 截断（2026-10-20 的挂死参数）。"""
        self._run("memcpy-partial")

    def test_memcpy_scatter(self):
        """`offset_src=False`（dp_gather 的方向）。"""
        self._run("memcpy-scatter")

    def test_mhc_pre_splitk(self):
        """mhc_pre split-k 拆解，服务真实形状（3 token / split_k=32）。"""
        self._run("splitk")

    def test_mhc_pre_tf32(self):
        """mhc_pre 的 tf32_hc_prenorm_gemm 拆解。"""
        self._run("tf32")


if __name__ == "__main__":
    unittest.main()
