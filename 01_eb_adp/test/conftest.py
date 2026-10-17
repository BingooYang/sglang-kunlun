"""在收集 EB5 kernel 单测前，先应用昆仑适配层（融合算子 -> torch 拆解）。"""
import os
import sys
import pathlib

# 用例会直接对 fp8 张量做 `.float()` / `.to(...)`，需要适配层给 `aten::_to_copy`
# 装的那层兜底。组网推理里这层是负担（见 eb5_mla_adapt._install_fp8_upcast），
# 所以默认关闭，只有单测把它打开。
os.environ.setdefault("EB5_KUNLUN_FP8_UPCAST", "1")

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    import eb5_kunlun_adapt

    print("[eb5-kunlun-adapt] patched:", eb5_kunlun_adapt.apply())
except Exception as _e:  # never block collection
    print("[eb5-kunlun-adapt] apply failed:", _e)
