"""EB5 运行时分支的「插件 ↔ fork API」兼容性回归（**不需要 XPU，CPU 就能跑**）。

为什么需要这个文件
------------------
2026-10-20 在 EB5 服务里连续踩到三处错配，共同点是**插件按旧版 sglang（0.5.x）的
API 写、而 fork 已经改名/改签名**，而且都要等整网首次前向才炸（import 阶段、
kernel 级单测都发现不了）：

=================================  ==========================================  ====================================
插件里的旧写法                       fork 的实际契约                              本文件的用例
=================================  ==========================================  ====================================
``forward_batch.attn_backend``      ``ForwardBatch`` 已删除该属性；改为            TestForwardBatchAttnBackend
                                   ``sglang.srt.model_executor.forward_context``
                                   ``.get_attn_backend()``
``self.nsa_enable_prefill_cp``      改名为 ``dsa_enable_prefill_cp``              TestPrefillCPRename
``self._get_k_bf16(x, ...)``        签名固定为 ``(x, positions)``                 TestGetKBf16Signature
=================================  ==========================================  ====================================

前三条是**行为级**用例：构造最小 ``ForwardBatch`` / fake indexer，断言真的走了新的
API（而不是只 grep 源码）。最后一条 TestStaticApiRegression 是**低成本静态回归** ——
直接扫 ``sglang_kunlun/`` 下的 ``.py``，出现旧 API 即 fail 并打印 ``文件:行``，
这样以后再遇到同类版本改名，第一时间在这里报出来。

为什么刻意不依赖 XPU
--------------------
本文件**不 import** ``_eb5_kernel_test_common``（它在 import 阶段就
``init_process_group(backend="nccl")``），也不建任何设备张量，所以设备半坏 / 被别的
任务占着时也能跑：

    CUDA_VISIBLE_DEVICES=4 python -m pytest test_eb5_runtime_compat.py -q

（``CUDA_VISIBLE_DEVICES`` 只是让 ``import torch`` 走本仓库约定的卡；用例本身不碰设备。）
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import pathlib
import re
import unittest
from types import SimpleNamespace

import torch

_HERE = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 插件源码定位 + 静态扫描基础设施
# ---------------------------------------------------------------------------

# 每条：(正则, 人类可读的旧 API, 替换建议)
_FORBIDDEN_LEGACY_APIS = (
    (
        re.compile(r"(?<![\w.])forward_batch\s*\.\s*attn_backend\b"),
        "forward_batch.attn_backend",
        "fork 的 ForwardBatch 已无此属性，改用 "
        "sglang.srt.model_executor.forward_context.get_attn_backend()",
    ),
    (
        re.compile(r"\.nsa_enable_prefill_cp\b"),
        "self.nsa_enable_prefill_cp",
        "fork 已改名为 dsa_enable_prefill_cp（旧名只应出现在 _enable_prefill_cp 的兼容分支）",
    ),
)


def _plugin_package_dir() -> pathlib.Path:
    """``.../sglang_kunlun`` 包目录。

    三层兜底：环境变量（staging 到别处时用）→ 已安装/可发现的包 → 源码树相对路径。
    """
    env_root = os.environ.get("SGLANG_KUNLUN_ROOT")
    if env_root:
        cand = pathlib.Path(env_root) / "sglang_kunlun"
        if cand.is_dir():
            return cand
    spec = importlib.util.find_spec("sglang_kunlun")
    if spec is not None and spec.origin:
        return pathlib.Path(spec.origin).resolve().parent
    # 01_eb_adp/test/ -> 01_eb_adp -> sglang-kunlun/
    cand = _HERE.parents[1] / "sglang_kunlun"
    if cand.is_dir():
        return cand
    raise RuntimeError(
        "找不到 sglang_kunlun 包目录：请设 SGLANG_KUNLUN_ROOT，或确认插件已安装"
    )


def _iter_plugin_sources():
    """遍历插件下的 .py，产出 ``(path, lines)``；跳过 __pycache__ / 隐藏目录。"""
    root = _plugin_package_dir()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts or any(p.startswith(".") for p in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # 非 UTF-8 的生成文件直接跳过
            continue
        yield path, text.splitlines()


def _import_nsa_indexer():
    """插件里被 hook 到 ``sglang...dsa_indexer.Indexer`` 的那份实现。"""
    import sglang_kunlun.hooks.layers.attention.nsa.nsa_indexer as mod

    return mod


def _minimal_forward_batch(forward_mode, seq_lens_cpu=None):
    """最小可构造的**真实** ``ForwardBatch``（CPU 张量，不碰设备）。

    刻意用真类而不是 SimpleNamespace：这样 "fork 里到底还有没有 attn_backend"
    这件事是**由 fork 自己回答**的，插件一旦回退到旧属性就会 AttributeError。
    """
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    return ForwardBatch(
        forward_mode=forward_mode,
        batch_size=1,
        input_ids=torch.tensor([1], dtype=torch.int64),
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([4], dtype=torch.int64),
        out_cache_loc=torch.tensor([0], dtype=torch.int64),
        seq_lens_sum=4,
        seq_lens_cpu=seq_lens_cpu,
    )


class _RecordingIndexerMetadataBackend:
    """只实现 ``get_indexer_metadata`` 的假后端，记录被调用的情况。"""

    def __init__(self, metadata=None):
        self.metadata = metadata
        self.calls = []

    def get_indexer_metadata(self, layer_id, forward_batch):
        self.calls.append((layer_id, forward_batch))
        return self.metadata


# ---------------------------------------------------------------------------
# 错配 1：forward_batch.attn_backend -> get_attn_backend()
# ---------------------------------------------------------------------------


class TestForwardBatchAttnBackend(unittest.TestCase):
    """:mod:`...nsa.nsa_indexer` 必须从 forward context 取后端，不能读 ForwardBatch。"""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        cls.ForwardMode = ForwardMode
        cls.nsa_indexer = _import_nsa_indexer()

    def test_forward_batch_has_no_attn_backend_attribute(self):
        """前提：fork 的 ForwardBatch **确实没有** attn_backend（旧代码在这里就炸）。"""
        fb = _minimal_forward_batch(self.ForwardMode.DECODE)
        self.assertFalse(
            hasattr(fb, "attn_backend"),
            "fork 的 ForwardBatch 又出现了 attn_backend 属性 —— 请复核 "
            "nsa_indexer 用 get_attn_backend() 取后端是否仍然正确，再决定要不要保留本用例。",
        )

    def test_forward_cuda_resolves_backend_from_forward_context(self):
        """行为：只 publish ForwardContext，``forward_cuda`` 就能拿到后端（不抛 AttributeError）。"""
        import sglang.srt.model_executor.forward_context as forward_context

        backend = _RecordingIndexerMetadataBackend(metadata=None)
        fb = _minimal_forward_batch(self.ForwardMode.DECODE)
        indexer = object()  # metadata 为 None 时不会碰 self 的任何成员

        with forward_context.forward_context(
            forward_context.ForwardContext(attn_backend=backend)
        ):
            out = self.nsa_indexer.forward_cuda(
                indexer,
                torch.zeros(1, 4),
                None,
                torch.zeros(1, dtype=torch.int64),
                fb,
                7,
            )

        self.assertIsNone(out, "metadata 为 None 时应早退，返回 None")
        self.assertEqual(
            len(backend.calls),
            1,
            "forward_cuda 没有通过 get_attn_backend() 取 indexer metadata",
        )
        layer_id, seen_fb = backend.calls[0]
        self.assertEqual(layer_id, 7)
        self.assertIs(seen_fb, fb)

    def test_forward_cuda_without_forward_context_fails_loudly(self):
        """没有 publish context 时要报 ``get_forward_context`` 的 assert，而不是静默用错后端。"""
        nsa_indexer = self.nsa_indexer
        fb = _minimal_forward_batch(self.ForwardMode.DECODE)
        with self.assertRaises(AssertionError) as ctx:
            nsa_indexer.forward_cuda(
                object(),
                torch.zeros(1, 4),
                None,
                torch.zeros(1, dtype=torch.int64),
                fb,
                0,
            )
        self.assertIn("forward context", str(ctx.exception))


# ---------------------------------------------------------------------------
# 错配 2：self.nsa_enable_prefill_cp -> dsa_enable_prefill_cp
# ---------------------------------------------------------------------------


class TestPrefillCPRename(unittest.TestCase):
    """``nsa_enable_prefill_cp`` 在 fork 里已改名 ``dsa_enable_prefill_cp``。"""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        cls.ForwardMode = ForwardMode
        cls.nsa_indexer = _import_nsa_indexer()

    # -- 兼容 helper：两种属性名都要能取值，且语义一致 -----------------------

    def test_compat_helper_accepts_both_attribute_names(self):
        helper = getattr(self.nsa_indexer, "_enable_prefill_cp", None)
        self.assertIsNotNone(
            helper,
            "nsa_indexer 丢了 _enable_prefill_cp 兼容 helper（旧/新 fork 双名兜底）",
        )
        cases = {
            "只 dsa_（fork 现状）": SimpleNamespace(dsa_enable_prefill_cp=True),
            "只 nsa_（旧接口）": SimpleNamespace(nsa_enable_prefill_cp=True),
            "两个都有（都 True）": SimpleNamespace(
                dsa_enable_prefill_cp=True, nsa_enable_prefill_cp=True
            ),
        }
        for name, obj in cases.items():
            with self.subTest(case=name):
                self.assertTrue(helper(obj))

    def test_compat_helper_off_and_absent(self):
        helper = self.nsa_indexer._enable_prefill_cp
        self.assertFalse(helper(SimpleNamespace(dsa_enable_prefill_cp=False)))
        self.assertFalse(helper(SimpleNamespace(nsa_enable_prefill_cp=False)))
        self.assertFalse(
            helper(SimpleNamespace()), "两个属性都没有时应视为关闭，而不是抛异常"
        )

    def test_compat_helper_prefers_dsa_when_conflicting(self):
        """同名冲突时以 fork 的新名字为准（旧字段若残留不应左右行为）。"""
        helper = self.nsa_indexer._enable_prefill_cp
        self.assertTrue(
            helper(
                SimpleNamespace(
                    dsa_enable_prefill_cp=True, nsa_enable_prefill_cp=False
                )
            )
        )
        self.assertFalse(
            helper(
                SimpleNamespace(
                    dsa_enable_prefill_cp=False, nsa_enable_prefill_cp=True
                )
            )
        )

    # -- 行为：forward_cuda 的 "k-only 快路径" 由 dsa_ 这个开关决定 ----------

    def _run_forward_cuda_with_cp_flag(self, *, dsa_value):
        """让 ``forward_cuda`` 在 "skip_logits + CP 关闭" 时才走 ``_forward_cuda_k_only``。

        返回 (走没走 k_only, 是否进入了更深的分支)。两条分支都用 sentinel 截断，
        所以不需要真的张量 / 设备。
        """
        import sglang.srt.model_executor.forward_context as forward_context

        class _DeepPathReached(RuntimeError):
            pass

        k_only_calls = []

        indexer = SimpleNamespace(
            index_topk=8,
            dsa_enable_prefill_cp=dsa_value,
            _forward_cuda_k_only=lambda *a, **k: k_only_calls.append(a) or "k_only",
            _get_q_k_bf16=lambda *a, **k: (_ for _ in ()).throw(_DeepPathReached()),
        )
        fb = _minimal_forward_batch(
            self.ForwardMode.EXTEND, seq_lens_cpu=torch.tensor([4], dtype=torch.int64)
        )
        backend = _RecordingIndexerMetadataBackend(metadata=object())

        deep = False
        with forward_context.forward_context(
            forward_context.ForwardContext(attn_backend=backend)
        ):
            try:
                self.nsa_indexer.forward_cuda(
                    indexer,
                    torch.zeros(4, 4),
                    None,
                    torch.zeros(4, dtype=torch.int64),
                    fb,
                    0,
                )
            except _DeepPathReached:
                deep = True
        return bool(k_only_calls), deep

    def test_cp_enabled_skips_k_only_fast_path(self):
        """``dsa_enable_prefill_cp=True`` 时**不能**走 k-only 快路径（CP 需要完整 logits）。"""
        k_only, deep = self._run_forward_cuda_with_cp_flag(dsa_value=True)
        self.assertFalse(
            k_only, "CP 打开时仍走了 _forward_cuda_k_only（旧属性名读串了？）"
        )
        self.assertTrue(deep, "CP 打开时应继续走向 _get_q_k_bf16 那条路")

    def test_cp_disabled_takes_k_only_fast_path(self):
        """``dsa_enable_prefill_cp=False`` 时走 k-only 快路径（短 prefill 的早退分支）。"""
        k_only, deep = self._run_forward_cuda_with_cp_flag(dsa_value=False)
        self.assertTrue(k_only, "CP 关闭时没走 _forward_cuda_k_only，分发逻辑变了")
        self.assertFalse(deep)

    def test_old_attribute_name_alone_is_not_read_by_hot_path(self):
        """热路径只认 ``dsa_``：只挂旧名 ``nsa_`` 的 fake indexer 应被当作 "CP 关闭"。

        这条用例把 "旧名还能不能用" 这个隐含假设钉死：``_enable_prefill_cp``
        才是双名兜底的地方，热路径不兜底（fork 里只有新名）。
        """
        import sglang.srt.model_executor.forward_context as forward_context

        k_only_calls = []
        indexer = SimpleNamespace(
            index_topk=8,
            nsa_enable_prefill_cp=True,  # 只有旧名
            _forward_cuda_k_only=lambda *a, **k: k_only_calls.append(a) or "k_only",
        )
        fb = _minimal_forward_batch(
            self.ForwardMode.EXTEND, seq_lens_cpu=torch.tensor([4], dtype=torch.int64)
        )
        backend = _RecordingIndexerMetadataBackend(metadata=object())
        with forward_context.forward_context(
            forward_context.ForwardContext(attn_backend=backend)
        ):
            out = self.nsa_indexer.forward_cuda(
                indexer,
                torch.zeros(4, 4),
                None,
                torch.zeros(4, dtype=torch.int64),
                fb,
                0,
            )
        self.assertEqual(out, "k_only")
        self.assertEqual(len(k_only_calls), 1)


# ---------------------------------------------------------------------------
# 错配 3：_get_k_bf16 参数个数
# ---------------------------------------------------------------------------


class TestGetKBf16Signature(unittest.TestCase):
    """``_get_k_bf16`` 的 fork 签名是 ``(x, positions)``；插件调用点必须只传两个。"""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.dsa import dsa_indexer

        cls.dsa_indexer = dsa_indexer
        cls.nsa_indexer = _import_nsa_indexer()

    def test_fork_signature_is_x_positions(self):
        params = list(
            inspect.signature(self.dsa_indexer.Indexer._get_k_bf16).parameters
        )
        self.assertEqual(
            params,
            ["self", "x", "positions"],
            f"fork 的 _get_k_bf16 签名变了：{params}；插件的 _forward_cuda_k_only 需要同步",
        )

    def test_plugin_call_site_passes_exactly_two_args(self):
        """AST 找 ``self._get_k_bf16(...)``：必须是 2 个位置参数、无关键字参数。"""
        src = pathlib.Path(self.nsa_indexer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        calls = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_get_k_bf16"
            ):
                calls.append(
                    (node.lineno, len(node.args), [kw.arg for kw in node.keywords])
                )
        self.assertTrue(calls, "插件里找不到 self._get_k_bf16(...) 调用点（被删了？）")
        for lineno, n_args, kwargs in calls:
            with self.subTest(line=lineno):
                self.assertEqual(
                    n_args,
                    2,
                    f"{self.nsa_indexer.__file__}:{lineno} 传了 {n_args} 个位置参数",
                )
                self.assertFalse(
                    kwargs,
                    f"{self.nsa_indexer.__file__}:{lineno} 用了关键字参数 {kwargs}",
                )

    def test_get_k_bf16_runs_with_two_args_and_returns_key_shape(self):
        """最小张量（CPU）走一遍：返回 ``(num_tokens, head_dim)`` 的 key。"""
        head_dim, num_tokens = 64, 4
        positions = torch.arange(num_tokens, dtype=torch.int64)

        class _Fake:
            def __init__(self):
                self.wk_called_with = None
                self.rope_positions = None

            def wk(self, x):
                self.wk_called_with = tuple(x.shape)
                return torch.zeros(x.shape[0], head_dim, dtype=torch.bfloat16), None

            def k_norm(self, key):
                return key

            def rotary_emb(self, pos, q, k):
                self.rope_positions = pos
                return q, k

            _update_rope_guarded = staticmethod(
                self.dsa_indexer.Indexer._update_rope_guarded
            )

        fake = _Fake()
        x = torch.zeros(num_tokens, 128, dtype=torch.bfloat16)

        # rotate_activation 走 hadamard（设备算子），单测里换成恒等即可 ——
        # 本用例只验证**调用签名 + 返回形状**，不验证数值。
        original = self.dsa_indexer.rotate_activation
        self.dsa_indexer.rotate_activation = lambda t: t
        try:
            key = self.dsa_indexer.Indexer._get_k_bf16(fake, x, positions)
        finally:
            self.dsa_indexer.rotate_activation = original

        self.assertEqual(tuple(key.shape), (num_tokens, head_dim))
        self.assertEqual(key.dtype, torch.bfloat16)
        self.assertEqual(fake.wk_called_with, (num_tokens, 128))
        self.assertIs(fake.rope_positions, positions)

    def test_old_three_arg_call_raises_type_error(self):
        """旧写法（多传 forward_batch）必须炸 —— 这就是服务里那次 TypeError。"""
        head_dim = 8

        class _Fake:
            def wk(self, x):
                return torch.zeros(x.shape[0], head_dim), None

            def k_norm(self, key):
                return key

            def rotary_emb(self, pos, q, k):
                return q, k

            _update_rope_guarded = staticmethod(
                self.dsa_indexer.Indexer._update_rope_guarded
            )

        x = torch.zeros(2, 16)
        positions = torch.zeros(2, dtype=torch.int64)
        with self.assertRaises(TypeError):
            self.dsa_indexer.Indexer._get_k_bf16(_Fake(), x, positions, None)


# ---------------------------------------------------------------------------
# 低成本静态回归：旧 API 不许回潮
# ---------------------------------------------------------------------------


class TestStaticApiRegression(unittest.TestCase):
    """扫 ``sglang_kunlun/**/*.py``，出现被 fork 移除/改名的 API 就 fail。"""

    def test_plugin_sources_are_discoverable(self):
        """自检：扫描本身没坏（否则下面几条会假绿）。"""
        files = list(_iter_plugin_sources())
        self.assertGreater(
            len(files), 50, f"只扫到 {len(files)} 个 .py，源码定位可能坏了"
        )
        names = {p.name for p, _ in files}
        self.assertIn("nsa_indexer.py", names)

    def test_no_forbidden_legacy_apis(self):
        hits = []
        for path, lines in _iter_plugin_sources():
            for lineno, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue  # 纯注释不算（说明历史可以，代码里不许留）
                for pattern, legacy, hint in _FORBIDDEN_LEGACY_APIS:
                    if pattern.search(line):
                        hits.append(
                            f"  {path}:{lineno}  {legacy}\n"
                            f"      {stripped}\n"
                            f"      -> {hint}"
                        )
        self.assertFalse(
            hits,
            "插件里出现已被 fork 移除/改名的 API（这就是服务里挂死/崩的那类错配）：\n"
            + "\n".join(hits),
        )

    def test_replacement_apis_are_actually_used(self):
        """正向断言：光 "没有旧名" 不够，新名/新入口必须真的在。"""
        src = (
            _plugin_package_dir() / "hooks/layers/attention/nsa/nsa_indexer.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "from sglang.srt.model_executor.forward_context import get_attn_backend",
            src,
            "nsa_indexer 没有 import get_attn_backend（错配 1 的修复被回退了？）",
        )
        self.assertIn(
            "get_attn_backend()", src, "nsa_indexer 没有调用 get_attn_backend()"
        )
        self.assertIn(
            "dsa_enable_prefill_cp",
            src,
            "nsa_indexer 里找不到 dsa_enable_prefill_cp（错配 2 的修复被回退了？）",
        )


if __name__ == "__main__":
    unittest.main()
