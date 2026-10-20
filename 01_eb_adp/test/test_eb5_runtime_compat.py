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

    def test_decoy_attn_backend_on_batch_is_never_read(self):
        """即便 batch 上有 ``attn_backend`` 诱饵，也不能被读到（防"fork 又加回来"时静默走旧路）。"""
        import sglang.srt.model_executor.forward_context as forward_context

        class _Decoy:
            def get_indexer_metadata(self, *a, **k):  # noqa: D401
                raise AssertionError(
                    "插件读了 forward_batch.attn_backend（fork 已移除的旧 API）"
                )

        batch_like = SimpleNamespace(attn_backend=_Decoy())
        backend = _RecordingIndexerMetadataBackend(metadata=None)
        with forward_context.forward_context(
            forward_context.ForwardContext(attn_backend=backend)
        ):
            out = self.nsa_indexer.forward_cuda(
                object(),
                torch.zeros(1, 4),
                None,
                torch.zeros(1, dtype=torch.int64),
                batch_like,
                0,
            )
        self.assertIsNone(out)
        self.assertEqual(len(backend.calls), 1, "没有走 forward context 的后端")

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


class _StopPathReached(RuntimeError):
    """属性触碰记录器的截断信号（不是被测代码的异常）。"""


class _AttrTouchRecorder:
    """记录 ``forward_cuda`` 读过的 self 属性；未预置的属性一读就截断。"""

    def __init__(self, **known):
        object.__setattr__(self, "_known", known)
        object.__setattr__(self, "touched", [])

    def __getattr__(self, name):
        known = object.__getattribute__(self, "_known")
        if name in known:
            return known[name]
        touched = object.__getattribute__(self, "touched")
        touched.append(name)
        raise _StopPathReached(name)


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

    # -- 行为：热路径必须只认 fork 的新名字，且不因属性缺失抛异常 -------------
    #
    # 2026-10-20 服务里那次的形态是 ``AttributeError: ... has no attribute
    # 'nsa_enable_prefill_cp'``（直接读旧名、且 fork 上没有该属性）。
    # 这里不绑定具体实现（helper 叫什么、k-only 是方法还是模块级函数都在变），
    # 只观察两个稳定事实：① 不会去读旧名字；② 只挂新名字的 indexer 不会炸。

    def _probe(self, **known):
        """跑一次 ``forward_cuda``，返回 ``(读过的属性名, 抛出的异常或 None)``。

        ``known`` 是决策必需的属性（``index_topk`` / ``dsa_enable_prefill_cp`` /
        旧名 ``nsa_enable_prefill_cp``）；其余任何属性一被读到就记录并抛
        :class:`_StopPathReached` 截断 —— 既不需要真张量，也不依赖后续实现细节。
        """
        import sglang.srt.model_executor.forward_context as forward_context

        indexer = _AttrTouchRecorder(**known)
        fb = _minimal_forward_batch(
            self.ForwardMode.EXTEND, seq_lens_cpu=torch.tensor([4], dtype=torch.int64)
        )
        backend = _RecordingIndexerMetadataBackend(metadata=object())
        exc = None
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
            except _StopPathReached:
                pass
            except Exception as e:  # noqa: BLE001 - 只做记录，下面按类型断言
                exc = e
        return indexer.touched, exc

    def test_hot_path_never_reads_old_attribute_name(self):
        """不管 CP 开关取值，热路径都不许读 ``nsa_enable_prefill_cp``。

        "新名字真的被用到" 由 TestStaticApiRegression 正向断言
        （扫源码里的 ``dsa_enable_prefill_cp``）兜底 —— 这里只钉行为：
        读旧名 / 因属性缺失抛 AttributeError 都算错配。
        """
        for cp_on in (True, False):
            with self.subTest(dsa_enable_prefill_cp=cp_on):
                touched, _ = self._probe(
                    index_topk=8, dsa_enable_prefill_cp=cp_on
                )
                self.assertNotIn(
                    "nsa_enable_prefill_cp",
                    touched,
                    "forward_cuda 仍在读旧名 nsa_enable_prefill_cp"
                    f"（触碰过的属性：{touched}）",
                )

    def test_fork_shaped_indexer_never_raises_attribute_error(self):
        """只挂 ``dsa_``（fork 现状）的 indexer 不会因为 CP 属性缺失而 AttributeError。"""
        for cp_on in (True, False):
            with self.subTest(dsa_enable_prefill_cp=cp_on):
                _, exc = self._probe(index_topk=8, dsa_enable_prefill_cp=cp_on)
                self.assertFalse(
                    isinstance(exc, AttributeError),
                    f"forward_cuda 抛了 AttributeError（旧 API 错配的典型形态）：{exc}",
                )


# ---------------------------------------------------------------------------
# 错配 3：_get_k_bf16 参数个数
# ---------------------------------------------------------------------------


class TestGetKBf16Signature(unittest.TestCase):
    """K-only 路径（``_get_k_bf16`` 家族）的签名/调用点必须和 fork 对得上。

    这条错配的**本质**是"调用点实参个数 ≠ 目标签名"，所以除了比对 fork 的
    ``(x, positions)``，还做一条与具体实现名无关的 AST 检查：插件里任何
    ``*_get_k_bf16*`` 的调用，实参个数都必须落在目标签名的可接受范围内。
    这样插件把 helper 改名（``_get_k_bf16`` -> ``_get_k_bf16_compat``）、或换成
    ``_k_proj`` 之类的新实现，本用例都不需要跟着改。
    """

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
            f"fork 的 _get_k_bf16 签名变了：{params}；插件的 K-only 路径需要同步",
        )

    @staticmethod
    def _accepted_positional(fn, *, is_bound_method: bool) -> tuple[int, set]:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        n_pos = 0
        names = set()
        for i, prm in enumerate(params):
            if prm.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                if is_bound_method and i == 0:
                    continue  # self
                n_pos += 1
                names.add(prm.name)
            elif prm.kind == inspect.Parameter.KEYWORD_ONLY:
                names.add(prm.name)
        return n_pos, names

    def test_k_only_call_sites_match_target_signature(self):
        """AST 遍历：``*_get_k_bf16*`` 的每个调用点，实参都能被目标签名接住。"""
        import sglang_kunlun.hooks.layers.attention.nsa.nsa_indexer as plugin

        src = pathlib.Path(plugin.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        indexer_cls = self.dsa_indexer.Indexer

        checked, problems, unresolved = 0, [], []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Attribute) and "_get_k_bf16" in fn.attr:
                target = getattr(indexer_cls, fn.attr, None)
                is_bound = target is not None
            elif isinstance(fn, ast.Name) and "_get_k_bf16" in fn.id:
                target = getattr(plugin, fn.id, None)
                is_bound = False
            else:
                continue
            if target is None or not callable(target):
                unresolved.append(f"{plugin.__file__}:{node.lineno} {ast.unparse(fn)}")
                continue
            try:
                n_pos, kw_names = self._accepted_positional(
                    target, is_bound_method=is_bound
                )
            except (TypeError, ValueError):
                unresolved.append(f"{plugin.__file__}:{node.lineno} {ast.unparse(fn)}")
                continue
            checked += 1
            if len(node.args) > n_pos:
                problems.append(
                    f"{plugin.__file__}:{node.lineno} 调用 {ast.unparse(fn)} 传了 "
                    f"{len(node.args)} 个位置参数，签名只接 {n_pos} 个"
                )
            for kw in node.keywords:
                if kw.arg is not None and kw.arg not in kw_names:
                    problems.append(
                        f"{plugin.__file__}:{node.lineno} 关键字参数 {kw.arg} 不在签名里"
                    )
        self.assertTrue(
            checked, f"没找到任何可解析的 *_get_k_bf16* 调用点（unresolved={unresolved}）"
        )
        self.assertFalse(
            problems,
            "K-only 路径的调用点与目标签名不匹配（服务里那次 TypeError 就长这样）：\n  "
            + "\n  ".join(problems),
        )

    def test_get_k_bf16_runs_with_two_args_and_returns_key_shape(self):
        """最小张量（CPU）按 ``(x, positions)`` 调 fork 实现：返回 ``(T, head_dim)``。"""
        head_dim, rope_head_dim, num_tokens = 64, 32, 4
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

        # fork 的 _get_k_bf16 读这两个属性（类属性在类体里不能写成 `x = x`）
        _Fake.head_dim = head_dim
        _Fake.rope_head_dim = rope_head_dim
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
            rope_head_dim = 4

            def wk(self, x):
                return torch.zeros(x.shape[0], head_dim, dtype=torch.bfloat16), None

            def k_norm(self, key):
                return key

            def rotary_emb(self, pos, q, k):
                return q, k

            _update_rope_guarded = staticmethod(
                self.dsa_indexer.Indexer._update_rope_guarded
            )

        _Fake.head_dim = head_dim
        x = torch.zeros(2, 16, dtype=torch.bfloat16)
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
        """正向断言：光 "没有旧名" 不够，新名/新入口必须真的在。

        用 AST 而不是字符串匹配 —— 插件的 import 形态会变（单行 / 括号多行 /
        顺带多导入 ``get_token_to_kv_pool``），字符串断言会假红。
        """
        import sglang_kunlun.hooks.layers.attention.nsa.nsa_indexer as plugin

        src = pathlib.Path(plugin.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)

        imported_from_fc = False
        calls_get_attn_backend = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module == "sglang.srt.model_executor.forward_context"
            ):
                imported_from_fc = imported_from_fc or any(
                    a.name == "get_attn_backend" for a in node.names
                )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_attn_backend"
            ):
                calls_get_attn_backend = True

        self.assertTrue(
            imported_from_fc,
            "nsa_indexer 没有从 forward_context import get_attn_backend（错配 1 的修复被回退了？）",
        )
        self.assertTrue(
            calls_get_attn_backend, "nsa_indexer 没有调用 get_attn_backend()"
        )
        self.assertIn(
            "dsa_enable_prefill_cp",
            src,
            "nsa_indexer 里找不到 dsa_enable_prefill_cp（错配 2 的修复被回退了？）",
        )

if __name__ == "__main__":
    unittest.main()
