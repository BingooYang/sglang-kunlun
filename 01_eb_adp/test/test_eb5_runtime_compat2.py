"""EB5 运行时兼容性回归（第二批）：**2026-10-20 实际踩到的 9 类问题** + kernel 级替换自检。

为什么另开一个文件（而不是继续往 `test_eb5_runtime_compat.py` 里加）
--------------------------------------------------------------
`test_eb5_runtime_compat.py` 钉的是"插件 ↔ fork 的 API 改名"（三处错配 + 静态
回潮扫描），本文件钉的是**另一类**：EB5 运行时里"取错池子 / 形参契约 / 越界填充值 /
Triton 化之后的分支残留"，外加"融合 kernel 已被 kernel 级 torch 实现替换"的自检。
两者都不需要 XPU，工具函数（插件源码定位、最小 `ForwardBatch`）直接复用前者。

覆盖项（编号与任务书一致）
--------------------------
1. `TestKvPoolAccessor`                  —— `forward_batch.token_to_kv_pool` 已不存在，
   必须走 `get_token_to_kv_pool()`；EB5 要取**包装后**的
   `dsa_indexer.get_token_to_kv_pool` 并解包 `full_kv_pool`
2. `TestSetMlaKvBufferSignature`         —— `set_mla_kv_buffer_kernel` 的 10 个位置形参 +
   `reserved_skip_index` + `BLOCK` 关键字 + `DCP_RANK/DCP_WORLD_SIZE` 被 kwargs 吞掉
3. `TestKernelLevelTorchReplacements`    —— tilelang / deep_gemm 融合 kernel 必须在
   **kernel 级**指向 torch 实现（不是只绕开调用层）
4. `TestTopkAllVisible`                  —— `_topk_all_visible` 的 shape / dtype / 右侧
   `-1` 填充 / 与 `torch.topk` 的 index 一致（含 tokens=0、topk>seq）
5. `TestTransformIndexPageTable`         —— `transform_index_page_table_{prefill,decode}`
   （torch 版）与文件内 ref 逐元素对照：8 组 prefill 形状 + decode + 空洞/越界填充
6. `TestAppendIndexerSwa`                —— `dsa_swa_index.append_indexer_swa` 的 3 种
   mode（ragged / paged-identity / paged-mapped）与 ref 对照
7. `TestTopkRaggedChunkPath`             —— `_get_topk_ragged` 的 chunk 分支不再引用
   未定义的 `k_interval_*`（AST 作用域分析，比字符串匹配更严）
8. `TestNsaBackendDsaCpSymbols`          —— fork 已删除 5 个 `dsa_cp_*` 符号；插件用
   fork 现有别名 + 纯 torch 实现补齐，`_nsa_backend_impl` 必须能 import，且
   redirect 真正装进 `sys.modules`
9. `TestTopkRaggedWithCpRename`          —— 插件不再引用已删除的旧名
   `_get_topk_ragged_with_cp`（CP-v1 in-seq-split 改成显式 NotImplementedError 守卫）
10. `TestKvCacheDtypeGuard`              —— `run_eb5_kunlun.sh` 默认 `KV_CACHE_DTYPE=bfloat16`

为什么刻意不依赖 XPU
--------------------
与 `test_eb5_runtime_compat.py` 相同：不 import `_eb5_kernel_test_common`
（它在 import 阶段 `init_process_group(backend="nccl")`），所有张量都在 CPU 上构造。
设备半坏 / 被别的任务占着时也能跑：

    XPU_VISIBLE_DEVICES=4 python -m pytest test_eb5_runtime_compat2.py -q
"""

from __future__ import annotations

import ast
import builtins
import inspect
import os
import pathlib
import re
import sys
import unittest

import torch

import test_eb5_runtime_compat as rc

_HERE = pathlib.Path(__file__).resolve().parent
_ADP = _HERE.parent
_PLUGIN = rc._plugin_package_dir()


def _plugin_file(rel: str) -> pathlib.Path:
    path = _PLUGIN / rel
    assert path.is_file(), f"插件源码不存在: {path}"
    return path


# ---------------------------------------------------------------------------
# 1) KV pool 取法：get_token_to_kv_pool() + 解包 full_kv_pool
# ---------------------------------------------------------------------------


class _FakePool:
    """只有 `full_kv_pool` 的"外层 HCA 池"（page_size 与 indexer 要的 64 不同）。"""

    def __init__(self, full_kv_pool=None, *, has_attr: bool = True):
        if has_attr:
            self.full_kv_pool = full_kv_pool
        self.page_size = 256

    def __repr__(self):  # 便于断言失败时看对象
        return (
            f"_FakePool(page_size={self.page_size}, "
            f"full={getattr(self, 'full_kv_pool', '<absent>')!r})"
        )


class TestKvPoolAccessor(unittest.TestCase):
    """indexer 侧的 KV pool 必须从 `get_token_to_kv_pool()` 拿，并解包 EB5 子池。"""

    @classmethod
    def setUpClass(cls):
        cls.nsa_indexer = rc._import_nsa_indexer()
        import sglang.srt.layers.attention.dsa.dsa_indexer as dsa_indexer

        cls.dsa_indexer = dsa_indexer

    def setUp(self):
        self._saved_module_fn = getattr(self.dsa_indexer, "get_token_to_kv_pool", None)
        self._saved_local_fn = getattr(self.nsa_indexer, "get_token_to_kv_pool", None)

    def tearDown(self):
        if self._saved_module_fn is not None:
            self.dsa_indexer.get_token_to_kv_pool = self._saved_module_fn
        if self._saved_local_fn is not None:
            self.nsa_indexer.get_token_to_kv_pool = self._saved_local_fn

    # -- 前提：旧属性确实没了 ------------------------------------------------

    def test_forward_batch_has_no_token_to_kv_pool_attribute(self):
        """前提：fork 的 `ForwardBatch` 已经没有 `token_to_kv_pool`（旧写法必 AttributeError）。"""
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        self.assertFalse(
            hasattr(ForwardBatch, "token_to_kv_pool"),
            "fork 的 ForwardBatch 又出现 token_to_kv_pool —— 请复核插件是否该回退旧写法",
        )

    @staticmethod
    def _old_pool_attr_hits(rel: str):
        """扫一个插件文件里对 `forward_batch.token_to_kv_pool` 的引用。"""
        pattern = re.compile(r"(?<![\w.])forward_batch\s*\.\s*token_to_kv_pool\b")
        path = _plugin_file(rel)
        hits = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line) and not line.strip().startswith("#"):
                hits.append(f"{path}:{lineno}: {line.strip()}")
        return hits

    def test_live_indexer_never_reads_forward_batch_token_to_kv_pool(self):
        """静态：**在跑的那条** indexer 路径不许出现 `forward_batch.token_to_kv_pool`。"""
        hits = self._old_pool_attr_hits("hooks/layers/attention/nsa/nsa_indexer.py")
        self.assertFalse(
            hits,
            "插件仍在读 forward_batch.token_to_kv_pool（fork 已删除该属性）：\n  "
            + "\n  ".join(hits),
        )

    def test_nsa_backend_impl_never_reads_forward_batch_token_to_kv_pool(self):
        """`_nsa_backend_impl.py` 的 8 处旧属性已全部换成 `get_token_to_kv_pool()`。

        EB5 侧用 ``_dsa_kv_pool()`` 解包 ``full_kv_pool`` 做 dtype / 容量判断，
        MLA buffer 读写走 ``_kv_pool()``（由 hybrid pool 自己按 full-layer id 路由）。
        """
        hits = self._old_pool_attr_hits("hooks/layers/attention/_nsa_backend_impl.py")
        self.assertFalse(
            hits,
            "_nsa_backend_impl 仍在读 forward_batch.token_to_kv_pool：\n  "
            + "\n  ".join(hits),
        )

    # -- 行为：解包路径 ------------------------------------------------------

    def test_unwraps_full_kv_pool_from_outer_pool(self):
        """外层池有 `full_kv_pool` → 必须返回子池（否则 page_size 断言 64 会炸）。"""
        inner = _FakePool(has_attr=False)
        inner.page_size = 64
        outer = _FakePool(full_kv_pool=inner)
        self.dsa_indexer.get_token_to_kv_pool = lambda: outer

        got = self.nsa_indexer._indexer_kv_pool()
        self.assertIs(got, inner, f"没有解包 full_kv_pool，拿到的是 {got!r}")
        self.assertEqual(got.page_size, 64)

    def test_full_kv_pool_none_falls_back_to_outer_pool(self):
        """`full_kv_pool is None` 时按"没有该属性"处理（不能返回 None）。"""
        outer = _FakePool(full_kv_pool=None)
        self.dsa_indexer.get_token_to_kv_pool = lambda: outer
        self.assertIs(self.nsa_indexer._indexer_kv_pool(), outer)

    def test_dsv4_shaped_pool_without_attribute_is_returned_as_is(self):
        """DSV4 的池没有 `full_kv_pool`：行为不变，原样返回。"""
        outer = _FakePool(has_attr=False)
        self.dsa_indexer.get_token_to_kv_pool = lambda: outer
        self.assertIs(self.nsa_indexer._indexer_kv_pool(), outer)

    def test_uses_patched_module_attribute_not_the_imported_binding(self):
        """EB5 的 `register_dsa_full_pool_patch` 换的是**模块属性**。

        插件用 `from ... import get_token_to_kv_pool` 绑的是原函数，直接调用会拿到
        外层 HCA 池 → `No indexer pool for compression ratio 128`。这里用两个返回值
        不同的假函数把两条路区分开。
        """
        inner = _FakePool(has_attr=False)
        inner.page_size = 64
        outer = _FakePool(full_kv_pool=inner)

        self.dsa_indexer.get_token_to_kv_pool = lambda: outer          # 被 patch 过的模块属性
        self.nsa_indexer.get_token_to_kv_pool = lambda: _FakePool(     # 旧的 import 绑定
            has_attr=False
        )

        self.assertIs(self.nsa_indexer._indexer_kv_pool(), inner)

    def test_falls_back_to_imported_binding_when_module_attr_missing(self):
        """拿不到模块属性时，退回本文件 import 的那个函数（老 fork / 未打补丁场景）。"""
        sentinel = _FakePool(has_attr=False)
        if hasattr(self.dsa_indexer, "get_token_to_kv_pool"):
            del self.dsa_indexer.get_token_to_kv_pool
        self.nsa_indexer.get_token_to_kv_pool = lambda: sentinel

        self.assertIs(self.nsa_indexer._indexer_kv_pool(), sentinel)


# ---------------------------------------------------------------------------
# 2) set_mla_kv_buffer_kernel 形参契约
# ---------------------------------------------------------------------------

# fork 的 `_set_mla_kv_buffer_impl` 按这个顺序传 10 个位置参数
# （`python/sglang/kernels/ops/kvcache/mla_buffer.py:222-237`）
_EXPECTED_MLA_KV_PARAMS = [
    "kv_buffer",
    "cache_k_nope",
    "cache_k_rope",
    "loc",
    "reserved_skip_index",
    "buffer_stride",
    "nope_stride",
    "rope_stride",
    "nope_dim",
    "rope_dim",
]


class TestSetMlaKvBufferSignature(unittest.TestCase):
    """`set_mla_kv_buffer_kernel` 必须能接住 fork 的 10 个位置参数 + 3 个新 kwargs。"""

    @classmethod
    def setUpClass(cls):
        import sglang_kunlun.kernels.kernel_ops as kernel_ops

        cls.kernel_ops = kernel_ops
        # staticmethod：普通函数挂类属性会变成 bound method（调用时多一个 self）
        cls.fn = staticmethod(kernel_ops.set_mla_kv_buffer_kernel)
        cls.sig = inspect.signature(kernel_ops.set_mla_kv_buffer_kernel)

    def test_ten_positional_params_in_order(self):
        positional = [
            p.name
            for p in self.sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        self.assertEqual(
            positional,
            _EXPECTED_MLA_KV_PARAMS,
            "set_mla_kv_buffer_kernel 的位置参数与 fork 调用点不一致；少一个就是服务里那次 "
            "`TypeError: takes 9 positional arguments but 10 were given`",
        )

    def test_reserved_skip_index_is_fifth_and_defaults_to_zero(self):
        params = list(self.sig.parameters.values())
        self.assertEqual(params[4].name, "reserved_skip_index")
        self.assertEqual(params[4].default, 0)
        self.assertIs(params[4].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)

    def test_block_is_keyword_only_and_kwargs_swallows_dcp(self):
        self.assertIn(
            "BLOCK", self.sig.parameters, "BLOCK 必须是关键字参数（fork 用 BLOCK=... 传）"
        )
        self.assertIs(
            self.sig.parameters["BLOCK"].kind, inspect.Parameter.KEYWORD_ONLY
        )
        self.assertTrue(
            any(
                p.kind is inspect.Parameter.VAR_KEYWORD
                for p in self.sig.parameters.values()
            ),
            "缺少 **kwargs —— fork 的 DCP_RANK / DCP_WORLD_SIZE 会直接 TypeError",
        )

    def test_binding_accepts_fork_call_shape(self):
        """按 fork 的真实调用形状 bind：10 位置 + BLOCK + DCP_RANK + DCP_WORLD_SIZE。"""
        args = [object()] * 10
        self.sig.bind(*args, BLOCK=64, DCP_RANK=0, DCP_WORLD_SIZE=1, USE_GDC=False)
        # BLOCK 是必填的关键字参数：漏了就 bind 不上（fork 一定会传）
        with self.assertRaises(TypeError):
            self.sig.bind(*args)

    def test_reserved_skip_index_guard_is_loud(self):
        """`reserved_skip_index != 0` 在昆仑上未实现，必须显式断言（不能静默整行写）。"""
        args = [object()] * 10
        args[4] = 7
        with self.assertRaises(AssertionError) as ctx:
            self.fn(*args, BLOCK=64)
        self.assertIn("reserved_skip_index", str(ctx.exception))

    def test_fork_call_site_still_passes_ten_positional_args(self):
        """反向钉住 fork：调用点一旦改签名，这里立刻报（而不是等整网前向）。"""
        import sglang.kernels.ops.kvcache.mla_buffer as mla_buffer

        tree = ast.parse(pathlib.Path(mla_buffer.__file__).read_text(encoding="utf-8"))
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Subscript) and isinstance(fn.value, ast.Name)):
                continue
            if fn.value.id != "set_mla_kv_buffer_kernel":
                continue
            calls.append(node)
        self.assertEqual(
            len(calls), 1, f"{mla_buffer.__file__} 里 set_mla_kv_buffer_kernel 调用点数变了"
        )
        call = calls[0]
        self.assertEqual(
            len(call.args),
            10,
            f"{mla_buffer.__file__}:{call.lineno} 位置参数个数变了，插件的形参要同步",
        )
        kwargs = {kw.arg for kw in call.keywords if kw.arg}
        for name in ("BLOCK", "DCP_RANK", "DCP_WORLD_SIZE"):
            self.assertIn(name, kwargs, f"fork 调用点不再传 {name}")


# ---------------------------------------------------------------------------
# 3) 融合 kernel 的 kernel 级 torch 替代（tilelang / deep_gemm）
# ---------------------------------------------------------------------------


class TestKernelLevelTorchReplacements(unittest.TestCase):
    """`tilelang` / `deep_gemm` 的融合 kernel 必须被**kernel 级** torch 实现替换。

    2026-10-20 起 EB5 不再"只在调用层绕开"：`eb5_kunlun_adapt.apply()` 把这些符号
    本身换成 `*_torch` 实现，所以 `test_eb5_hc_pre.py` / `test_eb5_hc_post.py` 里
    直调 kernel 的用例仍然有意义 —— 它们验的就是这些 torch 实现。

    本类只做**绑定关系 + 可调用性**的自检（纯 CPU）；数值由那两个设备用例覆盖。
    """

    @classmethod
    def setUpClass(cls):
        import eb5_kunlun_adapt as adapt
        import sglang.kernels.ops.layernorm.mhc as mhc
        import sglang.srt.layers.deep_gemm_wrapper.entrypoint as dg

        cls.adapt = adapt
        cls.mhc = mhc
        cls.dg = dg

    def test_mhc_module_symbols_point_to_adapt_impls(self):
        expected = {
            "mhc_post_tilelang": "mhc_post_tilelang",
            "mhc_pre_big_fuse_with_norm_tilelang": "mhc_pre_big_fuse_with_norm_tilelang",
            "mhc_pre_big_fuse_tilelang": "mhc_pre_big_fuse_tilelang",
            "mhc_pre_gemm_sqrsum_tilelang": "mhc_pre_gemm_sqrsum_tilelang",
            "mhc_pre_gemm_sqrsum_splitk_kernel": "mhc_pre_gemm_sqrsum_splitk_kernel",
            # fork 里这个符号是**私有**名（`_mhc_pre_gemm_sqrsum_dispatch`），
            # 适配层保持同名，别写成不带下划线的版本。
            "_mhc_pre_gemm_sqrsum_dispatch": "_mhc_pre_gemm_sqrsum_dispatch",
        }
        problems = []
        for target, impl_name in expected.items():
            impl = getattr(self.adapt, impl_name, None)
            if impl is None:
                problems.append(f"adapt 里没有 {impl_name}")
                continue
            got = getattr(self.mhc, target, None)
            if got is not impl:
                problems.append(f"mhc.{target} 不是 adapt.{impl_name}（当前 {got!r}）")
        self.assertFalse(
            problems,
            "融合 kernel 没有在 kernel 级被 torch 实现替换：\n  " + "\n  ".join(problems),
        )

    def test_deep_gemm_tf32_hc_prenorm_gemm_is_replaced(self):
        self.assertIs(
            self.dg.tf32_hc_prenorm_gemm,
            self.adapt.tf32_hc_prenorm_gemm,
            "deep_gemm_wrapper.entrypoint.tf32_hc_prenorm_gemm 仍是 deep_gemm 实现",
        )

    def test_splitk_factory_returns_two_callables(self):
        """`mhc_pre` 按 `kernel_0, _ = ...` 解包，工厂必须返回二元组。"""
        out = self.mhc.mhc_pre_gemm_sqrsum_splitk_kernel(24, 16384, split_k=32)
        self.assertEqual(len(out), 2)
        self.assertTrue(callable(out[0]))
        self.assertTrue(callable(out[1]))

    def test_splitk_factory_validates_shape_contract(self):
        """原 kernel 的硬约束（hc_mult3<=32、K 能被 split_k 整除）不能丢。"""
        with self.assertRaises(ValueError):
            self.mhc.mhc_pre_gemm_sqrsum_splitk_kernel(33, 16384, split_k=32)
        with self.assertRaises(ValueError):
            self.mhc.mhc_pre_gemm_sqrsum_splitk_kernel(24, 16384, split_k=7)

    def test_dispatch_returns_torch_impl(self):
        """`_mhc_pre_gemm_sqrsum_dispatch()` 必须返回**torch 版**（不再是 tilelang）。"""
        fn = self.mhc._mhc_pre_gemm_sqrsum_dispatch()
        self.assertIs(
            fn,
            self.adapt.mhc_pre_gemm_sqrsum_tilelang,
            "dispatch 没有指向 torch 实现（fork 的 sm120 分支会返回重新 jit 的 tilelang）",
        )


# ---------------------------------------------------------------------------
# 4) _topk_all_visible
# ---------------------------------------------------------------------------


class _FakeIndexerMetadata:
    """`_topk_all_visible` 只读这三个方法（全部 CPU）。"""

    def __init__(self, page_table_1, seqlens_expanded, extend_len_cpu=None):
        self._page_table_1 = page_table_1
        self._seqlens_expanded = seqlens_expanded
        self._extend_len_cpu = list(extend_len_cpu or [])

    def get_page_table_1(self):
        return self._page_table_1

    def get_seqlens_expanded(self):
        return self._seqlens_expanded

    def get_dsa_extend_len_cpu(self):
        return list(self._extend_len_cpu)


class TestTopkAllVisible(unittest.TestCase):
    """`kv_len <= topk` 的免 logits 快路径：直接拼 page table，右侧补 -1。"""

    @classmethod
    def setUpClass(cls):
        # staticmethod：普通函数挂类属性会变成 bound method（多传一个 self）
        cls.fn = staticmethod(rc._import_nsa_indexer()._topk_all_visible)

    def test_shape_dtype_and_right_padding(self):
        topk = 8
        page_table = torch.arange(3 * 16, dtype=torch.int32).reshape(3, 16) + 100
        visible = torch.tensor([1, 5, 8], dtype=torch.int32)
        out = self.fn(
            _FakeIndexerMetadata(page_table, visible), topk, torch.device("cpu")
        )

        self.assertEqual(tuple(out.shape), (3, topk))
        self.assertEqual(out.dtype, torch.int32)
        expected = torch.full((3, topk), -1, dtype=torch.int32)
        for row, n in enumerate((1, 5, 8)):
            expected[row, :n] = page_table[row, :n]
        self.assertTrue(
            torch.equal(out, expected),
            f"右侧 -1 填充/槽位映射不对:\n got={out}\n exp={expected}",
        )

    def test_expands_page_table_rows_by_extend_lens(self):
        """`page_table_1` 是 per-request 时，按 `extend_len_cpu` 展开成 per-token。"""
        topk = 4
        page_table = torch.tensor(
            [[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32
        )
        visible = torch.tensor([1, 2, 3, 4], dtype=torch.int32)  # 4 token / 2 request
        out = self.fn(
            _FakeIndexerMetadata(page_table, visible, extend_len_cpu=[1, 3]),
            topk,
            torch.device("cpu"),
        )
        expected = torch.tensor(
            [
                [10, -1, -1, -1],
                [20, 21, -1, -1],
                [20, 21, 22, -1],
                [20, 21, 22, 23],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(out, expected), f"got={out}\nexp={expected}")

    def test_topk_larger_than_page_table_width(self):
        """`topk > 行宽` 时先取满行宽，再把其余列补 -1。"""
        topk, width = 9, 5
        page_table = torch.arange(2 * width, dtype=torch.int32).reshape(2, width) + 7
        visible = torch.tensor([2, width], dtype=torch.int32)
        out = self.fn(
            _FakeIndexerMetadata(page_table, visible), topk, torch.device("cpu")
        )
        self.assertEqual(tuple(out.shape), (2, topk))
        self.assertTrue(torch.equal(out[0, :2], page_table[0, :2]))
        self.assertTrue(torch.equal(out[1, :width], page_table[1]))
        self.assertTrue(bool((out[0, 2:] == -1).all()))
        self.assertTrue(bool((out[1, width:] == -1).all()))

    def test_tokens_zero(self):
        """tokens=0（DP rank 上没分到 token）不能崩，shape 仍是 (0, topk)。"""
        page_table = torch.zeros(0, 8, dtype=torch.int32)
        visible = torch.zeros(0, dtype=torch.int32)
        out = self.fn(
            _FakeIndexerMetadata(page_table, visible), 8, torch.device("cpu")
        )
        self.assertEqual(tuple(out.shape), (0, 8))
        self.assertEqual(out.dtype, torch.int32)

    def test_indices_match_torch_topk_on_monotonic_logits(self):
        """与 `torch.topk` 的 index 一致：logits 随列严格递减 + 恒等 page table 时，
        topk 就是 `[0, 1, ..., topk-1]`（即"全部可见"的语义）。"""
        tokens, topk = 3, 8
        logits = -torch.arange(topk, dtype=torch.float32).unsqueeze(0).repeat(tokens, 1)
        ref_idx = torch.topk(logits, topk, dim=1).indices.to(torch.int32)
        self.assertTrue(
            torch.equal(
                ref_idx,
                torch.arange(topk, dtype=torch.int32).unsqueeze(0).repeat(tokens, 1),
            ),
            "前提不成立：logits 不是严格递减",
        )

        page_table = torch.arange(topk, dtype=torch.int32).unsqueeze(0).repeat(tokens, 1)
        visible = torch.full((tokens,), topk, dtype=torch.int32)
        out = self.fn(
            _FakeIndexerMetadata(page_table, visible), topk, torch.device("cpu")
        )
        self.assertTrue(torch.equal(out, ref_idx), f"got={out}\nref={ref_idx}")


# ---------------------------------------------------------------------------
# 5) transform_index_page_table_{prefill,decode}（torch 版）
# ---------------------------------------------------------------------------


class TestTransformIndexPageTable(unittest.TestCase):
    """ torch 化后的 prefill / decode 索引表变换，与文件内 ref 逐元素一致。

    ``fast`` 版与 ``ref`` 版的差别只有"越界下标"：``ref`` 的 ``gather`` 要求下标
    ``< 行宽``，越界会抛异常；``fast`` 版按 kernel 语义写 ``-1``。所以越界单独测。
    """

    TOPK = 2048
    P = 48  # page_table 行宽（远小于 topk，天然覆盖"topk 靠后全空洞"）

    @classmethod
    def setUpClass(cls):
        import sglang.kernels.ops.attention.dsa.transform_index as m

        cls.m = m

    def _inputs(self, bs, n_rows):
        g = torch.Generator().manual_seed(20261020 + bs * 7 + n_rows)
        # page_table 里留一些 -1（掩码空洞）
        page_table = torch.randint(0, 5000, (bs, self.P), generator=g, dtype=torch.int32)
        page_table[:, ::11] = -1
        topk_indices = torch.randint(
            -1, self.P, (n_rows, self.TOPK), generator=g, dtype=torch.int32
        )
        topk_indices[:, ::13] = -1
        return page_table, topk_indices

    def _prefill_cases(self):
        """8 组 prefill 形状（bs / extend 长度 / expanded / spec padding / cu_seqlens）。"""
        return [
            dict(bs=1, extend=[1], expanded=False),
            dict(bs=1, extend=[7], expanded=False),
            dict(bs=2, extend=[1, 3], expanded=False),
            dict(bs=3, extend=[0, 4, 2], expanded=False),      # 含 0 长度请求
            dict(bs=1, extend=[5], expanded=True),
            dict(bs=2, extend=[2, 3], expanded=True),          # page_table 行 = token 行
            dict(bs=2, extend=[2, 2], expanded=False, output_num_tokens=6),  # spec padding
            dict(bs=2, extend=[3, 1], expanded=False, use_cu=True),
        ]

    def test_prefill_matches_ref_for_eight_shapes(self):
        for i, case in enumerate(self._prefill_cases()):
            with self.subTest(case=i, **case):
                bs, extend = case["bs"], case["extend"]
                real = sum(extend)
                expanded = case["expanded"]
                # expanded 时 page_table 的行就是 token 行（real 行）；否则是请求行
                pt_rows = real if expanded else bs
                page_table, topk_indices = self._inputs(pt_rows, real)

                out_n = case.get("output_num_tokens")
                kwargs = dict(
                    page_size=1,
                    output_num_tokens=out_n,
                    page_table_is_expanded=expanded,
                )
                cu = None
                if case.get("use_cu"):
                    cu = torch.tensor(
                        [0] + list(torch.cumsum(torch.tensor(extend), 0).tolist()),
                        dtype=torch.int32,
                    )
                # fork 的 wrapper 是 `def f(**kwargs)`，只能关键字调用
                got = self.m.transform_index_page_table_prefill(
                    page_table=page_table,
                    topk_indices=topk_indices,
                    extend_lens_cpu=extend,
                    cu_seqlens_q=cu,
                    **kwargs,
                )
                ref = self.m.transform_index_page_table_prefill_ref(
                    page_table, topk_indices, extend, **kwargs
                )
                self.assertEqual(got.dtype, torch.int32)
                self.assertEqual(tuple(got.shape), tuple(ref.shape))
                self.assertTrue(
                    torch.equal(got, ref),
                    f"case={i} prefill 与 ref 不一致: "
                    f"{(got != ref).sum().item()} 个元素不同",
                )

    def test_prefill_out_of_range_and_holes_fill_minus_one(self):
        """越界下标 / 行尾 padding 一律写 -1（ref 的 gather 会抛异常，这里只测 fast）。"""
        extend = [3]
        page_table = torch.full((1, 8), 5, dtype=torch.int32)
        topk_indices = torch.full((3, self.TOPK), -1, dtype=torch.int32)
        topk_indices[0, 0] = 0
        topk_indices[0, 1] = 7           # 行内最后一个合法列
        topk_indices[0, 2] = 8           # 越界（== 行宽）
        topk_indices[0, 3] = self.P + 9  # 远远越界
        topk_indices[1, 0] = 3
        out = self.m.transform_index_page_table_prefill(
            page_table=page_table, topk_indices=topk_indices, extend_lens_cpu=extend
        )
        self.assertEqual(out[0, 0].item(), 5)
        self.assertEqual(out[0, 1].item(), 5)
        self.assertEqual(out[0, 2].item(), -1)
        self.assertEqual(out[0, 3].item(), -1)
        self.assertTrue(bool((out[0, 4:] == -1).all()))
        self.assertEqual(out[1, 0].item(), 5)
        self.assertTrue(bool((out[2] == -1).all()), "全是 -1 的行必须保持 -1")

    def test_prefill_zero_tokens(self):
        page_table = torch.zeros(0, 8, dtype=torch.int32)
        topk_indices = torch.zeros(0, self.TOPK, dtype=torch.int32)
        out = self.m.transform_index_page_table_prefill(
            page_table=page_table, topk_indices=topk_indices, extend_lens_cpu=[]
        )
        self.assertEqual(tuple(out.shape), (0, self.TOPK))

    def test_decode_matches_ref(self):
        bs, qo = 4, 4
        g = torch.Generator().manual_seed(7)
        page_table = torch.randint(0, 3000, (bs, 64), generator=g, dtype=torch.int32)
        page_table[:, ::9] = -1  # 空洞
        topk_indices = torch.randint(
            -1, 64, (qo, self.TOPK), generator=g, dtype=torch.int32
        )
        topk_indices[:, ::17] = -1

        got = self.m.transform_index_page_table_decode(
            page_table=page_table, topk_indices=topk_indices
        )
        ref = self.m.transform_index_page_table_decode_ref(page_table, topk_indices)
        self.assertEqual(got.dtype, torch.int32)
        self.assertTrue(torch.equal(got, ref), "decode 与 ref 不一致")

    def test_decode_out_of_range_fills_minus_one(self):
        page_table = torch.full((1, 4), 9, dtype=torch.int32)
        topk_indices = torch.full((1, self.TOPK), -1, dtype=torch.int32)
        topk_indices[0, 0] = 0
        topk_indices[0, 1] = 3
        topk_indices[0, 2] = 4        # 越界
        out = self.m.transform_index_page_table_decode(
            page_table=page_table, topk_indices=topk_indices
        )
        self.assertEqual(out[0, 0].item(), 9)
        self.assertEqual(out[0, 1].item(), 9)
        self.assertEqual(out[0, 2].item(), -1)
        self.assertTrue(bool((out[0, 3:] == -1).all()))

    def test_decode_result_buffer_is_reused(self):
        """传了 `result` 时必须原地写（fork 用自己的 buffer 省一次分配）。"""
        page_table = torch.arange(8, dtype=torch.int32).reshape(1, 8)
        topk_indices = torch.arange(self.TOPK, dtype=torch.int32).reshape(1, -1) % 8
        buf = torch.empty_like(topk_indices, dtype=torch.int32)
        out = self.m.transform_index_page_table_decode(
            page_table=page_table, topk_indices=topk_indices, result=buf
        )
        self.assertIs(out, buf)


# ---------------------------------------------------------------------------
# 6) dsa_swa_index.append_indexer_swa（3 种 mode）
# ---------------------------------------------------------------------------

_SWA_RAGGED, _SWA_PAGED_IDENTITY, _SWA_PAGED_MAPPED = 0, 1, 2


def _swa_ref(
    topk_indices,
    seq_lens,
    *,
    num_rows,
    topk,
    window,
    alignment,
    mode,
    page_table=None,
    page_size=1,
    row_aux=None,
):
    """按 `_append_indexer_swa_kernel` 逐行复刻（Python 版，只用于对照）。"""
    width = -(-(topk + window) // alignment) * alignment
    out = torch.full((num_rows, width), -1, dtype=torch.int32)
    lengths = torch.zeros(num_rows, dtype=torch.int32)
    live = seq_lens.numel()
    for r in range(num_rows):
        if r >= live:
            # kernel 里 `row_is_live=False`：主拷贝的 mask 全不成立，整行保持 -1
            continue
        sl = max(int(seq_lens[r]), 0)
        ilen = min(max(sl - window, 0), topk)
        swa = min(sl, window)
        for c in range(width):
            if c < ilen or (c < topk and c >= ilen + window):
                if c < topk_indices.shape[1]:
                    out[r, c] = topk_indices[r, c]
        for w in range(window):
            if not (r < live and w < swa):
                continue
            pos = sl - swa + w
            if mode == _SWA_RAGGED:
                val = int(row_aux[r]) + pos
            else:
                prow = int(row_aux[r]) if mode == _SWA_PAGED_MAPPED else r
                pcol = pos // page_size
                if 0 <= prow < page_table.shape[0] and pcol < page_table.shape[1]:
                    val = int(page_table[prow, pcol]) * page_size + pos % page_size
                else:
                    val = -1
            out[r, ilen + w] = val
        lengths[r] = ilen + swa
    return out, lengths


class TestAppendIndexerSwa(unittest.TestCase):
    """`top-k ∪ 最近 window` 的索引行拼接：3 种 mode 与 ref 一致。"""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.internal.layers.attention.mla_hca.dsa_swa_index import (
            append_indexer_swa_triton,
        )

        cls.fn = staticmethod(append_indexer_swa_triton)

    TOPK, WINDOW, ALIGN = 8, 4, 4
    SEQ_LENS = [0, 1, 3, 5, 12]
    NUM_ROWS = 6  # 最后一行是 CUDA-graph padding（超出 live_rows）

    def _topk_indices(self):
        g = torch.Generator().manual_seed(11)
        return torch.randint(
            -1, 900, (self.NUM_ROWS, self.TOPK), generator=g, dtype=torch.int32
        )

    def _common(self):
        return dict(
            num_rows=self.NUM_ROWS,
            topk=self.TOPK,
            window=self.WINDOW,
            alignment=self.ALIGN,
        )

    def test_ragged_mode_matches_ref(self):
        seq_lens = torch.tensor(self.SEQ_LENS, dtype=torch.int32)
        ragged = torch.tensor([0, 100, 200, 300, 400], dtype=torch.int64)
        topk_indices = self._topk_indices()
        got, got_len = self.fn(
            topk_indices, seq_lens, ragged_offsets=ragged, **self._common()
        )
        ref, ref_len = _swa_ref(
            topk_indices,
            seq_lens,
            mode=_SWA_RAGGED,
            row_aux=ragged,
            **self._common(),
        )
        self.assertEqual(got.dtype, torch.int32)
        self.assertTrue(torch.equal(got, ref), f"ragged 行不一致:\n{got}\n{ref}")
        self.assertTrue(torch.equal(got_len, ref_len), "ragged 行长度不一致")

    def test_paged_identity_mode_matches_ref(self):
        seq_lens = torch.tensor(self.SEQ_LENS, dtype=torch.int32)
        page_table = torch.arange(self.NUM_ROWS * 16, dtype=torch.int32).reshape(
            self.NUM_ROWS, 16
        )
        topk_indices = self._topk_indices()
        got, got_len = self.fn(
            topk_indices,
            seq_lens,
            page_table=page_table,
            page_size=2,
            **self._common(),
        )
        ref, ref_len = _swa_ref(
            topk_indices,
            seq_lens,
            mode=_SWA_PAGED_IDENTITY,
            page_table=page_table,
            page_size=2,
            **self._common(),
        )
        self.assertTrue(torch.equal(got, ref), f"paged-identity 行不一致:\n{got}\n{ref}")
        self.assertTrue(torch.equal(got_len, ref_len))

    def test_paged_mapped_mode_matches_ref(self):
        """有 `page_row_indices`（token → 请求行）时按映射取页。"""
        seq_lens = torch.tensor(self.SEQ_LENS, dtype=torch.int32)
        page_table = torch.arange(2 * 16, dtype=torch.int32).reshape(2, 16)
        page_rows = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int64)
        topk_indices = self._topk_indices()
        got, got_len = self.fn(
            topk_indices,
            seq_lens,
            page_table=page_table,
            page_size=4,
            page_row_indices=page_rows,
            **self._common(),
        )
        ref, ref_len = _swa_ref(
            topk_indices,
            seq_lens,
            mode=_SWA_PAGED_MAPPED,
            page_table=page_table,
            page_size=4,
            row_aux=page_rows,
            **self._common(),
        )
        self.assertTrue(torch.equal(got, ref), f"paged-mapped 行不一致:\n{got}\n{ref}")
        self.assertTrue(torch.equal(got_len, ref_len))

    def test_output_width_is_aligned_and_lengths_are_capped(self):
        """输出宽度按 `alignment` 上取整；行长度 = ilen + swa_len。"""
        seq_lens = torch.tensor([100], dtype=torch.int32)  # 远大于 topk+window
        topk_indices = torch.zeros(1, self.TOPK, dtype=torch.int32)
        got, got_len = self.fn(
            topk_indices,
            seq_lens,
            ragged_offsets=torch.zeros(1, dtype=torch.int64),
            num_rows=1,
            topk=self.TOPK,
            window=self.WINDOW,
            alignment=8,
        )
        self.assertEqual(got.shape[1], 16)  # ceil((8+4)/8)*8
        self.assertEqual(got_len[0].item(), self.TOPK + self.WINDOW)
        self.assertTrue(bool((got[0, : self.TOPK] == 0).all()))
        # window 段写在 [indexer_len, indexer_len+window) = [8, 12)
        self.assertTrue(
            torch.equal(
                got[0, self.TOPK : self.TOPK + self.WINDOW],
                torch.arange(100 - self.WINDOW, 100, dtype=torch.int32),
            )
        )
        self.assertTrue(bool((got[0, self.TOPK + self.WINDOW :] == -1).all()))

    def test_num_rows_zero(self):
        empty_topk = torch.zeros(0, self.TOPK, dtype=torch.int32)
        kwargs = self._common()
        kwargs["num_rows"] = 0
        got, got_len = self.fn(
            empty_topk,
            torch.zeros(0, dtype=torch.int32),
            ragged_offsets=torch.zeros(0, dtype=torch.int64),
            **kwargs,
        )
        self.assertEqual(got.shape[0], 0)
        self.assertEqual(got_len.shape[0], 0)


# ---------------------------------------------------------------------------
# 7) _get_topk_ragged 的 chunk 路径：未定义名（历史 bug: k_interval_*）
# ---------------------------------------------------------------------------


def _module_level_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.If):
            # `if TYPE_CHECKING:` 里的 import 在运行期确实不存在，但它只用于
            # `if TYPE_CHECKING: assert isinstance(...)`，按"已绑定"处理。
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for a in sub.names:
                        names.add((a.asname or a.name).split(".")[0])
    return names


def _function_bindings(fn: ast.AST) -> set[str]:
    """函数（含嵌套）里出现的所有"被绑定"的名字：参数 / 赋值 / import / for / with …"""
    names: set[str] = set()
    a = fn.args
    for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
        names.add(arg.arg)
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                names.add((al.asname or al.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


class TestTopkRaggedChunkPath(unittest.TestCase):
    """chunk 分支曾经引用过未定义的 `k_interval_cpu/xpu`（改名成 `q_interval_*` 时漏改）。"""

    @classmethod
    def setUpClass(cls):
        cls.plugin_path = _plugin_file("hooks/layers/attention/nsa/nsa_indexer.py")
        cls.src = cls.plugin_path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.src)
        cls.fn_node = next(
            n
            for n in ast.walk(cls.tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_get_topk_ragged"
        )

    def test_no_k_interval_identifiers(self):
        hits = [
            (n.id, n.lineno)
            for n in ast.walk(self.fn_node)
            if isinstance(n, ast.Name) and "k_interval" in n.id
        ]
        hits += [
            (n.attr, n.lineno)
            for n in ast.walk(self.fn_node)
            if isinstance(n, ast.Attribute) and "k_interval" in n.attr
        ]
        self.assertFalse(
            hits,
            "_get_topk_ragged 仍引用未定义的 k_interval_*（NameError 会在 chunk 路径炸）："
            f"{hits}",
        )

    def test_chunk_branch_is_present_and_uses_q_interval(self):
        """自检：chunk 分支还在（否则上一条会假绿）。"""
        self.assertIn("q_interval_cpu", self.src)
        self.assertIn("q_interval_xpu", self.src)
        self.assertIn("need_chunk", self.src)

    def test_no_undefined_names_in_function(self):
        """作用域分析：函数体内所有 `Load` 的名字都必须有绑定（参数/局部/模块级/builtin）。

        这条比"grep k_interval"更通用 —— 任何漏改的局部变量都会在这里报出来。
        """
        bound = (
            _module_level_names(self.tree)
            | _function_bindings(self.fn_node)
            | set(dir(builtins))
        )
        missing = sorted(
            {
                (n.id, n.lineno)
                for n in ast.walk(self.fn_node)
                if isinstance(n, ast.Name)
                and isinstance(n.ctx, ast.Load)
                and n.id not in bound
            }
        )
        self.assertFalse(
            missing,
            "函数里有未绑定的名字（运行时 NameError）：\n  "
            + "\n  ".join(
                f"{name} @ {self.plugin_path}:{lineno}" for name, lineno in missing
            ),
        )


# ---------------------------------------------------------------------------
# 8) _nsa_backend_impl 的 5 个 dsa_cp_* 符号
# ---------------------------------------------------------------------------

_DSA_CP_SYMBOLS = (
    "can_dsa_prefill_cp_round_robin_split",
    "is_dsa_prefill_cp_in_seq_split",
    "is_dsa_prefill_cp_round_robin_split",
    "dsa_cp_round_robin_split_data",
    "dsa_cp_round_robin_split_q_seqs",
)


class TestNsaBackendDsaCpSymbols(unittest.TestCase):
    """fork 删掉的 5 个 `dsa_cp_*` 符号由插件自己补齐。

    2026-10-20 之前 `_nsa_backend_impl` 直接 `from dsa.utils import ...`，而 fork
    在 CP V1 deprecation 里删掉了 round-robin 家族，于是整个模块 import 失败、
    `sglang.srt.layers.attention.nsa_backend` 从未被 redirect。

    现在：
      1. `can_dsa_prefill_cp_round_robin_split` / `is_dsa_prefill_cp_round_robin_split`
         改用 fork 现有等价符号（interleave 别名）；
      2. `is_dsa_prefill_cp_in_seq_split` 改成 `getattr` 守卫（不存在 = 该模式未启用）；
      3. `dsa_cp_round_robin_split_data` / `dsa_cp_round_robin_split_q_seqs` 用纯 torch
         按原语义补一份，若未来 fork 恢复同名符号则优先用 fork 的。
    EB5 的 prefill CP 关闭，这些只是"显式守卫"而非热路径。
    """

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.dsa import utils as dsa_utils

        cls.dsa_utils = dsa_utils

    def _missing(self):
        return [s for s in _DSA_CP_SYMBOLS if not hasattr(self.dsa_utils, s)]

    # fork 名字 -> 插件里的绑定名（nsa 前缀是历史命名，保留以免改所有调用点）
    _PLUGIN_ALIASES = {
        "can_dsa_prefill_cp_round_robin_split": (
            "can_nsa_prefill_cp_round_robin_split"
        ),
        "is_dsa_prefill_cp_in_seq_split": "is_nsa_prefill_cp_in_seq_split",
        "is_dsa_prefill_cp_round_robin_split": (
            "is_nsa_prefill_cp_round_robin_split"
        ),
        "dsa_cp_round_robin_split_data": "nsa_cp_round_robin_split_data",
        "dsa_cp_round_robin_split_q_seqs": "nsa_cp_round_robin_split_q_seqs",
    }

    def test_missing_fork_symbols_are_shimmed(self):
        """fork 缺的符号由插件补齐：模块 import 成功，且每个缺失名都有绑定。"""
        import importlib

        mod = importlib.import_module(
            "sglang_kunlun.hooks.layers.attention._nsa_backend_impl"
        )
        missing = self._missing()
        absent = [
            name
            for name in missing
            if not hasattr(mod, self._PLUGIN_ALIASES[name])
        ]
        self.assertFalse(
            absent,
            f"fork 缺失且插件也没有等价实现：{absent}（fork missing={missing}）",
        )

    def test_redirect_installs_impl_in_sys_modules(self):
        """redirect 必须把 `_nsa_backend_impl` 真正装成 fork 的 nsa_backend。"""
        import sglang_kunlun.hooks.layers.attention as attn_pkg
        from sglang_kunlun.hooks.layers.attention import _nsa_backend_impl

        target = "sglang.srt.layers.attention.nsa_backend"
        saved = sys.modules.pop(target, None)
        try:
            attn_pkg._redirect_nsa_backend()
            self.assertIn(target, sys.modules, "redirect 没有安装任何模块")
            self.assertIs(
                sys.modules[target],
                _nsa_backend_impl,
                "redirect 装进去的不是昆仑实现",
            )
        finally:
            sys.modules.pop(target, None)
            if saved is not None:
                sys.modules[target] = saved

    def test_nsa_backend_impl_imports_and_redirects(self):
        """`_nsa_backend_impl` 能用 fork 现有符号 import，并被装成 nsa_backend。"""
        import importlib

        mod = importlib.import_module(
            "sglang_kunlun.hooks.layers.attention._nsa_backend_impl"
        )
        self.assertTrue(hasattr(mod, "NativeSparseAttnBackend"))
        self.assertIs(
            sys.modules.get("sglang.srt.layers.attention.nsa_backend"),
            mod,
            "nsa_backend 没有被重定向到昆仑实现",
        )


# ---------------------------------------------------------------------------
# 9) 旧名 Indexer._get_topk_ragged_with_cp
# ---------------------------------------------------------------------------


class TestTopkRaggedWithCpRename(unittest.TestCase):
    """fork 的 `Indexer` 已没有 `_get_topk_ragged_with_cp`，插件不应再引用旧名。"""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.attention.dsa import dsa_indexer

        cls.indexer_cls = dsa_indexer.Indexer
        cls.plugin_path = _plugin_file("hooks/layers/attention/nsa/nsa_indexer.py")
        cls.tree = ast.parse(cls.plugin_path.read_text(encoding="utf-8"))

    def test_fork_indexer_has_no_old_method(self):
        self.assertFalse(
            hasattr(self.indexer_cls, "_get_topk_ragged_with_cp"),
            "fork 又长回了 _get_topk_ragged_with_cp —— 插件侧结论需要复核",
        )

    def _old_name_refs(self):
        refs = []
        for node in ast.walk(self.tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "_get_topk_ragged_with_cp"
            ):
                refs.append(("def", node.lineno))
            elif (
                isinstance(node, ast.Attribute)
                and node.attr == "_get_topk_ragged_with_cp"
            ):
                refs.append(("attr", node.lineno))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if "_get_topk_ragged_with_cp" in node.value:
                    refs.append(("str", node.lineno))
        return refs

    def test_fork_nsa_utils_shim_lacks_the_symbol(self):
        """CP 分支 import 的符号在 `nsa.utils`（deprecated shim）/ `dsa.utils` 里都没有。

        `nsa/utils.py` 现在是 `from dsa.utils import *` 的兼容 shim，模块本身还在，
        但 `is_nsa_prefill_cp_in_seq_split` 已经从 fork 里删掉了 —— 所以插件 1223 行
        的 `from ...nsa.utils import is_nsa_prefill_cp_in_seq_split` 仍然会 ImportError。
        """
        import sglang.srt.layers.attention.nsa.utils as nsa_utils
        from sglang.srt.layers.attention.dsa import utils as dsa_utils

        for mod in (nsa_utils, dsa_utils):
            self.assertFalse(
                hasattr(mod, "is_nsa_prefill_cp_in_seq_split"),
                f"{mod.__name__} 又有 is_nsa_prefill_cp_in_seq_split 了 —— 请复核插件 1223 行",
            )

    def test_plugin_no_longer_references_old_name(self):
        """插件不再引用旧名：hook target / def / 调用点全部删除。

        原 CP-v1 in-seq-split 分支改成显式守卫：只有 fork 真的重新提供
        `ForwardBatch.nsa_cp_metadata` 与 `is_dsa_prefill_cp_in_seq_split` 时才
        抛 `NotImplementedError`；CP 关闭时直接落到 `_get_topk_ragged`。
        """
        refs = self._old_name_refs()
        self.assertFalse(
            refs,
            f"{self.plugin_path} 仍引用 fork 已删除的 _get_topk_ragged_with_cp：{refs}",
        )


# ---------------------------------------------------------------------------
# 10) KV cache dtype 护栏（避免回落到 FP8 布局触发 `_e4m3_to_uint8` 的 .cpu()）
# ---------------------------------------------------------------------------

def _find_run_script():
    """定位 `run_eb5_kunlun.sh`。

    同一份文件会被 staging 到 `code/sglang/test_internal/internal/models/eb5/layer/`，
    那里没有 `01_eb_adp/tool/`，所以按"向上找 + 两种布局"解析，找不到就 skip
    而不是 collection error。也可用 `EB5_RUN_SCRIPT` 直接指定。
    """
    env = os.environ.get("EB5_RUN_SCRIPT")
    if env and pathlib.Path(env).is_file():
        return pathlib.Path(env)
    for parent in _HERE.parents:
        for rel in (
            ("01_eb_adp", "tool", "run_eb5_kunlun.sh"),
            ("sglang-kunlun", "01_eb_adp", "tool", "run_eb5_kunlun.sh"),
        ):
            cand = parent.joinpath(*rel)
            if cand.is_file():
                return cand
    return None


_RUN_SCRIPT = _find_run_script()


class TestKvCacheDtypeGuard(unittest.TestCase):
    """`run_eb5_kunlun.sh` 必须默认 bf16 KV cache。

    `--kv-cache-dtype auto` 会落到 FP8(uint8) 布局，SWA 的 K 写入走
    `sglang_kunlun/kernels/dsv4_torch_shim.py::_e4m3_to_uint8` 里的 `.cpu()`
    → 4 rank 首次前向永久停在该 D2H（doc/3 §11.7-1）。
    """

    @classmethod
    def setUpClass(cls):
        if _RUN_SCRIPT is None:
            raise unittest.SkipTest(
                "找不到 run_eb5_kunlun.sh（staging 布局下请设 EB5_RUN_SCRIPT）"
            )
        cls.src = _RUN_SCRIPT.read_text(encoding="utf-8")

    def test_default_is_bfloat16(self):
        self.assertRegex(
            self.src,
            r"KV_CACHE_DTYPE=\$\{KV_CACHE_DTYPE:-bfloat16\}",
            "KV_CACHE_DTYPE 的默认值不再是 bfloat16 —— FP8 布局会触发 "
            "_e4m3_to_uint8 的 .cpu()",
        )

    def test_default_is_actually_forwarded(self):
        self.assertIn(
            '--kv-cache-dtype "$KV_CACHE_DTYPE"',
            self.src,
            "脚本没有把 KV_CACHE_DTYPE 传给 launcher（默认值形同虚设）",
        )

    def test_no_hardcoded_auto(self):
        self.assertNotRegex(
            self.src,
            r"--kv-cache-dtype\s+[\"']?auto",
            "脚本里出现硬编码 --kv-cache-dtype auto（会回落到 FP8 布局）",
        )

    def test_hazard_still_exists_in_shim(self):
        """护栏的必要性：`_e4m3_to_uint8` 里仍有 `.cpu()`。

        若哪天去掉这个 D2H，可以放宽 `test_default_is_bfloat16`（先确认 FP8 布局
        在昆仑上真的能跑）。
        """
        shim = _plugin_file("kernels/dsv4_torch_shim.py").read_text(encoding="utf-8")
        fn_src = shim.split("def _e4m3_to_uint8", 1)[1].split("\ndef ", 1)[0]
        self.assertIn(
            ".cpu()",
            fn_src,
            "_e4m3_to_uint8 不再有 .cpu() —— 如果 FP8 布局已可用，请复核 KV_CACHE_DTYPE 护栏",
        )


if __name__ == "__main__":
    unittest.main()
