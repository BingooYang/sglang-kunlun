"""EB5 `nsa_indexer` 插件专项单测（**纯 CPU**，不 import `_eb5_kernel_test_common`）。

为什么要单独一套
----------------
`nsa_indexer.py` 是 EB5 在昆仑上唯一一条 indexer 通路（`forward_cuda` → 默认
`EB5_INDEXER_TORCH_REF=1` → `_indexer_torch_ref`），但它真正的失败点在**和 vendor
算子 / fork API 的接触面**上，而不是数值上：

=================================  ==========================================  ====================================
2026-10-20 实际报错                  根因                                        本文件用例
=================================  ==========================================  ====================================
``dtype of kv_cache (Byte) must be   池里的 index-k buffer 是 uint8，           ``TestGatherPagedScaleDtype``
one of [Char]`` (nsa_indexer:619)   ``gather_paged_scale`` 只收 int8
``Check torch_lod_cpu_opt->dtype()   ``forward_batch.seq_lens_cpu`` 是 int64    ``TestInt32LodInputs``
== int32 ... left operand=long``     （fork 在 schedule_batch 里显式 int64）
``AttributeError: 'ForwardBatch'     fork 删了该属性                            ``TestIndexerKvPool``
has no attribute 'token_to_kv_pool'``  → 走 `_indexer_kv_pool()` + 解包子池
``AttributeError: ... get_seqlens_   fork 没有 *_cpu 访问器                      ``TestSeqlensCpuNoD2H``
int32_cpu``（decode 每层）            → 优先 `seq_lens_cpu`，零 D2H
``NotImplementedError: fast_topk_   sgl_kernel 是 stub                          ``TestSkipLogitsBranch``
transform_fused``                    → `_topk_all_visible`
fork 删除的 CP-v1 符号                插件不能再引用                              ``TestNoLegacyCpSymbols``
=================================  ==========================================  ====================================

几条用例用**假算子模拟 vendor 的真实校验**（dtype 不对就抛和线上同款的
RuntimeError / ValueError），这样把实现改回错误写法时测试会以线上那条错误失败，
而不是只比对一个不痛不痒的 dtype 断言。

设备无关：`torch.ops.xspeedgate_ops` 与 `kunlun_ops` 都被替换成假命名空间，
所有张量都在 CPU 上，`python -m pytest test_eb5_nsa_indexer.py -q` 可直接跑。
"""

from __future__ import annotations

import ast
import builtins
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import sglang_kunlun.hooks.layers.attention.nsa.nsa_indexer as nsa
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

_HERE = pathlib.Path(__file__).resolve().parent
# 静态扫描要扫"**正在跑的那份**插件"，所以从 import 进来的模块反推包目录：
# 本文件同时存在于 `01_eb_adp/test/` 与 staging 的
# `test_internal/internal/models/eb5/layer/`，用相对路径会指错地方。
_INDEXER_PATH = pathlib.Path(nsa.__file__).resolve()
_PLUGIN_ROOT = _INDEXER_PATH.parents[4]
assert (_PLUGIN_ROOT / "hooks").is_dir(), f"插件包目录推断失败: {_PLUGIN_ROOT}"

_PAGE_SIZE = 64
_HEAD_DIM = 128


# ---------------------------------------------------------------------------
# vendor 侧假算子：照抄线上的校验，dtype 不对就抛和线上一样的错
# ---------------------------------------------------------------------------

_DTYPE_ALIAS = {
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float32: "Float",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
}


def _alias(dtype) -> str:
    return _DTYPE_ALIAS.get(dtype, str(dtype))


def _require_int32_lod(what: str, tensor) -> None:
    """模拟 XCCL ``torch_lod_cpu_opt`` 的 int32 断言（2026-10-20 22:28 线上原话）。"""
    if tensor is None:
        return
    if tensor.dtype is not torch.int32:
        raise ValueError(
            "Check torch_lod_cpu_opt->dtype() == CPPTypeToTorchType<int32_t>() "
            f"failed, left operand={'long int' if tensor.dtype is torch.int64 else _alias(tensor.dtype)} "
            f"({what})"
        )


class _VendorRecord:
    """记录假 vendor 算子收到的参数，供用例断言。"""

    def __init__(self):
        self.calls = {}

    def put(self, name, kwargs):
        self.calls[name] = kwargs
        return kwargs

    def __getitem__(self, name):
        assert name in self.calls, f"没有调用 {name}（实际调用：{sorted(self.calls)}）"
        return self.calls[name]

    def __contains__(self, name):
        return name in self.calls


def _make_xspeedgate_ops(rec: _VendorRecord):
    def gather_paged_scale(
        *,
        kv_cache,
        block_tables,
        page_size,
        head_dim,
        max_seq_len,
        scale_offset,
        context_lens,
        q_is_int8,
    ):
        rec.put(
            "gather_paged_scale",
            dict(
                kv_cache=kv_cache,
                block_tables=block_tables,
                page_size=page_size,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                scale_offset=scale_offset,
                context_lens=context_lens,
                q_is_int8=q_is_int8,
            ),
        )
        # 线上 vendor 只接受 Char(int8)：
        #   RuntimeError: dtype of kv_cache (Byte) must be one of [Char]
        if kv_cache.dtype is not torch.int8:
            raise RuntimeError(
                f"dtype of kv_cache ({_alias(kv_cache.dtype)}) must be one of [Char]"
            )
        _require_int32_lod("gather_paged_scale.context_lens", context_lens)
        return torch.zeros(kv_cache.shape[0], page_size, dtype=torch.float32)

    def mask_for_I8_mqa_logits(*, seq_len_kv, cu_seqlen_ks, cu_seqlen_ke, logits):
        rec.put(
            "mask_for_I8_mqa_logits",
            dict(
                seq_len_kv=seq_len_kv,
                cu_seqlen_ks=cu_seqlen_ks,
                cu_seqlen_ke=cu_seqlen_ke,
                logits=logits,
            ),
        )
        _require_int32_lod("mask.cu_seqlen_ks", cu_seqlen_ks)
        _require_int32_lod("mask.cu_seqlen_ke", cu_seqlen_ke)

    def topk_transform(
        *, score, lengths, src_page_table, dst_page_table, topk, cu_seqlens_q
    ):
        rec.put(
            "topk_transform",
            dict(
                score=score,
                lengths=lengths,
                src_page_table=src_page_table,
                dst_page_table=dst_page_table,
                topk=topk,
                cu_seqlens_q=cu_seqlens_q,
            ),
        )
        _require_int32_lod("topk_transform.lengths", lengths)
        _require_int32_lod("topk_transform.cu_seqlens_q", cu_seqlens_q)
        _require_int32_lod("topk_transform.src_page_table", src_page_table)
        dst_page_table.zero_()
        return dst_page_table

    def hadamard_transform(x, scale=None):
        rec.put("hadamard_transform", dict(scale=scale))
        return x if scale is None else x * scale

    return SimpleNamespace(
        gather_paged_scale=gather_paged_scale,
        mask_for_I8_mqa_logits=mask_for_I8_mqa_logits,
        topk_transform=topk_transform,
        hadamard_transform=hadamard_transform,
    )


def _make_kunlun_ops(rec: _VendorRecord):
    mod = types.ModuleType("kunlun_ops")

    def quant2d(src, dst, scale, force_sdnn=False):
        rec.put("quant2d", dict(src=src, dst=dst, scale=scale, force_sdnn=force_sdnn))
        assert dst.dtype is torch.int8, f"quant2d 需要 int8 输出，收到 {dst.dtype}"
        assert scale.dtype is torch.float32
        dst.fill_(1)
        scale.fill_(0.5)

    def I8_paged_mqa_logits(
        *,
        q,
        fused_kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        clean_logits,
        out,
        use_xfa_boost,
    ):
        rec.put(
            "I8_paged_mqa_logits",
            dict(
                q=q,
                fused_kv_cache=fused_kv_cache,
                weights=weights,
                context_lens=context_lens,
                block_table=block_table,
                max_context_len=max_context_len,
                clean_logits=clean_logits,
                out=out,
                use_xfa_boost=use_xfa_boost,
            ),
        )
        assert len(context_lens) == 2, "context_lens 必须是 (cpu_lod, xpu_lod)"
        _require_int32_lod("I8_paged_mqa_logits.context_lens_cpu", context_lens[0])
        _require_int32_lod("I8_paged_mqa_logits.context_lens_xpu", context_lens[1])
        _require_int32_lod("I8_paged_mqa_logits.block_table", block_table)
        assert q.dtype is torch.int8
        assert out.shape == (q.shape[0], q.shape[1], max_context_len), (
            f"out {tuple(out.shape)} 与 q {tuple(q.shape)} / max_context_len "
            f"{max_context_len} 不匹配"
        )
        out.zero_()

    def I8_mqa_logits(
        *,
        q,
        fused_kv_cache,
        weights,
        context_q_lens,
        context_k_lens,
        logits,
        clean_logits,
        use_xfa_boost,
    ):
        rec.put(
            "I8_mqa_logits",
            dict(
                q=q,
                fused_kv_cache=fused_kv_cache,
                weights=weights,
                context_q_lens=context_q_lens,
                context_k_lens=context_k_lens,
                logits=logits,
                clean_logits=clean_logits,
                use_xfa_boost=use_xfa_boost,
            ),
        )
        for name, pair in (
            ("context_q_lens", context_q_lens),
            ("context_k_lens", context_k_lens),
        ):
            assert len(pair) == 2, f"{name} 必须是 (cpu, xpu)"
            _require_int32_lod(f"I8_mqa_logits.{name}_cpu", pair[0])
            _require_int32_lod(f"I8_mqa_logits.{name}_xpu", pair[1])
        logits.fill_(0.25)

    mod.quant2d = quant2d
    mod.I8_paged_mqa_logits = I8_paged_mqa_logits
    mod.I8_mqa_logits = I8_mqa_logits
    return mod


@pytest.fixture()
def vendor(monkeypatch):
    """把 `torch.ops.xspeedgate_ops` 与 `kunlun_ops` 换成带真实 dtype 校验的假算子。"""
    rec = _VendorRecord()
    monkeypatch.setattr(torch.ops, "xspeedgate_ops", _make_xspeedgate_ops(rec))
    monkeypatch.setitem(sys.modules, "kunlun_ops", _make_kunlun_ops(rec))
    return rec


# ---------------------------------------------------------------------------
# 被测实现的假依赖
# ---------------------------------------------------------------------------


class FakeIndexPool:
    """indexer 子池（page_size=64，index-k buffer 是 **uint8**，和线上一致）。"""

    def __init__(self, *, page_size=_PAGE_SIZE, head_dim=_HEAD_DIM, num_pages=16):
        self.page_size = page_size
        self.index_head_dim = head_dim
        # 线上 layout: [:, :page*head_dim] 是 k（fp8 但 dtype=uint8），
        # 其后 page*4 字节是每个 page 的 scale。
        self.buffer = torch.zeros(
            num_pages, page_size * head_dim + page_size * 4, dtype=torch.uint8
        )
        self.layer_ids = []
        self.set_calls = []

    def get_index_k_with_scale_buffer(self, layer_id):
        self.layer_ids.append(layer_id)
        return self.buffer

    def get_index_k_continuous(self, layer_id, seq_len, page_indices):
        self.layer_ids.append(layer_id)
        return torch.zeros(int(seq_len), self.index_head_dim, dtype=torch.uint8)

    def get_index_k_scale_continuous(self, layer_id, seq_len, page_indices):
        self.layer_ids.append(layer_id)
        return torch.zeros(int(seq_len), 4, dtype=torch.uint8)

    def set_index_k_scale_buffer(self, *, layer_id, loc, index_k, index_k_scale):
        self.set_calls.append(
            dict(
                layer_id=layer_id,
                loc=loc,
                index_k=index_k,
                index_k_scale=index_k_scale,
            )
        )


class FakeIndexerMetadata:
    """`DSAIndexerMetadata` 的最小替身（只实现插件真正调用的访问器）。

    注意 `get_cu_seqlens_q`：**fork 的 `DSAIndexerMetadata` 上并没有这个方法**
    （只有 `attn_metadata.cu_seqlens_q` 字段），插件却在 `_get_topk_ragged` 里直接
    调用它。为了让本文件聚焦在 dtype/dispatch 上，这里把该方法补上；
    该缺陷本身由 ``test_fork_indexer_metadata_has_cu_seqlens_q`` 记录。
    """

    def __init__(
        self,
        *,
        seqlens_int32=None,
        seqlens_expanded=None,
        page_table_64=None,
        page_table_1=None,
        indexer_seq_len_cpu=None,
        extend_lens=(1,),
        cu_seqlens_q=None,
        kv_range=None,
        token_to_batch_idx=None,
        topk_indices_offset=None,
        topk_fn=None,
        with_cpu_accessors=False,
    ):
        self._seqlens_int32 = seqlens_int32
        self._seqlens_expanded = seqlens_expanded
        self._page_table_64 = page_table_64
        self._page_table_1 = page_table_1
        self._indexer_seq_len_cpu = indexer_seq_len_cpu
        self._extend_lens = list(extend_lens)
        self._cu_seqlens_q = cu_seqlens_q
        self._kv_range = kv_range
        self._token_to_batch_idx = token_to_batch_idx
        self.topk_calls = []
        self._topk_fn = topk_fn
        self.attn_metadata = SimpleNamespace(
            cu_seqlens_q=cu_seqlens_q, topk_indices_offset=topk_indices_offset
        )
        if with_cpu_accessors:
            self.get_seqlens_int32_cpu = lambda: self._seqlens_int32.clone()
            self.get_seqlens_expanded_cpu = lambda: self._seqlens_expanded.clone()

    def get_seqlens_int32(self):
        return self._seqlens_int32

    def get_seqlens_expanded(self):
        return self._seqlens_expanded

    def get_page_table_64(self):
        return self._page_table_64

    def get_page_table_1(self):
        return self._page_table_1

    def get_indexer_seq_len_cpu(self):
        return self._indexer_seq_len_cpu

    def get_dsa_extend_len_cpu(self):
        return self._extend_lens

    def get_cu_seqlens_q(self):
        return self._cu_seqlens_q

    def get_indexer_kvcache_range(self):
        return self._kv_range

    def get_token_to_batch_idx(self):
        return self._token_to_batch_idx

    def topk_transform(self, logits, topk, **kwargs):
        self.topk_calls.append(dict(logits=logits, topk=topk, **kwargs))
        if self._topk_fn is not None:
            return self._topk_fn(logits, topk, **kwargs)
        return torch.zeros(logits.shape[0], topk, dtype=torch.int32)


class _NoTopkTransform(FakeIndexerMetadata):
    """任何 `topk_transform` 都视为"踩到了 stub"，直接 fail。"""

    def topk_transform(self, logits, topk, **kwargs):  # noqa: D102
        raise AssertionError(
            "skip_logits 分支不应触碰 metadata.topk_transform "
            "(sgl_kernel.fast_topk_transform_fused 在昆仑上未实现)"
        )


def _forward_batch(mode, *, seq_lens_cpu, seq_lens=None, out_cache_loc=None, **extra):
    """真实的 `ForwardBatch`（CPU 张量），刻意不用 SimpleNamespace。"""
    bs = len(seq_lens_cpu)
    seq_lens = seq_lens if seq_lens is not None else seq_lens_cpu
    fb = ForwardBatch(
        forward_mode=mode,
        batch_size=bs,
        input_ids=torch.zeros(bs, dtype=torch.int64),
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
        seq_lens=seq_lens,
        out_cache_loc=(
            out_cache_loc
            if out_cache_loc is not None
            else torch.arange(bs, dtype=torch.int64)
        ),
        seq_lens_sum=int(seq_lens.sum()),
        seq_lens_cpu=seq_lens_cpu,
    )
    for key, value in extra.items():
        setattr(fb, key, value)
    return fb


def _page_table_1(bs, width=8 * _PAGE_SIZE, num_pages=16):
    """page_size=1 的页表：宽度 = max_seq_len，值是 slot 号。"""
    row = torch.arange(width, dtype=torch.int32).remainder(num_pages)
    return row.unsqueeze(0).repeat(bs, 1).contiguous()


def _paged_metadata(*, seq_lens, extend_lens=(1, 1), page_table=None, **kw):
    bs = len(seq_lens)
    pts = (
        page_table
        if page_table is not None
        else torch.zeros(bs, 8, dtype=torch.int32)
    )
    return FakeIndexerMetadata(
        seqlens_int32=torch.tensor(seq_lens, dtype=torch.int32),
        seqlens_expanded=torch.tensor(
            [x for x in seq_lens for _ in range(1)], dtype=torch.int32
        ),
        page_table_64=pts,
        page_table_1=_page_table_1(bs, pts.shape[1] * _PAGE_SIZE),
        indexer_seq_len_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        extend_lens=extend_lens,
        cu_seqlens_q=torch.tensor(
            [0]
            + list(
                torch.cumsum(torch.tensor(seq_lens, dtype=torch.int32), 0).tolist()
            ),
            dtype=torch.int32,
        ).unsqueeze(0),
        kv_range=(
            torch.zeros(sum(seq_lens), dtype=torch.int32),
            torch.ones(sum(seq_lens), dtype=torch.int32),
        ),
        token_to_batch_idx=torch.repeat_interleave(
            torch.arange(bs, dtype=torch.int32),
            torch.tensor(extend_lens, dtype=torch.int32),
        ),
        **kw,
    )


# ---------------------------------------------------------------------------
# 1) gather_paged_scale 的 kv_cache 必须是 int8 视图（不能是 uint8，也不能拷贝）
# ---------------------------------------------------------------------------


class TestGatherPagedScaleDtype:
    """2026-10-20 22:24：`RuntimeError: dtype of kv_cache (Byte) must be one of [Char]`。"""

    def _run(self, monkeypatch, vendor, *, seq_lens_cpu_dtype=torch.int32):
        pool = FakeIndexPool()
        monkeypatch.setattr(nsa, "_indexer_kv_pool", lambda: pool)
        md = _paged_metadata(seq_lens=[128, 300])
        fb = _forward_batch(
            ForwardMode.DECODE,
            seq_lens_cpu=torch.tensor([128, 300], dtype=seq_lens_cpu_dtype),
            seq_lens=torch.tensor([128, 300], dtype=torch.int64),
        )
        self_obj = SimpleNamespace(index_topk=64, block_size=_HEAD_DIM)
        q = torch.ones(2, 8, _HEAD_DIM, dtype=torch.int8)
        w = torch.full((2, 8), 0.5, dtype=torch.float32)
        out = nsa._get_topk_paged(self_obj, fb, 3, q, w, md)
        return pool, md, out

    def test_kv_cache_is_zero_copy_int8_view(self, monkeypatch, vendor):
        pool, _, _ = self._run(monkeypatch, vendor)
        got = vendor["gather_paged_scale"]["kv_cache"]
        assert got.dtype is torch.int8, (
            "gather_paged_scale 收到的是 "
            f"{got.dtype}（vendor 只接受 int8，线上报 "
            "'dtype of kv_cache (Byte) must be one of [Char]'）"
        )
        assert got.data_ptr() == pool.buffer.data_ptr(), (
            "int8 必须是 view（零拷贝）而不是 .to(int8)/clone："
            "data_ptr 变了说明多做了一次拷贝"
        )
        assert (
            got.untyped_storage().data_ptr()
            == pool.buffer.untyped_storage().data_ptr()
        )
        assert got.shape == pool.buffer.shape

    def test_paged_geometry_and_lens(self, monkeypatch, vendor):
        # decode 的 top-k 有两条合法实现：`metadata.topk_transform`（sgl-kernel，
        # 昆仑是 stub）与插件自带的 `_topk_paged_torch`。这里只要求"确实走了其中
        # 一条并且形状/dtype 对"，避免和正在迁移的实现互相打架。
        torch_topk = []
        real = getattr(nsa, "_topk_paged_torch", None)
        if real is not None:
            monkeypatch.setattr(
                nsa,
                "_topk_paged_torch",
                lambda *a, **k: torch_topk.append(a) or real(*a, **k),
            )
        _, md, out = self._run(monkeypatch, vendor)
        call = vendor["gather_paged_scale"]
        assert call["page_size"] == _PAGE_SIZE
        assert call["head_dim"] == _HEAD_DIM
        assert call["scale_offset"] == _PAGE_SIZE * _HEAD_DIM, (
            "scale 紧跟在 k 后面：scale_offset 必须是 page_size*head_dim"
        )
        assert call["max_seq_len"] == 8 * _PAGE_SIZE
        assert call["q_is_int8"] is True
        assert call["block_tables"].dtype is torch.int32
        # k 切片后再喂给 logits 算子
        pg = vendor["I8_paged_mqa_logits"]
        assert pg["q"].dtype is torch.int8
        assert pg["fused_kv_cache"][0].shape == (16, _PAGE_SIZE, 1, _HEAD_DIM)
        assert out.dtype is torch.int32 and out.shape == (2, 64)
        assert len(md.topk_calls) + len(torch_topk) >= 1, (
            "decode 既没走 metadata.topk_transform 也没走 _topk_paged_torch："
            "top-k 根本没做"
        )


# ---------------------------------------------------------------------------
# 1b) `_topk_paged_torch`：替掉 sgl_kernel stub 的 paged top-k + slot 映射
# ---------------------------------------------------------------------------


def _ref_topk_paged(logits, lengths, page_table, topk):
    rows, width = logits.shape
    k = min(topk, width)
    out = torch.full((rows, topk), -1, dtype=torch.int32)
    for r in range(rows):
        row = page_table[r % page_table.shape[0]]
        n = min(int(lengths[r]), width)
        order = torch.argsort(logits[r, :n], descending=True, stable=True)[:k]
        out[r, : order.numel()] = row[order].to(torch.int32)
    return out


class TestTopkPagedTorch:
    """paged top-k 必须"按可见长度 mask + 经 page_table_1 映射 + 右侧补 -1"。"""

    def test_matches_reference(self):
        torch.manual_seed(0)
        rows, width, topk = 6, 16, 4
        logits = torch.randn(rows, width, dtype=torch.float32)
        lengths = torch.tensor([3, 16, 1, 9, 16, 2], dtype=torch.int32)
        # 页表行数与 logits 行数一致（decode / per-token 展开后的正常形态），
        # 这样"位置 -> slot"的映射是唯一的，可以逐元素对照。
        page_table = torch.arange(rows * width, dtype=torch.int32).view(rows, width) + 100
        md = FakeIndexerMetadata(page_table_1=page_table)
        got = nsa._topk_paged_torch(md, logits, topk, lengths)
        want = _ref_topk_paged(logits, lengths, page_table, topk)
        assert torch.equal(got, want), f"got={got.tolist()}\nwant={want.tolist()}"
        assert got.dtype is torch.int32 and got.shape == (rows, topk)

    def test_topk_wider_than_kv_pads_with_minus_one(self):
        logits = torch.randn(2, 3, dtype=torch.float32)
        lengths = torch.tensor([3, 2], dtype=torch.int32)
        md = FakeIndexerMetadata(page_table_1=torch.arange(6, dtype=torch.int32).view(2, 3))
        got = nsa._topk_paged_torch(md, logits, 8, lengths)
        assert got.shape == (2, 8)
        assert (got[:, 3:] == -1).all(), "width < topk 的部分必须补 -1"
        # 每个 request 只有前 `length` 个位置可见
        assert got[1, 2:].eq(-1).all()


# ---------------------------------------------------------------------------
# 2) CPU 侧 lod / 长度类输入必须是 int32
# ---------------------------------------------------------------------------


class TestInt32LodInputs:
    """2026-10-20 22:28：`torch_lod_cpu_opt->dtype() == ...<int32_t>() failed, left=long`。

    fork 在 `schedule_batch.py` 里把 `seq_lens_cpu` 显式建成 **int64**
    （``torch.tensor(seq_lens, dtype=torch.int64)``），而 `_seqlens_cpu` 早期直接把它
    透传给 vendor 算子 → 线上 int64 打 int32 断言。
    """

    def _run_decode(self, monkeypatch, vendor, seq_lens_cpu_dtype):
        pool = FakeIndexPool()
        monkeypatch.setattr(nsa, "_indexer_kv_pool", lambda: pool)
        md = _paged_metadata(seq_lens=[128, 300])
        fb = _forward_batch(
            ForwardMode.DECODE,
            seq_lens_cpu=torch.tensor([128, 300], dtype=seq_lens_cpu_dtype),
            seq_lens=torch.tensor([128, 300], dtype=torch.int64),
        )
        self_obj = SimpleNamespace(index_topk=64, block_size=_HEAD_DIM)
        q = torch.ones(2, 8, _HEAD_DIM, dtype=torch.int8)
        w = torch.full((2, 8), 0.5, dtype=torch.float32)
        return nsa._get_topk_paged(self_obj, fb, 0, q, w, md)

    @pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.int16])
    def test_decode_cpu_lod_is_int32(self, monkeypatch, vendor, dtype):
        # `seq_lens_cpu` 无论来自 fork 的 int64 镜像还是别的整数宽度，
        # 传给 vendor 之前都必须归一成 int32；否则假算子里会抛线上那条 ValueError。
        self._run_decode(monkeypatch, vendor, dtype)
        assert vendor["I8_paged_mqa_logits"]["context_lens"][0].dtype is torch.int32
        assert vendor["gather_paged_scale"]["context_lens"].dtype is torch.int32

    def test_expanded_lod_is_int32(self, monkeypatch, vendor):
        """target_verify / draft_extend 走 expanded 分支，两边的 lod 也必须 int32。"""
        pool = FakeIndexPool()
        monkeypatch.setattr(nsa, "_indexer_kv_pool", lambda: pool)
        # target_verify：每请求 2 个 draft token → per-token 长度 4 条。
        expanded = torch.tensor([130, 130, 302, 302], dtype=torch.int32)
        md = FakeIndexerMetadata(
            seqlens_int32=torch.tensor([128, 300], dtype=torch.int32),
            # device 侧按 fork 现状给 int32（`dsa_seqlens_expanded` 由
            # `(seq_lens + draft).to(torch.int32)` 产出）；CPU 侧回来必须是 int32。
            seqlens_expanded=expanded,
            page_table_64=torch.zeros(2, 8, dtype=torch.int32),
            page_table_1=_page_table_1(2, 8 * _PAGE_SIZE),
            indexer_seq_len_cpu=torch.tensor([128, 300], dtype=torch.int64),
            extend_lens=[2, 2],
            cu_seqlens_q=torch.tensor([[0, 2, 4]], dtype=torch.int32),
            token_to_batch_idx=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        )
        md.get_seqlens_expanded = lambda: expanded
        fb = _forward_batch(
            ForwardMode.TARGET_VERIFY,
            seq_lens_cpu=torch.tensor([128, 300], dtype=torch.int64),
            seq_lens=torch.tensor([130, 302], dtype=torch.int64),
        )
        self_obj = SimpleNamespace(index_topk=64, block_size=_HEAD_DIM)
        q = torch.ones(4, 8, _HEAD_DIM, dtype=torch.int8)
        w = torch.full((4, 8), 0.5, dtype=torch.float32)
        out = nsa._get_topk_paged(self_obj, fb, 0, q, w, md)
        assert out.shape == (4, 64)
        assert vendor["I8_paged_mqa_logits"]["context_lens"][0].dtype is torch.int32

    def test_ragged_lengths_cu_seqlens_and_topk_dtypes(self, monkeypatch, vendor):
        """ragged（非 chunk）路径：所有 lod + topk 输入输出都必须是 int32。"""
        pool = FakeIndexPool()
        monkeypatch.setattr(nsa, "_indexer_kv_pool", lambda: pool)
        md = self._ragged_metadata(seq_lens=[40, 70], extend_lens=[3, 2])
        fb = _forward_batch(
            ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor([40, 70], dtype=torch.int64),
            seq_lens=torch.tensor([40, 70], dtype=torch.int64),
            extend_seq_lens_cpu=[3, 2],
        )
        self_obj = SimpleNamespace(
            index_topk=64,
            block_size=_HEAD_DIM,
            _should_chunk_mqa_logits=lambda q, k, d: (False, 0),
        )
        q = torch.ones(5, 8, _HEAD_DIM, dtype=torch.int8)
        w = torch.full((5, 8, 1), 0.5, dtype=torch.float32)
        out = nsa._get_topk_ragged(self_obj, fb, 0, q, w, md)

        assert out.dtype is torch.int32 and out.shape == (5, 64)
        mqa = vendor["I8_mqa_logits"]
        assert mqa["context_q_lens"][0].dtype is torch.int32
        assert mqa["context_q_lens"][1].dtype is torch.int32
        assert mqa["context_k_lens"][0].dtype is torch.int32
        assert mqa["context_k_lens"][1].dtype is torch.int32
        assert mqa["logits"].dtype is torch.float32
        tk = vendor["topk_transform"]
        assert tk["lengths"].dtype is torch.int32
        assert tk["cu_seqlens_q"].dtype is torch.int32
        assert tk["src_page_table"].dtype is torch.int32
        assert tk["dst_page_table"].dtype is torch.int32
        assert tk["topk"] == 64

    def test_make_cu_seqlens_from_lens_int32(self):
        """`_make_cu_seqlens_from_lens`：int64 输入也要在 host 侧 cu 成 int32。"""
        cu_cpu, cu_xpu = nsa._make_cu_seqlens_from_lens(
            torch.tensor([40, 70], dtype=torch.int64), torch.device("cpu")
        )
        assert cu_cpu.dtype is torch.int32 and cu_xpu.dtype is torch.int32
        assert cu_cpu.tolist() == [0, 40, 110]

    @staticmethod
    def _ragged_metadata(*, seq_lens, extend_lens):
        tokens = sum(extend_lens)
        bs = len(seq_lens)
        return FakeIndexerMetadata(
            seqlens_int32=torch.tensor(seq_lens, dtype=torch.int32),
            seqlens_expanded=torch.repeat_interleave(
                torch.tensor(extend_lens, dtype=torch.int32),
                torch.tensor(extend_lens, dtype=torch.int32),
            ),
            page_table_64=torch.arange(bs * 2, dtype=torch.int32).view(bs, 2),
            page_table_1=_page_table_1(tokens, 8 * _PAGE_SIZE),
            indexer_seq_len_cpu=torch.tensor(seq_lens, dtype=torch.int64),
            extend_lens=extend_lens,
            cu_seqlens_q=torch.tensor(
                [[0] + list(torch.cumsum(torch.tensor(extend_lens), 0).tolist())],
                dtype=torch.int32,
            ),
            kv_range=(
                torch.zeros(tokens, dtype=torch.int32),
                torch.tensor(
                    [x for x, n in zip(seq_lens, extend_lens) for _ in range(n)],
                    dtype=torch.int32,
                ),
            ),
            token_to_batch_idx=torch.repeat_interleave(
                torch.arange(bs, dtype=torch.int32),
                torch.tensor(extend_lens, dtype=torch.int32),
            ),
        )


# ---------------------------------------------------------------------------
# 3) `_indexer_kv_pool()`：fork 删了 forward_batch.token_to_kv_pool
# ---------------------------------------------------------------------------


class TestIndexerKvPool:
    """必须走（可能被 patch 过的）`get_token_to_kv_pool()`，并解包 `full_kv_pool`。"""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        import sglang.srt.layers.attention.dsa.dsa_indexer as dsa_indexer

        self.dsa_indexer = dsa_indexer
        self.monkeypatch = monkeypatch

    def test_forward_batch_has_no_legacy_pool_attribute(self):
        fb = _forward_batch(
            ForwardMode.DECODE, seq_lens_cpu=torch.tensor([4], dtype=torch.int32)
        )
        assert not hasattr(fb, "token_to_kv_pool"), (
            "fork 又加回了 ForwardBatch.token_to_kv_pool —— 请复核插件取池路径"
        )
        assert not hasattr(SimpleNamespace(), "token_to_kv_pool")

    def test_unwraps_full_kv_pool(self):
        inner = FakeIndexPool()
        outer = SimpleNamespace(full_kv_pool=inner, page_size=256)
        self.monkeypatch.setattr(
            self.dsa_indexer, "get_token_to_kv_pool", lambda: outer, raising=False
        )
        assert nsa._indexer_kv_pool() is inner

    def test_dsv4_pool_without_full_kv_pool_is_returned_as_is(self):
        outer = FakeIndexPool()
        outer.page_size = 64
        self.monkeypatch.setattr(
            self.dsa_indexer, "get_token_to_kv_pool", lambda: outer, raising=False
        )
        assert nsa._indexer_kv_pool() is outer

    def test_full_kv_pool_none_falls_back(self):
        outer = SimpleNamespace(full_kv_pool=None)
        self.monkeypatch.setattr(
            self.dsa_indexer, "get_token_to_kv_pool", lambda: outer, raising=False
        )
        assert nsa._indexer_kv_pool() is outer

    def test_module_attribute_wins_over_imported_binding(self):
        """EB5 的 runtime patch 换的是**模块属性**，插件不能用 import 时那份绑定。"""
        inner = FakeIndexPool()
        outer = SimpleNamespace(full_kv_pool=inner)
        self.monkeypatch.setattr(
            self.dsa_indexer, "get_token_to_kv_pool", lambda: outer, raising=False
        )
        self.monkeypatch.setattr(
            nsa, "get_token_to_kv_pool", lambda: FakeIndexPool(), raising=True
        )
        assert nsa._indexer_kv_pool() is inner

    def test_indexer_reads_go_through_unwrapped_pool(self, monkeypatch, vendor):
        """行为：`_get_topk_paged` 读的是子池（page_size 64），不是外层 256 池。"""
        inner = FakeIndexPool()
        outer = SimpleNamespace(full_kv_pool=inner, page_size=256)
        monkeypatch.setattr(
            self.dsa_indexer, "get_token_to_kv_pool", lambda: outer, raising=False
        )
        md = _paged_metadata(seq_lens=[128, 300])
        fb = _forward_batch(
            ForwardMode.DECODE,
            seq_lens_cpu=torch.tensor([128, 300], dtype=torch.int32),
            seq_lens=torch.tensor([128, 300], dtype=torch.int64),
        )
        nsa._get_topk_paged(
            SimpleNamespace(index_topk=64, block_size=_HEAD_DIM),
            fb,
            5,
            torch.ones(2, 8, _HEAD_DIM, dtype=torch.int8),
            torch.full((2, 8), 0.5),
            md,
        )
        assert inner.layer_ids == [5]
        assert vendor["gather_paged_scale"]["page_size"] == 64


# ---------------------------------------------------------------------------
# 4) `_seqlens_cpu`：decode 热路径零 D2H，且 int32
# ---------------------------------------------------------------------------


class TestSeqlensCpuNoD2H:
    """2026-10-20 22:02：`DSAIndexerMetadata` 没有 `get_seqlens_int32_cpu`。"""

    @pytest.fixture()
    def cpu_calls(self, monkeypatch):
        calls = []
        orig = torch.Tensor.cpu

        def counting(self, *a, **k):
            calls.append(tuple(self.shape))
            return orig(self, *a, **k)

        monkeypatch.setattr(torch.Tensor, "cpu", counting)
        return calls

    def _fb(self, dtype=torch.int64):
        return _forward_batch(
            ForwardMode.DECODE, seq_lens_cpu=torch.tensor([128, 300], dtype=dtype)
        )

    def test_decode_uses_seq_lens_cpu_with_zero_d2h(self, cpu_calls):
        fb = self._fb(torch.int32)
        md = FakeIndexerMetadata()
        got = nsa._seqlens_cpu(md, fb, expanded=False)
        assert got is fb.seq_lens_cpu, (
            "decode 必须直接复用 ForwardBatch.seq_lens_cpu（不能 .cpu()/clone）"
        )
        assert cpu_calls == [], f"decode 路径发生了 {len(cpu_calls)} 次 .cpu()：{cpu_calls}"

    def test_int64_mirror_is_normalized_on_host(self, cpu_calls):
        fb = self._fb(torch.int64)
        got = nsa._seqlens_cpu(FakeIndexerMetadata(), fb, expanded=False)
        assert got.dtype is torch.int32, (
            "fork 的 seq_lens_cpu 是 int64，直接用会让 vendor 的 torch_lod_cpu_opt "
            "断言失败：left operand=long int"
        )
        assert got.tolist() == [128, 300]
        assert cpu_calls == [], "int32 归一化必须在 host 上做，不能引入 D2H"

    def test_metadata_cpu_accessor_takes_precedence(self, cpu_calls):
        fb = self._fb(torch.int64)
        md = FakeIndexerMetadata(
            seqlens_int32=torch.tensor([128, 300], dtype=torch.int32),
            with_cpu_accessors=True,
        )
        got = nsa._seqlens_cpu(md, fb, expanded=False)
        assert got.tolist() == [128, 300]
        assert got.dtype is torch.int32
        assert cpu_calls == []

    def test_expanded_falls_back_to_d2h_and_casts(self, cpu_calls):
        fb = self._fb(torch.int64)
        expanded = torch.tensor([130, 130], dtype=torch.int64)
        md = FakeIndexerMetadata(seqlens_expanded=expanded)
        got = nsa._seqlens_cpu(md, fb, expanded=True)
        assert got.dtype is torch.int32
        assert got.tolist() == [130, 130]
        assert got.data_ptr() != expanded.data_ptr()

    def test_extend_lens_prefers_nsa_then_dsa(self):
        md = FakeIndexerMetadata(extend_lens=[3, 2])
        md.get_nsa_extend_len_cpu = lambda: [7, 7]
        assert nsa._extend_lens_cpu(md) == [7, 7], "有旧名时优先用旧名"
        del md.get_nsa_extend_len_cpu
        assert nsa._extend_lens_cpu(md) == [3, 2], (
            "旧名不存在时回退 get_dsa_extend_len_cpu"
        )


# ---------------------------------------------------------------------------
# 5) skip_logits 分支：只走 `_topk_all_visible`，不碰 topk_transform
# ---------------------------------------------------------------------------


class TestSkipLogitsBranch:
    """`metadata.topk_transform` → `sgl_kernel.fast_topk_transform_fused` 是 stub。"""

    def _run(self, monkeypatch, vendor, *, seq_lens, extend_lens, topk=64):
        pool = FakeIndexPool()
        monkeypatch.setattr(nsa, "_indexer_kv_pool", lambda: pool)
        monkeypatch.setattr(nsa, "_fix_kernel_bindings_once", lambda: None)
        monkeypatch.setattr(
            nsa,
            "_k_from_weights",
            lambda self, x, pos: torch.zeros(
                x.shape[0], _HEAD_DIM, dtype=torch.bfloat16
            ),
        )
        md = _NoTopkTransform(
            page_table_1=torch.arange(len(seq_lens) * 8, dtype=torch.int32).view(
                len(seq_lens), 8
            ),
            seqlens_expanded=torch.repeat_interleave(
                torch.tensor(extend_lens, dtype=torch.int32),
                torch.tensor(extend_lens, dtype=torch.int32),
            ),
            extend_lens=extend_lens,
            page_table_64=torch.zeros(len(seq_lens), 2, dtype=torch.int32),
        )
        backend = SimpleNamespace(get_indexer_metadata=lambda lid, fb: md)
        monkeypatch.setattr(nsa, "get_attn_backend", lambda: backend)

        visible_calls = []
        real = nsa._topk_all_visible

        def spy(metadata, k, device):
            visible_calls.append(k)
            return real(metadata, k, device)

        monkeypatch.setattr(nsa, "_topk_all_visible", spy)

        fb = _forward_batch(
            ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
            extend_seq_lens_cpu=list(extend_lens),
            out_cache_loc=torch.arange(sum(extend_lens), dtype=torch.int64),
        )
        self_obj = SimpleNamespace(index_topk=topk, block_size=_HEAD_DIM)
        out = nsa._indexer_torch_ref(
            self_obj,
            torch.zeros(sum(extend_lens), _HEAD_DIM, dtype=torch.bfloat16),
            None,
            torch.zeros(sum(extend_lens), dtype=torch.int64),
            fb,
            0,
        )
        return out, md, pool, visible_calls

    def test_uses_topk_all_visible_only(self, monkeypatch, vendor):
        out, md, pool, visible_calls = self._run(
            monkeypatch, vendor, seq_lens=[5, 9], extend_lens=[3, 2]
        )
        assert visible_calls == [64], "skip_logits 分支必须调用 _topk_all_visible"
        assert md.topk_calls == [], "不得调用 metadata.topk_transform（昆仑是 stub）"
        assert out.dtype is torch.int32 and out.shape == (5, 64)

    def test_result_matches_page_table_expansion(self, monkeypatch, vendor):
        out, _, _, _ = self._run(
            monkeypatch, vendor, seq_lens=[5, 9], extend_lens=[3, 2]
        )
        pt = torch.arange(2 * 8, dtype=torch.int32).view(2, 8)
        visible = [3, 3, 3, 2, 2]
        want = torch.full((5, 64), -1, dtype=torch.int32)
        for t, n in enumerate(visible):
            want[t, :n] = pt[0 if t < 3 else 1, :n]
        assert torch.equal(out, want)

    def test_index_k_scale_written_to_indexer_pool(self, monkeypatch, vendor):
        _, _, pool, _ = self._run(
            monkeypatch, vendor, seq_lens=[5, 9], extend_lens=[3, 2]
        )
        assert len(pool.set_calls) == 1
        call = pool.set_calls[0]
        assert call["layer_id"] == 0
        assert call["index_k"].dtype is torch.int8
        assert call["index_k_scale"].dtype is torch.float32
        assert call["loc"].tolist() == [0, 1, 2, 3, 4]
        assert vendor["quant2d"]["force_sdnn"] is True

    def test_long_context_does_not_take_skip_branch(self, monkeypatch, vendor):
        """seq_lens.max() > index_topk 时不走 skip 分支（交给 paged/ragged）。"""
        monkeypatch.setattr(nsa, "_fix_kernel_bindings_once", lambda: None)
        md = _NoTopkTransform()
        monkeypatch.setattr(
            nsa,
            "get_attn_backend",
            lambda: SimpleNamespace(get_indexer_metadata=lambda lid, fb: md),
        )
        monkeypatch.setattr(
            nsa,
            "_topk_all_visible",
            lambda *a, **k: pytest.fail("不该走 skip 分支"),
        )
        fb = _forward_batch(
            ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor([512], dtype=torch.int32),
            seq_lens=torch.tensor([512], dtype=torch.int64),
        )
        self_obj = SimpleNamespace(index_topk=64, block_size=_HEAD_DIM)

        def boom(*a, **k):
            raise RuntimeError("reached-projections")

        monkeypatch.setattr(nsa, "_qk_from_weights", boom)
        with pytest.raises(RuntimeError, match="reached-projections"):
            nsa._indexer_torch_ref(
                self_obj,
                torch.zeros(1, _HEAD_DIM, dtype=torch.bfloat16),
                None,
                torch.zeros(1, dtype=torch.int64),
                fb,
                0,
            )


# ---------------------------------------------------------------------------
# 6) 静态扫描：fork 已删除的 CP-v1 符号
# ---------------------------------------------------------------------------

_FORBIDDEN_SYMBOLS = (
    "_get_topk_ragged_with_cp",
    "is_nsa_prefill_cp_in_seq_split",
)


def _parent_map(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _is_guarded_getattr(node, parents) -> bool:
    """``getattr(mod, "name", None)`` / ``hasattr(mod, "name")`` 里的字符串算加守卫。"""
    call = parents.get(node)
    if not isinstance(call, ast.Call):
        return False
    fn = call.func
    name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
    if name not in ("getattr", "hasattr"):
        return False
    return len(call.args) >= 2 and call.args[1] is node


def _symbol_violations(source: str, path: str = "<src>"):
    tree = ast.parse(source)
    parents = _parent_map(tree)
    # 本模块自己定义的 helper 不算"引用 fork 符号"：例如
    # `_nsa_backend_impl.is_nsa_prefill_cp_in_seq_split` 是插件自带的守卫包装，
    # 内部走 getattr(_dsa_utils, "is_dsa_prefill_cp_in_seq_split", None)。
    local_defs = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_SYMBOLS:
            hits.append(f"{path}:{node.lineno}: 属性 {node.attr}")
        elif (
            isinstance(node, ast.Name)
            and node.id in _FORBIDDEN_SYMBOLS
            and node.id not in local_defs
        ):
            hits.append(f"{path}:{node.lineno}: 名字 {node.id}")
        elif isinstance(node, ast.Constant) and node.value in _FORBIDDEN_SYMBOLS:
            if not _is_guarded_getattr(node, parents):
                hits.append(
                    f"{path}:{node.lineno}: 字符串 {node.value!r}（未加 getattr 守卫）"
                )
    return hits


class TestNoLegacyCpSymbols:
    """`Indexer._get_topk_ragged_with_cp` / `is_nsa_prefill_cp_in_seq_split` 已删。"""

    def test_scanner_is_not_vacuous(self):
        """自检：扫描器必须能抓到裸引用，同时放过 getattr 守卫。"""
        src = (
            "def is_nsa_prefill_cp_in_seq_split():\n"
            "    fn = getattr(utils, 'is_nsa_prefill_cp_in_seq_split', None)\n"
            "    return bool(fn()) if fn is not None else False\n"
            "mod.get_topk = Indexer._get_topk_ragged_with_cp\n"
            "x = fb.nsa_cp_metadata is not None and is_nsa_prefill_cp_in_seq_split()\n"
            "y = utils.is_nsa_prefill_cp_in_seq_split()\n"
        )
        hits = _symbol_violations(src)
        # 本模块 def 出来的同名 helper + getattr 守卫都应放过；只抓
        # `Indexer._get_topk_ragged_with_cp` 和 `utils.is_nsa_prefill_cp_in_seq_split()`。
        assert len(hits) == 2, hits
        assert "_get_topk_ragged_with_cp" in hits[0]
        assert "属性 is_nsa_prefill_cp_in_seq_split" in hits[1]

    def test_package_has_no_unguarded_references(self):
        assert _PLUGIN_ROOT.is_dir(), f"插件源码不存在: {_PLUGIN_ROOT}"
        hits = []
        for path in sorted(_PLUGIN_ROOT.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            hits.extend(
                _symbol_violations(path.read_text(encoding="utf-8"), str(path))
            )
        assert not hits, (
            "插件仍在引用 fork 已删除的 CP-v1 符号（只能作为 "
            "getattr(mod, '<name>', None) 出现）：\n  " + "\n  ".join(hits)
        )


# ---------------------------------------------------------------------------
# 7) `_get_topk_ragged` 的 chunk 路径不得引用未定义的 `k_interval_*`
# ---------------------------------------------------------------------------


def _collect_bound(node):
    """收集一个函数体内被绑定的名字（赋值 / for / with / 推导式 / import / 形参）。"""
    bound = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.arguments):
            for arg in (
                list(sub.posonlyargs)
                + list(sub.args)
                + list(sub.kwonlyargs)
                + [sub.vararg, sub.kwarg]
            ):
                if arg is not None:
                    bound.add(arg.arg)
        elif isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            bound.add(sub.id)
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(sub.name)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            bound.add(sub.name)
        elif isinstance(sub, (ast.Import, ast.ImportFrom)):
            for alias in sub.names:
                bound.add((alias.asname or alias.name).split(".")[0])
    return bound


def _module_globals(tree):
    names = set(dir(builtins)) | {"__name__", "__file__"}
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                names.add(sub.id)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(sub.name)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for alias in sub.names:
                    names.add((alias.asname or alias.name).split(".")[0])
    return names


def _undefined_loads(func_node, module_names):
    bound = _collect_bound(func_node)
    missing = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in bound and node.id not in module_names:
                missing.add((node.id, node.lineno))
    return sorted(missing, key=lambda x: x[1])


_INDEXER_SOURCE = _INDEXER_PATH.read_text(encoding="utf-8")
_INDEXER_TREE = ast.parse(_INDEXER_SOURCE)
_MODULE_NAMES = _module_globals(_INDEXER_TREE)
_FUNCS = {
    node.name: node
    for node in _INDEXER_TREE.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
}


class TestRaggedChunkScope:
    """2026-10-20：chunk 路径曾引用未定义的 `k_interval_*`（NameError）。"""

    source = _INDEXER_SOURCE
    tree = _INDEXER_TREE
    module_names = _MODULE_NAMES
    funcs = _FUNCS

    def test_checker_catches_k_interval(self):
        """自检：把 `k_interval_ks` 塞进一个函数，扫描器必须报出来。"""
        tree = ast.parse(
            "def f(a):\n"
            "    x = a + 1\n"
            "    y = k_interval_ks * 2\n"
            "    return x + y\n"
        )
        missing = _undefined_loads(tree.body[0], set(dir(builtins)))
        assert ("k_interval_ks", 3) in missing

    def test_chunk_path_has_no_k_interval_names(self):
        func = self.funcs["_get_topk_ragged"]
        names = {
            node.id
            for node in ast.walk(func)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        bad = sorted(n for n in names if n.startswith("k_interval"))
        assert not bad, f"_get_topk_ragged 仍在引用未定义的 {bad}"

    def test_no_undefined_names_in_indexer_functions(self):
        """通用作用域检查：插件里所有函数都不许 load 未绑定的名字。"""
        problems = []
        for name, func in self.funcs.items():
            for var, lineno in _undefined_loads(func, self.module_names):
                problems.append(f"nsa_indexer.py:{lineno}: {name}() 里的 {var!r} 未定义")
        assert not problems, "\n  ".join([""] + problems)

    def test_chunk_path_runtime_has_no_name_error(self, monkeypatch, vendor):
        """行为：真的把 chunk 分支跑起来（2 个 chunk），确保不是"AST 过了但运行时炸"。"""
        pool = FakeIndexPool()
        monkeypatch.setattr(nsa, "_indexer_kv_pool", lambda: pool)
        seq_lens, extend_lens = [6, 4], [6, 4]
        tokens = sum(extend_lens)
        md = TestInt32LodInputs._ragged_metadata(
            seq_lens=seq_lens, extend_lens=extend_lens
        )
        md.get_seqlens_expanded = lambda: torch.tensor(
            [1, 2, 3, 4, 5, 6, 1, 2, 3, 4], dtype=torch.int32
        )
        md.attn_metadata.topk_indices_offset = None
        chunks = []

        def topk_fn(logits, topk, **kw):
            chunks.append((logits.shape, kw.get("cu_seqlens_q")))
            return torch.full((logits.shape[0], topk), len(chunks), dtype=torch.int32)

        md._topk_fn = topk_fn
        fb = _forward_batch(
            ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
            extend_seq_lens_cpu=extend_lens,
        )
        # free_mem 让 max_rows = (400*0.5)//(10*4) = 5 → 10 个 token 切 2 块
        self_obj = SimpleNamespace(
            index_topk=8,
            block_size=_HEAD_DIM,
            _should_chunk_mqa_logits=lambda q, k, d: (True, 400),
        )
        out = nsa._get_topk_ragged(
            self_obj,
            fb,
            0,
            torch.ones(tokens, 8, _HEAD_DIM, dtype=torch.int8),
            torch.full((tokens, 8), 0.5, dtype=torch.float32),
            md,
        )
        assert len(chunks) == 2, f"应当切成 2 个 chunk，实际 {len(chunks)}"
        assert chunks[0][0] == (5, tokens) and chunks[1][0] == (5, tokens)
        assert all(c[1] is not None and c[1].dtype is torch.int32 for c in chunks), (
            "chunk 路径的 cu_seqlens_q 必须是 int32"
        )
        assert out.dtype is torch.int32 and out.shape == (tokens, 8)
        assert out[0, 0].item() == 1 and out[-1, 0].item() == 2, (
            "两个 chunk 的结果都要回填到 topk_result"
        )


# ---------------------------------------------------------------------------
# 8) `_enable_prefill_cp` / `_k_proj` / `_get_k_bf16_compat`
# ---------------------------------------------------------------------------


class TestPrefillCpFlag:
    def test_dsa_and_nsa_names_both_accepted(self):
        assert nsa._enable_prefill_cp(SimpleNamespace(dsa_enable_prefill_cp=True))
        assert nsa._enable_prefill_cp(SimpleNamespace(nsa_enable_prefill_cp=True))
        assert not nsa._enable_prefill_cp(SimpleNamespace())
        assert not nsa._enable_prefill_cp(SimpleNamespace(dsa_enable_prefill_cp=False))

    def test_dsa_wins_on_conflict(self):
        assert nsa._enable_prefill_cp(
            SimpleNamespace(dsa_enable_prefill_cp=True, nsa_enable_prefill_cp=False)
        )
        assert not nsa._enable_prefill_cp(
            SimpleNamespace(dsa_enable_prefill_cp=False, nsa_enable_prefill_cp=True)
        )


class _FakeNorm(torch.nn.Module):
    def forward(self, x):
        return x


class _FakeRotary(torch.nn.Module):
    def forward(self, positions, q, k):
        return q, k


class TestKProjection:
    """EB5 是 fused indexer（只有 `wk_weights_proj`，没有 `wk`）。"""

    def _module(self, fusion: bool, *, hidden=32, head_dim=_HEAD_DIM, rope=64):
        weight = (torch.randn(head_dim + 4, hidden) * 0.02).to(torch.bfloat16)
        mod = SimpleNamespace(
            use_dsa_indexer_fusion=fusion,
            head_dim=head_dim,
            rope_head_dim=rope,
            k_norm=_FakeNorm(),
            rotary_emb=_FakeRotary(),
            wk_weights_proj=SimpleNamespace(weight=weight),
        )
        if not fusion:
            mod.wk = SimpleNamespace(
                weight=(torch.randn(head_dim, hidden) * 0.02).to(torch.bfloat16)
            )
        return mod

    def test_k_proj_fused_and_unfused_shapes(self):
        x = torch.randn(5, 32, dtype=torch.bfloat16)
        fused = nsa._k_proj(self._module(True), x)
        plain = nsa._k_proj(self._module(False), x)
        assert fused.shape == (5, _HEAD_DIM) and fused.dtype is torch.bfloat16
        assert plain.shape == (5, _HEAD_DIM) and plain.dtype is torch.bfloat16

    def test_k_bf16_compat_fused_skips_hadamard(self, vendor):
        mod = self._module(True)
        key = nsa._get_k_bf16_compat(
            mod,
            torch.randn(3, 32, dtype=torch.bfloat16),
            torch.zeros(3, dtype=torch.int64),
        )
        assert key.shape == (3, _HEAD_DIM)
        assert "hadamard_transform" not in vendor, (
            "fused indexer 在训练时已去掉 Hadamard，推理侧不能再转一次"
        )

    def test_k_bf16_compat_unfused_rotates(self, vendor):
        mod = self._module(False)
        key = nsa._get_k_bf16_compat(
            mod,
            torch.randn(3, 32, dtype=torch.bfloat16),
            torch.zeros(3, dtype=torch.int64),
        )
        assert key.shape == (3, _HEAD_DIM)
        assert vendor["hadamard_transform"]["scale"] == pytest.approx(
            _HEAD_DIM ** -0.5
        )

    def test_maybe_rotate_prefers_module_method(self):
        sentinel = torch.zeros(2, 4)
        mod = SimpleNamespace(_maybe_rotate=lambda t: sentinel)
        assert nsa._maybe_rotate(mod, torch.ones(2, 4)) is sentinel


# ---------------------------------------------------------------------------
# 9) `forward_cuda` 的 dispatch 与 CP 守卫
# ---------------------------------------------------------------------------


class TestForwardCudaDispatch:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("EB5_INDEXER_TORCH_REF", raising=False)
        monkeypatch.setattr(nsa, "_fix_kernel_bindings_once", lambda: None)

    def _backend(self, monkeypatch, md):
        monkeypatch.setattr(
            nsa,
            "get_attn_backend",
            lambda: SimpleNamespace(get_indexer_metadata=lambda lid, fb: md),
        )

    def test_default_env_goes_torch_ref(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            nsa, "_indexer_torch_ref", lambda *a, **k: calls.append(a) or "torch-ref"
        )
        out = nsa.forward_cuda(
            SimpleNamespace(), torch.zeros(1, 8), None, torch.zeros(1), object(), 0
        )
        assert out == "torch-ref"
        assert len(calls) == 1

    def test_metadata_none_returns_none(self, monkeypatch):
        self._backend(monkeypatch, None)
        out = nsa.forward_cuda(
            SimpleNamespace(), torch.zeros(1, 8), None, torch.zeros(1), object(), 0
        )
        assert out is None

    def test_env_zero_uses_legacy_implementation(self, monkeypatch, vendor):
        monkeypatch.setenv("EB5_INDEXER_TORCH_REF", "0")
        self._backend(monkeypatch, SimpleNamespace())
        monkeypatch.setattr(
            nsa, "_indexer_torch_ref", lambda *a, **k: pytest.fail("不该走 torch ref")
        )
        fb = _forward_batch(
            ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor([4], dtype=torch.int32),
            seq_lens=torch.tensor([4], dtype=torch.int64),
        )
        legacy = []
        self_obj = SimpleNamespace(
            index_topk=64,
            dsa_enable_prefill_cp=False,
            _forward_cuda_k_only=lambda *a, **k: legacy.append(a) or "k-only",
        )
        out = nsa.forward_cuda(
            self_obj,
            torch.zeros(4, _HEAD_DIM, dtype=torch.bfloat16),
            None,
            torch.zeros(4, dtype=torch.int64),
            fb,
            0,
        )
        assert out == "k-only", "EB5_INDEXER_TORCH_REF=0 应走原实现的 k-only 快路径"
        assert len(legacy) == 1

    def _legacy_self(self, monkeypatch, *, result="ragged"):
        monkeypatch.setattr(
            nsa,
            "_qk_from_weights",
            lambda self, q_lora, x, pos: (
                torch.zeros(x.shape[0], 8, _HEAD_DIM, dtype=torch.bfloat16),
                torch.zeros(x.shape[0], _HEAD_DIM, dtype=torch.bfloat16),
                None,
            ),
        )
        monkeypatch.setattr(nsa, "_indexer_kv_pool", lambda: FakeIndexPool())
        return SimpleNamespace(
            index_topk=64,
            block_size=_HEAD_DIM,
            dsa_enable_prefill_cp=False,
            _get_q_k_bf16=lambda *a, **k: (
                torch.zeros(a[1].shape[0], 8, _HEAD_DIM, dtype=torch.bfloat16),
                torch.zeros(a[1].shape[0], _HEAD_DIM, dtype=torch.bfloat16),
            ),
            _get_logits_head_gate=lambda x, q_scale: torch.full(
                (x.shape[0], 8), 0.5, dtype=torch.float32
            ),
            _get_topk_ragged=lambda *a, **k: result,
            _get_topk_paged=lambda *a, **k: result,
        )

    def test_legacy_extend_path_calls_ragged(self, monkeypatch, vendor):
        monkeypatch.setenv("EB5_INDEXER_TORCH_REF", "0")
        self._backend(monkeypatch, FakeIndexerMetadata())
        self_obj = self._legacy_self(monkeypatch, result="ragged")
        fb = _forward_batch(
            ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor([512], dtype=torch.int32),
            seq_lens=torch.tensor([512], dtype=torch.int64),
        )
        out = nsa.forward_cuda(
            self_obj,
            torch.zeros(4, _HEAD_DIM, dtype=torch.bfloat16),
            None,
            torch.zeros(4, dtype=torch.int64),
            fb,
            0,
        )
        assert out == "ragged"

    def test_cp_in_seq_split_guard_raises_loudly(self, monkeypatch, vendor):
        """fork 删了 `_get_topk_ragged_with_cp`：CP-v1 in-seq-split 必须显式报错。"""
        import sglang.srt.layers.attention.dsa.utils as dsa_utils

        monkeypatch.setenv("EB5_INDEXER_TORCH_REF", "0")
        self._backend(monkeypatch, FakeIndexerMetadata())
        monkeypatch.setattr(
            dsa_utils,
            "is_dsa_prefill_cp_in_seq_split",
            lambda: True,
            raising=False,
        )
        self_obj = self._legacy_self(monkeypatch)
        fb = _forward_batch(
            ForwardMode.EXTEND,
            seq_lens_cpu=torch.tensor([512], dtype=torch.int32),
            seq_lens=torch.tensor([512], dtype=torch.int64),
        )
        fb.nsa_cp_metadata = SimpleNamespace()
        with pytest.raises(NotImplementedError, match="in-seq-split"):
            nsa.forward_cuda(
                self_obj,
                torch.zeros(4, _HEAD_DIM, dtype=torch.bfloat16),
                None,
                torch.zeros(4, dtype=torch.int64),
                fb,
                0,
            )


# ---------------------------------------------------------------------------
# 10) 已知实现缺陷（xfail 记录，修好会自动 XPASS）
# ---------------------------------------------------------------------------


def test_fork_indexer_metadata_has_cu_seqlens_q():
    """`_get_topk_ragged` 直接调 `metadata.get_cu_seqlens_q()`，fork 的类上却没有。

    `DSAIndexerMetadata`（dsa_indexer_metadata.py:94）只暴露 `attn_metadata`
    字段，访问器里没有 `get_cu_seqlens_q`。EB5 目前靠 skip_logits / decode 绕过
    这条路径，一旦出现 ``seq_len > index_topk`` 的长 prefill 就会 AttributeError。
    """
    from sglang.srt.layers.attention.dsa.dsa_indexer_metadata import DSAIndexerMetadata

    if not hasattr(DSAIndexerMetadata, "get_cu_seqlens_q"):
        pytest.xfail(
            "DSAIndexerMetadata 缺 get_cu_seqlens_q，而 nsa_indexer._get_topk_ragged "
            "直接调用它（长 prefill 必 AttributeError）"
        )
