#!/usr/bin/env python3
"""EB5 在线 INT8（昆仑 P800，8 卡整网）异步并发基准。

设计要点
--------
* `aiohttp` + `stream: true`，逐 SSE 块读取，记录首块时间（TTFT）。
* 并发下限 8：服务是 tp=8 / dp=8 / ep=8 + `--enable-dp-attention`，每张卡一个
  DP rank；客户端并发 < 8 时会有 rank 拿不到请求，DP 集合通信空等而死锁。
* 请求走 sglang 原生 `/generate`，`meta_info` 里有 `prompt_tokens` /
  `completion_tokens`，比按字符估算可靠。
* 每条请求都带一个唯一 nonce 前缀，避免 radix prefix cache 让 TTFT 失真。
* 每组前后采样 `xpu-smi`，记录 8 卡 util / 显存的峰值。

用法
----
    env/bin/python code/sglang-kunlun/01_eb_adp/tool/bench_eb5_int8.py \
        --port 30082 --out .build/perf_int8_8card.md

固定条件：`--disable-cuda-graph`（run_eb5_kunlun.sh 固定打开）、`temperature=0`、
`ignore_eos=true`；`max_new_tokens` 见各场景；prompt 用重复句构造并按
`/tokenize` 校准到目标长度。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

SHORT_UNIT = "The quick brown fox jumps over the lazy dog and keeps running. "
LONG_UNIT = (
    "In a distributed inference system, the scheduler assigns requests to data "
    "parallel ranks, and each rank executes its own prefill and decode steps. "
)
MEM_RE = re.compile(r"(\d+)MiB\s*/\s*(\d+)MiB")
UTIL_RE = re.compile(r"(\d+)%\s+Default")


# --------------------------------------------------------------------------- #
# GPU sampling
# --------------------------------------------------------------------------- #
args_host = "127.0.0.1"
args_port = 30082


async def port_alive(host: str, port: int) -> bool:
    try:
        _r, w = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=3)
        w.close()
        return True
    except Exception:
        return False


class XpuSampler:
    """后台跑 `xpu-smi`，记录 8 卡 util / 显存峰值。"""

    def __init__(self, interval: float = 2.0) -> None:
        self.interval = interval
        self.peak_mem: List[int] = []
        self.peak_util: List[int] = []
        self.samples = 0
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def _probe(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "xpu-smi",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        text = out.decode("utf-8", "ignore")
        mems = [int(m.group(1)) for m in MEM_RE.finditer(text)]
        utils = [int(u.group(1)) for u in UTIL_RE.finditer(text)]
        if not mems:
            return
        self.samples += 1
        if not self.peak_mem:
            self.peak_mem = list(mems)
        else:
            for i, v in enumerate(mems):
                if i < len(self.peak_mem):
                    self.peak_mem[i] = max(self.peak_mem[i], v)
        if utils:
            if not self.peak_util:
                self.peak_util = list(utils)
            else:
                for i, v in enumerate(utils):
                    if i < len(self.peak_util):
                        self.peak_util[i] = max(self.peak_util[i], v)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._probe()
            except Exception:  # 采样失败不影响基准
                pass
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass

    async def start(self) -> None:
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())
        await asyncio.sleep(0)

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
async def post_json(
    session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], timeout: float
) -> Dict[str, Any]:
    async with session.post(
        url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as r:
        if r.status != 200:
            body = await r.text()
            raise RuntimeError(f"HTTP {r.status}: {body[:400]}")
        return await r.json()


async def count_tokens(
    session: aiohttp.ClientSession, base: str, text: str
) -> Optional[int]:
    """用服务端 /tokenize 精确计数；失败返回 None。"""
    for payload in ({"prompt": text}, {"text": text}):
        try:
            data = await post_json(session, base + "/tokenize", payload, timeout=120)
        except Exception:
            continue
        if isinstance(data, dict):
            if "count" in data:
                return int(data["count"])
            if "tokens" in data:
                return len(data["tokens"])
            if "input_ids" in data:
                return len(data["input_ids"])
        if isinstance(data, list):
            return len(data)
    return None


def build_prompt(unit: str, repeats: int) -> str:
    return unit * max(1, repeats)


async def calibrate_prompt(
    session: aiohttp.ClientSession, base: str, unit: str, target: int
) -> Tuple[str, int, int]:
    """返回 (prompt, 实际 token 数, 重复次数)，按 /tokenize 线性校准。"""
    per_unit = await count_tokens(session, base, unit)
    if not per_unit:
        repeats = max(1, round(target / max(1, len(unit) // 4)))
        prompt = build_prompt(unit, repeats)
        return prompt, 0, repeats

    repeats = max(1, round(target / per_unit))
    prompt = build_prompt(unit, repeats)
    n = 0
    for _ in range(6):
        prompt = build_prompt(unit, repeats)
        got = await count_tokens(session, base, prompt)
        if got is None:
            return prompt, 0, repeats
        n = got
        if abs(n - target) <= max(4, target // 32):  # ±3%
            break
        delta = target - n
        step = max(1, round(abs(delta) / per_unit))
        if delta > 0:
            repeats += step
        elif repeats > step:
            repeats -= step
        else:
            repeats = max(1, repeats - 1)
    return prompt, n, repeats


@dataclass
class ReqResult:
    ok: bool
    ttft: float = 0.0
    total: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""

    @property
    def decode_tps(self) -> float:
        dt = self.total - self.ttft
        if self.completion_tokens <= 1 or dt <= 0:
            return 0.0
        return (self.completion_tokens - 1) / dt

    @property
    def e2e_tps(self) -> float:
        if self.total <= 0:
            return 0.0
        return self.completion_tokens / self.total


async def one_request(
    session: aiohttp.ClientSession,
    base: str,
    prompt: str,
    max_new_tokens: int,
    timeout: float,
) -> ReqResult:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = 0.0
    completion = 0
    prompt_tokens = 0
    saw_chunk = False
    buf = ""
    try:
        async with session.post(
            base + "/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout, sock_read=timeout),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return ReqResult(False, error=f"HTTP {resp.status}: {body[:300]}")
            async for raw in resp.content.iter_any():
                now = time.perf_counter()
                buf += raw.decode("utf-8", "ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data in ("", "[DONE]"):
                        continue
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    meta = obj.get("meta_info") or {}
                    usage = obj.get("usage") or {}
                    if not prompt_tokens:
                        prompt_tokens = int(
                            meta.get("prompt_tokens")
                            or usage.get("prompt_tokens")
                            or 0
                        )
                    ct = (
                        meta.get("completion_tokens")
                        or usage.get("completion_tokens")
                        or 0
                    )
                    if ct:
                        completion = int(ct)
                    if (obj.get("text") or meta.get("output_ids")) and not saw_chunk:
                        saw_chunk = True
                        if not ttft:
                            ttft = now - t0
                    if meta.get("finish_reason") and not ttft:
                        ttft = now - t0
        total = time.perf_counter() - t0
        if not ttft:
            ttft = total
        if completion <= 0:
            return ReqResult(
                False, ttft=ttft, total=total, prompt_tokens=prompt_tokens,
                error="no completion_tokens in stream",
            )
        return ReqResult(
            True, ttft=ttft, total=total,
            prompt_tokens=prompt_tokens, completion_tokens=completion,
        )
    except asyncio.TimeoutError:
        return ReqResult(False, error=f"timeout>{timeout:.0f}s")
    except Exception as exc:  # noqa: BLE001
        return ReqResult(False, error=f"{type(exc).__name__}: {exc}"[:300])


# --------------------------------------------------------------------------- #
# Scenario runner
# --------------------------------------------------------------------------- #
@dataclass
class GroupStat:
    name: str
    concurrency: int
    in_len: int
    out_len: int
    rounds: int
    results: List[ReqResult] = field(default_factory=list)
    wall: float = 0.0
    peak_mem: List[int] = field(default_factory=list)
    peak_util: List[int] = field(default_factory=list)
    error: str = ""

    @property
    def ok_results(self) -> List[ReqResult]:
        return [r for r in self.results if r.ok]

    def mean(self, attr: str) -> float:
        vals = [getattr(r, attr) for r in self.ok_results]
        return statistics.fmean(vals) if vals else float("nan")

    @property
    def agg_out_tps(self) -> float:
        if self.wall <= 0:
            return float("nan")
        return sum(r.completion_tokens for r in self.ok_results) / self.wall

    @property
    def agg_total_tps(self) -> float:
        if self.wall <= 0:
            return float("nan")
        return (
            sum(r.prompt_tokens + r.completion_tokens for r in self.ok_results)
            / self.wall
        )


async def run_group(
    session: aiohttp.ClientSession,
    base: str,
    sampler: XpuSampler,
    name: str,
    unit: str,
    in_target: int,
    out_len: int,
    concurrency: int,
    rounds: int,
    timeout: float,
    max_attempts: int = 2,
) -> GroupStat:
    prompt, real_in, repeats = await calibrate_prompt(session, base, unit, in_target)
    stat = GroupStat(name, concurrency, real_in, out_len, rounds)
    print(
        f"[group] {name}: conc={concurrency} in_target={in_target} "
        f"(real={real_in}, repeats={repeats}) out={out_len} rounds={rounds}",
        flush=True,
    )

    for attempt in range(1, max_attempts + 1):
        stat.results.clear()
        await sampler.start()
        t0 = time.perf_counter()
        for rnd in range(rounds):
            tasks = [
                one_request(
                    session,
                    base,
                    f"[req {uuid.uuid4().hex}] {prompt}",
                    out_len,
                    timeout,
                )
                for _ in range(concurrency)
            ]
            res = await asyncio.gather(*tasks)
            stat.results.extend(res)
            n_ok = sum(1 for r in res if r.ok)
            errs = [r.error for r in res if not r.ok]
            print(
                f"  round {rnd + 1}/{rounds}: ok={n_ok}/{concurrency}"
                + (f" err={errs[:2]}" if errs else ""),
                flush=True,
            )
        stat.wall = time.perf_counter() - t0
        await sampler.stop()
        stat.peak_mem = list(sampler.peak_mem)
        stat.peak_util = list(sampler.peak_util)

        n_ok = len(stat.ok_results)
        if n_ok >= max(1, concurrency * rounds // 2):
            break
        stat.error = f"attempt {attempt}: only {n_ok} ok"
        if await port_alive(args_host, args_port) is False:
            stat.error += " (service port closed)"
            print(f"  [fatal] {stat.error}; no retry", flush=True)
            break
        if attempt < max_attempts:
            print(f"  [warn] {stat.error}, retrying", flush=True)
            await asyncio.sleep(3)

    if not stat.ok_results and not stat.error:
        stat.error = "all requests failed"
    return stat


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt(v: float, nd: int = 2) -> str:
    if v != v:  # NaN
        return "—"
    return f"{v:.{nd}f}"


def render_markdown(groups: List[GroupStat], cfg: Dict[str, Any]) -> str:
    lines: List[str] = []
    main = [g for g in groups if not g.name.startswith("warmup")]
    lines.append("## 九、性能实测\n")
    lines.append(
        f"> {cfg.get('run_time', '')} 启动的 8 卡整网 INT8 服务"
        f"（port {cfg.get('port', 30082)}）。"
        "固定 `--disable-cuda-graph`、`temperature=0`、`ignore_eos=true`，"
        "`tp=8 / dp=8 / ep=8`、`--enable-dp-attention`、ctx 32768、"
        "`MEM_FRACTION=0.88`、`MAX_RUNNING_REQUESTS=8`。"
    )
    lines.append("")
    lines.append(
        "prompt 构造：短输入用 `\"The quick brown fox jumps over the lazy dog and keeps running. \"` "
        "重复，长输入用 `\"In a distributed inference system, ...\"` 重复，按服务端 `/tokenize` "
        "校准到目标 token 数；每条请求前置 `[req <uuid>]` 唯一前缀避免 radix prefix cache 命中。"
        "`输出 tok` 即 `max_new_tokens`（`ignore_eos=true`，实际生成长度=设定值）。"
    )
    lines.append("")
    lines.append(
        "| 并发 | 输入 tok | 输出 tok | 平均 TTFT (s) | 平均总延迟 (s) | "
        "单请求 tok/s | 聚合 output tok/s | 聚合 total tok/s | 成功/总数 |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in main:
        n_ok = len(g.ok_results)
        n_all = len(g.results)
        if n_ok == 0:
            lines.append(
                f"| {g.concurrency} | {g.in_len or '—'} | {g.out_len} | — | — | — | — | — | "
                f"0/{n_all} (跳过) |"
            )
            continue
        lines.append(
            f"| {g.concurrency} | {g.in_len} | {g.out_len} | {fmt(g.mean('ttft'))} | "
            f"{fmt(g.mean('total'))} | {fmt(g.mean('decode_tps'))} | "
            f"{fmt(g.agg_out_tps)} | {fmt(g.agg_total_tps)} | {n_ok}/{n_all} |"
        )
    lines.append("")

    mem_groups = [g for g in main if g.peak_mem]
    if mem_groups:
        lines.append("### 显存 / 利用率峰值（`xpu-smi`，每组约 2 s 采样一次）\n")
        lines.append("| 组 | 并发 | 显存峰值 MiB (min–max) | 显存峰值 GiB | util 峰值 |")
        lines.append("|---|---:|---:|---:|---:|")
        for g in mem_groups:
            lo, hi = min(g.peak_mem), max(g.peak_mem)
            lines.append(
                f"| {g.name} | {g.concurrency} | {lo}–{hi} | {hi / 1024:.2f} | "
                f"{max(g.peak_util) if g.peak_util else '—'}% |"
            )
        lines.append("")

    failures = [g for g in main if g.error]
    if failures:
        lines.append("### 异常\n")
        for g in failures:
            errs = sorted({r.error for r in g.results if not r.ok})
            lines.append(
                f"- `{g.name}` (conc {g.concurrency}): {g.error} "
                f"{('; '.join(errs))[:400]}"
            )
        lines.append("")

    lines.append("### 复现条件\n")
    lines.append("```json")
    lines.append(json.dumps(cfg, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main_async(args: argparse.Namespace) -> int:
    global args_host, args_port
    args_host, args_port = args.host, args.port
    base = f"http://{args.host}:{args.port}"
    sampler = XpuSampler(interval=args.sample_interval)
    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        ready = False
        for i in range(60):
            try:
                async with session.get(
                    base + "/v1/models", timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    if r.status == 200:
                        ready = True
                        break
            except Exception:
                pass
            if i % 6 == 0:
                print(f"[wait] {base}/v1/models not ready ...", flush=True)
            await asyncio.sleep(5)
        if not ready:
            print("[fatal] service not ready", flush=True)
            return 2

        groups: List[GroupStat] = []

        if args.warmup:
            print("[warmup] 8 并发 × 1 轮（结果丢弃）", flush=True)
            await run_group(
                session, base, sampler, "warmup", SHORT_UNIT, 128, 64,
                concurrency=8, rounds=1, timeout=args.timeout,
            )

        for conc in [c for c in args.short_conc if c >= 8]:
            g = await run_group(
                session, base, sampler, f"short-c{conc}", SHORT_UNIT, 128, 64,
                concurrency=conc, rounds=args.rounds, timeout=args.timeout,
            )
            groups.append(g)
            if not g.ok_results:
                print("[fatal] short group failed entirely; skip remaining", flush=True)
                break

        # 长输入放最后：>index_topk(2048) 会走 ragged 分支，若该分支有 bug 会
        # 打挂 scheduler，不能让它影响前面已经采到的短输入数据。
        if all(g.ok_results for g in groups):
            for conc in [c for c in args.long_conc if c >= 8]:
                g = await run_group(
                    session, base, sampler, f"long-c{conc}", LONG_UNIT, 2048, 16,
                    concurrency=conc, rounds=args.rounds,
                    timeout=args.long_timeout, max_attempts=1,
                )
                groups.append(g)
                if not g.ok_results:
                    print("[fatal] long group failed; stop", flush=True)
                    break

    cfg = {
        "run_time": time.strftime("%Y-%m-%d %H:%M"),
        "port": args.port,
        "endpoint": base + "/generate",
        "stream": True,
        "temperature": 0,
        "ignore_eos": True,
        "disable_cuda_graph": True,
        "tp": 8, "dp": 8, "ep": 8,
        "enable_dp_attention": True,
        "context_length": 32768,
        "mem_fraction_static": 0.88,
        "max_running_requests": 8,
        "short_scenario": {
            "input_tokens": 128, "max_new_tokens": 64,
            "concurrency": args.short_conc, "rounds": args.rounds,
        },
        "long_scenario": {
            "input_tokens": 2048, "max_new_tokens": 16,
            "concurrency": args.long_conc, "rounds": args.rounds,
            "note": "输入 >index_topk(2048)，走 ragged 分支",
        },
        "warmup": "8 并发 × 1 轮，结果丢弃",
    }

    md = render_markdown(groups, cfg)
    print("\n" + md, flush=True)
    if args.out:
        header = (
            "# EB5 8 卡整网 在线 INT8 性能实测\n\n"
            f"- 时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- 服务：{base}\n"
            "- 脚本：`code/sglang-kunlun/01_eb_adp/tool/bench_eb5_int8.py`\n\n"
        )
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(header + md + "\n")
        print(f"[out] wrote {args.out}", flush=True)

    bad = [g for g in groups if not g.ok_results]
    return 0 if not bad else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EB5 online INT8 benchmark (8-card full net)"
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30082)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--timeout", type=float, default=1200.0)
    p.add_argument("--long-timeout", type=float, default=300.0)
    p.add_argument("--sample-interval", type=float, default=2.0)
    p.add_argument("--short-conc", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--long-conc", type=int, nargs="+", default=[8])
    p.add_argument("--no-warmup", dest="warmup", action="store_false")
    p.add_argument("--out", default="")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main_async(parse_args())))
