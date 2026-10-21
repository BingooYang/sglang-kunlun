# DeepSeek-V4-Flash on Kunlun P800 (SGLang)

Single-node, 8 × P800 OAM (96 GB) recipe used for validation. `sglang-kunlun`
provides the platform plugin; this directory only contains the launcher and the
runtime environment.

On the `eb5-adapt` branch the DSV4/ERNIE adaptations live in the
`sglang_kunlun.eb5_adapt` overlay and are applied by the `eb5_kunlun_patch`
plugin entry point. `env.sh` therefore sets the `SGLANG_PLUGINS` whitelist
(`kunlun,kunlun_hooks,eb5_kunlun_patch`) — if that variable is a whitelist and
`kunlun` is omitted, the platform silently falls back to CUDA.

## Verified configuration

- sglang: community `main` (`16eb87c`) or `sglang-ERNIE-5.5-Flash` rebased on it
- sglang-kunlun: this branch (`eb5-adapt`)
- torch: 2.9.0 + xpytorch v3.7.0 (`cp310`, `ubuntu2004`)
- kunlun_ops: 0.1.227
- xspeedgate_ops: 1.5.1 (`torch29`)
- deep_ep: 3.2.1.6 (`cuda12`)
- model: `DeepSeek-V4-Flash-FP8-W8A8-INT8-Dynamic` (bf16 KV cache)

## Environment

```bash
conda create -n sglang_main python=3.10 -y
conda activate sglang_main
pip install torch==2.9.0
# xpytorch (installs torch_xmlir, xcudart, flash_attn, xformers, xmlir, ...)
bash xpytorch-cp310-torch290-ubuntu2004-x64.run
pip install kunlun_ops-0.1.227+2b100f96-cp310-cp310-linux_x86_64.whl
pip install xspeedgate_ops-1.5.1+87067b3.torch29-cp310-cp310-linux_x86_64.whl
pip install deep_ep-3216-cp310-cp310-linux_x86_64.whl
```

Install sglang and this plugin in editable mode:

```bash
SGLANG_BUILD_RUST_EXTS=none pip install --no-deps --no-build-isolation -e /path/to/sglang/python
pip install --no-deps -e /path/to/sglang-kunlun
```

> The Rust tree extension currently requires torch 2.11+, so build with
> `SGLANG_BUILD_RUST_EXTS=none` on the torch 2.9 XPU stack.

## Launch

```bash
export MODEL_PATH=/path/to/DeepSeek-V4-Flash-FP8-W8A8-INT8-Dynamic
cd examples/dsv4-kunlun

# simplest: TP8/EP8/DP8, DeepEP normal, no graph
MODE=bringup CONDA_ENV=sglang_main PORT=38080 bash launch.sh

# low_latency + EAGLE (3 draft steps), no graph
MODE=ll CONDA_ENV=sglang_main PORT=38080 bash launch.sh

# low_latency + EAGLE + CUDA graph (recommended)
MODE=llgraph CONDA_ENV=sglang_main PORT=38080 bash launch.sh
```

All modes use `--kv-cache-dtype bfloat16 --dtype bfloat16`; float16 KV is not
supported by this checkpoint.

## Expected results (8 × P800)

- bringup: correct, ~1.3 tok/s
- ll (low_latency + EAGLE, no graph): correct, ~2.1 tok/s
- llgraph (low_latency + EAGLE + CUDA graph): correct, ~3.9 tok/s

`llgraph` logs show `cuda graph: True` and `accept len ≈ 2.0–2.4`.
`prefill` CUDA graph is auto-disabled for DeepSeek-V4 by the runtime; decode /
target-verify graphs are captured.

## Notes

- `env.sh` sets the `SGLANG_PLUGINS` whitelist so the `eb5_kunlun_patch`
  overlay loads; it also contains the KLX/XCCL variables (`XSHMEM_MODE=1`,
  `BKCL_RDMA_NICS=...`) that the self-built environment must set manually.
- `env_ll.sh` adds the low-latency overlay (`XSHMEM_SYMMETRIC_SIZE`,
  DeepEP LL, EAGLE).
- The DP-idle / short-prompt zero-length kernel guards and the ERNIE-5.5-Flash
  `sigmoid_gate_mul` / `multimax` torch replacements are applied by
  `eb5_kunlun_patch` (module `sglang_kunlun.eb5_adapt.eb5_xy_adapt`). A
  dedicated Kunlun attention backend for the ERNIE model is not included; the
  values above are validated with the DeepSeek-V4 checkpoint.
