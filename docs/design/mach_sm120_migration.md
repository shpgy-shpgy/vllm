# Opt-in Mach SM120 adapters

The pre-migration working tree is preserved in commit `4a01a1c51d`.
Source: `/data/lxy/detailed_benchmark/vllm-mach`, revision `c507ef3`
plus its local validated implementation. This is a native vLLM integration,
not a plugin installation or a dependency on `vllm.mach.mxfp6.serve`.
Continue using the standard vLLM server entrypoints.

## Scope and baseline contract

All newly introduced optimization flags default to **off**. No launcher changes
the attention backend, compilation policy, scheduling limits, precision, GPU
selection or sampling parameters. Existing Quark weight preparation, MoE scratch
ownership fixes, compact sampling and hybrid NVFP4 head are retained.

| Path | Integration / eligibility |
| --- | --- |
| Native W6A8 and MoE small/grouped schedules | Already in Quark/MXFP6; retained, not replaced with plugin kernels |
| Compact vocab-parallel sampling, hybrid NVFP4 head | Already in V2 runner and logits processor; retained |
| MoE compiled AR/residual/GemmaRMSNorm | Preserved in snapshot; `VLLM_QWEN3_5_FUSED_AR_NORM=1` |
| MoE projection dispatch | 37 frozen configurations from `58014e1`; `VLLM_MACH_MOE_PROJECTION_TUNING=1`, `MXFP6_AUTOTUNE=off`; native ABI `native-w6a8-30-v5` |
| Persistent GDN | `VLLM_MACH_GDN_PERSISTENT=1`; physical M=1,2,4,8 |
| BA projection overlap | `VLLM_MACH_GDN_BA_OVERLAP=1`; physical M=16,24,32 |
| SwiGLU/MXFP8 producer | `VLLM_MACH_FUSED_SWIGLU_QUANT=1`; Dense 5120 TP2 |
| GDN norm/MXFP8 producer | `VLLM_MACH_FUSED_GDN_QUANT=1`; Dense, requires persistent/BA adapter |
| AR/norm/MXFP8 producer | `VLLM_MACH_FUSED_AR_QUANT=1`; Dense M=2,4,8,16,24,32, requires AR-Norm and preceding producers |
| Lossless prefill transport | `VLLM_SM120_LOSSLESS_PREFILL=1`; Dense 5120 TP2, validated shapes only |
| Owner prefill | `VLLM_SM120_OWNER_PREFILL=1`; Dense 64-layer 5120 TP2, M=512..4096, no graph capture |
| Narrow FP16 SSM admission | `VLLM_QWEN3_5_FP16_SSM=1` plus `--mamba-ssm-cache-dtype float16`; does not change allocation by itself |

The new Python GDN/producer/owner adapters currently require compilation mode
`NONE`; decode CUDA Graphs remain enabled with `FULL_DECODE_ONLY`. MoE's existing
compiled AR-Norm and projection table paths do **not** require that restriction. Do not disable MoE
prefill compilation merely to enable GDN adapters: the source benchmark did not
establish a net benefit from that tradeoff. Unsupported speculative/LoRA adapters
are rejected before mutation. Strict producer mode `1` rejects zero eligible
layers; `auto` permits unsupported layers to remain on their original paths.

An option being set is not evidence of speedup. GDN logs report both preparation
and selected physical shapes. Host counters count dispatch/capture, not graph
replays. Check preparation, eligible request shapes and GPU traces before
claiming a path was exercised.

## Native dependencies

Validated dependency contract: FlashInfer 0.6.18, mxfp6-sm120 0.2.1, CUDA 13.0,
SM120 and TP2. Native sources are vendored under `native/mach/` with their
original Apache-2.0 licenses. Install into the environment used by the server:

```bash
uv pip install --python .venv/bin/python --no-deps --no-build-isolation ./native/mach/ar_norm
uv pip install --python .venv/bin/python --no-deps --no-build-isolation ./native/mach/lossless_prefill
uv pip install --python .venv/bin/python --no-deps --no-build-isolation ./native/mach/owner_prefill
```

On this host, PyTorch's CUDA headers are distributed in
`/usr/local/lib/python3.12/site-packages/nvidia/cu13/include`; prefix native builds
with `CPATH=/usr/local/lib/python3.12/site-packages/nvidia/cu13/include` if the
toolkit cannot find `cusparse.h`. This does not modify system packages.
Persistent GDN JIT source ships with vLLM and is warmed before graph capture.
No native extension is compiled implicitly by the lossless/owner/AR loader.

Owner prefill allocates partner MLP replicas during memory profiling, before KV
cache allocation. Expect approximately 3.1 GiB extra per rank for the default
32 replicated Dense MLP layers; this can reduce serving capacity.

## Accuracy and validation

Lossless codec and producer fusion preserve specified intermediate rounding;
verification flags compare against the unfused computation, not model quality:
`VLLM_SM120_LOSSLESS_PREFILL_VERIFY=1`, `VLLM_SM120_OWNER_VERIFY=1`.
Do not include verification overhead in performance comparisons.
FP16 SSM changes recurrent-state precision. Persistent GDN also changes arithmetic
ordering and is not bitwise identical to the reference recurrence. Neither is
a claim of lossless model output.

Tests:

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest --confcutdir=tests/kernels/mamba tests/kernels/mamba/test_mach_gdn.py -q
CUDA_VISIBLE_DEVICES=4 .venv/bin/python benchmarks/kernels/verify_mach_gdn.py --model 27b --output /tmp/mach-gdn-27b.json
CUDA_VISIBLE_DEVICES=4 .venv/bin/python benchmarks/kernels/verify_mach_gdn.py --model 35b --output /tmp/mach-gdn-35b.json
```

Performance comparisons must use identical prompts, actual output-token counts,
sampling, cache dtype, GPU pair and compilation/scheduling configuration. Source
tables combine configuration changes and are not promised speedups here.

## Local acceptance (2026-09-28)

On GPU 4/5, CUDA 13 / PyTorch 2.13 / FlashInfer 0.6.18:

- Persistent GDN: both Dense and MoE geometries, FP16/FP32, SD/DS layouts,
  M=1/2/4/8, four changing-input iterations per case. Graph equals eager;
  null slots and canaries remain unchanged. These are 32 cases in total.
- BA adapter matches reference output/conv/state bitwise at M=16/24/32.
- Lossless/owner: all supported codec shapes, ragged partitions and alternating
  graph replays pass. AR quant's BF16 outputs, MXFP8 codes and logical scales
  match the reference bitwise at M=2/4/8/16/24/32 with changing graph inputs.
- Dense model: all 48 GDN, 64 MLP and 64 AR quant layers prepared per rank;
  producer logs confirm execution. Owner verification passes 717 checks per
  rank on a real forward; lossless M3000 verification passes.
- Default compile factors exactly match the saved HEAD (259 fields); new
  disabled flags do not change the environment portion of the cache identity.

Dense HTTP **smoke throughput**, not the historical serving benchmark:
3,000 input / 256 output tokens, greedy, no EOS stop, two rounds of C requests,
FP32 SSM, FLASHINFER, batched tokens 4096, max sequences 64, hybrid head enabled
in both arms. Both use `NONE/FULL_DECODE_ONLY`, capture sizes
`[1,2,4,8,16,24,32,64]`. Only optimization flags differ; verification is off.

| Concurrency | All adapters off (tok/s) | All Dense adapters on (tok/s) |
| --- | ---: | ---: |
| 1 | 84.92 | 107.46 |
| 4 | 231.61 | 334.52 |
| 16 | 484.39 | 689.45 |
| 32 | 597.80 | 859.91 |

All 106 scored long requests in each arm completed their 256 output tokens;
short Chinese arithmetic and English factual prompts returned `156` and `Paris`
in both arms. Long outputs remained readable but were not identical, consistent
with persistent GDN's changed arithmetic ordering. This is a bounded smoke test,
not a task-quality evaluation or a statistically established speedup. In
particular, it does not compare against the original compiled baseline or the
historical 3,000/1,000 benchmarks. Clocks were not locked.

Logs and raw results from this run are under `/tmp/mach-migration-*`;
kernel acceptance results are `/tmp/mach-gdn-27b-acceptance.json` and
`/tmp/mach-gdn-35b-acceptance.json`.

The final combined regression completed with **171 passed** (119.51 seconds),
covering GDN dispatch and FP16 admission, AR-quant handoff, TP2 prefill/graph
correctness, frozen MoE projection profiles and unchanged default compilation
factors. The log is `/tmp/mach-final-pytest.log`. The standalone GDN GPU
acceptance additionally passed all 32 geometry/dtype/layout/batch combinations.
All temporary API servers used for this validation were stopped.

Historical EXL3/vLLM 0.28 profiles and the source's rejected extra-loss expert
experiment are not native MXFP6 optimizations and are not applied here.

MoE acceptance used the standard API server, `VLLM_COMPILE/FULL_AND_PIECEWISE`,
FLASHINFER, FP32 SSM, 2048 batched tokens and GPUs 4/5. Both ranks installed
all 37 projection entries; small-batch, grouped and compact sampling executed.
All 106 scored 3000/256 requests completed at C=1/4/16/32. A separate batch of
32 distinct Chinese addition questions passed 32/32. Repeated-text completion
prompts often continued repeating the prompt, so these are load probes rather
than task-quality evidence. No paired MoE performance gain is claimed here.

### Subsequent paired 3000/1000 serving test

The follow-up used GPU 4/5, FP32 SSM, FLASHINFER, hybrid NVFP4 head in every arm,
four fixed repeated-text prompts, greedy sampling and ignored EOS. BS1 had three
measured requests; BS32 had two rounds of 160 requests after warmup. All 1,938
measured requests completed exactly 1,000 output tokens. Throughput below pools
tokens and elapsed time across repeats; startup/warmup is excluded.

| Configuration | BS1 output tok/s | BS32 output tok/s |
| --- | ---: | ---: |
| Dense compiled baseline | 97.12 | 1419.20 |
| Dense NONE, adapters off | 80.07 | 1152.09 |
| Dense NONE, all adapters on | 103.18 | 1533.06 |
| MoE compiled baseline | 314.36 | 3103.15 |
| MoE compiled + AR-Norm | 325.85 | 3195.93 |
| MoE compiled + AR-Norm + projection | 323.51 | 3216.52 |

Dense used 4096 batched tokens; MoE used 2048. Baselines and MoE retained default
compilation; Dense adapters used NONE/FULL_DECODE_ONLY. Every arm supplied the
same full 1..2048 capture-size list, with the normal mode-specific filtering.
Dense net gains over its compiled baseline were 6.24%/8.02%, not the larger
28.86%/33.07% comparison against its uncompiled control. Owner reduced Dense KV
capacity from 434,366 to 328,442 tokens. MoE AR-Norm gains were 3.66%/2.99%;
projection's additional -0.72%/+0.64% does not establish a stable incremental gain.
These are current-worktree fixed-load comparisons, not historical FP16 SSM
benchmarks, cross-version regression tests or per-adapter Dense ablations.
Clocks were not locked and runs were sequential. Full commands, samples and
logs are in `/data/luyufan/mach-perf-20260928/RESULTS.md` on the validation host.

## Enabling without a custom launcher

Use `.venv/bin/python -m vllm.entrypoints.openai.api_server` (or the normal vLLM
CLI) in the environment containing the extensions. The AR wheel built in this
validation is installed only in the project's `.venv`, not system Python.

For the validated Dense adapter configuration, add these environment values:

```bash
VLLM_QWEN3_5_FUSED_AR_NORM=1
VLLM_MACH_GDN_PERSISTENT=1
VLLM_MACH_GDN_BA_OVERLAP=1
VLLM_MACH_FUSED_SWIGLU_QUANT=1
VLLM_MACH_FUSED_GDN_QUANT=1
VLLM_MACH_FUSED_AR_QUANT=1
VLLM_SM120_LOSSLESS_PREFILL=1
VLLM_SM120_OWNER_PREFILL=1
```

These are environment values, not a complete shell command: export them or
prefix the server command with space/backslash-separated assignments. Set
compilation explicitly to `NONE/FULL_DECODE_ONLY`, specifying capture sizes in
only one place. No `--enforce-eager` is needed. Keep your original baseline
command separately; this is an opt-in profile, not a rewrite of the baseline.

For MoE compiled serving, the relevant opt-ins are:

```bash
VLLM_QWEN3_5_FUSED_AR_NORM=1
VLLM_MACH_MOE_PROJECTION_TUNING=1
MXFP6_AUTOTUNE=off
```

Keep the normal compiled/prefill-graph policy; leave the Dense-only adapters
off. FP16 SSM and hybrid NVFP4 head remain independent precision/sampling
choices, not silently enabled by either group of flags.

## Qwen3.5-2B MXFP8 BF16 GDN follow-up (2026-10-15)

The BF16 ordered port needs the mixed decode companion as well as pure decode.
With only pure decode enabled, a batch containing both prefill and decode
materializes pending state for all requests and falls back to stock recurrence.
The companion retains ordered history for decode requests while preserving the
stock convolution and prefill branch. The mixed kernel must round state to
BF16 after every replayed token, matching a stock state store/load boundary;
that upstream rounding step was missing from the original local mixed port.
Allocation hooks now reset metadata only for blocks assigned to GDN groups,
not attention-only KV growth.

Automatic admission selects ordered and mixed together for the validated
SM120 Qwen3.5-2B MXFP8 native profile: V2, TP1, BF16 model/SSM/KV,
FLASHINFER, 64 maximum sequences, no prefix caching, ReplaySSM, MTP, LoRA,
offload, transfer connector or fixed block-count override. Explicitly setting
either `VLLM_MACH_MXFP8_2B_GDN_ORDERED=0` or
`VLLM_MACH_MXFP8_2B_GDN_MIXED=0` suppresses this automatic pair. Other profiles
retain their previous behavior; in particular 4B does not acquire this default.

Same-GPU6 sequential comparisons used the unchanged 3000/1000 benchmark,
one complete warmup sweep and two scored sweeps each, with BF16 SSM and default
BF16 KV. The server retained the original capture sizes and launch arguments.

| 2B BS64 route | Run 1 output tok/s | Run 2 output tok/s | Mean |
| --- | ---: | ---: | ---: |
| Stock GDN | 10189.95 | 10136.69 | 10163.32 |
| Ordered, mixed off | 9951.52 | 9848.78 | 9900.15 |
| Ordered + BF16-correct mixed | 10556.54 | 10560.93 | 10558.74 |

The complete pair improves BS64 by 3.89% over stock and 6.65% over pure-only
ordered. Mean ITL changes from 5.868 to 5.649 ms versus stock. Small-batch
throughput is essentially unchanged; this is not an all-batch speedup claim.
The benchmark retokenizes output text, so elapsed time and ITL were also
checked: BS64 mean elapsed time changes from 31.476 to 30.306 seconds.
Requests use the benchmark's unchanged random seed policy and clocks are not
locked, so these are repeated serving measurements, not a statistical bound.

A subsequent unchanged-command run with neither GDN switch in the environment
confirmed automatic admission and mixed branch hits. Its scored BS64 duration
corresponds to 10617.43 generated tokens/s (320000 tokens); the CSV reports
10716.80 because output retokenization counted 322995 tokens. This run verifies
default routing, rather than establishing an additional optimization gain.
The unchanged 4B GPU3 command, using
`/data/models/Qwen3.5-4B-MXFP8-test`, scored 4855.76 output tokens/s at BS64
(4855.89 from fixed generated tokens and elapsed time). Its worker receipt
shows neither 2B ordered/mixed defaults nor ordered scratch reservation.
The earlier same-GPU3 stock runs were about 4861 tokens/s, so this single
4B confirmation run does not indicate a material regression.

Kernel/lifecycle/default-routing regression passes 176 cases, including
129-step FP32/BF16 mixed recurrence, pure/mixed handoffs, graph replay,
materialization and slot reuse. This is not a new task-accuracy evaluation.
Raw commands, manifests, CSVs and diagnostic findings are under
`/data/luyufan/gdn_bf16_20261014/` on the validation host.
