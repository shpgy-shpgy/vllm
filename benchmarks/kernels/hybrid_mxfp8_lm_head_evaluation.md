# Hybrid MXFP8/BF16 lm-head evaluation (C=128)

Date: 2026-08-06

## Conclusion

The opt-in lm-head path uses OCP MXFP8 (E4M3 values with one E8M0 scale per
32 elements) for a full-vocabulary coarse projection, selects 128 candidates,
and recomputes those logits from the original BF16 weight. It is shape-generic
through the configured row limit and applies to both the main and MTP heads.

The final non-eager V2/TP2/MTP2 serving A/B is positive at every requested
concurrency point from BS1 through BS96. Against a fresh same-process-setup
BF16 baseline, mean TPOT falls by 10.37%, 5.62%, 3.69%, 0.96%, and 2.08% at
BS1, 8, 32, 64, and 96. Output throughput rises by 7.46%, 4.24%, 3.01%,
1.10%, and 1.89%, respectively.

The important qualification is that the fastest isolated refinement kernel
was not the fastest serving configuration. Tiling several BF16 candidates in
one Triton program saves 4-12 us at large M, but changes the floating-point
reduction order. At M>=64 this was enough to perturb near-tied continuations
and downstream execution: BS64/96 became 1.18%/1.93% slower than BF16. The
final dispatcher therefore uses four-candidate tiles only for 16<=M<64 and
keeps the original scalar reduction for M<16 and M>=64. That policy restores
positive serving results across BS1-96.

The other two attempted Top-K changes do not belong in production:

- row-wise CUB `DeviceTopK` is slower because its API processes one sequence;
- exact hierarchical Top-K saves about 2 us only at M=1 and is 1.7-1.9x
  slower at representative medium/large shapes.

No eager serving number is used in this report.

## Final implementation

Each TP rank retains the original BF16 lm-head and creates one persistent
MXFP8 copy. Runtime execution is:

1. quantize contiguous BF16 hidden states to E4M3 with per-32 E8M0 scales;
2. run a shape-autotuned FlashInfer-CUTLASS MXFP8 GEMM over the local vocab;
3. select an exact unsorted C=128 set with FlashInfer `top_k`, whose backend
   is auto-dispatched for the current GPU and shape;
4. recompute selected logits directly from BF16 weights with Triton, using
   candidate tiles of four only for 16<=M<64 and scalar programs otherwise;
5. reduce only compact winners/candidates across TP ranks; and
6. for presence-only penalties, touch only persistent unique output-token IDs
   at M>=32, while retaining the dense exact fallback for small M.

The indexed refinement replaces `weight[candidate_indices]` plus BF16 `bmm`.
At M=288, the old path materialized a 144 MiB `[288,128,2048]` BF16 tensor.
The indexed kernel removes that allocation and its write/read while retaining
only the small `[M,128]` refined-logit output.

Candidate selection stays in BF16 whenever all preceding transforms are
monotonic. This avoids a full `[M,N]` FP32 copy and twice-width Top-K input.
The refined candidate logits are still converted to FP32 before exact
soft-cap, scale, temperature, and penalty processing.

Vocabulary rows are padded to a multiple of 32 for CUTLASS and sliced back to
the logical size. Hidden size must be divisible by 32. Vocabulary size, hidden
size, and row count are not hard-coded. Unsupported device capability, dtype,
layout, top-k, vocabulary, or a row count above the configured limit falls
back to the original BF16 path.

The largest configured row shape is autotuned during model loading. The
FlashInfer 0.6.13 profiling-delay helper cannot compile in this CUDA 13 test
environment because the toolkit lacks compatible cuBLAS development headers;
vLLM replaces only that delay operation with `torch.cuda._sleep`. Tactic
enumeration, measurement, and caching remain FlashInfer's. Without this
workaround, valid tactics are rejected and tactic 0 is slower than BF16.

## Candidate-selection algorithms

### CUB DeviceTopK/AIR

The standalone benchmark uses the FlashInfer-vendored CCCL 3.3.2 headers,
BF16 key/value pairs, unsorted output, non-deterministic ordering, and CUDA
Graph replay. Equal BF16 values at the kth boundary may legally return
different tied indices, so correctness compares selected values.

Configuration: RTX 5090 (SM120), `N=124160`, C=128.

| M | CUB `DeviceTopK` AIR | FlashInfer `top_k` | `torch.topk` |
| ---: | ---: | ---: | ---: |
| 1 | 20.458 us | 18.482 us | 26.468 us |
| 2 | 38.858 us | 18.490 us | - |
| 4 | 75.756 us | 18.712 us | - |
| 8 | 145.090 us | 18.601 us | - |
| 16 | 296.071 us | 19.970 us | - |
| 32 | 586.269 us | 20.518 us | - |

`DeviceTopK` accepts one sequence, so the extension loops over rows. Newer
`DeviceBatchedTopK` targets segments small enough for one CTA and does not fit
124,160-element local-vocabulary rows. The useful CUB idea is adaptive radix
filtering fused across iterations, but its public execution model is not a
drop-in batched selector here.

FlashInfer on SM120 can auto-dispatch among cluster, filtered, and multi-CTA
exact implementations. Its selector is roughly 10-19% of the isolated hybrid
lm-head path, but only around 1-2% of the complete high-BS MTP decode
iteration. Replacing it cannot explain or recover millisecond-level serving
differences.

### Exact hierarchical Top-K

For any partition of a row, global top-C must be contained in the union of
each partition's local top-C. The benchmark applies this exact property using
FlashInfer for the first stage and a final merge.

| M | Direct Top128 | Best useful partition | Hierarchical | Relative time |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 18.438 us | 20 x 6,208 | 16.425 us | 0.89x |
| 32 | 20.466 us | 20 x 6,208 | 34.797 us | 1.70x |
| 96 | 36.710 us | 20 x 6,208 | 69.674 us | 1.90x |
| 288 | 96.333 us | 20 x 6,208 | 171.941 us | 1.78x |

Five partitions reach 94.307 us at M=288, only about 2 us faster than direct
selection, while becoming slower at smaller rows. A standard CUTLASS SM120
linear-combination epilogue is pointwise and its CTA-N choices top out at 256.
An exact Top128 epilogue would still emit at least half of each CTA's outputs,
then require inter-CTA reduction and a second Top-K. This needs a custom
collective rewrite rather than a small epilogue extension, with no measured
margin to repay the added synchronization. It is therefore not integrated.

References:

- GPU Top-K paper and Bitonic/RadixSelect analysis:
  <https://anilshanbhag.com/static/papers/gputopk_sigmod18.pdf>
- CUB `DeviceTopK` implementation:
  <https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_topk.cuh>
- CUB `DeviceBatchedTopK` contract:
  <https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceBatchedTopK.html>

## Isolated CUDA-Graph performance

Configuration: RTX 5090 (SM120), one actual TP rank, `N=124160`, `K=2048`,
C=128, BF16 output, and FlashInfer exact candidate selection.

| M | Native BF16 | Hybrid MXFP8/BF16 | Indexed refine | Hybrid speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 316.916 us | 180.333 us | 4.185 us | 1.757x |
| 32 | 316.685 us | 197.161 us | 6.185 us | 1.606x |
| 64 | 324.280 us | 224.971 us | 14.364 us | 1.441x |
| 96 | 354.248 us | 240.534 us | 18.475 us | 1.473x |
| 128 | 363.772 us | 268.582 us | 22.570 us | 1.354x |
| 192 | 510.583 us | 352.683 us | 33.021 us | 1.448x |
| 288 | 821.212 us | 513.753 us | 61.142 us | 1.598x |

At M=1, hidden quantization is about 3.85 us, the selected CUTLASS MXFP8 GEMM
tactic is about 158 us, and candidate selection is about 18.5 us. The
default/fallback CUTLASS tactic 0 takes about 881.6 us and is not valid for
performance conclusions. Tactic choice changes with M, so loading autotunes
the configured maximum shape instead of hard-coding one tactic.

### BF16 refinement tiles

The isolated tiled kernel is faster, but the final dispatcher deliberately
uses only the M=16-48 portion of this result.

| M | Scalar refine | Best tiled refine | Isolated reduction |
| ---: | ---: | ---: | ---: |
| 16 | 6.152 us | 4.052 us (tile 4) | 34.1% |
| 32 | 8.190 us | 6.142 us (tile 4) | 25.0% |
| 48 | 10.241 us | 8.186 us (tile 8) | 20.1% |
| 64 | 12.296 us | 9.337 us (tile 4) | 24.1% |
| 96 | 16.399 us | 12.284 us (tile 4) | 25.1% |
| 128 | 22.516 us | 14.742 us (tile 4) | 34.5% |
| 192 | 32.752 us | 20.521 us (tile 8) | 37.3% |
| 288 | 59.376 us | 49.144 us (tile 4) | 17.2% |

Four warps occasionally differed from the eight-warp result by 0.125, so all
production variants use eight warps. More importantly, large-M tiling changed
the reduction tree even with eight warps. The service-level cost of changed
continuations exceeded the microsecond kernel saving, motivating the M>=64
scalar policy.

### Avoiding the full FP32 coarse copy

Selecting directly from BF16 instead of `.to(float32)` plus Top-K saves:

| M | Saved CUDA-Graph time |
| ---: | ---: |
| 1 | 2.041 us |
| 32 | 22.578 us |
| 64 | 30.553 us |
| 96 | 51.234 us |
| 128 | 61.471 us |
| 192 | 150.470 us |
| 288 | 248.560 us |

This is valid only before monotonic transforms. Non-positive scale or a
presence penalty combined with non-trivial soft-cap/scale materializes FP32
and uses the exact general path.

## Precision and penalty coverage

Five-seed real-weight scans covered M=1, 16, 32, 64, 96, 128, 192, and 288.
All 4,085 native greedy winners were present in the C=128 candidate set and
all final winners agreed. Two high-M refined values differed because of BF16
reduction ties; the final large-M scalar dispatch is at least as conservative
as this audit.

For presence penalty 1.0 with a 128-token generated history, 20-seed tests at
M=1, 32, 96, and 288 retained all 8,340 greedy winners and all 309,120 native
top-40 candidates. The exact per-shape top-40 counts were 800, 25,600, 76,800,
and 230,400.

Presence-only penalties now keep a persistent per-request list of unique
generated token IDs. The hot kernel updates this list only for rows whose
presence penalty is non-zero; penalty-free requests do not pay the extra
stores. At M<32 the existing dense counts kernel remains faster.

CUDA-Graph comparison with 128-token history:

| M | Dense counts | Sparse unique IDs | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 4.080 us | 4.067 us | 1.00x |
| 8 | 4.178 us | 4.046 us | 1.03x |
| 16 | 4.094 us | 4.207 us | 0.97x |
| 32 | 6.153 us | 4.090 us | 1.50x |
| 64 | 10.238 us | 3.989 us | 2.57x |
| 96 | 12.526 us | 4.062 us | 3.08x |
| 288 | 34.821 us | 4.050 us | 8.60x |

Supported sampling semantics are unchanged:

| Path | Hybrid coarse/refine | Semantics |
| --- | --- | --- |
| Greedy / argmax | Yes | Exact winner in the measured audit |
| Bounded top-k, then top-p | Yes | Candidate based; BF16 boundary ties possible |
| Presence-only penalty | Yes | Sparse coarse penalty plus exact refinement |
| Full-vocabulary random sampling | No | BF16 fallback retains every token |
| Frequency/repetition penalty or arbitrary bias | No | BF16 fallback |

BF16 refinement can recover a native token only when MXFP8 placed it within
C=128. The method remains approximate in principle even though every tested
winner/top-40 token was retained.

## Non-eager V2/TP2/MTP2 serving A/B, BS1-96

Configuration: GPUs 4/5, `VLLM_USE_V2_MODEL_RUNNER=1`, TP2, MTP2, C=128,
`VLLM_USE_DEEP_GEMM=0`, `enforce_eager=False`, max sequences 96, input 3000,
output 200, temperature 0, ignore EOS, fixed seed 1785902258, and CUDA Graph
capture sizes `[1,2,3,4,8,12,16,24,32,48,64,96,128,192,288]`.

Each concurrency point was exercised once before formal measurement. BF16
and hybrid services used identical CUTLASS/Triton model backends. Historical
files were not mixed into the final A/B because run-to-run platform drift was
large enough to move BS8 by about 9%; both sides below are fresh controlled
runs.

| BS | BF16 TPOT | Hybrid TPOT | TPOT delta | BF16 output tok/s | Hybrid output tok/s | Throughput delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.199005 ms | 1.971075 ms | -10.37% | 298.429 | 320.689 | +7.46% |
| 8 | 10.254591 ms | 9.678340 ms | -5.62% | 647.033 | 674.462 | +4.24% |
| 32 | 29.131994 ms | 28.057508 ms | -3.69% | 904.883 | 932.146 | +3.01% |
| 64 | 43.130978 ms | 42.716775 ms | -0.96% | 1000.822 | 1011.790 | +1.10% |
| 96 | 44.575357 ms | 43.648652 ms | -2.08% | 1000.834 | 1019.751 | +1.89% |

| BS | BF16 ITL | Hybrid ITL | ITL delta | BF16 TTFT | Hybrid TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.994450 ms | 5.563644 ms | -7.19% | 232.252 ms | 231.053 ms |
| 8 | 28.099605 ms | 26.343830 ms | -6.25% | 403.516 ms | 419.584 ms |
| 32 | 79.672957 ms | 76.508108 ms | -3.97% | 1185.304 ms | 1196.962 ms |
| 64 | 117.165091 ms | 116.301478 ms | -0.74% | 3985.270 ms | 3873.780 ms |
| 96 | 120.692313 ms | 118.454061 ms | -1.85% | 9669.471 ms | 9507.427 ms |

The large-M dispatcher decision was measured explicitly:

| BS | BF16 TPOT | Tiled at all M | Final scalar at M>=64 |
| ---: | ---: | ---: | ---: |
| 64 | 43.130978 ms | 43.646811 ms | 42.716775 ms |
| 96 | 44.575357 ms | 45.454200 ms | 43.648652 ms |

This explains why a faster isolated GEMM/refinement did not initially lower
overall TPOT: a 4-12 us local refinement saving changed the generated path,
whereas each high-BS MTP iteration is about 120 ms. Preserving the stable
scalar reduction produces a much larger end-to-end improvement than keeping
the locally fastest kernel.

## Hidden costs and feasibility

- Original BF16 local head: 485.00 MiB per GPU.
- Persistent MXFP8 values: 242.50 MiB per GPU.
- Persistent MXFP8 E8M0 scales: 7.58 MiB per GPU.
- Total auxiliary MXFP8 copy: 250.08 MiB per GPU per prepared head.
- Model-load conversion: about 1.03 seconds for the actual shard.
- CUTLASS autotuning: about 0.64-0.73 seconds per rank on a cold load.
- CUDA Graph capture for all validated shapes: about 0.55 GiB graph memory.

The previous gathered-refinement implementation also had an M-proportional
transient allocation (144 MiB at M=288 and 256 MiB at M=512); the indexed
kernel removes it. The unique-token table costs at most
`max_num_reqs * min(max_model_len, vocab_size) * 4` bytes and is updated only
for presence-penalty rows.

The requested BS1-96 range is feasible and beneficial, but the margin at BS64
is only about 1%. The main deployment tradeoff is the persistent MXFP8 copy;
the next worthwhile optimization would require a genuinely fused batched
selection/epilogue, not row-wise CUB or a standalone hierarchical pass.

## Enablement

The feature remains disabled by default:

```bash
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_HYBRID_MXFP8_LM_HEAD=1
VLLM_HYBRID_MXFP8_LM_HEAD_CANDIDATES=128
VLLM_HYBRID_MXFP8_LM_HEAD_MAX_ROWS=512
VLLM_HYBRID_MXFP8_LM_HEAD_USE_FLASHINFER_TOPK=1
```

The validated Qwen3.5 service additionally uses `VLLM_USE_DEEP_GEMM=0` so the
model backend matches the historical CUTLASS/Triton configuration.

## Verification status

Focused CPU and GPU tests cover shape-generic MXFP8 preparation, C=128
candidate recall, scalar/tiled indexed BF16 refinement, sparse presence state,
hybrid-disabled fallback, and V2 compact greedy/top-k sampling. The real-shape
accuracy scans, sparse-penalty scan, CUB comparison, hierarchical Top-K
benchmark, and all serving A/B measurements used GPUs 4/5. Ruff,
Markdownlint, SPDX, bytecode, and `git diff --check` pass. The final focused
CPU run reports 33 passed and 10 skipped; the focused GPU run reports 39
passed.
