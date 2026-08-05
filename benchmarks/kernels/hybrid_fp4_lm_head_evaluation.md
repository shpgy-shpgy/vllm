# Hybrid NVFP4/BF16 lm-head evaluation (C=128)

Date: 2026-08-05

## Conclusion

The production path has been changed from FP8 coarse search to NVFP4 E2M1
coarse search with BF16 candidate refinement.  On the authorized GPU 4/5 TP2
service, the same non-eager V2/MTP2 CUDA-Graph workload reduced mean TPOT from
2.2593 ms to 2.0579 ms and mean streamed ITL from 6.0027 ms to 5.6939 ms.

The gain is real, but the path is not generally bit-exact.  C=128 recovered the
full-BF16 greedy winner in every audited real MTP row.  Bounded top-k had a very
small FP4 candidate-miss rate and additional near-boundary differences caused
by the different BF16 accumulation order in candidate `bmm`.  The feature
therefore remains opt-in and full-distribution sampling continues to use the
original BF16 lm-head.

## Implementation

Each TP rank retains the original BF16 lm-head and creates one packed NVFP4
weight with per-16 FP8 scale factors.  At runtime it:

1. quantizes the BF16 hidden states to NVFP4;
2. runs FlashInfer-CUTLASS FP4 matrix multiplication over the local vocabulary;
3. selects 128 coarse candidates with FlashInfer radix top-k;
4. gathers those rows from the original BF16 weight and recomputes their logits
   with BF16 `bmm`; and
5. communicates only compact winners/candidates across TP ranks.

The implementation dispatches from runtime dimensions and uses CUTLASS padding
and backend selection.  It is not specialized for one vocabulary or hidden
shape.  Unsupported dtype, device, layout, shape, top-k, vocabulary layout, or
workspace size falls back to the original implementation.

## Service configuration

- Model: `Qwen3.5-35B-A3B-FP8`
- GPUs: 4/5, TP=2
- Engine: V2 Model Runner
- Speculative decoding: MTP2, local argmax
- Candidate count: 128 per TP rank
- Input calibration amax: 48
- Execution: `enforce_eager=False`, CUDA Graph `FULL_AND_PIECEWISE`, capture
  sizes 1/2/3
- Attention: FlashInfer
- All-reduce: CUSTOM/PyNCCL
- Workload: 10 sequential requests per run, random 3000-token input,
  1000-token output, batch size 1, temperature 0, seed 20260729
- Measurement: one warmup run followed by three formal runs for each mode

No eager-mode latency is used in this report.  FP4 and BF16 require separate
service starts, so the system test is same-configuration repeated A/B rather
than an in-process kernel toggle.  The isolated kernel comparison below is
interleaved A-B-B-A.

## End-to-end result

Values are means of the three post-warmup runs.

| Metric | Native BF16 | NVFP4/BF16 | Change |
| --- | ---: | ---: | ---: |
| mean TPOT | 2.2593 ms | 2.0579 ms | -0.2014 ms (-8.91%) |
| median TPOT | 2.3066 ms | 1.9939 ms | -0.3127 ms (-13.56%) |
| mean streamed ITL | 6.0027 ms | 5.6939 ms | -0.3089 ms (-5.15%) |
| median streamed ITL | 6.0051 ms | 5.7035 ms | -0.3016 ms (-5.02%) |
| output throughput | 400.21 tok/s | 436.39 tok/s | +9.04% |
| request duration | 24.9870 s | 22.9107 s | -8.31% |
| mean TTFT | 241.39 ms | 235.32 ms | -2.51% |

The FP4 TPOT standard deviation was 0.00123 ms across the three runs.  The
contemporaneous 2.2593 ms BF16 result is also consistent with the previously
observed approximately 2.22 ms production range.

MTP acceptance and the generated trajectory can change after any approximate
logit path, so TPOT and throughput do not isolate lm-head cost.  Streamed ITL
is the more conservative system result.  Per-iteration server logs were also
approximately 5.5 ms for FP4 versus 5.8 ms for BF16 during steady decode.

## Isolated FP8-to-FP4 result

For the actual TP-local shape `N=124160, K=2048`, C=128 refinement and
non-eager CUDA-Graph replay:

| Captured rows | FP8 complete path | NVFP4 complete path | Saving |
| --- | ---: | ---: | ---: |
| 1 | 179.82 us | 158.21 us | 21.61 us (-12.02%) |
| 2 | 181.11 us | 159.86 us | 21.25 us (-11.73%) |
| 3 | 182.80 us | 160.94 us | 21.87 us (-11.96%) |

For one row, coarse projection alone changed from about 156.00 us to 133.66 us.
Direct CUTLASS and the experimental b12x backend were slower for this shape
than the FlashInfer-CUTLASS wrapper selected by vLLM.  FP4 therefore saves
about 12%, not 50%, from the already optimized FP8 coarse/refine path.  Hidden
quantization, top-k, indexed BF16 weight gather, refinement, and launches do
not shrink with the tensor-core datatype.

## Precision audit

### Greedy and MTP

Instrumented non-eager CUDA-Graph services computed the full-BF16 reference in
parallel with FP4 coarse/refine.  The audit computation was excluded from all
performance measurements.

- At input amax 16, each TP rank audited 3364 real greedy/MTP rows: candidate
  misses 0, final token mismatches 0.
- At input amax 48, each TP rank audited another 3572 real greedy/MTP rows:
  candidate misses 0, final token mismatches 0.
- Random hidden-state tests using the actual lm-head weight recovered the
  full-BF16 winner for 4096/4096 rows.

Thus the audited greedy result is exact at C=128.  It is an empirical result,
not a mathematical guarantee for unseen hidden states.

### Bounded top-k

With input amax 48 and top-k 40, each rank audited 5350 real rows, or 214,000
full-BF16 local top-k token entries:

| TP rank | BF16 top-40 tokens absent from FP4 C=128 | Rows with an absent token | Refined top-40 set mismatch rows |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0 | 1077 |
| 1 | 22 (0.01028%) | 20 (0.374%) | 1194 |

The last column includes rows whose full-BF16 top-40 candidates were all
present.  Those differences occur at the top-k boundary because a 128-wide
BF16 `bmm` and the full-vocabulary BF16 GEMM do not use the same accumulation
order.  Top-k sampling is therefore covered functionally but is approximate,
not bit-exact.

At input amax 16, top-k 20 missed 2 of 105,800 BF16 local top-k entries on one
rank and 0 on the other.  The amax-16 run observed a hidden maximum of 38.75
and clipped 0.1247% of hidden elements.  Raising the default to 48 observed a
maximum of 44.5 and zero clipped elements; the remaining misses are FP4 rank
error rather than input clipping.

### Presence penalty and other sampling modes

| Path | NVFP4 coarse/refine | Semantics |
| --- | --- | --- |
| Greedy / argmax | Yes | Exact in the audited rows |
| Bounded top-k, then top-p | Yes | Approximate as quantified above |
| Presence-only penalty | Yes | Applied before coarse selection and again during BF16 refinement |
| Full-vocabulary random sampling | No | Full BF16 fallback preserves the complete distribution |
| Frequency/repetition penalty, arbitrary logit bias | No | Full BF16 fallback preserves semantics |

In the MTP2 presence-only test, the first non-speculative row of each request
used the hybrid path and had 0 candidate/refinement misses across 20 rows per
rank.  Speculative validation with presence penalty deliberately falls back to
full BF16, so the scenario is correct but receives little lm-head speedup.

An exact full-distribution sampler cannot discard all but 128 tokens before
sampling because every finite-probability token must remain eligible.  The
existing local Gumbel-max path reduces TP communication but intentionally does
not replace the BF16 lm-head computation.

Increasing C preserves or increases candidate recall for the same deterministic
coarse logits.  BF16 refinement can recover the native result only when the
relevant BF16 token is in the candidate set; it cannot undo a coarse candidate
miss or guarantee bit-identical GEMM accumulation.

## Memory and hidden costs

- Packed NVFP4 weight plus FP8 scales: 136.41 MiB per GPU in addition to the
  original BF16 head.
- The prior block-FP8 copy used 242.56 MiB; FP4 saves 106.15 MiB, or 43.8%,
  from that auxiliary allocation.
- Actual M=3 candidate refinement materializes approximately 1.5 MiB of BF16
  selected weights per rank, plus coarse logits and quantization workspaces.
- Runtime still pays for hidden-state quantization, a full local-vocabulary
  coarse GEMM, radix top-k, indexed gather, BF16 `bmm`, compact TP reduction,
  and multiple kernel launches.
- Online weight quantization occurs once at model preparation.  The tested
  FlashInfer-CUTLASS FP4 shape autotuning added roughly 95 seconds on service
  starts even when the general compilation cache was warm.  This is the main
  deployment hidden cost and should be cached or pre-tuned before default-on
  use.
- MTP target validation and unsupported sampling/penalty paths retain their
  full-BF16 cost, limiting the end-to-end fraction that FP4 can improve.

MTP draft/target lm-head sharing leaves one BF16 head and one NVFP4 copy.  The
discarded draft head releases its auxiliary FP4 buffers instead of retaining a
vocabulary-sized duplicate allocation.

## Shape dispatch and fallback

NVFP4 preparation requires FlashInfer-CUTLASS FP4 support, contiguous CUDA
BF16 weights, and a hidden dimension divisible by 16.  Runtime checks cover
batch rank, hidden dimension, active vocabulary, top-k versus C, device,
dtype, contiguity, and refinement workspace.  CUTLASS padding handles local
vocabulary alignment.  Any failed check returns to the original BF16 path.

The feature is disabled by default.  The evaluated setting is:

```bash
VLLM_HYBRID_FP4_LM_HEAD=1
VLLM_HYBRID_FP4_LM_HEAD_CANDIDATES=128
VLLM_HYBRID_FP4_LM_HEAD_INPUT_AMAX=48
```

## Verification

The final focused suite passes 29/29 tests, including the CUDA FP4/argmax tests
on GPU 4.  Python bytecode compilation and `git diff --check` also pass.  Ruff
was not available in the repository virtual environment and was not installed
as an unrequested dependency.
