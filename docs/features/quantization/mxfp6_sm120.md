# Native MXFP6 on SM120

The optional `mxfp6-sm120` backend executes Quark MXFP6 E3M2 weights with
dynamic MXFP8 E4M3 activations on SM120 GPUs. This integration is adapted from
vllm-mach and does not require the Mach plugin. Install the matching
`mxfp6-sm120` wheel in the same environment as vLLM; version 0.2.1 is the
validated runtime. Its native extension must match the installed PyTorch ABI.

The selector prefers `Mxfp6Sm120LinearKernel` for this format. Existing
weight-only and other MXFP6 formats retain their emulation fallback. An MXFP8
activation checkpoint requires the native backend; an absent or incompatible
wheel produces a kernel-selection error rather than silently dropping
activation quantization.

Weights remain packed at six bits. Loading converts checkpoint scale tensors
into the backend's packed layout. Workspace planning runs before memory
profiling, and each CUDA Graph capture stream is warmed before capture. Both
GPU model runners include these hooks. Loading a model without native MXFP6
layers does not load the optional runtime during warmup.

For example, with a compatible local checkpoint:

```bash
VLLM_PLUGINS='' MXFP6_AUTOTUNE=off CUDA_VISIBLE_DEVICES=0,1 \
  .venv/bin/vllm serve /path/to/Qwen3.8-27B-MXFP6 \
  --quantization quark --dtype bfloat16 --tensor-parallel-size 2 \
  --max-model-len 8192 --max-num-seqs 32 --attention-backend TRITON_ATTN \
  --no-enable-prefix-caching \
  --compilation-config '{"mode":"NONE","cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,24,32]}'
```

This preserves the checkpoint's existing embedding and output-head formats;
it does not introduce embedding or output-head requantization.

## Native MXFP8 W8A8 on SM120

For a pure `compressed-tensors` MXFP8 checkpoint, select the native
`mxfp6-sm120` W8A8 path explicitly with `--linear-backend mxfp6-sm120`.
This uses `mxfp6.mxfp8` for dynamic activation quantization and the MXFP8 GEMM
for medium/large decode M. To avoid the fixed launch and quantization overhead
at very small M, M<=16 is dispatched through the existing FlashInfer
`b12x-decode` path; `--attention-backend` remains independent and can still be
set to `FLASHINFER`.

```bash
MXFP6_LIBRARY_PATH=/path/to/site-packages/mxfp6/mxfp6_torch.so \
MXFP8_LIBRARY_PATH=/path/to/site-packages/mxfp6/mxfp8_torch.so \
CUDA_VISIBLE_DEVICES=5 python3 -m vllm.entrypoints.openai.api_server \
  --model /data/models/Qwen3.5-4B-MXFP8 \
  --attention-backend FLASHINFER \
  --linear-backend mxfp6-sm120
```

The native W8A8 implementation requires SM120 and positive `N/K` dimensions
that are multiples of 128. The Qwen3.5-4B MXFP8 checkpoint's quantized linear
weights satisfy this constraint; BF16-only layers continue through their
ordinary path.

## Manual AllReduce and RMSNorm fusion

Set `VLLM_QWEN3_5_FUSED_AR_NORM=1` to delay row-parallel reductions into the
following residual-add/GemmaRMSNorm. This includes the final model norm. The
opt-in path is limited to the validated dense Qwen3.5 geometry (hidden size
5120), BF16, Quark, SM120, TP2, PP1, and `mode=NONE`. LoRA, speculative
execution, layer scaling, and auxiliary hidden-state output are not supported
by this path. Other configurations retain the original forward path.

This fusion returns ordinary BF16 activations; it does not enable packed
MXFP8 norm/MLP fusion. Use `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` for the
validated single-node IPC setup. The existing helper falls back to explicit
AllReduce plus RMSNorm when its fused backend is unavailable.

On the tested RTX 5090 environment, FlashInfer 0.6.18 needs the local IPC
allocation compatibility patch in
[`tools/patches/flashinfer-0.6.18-local-ipc.patch`](../../../tools/patches/flashinfer-0.6.18-local-ipc.patch).
It removes the GPU-direct-RDMA requirement only for local TRT-LLM IPC buffers.
This external dependency patch is provided for review/reproduction, not applied
at vLLM import time. The test machine already has it installed. Do not mix a
FlashInfer JIT-cache wheel from a different FlashInfer version.

Local validation on two RTX 5090s (32 requests, 3000 input tokens, 112 generated
per request, last 64 steady decode steps, median of per-step TP-rank maxima)
measured model forward at 17.371 ms before manual fusion and 13.874 ms after it.
These timings exclude the output head, sampling, prefill and HTTP overhead.
The TP2 M=32/K=5120 kernel test asserts that the FlashInfer fast path is active
and checks its output/residual against explicit AllReduce and GemmaRMSNorm.
Eight short text regression prompts passed after Unicode normalization; this
small check is not a general model-quality evaluation.
