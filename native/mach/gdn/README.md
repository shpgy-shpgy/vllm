# Precompiled persistent GDN

Build from this complete vLLM checkout with its validated virtual environment:

```bash
env -u CPATH CUDA_VISIBLE_DEVICES=4,5 MAX_JOBS=2 \
  uv build --python .venv/bin/python --no-build-isolation --wheel \
  --out-dir /data/luyufan/vllm_template native/mach/gdn
```

Requires Python 3.12, PyTorch 2.13.0+cu130, CUDA toolkit 13.0,
FlashInfer 0.6.18 and apache-tvm-ffi 0.1.11. The builder compiles the
existing CUDA source without changing its flags or algorithm. Do not put
other CUDA runtime headers ahead of the toolkit headers using CPATH.

The wheel includes four geometry-specific SM120 modules: Dense27B and
MoE35B, each with FP16 and FP32 recurrent state. Batch size remains dynamic.
The manifest records geometry, flags, source digest, compiler version,
binary digests and runtime ABI. It contains no build intermediates.

The runtime loader prefers this wheel and never silently recompiles an
incompatible binary. Set `VLLM_MACH_GDN_REQUIRE_PRECOMPILED=1` to also reject
a missing wheel. This setting does not enable GDN or change baseline dispatch.
Other Triton, CuTe and FlashInfer kernels may still compile at runtime.
