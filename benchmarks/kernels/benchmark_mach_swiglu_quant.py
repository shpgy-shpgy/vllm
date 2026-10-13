# SPDX-License-Identifier: Apache-2.0
"""Compare the upstream SwiGLU quantizer against compiled producer + FlashInfer.

Kernel-only CUDA Graph timing; this does not predict serving throughput.
Run on SM120 with CUDA_VISIBLE_DEVICES selecting an idle GPU.
"""

import json

import torch
from flashinfer import mxfp8_quantize
from triton.testing import do_bench_cudagraph

from vllm.mach.mxfp6.swiglu_quant import quantize


def producer(x):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


@torch.inference_mode()
def main():
    from mxfp6 import load_library

    load_library()
    for model, width in (("2B", 12288), ("4B", 18432)):
        for rows in (1, 4, 16, 32, 64):
            x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
            compiled = torch.compile(producer, fullgraph=True)

            def reference(compiled=compiled, x=x):
                return mxfp8_quantize(compiled(x), is_sf_swizzled_layout=True)

            reference()
            quantize(x)
            ref_ms = do_bench_cudagraph(reference, rep=200)
            fused_ms = do_bench_cudagraph(lambda x=x: quantize(x), rep=200)
            native_ms = do_bench_cudagraph(
                lambda x=x: torch.ops.mxfp6.silu_and_mul_mxfp8_pdl(x), rep=200
            )
            print(
                json.dumps(
                    dict(
                        model=model,
                        rows=rows,
                        reference_us=ref_ms * 1000,
                        fused_us=fused_ms * 1000,
                        existing_cuda_us=native_ms * 1000,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
