# SPDX-License-Identifier: Apache-2.0
"""Compare GEMM consumers with identical quantized inputs; not serving TPS."""

import json

import torch
from flashinfer import autotune, mxfp8_quantize

from vllm.mach.mxfp6.two_b_native import gemm
from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import (
    load_mxfp6_sm120_mxfp8_libraries,
)
from vllm.utils.flashinfer import mm_mxfp8


def graph_ms(fn):
    # Native workspaces are stream-owned: warm the actual capture stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(64):
            fn()
    graph.replay()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(20):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / (64 * 20)


@torch.inference_mode()
def main():
    load_mxfp6_sm120_mxfp8_libraries()
    # 2B QKVZ, full-attention QKV, gate/up, down, and output projections.
    for n, k in ((8192, 2048), (5120, 2048), (12288, 2048), (2048, 6144), (2048, 2048)):
        weight, ws = mxfp8_quantize(
            torch.randn((n, k), device="cuda", dtype=torch.bfloat16),
            is_sf_swizzled_layout=True,
        )
        for rows in (1, 4, 16, 32, 64, 2048):
            values, sf = mxfp8_quantize(
                torch.randn((rows, k), device="cuda", dtype=torch.bfloat16),
                is_sf_swizzled_layout=True,
            )

            def reference(values=values, weight=weight, sf=sf, ws=ws):
                return mm_mxfp8(
                    values,
                    weight.t(),
                    sf,
                    ws,
                    out_dtype=torch.bfloat16,
                    backend="cutlass",
                )

            def native(values=values, weight=weight, sf=sf, ws=ws):
                return gemm(values, weight, sf, ws)

            with autotune(tune_mode=True):
                reference()
            native()
            fi_ms = graph_ms(reference)
            native_ms = graph_ms(native)
            print(
                json.dumps(
                    dict(
                        m=rows,
                        n=n,
                        k=k,
                        flashinfer_us=fi_ms * 1000,
                        native_us=native_ms * 1000,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
