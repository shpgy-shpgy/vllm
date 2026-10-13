# SPDX-License-Identifier: Apache-2.0
"""Measure complete W4 cycles against stock FLA; not end-to-end serving TPS."""

import argparse
import json
import statistics
from functools import partial

import torch

from vllm.mach.mxfp6.gdn import two_b_ordered as ordered
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode as stock,
)


def graph_us(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(8):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        # A multiple of four includes both deferred steps and dense writeback.
        for _ in range(64):
            fn()
    timings = []
    for _ in range(7):
        graph.replay()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / (64 * 20))
    return statistics.median(timings)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=int, default=256)
    args = parser.parse_args()
    if args.pages < 162:
        parser.error("--pages must be >=162")
    if torch.cuda.get_device_capability() != (12, 0):
        parser.error("requires SM120")
    torch.manual_seed(71)
    for rows in (32, 48, 64, 96, 128, 160):
        base = torch.randn(args.pages, 16, 128, 128, device="cuda")
        reference_base = base.clone()
        scratch = ordered.DeferredScratch(
            **{
                name: torch.zeros(
                    shape,
                    device="cuda",
                    dtype=torch.int32 if name == "age" else torch.float32,
                )
                for name, shape in ordered.required_shapes(args.pages, 4).items()
            }
        )
        scratch.prefix.fill_(1)
        qkv = torch.randn(rows, 8192, device="cuda", dtype=torch.bfloat16)[:, :6144]
        ab = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
        a, b = ab[:, :16], ab[:, 16:]
        alog = torch.randn(16, device="cuda") - 2
        bias = torch.randn(16, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(rows, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(1, rows + 1, dtype=torch.int32, device="cuda")

        reference = partial(
            stock, qkv, a, b, alog, bias, 128**-0.5, reference_base, out, ids, True
        )
        candidate = partial(
            ordered.decode, qkv, a, b, alog, bias, base, out, ids, scratch
        )

        stock_us = graph_us(reference)
        ordered_us = graph_us(candidate)
        print(
            json.dumps(
                dict(
                    rows=rows,
                    stock_us=stock_us,
                    ordered_us=ordered_us,
                    speedup=stock_us / ordered_us,
                    scratch_bytes_per_layer=sum(
                        t.numel() * t.element_size() for t in vars(scratch).values()
                    ),
                    pages=args.pages,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
