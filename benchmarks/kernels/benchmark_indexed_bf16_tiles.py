# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tune candidate tiling for indexed BF16 lm-head refinement."""

from __future__ import annotations

import argparse
from functools import partial

import torch

from vllm.model_executor.layers.hybrid_mxfp8_lm_head import indexed_bf16_dot


def _time_cuda_graph(fn, warmup: int, iterations: int) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    for _ in range(warmup):
        graph.replay()
    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", nargs="+", type=int, default=[1, 32, 64, 96, 128, 192, 288]
    )
    parser.add_argument("--vocab-size", type=int, default=62080)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--tiles", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--warps", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260806)
    weight = torch.randn(
        (args.vocab_size, args.hidden_size),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    for rows in args.rows:
        hidden = torch.randn(
            (rows, args.hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        indices = torch.randint(
            0,
            args.vocab_size,
            (rows, args.candidates),
            dtype=torch.int64,
            device=device,
            generator=generator,
        )
        reference = indexed_bf16_dot(hidden, weight, indices, candidate_tile=1)
        timings: dict[tuple[int, int], float] = {}
        for tile in args.tiles:
            for num_warps in args.warps:
                output = indexed_bf16_dot(
                    hidden,
                    weight,
                    indices,
                    candidate_tile=tile,
                    num_warps=num_warps,
                )
                torch.testing.assert_close(output.float(), reference.float())
                timings[(tile, num_warps)] = _time_cuda_graph(
                    partial(
                        indexed_bf16_dot,
                        hidden,
                        weight,
                        indices,
                        candidate_tile=tile,
                        num_warps=num_warps,
                    ),
                    args.warmup,
                    args.iterations,
                )
        best_config = min(timings, key=timings.__getitem__)
        timing_text = " ".join(
            f"C{tile}W{num_warps}={timings[(tile, num_warps)]:8.3f}us"
            for tile in args.tiles
            for num_warps in args.warps
        )
        print(f"M={rows:3d} {timing_text} best=C{best_config[0]}W{best_config[1]}")


if __name__ == "__main__":
    main()
