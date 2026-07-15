# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate exact partition-local Top-K followed by a global merge."""

from __future__ import annotations

import argparse
from functools import partial

import torch
from flashinfer import top_k as flashinfer_top_k


def _hierarchical_topk(
    logits: torch.Tensor,
    candidates: int,
    partitions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, vocab_size = logits.shape
    if vocab_size % partitions:
        raise ValueError("vocab_size must be divisible by partitions")
    partition_size = vocab_size // partitions
    partition_logits = logits.view(rows * partitions, partition_size)
    local_values, local_indices = flashinfer_top_k(
        partition_logits,
        candidates,
        sorted=False,
    )
    offsets = torch.arange(
        partitions, dtype=local_indices.dtype, device=logits.device
    ).mul_(partition_size)
    local_indices = local_indices.view(rows, partitions, candidates)
    local_indices = local_indices + offsets.view(1, partitions, 1)
    merged_values = local_values.view(rows, partitions * candidates)
    merged_indices = local_indices.view(rows, partitions * candidates)
    final_values, final_positions = flashinfer_top_k(
        merged_values,
        candidates,
        sorted=False,
    )
    final_indices = merged_indices.gather(1, final_positions)
    return final_values, final_indices


def _fp32_copy_topk(
    logits: torch.Tensor,
    candidates: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return flashinfer_top_k(logits.to(torch.float32), candidates, sorted=False)


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
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 32, 96, 288])
    parser.add_argument("--vocab-size", type=int, default=124160)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--partitions", nargs="+", type=int, default=[5, 10, 20, 97])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260806)
    for rows in args.rows:
        logits = torch.randn(
            (rows, args.vocab_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        direct_values, _ = flashinfer_top_k(logits, args.candidates, sorted=False)
        direct_us = _time_cuda_graph(
            partial(flashinfer_top_k, logits, args.candidates, sorted=False),
            args.warmup,
            args.iterations,
        )
        fp32_copy_us = _time_cuda_graph(
            partial(_fp32_copy_topk, logits, args.candidates),
            args.warmup,
            args.iterations,
        )
        print(
            f"M={rows:3d} direct_bf16={direct_us:8.3f}us "
            f"copy_fp32_select={fp32_copy_us:8.3f}us "
            f"saved={fp32_copy_us - direct_us:8.3f}us"
        )
        direct_sorted = direct_values.sort(dim=-1).values
        for partitions in args.partitions:
            if args.vocab_size % partitions:
                continue
            values, _ = _hierarchical_topk(logits, args.candidates, partitions)
            torch.testing.assert_close(
                values.sort(dim=-1).values,
                direct_sorted,
            )
            hierarchical_us = _time_cuda_graph(
                partial(
                    _hierarchical_topk,
                    logits,
                    args.candidates,
                    partitions,
                ),
                args.warmup,
                args.iterations,
            )
            print(
                f"  partitions={partitions:3d} "
                f"segment={args.vocab_size // partitions:6d} "
                f"hierarchical={hierarchical_us:8.3f}us "
                f"ratio={hierarchical_us / direct_us:6.2f}x"
            )


if __name__ == "__main__":
    main()
