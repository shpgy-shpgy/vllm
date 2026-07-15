# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare dense-count and sparse-index presence-penalty kernels."""

from __future__ import annotations

import argparse
from functools import partial

import torch

from vllm.model_executor.layers.presence_penalty_triton import (
    apply_presence_penalty_from_counts,
    apply_sparse_presence_penalty,
)


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
    parser.add_argument("--local-vocab-size", type=int, default=62080)
    parser.add_argument("--global-vocab-size", type=int, default=124160)
    parser.add_argument("--max-requests", type=int, default=96)
    parser.add_argument("--history", nargs="+", type=int, default=[128, 1000])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260806)
    for history in args.history:
        unique_ids = torch.stack(
            [
                torch.randperm(
                    args.local_vocab_size,
                    dtype=torch.int64,
                    device=device,
                    generator=generator,
                )[:history]
                for _ in range(args.max_requests)
            ]
        ).to(torch.int32)
        num_unique = torch.full(
            (args.max_requests,), history, dtype=torch.int32, device=device
        )
        counts = torch.zeros(
            (args.max_requests, args.global_vocab_size),
            dtype=torch.int32,
            device=device,
        )
        counts.scatter_(1, unique_ids.to(torch.int64), 1)

        for rows in args.rows:
            logits = torch.randn(
                (rows, args.local_vocab_size),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            request_indices = torch.arange(
                rows, dtype=torch.int32, device=device
            ).remainder_(args.max_requests)
            penalties = torch.ones(rows, dtype=torch.float32, device=device)

            dense_logits = logits.clone()
            sparse_logits = logits.clone()
            dense = partial(
                apply_presence_penalty_from_counts,
                dense_logits,
                counts,
                request_indices,
                penalties,
                org_vocab_start=0,
                num_org_elements=args.local_vocab_size,
                num_org_elements_padded=args.local_vocab_size,
                added_vocab_start=args.global_vocab_size,
                num_added_elements=0,
            )
            sparse = partial(
                apply_sparse_presence_penalty,
                sparse_logits,
                unique_ids,
                num_unique,
                request_indices,
                penalties,
                org_vocab_start=0,
                num_org_elements=args.local_vocab_size,
                num_org_elements_padded=args.local_vocab_size,
                added_vocab_start=args.global_vocab_size,
                num_added_elements=0,
            )

            dense_once = logits.clone()
            sparse_once = logits.clone()
            apply_presence_penalty_from_counts(
                dense_once,
                counts,
                request_indices,
                penalties,
                org_vocab_start=0,
                num_org_elements=args.local_vocab_size,
                num_org_elements_padded=args.local_vocab_size,
                added_vocab_start=args.global_vocab_size,
                num_added_elements=0,
            )
            apply_sparse_presence_penalty(
                sparse_once,
                unique_ids,
                num_unique,
                request_indices,
                penalties,
                org_vocab_start=0,
                num_org_elements=args.local_vocab_size,
                num_org_elements_padded=args.local_vocab_size,
                added_vocab_start=args.global_vocab_size,
                num_added_elements=0,
            )
            torch.testing.assert_close(sparse_once, dense_once)

            dense_us = _time_cuda_graph(dense, args.warmup, args.iterations)
            sparse_us = _time_cuda_graph(sparse, args.warmup, args.iterations)
            print(
                f"history={history:4d} M={rows:3d} "
                f"dense={dense_us:8.3f}us sparse={sparse_us:8.3f}us "
                f"speedup={dense_us / sparse_us:6.2f}x"
            )


if __name__ == "__main__":
    main()
