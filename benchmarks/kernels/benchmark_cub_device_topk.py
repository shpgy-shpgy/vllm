# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare CUB AIR DeviceTopK with row-wise FlashInfer and PyTorch top-k."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from statistics import median

import torch
from torch.utils.cpp_extension import load


def _time_cuda_graph(
    function: Callable[[], object],
    *,
    steps: int,
    warmup: int,
    trials: int,
) -> float:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = function()
    torch.accelerator.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()

    samples = []
    for _ in range(trials):
        start = torch.Event(enable_timing=True)
        end = torch.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / steps)
    del outputs
    return median(samples)


def _same_values(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.sort(dim=-1).values, right.sort(dim=-1).values))


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", required=True)
    parser.add_argument("--width", type=int, default=124160)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    cccl_root = root / ".venv/lib/python3.12/site-packages/flashinfer/data/cccl"
    module = load(
        name="vllm_benchmark_cub_device_topk",
        sources=[str(Path(__file__).with_name("cub_device_topk_extension.cu"))],
        extra_include_paths=[
            str(cccl_root / "cub"),
            str(cccl_root / "thrust"),
            str(cccl_root / "libcudacxx/include"),
        ],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )

    from flashinfer import top_k as flashinfer_top_k

    for rows in args.rows:
        logits = torch.randn(
            (rows, args.width),
            dtype=torch.bfloat16,
            device="cuda",
        )
        reference_values, _ = torch.topk(
            logits,
            args.k,
            dim=-1,
            sorted=False,
        )
        cub_values, cub_indices = module.topk(logits, args.k)
        flash_values, flash_indices = flashinfer_top_k(
            logits,
            args.k,
            sorted=False,
        )
        torch.accelerator.synchronize()
        if not _same_values(cub_values, reference_values):
            raise AssertionError(f"CUB value mismatch for M={rows}")
        if not _same_values(flash_values, reference_values):
            raise AssertionError(f"FlashInfer value mismatch for M={rows}")
        del cub_values, cub_indices, flash_values, flash_indices

        timings = {
            "cub_air": _time_cuda_graph(
                lambda logits=logits: module.topk(logits, args.k),
                steps=args.steps,
                warmup=args.warmup,
                trials=args.trials,
            ),
            "flashinfer": _time_cuda_graph(
                lambda logits=logits: flashinfer_top_k(logits, args.k, sorted=False),
                steps=args.steps,
                warmup=args.warmup,
                trials=args.trials,
            ),
            "torch": _time_cuda_graph(
                lambda logits=logits: torch.topk(logits, args.k, dim=-1, sorted=False),
                steps=args.steps,
                warmup=args.warmup,
                trials=args.trials,
            ),
        }
        print(
            f"m={rows} n={args.width} k={args.k} "
            + " ".join(f"{name}={value:.3f}us" for name, value in timings.items()),
            flush=True,
        )

    return 0


if __name__ == "__main__":
    os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0a"
    raise SystemExit(main())
