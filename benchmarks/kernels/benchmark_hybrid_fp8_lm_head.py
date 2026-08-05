# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark shape-generic FP8-coarse/BF16-refined lm-head top-k.

This is an evaluation harness, not a serving path. It intentionally composes
existing vLLM and PyTorch kernels so their intermediate allocations and launch
overheads remain visible.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.model_executor.layers.argmax_triton import (
    indexed_argmax_triton,
    local_argmax_triton,
)
from vllm.model_executor.layers.hybrid_fp4_lm_head import (
    select_lm_head_candidates,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.utils.deep_gemm import per_block_cast_to_fp8

_EMBEDDING_WEIGHT_KEYS = (
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
)


@dataclass
class Fp8Weight:
    weight: torch.Tensor
    scale: torch.Tensor
    scheme: str


@dataclass
class CandidateResult:
    values: torch.Tensor
    indices: torch.Tensor
    coarse_indices: torch.Tensor


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _resolve_weight(model_dir: Path) -> tuple[Path, str]:
    index_paths = sorted(model_dir.glob("*safetensors.index.json"))
    if len(index_paths) != 1:
        raise ValueError(
            f"expected one safetensors index in {model_dir}, got {index_paths}"
        )
    weight_map = json.loads(index_paths[0].read_text(encoding="utf-8"))["weight_map"]
    for key in _EMBEDDING_WEIGHT_KEYS:
        if key in weight_map:
            return model_dir / weight_map[key], key
    raise KeyError(f"none of {_EMBEDDING_WEIGHT_KEYS} is present in {index_paths[0]}")


def _load_weight_shard(
    model_dir: Path,
    *,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, str]:
    shard_path, weight_key = _resolve_weight(model_dir)
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        weight_slice = handle.get_slice(weight_key)
        shape = weight_slice.get_shape()
        if len(shape) != 2:
            raise ValueError(f"lm-head weight must be 2D, got {shape}")
        if shape[0] % tp_size:
            raise ValueError(
                f"vocabulary {shape[0]} is not divisible by TP size {tp_size}"
            )
        rows_per_rank = shape[0] // tp_size
        start = tp_rank * rows_per_rank
        weight = weight_slice[start : start + rows_per_rank]
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 lm-head weight, got {weight.dtype}")
    return weight.contiguous().to(device), start, weight_key


def _quantize_weight(weight: torch.Tensor, scheme: str) -> Fp8Weight:
    if scheme == "block128":
        quantized, scale = per_block_cast_to_fp8(
            weight,
            block_size=[128, 128],
            use_ue8m0=False,
        )
    elif scheme == "channel":
        quantized, scale = ops.scaled_fp8_quant(
            weight,
            use_per_token_if_dynamic=True,
        )
    else:
        raise ValueError(f"unknown FP8 weight scheme: {scheme}")
    return Fp8Weight(quantized, scale, scheme)


def _fp8_linear(hidden: torch.Tensor, weight: Fp8Weight) -> torch.Tensor:
    if weight.scheme == "block128":
        hidden_q, hidden_scale = per_token_group_quant_fp8(
            hidden.contiguous(),
            group_size=128,
            column_major_scales=True,
            use_ue8m0=False,
        )
        weight_scale = weight.scale.T
    else:
        hidden_q, hidden_scale = ops.scaled_fp8_quant(
            hidden.contiguous(),
            use_per_token_if_dynamic=True,
        )
        weight_scale = weight.scale

    return ops.cutlass_scaled_mm(
        hidden_q,
        weight.weight.T,
        scale_a=hidden_scale,
        scale_b=weight_scale,
        out_dtype=torch.bfloat16,
    )


def _make_presence_inputs(
    *,
    batch_size: int,
    vocab_size: int,
    history_tokens: int,
    penalty: float,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if penalty == 0.0 or history_tokens == 0:
        return None, None
    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size, history_tokens),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    penalties = torch.full(
        (batch_size,),
        penalty,
        dtype=torch.float32,
        device=device,
    )
    return token_ids, penalties


def _presence_mask(
    logits: torch.Tensor,
    token_ids: torch.Tensor | None,
    penalties: torch.Tensor | None,
) -> torch.Tensor | None:
    if token_ids is None:
        return None
    assert penalties is not None
    mask = torch.zeros_like(logits)
    values = penalties.to(logits.dtype).unsqueeze(1).expand_as(token_ids)
    mask.scatter_(1, token_ids, values)
    return mask


def _apply_presence_penalty(
    logits: torch.Tensor,
    token_ids: torch.Tensor | None,
    penalties: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    mask = _presence_mask(logits, token_ids, penalties)
    if mask is not None:
        logits = logits - mask
    return logits, mask


def _native_topk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    top_k: int,
    token_ids: torch.Tensor | None,
    penalties: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden, weight)
    if top_k > 1 or token_ids is not None:
        logits = logits.float()
    logits, _ = _apply_presence_penalty(logits, token_ids, penalties)
    if top_k == 1:
        values, indices = local_argmax_triton(
            logits,
            vocab_start=0,
            active_vocab_size=logits.shape[-1],
        )
        return values.unsqueeze(-1), indices.unsqueeze(-1)
    return torch.topk(logits, top_k, dim=-1)


def _hybrid_coarse_logits(
    hidden: torch.Tensor,
    fp8_weight: Fp8Weight,
    *,
    token_ids: torch.Tensor | None,
    penalties: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    coarse_logits = _fp8_linear(hidden, fp8_weight)
    if token_ids is not None:
        coarse_logits = coarse_logits.float()
    return _apply_presence_penalty(
        coarse_logits,
        token_ids,
        penalties,
    )


def _refine_topk(
    hidden: torch.Tensor,
    bf16_weight: torch.Tensor,
    coarse_logits: torch.Tensor,
    penalty_mask: torch.Tensor | None,
    *,
    top_k: int,
    candidates: int,
) -> CandidateResult:
    coarse_indices = select_lm_head_candidates(coarse_logits, candidates)

    # Advanced indexing intentionally materializes [M, candidates, K]. This is
    # the generic PyTorch cost that a fused gathered-dot kernel would remove.
    selected_weight = bf16_weight[coarse_indices]
    exact_logits = torch.bmm(
        selected_weight,
        hidden.unsqueeze(-1),
    ).squeeze(-1)
    if top_k > 1 or penalty_mask is not None:
        exact_logits = exact_logits.float()
    if penalty_mask is not None:
        exact_logits -= penalty_mask.gather(1, coarse_indices)

    if top_k == 1:
        values, indices = indexed_argmax_triton(
            exact_logits,
            coarse_indices,
        )
        values = values.unsqueeze(-1)
        indices = indices.unsqueeze(-1)
    else:
        values, positions = torch.topk(exact_logits, top_k, dim=-1)
        indices = coarse_indices.gather(1, positions)
    return CandidateResult(values, indices, coarse_indices)


def _hybrid_topk(
    hidden: torch.Tensor,
    bf16_weight: torch.Tensor,
    fp8_weight: Fp8Weight,
    *,
    top_k: int,
    candidates: int,
    token_ids: torch.Tensor | None,
    penalties: torch.Tensor | None,
) -> CandidateResult:
    coarse_logits, penalty_mask = _hybrid_coarse_logits(
        hidden,
        fp8_weight,
        token_ids=token_ids,
        penalties=penalties,
    )
    return _refine_topk(
        hidden,
        bf16_weight,
        coarse_logits,
        penalty_mask,
        top_k=top_k,
        candidates=candidates,
    )


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
    return ordered[index]


def _time_cuda_graph(
    function: Callable[[], object],
    *,
    steps: int,
    warmup: int,
    trials: int,
) -> dict[str, float | list[float]]:
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
    return {
        "median_us": median(samples),
        "p95_us": _percentile(samples, 0.95),
        "samples_us": samples,
    }


def _candidate_recall(
    native_indices: torch.Tensor,
    coarse_indices: torch.Tensor,
) -> tuple[int, int]:
    matches = native_indices.unsqueeze(-1) == coarse_indices.unsqueeze(-2)
    hits = matches.any(dim=-1).sum().item()
    return int(hits), native_indices.numel()


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tp-size", type=_positive_int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument(
        "--weight-scheme",
        choices=("block128", "channel"),
        default="block128",
    )
    parser.add_argument("--batch-sizes", type=_positive_int, nargs="+", default=[1])
    parser.add_argument("--top-k", type=_positive_int, nargs="+", default=[1, 64])
    parser.add_argument(
        "--candidates", type=_positive_int, nargs="+", default=[64, 128, 256]
    )
    parser.add_argument("--seeds", type=_positive_int, default=100)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--history-tokens", type=int, default=0)
    parser.add_argument("--steps", type=_positive_int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trials", type=_positive_int, default=5)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not 0 <= args.tp_rank < args.tp_size:
        raise ValueError("tp-rank must be in [0, tp-size)")
    if args.history_tokens < 0:
        raise ValueError("history-tokens must be non-negative")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")

    device = torch.device("cuda")
    weight, vocab_start, weight_key = _load_weight_shard(
        args.model_dir,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        device=device,
    )
    local_vocab, hidden_size = weight.shape
    if hidden_size % 128:
        raise ValueError(f"hidden size must be divisible by 128, got {hidden_size}")
    if local_vocab % 16:
        raise ValueError(f"local vocabulary must be divisible by 16, got {local_vocab}")
    if max(args.top_k) > local_vocab:
        raise ValueError("top-k exceeds the local vocabulary")
    if max(args.candidates) > local_vocab:
        raise ValueError("candidate count exceeds the local vocabulary")
    if min(args.candidates) < max(args.top_k):
        raise ValueError("every candidate count must be at least every top-k")

    quant_start = perf_counter()
    fp8_weight = _quantize_weight(weight, args.weight_scheme)
    torch.accelerator.synchronize()
    quantize_seconds = perf_counter() - quant_start

    payload: dict[str, object] = {
        "standard": "shape-generic-fp8-coarse-bf16-refined-lm-head",
        "model_dir": str(args.model_dir),
        "weight_key": weight_key,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "vocab_start": vocab_start,
        "shape": {"n_local": local_vocab, "k": hidden_size},
        "weight_scheme": args.weight_scheme,
        "candidate_selector": "flashinfer_radix_or_torch_unsorted",
        "quantize_seconds": quantize_seconds,
        "weight_bytes": weight.numel() * weight.element_size(),
        "fp8_weight_bytes": fp8_weight.weight.numel()
        * fp8_weight.weight.element_size(),
        "fp8_scale_bytes": fp8_weight.scale.numel() * fp8_weight.scale.element_size(),
        "presence_penalty": args.presence_penalty,
        "history_tokens": args.history_tokens,
        "results": {},
    }

    for batch_size in args.batch_sizes:
        generator = torch.Generator(device=device).manual_seed(20260804 + batch_size)
        hidden = torch.randn(
            (batch_size, hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        token_ids, penalties = _make_presence_inputs(
            batch_size=batch_size,
            vocab_size=local_vocab,
            history_tokens=args.history_tokens,
            penalty=args.presence_penalty,
            device=device,
            generator=generator,
        )

        batch_results: dict[str, object] = {}
        for top_k in args.top_k:
            correctness = {
                candidates: {
                    "exact_topk_rows": 0,
                    "ordered_topk_rows": 0,
                    "candidate_hits": 0,
                    "candidate_total": 0,
                    "exact_value_rows": 0,
                    "value_max_abs": 0.0,
                }
                for candidates in args.candidates
                if candidates >= top_k
            }
            candidate_sizes = sorted(correctness)
            nesting = {
                f"{smaller}_in_{larger}": {
                    "nested_rows": 0,
                    "rows_tested": args.seeds * batch_size,
                }
                for smaller, larger in zip(
                    candidate_sizes[:-1],
                    candidate_sizes[1:],
                    strict=True,
                )
            }
            for seed in range(args.seeds):
                seed_generator = torch.Generator(device=device).manual_seed(seed)
                seed_hidden = torch.randn(
                    (batch_size, hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=seed_generator,
                )
                native_values, native_indices = _native_topk(
                    seed_hidden,
                    weight,
                    top_k=top_k,
                    token_ids=token_ids,
                    penalties=penalties,
                )
                native_sorted = native_indices.sort(dim=-1).values
                coarse_logits, penalty_mask = _hybrid_coarse_logits(
                    seed_hidden,
                    fp8_weight,
                    token_ids=token_ids,
                    penalties=penalties,
                )
                candidate_results: dict[int, CandidateResult] = {}
                for candidates, stats in correctness.items():
                    candidate = _refine_topk(
                        seed_hidden,
                        weight,
                        coarse_logits,
                        penalty_mask,
                        top_k=top_k,
                        candidates=candidates,
                    )
                    candidate_results[candidates] = candidate
                    hits, total = _candidate_recall(
                        native_indices,
                        candidate.coarse_indices,
                    )
                    stats["candidate_hits"] += hits
                    stats["candidate_total"] += total
                    exact_set = candidate.indices.sort(dim=-1).values == native_sorted
                    stats["exact_topk_rows"] += int(exact_set.all(dim=-1).sum().item())
                    exact_order = candidate.indices == native_indices
                    stats["ordered_topk_rows"] += int(
                        exact_order.all(dim=-1).sum().item()
                    )
                    value_diff = (candidate.values - native_values).abs()
                    stats["exact_value_rows"] += int(
                        (value_diff == 0).all(dim=-1).sum().item()
                    )
                    stats["value_max_abs"] = max(
                        float(stats["value_max_abs"]),
                        float(value_diff.max().item()),
                    )
                for smaller, larger in zip(
                    candidate_sizes[:-1],
                    candidate_sizes[1:],
                    strict=True,
                ):
                    smaller_indices = candidate_results[smaller].coarse_indices
                    larger_indices = candidate_results[larger].coarse_indices
                    membership = (
                        smaller_indices.unsqueeze(-1) == larger_indices.unsqueeze(-2)
                    ).any(dim=-1)
                    nesting[f"{smaller}_in_{larger}"]["nested_rows"] += int(
                        membership.all(dim=-1).sum().item()
                    )

            timings: dict[str, object] = {
                "native": _time_cuda_graph(
                    partial(
                        _native_topk,
                        hidden,
                        weight,
                        top_k=top_k,
                        token_ids=token_ids,
                        penalties=penalties,
                    ),
                    steps=args.steps,
                    warmup=args.warmup,
                    trials=args.trials,
                )
            }
            native_us = float(timings["native"]["median_us"])
            for candidates in correctness:
                timing = _time_cuda_graph(
                    partial(
                        _hybrid_topk,
                        hidden,
                        weight,
                        fp8_weight,
                        top_k=top_k,
                        candidates=candidates,
                        token_ids=token_ids,
                        penalties=penalties,
                    ),
                    steps=args.steps,
                    warmup=args.warmup,
                    trials=args.trials,
                )
                timing["speedup"] = native_us / float(timing["median_us"])
                timings[f"hybrid_candidates_{candidates}"] = timing

            topk_result = {
                "rows_tested": args.seeds * batch_size,
                "correctness": correctness,
                "candidate_nesting": nesting,
                "timings": timings,
            }
            batch_results[str(top_k)] = topk_result
            for candidates, stats in correctness.items():
                hybrid_timing = timings[f"hybrid_candidates_{candidates}"]
                print(
                    f"m={batch_size} top_k={top_k} candidates={candidates} "
                    f"exact_set={stats['exact_topk_rows']}/"
                    f"{args.seeds * batch_size} "
                    f"exact_values={stats['exact_value_rows']}/"
                    f"{args.seeds * batch_size} "
                    f"recall={stats['candidate_hits']}/"
                    f"{stats['candidate_total']} "
                    f"native={native_us:.3f}us "
                    f"hybrid={hybrid_timing['median_us']:.3f}us "
                    f"speedup={hybrid_timing['speedup']:.3f}x",
                    flush=True,
                )
            for relation, stats in nesting.items():
                print(
                    f"m={batch_size} top_k={top_k} nesting={relation} "
                    f"rows={stats['nested_rows']}/{stats['rows_tested']}",
                    flush=True,
                )
        payload["results"][str(batch_size)] = batch_results

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
