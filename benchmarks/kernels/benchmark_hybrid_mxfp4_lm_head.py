# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark shape-generic MXFP4-coarse/BF16-refined lm-head top-k.

This is an evaluation harness, not a serving path.  It mirrors
``benchmark_hybrid_mxfp8_lm_head.py`` so that MXFP4 and MXFP8 can be compared
with the same CUDA-Graph/CUDA-Event timing and candidate-recall checks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.hybrid_mxfp8_lm_head import (
    indexed_bf16_dot,
    select_lm_head_candidates,
)
from vllm.utils.flashinfer import (
    autotune_with_torch_cuda_delay as flashinfer_autotune,
    flashinfer_mxfp4_quantize,
    flashinfer_scaled_fp4_mm,
)

# The FP8 harness already contains the model-shard loader, native top-k,
# candidate/refine helpers and the common CUDA-Graph timer.  Importing those
# private helpers keeps both benchmark scripts on exactly the same protocol.
# When this file is invoked directly, Python puts benchmarks/kernels on
# sys.path, so import the sibling harness by its filename (the benchmarks
# directory is intentionally not a Python package in this checkout).
from benchmark_hybrid_mxfp8_lm_head import (  # noqa: E402
    _apply_presence_penalty,
    _candidate_recall,
    _load_weight_shard,
    _make_presence_inputs,
    _native_topk,
    _positive_int,
    _refine_selected_topk,
    _refine_topk,
    _annotate_lm_head_timing,
    _time_cuda_graph,
)

_BLOCK_SIZE = 32


@dataclass
class Mxfp4Weight:
    weight: torch.Tensor
    scale: torch.Tensor
    output_size: int
    backend: str
    quant_backend: str


@dataclass
class CandidateResult:
    values: torch.Tensor
    indices: torch.Tensor
    coarse_indices: torch.Tensor


def _mxfp4_quantize(
    tensor: torch.Tensor,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize MXFP4 activations/weights using the selected FlashInfer path."""
    if backend == "cuda":
        # This custom op is the exact path used by hybrid_mxfp4_lm_head.py.
        return flashinfer_mxfp4_quantize(tensor)
    if backend == "cute-dsl":
        # CuTe-DSL is experimental, but useful for isolating quantizer cost.
        from flashinfer import mxfp4_quantize

        return mxfp4_quantize(tensor, backend="cute-dsl")
    raise ValueError(f"unknown MXFP4 quantization backend: {backend}")


def _quantize_weight(weight: torch.Tensor, backend: str) -> Mxfp4Weight:
    quantized, scale = _mxfp4_quantize(weight, backend)
    return Mxfp4Weight(
        quantized,
        scale,
        weight.shape[0],
        backend="auto",
        quant_backend=backend,
    )


def _quantize_hidden(
    hidden: torch.Tensor,
    quant_backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _mxfp4_quantize(hidden.contiguous(), quant_backend)


def _mxfp4_mm(
    hidden_q: torch.Tensor,
    hidden_scale: torch.Tensor,
    weight: Mxfp4Weight,
) -> torch.Tensor:
    logits = flashinfer_scaled_fp4_mm(
        hidden_q,
        weight.weight,
        hidden_scale,
        weight.scale,
        alpha=None,
        out_dtype=torch.bfloat16,
        backend=weight.backend,
        block_size=_BLOCK_SIZE,
        use_nvfp4=False,
    )
    return logits[:, : weight.output_size]


def _mxfp4_linear(hidden: torch.Tensor, weight: Mxfp4Weight) -> torch.Tensor:
    hidden_q, hidden_scale = _quantize_hidden(hidden, weight.quant_backend)
    return _mxfp4_mm(hidden_q, hidden_scale, weight)


def _hybrid_coarse_logits(
    hidden: torch.Tensor,
    mxfp4_weight: Mxfp4Weight,
    *,
    token_ids: torch.Tensor | None,
    penalties: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    coarse_logits = _mxfp4_linear(hidden, mxfp4_weight)
    return _apply_presence_penalty(coarse_logits, token_ids, penalties)


def _hybrid_topk(
    hidden: torch.Tensor,
    bf16_weight: torch.Tensor,
    mxfp4_weight: Mxfp4Weight,
    *,
    top_k: int,
    candidates: int,
    token_ids: torch.Tensor | None,
    penalties: torch.Tensor | None,
) -> CandidateResult:
    coarse_logits, penalty_mask = _hybrid_coarse_logits(
        hidden,
        mxfp4_weight,
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


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tp-size", type=_positive_int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
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
    parser.add_argument(
        "--backend",
        choices=("auto", "cutlass", "cudnn", "cute-dsl"),
        default="auto",
        help=(
            "FlashInfer FP4 GEMM backend; auto matches the serving path. "
            "cutlass is accepted as a compatibility alias for auto because "
            "this FlashInfer release does not support MXFP4 on cutlass."
        ),
    )
    parser.add_argument(
        "--quant-backend",
        choices=("cuda", "cute-dsl"),
        default="cuda",
        help="MXFP4 activation/weight quantizer; cuda matches the serving path.",
    )
    parser.add_argument("--skip-autotune", action="store_true")
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
    if hidden_size % _BLOCK_SIZE:
        raise ValueError(
            f"hidden size must be divisible by {_BLOCK_SIZE}, got {hidden_size}"
        )
    if max(args.top_k) > local_vocab:
        raise ValueError("top-k exceeds the local vocabulary")
    if max(args.candidates) > local_vocab:
        raise ValueError("candidate count exceeds the local vocabulary")
    if min(args.candidates) < max(args.top_k):
        raise ValueError("every candidate count must be at least every top-k")

    quant_start = perf_counter()
    mxfp4_weight = _quantize_weight(weight, args.quant_backend)
    requested_backend = args.backend
    effective_backend = args.backend
    if requested_backend == "cutlass":
        print(
            "WARNING: FlashInfer MXFP4 does not support backend=cutlass; "
            "using backend=auto (the serving-path dispatch).",
            flush=True,
        )
        effective_backend = "auto"
    mxfp4_weight.backend = effective_backend
    torch.accelerator.synchronize()
    quantize_seconds = perf_counter() - quant_start

    autotune_seconds = 0.0
    if not args.skip_autotune:
        tune_start = perf_counter()
        max_batch_size = max(args.batch_sizes)
        tune_rows = 1 << (max_batch_size - 1).bit_length()
        tune_hidden = torch.zeros(
            (tune_rows, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        with flashinfer_autotune(tune_mode=True):
            _mxfp4_linear(tune_hidden, mxfp4_weight)
        torch.accelerator.synchronize()
        autotune_seconds = perf_counter() - tune_start

    payload: dict[str, object] = {
        "standard": "shape-generic-mxfp4-coarse-bf16-refined-lm-head",
        "model_dir": str(args.model_dir),
        "weight_key": weight_key,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "vocab_start": vocab_start,
        "shape": {"n_local": local_vocab, "k": hidden_size},
        "weight_scheme": "mxfp4-e2m1-ue8m0-block32",
        "backend": effective_backend,
        "requested_backend": requested_backend,
        "quant_backend": args.quant_backend,
        "candidate_selector": "flashinfer_radix_or_torch_unsorted",
        "quantize_seconds": quantize_seconds,
        "autotune_seconds": autotune_seconds,
        "weight_bytes": weight.numel() * weight.element_size(),
        "mxfp4_weight_bytes": mxfp4_weight.weight.numel()
        * mxfp4_weight.weight.element_size(),
        "mxfp4_scale_bytes": mxfp4_weight.scale.numel()
        * mxfp4_weight.scale.element_size(),
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
                    candidate_sizes[:-1], candidate_sizes[1:], strict=True
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
                    mxfp4_weight,
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
                        native_indices, candidate.coarse_indices
                    )
                    stats["candidate_hits"] += hits
                    stats["candidate_total"] += total
                    exact_set = (
                        candidate.indices.sort(dim=-1).values == native_sorted
                    )
                    stats["exact_topk_rows"] += int(
                        exact_set.all(dim=-1).sum().item()
                    )
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
                    candidate_sizes[:-1], candidate_sizes[1:], strict=True
                ):
                    smaller_indices = candidate_results[smaller].coarse_indices
                    larger_indices = candidate_results[larger].coarse_indices
                    membership = (
                        smaller_indices.unsqueeze(-1)
                        == larger_indices.unsqueeze(-2)
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
            hidden_q, hidden_scale = _quantize_hidden(
                hidden, args.quant_backend
            )
            coarse_logits, penalty_mask = _hybrid_coarse_logits(
                hidden,
                mxfp4_weight,
                token_ids=token_ids,
                penalties=penalties,
            )
            for candidates in correctness:
                coarse_indices = select_lm_head_candidates(
                    coarse_logits, candidates
                )
                components = {
                    "activation_quant": _time_cuda_graph(
                        partial(_quantize_hidden, hidden, args.quant_backend),
                        steps=args.steps,
                        warmup=args.warmup,
                        trials=args.trials,
                    ),
                    "gemm": _time_cuda_graph(
                        partial(
                            _mxfp4_mm,
                            hidden_q,
                            hidden_scale,
                            mxfp4_weight,
                        ),
                        steps=args.steps,
                        warmup=args.warmup,
                        trials=args.trials,
                    ),
                    "coarse": _time_cuda_graph(
                        partial(
                            _hybrid_coarse_logits,
                            hidden,
                            mxfp4_weight,
                            token_ids=token_ids,
                            penalties=penalties,
                        ),
                        steps=args.steps,
                        warmup=args.warmup,
                        trials=args.trials,
                    ),
                    "selector": _time_cuda_graph(
                        partial(
                            select_lm_head_candidates,
                            coarse_logits,
                            candidates,
                        ),
                        steps=args.steps,
                        warmup=args.warmup,
                        trials=args.trials,
                    ),
                    "refine": _time_cuda_graph(
                        partial(
                            _refine_selected_topk,
                            hidden,
                            weight,
                            coarse_indices,
                            penalty_mask,
                            top_k=top_k,
                        ),
                        steps=args.steps,
                        warmup=args.warmup,
                        trials=args.trials,
                    ),
                }
                timing = _time_cuda_graph(
                    partial(
                        _hybrid_topk,
                        hidden,
                        weight,
                        mxfp4_weight,
                        top_k=top_k,
                        candidates=candidates,
                        token_ids=token_ids,
                        penalties=penalties,
                    ),
                    steps=args.steps,
                    warmup=args.warmup,
                    trials=args.trials,
                )
                timing["components"] = components
                _annotate_lm_head_timing(timing, timings["native"], components)
                timings[f"hybrid_candidates_{candidates}"] = timing

            batch_results[str(top_k)] = {
                "rows_tested": args.seeds * batch_size,
                "correctness": correctness,
                "candidate_nesting": nesting,
                "timings": timings,
            }
            for candidates, stats in correctness.items():
                hybrid_timing = timings[f"hybrid_candidates_{candidates}"]
                components = hybrid_timing["components"]
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
                    f"speedup={hybrid_timing['speedup']:.3f}x "
                    f"saved={hybrid_timing['lm_head_saved_us']:.3f}us "
                    f"saved_p95={hybrid_timing['lm_head_saved_p95_us']:.3f}us "
                    f"components=q:{components['activation_quant']['median_us']:.3f},"
                    f"gemm:{components['gemm']['median_us']:.3f},"
                    f"coarse:{components['coarse']['median_us']:.3f},"
                    f"select:{components['selector']['median_us']:.3f},"
                    f"refine:{components['refine']['median_us']:.3f},"
                    f"unaccounted:{hybrid_timing['hybrid_unaccounted_us']:.3f}us",
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
