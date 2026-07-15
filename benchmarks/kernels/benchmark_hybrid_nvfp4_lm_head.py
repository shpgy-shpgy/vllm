# SPDX-License-Identifier: Apache-2.0
"""Small NVFP4-coarse/BF16-refined lm-head comparison.

This is deliberately separate from the MXFP4 harness: NVFP4 uses a global
scale and block-size 16, while MXFP4 uses block-size 32.  The script measures
the same end-to-end coarse/selector/refine path as the hybrid MXFP4 harness,
and also checks the candidate set against the BF16 reference.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from vllm.model_executor.layers.hybrid_mxfp8_lm_head import (
    indexed_bf16_dot,
    select_lm_head_candidates,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

from benchmark_hybrid_mxfp8_lm_head import (  # noqa: E402
    _annotate_lm_head_timing,
    _candidate_recall,
    _load_weight_shard,
    _native_topk,
    _refine_selected_topk,
    _time_cuda_graph,
)


@dataclass
class Nvfp4Weight:
    weight: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    output_size: int
    backend: str


@dataclass
class CandidateResult:
    values: torch.Tensor
    indices: torch.Tensor
    coarse_indices: torch.Tensor


def _quantize_nvfp4(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize with the 128x4 layout used by b12x/cuDNN/CUTLASS."""
    from flashinfer import SfLayout, nvfp4_quantize

    # This is the global-scale convention used by FlashInfer's NVFP4 examples.
    global_scale = (448 * 6) / tensor.float().abs().nan_to_num().max()
    quantized, scale = nvfp4_quantize(
        tensor.contiguous(),
        global_scale,
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    return quantized, scale, global_scale


def _quantize_weight(weight: torch.Tensor, backend: str) -> Nvfp4Weight:
    quantized, scale, global_scale = _quantize_nvfp4(weight)
    return Nvfp4Weight(
        quantized,
        scale,
        global_scale,
        weight.shape[0],
        backend,
    )


def _nvfp4_linear(
    hidden: torch.Tensor,
    weight: Nvfp4Weight,
) -> torch.Tensor:
    hidden_q, hidden_scale, hidden_global_scale = _quantize_nvfp4(hidden)
    alpha = 1.0 / (hidden_global_scale * weight.global_scale)
    logits = flashinfer_scaled_fp4_mm(
        hidden_q,
        weight.weight,
        hidden_scale,
        weight.scale,
        alpha=alpha,
        out_dtype=torch.bfloat16,
        backend=weight.backend,
        block_size=16,
        use_nvfp4=True,
    )
    return logits[:, : weight.output_size]


def _nvfp4_mm(
    hidden_q: torch.Tensor,
    hidden_scale: torch.Tensor,
    hidden_global_scale: torch.Tensor,
    weight: Nvfp4Weight,
) -> torch.Tensor:
    return flashinfer_scaled_fp4_mm(
        hidden_q,
        weight.weight,
        hidden_scale,
        weight.scale,
        alpha=1.0 / (hidden_global_scale * weight.global_scale),
        out_dtype=torch.bfloat16,
        backend=weight.backend,
        block_size=16,
        use_nvfp4=True,
    )[:, : weight.output_size]


def _hybrid_topk(
    hidden: torch.Tensor,
    bf16_weight: torch.Tensor,
    weight: Nvfp4Weight,
    *,
    top_k: int,
    candidates: int,
) -> CandidateResult:
    coarse_logits = _nvfp4_linear(hidden, weight)
    coarse_indices = select_lm_head_candidates(coarse_logits, candidates)
    values, indices = _refine_selected_topk(
        hidden,
        bf16_weight,
        coarse_indices,
        None,
        top_k=top_k,
    )
    return CandidateResult(values, indices, coarse_indices)


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16, 32])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 20])
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--backend",
        choices=("auto", "b12x", "cutlass", "cudnn"),
        default="auto",
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")
    weight, vocab_start, weight_key = _load_weight_shard(
        args.model_dir,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        device=device,
    )
    if weight.shape[1] % 32:
        raise ValueError(f"hidden size must be divisible by 32, got {weight.shape[1]}")
    if max(args.top_k) > weight.shape[0] or args.candidates > weight.shape[0]:
        raise ValueError("top-k/candidates exceed the local vocabulary")
    if min(args.top_k) > args.candidates:
        raise ValueError("candidates must be at least every top-k")

    quant_start = torch.cuda.Event(enable_timing=True)
    quant_end = torch.cuda.Event(enable_timing=True)
    quant_start.record()
    nvfp4_weight = _quantize_weight(weight, args.backend)
    quant_end.record()
    quant_end.synchronize()
    weight_quantize_us = quant_start.elapsed_time(quant_end) * 1000.0

    payload: dict[str, object] = {
        "standard": "shape-generic-nvfp4-coarse-bf16-refined-lm-head",
        "model_dir": str(args.model_dir),
        "weight_key": weight_key,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "vocab_start": vocab_start,
        "shape": {"n_local": weight.shape[0], "k": weight.shape[1]},
        "weight_scheme": "nvfp4-e2m1-ue4m3-block16",
        "backend": args.backend,
        "candidate_selector": "flashinfer_radix_or_torch_unsorted",
        "weight_quantize_us": weight_quantize_us,
        "results": {},
    }

    for batch_size in args.batch_sizes:
        generator = torch.Generator(device=device).manual_seed(20260811 + batch_size)
        hidden = torch.randn(
            (batch_size, weight.shape[1]),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        batch_results: dict[str, object] = {}
        for top_k in args.top_k:
            exact_rows = 0
            candidate_hits = 0
            candidate_total = 0
            native_values = native_indices = None
            for seed in range(args.seeds):
                seed_generator = torch.Generator(device=device).manual_seed(seed)
                seed_hidden = torch.randn(
                    (batch_size, weight.shape[1]),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=seed_generator,
                )
                ref_values, ref_indices = _native_topk(
                    seed_hidden,
                    weight,
                    top_k=top_k,
                    token_ids=None,
                    penalties=None,
                )
                candidate = _hybrid_topk(
                    seed_hidden,
                    weight,
                    nvfp4_weight,
                    top_k=top_k,
                    candidates=args.candidates,
                )
                hits, total = _candidate_recall(ref_indices, candidate.coarse_indices)
                candidate_hits += hits
                candidate_total += total
                exact_rows += int(
                    (
                        candidate.indices.sort(dim=-1).values
                        == ref_indices.sort(dim=-1).values
                    )
                    .all(dim=-1)
                    .sum()
                    .item()
                )
                native_values, native_indices = ref_values, ref_indices

            native_timing = _time_cuda_graph(
                partial(
                    _native_topk,
                    hidden,
                    weight,
                    top_k=top_k,
                    token_ids=None,
                    penalties=None,
                ),
                steps=args.steps,
                warmup=args.warmup,
                trials=args.trials,
            )
            hidden_q, hidden_scale, hidden_global_scale = _quantize_nvfp4(hidden)
            coarse_timing = _time_cuda_graph(
                partial(_nvfp4_linear, hidden, nvfp4_weight),
                steps=args.steps,
                warmup=args.warmup,
                trials=args.trials,
            )
            gemm_timing = _time_cuda_graph(
                partial(
                    _nvfp4_mm,
                    hidden_q,
                    hidden_scale,
                    hidden_global_scale,
                    nvfp4_weight,
                ),
                steps=args.steps,
                warmup=args.warmup,
                trials=args.trials,
            )
            coarse_logits = _nvfp4_linear(hidden, nvfp4_weight)
            coarse_indices = select_lm_head_candidates(coarse_logits, args.candidates)
            selector_timing = _time_cuda_graph(
                partial(select_lm_head_candidates, coarse_logits, args.candidates),
                steps=args.steps,
                warmup=args.warmup,
                trials=args.trials,
            )
            refine_timing = _time_cuda_graph(
                partial(
                    _refine_selected_topk,
                    hidden,
                    weight,
                    coarse_indices,
                    None,
                    top_k=top_k,
                ),
                steps=args.steps,
                warmup=args.warmup,
                trials=args.trials,
            )
            hybrid_timing = _time_cuda_graph(
                partial(
                    _hybrid_topk,
                    hidden,
                    weight,
                    nvfp4_weight,
                    top_k=top_k,
                    candidates=args.candidates,
                ),
                steps=args.steps,
                warmup=args.warmup,
                trials=args.trials,
            )
            components = {
                "coarse": coarse_timing,
                "gemm": gemm_timing,
                "selector": selector_timing,
                "refine": refine_timing,
            }
            hybrid_timing["components"] = components
            _annotate_lm_head_timing(hybrid_timing, native_timing, components)
            result = {
                "exact_set_rows": exact_rows,
                "rows_tested": args.seeds * batch_size,
                "candidate_hits": candidate_hits,
                "candidate_total": candidate_total,
                "native": native_timing,
                "hybrid": hybrid_timing,
            }
            batch_results[str(top_k)] = result
            print(
                f"backend={args.backend} m={batch_size} top_k={top_k} "
                f"candidates={args.candidates} exact_set={exact_rows}/"
                f"{args.seeds * batch_size} recall={candidate_hits}/"
                f"{candidate_total} native={native_timing['median_us']:.3f}us "
                f"hybrid={hybrid_timing['median_us']:.3f}us "
                f"speedup={hybrid_timing['speedup']:.3f}x "
                f"q+gemm={coarse_timing['median_us']:.3f}us "
                f"gemm={gemm_timing['median_us']:.3f}us",
                flush=True,
            )
        payload["results"][str(batch_size)] = batch_results

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
