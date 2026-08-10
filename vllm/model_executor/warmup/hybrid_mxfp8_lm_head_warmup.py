# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup autotune and JIT warmup for hybrid MXFP8 lm heads.

Weight loading only quantizes and registers the persistent MXFP8 copies.
FlashInfer CUTLASS tactic enumeration (per row bucket) and Triton/selector
JIT compilation are deferred to this central warmup stage so they never
trigger during model execution or CUDA graph capture.
"""

from __future__ import annotations

from time import perf_counter

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.hybrid_mxfp8_lm_head import (
    autotune_hybrid_mxfp8_lm_head,
    autotune_row_buckets,
    get_hybrid_mxfp8_lm_head,
    warmup_hybrid_mxfp8_lm_head_kernels,
)

logger = init_logger(__name__)


def _collect_hybrid_lm_heads(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    heads: dict[int, torch.nn.Module] = {}
    for _, module in model.named_modules():
        if get_hybrid_mxfp8_lm_head(module) is not None:
            heads[id(module)] = module
    return heads


def _warmup_row_shapes(worker, max_rows: int) -> tuple[int, ...]:
    capture_sizes = worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
    row_shapes = sorted(
        {size for size in capture_sizes if isinstance(size, int) and 0 < size <= max_rows}
    )
    if not row_shapes:
        # No CUDA graph capture sizes (eager or capture disabled): fall back
        # to FlashInfer's own tactic buckets so first runtime use can't tune.
        row_shapes = list(autotune_row_buckets(max_rows))
    if row_shapes[0] != 1:
        # Draft bs1 steps always sample a single row.
        row_shapes.insert(0, 1)
    return tuple(row_shapes)


def hybrid_mxfp8_lm_head_warmup(worker) -> None:
    """Autotune every runtime row shape and warm JIT kernels before capture."""
    if not envs.VLLM_HYBRID_MXFP8_LM_HEAD:
        return

    heads = _collect_hybrid_lm_heads(worker.get_model())
    speculator = getattr(getattr(worker, "model_runner", None), "speculator", None)
    draft_model = getattr(speculator, "model", None)
    if isinstance(draft_model, torch.nn.Module):
        # Usually shares the target lm head and dedupes by id(); covered anyway
        # when the draft keeps a distinct prepared copy.
        heads.update(_collect_hybrid_lm_heads(draft_model))
    if not heads:
        logger.debug(
            "No prepared hybrid MXFP8 lm heads found; skipping lm-head warmup."
        )
        return

    started = perf_counter()
    row_count = 0
    for layer in heads.values():
        state = get_hybrid_mxfp8_lm_head(layer)
        assert state is not None
        row_shapes = _warmup_row_shapes(worker, state.max_rows)
        _, tuned_shapes = autotune_hybrid_mxfp8_lm_head(
            state,
            layer.weight,
            row_shapes,
        )
        row_count = max(row_count, len(tuned_shapes))
        warmup_hybrid_mxfp8_lm_head_kernels(
            state,
            layer.weight,
            tp_size=getattr(layer, "tp_size", 1),
        )
        logger.debug("Hybrid MXFP8 lm-head tuned row shapes: %s", tuned_shapes)
    torch.accelerator.synchronize()
    logger.info(
        "Warmed %d hybrid MXFP8 lm-head(s): CUTLASS autotune across %d row "
        "shapes (CUDA graph capture sizes) and selector/refine/reduction JIT "
        "in %.2fs.",
        len(heads),
        row_count,
        perf_counter() - started,
    )
