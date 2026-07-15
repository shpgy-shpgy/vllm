# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup autotune and JIT warmup for hybrid MXFP4/MXFP8 lm heads.

Weight loading only quantizes and registers the persistent auxiliary copies.
FlashInfer tactic enumeration (per row bucket) and Triton/selector JIT
compilation are deferred to this central warmup stage so they never trigger
during model execution or CUDA graph capture.
"""

from __future__ import annotations

from time import perf_counter

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.hybrid_nvfp4_lm_head import (
    autotune_hybrid_nvfp4_lm_head,
    get_hybrid_nvfp4_lm_head,
    warmup_hybrid_nvfp4_lm_head_kernels,
)
from vllm.model_executor.layers.hybrid_mxfp4_lm_head import (
    autotune_hybrid_mxfp4_lm_head,
    get_hybrid_mxfp4_lm_head,
    warmup_hybrid_mxfp4_lm_head_kernels,
)
from vllm.model_executor.layers.hybrid_mxfp8_lm_head import (
    autotune_hybrid_mxfp8_lm_head,
    autotune_row_buckets,
    get_hybrid_mxfp8_lm_head,
    warmup_hybrid_mxfp8_lm_head_kernels,
)

logger = init_logger(__name__)
_DEFAULT_WARMUP_MAX_ROWS = 2048


def _collect_hybrid_lm_heads(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    heads: dict[int, torch.nn.Module] = {}
    for _, module in model.named_modules():
        if (
            get_hybrid_nvfp4_lm_head(module) is not None
            or get_hybrid_mxfp4_lm_head(module) is not None
            or get_hybrid_mxfp8_lm_head(module) is not None
        ):
            heads[id(module)] = module
    return heads


def _warmup_row_shapes(worker, max_rows: int) -> tuple[int, ...]:
    capture_sizes = worker.vllm_config.compilation_config.cudagraph_capture_sizes or []
    row_shapes = sorted(
        {
            size
            for size in capture_sizes
            if isinstance(size, int)
            and 0 < size
            and (max_rows <= 0 or size <= max_rows)
        }
    )
    if not row_shapes:
        # No CUDA graph capture sizes (eager or capture disabled): fall back
        # to FlashInfer's own tactic buckets so first runtime use can't tune.
        row_shapes = list(
            autotune_row_buckets(max_rows or _DEFAULT_WARMUP_MAX_ROWS)
        )
    if row_shapes[0] != 1:
        # Draft bs1 steps always sample a single row.
        row_shapes.insert(0, 1)
    return tuple(row_shapes)


def hybrid_lm_head_warmup(worker) -> None:
    """Warm every enabled hybrid MXFP4/MXFP8 lm-head format before capture."""
    if not (
        envs.VLLM_HYBRID_NVFP4_LM_HEAD
        or envs.VLLM_HYBRID_MXFP4_LM_HEAD
        or envs.VLLM_HYBRID_MXFP8_LM_HEAD
    ):
        return

    heads = _collect_hybrid_lm_heads(worker.get_model())
    speculator = getattr(getattr(worker, "model_runner", None), "speculator", None)
    draft_model = getattr(speculator, "model", None)
    if isinstance(draft_model, torch.nn.Module):
        # Usually shares the target lm head and dedupes by id(); covered anyway
        # when the draft keeps a distinct prepared copy.
        heads.update(_collect_hybrid_lm_heads(draft_model))
    if not heads:
        logger.debug("No prepared hybrid lm heads found; skipping lm-head warmup.")
        return

    started = perf_counter()
    row_count = 0
    warmed_states: set[int] = set()
    for layer in heads.values():
        nvfp4_state = get_hybrid_nvfp4_lm_head(layer)
        mxfp4_state = get_hybrid_mxfp4_lm_head(layer)
        mxfp8_state = get_hybrid_mxfp8_lm_head(layer)
        state = nvfp4_state or mxfp4_state or mxfp8_state
        assert state is not None
        if id(state) in warmed_states:
            continue
        warmed_states.add(id(state))
        row_shapes = _warmup_row_shapes(worker, state.max_rows)
        if nvfp4_state is not None:
            _, tuned_shapes = autotune_hybrid_nvfp4_lm_head(
                nvfp4_state,
                layer.weight,
                row_shapes,
            )
        elif mxfp4_state is not None:
            _, tuned_shapes = autotune_hybrid_mxfp4_lm_head(
                mxfp4_state,
                layer.weight,
                row_shapes,
            )
        else:
            _, tuned_shapes = autotune_hybrid_mxfp8_lm_head(
                state,
                layer.weight,
                row_shapes,
            )
        row_count = max(row_count, len(tuned_shapes))
        if nvfp4_state is not None:
            warmup_hybrid_nvfp4_lm_head_kernels(
                nvfp4_state,
                layer.weight,
                tp_size=getattr(layer, "tp_size", 1),
            )
            logger.debug("Hybrid NVFP4 lm-head tuned row shapes: %s", tuned_shapes)
        elif mxfp4_state is not None:
            warmup_hybrid_mxfp4_lm_head_kernels(
                mxfp4_state,
                layer.weight,
                tp_size=getattr(layer, "tp_size", 1),
            )
            logger.debug("Hybrid MXFP4 lm-head tuned row shapes: %s", tuned_shapes)
        else:
            warmup_hybrid_mxfp8_lm_head_kernels(
                state,
                layer.weight,
                tp_size=getattr(layer, "tp_size", 1),
            )
            logger.debug("Hybrid MXFP8 lm-head tuned row shapes: %s", tuned_shapes)
    torch.accelerator.synchronize()
    logger.info(
        "Warmed %d hybrid lm-head state(s): FlashInfer autotune across %d row "
        "shapes (CUDA graph capture sizes) and selector/refine/reduction JIT "
        "in %.2fs.",
        len(warmed_states),
        row_count,
        perf_counter() - started,
    )


__all__ = ["hybrid_lm_head_warmup"]
