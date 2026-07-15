# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental MXFP4 coarse search for BF16 lm heads."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.argmax_triton import (
    indexed_argmax_triton,
    reduce_global_argmax_triton,
)
from vllm.model_executor.layers.hybrid_mxfp8_lm_head import (
    autotune_row_buckets,
    _candidate_count_for_topk,
    indexed_bf16_dot,
    select_lm_head_candidates,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    autotune_with_torch_cuda_delay,
    flashinfer_mxfp4_quantize,
    flashinfer_scaled_fp4_mm,
    has_flashinfer,
)

logger = init_logger(__name__)

_BLOCK_SIZE = 32
_MIN_GEMM_DIMENSION = 128
_WEIGHT_NAME = "_hybrid_mxfp4_lm_head_weight"
_SCALE_NAME = "_hybrid_mxfp4_lm_head_scale"
_STATE_NAME = "_hybrid_mxfp4_lm_head_state"
_SHARED_STATE_NAME = "_hybrid_mxfp4_lm_head_shared_state"
_BUFFER_NAMES = (_WEIGHT_NAME, _SCALE_NAME)
_DEFAULT_AUTOTUNE_MAX_ROWS = 2048


def _row_bucket_padding_enabled() -> bool:
    return envs.VLLM_HYBRID_MXFP4_LM_HEAD_ROW_BUCKET_PAD


@lru_cache(maxsize=4)
def _mxfp4_row_buckets(max_rows: int) -> tuple[int, ...]:
    """Return the same uncapped row buckets used by FlashInfer's FP4 tuner."""
    return autotune_row_buckets(max_rows)


def _mxfp4_row_bucket(rows: int) -> int:
    buckets = _mxfp4_row_buckets(_DEFAULT_AUTOTUNE_MAX_ROWS)
    index = bisect_left(buckets, rows)
    return buckets[index] if index < len(buckets) else rows


@dataclass
class HybridMxfp4LmHead:
    """Persistent MXFP4 weight copy plus BF16 candidate refinement."""

    weight: torch.Tensor
    scale: torch.Tensor
    input_size: int
    output_size: int
    candidates: int
    max_rows: int
    # Runtime diagnostics for compact-path eligibility.  These are deliberately
    # kept on the prepared state so target/MTP callers share the same counters
    # when they share one quantized lm-head copy.
    can_use_failure_counts: dict[str, int] = field(
        default_factory=dict,
        repr=False,
    )

    def _record_can_use_failure(
        self,
        reason: str,
        hidden_states: torch.Tensor,
        *,
        active_vocab_size: int,
        top_k: int,
    ) -> None:
        count = self.can_use_failure_counts.get(reason, 0) + 1
        self.can_use_failure_counts[reason] = count

        # Keep serving logs useful without logging every sampling step.  Powers
        # of two expose both the first failure and recurring mixed-batch
        # fallbacks while keeping the log volume bounded.
        if count & (count - 1):
            return
        logger.warning(
            "Hybrid MXFP4 lm-head can_use() rejected the compact path "
            "(count=%d, reason=%s, M=%d, hidden_shape=%s, hidden_stride=%s, "
            "hidden_dtype=%s, hidden_contiguous=%s, active_vocab=%d, top_k=%d, "
            "candidate_width=%d, max_rows=%d); falling back to the full "
            "lm-head for this call.",
            count,
            reason,
            hidden_states.shape[0] if hidden_states.ndim > 0 else -1,
            tuple(hidden_states.shape),
            tuple(hidden_states.stride()),
            hidden_states.dtype,
            hidden_states.is_contiguous(),
            active_vocab_size,
            top_k,
            self.candidate_count_for_topk(top_k),
            self.max_rows,
        )

    def candidate_count_for_topk(self, top_k: int) -> int:
        """Choose transient refinement width without another MXFP4 copy."""
        return _candidate_count_for_topk(
            self.candidates,
            top_k,
            output_size=self.output_size,
        )

    def can_use(
        self,
        hidden_states: torch.Tensor,
        *,
        bf16_weight: torch.Tensor,
        active_vocab_size: int,
        top_k: int,
    ) -> bool:
        reason: str | None = None
        if hidden_states.ndim != 2:
            reason = "hidden_ndim"
        elif hidden_states.dtype != torch.bfloat16:
            reason = "hidden_dtype"
        elif not hidden_states.is_cuda:
            reason = "hidden_not_cuda"
        elif not hidden_states.is_contiguous():
            reason = "hidden_not_contiguous"
        elif hidden_states.shape[1] != self.input_size:
            reason = "hidden_width"
        elif bf16_weight.dtype != torch.bfloat16:
            reason = "weight_dtype"
        elif bf16_weight.device != hidden_states.device:
            reason = "weight_device"
        elif not bf16_weight.is_contiguous():
            reason = "weight_not_contiguous"
        elif bf16_weight.shape != (self.output_size, self.input_size):
            reason = "weight_shape"
        elif active_vocab_size > self.output_size:
            reason = "active_vocab_too_large"
        else:
            candidate_width = self.candidate_count_for_topk(top_k)
            if top_k > candidate_width:
                reason = "top_k_exceeds_candidates"
            elif active_vocab_size < candidate_width:
                reason = "active_vocab_too_small"
            elif self.max_rows > 0 and hidden_states.shape[0] > self.max_rows:
                reason = "max_rows"

        if reason is not None:
            self._record_can_use_failure(
                reason,
                hidden_states,
                active_vocab_size=active_vocab_size,
                top_k=top_k,
            )
            return False
        return True

    def coarse_logits(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        # Pad dynamic M to the FlashInfer row bucket by default.  This keeps
        # the cuDNN graph/workspace shape stable when requests finish and M
        # changes; set ROW_BUCKET_PAD=0 to retain the raw dynamic-M path.
        actual_rows = hidden_states.shape[0]
        padded_rows = (
            _mxfp4_row_bucket(actual_rows)
            if _row_bucket_padding_enabled()
            else actual_rows
        )
        if padded_rows != actual_rows:
            hidden_states = torch.nn.functional.pad(
                hidden_states,
                (0, 0, 0, padded_rows - actual_rows),
            )
        hidden_q, hidden_scale = flashinfer_mxfp4_quantize(hidden_states)
        logits = flashinfer_scaled_fp4_mm(
            hidden_q,
            self.weight,
            hidden_scale,
            self.scale,
            alpha=None,
            out_dtype=torch.bfloat16,
            backend="auto",
            block_size=_BLOCK_SIZE,
            use_nvfp4=False,
        )
        logits = logits[:actual_rows, : self.output_size]
        if bias is not None:
            logits += bias
        return logits

    def select_candidates(
        self,
        coarse_logits: torch.Tensor,
        *,
        top_k: int | None = None,
    ) -> torch.Tensor:
        candidates = (
            self.candidates
            if top_k is None
            else self.candidate_count_for_topk(top_k)
        )
        candidates = min(candidates, coarse_logits.shape[-1])
        if top_k is not None and candidates > self.candidates:
            logger.info_once(
                "Hybrid MXFP4 lm-head expands transient candidates from %d "
                "to %d for top-k=%d; persistent copy remains at configured "
                "width %d.",
                self.candidates,
                candidates,
                top_k,
                self.candidates,
            )
        result = select_lm_head_candidates(
            coarse_logits,
            candidates,
            use_flashinfer_topk=envs.VLLM_HYBRID_MXFP4_LM_HEAD_USE_FLASHINFER_TOPK,
            format_name="MXFP4",
        )
        return result

    @staticmethod
    def refine_logits(
        hidden_states: torch.Tensor,
        bf16_weight: torch.Tensor,
        candidate_indices: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = indexed_bf16_dot(
            hidden_states,
            bf16_weight,
            candidate_indices,
        )
        if bias is not None:
            logits += bias[candidate_indices]
        return logits


@torch.inference_mode()
def warmup_hybrid_mxfp4_lm_head_kernels(
    state: HybridMxfp4LmHead,
    bf16_weight: torch.Tensor,
    tp_size: int,
) -> None:
    """Warm MXFP4, selector, compact refinement, and TP reduction kernels."""
    hidden_states = torch.zeros(
        (1, state.input_size),
        dtype=torch.bfloat16,
        device=state.weight.device,
    )
    coarse_logits = state.coarse_logits(hidden_states, None)
    candidate_sets = [state.select_candidates(coarse_logits)]
    # Also warm the widest accuracy-sensitive sampling path.  This keeps
    # candidate-width JIT work out of the first top-k=40 request while the
    # configured greedy/MTP width remains warmed independently.
    if hasattr(state, "candidate_count_for_topk"):
        wide_candidates = state.select_candidates(coarse_logits, top_k=40)
        if wide_candidates.shape[-1] != candidate_sets[0].shape[-1]:
            candidate_sets.append(wide_candidates)

    # The Triton indexed-argmax kernel is intentionally limited to 1024
    # candidates.  Larger configured candidate sets still work through the
    # regular torch reduction in the logits processor; only skip this warmup
    # call for those shapes.
    for candidate_indices in candidate_sets:
        exact_logits = state.refine_logits(
            hidden_states,
            bf16_weight,
            candidate_indices,
            None,
        )
        if exact_logits.shape[-1] <= 1024:
            indexed_argmax_triton(exact_logits, candidate_indices)
    if state.max_rows == 0 or state.max_rows >= 16:
        tiled_rows = 16 if state.max_rows == 0 else min(state.max_rows, 16)
        tiled_hidden_states = hidden_states.expand(tiled_rows, -1).contiguous()
        for candidate_indices in candidate_sets:
            tiled_candidate_indices = candidate_indices.expand(
                tiled_rows, -1
            ).contiguous()
            state.refine_logits(
                tiled_hidden_states,
                bf16_weight,
                tiled_candidate_indices,
                None,
            )
    if tp_size > 1:
        gathered_pairs = torch.zeros(
            (1, tp_size * 2),
            dtype=torch.float32,
            device=state.weight.device,
        )
        reduce_global_argmax_triton(gathered_pairs, tp_size=tp_size)


@torch.inference_mode()
def autotune_hybrid_mxfp4_lm_head(
    state: HybridMxfp4LmHead,
    bf16_weight: torch.Tensor,
    row_shapes: tuple[int, ...] | None = None,
) -> tuple[float, tuple[int, ...]]:
    """Tune FlashInfer MXFP4 tactics for all row shapes used at runtime."""
    if row_shapes is None:
        row_shapes = autotune_row_buckets(
            state.max_rows or _DEFAULT_AUTOTUNE_MAX_ROWS
        )
    hidden_states = torch.zeros(
        (max(row_shapes), state.input_size),
        dtype=torch.bfloat16,
        device=state.weight.device,
    )
    started = perf_counter()
    with autotune_with_torch_cuda_delay(tune_mode=True):
        for rows in row_shapes:
            hidden = hidden_states[:rows]
            coarse_logits = state.coarse_logits(hidden, None)
            candidate_indices = state.select_candidates(coarse_logits)
            state.refine_logits(hidden, bf16_weight, candidate_indices, None)
    torch.accelerator.synchronize()
    return perf_counter() - started, row_shapes


def _attach_state(layer: torch.nn.Module, state: HybridMxfp4LmHead) -> None:
    for name, value in (
        (_WEIGHT_NAME, state.weight),
        (_SCALE_NAME, state.scale),
    ):
        layer.register_buffer(
            name,
            value,
            persistent=False,
        )
    setattr(layer, _STATE_NAME, state)


def prepare_hybrid_mxfp4_lm_head(
    layer: torch.nn.Module,
    *,
    candidates: int,
) -> bool:
    """Create one MXFP4 block-scaled copy, or keep the BF16 lm-head path."""
    if hasattr(layer, _STATE_NAME):
        return True

    weight = layer.weight
    shared_state = getattr(weight, _SHARED_STATE_NAME, None)
    if isinstance(shared_state, HybridMxfp4LmHead):
        _attach_state(layer, shared_state)
        logger.info_once(
            "Reused the shared MXFP4 lm-head copy for weight %s.",
            tuple(weight.shape),
        )
        return True

    if (
        not has_flashinfer()
        or not current_platform.is_cuda()
        or not current_platform.has_device_capability(100)
        or weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
        or getattr(weight, "_vllm_is_uva_offloaded", False)
        or weight.shape[0] < _MIN_GEMM_DIMENSION
        or weight.shape[1] < _MIN_GEMM_DIMENSION
        or weight.shape[1] % _BLOCK_SIZE
    ):
        logger.warning_once(
            "Hybrid MXFP4 lm-head does not support weight %s (%s on %s); "
            "falling back to the original lm-head implementation.",
            tuple(weight.shape),
            weight.dtype,
            weight.device,
        )
        return False
    if candidates <= 0 or candidates > weight.shape[0]:
        logger.warning_once(
            "Hybrid MXFP4 lm-head candidate count %d is outside [1, %d]; "
            "falling back to the original lm-head implementation.",
            candidates,
            weight.shape[0],
        )
        return False

    max_rows = envs.VLLM_HYBRID_MXFP4_LM_HEAD_MAX_ROWS
    if max_rows < 0:
        logger.warning_once(
            "Hybrid MXFP4 lm-head max rows must be non-negative, got %d; "
            "use 0 for no limit. Falling back to the original lm-head "
            "implementation.",
            max_rows,
        )
        return False

    quantized, scale = flashinfer_mxfp4_quantize(weight)
    state = HybridMxfp4LmHead(
        weight=quantized,
        scale=scale,
        input_size=weight.shape[1],
        output_size=weight.shape[0],
        candidates=candidates,
        max_rows=max_rows,
    )
    layer.register_buffer(_WEIGHT_NAME, quantized, persistent=False)
    layer.register_buffer(_SCALE_NAME, scale, persistent=False)
    setattr(layer, _STATE_NAME, state)
    setattr(weight, _SHARED_STATE_NAME, state)

    extra_mib = sum(getattr(layer, name).nbytes for name in _BUFFER_NAMES) / (
        1024 * 1024
    )
    logger.info_once(
        "Prepared shape-generic hybrid MXFP4 lm-head for weight %s with %d "
        "candidates and M<=%s (%.2f MiB persistent overhead; FlashInfer "
        "autotune and kernel warmup run in the startup warmup stage).",
        tuple(weight.shape),
        candidates,
        "unlimited" if max_rows == 0 else str(max_rows),
        extra_mib,
    )
    return True


def get_hybrid_mxfp4_lm_head(layer: torch.nn.Module) -> HybridMxfp4LmHead | None:
    return getattr(layer, _STATE_NAME, None)


def release_hybrid_mxfp4_lm_head(layer: torch.nn.Module) -> int:
    """Drop a prepared MXFP4 copy from an lm head being discarded."""
    released_bytes = 0
    state = getattr(layer, _STATE_NAME, None)
    for name in _BUFFER_NAMES:
        value = getattr(layer, name, None)
        if isinstance(value, torch.Tensor):
            released_bytes += value.nbytes

    if hasattr(layer, _STATE_NAME):
        delattr(layer, _STATE_NAME)
    for name in _BUFFER_NAMES:
        if hasattr(layer, name):
            delattr(layer, name)
    weight = getattr(layer, "weight", None)
    if getattr(weight, _SHARED_STATE_NAME, None) is state:
        delattr(weight, _SHARED_STATE_NAME)
    return released_bytes


__all__ = [
    "HybridMxfp4LmHead",
    "autotune_hybrid_mxfp4_lm_head",
    "get_hybrid_mxfp4_lm_head",
    "prepare_hybrid_mxfp4_lm_head",
    "release_hybrid_mxfp4_lm_head",
    "warmup_hybrid_mxfp4_lm_head_kernels",
]
