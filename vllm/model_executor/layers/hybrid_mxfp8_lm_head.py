# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental shape-generic MXFP8 coarse search for BF16 lm heads."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.argmax_triton import (
    indexed_argmax_triton,
    reduce_global_argmax_triton,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    autotune_with_torch_cuda_delay,
    flashinfer_mm_mxfp8,
    has_flashinfer,
)

logger = init_logger(__name__)

_MIN_GEMM_DIMENSION = 128
_WEIGHT_NAME = "_hybrid_mxfp8_lm_head_weight"
_SCALE_NAME = "_hybrid_mxfp8_lm_head_scale"
_STATE_NAME = "_hybrid_mxfp8_lm_head_state"
_BUFFER_NAMES = (_WEIGHT_NAME, _SCALE_NAME)


def _select_indexed_bf16_candidate_tile(
    num_rows: int,
    num_candidates: int,
    input_size: int,
) -> int:
    # Tiling changes the floating-point reduction order. At larger row counts,
    # a handful of near-tied logits can therefore change greedy continuations
    # and downstream MoE routing for much more than the few microseconds saved
    # by the refinement kernel. Keep the original scalar reduction there.
    if num_rows < 16 or num_rows >= 64 or num_candidates < 64 or input_size > 2048:
        return 1
    return 4


@triton.jit
def _indexed_bf16_dot_kernel(
    HIDDEN,
    WEIGHT,
    INDICES,
    OUTPUT,
    HIDDEN_STRIDE_0: tl.constexpr,
    WEIGHT_STRIDE_0: tl.constexpr,
    INDEX_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    INPUT_SIZE: tl.constexpr,
    BLOCK_INPUT_SIZE: tl.constexpr,
):
    pair_id = tl.program_id(0)
    row = pair_id // NUM_CANDIDATES
    candidate = pair_id % NUM_CANDIDATES
    token_id = tl.load(INDICES + row * INDEX_STRIDE_0 + candidate)
    offsets = tl.arange(0, BLOCK_INPUT_SIZE)
    mask = offsets < INPUT_SIZE
    hidden = tl.load(
        HIDDEN + row * HIDDEN_STRIDE_0 + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(
        WEIGHT + token_id * WEIGHT_STRIDE_0 + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    value = tl.sum(hidden * weight, axis=0)
    tl.store(OUTPUT + row * OUTPUT_STRIDE_0 + candidate, value)


@triton.jit
def _tiled_indexed_bf16_dot_kernel(
    HIDDEN,
    WEIGHT,
    INDICES,
    OUTPUT,
    HIDDEN_STRIDE_0: tl.constexpr,
    WEIGHT_STRIDE_0: tl.constexpr,
    INDEX_STRIDE_0: tl.constexpr,
    OUTPUT_STRIDE_0: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    INPUT_SIZE: tl.constexpr,
    BLOCK_INPUT_SIZE: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
):
    row = tl.program_id(0)
    candidates = tl.program_id(1) * BLOCK_CANDIDATES + tl.arange(0, BLOCK_CANDIDATES)
    candidate_mask = candidates < NUM_CANDIDATES
    token_ids = tl.load(
        INDICES + row * INDEX_STRIDE_0 + candidates,
        mask=candidate_mask,
        other=0,
    )

    offsets = tl.arange(0, BLOCK_INPUT_SIZE)
    input_mask = offsets < INPUT_SIZE
    hidden = tl.load(
        HIDDEN + row * HIDDEN_STRIDE_0 + offsets,
        mask=input_mask,
        other=0.0,
    ).to(tl.float32)
    weights = tl.load(
        WEIGHT + token_ids[:, None] * WEIGHT_STRIDE_0 + offsets[None, :],
        mask=candidate_mask[:, None] & input_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    values = tl.sum(weights * hidden[None, :], axis=1)
    tl.store(
        OUTPUT + row * OUTPUT_STRIDE_0 + candidates,
        values,
        mask=candidate_mask,
    )


def indexed_bf16_dot(
    hidden_states: torch.Tensor,
    bf16_weight: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    candidate_tile: int | None = None,
    num_warps: int = 8,
) -> torch.Tensor:
    """Compute selected BF16 logits without materializing gathered weights."""
    assert hidden_states.ndim == 2
    assert bf16_weight.ndim == 2
    assert candidate_indices.ndim == 2
    assert hidden_states.shape[0] == candidate_indices.shape[0]
    assert hidden_states.shape[1] == bf16_weight.shape[1]
    assert hidden_states.dtype == torch.bfloat16
    assert bf16_weight.dtype == torch.bfloat16
    assert hidden_states.is_cuda
    assert bf16_weight.is_cuda
    assert candidate_indices.is_cuda
    assert hidden_states.is_contiguous()
    assert bf16_weight.is_contiguous()
    assert candidate_indices.is_contiguous()
    if num_warps not in (4, 8):
        raise ValueError(f"num_warps must be 4 or 8; got {num_warps}")

    output = torch.empty(
        candidate_indices.shape,
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    num_rows, num_candidates = candidate_indices.shape
    input_size = hidden_states.shape[1]
    block_input_size = triton.next_power_of_2(input_size)
    if candidate_tile is None:
        candidate_tile = _select_indexed_bf16_candidate_tile(
            num_rows,
            num_candidates,
            input_size,
        )
    if candidate_tile == 1:
        _indexed_bf16_dot_kernel[(num_rows * num_candidates,)](
            hidden_states,
            bf16_weight,
            candidate_indices,
            output,
            HIDDEN_STRIDE_0=hidden_states.stride(0),
            WEIGHT_STRIDE_0=bf16_weight.stride(0),
            INDEX_STRIDE_0=candidate_indices.stride(0),
            OUTPUT_STRIDE_0=output.stride(0),
            NUM_CANDIDATES=num_candidates,
            INPUT_SIZE=input_size,
            BLOCK_INPUT_SIZE=block_input_size,
            num_warps=num_warps,
        )
    else:
        if candidate_tile not in (2, 4, 8):
            raise ValueError(
                f"candidate_tile must be one of 1, 2, 4, or 8; got {candidate_tile}"
            )
        _tiled_indexed_bf16_dot_kernel[
            (num_rows, triton.cdiv(num_candidates, candidate_tile))
        ](
            hidden_states,
            bf16_weight,
            candidate_indices,
            output,
            HIDDEN_STRIDE_0=hidden_states.stride(0),
            WEIGHT_STRIDE_0=bf16_weight.stride(0),
            INDEX_STRIDE_0=candidate_indices.stride(0),
            OUTPUT_STRIDE_0=output.stride(0),
            NUM_CANDIDATES=num_candidates,
            INPUT_SIZE=input_size,
            BLOCK_INPUT_SIZE=block_input_size,
            BLOCK_CANDIDATES=candidate_tile,
            num_warps=num_warps,
        )
    return output


def select_lm_head_candidates(
    coarse_logits: torch.Tensor,
    candidates: int,
) -> torch.Tensor:
    """Select an unsorted exact top-k set with the fastest available backend."""
    if envs.VLLM_HYBRID_MXFP8_LM_HEAD_USE_FLASHINFER_TOPK and has_flashinfer():
        from flashinfer import top_k as flashinfer_top_k

        logger.info_once(
            "Hybrid MXFP8 lm-head is using FlashInfer exact unsorted top-k "
            "candidate selection; FlashInfer auto-dispatches its backend."
        )
        _, candidate_indices = flashinfer_top_k(
            coarse_logits,
            candidates,
            sorted=False,
        )
        return candidate_indices
    return torch.topk(
        coarse_logits,
        candidates,
        dim=-1,
        sorted=False,
    ).indices


@dataclass
class HybridMxfp8LmHead:
    """Persistent MXFP8 weight copy plus BF16 candidate refinement."""

    weight: torch.Tensor
    scale: torch.Tensor
    input_size: int
    output_size: int
    candidates: int
    max_rows: int

    def can_use(
        self,
        hidden_states: torch.Tensor,
        *,
        bf16_weight: torch.Tensor,
        active_vocab_size: int,
        top_k: int,
    ) -> bool:
        return not (
            hidden_states.ndim != 2
            or hidden_states.dtype != torch.bfloat16
            or not hidden_states.is_cuda
            or not hidden_states.is_contiguous()
            or hidden_states.shape[1] != self.input_size
            or bf16_weight.dtype != torch.bfloat16
            or bf16_weight.device != hidden_states.device
            or not bf16_weight.is_contiguous()
            or bf16_weight.shape != (self.output_size, self.input_size)
            or active_vocab_size > self.output_size
            or top_k > self.candidates
            or active_vocab_size < self.candidates
            or hidden_states.shape[0] > self.max_rows
        )

    def coarse_logits(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_q, hidden_scale = mxfp8_e4m3_quantize(
            hidden_states,
            is_sf_swizzled_layout=True,
            alignment=MXFP8_BLOCK_SIZE,
        )
        logits = flashinfer_mm_mxfp8(
            hidden_q,
            self.weight,
            hidden_scale,
            self.scale,
            torch.bfloat16,
            backend="cutlass",
        )
        logits = logits[:, : self.output_size]
        if bias is not None:
            logits += bias
        return logits

    def select_candidates(self, coarse_logits: torch.Tensor) -> torch.Tensor:
        return select_lm_head_candidates(coarse_logits, self.candidates)

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
def warmup_hybrid_mxfp8_lm_head_kernels(
    state: HybridMxfp8LmHead,
    bf16_weight: torch.Tensor,
    tp_size: int,
) -> None:
    """Move MXFP8, selector, and compact-reduction JIT work into loading."""
    hidden_states = torch.zeros(
        (1, state.input_size),
        dtype=torch.bfloat16,
        device=state.weight.device,
    )
    coarse_logits = state.coarse_logits(hidden_states, None)
    candidate_indices = state.select_candidates(coarse_logits)
    exact_logits = state.refine_logits(
        hidden_states,
        bf16_weight,
        candidate_indices,
        None,
    )
    indexed_argmax_triton(exact_logits, candidate_indices)
    if state.max_rows >= 16:
        tiled_rows = min(state.max_rows, 16)
        tiled_hidden_states = hidden_states.expand(tiled_rows, -1).contiguous()
        tiled_candidate_indices = candidate_indices.expand(tiled_rows, -1).contiguous()
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


def autotune_row_buckets(max_rows: int) -> tuple[int, ...]:
    """Row shapes whose first runtime hit would trigger live FlashInfer tuning.

    The CUTLASS mm inside :meth:`HybridMxfp8LmHead.coarse_logits` keys its
    tactic cache with FlashInfer's hybrid num-tokens buckets (power-of-two up
    to 256, then 256 steps). A first runtime call on any new bucket costs a
    live autotune pass, so mirror FlashInfer's bucket list here and tune all
    of them during loading instead.
    """
    try:
        from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

        return get_hybrid_num_tokens_buckets(max_rows)
    except Exception:
        buckets = [b for b in (1, 2, 4, 8, 16, 32, 64, 128, 256) if b <= max_rows]
        rows = 512
        while rows <= max_rows:
            buckets.append(rows)
            rows += 256
        if not buckets:
            buckets.append(max_rows)
        return tuple(sorted(set(buckets)))


@torch.inference_mode()
def autotune_hybrid_mxfp8_lm_head(
    state: HybridMxfp8LmHead,
    bf16_weight: torch.Tensor,
    row_shapes: tuple[int, ...] | None = None,
) -> tuple[float, tuple[int, ...]]:
    """Tune CUTLASS tactics for every row shape that can appear at runtime.

    Args:
        row_shapes: Concrete row counts to tune (e.g. CUDA graph capture
            sizes). Defaults to FlashInfer's hybrid num-token buckets.
    """
    if row_shapes is None:
        row_shapes = autotune_row_buckets(state.max_rows)
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
            state.refine_logits(
                hidden,
                bf16_weight,
                candidate_indices,
                None,
            )
    torch.accelerator.synchronize()
    return perf_counter() - started, row_shapes


def prepare_hybrid_mxfp8_lm_head(
    layer: torch.nn.Module,
    *,
    candidates: int,
) -> bool:
    """Create an MXFP8 block-scaled copy, or leave the original path intact."""
    if hasattr(layer, _STATE_NAME):
        return True
    weight = layer.weight
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
        or weight.shape[1] % MXFP8_BLOCK_SIZE
    ):
        logger.warning_once(
            "Hybrid MXFP8 lm-head does not support weight %s (%s on %s); "
            "falling back to the original lm-head implementation.",
            tuple(weight.shape),
            weight.dtype,
            weight.device,
        )
        return False
    if candidates <= 0 or candidates > weight.shape[0]:
        logger.warning_once(
            "Hybrid MXFP8 lm-head candidate count %d is outside [1, %d]; "
            "falling back to the original lm-head implementation.",
            candidates,
            weight.shape[0],
        )
        return False
    max_rows = envs.VLLM_HYBRID_MXFP8_LM_HEAD_MAX_ROWS
    if max_rows < 1:
        logger.warning_once(
            "Hybrid MXFP8 lm-head max rows must be positive, got %d; "
            "falling back to the original lm-head implementation.",
            max_rows,
        )
        return False

    quantized_output_size = (weight.shape[0] + MXFP8_BLOCK_SIZE - 1) // MXFP8_BLOCK_SIZE
    quantized_output_size *= MXFP8_BLOCK_SIZE
    weight_for_quant = weight
    if quantized_output_size != weight.shape[0]:
        weight_for_quant = F.pad(
            weight,
            (0, 0, 0, quantized_output_size - weight.shape[0]),
        )
    quantized, scale = mxfp8_e4m3_quantize(
        weight_for_quant,
        is_sf_swizzled_layout=True,
        alignment=MXFP8_BLOCK_SIZE,
    )

    layer.register_buffer(_WEIGHT_NAME, quantized, persistent=False)
    layer.register_buffer(_SCALE_NAME, scale, persistent=False)
    state = HybridMxfp8LmHead(
        weight=getattr(layer, _WEIGHT_NAME),
        scale=getattr(layer, _SCALE_NAME),
        input_size=weight.shape[1],
        output_size=weight.shape[0],
        candidates=candidates,
        max_rows=max_rows,
    )
    setattr(layer, _STATE_NAME, state)
    extra_mib = sum(getattr(layer, name).nbytes for name in _BUFFER_NAMES) / (
        1024 * 1024
    )
    logger.info_once(
        "Prepared shape-generic hybrid MXFP8 lm-head for weight %s with %d "
        "candidates and M<=%d (%.2f MiB persistent overhead; FlashInfer "
        "tactic autotune and kernel warmup run in the startup warmup stage).",
        tuple(weight.shape),
        candidates,
        max_rows,
        extra_mib,
    )
    return True


def get_hybrid_mxfp8_lm_head(layer: torch.nn.Module) -> HybridMxfp8LmHead | None:
    return getattr(layer, _STATE_NAME, None)


def release_hybrid_mxfp8_lm_head(layer: torch.nn.Module) -> int:
    """Drop a prepared MXFP8 copy from an lm head being discarded."""
    released_bytes = 0
    for name in _BUFFER_NAMES:
        value = getattr(layer, name, None)
        if isinstance(value, torch.Tensor):
            released_bytes += value.nbytes

    if hasattr(layer, _STATE_NAME):
        delattr(layer, _STATE_NAME)
    for name in _BUFFER_NAMES:
        if hasattr(layer, name):
            delattr(layer, name)
    return released_bytes
