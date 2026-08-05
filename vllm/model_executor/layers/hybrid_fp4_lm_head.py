# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental shape-generic NVFP4 coarse search for BF16 lm heads."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.argmax_triton import (
    indexed_argmax_triton,
    reduce_global_argmax_triton,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    cutlass_fp4_supported,
    pad_nvfp4_weight_for_cutlass,
    slice_nvfp4_output,
)
from vllm.scalar_type import scalar_types
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer

logger = init_logger(__name__)

_BLOCK_SIZE = 16
_FP4_MAX = scalar_types.float4_e2m1f.max()
_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
_MAX_REFINEMENT_ELEMENTS = 48 * 1024 * 1024
_WEIGHT_NAME = "_hybrid_fp4_lm_head_weight"
_SCALE_NAME = "_hybrid_fp4_lm_head_scale"
_INPUT_SCALE_NAME = "_hybrid_fp4_lm_head_input_scale"
_ALPHA_NAME = "_hybrid_fp4_lm_head_alpha"
_STATE_NAME = "_hybrid_fp4_lm_head_state"
_BUFFER_NAMES = (_WEIGHT_NAME, _SCALE_NAME, _INPUT_SCALE_NAME, _ALPHA_NAME)


def select_lm_head_candidates(
    coarse_logits: torch.Tensor,
    candidates: int,
) -> torch.Tensor:
    """Select an unsorted exact top-k set with the fastest available backend."""
    if envs.VLLM_HYBRID_FP4_LM_HEAD_USE_FLASHINFER_TOPK and has_flashinfer():
        from flashinfer import top_k as flashinfer_top_k

        logger.info_once(
            "Hybrid NVFP4 lm-head is using FlashInfer radix top-k candidate "
            "selection; the first use may JIT-compile the kernel."
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
class HybridFp4LmHead:
    """Persistent NVFP4 weight copy plus BF16 candidate refinement."""

    weight: torch.Tensor
    scale: torch.Tensor
    input_scale: torch.Tensor
    alpha: torch.Tensor
    input_size: int
    output_size: int
    weights_padding_bytes: int
    candidates: int

    def can_use(
        self,
        hidden_states: torch.Tensor,
        *,
        bf16_weight: torch.Tensor,
        active_vocab_size: int,
        top_k: int,
    ) -> bool:
        if (
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
        ):
            return False
        refinement_elements = (
            hidden_states.shape[0] * self.candidates * self.input_size
        )
        return refinement_elements <= _MAX_REFINEMENT_ELEMENTS

    def coarse_logits(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_q, hidden_scale = ops.scaled_fp4_quant(
            hidden_states,
            self.input_scale,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cutlass",
            padded_n=self.input_size + self.weights_padding_bytes * 2,
        )
        logits = flashinfer_scaled_fp4_mm(
            hidden_q,
            self.weight,
            hidden_scale,
            self.scale,
            self.alpha,
            torch.bfloat16,
            backend="cutlass",
        )
        logits = slice_nvfp4_output(logits, self.output_size)
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
        selected_weight = bf16_weight[candidate_indices]
        logits = torch.bmm(
            selected_weight,
            hidden_states.unsqueeze(-1),
        ).squeeze(-1)
        if bias is not None:
            logits += bias[candidate_indices]
        return logits


@torch.inference_mode()
def _warmup_hybrid_fp4_lm_head_kernels(
    state: HybridFp4LmHead,
    tp_size: int,
) -> None:
    """Move NVFP4, selector, and compact-reduction JIT work into loading."""
    hidden_states = torch.zeros(
        (1, state.input_size),
        dtype=torch.bfloat16,
        device=state.weight.device,
    )
    coarse_logits = state.coarse_logits(hidden_states, None)
    candidate_indices = state.select_candidates(coarse_logits)
    exact_logits = torch.zeros(
        (1, state.candidates),
        dtype=torch.bfloat16,
        device=state.weight.device,
    )
    indexed_argmax_triton(exact_logits, candidate_indices)
    if tp_size > 1:
        gathered_pairs = torch.zeros(
            (1, tp_size * 2),
            dtype=torch.float32,
            device=state.weight.device,
        )
        reduce_global_argmax_triton(gathered_pairs, tp_size=tp_size)


def prepare_hybrid_fp4_lm_head(
    layer: torch.nn.Module,
    *,
    candidates: int,
    input_amax: float,
) -> bool:
    """Create an NVFP4 block-scaled copy, or leave the original path intact."""
    if hasattr(layer, _STATE_NAME):
        return True
    weight = layer.weight
    if (
        not has_flashinfer()
        or not cutlass_fp4_supported()
        or weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
        or getattr(weight, "_vllm_is_uva_offloaded", False)
        or weight.shape[1] % _BLOCK_SIZE
    ):
        logger.warning_once(
            "Hybrid NVFP4 lm-head does not support weight %s (%s on %s); "
            "falling back to the original lm-head implementation.",
            tuple(weight.shape),
            weight.dtype,
            weight.device,
        )
        return False
    if candidates <= 0 or candidates > weight.shape[0]:
        logger.warning_once(
            "Hybrid NVFP4 lm-head candidate count %d is outside [1, %d]; "
            "falling back to the original lm-head implementation.",
            candidates,
            weight.shape[0],
        )
        return False
    if not math.isfinite(input_amax) or input_amax <= 0.0:
        logger.warning_once(
            "Hybrid NVFP4 lm-head input amax must be finite and positive, got "
            "%s; falling back to the original lm-head implementation.",
            input_amax,
        )
        return False

    weight_amax = weight.abs().amax().to(torch.float32).clamp_min_(
        torch.finfo(torch.float32).tiny
    )
    weight_quant_scale = (_FP8_MAX * _FP4_MAX) / weight_amax
    quantized, scale = ops.scaled_fp4_quant(
        weight,
        weight_quant_scale,
        is_sf_swizzled_layout=True,
        backend="flashinfer-cutlass",
    )
    quantized, weights_padding_bytes = pad_nvfp4_weight_for_cutlass(quantized)
    input_quant_scale = torch.full(
        (),
        (_FP8_MAX * _FP4_MAX) / input_amax,
        dtype=torch.float32,
        device=weight.device,
    )
    alpha = (input_quant_scale * weight_quant_scale).reciprocal()

    layer.register_buffer(_WEIGHT_NAME, quantized, persistent=False)
    layer.register_buffer(_SCALE_NAME, scale, persistent=False)
    layer.register_buffer(_INPUT_SCALE_NAME, input_quant_scale, persistent=False)
    layer.register_buffer(_ALPHA_NAME, alpha, persistent=False)
    state = HybridFp4LmHead(
        weight=getattr(layer, _WEIGHT_NAME),
        scale=getattr(layer, _SCALE_NAME),
        input_scale=getattr(layer, _INPUT_SCALE_NAME),
        alpha=getattr(layer, _ALPHA_NAME),
        input_size=weight.shape[1],
        output_size=weight.shape[0],
        weights_padding_bytes=weights_padding_bytes,
        candidates=candidates,
    )
    setattr(layer, _STATE_NAME, state)
    _warmup_hybrid_fp4_lm_head_kernels(
        state,
        tp_size=getattr(layer, "tp_size", 1),
    )
    extra_mib = sum(getattr(layer, name).nbytes for name in _BUFFER_NAMES) / (
        1024 * 1024
    )
    logger.info_once(
        "Prepared shape-generic hybrid NVFP4 lm-head for weight %s with %d "
        "candidates, input amax %.3f (%.2f MiB persistent overhead).",
        tuple(weight.shape),
        candidates,
        input_amax,
        extra_mib,
    )
    return True


def get_hybrid_fp4_lm_head(layer: torch.nn.Module) -> HybridFp4LmHead | None:
    return getattr(layer, _STATE_NAME, None)


def release_hybrid_fp4_lm_head(layer: torch.nn.Module) -> int:
    """Drop a prepared NVFP4 copy from an lm head being discarded."""
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
