# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for the optional ``mxfp6-sm120`` W6A8 backend."""

import importlib
from collections.abc import Iterable
from types import ModuleType
from typing import TypeAlias

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_W6A8Problem = tuple[int, int, torch.Tensor, torch.Tensor]
_W6A8Models: TypeAlias = torch.nn.Module | Iterable[torch.nn.Module]

_REQUIRED_API = (
    "MXFP8Tensor",
    "PackedMXFP6Tensor",
    "autotune_w6a8",
    "begin_workspace_planning",
    "finalize_workspace_planning",
    "gemm_w6a8",
    "is_available",
    "load_library",
    "pack_scales",
    "workspace_stats",
)

_MOE_REQUIRED_API = (
    "Qwen35GroupedWorkspace",
    "array_gemm_w6a8_reduce_out",
    "grouped_gemm_w6a8_out",
    "moe_reduce_out",
    "qwen35_grouped_gemm_out",
    "qwen35_grouped_moe_out",
    "qwen35_grouped_reduce_out",
    "qwen35_grouped_workspace_shapes",
    "qwen35_router_quant_out",
    "qwen35_w1_splitk_silu_mxfp8_out",
    "qwen35_w2_splitk_reduce_out",
    "quantize_mxfp8_logical",
    "route_mxfp8_out",
    "silu_and_mul_mxfp8_grouped",
    "to_mma_k64_weight",
)


def _mxfp6_sm120_qwen35_grouped_moe_impl(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
) -> None:
    mxfp6 = _import_mxfp6()
    workspace = mxfp6.Qwen35GroupedWorkspace.from_storage(
        output,
        workspace13,
        workspace2,
    )
    mxfp6.qwen35_grouped_moe_out(
        workspace,
        output,
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
    )


def _mxfp6_sm120_qwen35_grouped_moe_fake(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
) -> None:
    del output, hidden_states, w1, w1_scale, w2, w2_scale
    del topk_weights, topk_ids, workspace13, workspace2


direct_register_custom_op(
    op_name="mxfp6_sm120_qwen35_grouped_moe",
    op_func=_mxfp6_sm120_qwen35_grouped_moe_impl,
    mutates_args=["output", "workspace13", "workspace2"],
    fake_impl=_mxfp6_sm120_qwen35_grouped_moe_fake,
)


def mxfp6_sm120_qwen35_grouped_moe(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w1_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
) -> None:
    """Run the package-owned Qwen3.5 grouped routed schedule."""
    torch.ops.vllm.mxfp6_sm120_qwen35_grouped_moe(
        output,
        hidden_states,
        w1,
        w1_scale,
        w2,
        w2_scale,
        topk_weights,
        topk_ids,
        workspace13,
        workspace2,
    )


def _import_mxfp6() -> ModuleType:
    return importlib.import_module("mxfp6")


def is_mxfp6_sm120_available() -> bool:
    """Return whether the native extension can run on the current device."""
    if not current_platform.is_cuda() or not current_platform.is_device_capability(120):
        return False

    try:
        mxfp6 = _import_mxfp6()
        if not all(hasattr(mxfp6, name) for name in _REQUIRED_API):
            logger.debug("mxfp6-sm120 does not provide the required W6A8 API")
            return False
        mxfp6.load_library()
        return True
    except Exception:
        logger.debug("mxfp6-sm120 is unavailable", exc_info=True)
        return False


def is_mxfp6_sm120_moe_available() -> bool:
    """Return whether the optional backend provides its native MoE API."""
    if not is_mxfp6_sm120_available():
        return False
    try:
        mxfp6 = _import_mxfp6()
        return all(hasattr(mxfp6, name) for name in _MOE_REQUIRED_API)
    except Exception:
        logger.debug("mxfp6-sm120 MoE API is unavailable", exc_info=True)
        return False


def pack_mxfp6_sm120_scales(logical_scales: torch.Tensor) -> torch.Tensor:
    """Convert logical UE8M0 scales to the CUTLASS SM120 layout."""
    return _import_mxfp6().pack_scales(logical_scales.contiguous())


def pack_mxfp6_sm120_k64_weight(weight: torch.Tensor) -> torch.Tensor:
    """Convert packed ``[E,N,64]`` weights to the compact SM120 TMA layout."""
    return _import_mxfp6().to_mma_k64_weight(weight)


def _mxfp6_sm120_quantize_mxfp8_impl(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _import_mxfp6().quantize_mxfp8_logical(input)


def _mxfp6_sm120_quantize_mxfp8_fake(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (input.shape[0], input.shape[1] // 32),
        dtype=torch.uint8,
        device=input.device,
    )
    return values, scales


direct_register_custom_op(
    op_name="mxfp6_sm120_quantize_mxfp8",
    op_func=_mxfp6_sm120_quantize_mxfp8_impl,
    mutates_args=[],
    fake_impl=_mxfp6_sm120_quantize_mxfp8_fake,
)


def mxfp6_sm120_quantize_mxfp8(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an activation with row-major per-32 UE8M0 scales."""
    return torch.ops.vllm.mxfp6_sm120_quantize_mxfp8(input)


def _mxfp6_sm120_route_impl(
    permuted_activation: torch.Tensor,
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _import_mxfp6().route_mxfp8_out(
        activation,
        logical_scales,
        topk_ids,
        expert_map,
        local_experts,
        permuted_activation,
    )


def _mxfp6_sm120_route_fake(
    permuted_activation: torch.Tensor,
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del permuted_activation, logical_scales, expert_map
    routes = activation.shape[0] * topk_ids.shape[1]
    scale_rows = routes + local_experts * 127
    packed_scales = torch.empty(
        scale_rows * (activation.shape[1] // 32),
        dtype=torch.uint8,
        device=activation.device,
    )
    expert_offsets = torch.empty(
        local_experts + 1,
        dtype=torch.int64,
        device=activation.device,
    )
    scale_offsets = torch.empty_like(expert_offsets)
    inverse_permutation = torch.empty(
        routes,
        dtype=torch.int32,
        device=activation.device,
    )
    return (
        packed_scales,
        expert_offsets,
        scale_offsets,
        inverse_permutation,
    )


direct_register_custom_op(
    op_name="mxfp6_sm120_route",
    op_func=_mxfp6_sm120_route_impl,
    mutates_args=["permuted_activation"],
    fake_impl=_mxfp6_sm120_route_fake,
)


def mxfp6_sm120_route(
    permuted_activation: torch.Tensor,
    activation: torch.Tensor,
    logical_scales: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route MXFP8 rows and directly produce grouped CUTLASS scales."""
    return torch.ops.vllm.mxfp6_sm120_route(
        permuted_activation,
        activation,
        logical_scales,
        topk_ids,
        expert_map,
        local_experts,
    )


def _mxfp6_sm120_quantize_mxfp8_packed_impl(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    operand = _import_mxfp6().quantize_mxfp8(input)
    return operand.values.view(torch.float8_e4m3fn), operand.scales


def _mxfp6_sm120_quantize_mxfp8_packed_fake(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    padded_rows = (input.shape[0] + 127) // 128 * 128
    scales = torch.empty(
        padded_rows * (input.shape[1] // 32),
        dtype=torch.uint8,
        device=input.device,
    )
    return values, scales


direct_register_custom_op(
    op_name="mxfp6_sm120_quantize_mxfp8_packed",
    op_func=_mxfp6_sm120_quantize_mxfp8_packed_impl,
    mutates_args=[],
    fake_impl=_mxfp6_sm120_quantize_mxfp8_packed_fake,
)


def mxfp6_sm120_quantize_mxfp8_packed(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an activation directly to the SM120 CUTLASS scale layout."""
    return torch.ops.vllm.mxfp6_sm120_quantize_mxfp8_packed(input)


def _mxfp6_sm120_grouped_gemm_impl(
    output: torch.Tensor,
    activation: torch.Tensor,
    packed_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    mxfp6 = _import_mxfp6()
    mxfp6.grouped_gemm_w6a8_out(
        output,
        activation,
        packed_scales,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
    )


def _mxfp6_sm120_grouped_gemm_fake(
    output: torch.Tensor,
    activation: torch.Tensor,
    packed_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    del output, activation, packed_scales, weight, weight_scales
    del expert_offsets, scale_offsets


direct_register_custom_op(
    op_name="mxfp6_sm120_grouped_gemm",
    op_func=_mxfp6_sm120_grouped_gemm_impl,
    mutates_args=["output"],
    fake_impl=_mxfp6_sm120_grouped_gemm_fake,
)


def mxfp6_sm120_grouped_gemm(
    output: torch.Tensor,
    activation: torch.Tensor,
    packed_scales: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    """Run a native grouped W6A8 GEMM with routed packed scales."""
    torch.ops.vllm.mxfp6_sm120_grouped_gemm(
        output,
        activation,
        packed_scales,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
    )


def _mxfp6_sm120_silu_grouped_gemm_impl(
    output: torch.Tensor,
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    mxfp6 = _import_mxfp6()
    activation, scales, scale_offsets = mxfp6.silu_and_mul_mxfp8_grouped(
        gate_up, expert_offsets, scale_offsets
    )
    mxfp6.grouped_gemm_w6a8_out(
        output,
        activation,
        scales,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
    )


def _mxfp6_sm120_silu_grouped_gemm_fake(
    output: torch.Tensor,
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    del output, gate_up, weight, weight_scales
    del expert_offsets, scale_offsets


direct_register_custom_op(
    op_name="mxfp6_sm120_silu_grouped_gemm",
    op_func=_mxfp6_sm120_silu_grouped_gemm_impl,
    mutates_args=["output"],
    fake_impl=_mxfp6_sm120_silu_grouped_gemm_fake,
)


def mxfp6_sm120_silu_grouped_gemm(
    output: torch.Tensor,
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scales: torch.Tensor,
    expert_offsets: torch.Tensor,
    scale_offsets: torch.Tensor,
) -> None:
    """Fuse SiLU-and-mul quantization with the second grouped W6A8 GEMM."""
    torch.ops.vllm.mxfp6_sm120_silu_grouped_gemm(
        output,
        gate_up,
        weight,
        weight_scales,
        expert_offsets,
        scale_offsets,
    )


def _mxfp6_sm120_reduce_impl(
    output: torch.Tensor,
    routed_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> None:
    _import_mxfp6().moe_reduce_out(
        output,
        routed_output,
        topk_weights,
        inverse_permutation,
    )


def _mxfp6_sm120_reduce_fake(
    output: torch.Tensor,
    routed_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> None:
    del output, routed_output, topk_weights, inverse_permutation


direct_register_custom_op(
    op_name="mxfp6_sm120_reduce",
    op_func=_mxfp6_sm120_reduce_impl,
    mutates_args=["output"],
    fake_impl=_mxfp6_sm120_reduce_fake,
)


def mxfp6_sm120_reduce(
    output: torch.Tensor,
    routed_output: torch.Tensor,
    topk_weights: torch.Tensor,
    inverse_permutation: torch.Tensor,
) -> None:
    """Reduce routed rows using only the independent mxfp6-sm120 package."""
    torch.ops.vllm.mxfp6_sm120_reduce(
        output,
        routed_output,
        topk_weights,
        inverse_permutation,
    )


def _mxfp6_sm120_gemm_impl(
    quantized_x: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    mxfp6 = _import_mxfp6()
    mxfp6.load_library()
    rows, input_features = quantized_x.shape
    output = torch.ops.mxfp6.gemm_w6a8(
        quantized_x.view(torch.uint8),
        weight,
        input_scale,
        weight_scale,
        rows,
        output_features,
        input_features,
        1.0,
        output_dtype,
    )
    return output


def _mxfp6_sm120_gemm_fake(
    quantized_x: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    del input_scale, weight, weight_scale
    return torch.empty(
        (quantized_x.shape[0], output_features),
        dtype=output_dtype,
        device=quantized_x.device,
    )


direct_register_custom_op(
    op_name="mxfp6_sm120_gemm",
    op_func=_mxfp6_sm120_gemm_impl,
    mutates_args=[],
    fake_impl=_mxfp6_sm120_gemm_fake,
)


def mxfp6_sm120_gemm(
    quantized_x: torch.Tensor,
    input_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output_features: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Run native W6A8 GEMM with an already-quantized MXFP8 activation."""
    return torch.ops.vllm.mxfp6_sm120_gemm(
        quantized_x,
        input_scale,
        weight,
        weight_scale,
        output_features,
        output_dtype,
    )


def _collect_w6a8_problems(models: _W6A8Models) -> list[_W6A8Problem]:
    problems: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    if isinstance(models, torch.nn.Module):
        models = (models,)

    for model in models:
        for layer in model.modules():
            scheme = getattr(layer, "scheme", None)
            if not getattr(scheme, "use_mxfp6_sm120", False):
                continue

            weight = layer.weight
            weight_scale = layer.weight_scale
            output_features, packed_features = weight.shape
            input_features = packed_features * 4 // 3
            problems.setdefault(
                (output_features, input_features), (weight, weight_scale)
            )

    return [
        (output_features, input_features, weight, weight_scale)
        for (output_features, input_features), (
            weight,
            weight_scale,
        ) in problems.items()
    ]


def _normalize_warmup_inputs(
    token_sizes: list[int],
    dtype: torch.dtype,
) -> tuple[list[int], torch.dtype]:
    sizes = sorted({size for size in token_sizes if size > 0}, reverse=True)
    if dtype not in (torch.float16, torch.bfloat16):
        dtype = torch.bfloat16
    return sizes, dtype


def _run_w6a8_warmup(
    mxfp6: ModuleType,
    problems: list[_W6A8Problem],
    sizes: list[int],
    dtype: torch.dtype,
    *,
    autotune: bool,
    initial_lane_count: int | None = None,
) -> None:
    for output_features, input_features, weight, weight_scale in problems:
        packed_weight = mxfp6.PackedMXFP6Tensor(
            values=weight,
            scales=weight_scale,
            rows=output_features,
            k=input_features,
        )
        for num_tokens in sizes:
            x = torch.empty(
                (num_tokens, input_features), device=weight.device, dtype=dtype
            ).uniform_(-1.0, 1.0)
            quantized_x = mxfp6.quantize_mxfp8(x)
            if autotune:
                mxfp6.autotune_w6a8(quantized_x, packed_weight, out_dtype=dtype)
            output = mxfp6.gemm_w6a8(quantized_x, packed_weight, out_dtype=dtype)
            del quantized_x, x, output
            if (
                initial_lane_count is not None
                and mxfp6.workspace_stats(weight.device)["lanes"] > initial_lane_count
            ):
                return


@torch.inference_mode()
def warmup_mxfp6_sm120(
    models: _W6A8Models,
    token_sizes: list[int],
    dtype: torch.dtype,
) -> None:
    """Warm and autotune each distinct native W6A8 problem before capture."""
    problems = _collect_w6a8_problems(models)
    if not problems:
        return

    sizes, dtype = _normalize_warmup_inputs(token_sizes, dtype)
    if not sizes:
        return

    mxfp6 = _import_mxfp6()
    workspace_device = problems[0][2].device
    mxfp6.begin_workspace_planning(workspace_device)
    logger.info(
        "Warming mxfp6-sm120 for %d W6A8 shapes and %d token sizes",
        len(problems),
        len(sizes),
    )
    _run_w6a8_warmup(mxfp6, problems, sizes, dtype, autotune=True)

    workspace_stats = mxfp6.finalize_workspace_planning(workspace_device)
    torch.accelerator.synchronize()
    logger.info(
        "Frozen mxfp6-sm120 Stream-K workspace: %d layouts, %.2f MiB "
        "per CUDA stream lane",
        workspace_stats["layouts"],
        workspace_stats["arena_bytes"] / (1 << 20),
    )


@torch.inference_mode()
def warmup_mxfp6_sm120_stream(
    models: _W6A8Models,
    token_sizes: list[int],
    dtype: torch.dtype,
) -> None:
    """Register the current CUDA stream with the frozen Stream-K workspace."""
    problems = _collect_w6a8_problems(models)
    if not problems:
        return

    sizes, dtype = _normalize_warmup_inputs(token_sizes, dtype)
    if not sizes:
        return
    sizes.reverse()

    mxfp6 = _import_mxfp6()
    workspace_device = problems[0][2].device
    workspace_stats = mxfp6.workspace_stats(workspace_device)
    if workspace_stats["layouts"] == 0:
        return

    _run_w6a8_warmup(
        mxfp6,
        problems,
        sizes,
        dtype,
        autotune=False,
        initial_lane_count=workspace_stats["lanes"],
    )
    torch.accelerator.synchronize()
