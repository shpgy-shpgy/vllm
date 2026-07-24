# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for the optional ``mxfp6-sm120`` W6A8 backend."""

import importlib
from types import ModuleType

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_W6A8Problem = tuple[int, int, torch.Tensor, torch.Tensor]

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


def pack_mxfp6_sm120_scales(logical_scales: torch.Tensor) -> torch.Tensor:
    """Convert logical UE8M0 scales to the CUTLASS SM120 layout."""
    return _import_mxfp6().pack_scales(logical_scales.contiguous())


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


def _collect_w6a8_problems(model: torch.nn.Module) -> list[_W6A8Problem]:
    problems: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in model.modules():
        scheme = getattr(layer, "scheme", None)
        if not getattr(scheme, "use_mxfp6_sm120", False):
            continue

        weight = layer.weight
        weight_scale = layer.weight_scale
        output_features, packed_features = weight.shape
        input_features = packed_features * 4 // 3
        problems.setdefault((output_features, input_features), (weight, weight_scale))

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
            quantized_values, input_scale = mxfp8_e4m3_quantize(
                x, is_sf_swizzled_layout=True
            )
            quantized_x = mxfp6.MXFP8Tensor(
                values=quantized_values.view(torch.uint8),
                scales=input_scale,
                rows=num_tokens,
                k=input_features,
            )
            if autotune:
                mxfp6.autotune_w6a8(quantized_x, packed_weight, out_dtype=dtype)
            output = mxfp6.gemm_w6a8(quantized_x, packed_weight, out_dtype=dtype)
            del quantized_values, input_scale, quantized_x
            del x, output
            if (
                initial_lane_count is not None
                and mxfp6.workspace_stats(weight.device)["lanes"] > initial_lane_count
            ):
                return


@torch.inference_mode()
def warmup_mxfp6_sm120(
    model: torch.nn.Module,
    token_sizes: list[int],
    dtype: torch.dtype,
) -> None:
    """Warm and autotune each distinct native W6A8 problem before capture."""
    problems = _collect_w6a8_problems(model)
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
    model: torch.nn.Module,
    token_sizes: list[int],
    dtype: torch.dtype,
) -> None:
    """Register the current CUDA stream with the frozen Stream-K workspace."""
    problems = _collect_w6a8_problems(model)
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
