# SPDX-License-Identifier: Apache-2.0
"""Shape-gated native MXFP8 dispatch for the common Qwen3.5 decode path.

The normal ``auto`` MXFP8 selection uses FlashInfer CUTLASS for this model.
The SM120 native W8A8 kernel is materially faster for the decode-sized
Qwen3.5-4B projections at M=4..64, including the K=4096/9216 down and output
projections.  M=1 stays on the tuned FlashInfer B12x path.  This adapter keeps
the selected FlashInfer kernel class (so other integrations, including the
optional dual adapter, can wrap it) and redirects only the validated
model/shape range.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_MODEL_HIDDEN_SIZE = 2560
_MODEL_INTERMEDIATE_SIZE = 9216
_NATIVE_MIN_M = 4
_NATIVE_MAX_M = 64

# The SM120 native kernel is not uniformly best across the four projection
# shapes.  In particular, the QKVZ/GateUp tile selected for M=32 loses to the
# tuned FlashInfer B12x decode kernel, while the same native path remains the
# better choice for the other capture sizes.  Keep this policy inside the
# opaque dispatcher so M stays runtime-dynamic to TorchInductor/CUDA Graphs.
_FLASHINFER_M_BY_SHAPE = {
    (12_288, 2_560): frozenset({32}),  # QKVZ
    (18_432, 2_560): frozenset({32}),  # GateUp
}

_INSTALLED = False
_OP_REGISTERED = False
_ORIGINAL_APPLY: Any = None


def _eligible_config(model: torch.nn.Module, config) -> bool:
    """Return whether the common public native profile is safe to install."""
    del model
    if not envs.VLLM_MXFP6_SM120_MXFP8_NATIVE:
        return False
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability(120)
        and config.use_v2_model_runner
        and config.speculative_config is None
        and config.lora_config is None
        and config.model_config is not None
        and config.model_config.dtype == torch.bfloat16
        and config.model_config.quantization == "compressed-tensors"
        and getattr(config.kernel_config, "linear_backend", None) == "auto"
        and config.parallel_config.tensor_parallel_size == 1
    ):
        return False

    hf = config.model_config.hf_text_config
    return bool(
        getattr(hf, "model_type", None) == "qwen3_5_text"
        and getattr(hf, "hidden_size", None) == _MODEL_HIDDEN_SIZE
        and getattr(hf, "intermediate_size", None) == _MODEL_INTERMEDIATE_SIZE
    )


def _kernel(layer: torch.nn.Module):
    method = getattr(layer, "quant_method", None)
    scheme = getattr(layer, "scheme", None) or getattr(method, "scheme", None)
    return getattr(scheme, "kernel", None) or getattr(method, "kernel", None)


def _eligible_layer(layer: torch.nn.Module) -> bool:
    weight = getattr(layer, "weight", None)
    weight_scale = getattr(layer, "weight_scale", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(
        weight_scale, torch.Tensor
    ):
        return False
    if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
        return False
    n, k = (int(value) for value in weight.shape)
    return bool(
        n >= 128
        and k >= 128
        and n % 128 == 0
        and k % 128 == 0
        and k % 32 == 0
        and weight.is_cuda
        and weight.is_contiguous()
        and weight_scale.dtype == torch.uint8
        and weight_scale.is_cuda
        and weight_scale.device == weight.device
        and weight_scale.is_contiguous()
        and weight_scale.numel() >= n * (k // 32)
    )


def _use_native_shape(m: int, n: int, k: int) -> bool:
    """Return the measured native policy for one projection shape."""
    if not _NATIVE_MIN_M <= m <= _NATIVE_MAX_M:
        return False
    return m not in _FLASHINFER_M_BY_SHAPE.get((n, k), ())


def _public_native_impl(
    input_: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """Dispatch native/FI while keeping M dynamic to TorchInductor.

    This function is behind an opaque custom-op boundary.  It is therefore
    safe to inspect the runtime M here; doing the same check in
    ``apply_weights`` would make Dynamo specialize the compiled graph to the
    first profile shape and bypass the native path during CUDA-graph replay.
    """
    if not input_.is_contiguous():
        input_ = input_.contiguous()

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    input_mxfp8, input_scale = mxfp8_e4m3_quantize(
        input_, is_sf_swizzled_layout=True
    )
    m = int(input_.shape[0])
    if _use_native_shape(m, int(weight.shape[0]), int(weight.shape[1])):
        return torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm_pdl(
            input_mxfp8,
            weight,
            input_scale,
            weight_scale,
        )

    from vllm.utils import flashinfer as vllm_flashinfer

    return vllm_flashinfer.mm_mxfp8(
        input_mxfp8,
        weight.t(),
        input_scale,
        weight_scale,
        out_dtype=torch.bfloat16,
        backend=backend,
    )


def _public_native_fake(
    input_: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    del weight_scale, backend
    return torch.empty(
        (input_.shape[0], weight.shape[0]),
        device=input_.device,
        dtype=torch.bfloat16,
    )


def _register_op() -> None:
    global _OP_REGISTERED
    if _OP_REGISTERED:
        return
    direct_register_custom_op(
        op_name="mxfp6_sm120_public_mxfp8_gemm",
        op_func=_public_native_impl,
        mutates_args=[],
        fake_impl=_public_native_fake,
    )
    _OP_REGISTERED = True


def _install_apply() -> None:
    global _INSTALLED, _ORIGINAL_APPLY
    if _INSTALLED:
        return

    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )

    _ORIGINAL_APPLY = FlashInferCutlassMxfp8LinearKernel.apply_weights

    @wraps(_ORIGINAL_APPLY)
    def apply_weights(self, layer, x, bias=None):
        if not getattr(layer, "_mxfp6_sm120_public_native", False):
            return _ORIGINAL_APPLY(self, layer, x, bias)
        if (
            x.dtype != torch.bfloat16
            or x.device.type != "cuda"
            or x.ndim < 2
        ):
            return _ORIGINAL_APPLY(self, layer, x, bias)

        weight = getattr(layer, "mxfp6_sm120_mxfp8_weight_values", None)
        weight_scale = getattr(layer, "mxfp6_sm120_mxfp8_weight_scales", None)
        if (
            not isinstance(weight, torch.Tensor)
            or not isinstance(weight_scale, torch.Tensor)
            or weight.ndim != 2
            or x.shape[-1] != weight.shape[1]
        ):
            return _ORIGINAL_APPLY(self, layer, x, bias)

        input_2d = x.reshape(-1, x.shape[-1])
        output = torch.ops.vllm.mxfp6_sm120_public_mxfp8_gemm(
            input_2d,
            weight,
            weight_scale,
            getattr(layer, "mxfp8_backend", "b12x-decode"),
        )
        if output.dtype != x.dtype:
            output = output.to(x.dtype)
        output = output.view(*x.shape[:-1], layer.weight.shape[0])
        if bias is not None:
            output = output + bias
        return output

    FlashInferCutlassMxfp8LinearKernel.apply_weights = apply_weights
    FlashInferCutlassMxfp8LinearKernel._mxfp6_sm120_public_native = True
    _INSTALLED = True


def _tag_model(model: torch.nn.Module) -> int:
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )

    count = 0
    for layer in model.modules():
        if not isinstance(_kernel(layer), FlashInferCutlassMxfp8LinearKernel):
            continue
        if not _eligible_layer(layer):
            continue

        layer._mxfp6_sm120_public_native = True
        layer.mxfp6_sm120_mxfp8_weight_values = layer.weight.view(
            torch.float8_e4m3fn
        )
        layer.mxfp6_sm120_mxfp8_weight_scales = layer.weight_scale
        count += 1
    return count


def _warmup_sizes(config) -> tuple[int, ...]:
    sizes = {4, 8, 10, 12, 14, 16, 17, 20, 24, 28, 32, 36, 40, 48, 56, 64}
    capture_sizes = getattr(config.compilation_config, "cudagraph_capture_sizes", None)
    if capture_sizes:
        sizes.update(
            int(size)
            for size in capture_sizes
            if _NATIVE_MIN_M <= int(size) <= _NATIVE_MAX_M
        )
    return tuple(sorted(sizes))


def _warmup_small_workspaces(
    model: torch.nn.Module,
    sizes: tuple[int, ...],
    *,
    use_pdl: bool = False,
) -> int:
    """Warm native workspaces below the legacy helper's M>=17 cutoff."""
    from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import (
        load_mxfp6_sm120_mxfp8_libraries,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    small_sizes = tuple(sorted({int(m) for m in sizes if int(m) < 17}))
    if not small_sizes:
        return 0

    shapes: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in model.modules():
        weight = getattr(layer, "mxfp6_sm120_mxfp8_weight_values", None)
        weight_scale = getattr(layer, "mxfp6_sm120_mxfp8_weight_scales", None)
        if weight is None or weight_scale is None or not weight.is_cuda:
            continue
        n, k = (int(value) for value in weight.shape)
        device_index = weight.device.index
        if device_index is None:
            device_index = torch.accelerator.current_device_index()
        shapes.setdefault((device_index, n, k), (weight, weight_scale))

    if not shapes:
        return 0

    load_mxfp6_sm120_mxfp8_libraries()
    launches = 0
    for (device_index, n, k), (weight, weight_scale) in shapes.items():
        device = torch.device("cuda", device_index)
        for m in small_sizes:
            # M<4 is intentionally not in the native policy; the capture
            # list may still contain it for the original FlashInfer path.
            if not _use_native_shape(m, n, k):
                continue
            input_ = torch.zeros((m, k), device=device, dtype=torch.bfloat16)
            input_mxfp8, input_scale = mxfp8_e4m3_quantize(
                input_, is_sf_swizzled_layout=True
            )
            op = (
                torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm_pdl
                if use_pdl
                else torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm
            )
            op(input_mxfp8, weight, input_scale, weight_scale)
            launches += 1

    for device_index, _, _ in shapes:
        torch.accelerator.synchronize(device_index)
    return launches


def prepare_model(model: torch.nn.Module, config) -> int:
    """Install and warm the common native MXFP8 decode adapter if eligible."""
    if getattr(model, "_mxfp6_sm120_public_native_prepared", False):
        return int(getattr(model, "_mxfp6_sm120_public_native_count", 0))
    if not _eligible_config(model, config):
        return 0

    try:
        from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import (
            load_mxfp6_sm120_mxfp8_libraries,
            warmup_mxfp6_sm120_mxfp8_workspaces,
        )

        load_mxfp6_sm120_mxfp8_libraries()
    except (AttributeError, ImportError, OSError, RuntimeError) as exc:
        logger.warning_once(
            "Skipping common native MXFP8 decode adapter: %s", exc
        )
        return 0

    _register_op()
    _install_apply()
    count = _tag_model(model)
    if not count:
        model._mxfp6_sm120_public_native_prepared = True
        model._mxfp6_sm120_public_native_count = 0
        return 0

    sizes = _warmup_sizes(config)
    with torch.inference_mode():
        launches = warmup_mxfp6_sm120_mxfp8_workspaces(
            model, sizes, use_pdl=True
        )
        launches += _warmup_small_workspaces(model, sizes, use_pdl=True)

    model._mxfp6_sm120_public_native_prepared = True
    model._mxfp6_sm120_public_native_count = count
    logger.info(
        "Enabled common native MXFP8 decode for %d FlashInfer projections "
        "at M=%d..%d; warmed %d workspace launches for M=%s",
        count,
        _NATIVE_MIN_M,
        _NATIVE_MAX_M,
        launches,
        sizes,
    )
    return count


__all__ = ["prepare_model"]
