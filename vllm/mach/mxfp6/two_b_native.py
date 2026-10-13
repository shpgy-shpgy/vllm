# SPDX-License-Identifier: Apache-2.0
"""GEMM-only 2B adapter from vllm-mach's MXFP8 native_backend (05aeea3).

Preserve FlashInfer activation quantization and swizzled weights. Upstream
2B uses native GEMM at all eligible row counts, including prefill. Reuse
shpgy's stream-owned workspace implementation, without the 4B row policy.
"""

from functools import wraps

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)
_INSTALLED = False
_PDL_REGISTERED = False


def _scheduled_gemm(
    values: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import (
        _native_gemm_impl_with_options,
    )

    from . import two_b_pdl

    rows = int(values.shape[0])
    if two_b_pdl.warming_full(rows):
        # Workspace keys include PDL and stream. Prime the capture stream too.
        _native_gemm_impl_with_options(
            values, weight, scales, weight_scales, use_pdl=True
        )
    return _native_gemm_impl_with_options(
        values, weight, scales, weight_scales, use_pdl=two_b_pdl.native_selected(rows)
    )


def install_pdl_dispatch() -> None:
    global _PDL_REGISTERED
    if _PDL_REGISTERED:
        return
    from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import _native_gemm_fake
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="mach_2b_scheduled_gemm",
        op_func=_scheduled_gemm,
        mutates_args=[],
        fake_impl=_native_gemm_fake,
    )
    _PDL_REGISTERED = True


def eligible_config(config) -> bool:
    mc = config.model_config
    if mc is None:
        return False
    hf = mc.hf_text_config
    return bool(
        envs.VLLM_MACH_MXFP8_2B_NATIVE
        and current_platform.is_cuda()
        and current_platform.is_device_capability(120)
        and config.use_v2_model_runner
        and config.parallel_config.tensor_parallel_size == 1
        and config.lora_config is None
        and config.speculative_config is None
        and config.kernel_config.linear_backend in ("auto", "flashinfer_cutlass")
        and mc.dtype == torch.bfloat16
        and mc.quantization == "compressed-tensors"
        and getattr(hf, "model_type", None) == "qwen3_5_text"
        and getattr(hf, "hidden_size", None) == 2048
        and getattr(hf, "intermediate_size", None) == 6144
        and getattr(hf, "num_hidden_layers", None) == 24
    )


def gemm(
    values: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    if envs.VLLM_MACH_MXFP8_2B_PDL:
        return torch.ops.vllm.mach_2b_scheduled_gemm(
            values, weight, scales, weight_scales
        )
    return torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm(
        values,
        weight,
        scales,
        weight_scales,
    )


def apply_weights(layer, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    n, k = layer.weight.shape
    values, scales = mxfp8_e4m3_quantize(
        x.reshape(-1, k),
        is_sf_swizzled_layout=True,
    )
    output = gemm(values, layer.weight, scales, layer.weight_scale)
    if bias is not None:
        output = output + bias
    return output.view(*x.shape[:-1], n)


def _install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )

    original = FlashInferCutlassMxfp8LinearKernel.apply_weights

    @wraps(original)
    def apply(self, layer, x, bias=None):
        if (
            not getattr(layer, "_mach_2b_native", False)
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or x.ndim < 1
            or x.shape[-1] != layer.weight.shape[1]
            or x.device != layer.weight.device
        ):
            return original(self, layer, x, bias)
        return apply_weights(layer, x, bias)

    FlashInferCutlassMxfp8LinearKernel.apply_weights = apply
    _INSTALLED = True


def prepare_model(model, config) -> int:
    if not eligible_config(config):
        return 0
    if getattr(model, "_mach_2b_native_prepared", False):
        return model._mach_2b_native_count
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import (
        load_mxfp6_sm120_mxfp8_libraries,
    )

    from .native import _eligible_layer, _kernel

    layers = [
        layer
        for name, layer in model.named_modules()
        if not name.endswith("lm_head")
        and type(_kernel(layer)) is FlashInferCutlassMxfp8LinearKernel
        and _eligible_layer(layer)
    ]
    if not layers:
        return 0
    load_mxfp6_sm120_mxfp8_libraries()
    _install()
    shapes = {}
    for layer in layers:
        layer._mach_2b_native = True
        shapes.setdefault((layer.weight.device, *layer.weight.shape), layer)
    sizes = {1, 4, 16, 24, 32, 64, config.scheduler_config.max_num_batched_tokens}
    sizes.update(config.compilation_config.cudagraph_capture_sizes or ())
    with torch.inference_mode():
        for (device, _, k), layer in shapes.items():
            for m in sorted(sizes):
                x = torch.zeros((m, k), dtype=torch.bfloat16, device=device)
                apply_weights(layer, x, None)
        for device, _, _ in shapes:
            torch.cuda.synchronize(device)
    model._mach_2b_native_prepared = True
    model._mach_2b_native_count = len(layers)
    logger.info(
        "Prepared upstream 2B native MXFP8 GEMM: %d layers, "
        "%d shapes, sizes=%s; FlashInfer quantization unchanged, PDL=%s",
        len(layers),
        len(shapes),
        sorted(sizes),
        "FULL16/24" if envs.VLLM_MACH_MXFP8_2B_PDL else "off",
    )
    return len(layers)
