# SPDX-License-Identifier: Apache-2.0
"""Mach all/wide-only GEMM policy; preserve stock FlashInfer quantization."""

from functools import wraps

import torch

from vllm import envs
from vllm.platforms import current_platform


def selected(policy, n):
    if policy not in ("all", "wide-only"):
        raise ValueError("4B native policy must be all or wide-only")
    return policy == "all" or n != 2560


def apply_weights(layer, x, bias=None):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    n, k = layer.weight.shape
    values, scales = mxfp8_e4m3_quantize(x.reshape(-1, k), is_sf_swizzled_layout=True)
    out = torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm(
        values, layer.weight, scales, layer.weight_scale
    )
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], n)


def prepare_model(model, cfg):
    policy = envs.VLLM_MACH_MXFP8_4B_NATIVE_POLICY
    selected(policy, 2560)
    if getattr(model, "_mach_4b_native_policy", None) == policy:
        return
    hf = cfg.model_config.hf_text_config
    if not (
        current_platform.is_cuda()
        and current_platform.is_device_capability(120)
        and cfg.use_v2_model_runner
        and cfg.parallel_config.tensor_parallel_size == 1
        and cfg.parallel_config.pipeline_parallel_size == 1
        and cfg.lora_config is None
        and cfg.speculative_config is None
        and cfg.kernel_config.linear_backend in ("auto", "flashinfer_cutlass")
        and cfg.model_config.dtype == torch.bfloat16
        and cfg.model_config.quantization == "compressed-tensors"
        and hf.model_type == "qwen3_5_text"
        and hf.hidden_size == 2560
        and hf.intermediate_size == 9216
        and hf.num_hidden_layers == 32
    ):
        raise ValueError("4B native policy requires SM120 Qwen3.5-4B MXFP8 V2 TP1/PP1")
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel as Kernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import (
        load_mxfp6_sm120_mxfp8_libraries,
    )

    from .native import _eligible_layer, _kernel

    layers = [
        m
        for name, m in model.named_modules()
        if not name.endswith("lm_head")
        and type(_kernel(m)) is Kernel
        and _eligible_layer(m)
        and selected(policy, m.weight.shape[0])
    ]
    if not layers:
        raise ValueError("4B native policy found no eligible MXFP8 layers")
    load_mxfp6_sm120_mxfp8_libraries()
    original = Kernel.apply_weights

    @wraps(original)
    def apply(self, layer, x, bias=None):
        if (
            getattr(layer, "_mach_4b_native", False)
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.ndim >= 1
            and x.device == layer.weight.device
            and x.shape[-1] == layer.weight.shape[1]
        ):
            return apply_weights(layer, x, bias)
        return original(self, layer, x, bias)

    shapes = {}
    for layer in layers:
        shapes.setdefault((layer.weight.device, *layer.weight.shape), layer)
    sizes = {1, 4, 16, 32, 64, cfg.scheduler_config.max_num_batched_tokens}
    sizes.update(cfg.compilation_config.cudagraph_capture_sizes or ())
    with torch.inference_mode():
        for (device, _, k), layer in shapes.items():
            for rows in sorted(sizes):
                apply_weights(
                    layer, torch.zeros((rows, k), device=device, dtype=torch.bfloat16)
                )
    for layer in layers:
        layer._mach_4b_native = True
    Kernel.apply_weights = apply
    model._mach_4b_native_policy = policy
    model._mach_4b_native_layers = len(layers)
