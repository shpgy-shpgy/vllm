# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare optional model adapters before profiling and graph capture."""

import torch

from vllm import envs
from vllm.config import CompilationMode, set_current_vllm_config
from vllm.logger import init_logger


def prepare_model(model, config) -> None:
    if envs.VLLM_MACH_MOE_PROJECTION_TUNING != "0" and not getattr(
        model, "_mach_projection_prepared", False
    ):
        from vllm.model_executor.layers.quantization.utils.mxfp6_sm120_utils import (
            _collect_w6a8_problems,
        )

        from .moe_projection import prepare

        count = prepare(model, _collect_w6a8_problems(model), config.model_config.dtype)
        if envs.VLLM_MACH_MOE_PROJECTION_TUNING == "1" and not count:
            raise ValueError("MoE projection tuning enabled but model is not eligible")
        model._mach_projection_prepared = bool(count)
    requested = (
        envs.VLLM_MACH_GDN_PERSISTENT
        or envs.VLLM_MACH_GDN_BA_OVERLAP
        or envs.VLLM_MACH_FUSED_SWIGLU_QUANT != "0"
        or envs.VLLM_MACH_FUSED_GDN_QUANT != "0"
        or envs.VLLM_MACH_FUSED_AR_QUANT != "0"
        or envs.VLLM_SM120_OWNER_PREFILL
        or envs.VLLM_SM120_LOSSLESS_PREFILL
    )
    if not requested:
        return
    if config.speculative_config is not None or config.lora_config is not None:
        raise ValueError(
            "Mach model adapters require non-speculative inference without LoRA"
        )
    if config.compilation_config.mode != CompilationMode.NONE:
        raise ValueError(
            "Mach Python GDN/prefill/producer adapters require compilation mode NONE; "
            "FULL_DECODE_ONLY CUDA graphs remain supported. MoE AR-Norm alone "
            "supports VLLM_COMPILE."
        )
    from . import ar_norm, fused_mlp, gdn_decode

    modules = list(model.modules())
    if envs.VLLM_SM120_OWNER_PREFILL or envs.VLLM_SM120_LOSSLESS_PREFILL:
        dense_models = [
            layer
            for layer in modules
            if type(layer).__name__ == "Qwen3_5Model"
            and layer.config.model_type == "qwen3_5_text"
            and layer.config.hidden_size == 5120
            and layer.use_fused_ar_gemma_norm
        ]
        if not dense_models:
            raise ValueError(
                "Owner/lossless prefill requires Dense hidden=5120 TP2 with "
                "VLLM_QWEN3_5_FUSED_AR_NORM=1."
            )
        if envs.VLLM_SM120_OWNER_PREFILL:
            for dense in dense_models:
                if len(dense.layers) != 64 or any(
                    layer.layer_type == "linear_attention"
                    and not layer.linear_attn.enable_fused_gdn_decode
                    for layer in dense.layers
                ):
                    raise ValueError(
                        "Owner prefill requires 64 layers and admitted packed GDN; "
                        "FP16 SSM needs VLLM_QWEN3_5_FP16_SSM=1."
                    )
    with torch.inference_mode(), set_current_vllm_config(config):
        gdn_decode.prepare(model)
        fused_mlp.prepare(model)
        ar_norm.prepare(model)
    required = {
        "GDN persistent/BA": (
            envs.VLLM_MACH_GDN_PERSISTENT or envs.VLLM_MACH_GDN_BA_OVERLAP,
            "_mach_gdn_prepared",
        ),
        "SwiGLU quant": (
            envs.VLLM_MACH_FUSED_SWIGLU_QUANT == "1",
            "_mach_swiglu_prepared",
        ),
        "GDN output quant": (
            envs.VLLM_MACH_FUSED_GDN_QUANT == "1",
            "_mach_gdn_output_fused",
        ),
        "AR quant": (envs.VLLM_MACH_FUSED_AR_QUANT == "1", "_mach_ar_quant_prepared"),
    }
    for name, (enabled, marker) in required.items():
        if enabled and not any(getattr(layer, marker, False) for layer in modules):
            raise ValueError(
                f"{name} explicitly enabled but no eligible layers were prepared. "
                "Check model geometry, native MXFP6 weights, AR-Norm and SSM dtype."
            )
    init_logger(__name__).info("Prepared opt-in Mach adapters before memory profiling")
