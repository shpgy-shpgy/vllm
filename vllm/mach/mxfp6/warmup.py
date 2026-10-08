# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare optional model adapters before profiling and graph capture."""

import os
from importlib.util import find_spec

import torch

from vllm import envs
from vllm.config import CompilationMode, CUDAGraphMode, set_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

DENSE_FLAGS = (
    "VLLM_QWEN3_5_FUSED_AR_NORM",
    "VLLM_MACH_GDN_PERSISTENT",
    "VLLM_MACH_GDN_BA_OVERLAP",
    "VLLM_MACH_FUSED_SWIGLU_QUANT",
    "VLLM_MACH_FUSED_GDN_QUANT",
    "VLLM_MACH_FUSED_AR_QUANT",
    "VLLM_SM120_LOSSLESS_PREFILL",
    "VLLM_SM120_OWNER_PREFILL",
)


def _mxfp6_sm120_workspace_warmup_sizes() -> tuple[int, ...]:
    raw = os.getenv(
        "VLLM_MXFP6_SM120_WORKSPACE_WARMUP_SIZES",
        "17,32,64,128,256,512,1024,2048",
    )
    try:
        values = tuple(
            sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
        )
    except ValueError as exc:
        raise ValueError(
            "VLLM_MXFP6_SM120_WORKSPACE_WARMUP_SIZES must be a comma-separated "
            "list of positive integers"
        ) from exc
    if any(value <= 0 for value in values):
        raise ValueError(
            "VLLM_MXFP6_SM120_WORKSPACE_WARMUP_SIZES must contain positive integers"
        )
    return values


def _prepare_mxfp8_sm120_adapters(model, config) -> None:
    """Prepare the opt-in 4B adapters before profiling and graph capture."""
    linear_backend = getattr(config.kernel_config, "linear_backend", None)
    if linear_backend in ("mxfp6_sm120", "mxfp6-sm120"):
        from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import (
            warmup_mxfp6_sm120_mxfp8_workspaces,
        )

        sizes = _mxfp6_sm120_workspace_warmup_sizes()
        launches = warmup_mxfp6_sm120_mxfp8_workspaces(model, sizes)
        if launches:
            init_logger(__name__).info(
                "Warmed %d mxfp6-sm120 native MXFP8 workspace launches for sizes=%s",
                launches,
                sizes,
            )

    if envs.VLLM_MXFP6_SM120_BA_GEMV:
        from .ba import prepare_model as prepare_ba_model

        prepare_ba_model(model)


def activate_dense_defaults(config) -> None:
    """Restore the serialized decision in spawned workers before model loading."""
    extra = config.additional_config
    enabled = isinstance(extra, dict) and extra.get("mach_dense_auto") is True
    envs.set_mach_dense_defaults(
        enabled,
        fp16=enabled and config.cache_config.mamba_ssm_cache_dtype == "float16",
    )


def configure_dense_defaults(config) -> None:
    """Select the supported Dense27B bundle without changing the SSM dtype."""
    envs.set_mach_dense_defaults(False)
    extra = config.additional_config
    if not isinstance(extra, dict):
        return
    extra.pop("mach_dense_auto", None)
    if not envs.VLLM_MACH_DENSE_AUTO or any(
        os.getenv(flag) == "0" for flag in DENSE_FLAGS
    ):
        return
    model, parallel = config.model_config, config.parallel_config
    if model is None or model.enforce_eager:
        return
    state_dtype = config.cache_config.mamba_ssm_cache_dtype
    if state_dtype == "float16" and os.getenv("VLLM_QWEN3_5_FP16_SSM") == "0":
        return
    hf = model.hf_text_config
    geometry = {
        "model_type": "qwen3_5_text",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
    }
    if any(getattr(hf, key, None) != value for key, value in geometry.items()):
        return
    quant = config.quant_config
    scheme = getattr(quant, "quant_config", {}).get("global_quant_config", {})
    if not (
        model.dtype == torch.bfloat16
        and not getattr(hf, "layer_scale", False)
        and parallel.tensor_parallel_size == 2
        and parallel.pipeline_parallel_size == parallel.data_parallel_size == 1
        and parallel.prefill_context_parallel_size == 1
        and parallel.decode_context_parallel_size == 1
        and not parallel.enable_expert_parallel
        and not parallel.enable_eplb
        and not parallel.use_sequence_parallel_moe
        and not parallel.use_ubatching
        and config.speculative_config is None
        and config.lora_config is None
        and config.kv_transfer_config is None
        and config.use_v2_model_runner
        and state_dtype in ("float32", "float16")
        and config.cache_config.mamba_cache_dtype in ("auto", "bfloat16")
        and config.cache_config.mamba_cache_mode
        == ("align" if config.cache_config.enable_prefix_caching else "none")
        and not config.cache_config.use_replayssm
        and config.compilation_config.backend == "inductor"
        and config.compilation_config.mode
        in (None, CompilationMode.NONE, CompilationMode.VLLM_COMPILE)
        and config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        and config.profiler_config.profiler != "proton"
        and not envs.VLLM_USE_BREAKABLE_CUDAGRAPH
        and quant is not None
        and quant.get_name() == "quark"
        and scheme.get("weight", {}).get("dtype") == "fp6_e3m2"
        and scheme.get("input_tensors", {}).get("dtype") == "fp8_e4m3"
        and current_platform.is_cuda()
        and current_platform.is_device_capability(120)
    ):
        return
    modules = (
        "vllm_mach_gdn",
        "mach_ar_norm_ext",
        "mach_owner_prefill_ext",
        "mach_lossless_prefill_direct_ext",
        "mach_lossless_prefill_long_ext",
    )
    if any(find_spec(name) is None for name in modules):
        init_logger(__name__).warning_once(
            "Dense auto optimizations skipped: install the GDN, AR norm, owner "
            "and lossless prefill wheels. Keeping the original compilation config."
        )
        return
    extra["mach_dense_auto"] = True
    config.compilation_config.mode = CompilationMode.NONE
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
    activate_dense_defaults(config)
    init_logger(__name__).info(
        "Auto-enabled SM120 TP2 Dense27B %s SSM optimization bundle: "
        "%s; prefix_caching=%s, mamba_cache_mode=%s; "
        "compilation=NONE, CUDA graphs=FULL_DECODE_ONLY. "
        "Set VLLM_MACH_DENSE_AUTO=0 to keep the original configuration.",
        state_dtype,
        ", ".join(DENSE_FLAGS),
        config.cache_config.enable_prefix_caching,
        config.cache_config.mamba_cache_mode,
    )


def prepare_model(model, config) -> None:
    _prepare_mxfp8_sm120_adapters(model, config)

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
        envs.VLLM_MACH_GDN_ORDERED
        or envs.VLLM_MACH_GDN_PERSISTENT
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
    requires_python_model_adapter = (
        envs.VLLM_MACH_GDN_PERSISTENT
        or envs.VLLM_MACH_GDN_BA_OVERLAP
        or envs.VLLM_MACH_FUSED_SWIGLU_QUANT != "0"
        or envs.VLLM_MACH_FUSED_GDN_QUANT != "0"
        or envs.VLLM_MACH_FUSED_AR_QUANT != "0"
        or envs.VLLM_SM120_OWNER_PREFILL
        or envs.VLLM_SM120_LOSSLESS_PREFILL
    )
    if (
        requires_python_model_adapter
        and config.compilation_config.mode != CompilationMode.NONE
    ):
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
        if envs.VLLM_MACH_GDN_ORDERED:
            from .ordered_gdn import prepare as prepare_ordered_gdn

            prepare_ordered_gdn(model)
        gdn_decode.prepare(model)
        fused_mlp.prepare(model)
        ar_norm.prepare(model)
    required = {
        "Ordered GDN": (
            envs.VLLM_MACH_GDN_ORDERED,
            "_mach_ordered_gdn_prepared",
        ),
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
