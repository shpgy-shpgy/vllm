# SPDX-License-Identifier: Apache-2.0
"""Opt-in two-limb MXFP8 dispatch for the Qwen3.5-4B SM120 path.

The upstream ``mxfp6.mxfp8_dual`` API improves the activation approximation
by computing a high FP8 limb and a BF16-rounded residual limb.  This adapter
keeps the existing FlashInfer kernel as the fallback and only redirects the
shapes that were validated for the first migration:

* MoE gate/up projections: small captured M values through 64,
  N=18432, K=2560
* GDN QKVZ projections: the same small captured M values through 64,
  N=12288, K=2560

The policy is intentionally shape-gated: only the validated Qwen3.5-4B
projection geometry uses the dual kernel, while every other MXFP8 shape keeps
the existing FlashInfer/native path.
"""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_HIDDEN_SIZE = 2560
_GATEUP_N = 18432
_QKVZ_N = 12288
_SMALL_CAPTURE_M = frozenset(
    (1, 2, 4, 8, 10, 12, 14, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64)
)
_GATEUP_M = _SMALL_CAPTURE_M
_QKVZ_M = _SMALL_CAPTURE_M

_OP_REGISTERED = False
_INSTALLED = False
_DUAL_API: Any = None
_ORIGINAL_APPLIES: dict[type, Any] = {}


def _load_dual_api():
    """Load the source-added dual API and validate the native dispatcher."""
    global _DUAL_API
    if _DUAL_API is not None:
        return _DUAL_API

    try:
        dual = import_module("mxfp6.mxfp8_dual")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "MXFP8 dual dispatch requires a dual-enabled mxfp6-sm120 build. "
            "The original 0.2.1 wheel does not provide mxfp6.mxfp8_dual; "
            "rebuild/install the latest mxfp6-sm120 wheel first."
        ) from exc

    try:
        dual_path = Path(dual.load_library()).resolve()
        schema = str(torch.ops.mxfp8_sm120.dual_gemm_out.default._schema)
    except (AttributeError, ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "The installed MXFP8 library does not expose "
            "mxfp8_sm120::dual_gemm_out. Install a dual-enabled mxfp6-sm120 "
            "binary built from the current upstream/main."
        ) from exc

    required = (
        "Tensor hi",
        "Tensor weight",
        "Tensor s_hi",
        "Tensor s_weight",
        "Tensor res",
        "Tensor s_res",
        "Tensor(a!) out",
        "bool use_pdl=False",
    )
    if not all(fragment in schema for fragment in required):
        raise RuntimeError(
            "Unexpected mxfp8_sm120::dual_gemm_out schema: " f"{schema}"
        )

    # The ordinary and dual Python APIs must resolve to one dispatcher binary.
    # Loading two copies can register duplicate operators and is especially
    # difficult to diagnose once the model enters CUDA graph capture.
    try:
        ordinary = import_module("mxfp6.mxfp8")
        ordinary_path = Path(ordinary.load_library()).resolve()
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        raise RuntimeError(
            "Could not load the ordinary mxfp6.mxfp8 dispatcher while enabling "
            "dual MXFP8"
        ) from exc
    if dual_path != ordinary_path:
        raise RuntimeError(
            "Ordinary and dual MXFP8 APIs loaded different native libraries: "
            f"ordinary={ordinary_path}, dual={dual_path}. Set MXFP8_LIBRARY_PATH "
            "to the same dual-enabled mxfp8_torch.so for both loaders."
        )

    _DUAL_API = dual
    logger.info("Loaded dual MXFP8 dispatcher from %s", dual_path)
    return dual


def _dual_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """Opaque vLLM op: select dual or original FlashInfer by runtime M."""
    m = int(x.shape[0])
    n = int(weight.shape[0])
    use_dual = (n == _GATEUP_N and m in _GATEUP_M) or (
        n == _QKVZ_N and m in _QKVZ_M
    )

    if not use_dual:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )
        from vllm.utils import flashinfer as vllm_flashinfer

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            x, is_sf_swizzled_layout=True
        )
        return vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=torch.bfloat16,
            backend=backend,
        )

    dual = _load_dual_api()
    high, high_scale, residual, residual_scale = dual.quantize(x)
    output = torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16
    )
    dual.gemm_out(
        high,
        weight,
        high_scale,
        weight_scale.view(-1),
        residual,
        residual_scale,
        output,
        use_pdl=False,
    )
    return output


def _dual_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    del weight_scale, backend
    return torch.empty(
        (x.shape[0], weight.shape[0]), device=x.device, dtype=torch.bfloat16
    )


def _register_op() -> None:
    global _OP_REGISTERED
    if _OP_REGISTERED:
        return
    direct_register_custom_op(
        op_name="mxfp6_sm120_dual_mxfp8_gemm",
        op_func=_dual_gemm_impl,
        mutates_args=[],
        fake_impl=_dual_gemm_fake,
    )
    _OP_REGISTERED = True


def _eligible(layer: torch.nn.Module, x: torch.Tensor, n: int) -> bool:
    if x.ndim < 2:
        return False
    weight = getattr(layer, "weight", None)
    weight_scale = getattr(layer, "weight_scale", None)
    if weight is None or weight_scale is None:
        return False
    return bool(
        x.dtype == torch.bfloat16
        and x.device.type == "cuda"
        and x.shape[-1] == _HIDDEN_SIZE
        and tuple(weight.shape) == (n, _HIDDEN_SIZE)
        and weight.dtype == MXFP8_VALUE_DTYPE
        and weight_scale.dtype == MXFP8_SCALE_DTYPE
        and weight_scale.ndim == 1
        and weight_scale.numel() == n * (_HIDDEN_SIZE // 32)
        and x.is_contiguous()
        and weight.is_contiguous()
        and weight_scale.is_contiguous()
        and weight.device == x.device
        and weight_scale.device == x.device
    )


def _install_dispatch() -> None:
    """Patch both possible MXFP8 linear classes with a guarded dispatcher."""
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )
    for kernel_cls in (FlashInferCutlassMxfp8LinearKernel,):
        if kernel_cls in _ORIGINAL_APPLIES:
            continue
        original = kernel_cls.apply_weights
        _ORIGINAL_APPLIES[kernel_cls] = original

        @wraps(original)
        def apply_weights(self, layer, x, bias=None, _original=original):
            kind = getattr(layer, "_mxfp6_sm120_dual_kind", None)
            if kind == "gateup":
                eligible = _eligible(layer, x, _GATEUP_N)
            elif kind == "qkvz":
                eligible = _eligible(layer, x, _QKVZ_N)
            else:
                eligible = False

            if not eligible:
                # Preserve the exact pre-migration FlashInfer/native path for
                # all layers and all M values outside the targeted dual policy.
                return _original(self, layer, x, bias)

            input_shape = x.shape
            input_2d = x.reshape(-1, x.shape[-1]).contiguous()
            output = torch.ops.vllm.mxfp6_sm120_dual_mxfp8_gemm(
                input_2d,
                layer.weight,
                layer.weight_scale,
                layer.mxfp8_backend,
            )
            output = output.view(*input_shape[:-1], layer.weight.shape[0])
            if bias is not None:
                output = output + bias
            return output

        kernel_cls.apply_weights = apply_weights

    _INSTALLED = True


def _kernel(layer: torch.nn.Module):
    method = getattr(layer, "quant_method", None)
    scheme = getattr(layer, "scheme", None) or getattr(method, "scheme", None)
    return getattr(scheme, "kernel", None) or getattr(method, "kernel", None)


def _validate_projection(layer: torch.nn.Module, n: int, name: str) -> None:
    weight = getattr(layer, "weight", None)
    scale = getattr(layer, "weight_scale", None)
    if (
        weight is None
        or scale is None
        or tuple(weight.shape) != (n, _HIDDEN_SIZE)
        or weight.dtype != MXFP8_VALUE_DTYPE
        or scale.dtype != MXFP8_SCALE_DTYPE
        or scale.ndim != 1
        or scale.numel() != n * (_HIDDEN_SIZE // 32)
    ):
        weight_info = (
            None if weight is None else (tuple(weight.shape), weight.dtype)
        )
        scale_info = None if scale is None else (tuple(scale.shape), scale.dtype)
        raise RuntimeError(
            "Dual MXFP8 projection/layout guard failed for "
            f"{name}: weight={weight_info}, scale={scale_info}"
        )


def _tag_model(model: torch.nn.Module) -> tuple[int, int]:
    """Tag exactly the validated 4B Qwen3.5 projection set."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    gateup_count = 0
    qkvz_count = 0
    for name, module in model.named_modules():
        if isinstance(module, Qwen2MoeMLP) and module.expert_gate is None:
            layer = module.gate_up_proj
            _validate_projection(layer, _GATEUP_N, f"{name}.gate_up_proj")
            layer._mxfp6_sm120_dual_kind = "gateup"
            layer._mxfp6_sm120_dual_id = gateup_count
            gateup_count += 1
        if isinstance(module, QwenGatedDeltaNetAttention):
            layer = module.in_proj_qkvz
            _validate_projection(layer, _QKVZ_N, f"{name}.in_proj_qkvz")
            layer._mxfp6_sm120_dual_kind = "qkvz"
            layer._mxfp6_sm120_dual_id = qkvz_count
            qkvz_count += 1

    if (gateup_count, qkvz_count) != (32, 24):
        raise ValueError(
            "VLLM_MXFP6_SM120_MXFP8_DUAL is currently limited to the "
            "Qwen3.5-4B TP1 geometry (32 gate/up and 24 QKVZ projections); "
            f"found {gateup_count} gate/up and {qkvz_count} QKVZ projections"
        )
    return gateup_count, qkvz_count


def prepare_model(model: torch.nn.Module, config) -> tuple[int, int]:
    """Load, install, tag, and eagerly warm the validated small-M shapes."""
    if getattr(model, "_mxfp6_sm120_dual_prepared", False):
        return (32, 24)
    if not config.use_v2_model_runner:
        raise ValueError("Dual MXFP8 requires the V2 model runner")
    if config.speculative_config is not None or config.lora_config is not None:
        raise ValueError("Dual MXFP8 requires non-speculative inference without LoRA")
    if config.parallel_config.tensor_parallel_size != 1:
        raise ValueError(
            "Dual MXFP8 migration currently requires tensor parallel size 1"
        )
    hf_config = config.model_config.hf_text_config
    if (
        getattr(hf_config, "model_type", None) != "qwen3_5_text"
        or getattr(hf_config, "hidden_size", None) != _HIDDEN_SIZE
    ):
        raise ValueError(
            "Dual MXFP8 migration currently targets Qwen3.5 hidden_size=2560"
        )

    _load_dual_api()
    _register_op()
    _install_dispatch()
    counts = _tag_model(model)

    device = next(
        (parameter.device for parameter in model.parameters() if parameter.is_cuda),
        None,
    )
    if device is None or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError("Dual MXFP8 requires a CUDA SM120 model device")

    # Eagerly compile the quantizer and both native tile families before
    # memory profiling.  Later CUDA-graph capture sees the same opaque op.
    layers = dict(model.named_modules())
    first_gateup = next(
        layer
        for layer in layers.values()
        if getattr(layer, "_mxfp6_sm120_dual_kind", None) == "gateup"
    )
    first_qkvz = next(
        layer
        for layer in layers.values()
        if getattr(layer, "_mxfp6_sm120_dual_kind", None) == "qkvz"
    )
    with torch.inference_mode():
        for layer, m_values in (
            (first_gateup, _GATEUP_M),
            (first_qkvz, _QKVZ_M),
        ):
            for m in sorted(m_values):
                x = torch.zeros(
                    (m, _HIDDEN_SIZE),
                    device=layer.weight.device,
                    dtype=torch.bfloat16,
                )
                torch.ops.vllm.mxfp6_sm120_dual_mxfp8_gemm(
                    x,
                    layer.weight,
                    layer.weight_scale,
                    getattr(layer, "mxfp8_backend", "b12x-decode"),
                )
        torch.cuda.synchronize(first_qkvz.weight.device)

    model._mxfp6_sm120_dual_prepared = True
    logger.info(
        "Enabled dual MXFP8 dispatch for %d gate/up and %d QKVZ projections "
        "(MLP M=%s, QKVZ M=%s); all other MXFP8 shapes keep their original "
        "FlashInfer/native path",
        counts[0],
        counts[1],
        sorted(_GATEUP_M),
        sorted(_QKVZ_M),
    )
    return counts


__all__ = ["prepare_model"]
