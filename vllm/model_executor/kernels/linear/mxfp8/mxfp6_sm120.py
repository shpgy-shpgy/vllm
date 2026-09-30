# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
    swizzle_mxfp8_scale,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

# The native MXFP8 path has a fixed activation-quantization and launch cost.
# Keep the measured low-M decode range on the existing FlashInfer dispatcher;
# it selects the tuned B12x kernel for the Qwen3.5 shapes below M=32.
_MXFP6_NATIVE_MIN_M = 17


def load_mxfp6_sm120_mxfp8_libraries() -> None:
    """Load both native libraries before entering a compiled model graph."""
    import mxfp6.mxfp8 as mxfp8

    mxfp8.load_library()
    from mxfp6 import load_library as load_mxfp6_library

    load_mxfp6_library()


def _mxfp6_sm120_mxfp8_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if input.shape[0] < _MXFP6_NATIVE_MIN_M:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )
        from vllm.utils import flashinfer as vllm_flashinfer

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input, is_sf_swizzled_layout=True
        )
        return vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=torch.bfloat16,
            backend="b12x-decode",
        )

    return torch.ops.mxfp8_sm120.gemm_from_float(input, weight, weight_scale)


def _mxfp6_sm120_mxfp8_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    del weight_scale
    return torch.empty(
        (input.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=input.device,
    )


direct_register_custom_op(
    op_name="mxfp6_sm120_mxfp8_gemm",
    op_func=_mxfp6_sm120_mxfp8_gemm_impl,
    mutates_args=[],
    fake_impl=_mxfp6_sm120_mxfp8_gemm_fake,
)


class Mxfp6Sm120Mxfp8LinearKernel(Mxfp8LinearKernel):
    """SM120 MXFP8 W8A8 GEMM with a low-M FlashInfer fallback."""

    @staticmethod
    def _load_native():
        load_mxfp6_sm120_mxfp8_libraries()
        import mxfp6.mxfp8 as mxfp8

        return mxfp8

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "mxfp6-sm120 MXFP8 requires CUDA"
        if not current_platform.is_device_capability(120):
            return False, "mxfp6-sm120 MXFP8 requires an SM120 device"
        try:
            cls._load_native()
        except (ImportError, OSError, RuntimeError, AttributeError) as e:
            return False, f"mxfp6-sm120 MXFP8 library is unavailable: {e}"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        del c
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._load_native()

        weight = layer.weight.data
        if weight.dtype != MXFP8_VALUE_DTYPE:
            raise ValueError(
                "mxfp6-sm120 MXFP8 requires "
                f"{MXFP8_VALUE_DTYPE} weights, got {weight.dtype}"
            )
        if weight.ndim != 2:
            raise ValueError(
                f"mxfp6-sm120 MXFP8 requires 2D weights, got {weight.ndim}D"
            )

        out_features, in_features = map(int, weight.shape)
        if out_features % 128 or in_features % 128:
            raise ValueError(
                "mxfp6-sm120 MXFP8 requires N/K to be multiples of 128, "
                f"got N={out_features}, K={in_features}"
            )
        if in_features % MXFP8_BLOCK_SIZE:
            raise ValueError(
                "mxfp6-sm120 MXFP8 requires K to be divisible by "
                f"{MXFP8_BLOCK_SIZE}, got K={in_features}"
            )

        weight_scale = layer.weight_scale.data
        if weight_scale.dtype != MXFP8_SCALE_DTYPE or weight_scale.ndim != 2:
            raise ValueError(
                "mxfp6-sm120 MXFP8 requires a 2D uint8 weight_scale, "
                f"got {weight_scale.dtype} {tuple(weight_scale.shape)}"
            )

        scale_k = in_features // MXFP8_BLOCK_SIZE
        weight_scale = weight_scale[:out_features, :scale_k].contiguous()
        weight_scale = swizzle_mxfp8_scale(weight_scale, M=out_features, K=in_features)

        weight = weight.contiguous()
        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        # Keep the native operands around for the medium/large-M branch. The
        # low-M branch reuses the same weight and scale tensors with FlashInfer.
        layer.mxfp6_sm120_mxfp8_weight_values = layer.weight.view(torch.float8_e4m3fn)
        layer.mxfp6_sm120_mxfp8_weight_scales = layer.weight_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"mxfp6-sm120 MXFP8 requires FP16/BF16 activations, got {x.dtype}"
            )

        _, in_features = layer.weight.shape
        input_shape = x.shape
        input_2d = x.reshape(-1, in_features).contiguous()
        if input_2d.shape[0] == 0:
            output = input_2d.new_empty((0, layer.weight.shape[0]))
            output = output.view(*input_shape[:-1], layer.weight.shape[0])
        else:
            # Both native libraries were loaded during weight processing. Use a
            # vLLM custom op with a fake implementation so Dynamo never traces
            # the optional package's lazy loader or its threading lock. The op
            # also dispatches low-M decode shapes to FlashInfer B12x based on
            # the runtime shape, which avoids freezing the choice in Dynamo.
            output = torch.ops.vllm.mxfp6_sm120_mxfp8_gemm(
                input_2d,
                layer.mxfp6_sm120_mxfp8_weight_values,
                layer.mxfp6_sm120_mxfp8_weight_scales,
            )
            if output.dtype != x.dtype:
                output = output.to(x.dtype)
            output = output.view(*input_shape[:-1], layer.weight.shape[0])

        if bias is not None:
            output = output + bias
        return output
