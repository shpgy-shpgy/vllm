# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MxFp4LinearKernel,
    MxFp6LinearKernel,
    init_mxfp4_linear_kernel,
    init_mxfp6_linear_kernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_sm120_utils import (
    is_mxfp6_sm120_available,
    mxfp6_sm120_gemm,
    mxfp6_sm120_quantize_mxfp8_packed,
    pack_mxfp6_sm120_scales,
)
from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import (
    OCP_MX_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp6E2M3Dynamic,
    kMxfp6E2M3Static,
    kMxfp6E3M2Dynamic,
    kMxfp6E3M2Static,
    kMxfp8Dynamic,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)

_WEIGHT_QUANT_KEY_MAP: dict[str, QuantKey] = {
    "mxfp4": kMxfp4Static,
    "mxfp6_e3m2": kMxfp6E3M2Static,
    "mxfp6_e2m3": kMxfp6E2M3Static,
}

_ACTIVATION_QUANT_KEY_MAP: dict[str, QuantKey] = {
    "mxfp4": kMxfp4Dynamic,
    "mxfp6_e3m2": kMxfp6E3M2Dynamic,
    "mxfp6_e2m3": kMxfp6E2M3Dynamic,
    "fp8": kMxfp8Dynamic,
}


class QuarkOCP_MX(QuarkScheme):
    ocp_mx_linear: MxFp6LinearKernel | MxFp4LinearKernel

    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any] | None,
        dynamic_mxfp4_quant: bool = False,
    ):
        self.dynamic_mxfp4_quant = dynamic_mxfp4_quant
        # Keep the original Quark specs around. The SM120 capability check
        # below needs the serialized dtype names (for example fp8_e4m3),
        # while the linear-kernel selector uses normalized names.
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.weight_dtype = weight_quant_spec["dtype"].replace("fp", "mxfp")
        self.input_dtype: str | None = None
        if input_quant_spec is not None:
            input_dtype = input_quant_spec["dtype"]
            self.input_dtype = (
                "fp8"
                if input_dtype in {"fp8", "fp8_e4m3", "mxfp8_e4m3"}
                else input_dtype.replace("fp", "mxfp")
            )

        if self.input_dtype not in [None, *_ACTIVATION_QUANT_KEY_MAP]:
            raise ValueError(
                f"Unsupported input_dtype={self.input_dtype} for QuarkOCP_MX. "
                f"Supported activation dtypes are {_ACTIVATION_QUANT_KEY_MAP.keys()}, "
                "or None for weight-only quantization."
            )

        self.weight_quant_key = _WEIGHT_QUANT_KEY_MAP[self.weight_dtype]
        self.activation_quant_key = (
            _ACTIVATION_QUANT_KEY_MAP[self.input_dtype]
            if self.input_dtype is not None
            else None
        )

        if self.weight_dtype == "mxfp4":
            self.packed_factor: int | Fraction = 2
        else:
            self.packed_factor = Fraction(numerator=8, denominator=6)

        if input_quant_spec is None:
            self.static_input_scales = False
        else:
            self.static_input_scales = not input_quant_spec.get("is_dynamic")

        if self.static_input_scales:
            raise NotImplementedError(
                "QuarkOCP_MX with static input scales is currently not "
                "implemented. Please open an issue."
            )

        # This exact Quark configuration is the W6A8 format implemented by
        # the optional SM120 backend.  It must be selected before creating
        # the generic MXFP6 emulation kernel; otherwise every dense layer is
        # dequantized to a high-precision temporary during profiling.
        is_mxfp6_sm120_config = self._is_mxfp6_sm120_config()
        self.use_mxfp6_sm120 = (
            is_mxfp6_sm120_config and is_mxfp6_sm120_available()
        )
        self.output_size_per_partition: int | None = None
        self.input_size_per_partition: int | None = None
        self.padded_input_size_per_partition: int | None = None
        self.emulate = not self.use_mxfp6_sm120 and (
            not current_platform.supports_mx()
            or self.input_dtype != "mxfp4"
            or self.weight_dtype != "mxfp4"
        )

        if self.use_mxfp6_sm120:
            logger.info_once(
                "Using the mxfp6-sm120 native W6A8 kernel for Quark "
                "E3M2 weights and dynamic E4M3 activations."
            )
        elif (
            is_mxfp6_sm120_config
            and current_platform.is_cuda()
            and current_platform.is_device_capability(120)
        ):
            logger.warning_once(
                "The Quark W6A8 checkpoint is compatible with the native "
                "SM120 kernel, but mxfp6-sm120 could not be loaded. "
                "Falling back to MXFP6 emulation."
            )
        elif not current_platform.supports_mx():
            logger.warning_once(
                "The current platform does not support native MXFP4/MXFP6 "
                "computation. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

        if (
            not self.use_mxfp6_sm120
            and current_platform.supports_mx()
            and (self.input_dtype != "mxfp4" or self.weight_dtype != "mxfp4")
        ):
            logger.warning_once(
                "The current platform supports native MXFP4/MXFP6 "
                f"computation, but kernels for input_dtype={self.input_dtype} "
                f"and weight_dtype={self.weight_dtype} are not yet integrated "
                "in vLLM. Simulated weight dequantization and activation "
                "QDQ (quantize and dequantize) will be used, with the linear "
                "layers computed in high precision."
            )

    def _is_mxfp6_sm120_config(self) -> bool:
        input_spec = self.input_quant_spec
        return (
            self.weight_quant_spec.get("dtype") == "fp6_e3m2"
            and self.weight_quant_spec.get("qscheme") == "per_group"
            and self.weight_quant_spec.get("group_size") == OCP_MX_BLOCK_SIZE
            and self.weight_quant_spec.get("scale_format") == "e8m0"
            and self.weight_quant_spec.get("symmetric") is True
            and not self.weight_quant_spec.get("is_dynamic")
            and input_spec is not None
            and input_spec.get("dtype") in {"fp8_e4m3", "mxfp8_e4m3"}
            and input_spec.get("qscheme") == "per_group"
            and input_spec.get("group_size") == OCP_MX_BLOCK_SIZE
            and input_spec.get("scale_format") == "e8m0"
            and input_spec.get("symmetric") is True
            and input_spec.get("is_dynamic") is True
        )

    @staticmethod
    def _is_mxfp6_sm120_problem_supported(
        output_features: int, input_features: int
    ) -> bool:
        return output_features % 8 == 0 and input_features % 128 == 0

    def get_packed_dim(self, dim: int, quant_dtype: str):
        if quant_dtype == "mxfp4":
            assert dim % 2 == 0
            return dim // 2
        elif quant_dtype in {"mxfp6_e3m2", "mxfp6_e2m3"}:
            # FP6 packs 4 * 6 = 24 bits on 3 bytes.
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4
        else:
            raise NotImplementedError(
                "Unsupported quant_dtype in QuarkOCP_MX.get_packed_dim, "
                f"got quant_dtype={quant_dtype}. Something is wrong, please "
                "open an issue."
            )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_dynamic_mxfp4_weights_after_loading(
        self, layer: torch.nn.Module
    ) -> None:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        w_q, w_s = dynamic_mxfp4_quant(layer.weight)
        layer.weight_scale = torch.nn.Parameter(w_s, requires_grad=False)
        layer.weight = torch.nn.Parameter(w_q, requires_grad=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)

        if self.use_mxfp6_sm120:
            assert self.input_size_per_partition is not None
            assert self.padded_input_size_per_partition is not None
            if self.padded_input_size_per_partition != self.input_size_per_partition:
                packed_features = self.padded_input_size_per_partition * 3 // 4
                padded_weight = torch.zeros(
                    (layer.weight.shape[0], packed_features),
                    dtype=layer.weight.dtype,
                    device=layer.weight.device,
                )
                padded_weight[:, : layer.weight.shape[1]] = layer.weight.data
                layer.weight = torch.nn.Parameter(padded_weight, requires_grad=False)

                padded_scale = torch.full(
                    (
                        layer.weight_scale.shape[0],
                        self.padded_input_size_per_partition // OCP_MX_BLOCK_SIZE,
                    ),
                    127,
                    dtype=layer.weight_scale.dtype,
                    device=layer.weight_scale.device,
                )
                padded_scale[:, : layer.weight_scale.shape[1]] = (
                    layer.weight_scale.data
                )
                layer.weight_scale = torch.nn.Parameter(
                    padded_scale, requires_grad=False
                )

            layer.weight_scale = torch.nn.Parameter(
                pack_mxfp6_sm120_scales(layer.weight_scale.data),
                requires_grad=False,
            )
            return

        if self.dynamic_mxfp4_quant:
            self.process_dynamic_mxfp4_weights_after_loading(layer)

        self.ocp_mx_linear.process_weights_after_loading(layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        if input_size_per_partition % OCP_MX_BLOCK_SIZE != 0:
            layer_name = getattr(layer, "prefix", "") or type(layer).__name__
            raise ValueError(
                f"OCP MX linear layer {layer_name!r} has an input size per "
                f"partition of {input_size_per_partition}, which must be "
                f"divisible by the OCP MX group size {OCP_MX_BLOCK_SIZE}. "
                "Choose a compatible tensor-parallel size or avoid "
                "tensor-parallel sharding for this layer."
            )

        if self.dynamic_mxfp4_quant:
            weight = ModelWeightParameter(
                data=torch.empty(
                    sum(output_partition_sizes),
                    input_size_per_partition,
                    dtype=params_dtype,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )

            layer.register_parameter("weight", weight)
            set_weight_attrs(weight, kwargs)
        else:
            output_size_per_partition = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            self.output_size_per_partition = output_size_per_partition
            self.input_size_per_partition = input_size_per_partition
            self.padded_input_size_per_partition = input_size_per_partition

            if self.use_mxfp6_sm120 and not self._is_mxfp6_sm120_problem_supported(
                output_size_per_partition, input_size_per_partition
            ):
                if (
                    output_size_per_partition % 8 == 0
                    and input_size_per_partition % OCP_MX_BLOCK_SIZE == 0
                ):
                    self.padded_input_size_per_partition = (
                        (input_size_per_partition + 127) // 128 * 128
                    )
                    logger.info_once(
                        "Padding an mxfp6-sm120 dense partition from K=%d "
                        "to K=%d for native W6A8 execution.",
                        input_size_per_partition,
                        self.padded_input_size_per_partition,
                    )
                else:
                    self.use_mxfp6_sm120 = False
                    self.emulate = True
                    logger.warning_once(
                        "mxfp6-sm120 requires N to be divisible by 8 and K "
                        "to be divisible by 32, but a partition has N=%d and "
                        "K=%d. Falling back to MXFP6 emulation for that layer.",
                        output_size_per_partition,
                        input_size_per_partition,
                    )

            # WEIGHT
            weight = PackedvLLMParameter(
                data=torch.empty(
                    output_size_per_partition,
                    self.get_packed_dim(input_size_per_partition, self.weight_dtype),
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=self.packed_factor,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight", weight)

            # WEIGHT SCALE
            weight_scale = GroupQuantScaleParameter(
                data=torch.empty(
                    output_size_per_partition,
                    input_size_per_partition // OCP_MX_BLOCK_SIZE,
                    dtype=torch.uint8,
                ),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_scale", weight_scale)

        # The SM120 W6A8 path is implemented directly in apply_weights.  Do
        # not construct the generic MXFP6 kernel here: it is an emulation
        # fallback and its selection log would incorrectly suggest that the
        # native path is being used through that kernel.
        if self.use_mxfp6_sm120:
            return

        if self.weight_quant_key == kMxfp4Static:
            self.ocp_mx_linear = init_mxfp4_linear_kernel(
                activation_quant_key=self.activation_quant_key,
            )
        elif self.weight_quant_key in [kMxfp6E2M3Static, kMxfp6E3M2Static]:
            self.ocp_mx_linear = init_mxfp6_linear_kernel(
                weight_quant_key=self.weight_quant_key,
                activation_quant_key=self.activation_quant_key,
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_mxfp6_sm120:
            assert self.output_size_per_partition is not None
            assert self.input_size_per_partition is not None
            assert self.padded_input_size_per_partition is not None
            output_shape = (*x.shape[:-1], self.output_size_per_partition)
            x_2d = x.reshape(-1, x.shape[-1]).contiguous()
            if self.padded_input_size_per_partition != self.input_size_per_partition:
                x_2d = F.pad(
                    x_2d,
                    (
                        0,
                        self.padded_input_size_per_partition
                        - self.input_size_per_partition,
                    ),
                )
            quantized_x, input_scale = mxfp6_sm120_quantize_mxfp8_packed(x_2d)
            y = mxfp6_sm120_gemm(
                quantized_x,
                input_scale,
                layer.weight,
                layer.weight_scale,
                self.output_size_per_partition,
                x.dtype,
            ).reshape(output_shape)
            if bias is not None:
                y = y + bias
            return y

        return self.ocp_mx_linear.apply_weights(layer, x, bias)
