# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_BLOCK_SIZE
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _unpack_mxfp6(x: torch.Tensor) -> torch.Tensor:
    """Unpack four contiguous 6-bit values from every three bytes."""
    if x.ndim == 0 or x.shape[-1] == 0 or (x.shape[-1] * 4) % 3 != 0:
        raise ValueError(
            "Packed MXFP6 weights must have a non-empty last dimension whose "
            "size is divisible by 3."
        )

    packed = x.contiguous().reshape(-1, x.shape[-1]).to(torch.int32)
    byte0 = packed[:, 0::3]
    byte1 = packed[:, 1::3]
    byte2 = packed[:, 2::3]
    codes = torch.stack(
        (
            byte0 & 0x3F,
            (byte0 >> 6) | ((byte1 & 0x0F) << 2),
            (byte1 >> 4) | ((byte2 & 0x03) << 4),
            byte2 >> 2,
        ),
        dim=-1,
    )
    return codes.reshape(*x.shape[:-1], -1).to(torch.uint8)


def _decode_mxfp6(codes: torch.Tensor, quant_dtype: str) -> torch.Tensor:
    """Decode OCP MXFP6 E3M2/E2M3 codes to float32."""
    if quant_dtype == "fp6_e3m2":
        exponent_bits, mantissa_bits, exponent_bias = 3, 2, 3
    elif quant_dtype == "fp6_e2m3":
        exponent_bits, mantissa_bits, exponent_bias = 2, 3, 1
    else:
        raise ValueError(f"Unsupported MXFP6 dtype: {quant_dtype}")

    code = codes.to(torch.int32)
    sign = torch.where((code & 0x20) != 0, -1.0, 1.0)
    exponent = (code >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = code & ((1 << mantissa_bits) - 1)

    normal = (1.0 + mantissa.to(torch.float32) / (1 << mantissa_bits)) * (
        2.0 ** (exponent.to(torch.float32) - exponent_bias)
    )
    subnormal = mantissa.to(torch.float32) * (
        2.0 ** (1 - exponent_bias - mantissa_bits)
    )
    return sign * torch.where(exponent == 0, subnormal, normal)


def _dequant_mxfp6_torch(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype, quant_dtype: str
) -> torch.Tensor:
    """Quark-compatible MXFP6 dequantization without an external package."""
    unpacked_x = _unpack_mxfp6(x)
    if unpacked_x.shape[-1] % OCP_MX_BLOCK_SIZE != 0:
        raise ValueError(
            "Unpacked MXFP6 weights must have a last dimension divisible by "
            f"{OCP_MX_BLOCK_SIZE}."
        )

    values = _decode_mxfp6(unpacked_x, quant_dtype)
    num_groups = values.shape[-1] // OCP_MX_BLOCK_SIZE
    values = values.reshape(*values.shape[:-1], num_groups, OCP_MX_BLOCK_SIZE)

    # UE8M0 stores the power-of-two exponent with a bias of 127.  Scales are
    # normally [rows, K/32], but keeping the leading dimensions broadcastable
    # also covers expert weight tensors.
    scales = torch.exp2(scale.to(torch.int16).to(torch.float32) - 127.0)
    values = values * scales.unsqueeze(-1)
    return values.reshape(*unpacked_x.shape[:-1], -1).to(float_dtype)


def _quant_dequant_mxfp6(
    x: torch.Tensor,
    quant_dtype: str,
    scale_calculation_mode: str = "even",
) -> torch.Tensor:
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            fake_quantize_fp4_fp6_per_group_with_scale,
        )
        from quark.torch.quantization.utils import even_round, reshape_to_blocks
    except ImportError as err:
        raise ImportError(
            "The package `amd-quark` is required to use "
            "MX-FP6 models. Please install it with `pip install "
            "amd-quark`."
        ) from err

    axis = -1
    block_x = reshape_to_blocks(x, OCP_MX_BLOCK_SIZE, axis)
    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    amax = amax.squeeze(-1)

    # TODO: there are other rounding strategies supported in quark and in the
    # config.json that we do not check for here!
    if scale_calculation_mode != "even":
        raise NotImplementedError(
            f"Scale calculation mode {scale_calculation_mode} is not yet "
            "supported in MX-FP6 quantization"
        )
    scale = even_round(amax, quant_dtype)

    # Apply dequantize(quantize(x)).
    x = fake_quantize_fp4_fp6_per_group_with_scale(
        x,
        scale.to(x.device),
        axis=axis,
        group_size=OCP_MX_BLOCK_SIZE,
        quant_dtype=quant_dtype,
    )

    return x


def _quant_dequant_mxfp6_fake(
    x: torch.Tensor,
    quant_dtype: str,
    scale_calculation_mode: str = "even",
) -> torch.Tensor:
    return torch.empty_like(x)


def _dequant_mxfp6(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype, quant_dtype: str
) -> torch.Tensor:
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            dequantize_fp4_fp6_per_group,
        )
        from quark.torch.utils.pack import create_pack_method
    except ImportError as e:
        logger.warning_once(
            "The package `amd-quark` is unavailable; using the slower pure "
            "PyTorch MXFP6 dequantization fallback."
        )
        return _dequant_mxfp6_torch(x, scale, float_dtype, quant_dtype)

    pack_method = create_pack_method(None, dtype=quant_dtype)
    unpacked_x = pack_method.unpack(x, reorder=False)

    scale = 2 ** (scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)

    # TODO: `dequantize_fp4_fp6_per_group` and `prepare_inputs_per_group`
    # always return fp32.
    return dequantize_fp4_fp6_per_group(
        unpacked_x,
        scale,
        axis=-1,
        group_size=OCP_MX_BLOCK_SIZE,
        quant_dtype=quant_dtype,
    ).to(float_dtype)


def _dequant_mxfp6_fake(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype, quant_dtype: str
) -> torch.Tensor:
    assert (x.shape[-1] * 4) % 3 == 0
    return torch.empty(
        (*x.shape[:-1], (x.shape[-1] * 4) // 3), dtype=float_dtype, device=x.device
    )


# Protect these operations into a torch custom op to avoid errors as
# torch._dynamo.exc.Unsupported: Attempted to call function marked as skipped
# Explanation: Dynamo does not know how to trace the builtin
# `kernel_ext.PyCapsule.dq_uint8_mxfp4_to_half.` This function is either a
# Python builtin (e.g. _warnings.warn) or a third-party C/C++ Python
# extension (perhaps created with pybind).
# TODO: Make sure there is no way to avoid having these functions
# marked as skipped by dynamo.
try:
    direct_register_custom_op(
        op_name="quant_dequant_mxfp6",
        op_func=_quant_dequant_mxfp6,
        mutates_args=[],
        fake_impl=_quant_dequant_mxfp6_fake,
    )
except AttributeError as error:
    raise error


# Expose keyword arguments.
def quant_dequant_mxfp6(
    x: torch.Tensor,
    quant_dtype: str,
    scale_calculation_mode: str = "even",
) -> torch.Tensor:
    return torch.ops.vllm.quant_dequant_mxfp6(x, quant_dtype, scale_calculation_mode)


try:
    direct_register_custom_op(
        op_name="dequant_mxfp6",
        op_func=_dequant_mxfp6,
        mutates_args=[],
        fake_impl=_dequant_mxfp6_fake,
    )
except AttributeError as error:
    raise error


def dequant_mxfp6(
    x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype, quant_dtype: str
) -> torch.Tensor:
    return torch.ops.vllm.dequant_mxfp6(x, scale, float_dtype, quant_dtype)
