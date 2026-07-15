# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# MXFP8 constants
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32


def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Swizzle MXFP8 scales from row-major 2D to F8_128x4 layout."""
    scaling_vector_size = MXFP8_BLOCK_SIZE  # 32 for MXFP8
    factor = scaling_vector_size * 4  # 128

    num_m_tiles = (M + 127) // 128
    num_k_tiles = (K + factor - 1) // factor

    m_padded = num_m_tiles * 128
    k_scale_padded = num_k_tiles * 4

    scale_cols = K // scaling_vector_size
    sf_padded = torch.zeros(
        (m_padded, k_scale_padded), dtype=sf.dtype, device=sf.device
    )
    sf_padded[:M, :scale_cols] = sf

    sf_reshaped = sf_padded.view(num_m_tiles, 4, 32, num_k_tiles, 4)

    sf_swizzled = sf_reshaped.transpose(1, 3)

    return sf_swizzled.contiguous().view(-1)


def _mxfp8_e4m3_quantize_torch(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive MXFP8 quantization.
    For each block of 32 elements along the last dimension, compute a
    shared e8m0 scale that fits the block-wise amax into the finite
    float8_e4m3fn range, and quantize each element to float8_e4m3fn.

    Returns (quantized_values [same shape, fp8], scales uint8).
    Scale shape depends on is_sf_swizzled_layout:
      False -> [..., K//32]  (row-major 2D)
      True  -> [flat swizzled 1D]
    """
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    orig_shape = x.shape
    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE

    x_fp32 = x.to(torch.float32)
    x_blocked = x_fp32.view(*orig_shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    amax = x_blocked.abs().amax(dim=-1)
    amax = amax.clamp(min=torch.finfo(torch.float32).tiny)
    fp8_max = torch.finfo(MXFP8_VALUE_DTYPE).max
    scale_biased = torch.ceil(torch.log2(amax / fp8_max)) + 127.0
    scale_biased = scale_biased.clamp(0, 254)
    scales_uint8 = scale_biased.to(torch.uint8)

    descale = torch.exp2(scale_biased - 127.0)
    x_scaled = x_blocked / descale.unsqueeze(-1)

    x_fp8 = x_scaled.view(orig_shape).to(MXFP8_VALUE_DTYPE)

    if x.ndim == 2:
        M, K = x.shape
        scales_uint8 = scales_uint8.view(M, -1)
        if is_sf_swizzled_layout:
            scales_uint8 = swizzle_mxfp8_scale(scales_uint8, M=M, K=K)
    elif x.ndim == 3:
        B, M, K = x.shape
        scales_uint8 = scales_uint8.view(B, M, -1)
        if is_sf_swizzled_layout:
            swizzled = []
            for i in range(B):
                swizzled.append(swizzle_mxfp8_scale(scales_uint8[i], M=M, K=K))
            scales_uint8 = torch.cat(swizzled)

    return x_fp8, scales_uint8


def _mxfp8_quant_triton_kernel():
    """Lazily-built Triton kernel: per-32-block E8M0 scale + FP8-E4M3 quant.

    Fuses what ``_mxfp8_e4m3_quantize_torch`` does in several elementwise passes
    into one launch. Each program handles ``[BLOCK_M, 32]`` (one MX block).
    """
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _kernel(
        x_ptr,
        xq_ptr,
        s_ptr,
        M,
        K,
        sxm,
        sxk,
        sqm,
        sqk,
        ssm,
        ssk,
        BLOCK_M: tl.constexpr,
        FP8_MAX: tl.constexpr,
        TINY: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)  # which 32-element block along K
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_b * 32 + tl.arange(0, 32)
        m_mask = offs_m < M
        x = tl.load(
            x_ptr + offs_m[:, None] * sxm + offs_k[None, :] * sxk,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        # Mirror _mxfp8_e4m3_quantize_torch: the scale has to put the block amax
        # at the top of the e4m3 range rather than at 1.0, or small elements of
        # the block end up in the subnormals.
        amax = tl.maximum(tl.max(tl.abs(x), axis=1), TINY)  # [BLOCK_M]
        sb = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
        sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
        descale = tl.exp2(sb - 127.0)
        xq = (x / descale[:, None]).to(xq_ptr.dtype.element_ty)
        tl.store(
            xq_ptr + offs_m[:, None] * sqm + offs_k[None, :] * sqk,
            xq,
            mask=m_mask[:, None],
        )
        tl.store(s_ptr + offs_m * ssm + pid_b * ssk, sb.to(tl.uint8), mask=m_mask)

    return _kernel


_MXFP8_QUANT_KERNEL = None


def _mxfp8_e4m3_quantize_triton(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused 2D MXFP8 quant (non-swizzled, row-major [M, K//32] scales)."""
    from vllm.triton_utils import triton

    global _MXFP8_QUANT_KERNEL
    if _MXFP8_QUANT_KERNEL is None:
        _MXFP8_QUANT_KERNEL = _mxfp8_quant_triton_kernel()

    M, K = x.shape
    x = x.contiguous()
    xq = torch.empty((M, K), dtype=MXFP8_VALUE_DTYPE, device=x.device)
    scales = torch.empty(
        (M, K // MXFP8_BLOCK_SIZE), dtype=MXFP8_SCALE_DTYPE, device=x.device
    )
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M), K // MXFP8_BLOCK_SIZE)
    _MXFP8_QUANT_KERNEL[grid](
        x,
        xq,
        scales,
        M,
        K,
        x.stride(0),
        x.stride(1),
        xq.stride(0),
        xq.stride(1),
        scales.stride(0),
        scales.stride(1),
        BLOCK_M=BLOCK_M,
        FP8_MAX=float(torch.finfo(MXFP8_VALUE_DTYPE).max),
        TINY=float(torch.finfo(torch.float32).tiny),
    )
    return xq, scales


def _mxfp8_e4m3_quantize_impl(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.platforms import current_platform
    from vllm.utils.flashinfer import has_flashinfer

    if current_platform.has_device_capability(100) and has_flashinfer():
        from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize

        x_q, x_scales = flashinfer_mxfp8_quantize(
            x,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            alignment=alignment if alignment > 0 else 32,
            backend="cute-dsl",
        )
        if x_scales.ndim == 1 and x.ndim == 2 and not is_sf_swizzled_layout:
            x_scales = x_scales.view(x.size(0), -1)
        return x_q, x_scales

    # ROCm: a single fused Triton kernel beats the multi-pass torch path for the
    # common 2D, non-swizzled activation-quant case (used by the native MX
    # linear/MoE). Falls back to torch otherwise (3D weights, swizzled layout).
    if (
        current_platform.is_rocm()
        and not is_sf_swizzled_layout
        and x.ndim == 2
        and x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    ):
        return _mxfp8_e4m3_quantize_triton(x)

    return _mxfp8_e4m3_quantize_torch(x, is_sf_swizzled_layout)


def mxfp8_e4m3_quantize(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.mxfp8_quantize(x, is_sf_swizzled_layout, alignment)


def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Dequantize MXFP8 tensor to BF16."""
    x_float = x.to(torch.float32)

    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_float.view(*x.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    descale = torch.exp2(scales.to(torch.float32) - 127.0)

    dequantized = x_blocked * descale.unsqueeze(-1)

    dequantized = dequantized.view(*x.shape)

    return dequantized.to(torch.bfloat16)


def mxfp8_e4m3_quantize_fake(
    x: torch.Tensor,
    is_sf_swizzled_layout: bool = False,
    alignment: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for torch.compile tracing."""
    fp_data = torch.empty_like(x, dtype=MXFP8_VALUE_DTYPE)

    block_size = MXFP8_BLOCK_SIZE

    if x.ndim == 2:
        M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    elif x.ndim == 3:
        B, M, N = x.shape
        K = (N + block_size - 1) // block_size
        if is_sf_swizzled_layout:
            M_padded = ((M + 127) // 128) * 128
            K_padded = ((K + 3) // 4) * 4
            scales = torch.empty(
                B * M_padded * K_padded, dtype=MXFP8_SCALE_DTYPE, device=x.device
            )
        else:
            scales = torch.empty((B, M, K), dtype=MXFP8_SCALE_DTYPE, device=x.device)
    else:
        scale_shape = list(x.shape)
        scale_shape[-1] = (x.shape[-1] + block_size - 1) // block_size
        scales = torch.empty(scale_shape, dtype=MXFP8_SCALE_DTYPE, device=x.device)

    return fp_data, scales


direct_register_custom_op(
    op_name="mxfp8_quantize",
    op_func=_mxfp8_e4m3_quantize_impl,
    fake_impl=mxfp8_e4m3_quantize_fake,
)


def _empty_swizzled_mxfp8_scales(x: torch.Tensor, output_features: int) -> torch.Tensor:
    rows = x.numel() // x.shape[-1]
    scale_columns = output_features // MXFP8_BLOCK_SIZE
    padded_rows = ((rows + 127) // 128) * 128
    padded_scale_columns = ((scale_columns + 3) // 4) * 4
    return torch.empty(
        padded_rows * padded_scale_columns,
        dtype=MXFP8_SCALE_DTYPE,
        device=x.device,
    )


def _build_fused_mxfp8_quant_kernels():
    from vllm.triton_utils import tl, triton

    @triton.jit
    def _silu_and_mul_quant_kernel(
        input_ptr,
        output_ptr,
        scale_ptr,
        rows,
        input_stride,
        output_stride,
        K: tl.constexpr,
        GROUPS_PER_ROW: tl.constexpr,
        GROUPS_PER_CTA: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        group_start = tl.program_id(0) * GROUPS_PER_CTA
        row_start = tl.program_id(1).to(tl.int64) * BLOCK_M
        row_step = tl.num_programs(1).to(tl.int64) * BLOCK_M

        elements_per_cta: tl.constexpr = GROUPS_PER_CTA * 32
        columns = group_start * 32 + tl.arange(0, elements_per_cta)
        row_offsets = tl.arange(0, BLOCK_M)
        group_offsets = tl.arange(0, GROUPS_PER_CTA)
        column_mask = columns < K
        num_k_tiles: tl.constexpr = (GROUPS_PER_ROW + 3) // 4

        while row_start < rows:
            row_ids = row_start + row_offsets
            row_mask = row_ids < rows
            input_offsets = row_ids[:, None] * input_stride

            gate = tl.load(
                input_ptr + input_offsets + columns[None, :],
                mask=row_mask[:, None] & column_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up = tl.load(
                input_ptr + input_offsets + K + columns[None, :],
                mask=row_mask[:, None] & column_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gate = (
                (gate / (1.0 + tl.exp(-gate)))
                .to(input_ptr.dtype.element_ty)
                .to(tl.float32)
            )
            output = gate * up
            output = output.to(input_ptr.dtype.element_ty).to(tl.float32)
            output_blocked = tl.reshape(output, (BLOCK_M, GROUPS_PER_CTA, 32))

            amax = tl.max(tl.abs(output_blocked), axis=2)
            raw_scale = amax * (1.0 / 448.0)
            scale = tl.ceil(tl.log2(raw_scale)) + 127.0
            scale = tl.minimum(tl.maximum(scale, 0.0), 254.0)
            descale = tl.exp2(scale - 127.0)
            descale = tl.where(amax == 0.0, 1.0, descale)
            output_quant = output_blocked / descale[:, :, None]
            output_quant = tl.reshape(output_quant, (BLOCK_M, elements_per_cta))

            tl.store(
                output_ptr + row_ids[:, None] * output_stride + columns[None, :],
                output_quant.to(output_ptr.dtype.element_ty),
                mask=row_mask[:, None] & column_mask[None, :],
            )

            group_ids = group_start + group_offsets
            scale_offsets = (
                (
                    (row_ids[:, None] // 128 * num_k_tiles + group_ids // 4) * 32
                    + row_ids[:, None] % 32
                )
                * 4
                + (row_ids[:, None] % 128) // 32
            ) * 4 + group_ids % 4
            tl.store(
                scale_ptr + scale_offsets,
                scale.to(tl.uint8),
                mask=row_mask[:, None] & (group_ids[None, :] < GROUPS_PER_ROW),
            )
            row_start += row_step

    @triton.jit
    def _fused_add_rms_norm_quant_kernel(
        input_ptr,
        residual_ptr,
        weight_ptr,
        output_ptr,
        scale_ptr,
        residual_output_ptr,
        K,
        epsilon,
        GROUPS_PER_ROW: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_id = tl.program_id(0).to(tl.int64)
        columns = tl.arange(0, BLOCK_K)
        column_mask = columns < K
        offsets = row_id * K + columns

        values = tl.load(input_ptr + offsets, mask=column_mask, other=0.0).to(
            tl.float32
        )
        residual = tl.load(residual_ptr + offsets, mask=column_mask, other=0.0).to(
            tl.float32
        )
        values += residual
        values = values.to(input_ptr.dtype.element_ty).to(tl.float32)
        tl.store(
            residual_output_ptr + offsets,
            values.to(residual_output_ptr.dtype.element_ty),
            mask=column_mask,
        )

        variance = tl.sum(values * values, axis=0) / K
        inverse_rms = tl.rsqrt(variance + epsilon)
        weight = tl.load(weight_ptr + columns, mask=column_mask, other=0.0).to(
            tl.float32
        )
        output = values * inverse_rms * weight
        output = output.to(input_ptr.dtype.element_ty).to(tl.float32)

        block_groups: tl.constexpr = BLOCK_K // 32
        output_blocked = tl.reshape(output, (block_groups, 32))
        amax = tl.max(tl.abs(output_blocked), axis=1)
        raw_scale = amax * (1.0 / 448.0)
        scale = tl.ceil(tl.log2(raw_scale)) + 127.0
        scale = tl.minimum(tl.maximum(scale, 0.0), 254.0)
        descale = tl.exp2(scale - 127.0)
        descale = tl.where(amax == 0.0, 1.0, descale)
        output_quant = output_blocked / descale[:, None]
        output_quant = tl.reshape(output_quant, (BLOCK_K,))
        tl.store(
            output_ptr + offsets,
            output_quant.to(output_ptr.dtype.element_ty),
            mask=column_mask,
        )

        group_ids = tl.arange(0, block_groups)
        num_k_tiles: tl.constexpr = (GROUPS_PER_ROW + 3) // 4
        scale_offsets = (
            ((row_id // 128 * num_k_tiles + group_ids // 4) * 32 + row_id % 32) * 4
            + (row_id % 128) // 32
        ) * 4 + group_ids % 4
        tl.store(
            scale_ptr + scale_offsets,
            scale.to(tl.uint8),
            mask=group_ids < GROUPS_PER_ROW,
        )

    return _silu_and_mul_quant_kernel, _fused_add_rms_norm_quant_kernel


_SILU_AND_MUL_MXFP8_QUANT_KERNEL = None
_FUSED_ADD_RMS_NORM_MXFP8_QUANT_KERNEL = None


def _get_fused_mxfp8_quant_kernels():
    global _SILU_AND_MUL_MXFP8_QUANT_KERNEL
    global _FUSED_ADD_RMS_NORM_MXFP8_QUANT_KERNEL
    if _SILU_AND_MUL_MXFP8_QUANT_KERNEL is None:
        (
            _SILU_AND_MUL_MXFP8_QUANT_KERNEL,
            _FUSED_ADD_RMS_NORM_MXFP8_QUANT_KERNEL,
        ) = _build_fused_mxfp8_quant_kernels()
    return (
        _SILU_AND_MUL_MXFP8_QUANT_KERNEL,
        _FUSED_ADD_RMS_NORM_MXFP8_QUANT_KERNEL,
    )


def _silu_and_mul_mxfp8_quant_impl(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.triton_utils import triton

    assert input.shape[-1] % (2 * MXFP8_BLOCK_SIZE) == 0
    input = input.contiguous()
    hidden_size = input.shape[-1] // 2
    rows = input.numel() // input.shape[-1]
    output_shape = (*input.shape[:-1], hidden_size)
    output = torch.empty(output_shape, dtype=MXFP8_VALUE_DTYPE, device=input.device)
    scales = _empty_swizzled_mxfp8_scales(input, hidden_size)

    silu_kernel, _ = _get_fused_mxfp8_quant_kernels()
    groups_per_row = hidden_size // MXFP8_BLOCK_SIZE
    groups_per_cta = 32
    block_m = 1 if rows < 512 else 4
    grid = (
        triton.cdiv(groups_per_row, groups_per_cta),
        min(triton.cdiv(rows, block_m), 4096),
    )
    silu_kernel[grid](
        input,
        output,
        scales,
        rows,
        input.shape[-1],
        hidden_size,
        K=hidden_size,
        GROUPS_PER_ROW=groups_per_row,
        GROUPS_PER_CTA=groups_per_cta,
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=2,
    )
    return output, scales


def _silu_and_mul_mxfp8_quant_fake(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_size = input.shape[-1] // 2
    output = torch.empty(
        (*input.shape[:-1], hidden_size),
        dtype=MXFP8_VALUE_DTYPE,
        device=input.device,
    )
    return output, _empty_swizzled_mxfp8_scales(input, hidden_size)


def silu_and_mul_mxfp8_quant(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse SiLU-and-mul with MXFP8 quantization and scale swizzling."""
    return torch.ops.vllm.silu_and_mul_mxfp8_quant(input)


direct_register_custom_op(
    op_name="silu_and_mul_mxfp8_quant",
    op_func=_silu_and_mul_mxfp8_quant_impl,
    fake_impl=_silu_and_mul_mxfp8_quant_fake,
)


def _fused_add_rms_norm_mxfp8_quant_impl(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input = input.contiguous()
    residual = residual.contiguous()
    weight = weight.contiguous()
    hidden_size = input.shape[-1]
    assert hidden_size % MXFP8_BLOCK_SIZE == 0
    rows = input.numel() // hidden_size
    output = torch.empty_like(input, dtype=MXFP8_VALUE_DTYPE)
    scales = _empty_swizzled_mxfp8_scales(input, hidden_size)
    residual_output = torch.empty_like(input)
    _, rms_kernel = _get_fused_mxfp8_quant_kernels()

    from vllm.triton_utils import triton

    block_k = triton.next_power_of_2(hidden_size)
    num_warps = 8 if block_k >= 8192 else 4
    rms_kernel[(rows,)](
        input,
        residual,
        weight,
        output,
        scales,
        residual_output,
        hidden_size,
        epsilon,
        GROUPS_PER_ROW=hidden_size // MXFP8_BLOCK_SIZE,
        BLOCK_K=block_k,
        num_warps=num_warps,
    )
    return output, scales, residual_output


def _fused_add_rms_norm_mxfp8_quant_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del residual, weight, epsilon
    return (
        torch.empty_like(input, dtype=MXFP8_VALUE_DTYPE),
        _empty_swizzled_mxfp8_scales(input, input.shape[-1]),
        torch.empty_like(input),
    )


def fused_add_rms_norm_mxfp8_quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse residual-add RMSNorm with MXFP8 quantization and scale swizzling."""
    return torch.ops.vllm.fused_add_rms_norm_mxfp8_quant(
        input, residual, weight, epsilon
    )


direct_register_custom_op(
    op_name="fused_add_rms_norm_mxfp8_quant",
    op_func=_fused_add_rms_norm_mxfp8_quant_impl,
    fake_impl=_fused_add_rms_norm_mxfp8_quant_fake,
)


def xpu_mxfp8_quantize(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.vllm.xpu_mxfp8_quantize(x, dtype)
