# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered full-store GDN decode for the first SM120 TP1 migration stage.

The kernel is intentionally limited to the Qwen3.5-4B geometry and physical
decode rows 1/2/4/8.  It keeps the recurrent update in a fixed token/head
order and stores the complete FP32 state after every token.  Deferred M16
terms are drained by the scheduler transition hook before this kernel runs.
"""

from __future__ import annotations

import math

import torch

from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton

ROWS = (1, 2, 4, 8)
H = 16
HV = 32
V = 128
K = 128
BV = 32
NV = V // BV
QKV = (2 * H * K) + HV * V


@triton.jit
def _ordered_decode(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    out,
    base,
    indices,
    stride_mixed,
    stride_a,
    stride_b,
    stride_base_page,
    stride_index,
    pages,
    scale,
    H: tl.constexpr,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
):
    """One program per value tile and logical request/head pair."""

    value_tile = tl.program_id(0)
    request_head = tl.program_id(1)
    request = request_head // HV
    value_head = request_head % HV
    key_head = value_head // (HV // H)

    key_offsets = tl.arange(0, K)
    value_offsets = tl.arange(0, BV)
    value_offset = value_tile * BV + value_offsets

    state_idx = tl.load(indices + request * stride_index).to(tl.int32)
    output_ptr = out + (request * HV + value_head) * V + value_offset
    if (state_idx <= 0) | (state_idx >= pages):
        tl.store(
            output_ptr,
            tl.full((BV,), 0, tl.float32).to(out.dtype.element_ty),
        )
        return

    state_ptr = (
        base
        + state_idx * stride_base_page
        + value_head * V * K
        + value_offset[:, None] * K
        + key_offsets[None, :]
    )
    hidden = tl.load(state_ptr).to(tl.float32)

    mixed_ptr = mixed_qkv + request * stride_mixed
    query = tl.load(mixed_ptr + key_head * K + key_offsets).to(tl.float32)
    key = tl.load(mixed_ptr + H * K + key_head * K + key_offsets).to(tl.float32)
    value = tl.load(mixed_ptr + 2 * H * K + value_head * V + value_offset).to(
        tl.float32
    )

    query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
    key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
    query = query * scale

    a_value = tl.load(a + request * stride_a + value_head).to(tl.float32)
    b_value = tl.load(b + request * stride_b + value_head).to(tl.float32)
    a_log = tl.load(A_log + value_head).to(tl.float32)
    dt = tl.load(dt_bias + value_head).to(tl.float32)

    x = a_value + dt
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    g = -tl.exp(a_log) * softplus_x
    beta = tl.sigmoid(b_value)
    alpha = exp(g)

    hidden *= alpha
    value -= tl.sum(hidden * key[None, :], 1)
    value *= beta
    hidden += value[:, None] * key[None, :]
    output = tl.sum(hidden * query[None, :], 1)

    tl.store(output_ptr, output.to(out.dtype.element_ty))
    tl.store(state_ptr, hidden)


def _check_vector(name: str, tensor: torch.Tensor, shape: tuple[int, ...], dtype):
    if (
        tuple(tensor.shape) != shape
        or tensor.dtype != dtype
        or not tensor.is_cuda
        or tensor.stride(-1) != 1
    ):
        raise ValueError(f"{name} must be CUDA {dtype} with shape {shape}")


def decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    base: torch.Tensor,
    out: torch.Tensor,
    indices: torch.Tensor,
    scale: float = K**-0.5,
) -> torch.Tensor:
    """Run the ordered full-store kernel for one physical decode batch."""

    rows = int(mixed_qkv.shape[0])
    if rows not in ROWS or tuple(mixed_qkv.shape) != (rows, QKV):
        raise ValueError(f"ordered low-M GDN supports only rows {ROWS}")
    _check_vector("mixed_qkv", mixed_qkv, (rows, QKV), torch.bfloat16)
    _check_vector("a", a, (rows, HV), torch.bfloat16)
    _check_vector("b", b, (rows, HV), torch.bfloat16)
    if (
        base.ndim != 4
        or tuple(base.shape[1:]) != (HV, V, K)
        or base.dtype != torch.float32
        or not base.is_cuda
        or base.stride(-1) != 1
        or base.stride(-2) != K
        or base.stride(-3) != V * K
        or base.stride(0) < HV * V * K
    ):
        raise ValueError("base must be a contiguous-inner CUDA FP32 GDN state pool")
    if (
        tuple(A_log.shape) != (HV,)
        or A_log.dtype != torch.float32
        or A_log.device != base.device
        or A_log.stride(0) != 1
    ):
        raise ValueError("A_log must be CUDA FP32 [32]")
    if (
        tuple(dt_bias.shape) != (HV,)
        or dt_bias.dtype != torch.bfloat16
        or dt_bias.device != base.device
        or dt_bias.stride(0) != 1
    ):
        raise ValueError("dt_bias must be CUDA BF16 [32]")
    if (
        indices.ndim != 1
        or indices.shape[0] != rows
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.device != base.device
        or indices.stride(0) != 1
    ):
        raise ValueError("indices must be contiguous CUDA int32/int64 [rows]")
    if (
        tuple(out.shape) != (rows, 1, HV, V)
        or out.dtype != torch.bfloat16
        or out.device != base.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous CUDA BF16 [rows,1,32,128]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")

    _ordered_decode[(NV, rows * HV)](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        base,
        indices,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        base.stride(0),
        indices.stride(0),
        base.shape[0],
        scale,
        H,
        HV,
        V,
        K,
        BV,
        NV,
        num_warps=1,
        num_stages=3,
    )
    return out
