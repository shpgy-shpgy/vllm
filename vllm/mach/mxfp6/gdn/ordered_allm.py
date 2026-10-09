# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordered deferred GDN decode for physical M24..128 SM120 batches.

This is the high-M companion to :mod:`ordered_m16_deferred`.  The arithmetic
and W4 scratch protocol intentionally stay identical to the M16 kernel; only
the launch shape is widened to the graph-friendly rows used by the Mach
champion route.  Scratch allocation, materialization, and slot lifecycle are
shared with the M16 implementation so a batch can move between deferred
physical sizes without changing the KV-cache layout.
"""

from __future__ import annotations

import math

import torch

from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton

from . import ordered_m16_deferred as deferred

HV, H, V, K, BV = deferred.HV, deferred.H, deferred.V, deferred.K, deferred.BV
NV = deferred.NV
QKV = deferred.QKV
W = deferred.W
_LAST_KERNEL = None


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
    pending_k,
    pending_d,
    coeff,
    prefix,
    age,
    stride_mixed: tl.constexpr,
    stride_a: tl.constexpr,
    stride_b: tl.constexpr,
    stride_base_page: tl.constexpr,
    stride_index: tl.constexpr,
    P: tl.constexpr,
    W: tl.constexpr,
    stride_pending_k_page: tl.constexpr,
    stride_pending_d_page: tl.constexpr,
    stride_coeff_page: tl.constexpr,
    stride_prefix_page: tl.constexpr,
    stride_age_page: tl.constexpr,
    SCALE: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
):
    value_tile = tl.program_id(0)
    value_head = tl.program_id(1)
    request = tl.program_id(2)
    key_head = value_head // (HV // H)

    key_offsets = tl.arange(0, K)
    value_offsets = tl.arange(0, BV)
    value_offset = value_tile * BV + value_offsets
    state_idx = tl.load(indices + request * stride_index).to(tl.int32)
    output_ptr = out + (request * HV + value_head) * V + value_offset
    if (state_idx <= 0) | (state_idx >= P):
        tl.store(
            output_ptr,
            tl.full((BV,), 0, tl.float32).to(out.dtype.element_ty),
        )
        return

    meta = value_head * NV + value_tile
    meta_ptr = state_idx * stride_age_page + meta
    old_age = tl.load(age + meta_ptr)
    if (old_age < 0) | (old_age >= W):
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
    pending_k_head = pending_k + state_idx * stride_pending_k_page + value_head * W * K
    pending_d_tile = pending_d + state_idx * stride_pending_d_page + meta * W * BV
    coeff_tile = coeff + state_idx * stride_coeff_page + meta * W

    # Replay raw factors in token order before applying the current token.
    for j in range(W):
        if j < old_age:
            old_alpha = tl.load(coeff_tile + j).to(tl.float32)
            old_key = tl.load(pending_k_head + j * K + key_offsets).to(tl.float32)
            old_delta = tl.load(pending_d_tile + j * BV + value_offsets).to(tl.float32)
            hidden = tl.fma(old_delta[:, None], old_key[None, :], hidden * old_alpha)

    mixed_ptr = mixed_qkv + request * stride_mixed
    query = tl.load(mixed_ptr + key_head * K + key_offsets).to(tl.float32)
    key = tl.load(mixed_ptr + H * K + key_head * K + key_offsets).to(tl.float32)
    value = tl.load(mixed_ptr + 2 * H * K + value_head * V + value_offset).to(
        tl.float32
    )
    query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
    key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
    query = query * SCALE

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

    if old_age == W - 1:
        tl.store(state_ptr, hidden)
        tl.store(age + meta_ptr, 0)
        tl.store(prefix + state_idx * stride_prefix_page + meta, 1.0)
    else:
        # All value tiles share one K vector.  Tile zero is the single writer.
        if value_tile == 0:
            tl.store(pending_k_head + old_age * K + key_offsets, key)
        tl.store(pending_d_tile + old_age * BV + value_offsets, value)
        tl.store(coeff_tile + old_age, alpha)
        tl.store(age + meta_ptr, old_age + 1)


def decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    base: torch.Tensor,
    out: torch.Tensor,
    indices: torch.Tensor,
    scratch: deferred.DeferredScratch,
    scale: float = K**-0.5,
) -> torch.Tensor:
    """Run ordered deferred decode for M24..128 rows in steps of eight."""

    pages, window = deferred.validate(base, scratch)
    deferred._check_indices(indices, base)
    rows = int(mixed_qkv.shape[0])
    if rows < 24 or rows > 128 or rows % 8 or tuple(mixed_qkv.shape) != (rows, QKV):
        raise ValueError("ordered high-M GDN supports only M24..128 multiples of 8")
    if (
        mixed_qkv.dtype != torch.bfloat16
        or not mixed_qkv.is_cuda
        or mixed_qkv.device != base.device
        or mixed_qkv.stride(1) != 1
    ):
        raise ValueError("mixed_qkv must be CUDA BF16 [M,8192] stride1")
    if (
        tuple(a.shape) != (rows, HV)
        or tuple(b.shape) != (rows, HV)
        or a.dtype != torch.bfloat16
        or b.dtype != torch.bfloat16
        or a.device != base.device
        or b.device != base.device
        or a.stride(1) != 1
        or b.stride(1) != 1
    ):
        raise ValueError("a/b must be CUDA BF16 [M,32] stride1")
    if (
        tuple(A_log.shape) != (HV,)
        or A_log.dtype != torch.float32
        or tuple(dt_bias.shape) != (HV,)
        or dt_bias.dtype != torch.bfloat16
        or A_log.device != base.device
        or dt_bias.device != base.device
        or A_log.stride(0) != 1
        or dt_bias.stride(0) != 1
    ):
        raise ValueError("A_log FP32 and dt_bias BF16 need 32 contiguous heads")
    if (
        tuple(out.shape) != (rows, 1, HV, V)
        or out.dtype != torch.bfloat16
        or out.device != base.device
        or not out.is_contiguous()
        or tuple(indices.shape) != (rows,)
    ):
        raise ValueError("out contiguous BF16 [M,1,32,128], indices [M]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")

    global _LAST_KERNEL
    _LAST_KERNEL = _ordered_decode[(NV, HV, rows)](
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        out,
        base,
        indices,
        scratch.pending_k,
        scratch.pending_d,
        scratch.coeff,
        scratch.prefix,
        scratch.age,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        base.stride(0),
        indices.stride(0),
        pages,
        window,
        scratch.pending_k.stride(0),
        scratch.pending_d.stride(0),
        scratch.coeff.stride(0),
        scratch.prefix.stride(0),
        scratch.age.stride(0),
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


__all__ = ["decode"]
