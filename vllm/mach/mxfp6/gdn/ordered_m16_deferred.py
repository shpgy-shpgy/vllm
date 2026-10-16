# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred Ordered GDN decode for the frozen Qwen3.5-4B SM120 profile.

The kernel keeps the recurrent state in FP32 but defers the last W4 updates
to caller-owned scratch.  This makes the M16 decode path cheaper while still
allowing the vLLM scheduler to change physical batch sizes: pending updates
are materialized before any path that reads the base state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from vllm.third_party.flash_linear_attention.ops.op import exp
from vllm.triton_utils import tl, triton

HV, H, V, K, BV = 32, 16, 128, 128, 32
NV = V // BV
QKV = (2 * H * K) + HV * V
W = 4
BASE_PAGE_ELEMS = HV * V * K
PENDING_K_PAGE_ELEMS = HV * W * K
PENDING_D_PAGE_ELEMS = HV * NV * W * BV
COEFF_PAGE_ELEMS = HV * NV * W
META_PAGE_ELEMS = HV * NV
SCRATCH_PAGE_ELEMS = (
    PENDING_K_PAGE_ELEMS
    + PENDING_D_PAGE_ELEMS
    + COEFF_PAGE_ELEMS
    + META_PAGE_ELEMS
    + META_PAGE_ELEMS
)
_LAST_KERNEL = None
_LAST_MATERIALIZE_KERNEL = None


@dataclass
class DeferredScratch:
    pending_k: torch.Tensor  # [P,HV,W,K] FP32; Vtile0 owns current write
    pending_d: torch.Tensor  # [P,HV,NV,W,BV] FP32
    coeff: torch.Tensor  # [P,HV,NV,W] FP32 raw exp(g), not cumulative
    prefix: torch.Tensor  # [P,HV,NV] FP32 compatibility storage, unused
    age: torch.Tensor  # [P,HV,NV] int32


def allocate(pages: int, device: torch.device) -> DeferredScratch:
    """Allocate the fixed W4 scratch layout for one state pool."""

    if pages < 2:
        raise ValueError("ordered deferred GDN requires at least two pages")
    return DeferredScratch(
        pending_k=torch.empty((pages, HV, W, K), dtype=torch.float32, device=device),
        pending_d=torch.empty(
            (pages, HV, NV, W, BV), dtype=torch.float32, device=device
        ),
        coeff=torch.empty((pages, HV, NV, W), dtype=torch.float32, device=device),
        prefix=torch.ones((pages, HV, NV), dtype=torch.float32, device=device),
        age=torch.zeros((pages, HV, NV), dtype=torch.int32, device=device),
    )


def from_state(base: torch.Tensor) -> DeferredScratch:
    """Place W4 scratch in the per-page padding of a bound GDN state pool.

    Qwen3.5's mamba block is padded from 524288 dense FP32 state elements to
    a 589824-element page stride.  The 65536-element gap is private to each
    layer/page and is large enough for all deferred metadata and payloads.
    """

    if (
        base.ndim != 4
        or tuple(base.shape[1:]) != (HV, V, K)
        or base.dtype != torch.float32
        or not base.is_cuda
    ):
        raise ValueError("base must be a CUDA FP32 [P,32,128,128] state view")
    if base.stride(0) < BASE_PAGE_ELEMS + SCRATCH_PAGE_ELEMS:
        raise ValueError(
            "bound GDN state page has no sufficient private padding for W4 "
            f"scratch: stride={base.stride(0)} required>="
            f"{BASE_PAGE_ELEMS + SCRATCH_PAGE_ELEMS}"
        )

    page_offset = BASE_PAGE_ELEMS
    pending_d_offset = page_offset + PENDING_K_PAGE_ELEMS
    coeff_offset = pending_d_offset + PENDING_D_PAGE_ELEMS
    prefix_offset = coeff_offset + COEFF_PAGE_ELEMS
    age_offset = prefix_offset + META_PAGE_ELEMS
    pages = int(base.shape[0])
    page_stride = base.stride(0)
    pending_k = base.as_strided(
        (pages, HV, W, K),
        (page_stride, W * K, K, 1),
        storage_offset=base.storage_offset() + page_offset,
    )
    pending_d = base.as_strided(
        (pages, HV, NV, W, BV),
        (page_stride, NV * W * BV, W * BV, BV, 1),
        storage_offset=base.storage_offset() + pending_d_offset,
    )
    coeff = base.as_strided(
        (pages, HV, NV, W),
        (page_stride, NV * W, W, 1),
        storage_offset=base.storage_offset() + coeff_offset,
    )
    prefix = base.as_strided(
        (pages, HV, NV),
        (page_stride, NV, 1),
        storage_offset=base.storage_offset() + prefix_offset,
    )
    base_int = base.view(torch.int32)
    age = base_int.as_strided(
        (pages, HV, NV),
        (page_stride, NV, 1),
        storage_offset=base_int.storage_offset() + age_offset,
    )
    return DeferredScratch(pending_k, pending_d, coeff, prefix, age)


def validate(base: torch.Tensor, scratch: DeferredScratch) -> tuple[int, int]:
    if base.ndim != 4 or tuple(base.shape[1:]) != (HV, V, K):
        raise ValueError("base must be [P,32,128,128]")
    if base.dtype not in (torch.float32, torch.bfloat16) or not base.is_cuda:
        raise ValueError("base must be CUDA FP32 or BF16")
    if base.stride(3) != 1 or base.stride(2) != K or base.stride(1) != V * K:
        raise ValueError("base inner head,V,K dimensions must be dense")
    if base.stride(0) < HV * V * K:
        raise ValueError("base page stride overlaps another slot")
    pages = int(base.shape[0])
    specs = {
        "pending_k": ((pages, HV, W, K), (W * K, K, 1), PENDING_K_PAGE_ELEMS),
        "pending_d": (
            (pages, HV, NV, W, BV),
            (NV * W * BV, W * BV, BV, 1),
            PENDING_D_PAGE_ELEMS,
        ),
        "coeff": ((pages, HV, NV, W), (NV * W, W, 1), COEFF_PAGE_ELEMS),
        "prefix": ((pages, HV, NV), (NV, 1), META_PAGE_ELEMS),
        "age": ((pages, HV, NV), (NV, 1), META_PAGE_ELEMS),
    }
    for name, (shape, inner_strides, page_elems) in specs.items():
        tensor = getattr(scratch, name)
        dtype = torch.int32 if name == "age" else torch.float32
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or not tensor.is_cuda
            or tensor.device != base.device
            or tuple(tensor.stride()[1:]) != inner_strides
            or tensor.stride(0) < page_elems
        ):
            raise ValueError(f"invalid deferred {name}: {shape} {dtype}")
    return pages, W


def _check_indices(indices: torch.Tensor, base: torch.Tensor) -> None:
    if (
        indices.ndim != 1
        or indices.dtype not in (torch.int32, torch.int64)
        or not indices.is_cuda
        or indices.device != base.device
        or indices.stride(0) <= 0
    ):
        raise ValueError("indices must be CUDA int32/int64 [M] with positive stride")


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
    iv = tl.program_id(0)
    ihv = tl.program_id(1)
    it = tl.program_id(2)
    ik = ihv // (HV // H)
    ok = tl.arange(0, K)
    ov_local = tl.arange(0, BV)
    ov = iv * BV + ov_local
    state_idx = tl.load(indices + it * stride_index).to(tl.int32)
    output_ptr = out + (it * HV + ihv) * V + ov
    if (state_idx <= 0) | (state_idx >= P):
        tl.store(output_ptr, tl.full((BV,), 0, tl.float32).to(out.dtype.element_ty))
        return

    meta = ihv * NV + iv
    meta_ptr = state_idx * stride_age_page + meta
    old_age = tl.load(age + meta_ptr)
    if (old_age < 0) | (old_age >= W):
        tl.store(output_ptr, tl.full((BV,), 0, tl.float32).to(out.dtype.element_ty))
        return
    ptr_base = (
        base
        + state_idx * stride_base_page
        + ihv * V * K
        + ov[:, None] * K
        + ok[None, :]
    )
    b_h = tl.load(ptr_base).to(tl.float32)
    pk_head = pending_k + state_idx * stride_pending_k_page + ihv * W * K
    pd_tile = pending_d + state_idx * stride_pending_d_page + meta * W * BV
    pc_tile = coeff + state_idx * stride_coeff_page + meta * W

    # Replay stored raw factors in token order before applying this token.
    for j in range(W):
        if j < old_age:
            old_alpha = tl.load(pc_tile + j).to(tl.float32)
            old_k = tl.load(pk_head + j * K + ok).to(tl.float32)
            old_d = tl.load(pd_tile + j * BV + ov_local).to(tl.float32)
            scaled = b_h * old_alpha
            b_h = tl.fma(old_d[:, None], old_k[None, :], scaled)
            if base.dtype.element_ty == tl.bfloat16:
                b_h = b_h.to(tl.bfloat16).to(tl.float32)

    ptr_mixed = mixed_qkv + it * stride_mixed
    b_q = tl.load(ptr_mixed + ik * K + ok).to(tl.float32)
    b_k = tl.load(ptr_mixed + H * K + ik * K + ok).to(tl.float32)
    b_v = tl.load(ptr_mixed + 2 * H * K + ihv * V + ov).to(tl.float32)
    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * SCALE
    a_val = tl.load(a + it * stride_a + ihv).to(tl.float32)
    b_val = tl.load(b + it * stride_b + ihv).to(tl.float32)
    A_log_val = tl.load(A_log + ihv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + ihv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val)

    alpha = exp(g_val)
    b_h *= alpha
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(output_ptr, b_o.to(out.dtype.element_ty))

    if old_age == W - 1:
        tl.store(ptr_base, b_h)
        tl.store(age + meta_ptr, 0)
        tl.store(prefix + state_idx * stride_prefix_page + meta, 1.0)
    else:
        if iv == 0:
            tl.store(pk_head + old_age * K + ok, b_k)
        tl.store(pd_tile + old_age * BV + ov_local, b_v)
        tl.store(pc_tile + old_age, alpha)
        tl.store(age + meta_ptr, old_age + 1)


@triton.jit
def _ordered_materialize(
    base,
    indices,
    pending_k,
    pending_d,
    coeff,
    prefix,
    age,
    stride_base_page: tl.constexpr,
    stride_index: tl.constexpr,
    P: tl.constexpr,
    stride_pending_k_page: tl.constexpr,
    stride_pending_d_page: tl.constexpr,
    stride_coeff_page: tl.constexpr,
    stride_prefix_page: tl.constexpr,
    stride_age_page: tl.constexpr,
    W: tl.constexpr,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
):
    iv, ihv, it = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    state_idx = tl.load(indices + it * stride_index).to(tl.int32)
    if (state_idx <= 0) | (state_idx >= P):
        return
    meta = ihv * NV + iv
    meta_ptr = state_idx * stride_age_page + meta
    old_age = tl.load(age + meta_ptr)
    if (old_age <= 0) | (old_age >= W):
        return
    ok, ov_local = tl.arange(0, K), tl.arange(0, BV)
    ov = iv * BV + ov_local
    ptr_base = (
        base
        + state_idx * stride_base_page
        + ihv * V * K
        + ov[:, None] * K
        + ok[None, :]
    )
    b_h = tl.load(ptr_base).to(tl.float32)
    pk_head = pending_k + state_idx * stride_pending_k_page + ihv * W * K
    pd_tile = pending_d + state_idx * stride_pending_d_page + meta * W * BV
    pc_tile = coeff + state_idx * stride_coeff_page + meta * W
    for j in range(W):
        if j < old_age:
            old_alpha = tl.load(pc_tile + j).to(tl.float32)
            old_k = tl.load(pk_head + j * K + ok).to(tl.float32)
            old_d = tl.load(pd_tile + j * BV + ov_local).to(tl.float32)
            scaled = b_h * old_alpha
            b_h = tl.fma(old_d[:, None], old_k[None, :], scaled)
            if base.dtype.element_ty == tl.bfloat16:
                b_h = b_h.to(tl.bfloat16).to(tl.float32)
    tl.store(ptr_base, b_h)
    tl.store(prefix + state_idx * stride_prefix_page + meta, 1.0)
    tl.store(age + meta_ptr, 0)


@triton.jit
def _ordered_reset(
    base,
    indices,
    prefix,
    age,
    stride_base_page: tl.constexpr,
    stride_index: tl.constexpr,
    P: tl.constexpr,
    stride_prefix_page: tl.constexpr,
    stride_age_page: tl.constexpr,
    ZERO_BASE: tl.constexpr,
    HV: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
    NV: tl.constexpr,
):
    iv, ihv, it = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    state_idx = tl.load(indices + it * stride_index).to(tl.int32)
    if (state_idx <= 0) | (state_idx >= P):
        return
    meta = ihv * NV + iv
    tl.store(age + state_idx * stride_age_page + meta, 0)
    tl.store(prefix + state_idx * stride_prefix_page + meta, 1.0)
    if ZERO_BASE:
        ok, ov = tl.arange(0, K), iv * BV + tl.arange(0, BV)
        ptr_base = (
            base
            + state_idx * stride_base_page
            + ihv * V * K
            + ov[:, None] * K
            + ok[None, :]
        )
        tl.store(ptr_base, tl.full((BV, K), 0, tl.float32))


def decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    base: torch.Tensor,
    out: torch.Tensor,
    indices: torch.Tensor,
    scratch: DeferredScratch,
    scale: float = K**-0.5,
) -> torch.Tensor:
    pages, _ = validate(base, scratch)
    _check_indices(indices, base)
    rows = int(mixed_qkv.shape[0])
    if rows != 16 or tuple(mixed_qkv.shape) != (rows, QKV):
        raise ValueError("ordered deferred GDN supports only packed M16 QKV8192")
    if (
        mixed_qkv.dtype != torch.bfloat16
        or not mixed_qkv.is_cuda
        or mixed_qkv.device != base.device
        or mixed_qkv.stride(1) != 1
    ):
        raise ValueError("mixed_qkv must be CUDA BF16 [16,8192] stride1")
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
        raise ValueError("a/b must be CUDA BF16 [16,32] stride1")
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
        raise ValueError("out contiguous BF16 [16,1,32,128], indices [16]")
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
        W,
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


def materialize_slots(
    base: torch.Tensor, indices: torch.Tensor, scratch: DeferredScratch
) -> None:
    """Apply raw pending factors before a stock path reads the base state."""

    pages, _ = validate(base, scratch)
    _check_indices(indices, base)
    if indices.numel() == 0:
        return
    global _LAST_MATERIALIZE_KERNEL
    _LAST_MATERIALIZE_KERNEL = _ordered_materialize[(NV, HV, indices.numel())](
        base,
        indices,
        scratch.pending_k,
        scratch.pending_d,
        scratch.coeff,
        scratch.prefix,
        scratch.age,
        base.stride(0),
        indices.stride(0),
        pages,
        scratch.pending_k.stride(0),
        scratch.pending_d.stride(0),
        scratch.coeff.stride(0),
        scratch.prefix.stride(0),
        scratch.age.stride(0),
        W,
        HV,
        V,
        K,
        BV,
        NV,
        num_warps=1,
        num_stages=3,
    )


def reset_slots(
    base: torch.Tensor,
    indices: torch.Tensor,
    scratch: DeferredScratch,
    *,
    zero_base: bool = False,
) -> None:
    """Reset metadata when a scheduler slot is assigned to a new request."""

    pages, _ = validate(base, scratch)
    _check_indices(indices, base)
    if indices.numel() == 0:
        return
    _ordered_reset[(NV, HV, indices.numel())](
        base,
        indices,
        scratch.prefix,
        scratch.age,
        base.stride(0),
        indices.stride(0),
        pages,
        scratch.prefix.stride(0),
        scratch.age.stride(0),
        zero_base,
        HV,
        V,
        K,
        BV,
        NV,
        num_warps=1,
        num_stages=3,
    )
