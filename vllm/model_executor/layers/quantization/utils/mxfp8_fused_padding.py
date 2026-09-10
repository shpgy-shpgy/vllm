# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 activation quantization with CUTLASS scale padding in one launch."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _quantize_padded(
    X,
    Y,
    S,
    M: tl.constexpr,
    K: tl.constexpr,
    PK: tl.constexpr,
    PM: tl.constexpr,
    BG: tl.constexpr,
    SILU_MUL: tl.constexpr = False,
):
    group = tl.program_id(0) * BG + tl.arange(0, BG)
    row, kg = group // PK, group % PK
    lane = tl.arange(0, 32)
    valid = (row < M) & (kg < K // 32)
    offset = row[:, None] * K + kg[:, None] * 32 + lane[None, :]
    if SILU_MUL:
        source = row[:, None] * (2 * K) + kg[:, None] * 32 + lane[None, :]
        gate = tl.load(X + source, valid[:, None], other=0).to(tl.float32)
        up = tl.load(X + source + K, valid[:, None], other=0).to(tl.float32)
        # Match Inductor's fused native SiLU/mul: round only its final output.
        x = ((gate / (1.0 + tl.exp(-gate))) * up).to(tl.bfloat16).to(tl.float32)
    else:
        x = tl.load(X + offset, valid[:, None], other=0).to(tl.float32)
    magnitude = tl.where(x == x, tl.abs(x), 0.0)
    maximum = tl.max(magnitude, 1)
    raw = tl.maximum(maximum * (1.0 / 448.0), 1.0e-30)
    bits = (raw.to(tl.uint32, bitcast=True) + 0x007FFFFF) & 0x7F800000
    inverse = tl.div_rn(1.0, bits.to(tl.float32, bitcast=True))
    value = x * inverse[:, None]
    value = tl.minimum(
        tl.maximum(value, -448.0, propagate_nan=tl.PropagateNan.ALL),
        448.0,
        propagate_nan=tl.PropagateNan.ALL,
    )
    tl.store(Y + offset, value.to(tl.float8e4nv), valid[:, None])
    # CUTLASS SFA atom: ((32, 4), (32, 4)), strides ((16, 4), (0, 1)).
    sf_offset = (row // 128) * PK * 128 + (kg // 4) * 512
    sf_offset += (row % 32) * 16 + ((row % 128) // 32) * 4 + kg % 4
    code = tl.where(valid, bits >> 23, 127).to(tl.uint8)
    tl.store(S + sf_offset, code, row < PM)


def quantize_mxfp8_with_padding(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert m > 0 and k > 0 and k % 32 == 0
    pm = triton.cdiv(m, 128) * 128
    pk = triton.cdiv(k // 32, 4) * 4
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(pm * pk, device=x.device, dtype=torch.uint8)
    groups = 32 if m <= 128 else 64
    _quantize_padded[(triton.cdiv(pm * pk, groups),)](
        x, out, scales, m, k, pk, pm, groups, num_warps=4
    )
    return out, scales


def silu_mul_quantize_mxfp8_with_padding(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, twice_k = x.shape
    k = twice_k // 2
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert m > 0 and twice_k % 256 == 0
    pm = triton.cdiv(m, 128) * 128
    pk = k // 32
    out = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(pm * pk, device=x.device, dtype=torch.uint8)
    _quantize_padded[(triton.cdiv(pm * pk, 64),)](
        x, out, scales, m, k, pk, pm, 64, SILU_MUL=True, num_warps=4
    )
    return out, scales
