# SPDX-License-Identifier: Apache-2.0
"""Regression tests for GDN prefill output-buffer reuse."""

from __future__ import annotations

import pytest
import torch

from vllm.model_executor.layers.fla.ops.chunk import chunk_gated_delta_rule


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunk_gated_delta_rule_reuses_output_buffer() -> None:
    """The FLA chunk kernel must write into the supplied output view exactly."""
    torch.manual_seed(20260813)
    device = torch.device("cuda")
    batch, tokens, heads, head_dim = 1, 128, 8, 128

    q = torch.randn(
        batch, tokens, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn_like(q)
    g = torch.full(
        (batch, tokens, heads), -0.5, device=device, dtype=torch.bfloat16
    )
    beta = torch.full(
        (batch, tokens, heads), 0.5, device=device, dtype=torch.bfloat16
    )
    initial_state = (
        torch.randn(heads, head_dim, head_dim, device=device, dtype=torch.bfloat16)
        .unsqueeze(0)
        .mul_(0.01)
    )
    cu_seqlens = torch.tensor([0, tokens], device=device, dtype=torch.int32)

    reference, reference_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )

    output_buffer = torch.empty(
        tokens, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    actual, actual_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        core_attn_out=output_buffer,
    )
    torch.cuda.synchronize()

    assert actual.data_ptr() == output_buffer.data_ptr()
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, reference_state, atol=0, rtol=0)
