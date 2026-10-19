# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for GDN prefill output-buffer reuse."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.chunk import (
    chunk_gated_delta_rule,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_chunk_gated_delta_rule_reuses_output_buffer() -> None:
    """The FLA chunk kernel must write into the supplied output view exactly."""
    torch.manual_seed(20260813)
    device = torch.device("cuda")
    batch, tokens, heads, head_dim = 1, 128, 8, 128

    q = torch.randn(batch, tokens, heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn_like(q)
    g = torch.full((batch, tokens, heads), -0.5, device=device, dtype=torch.bfloat16)
    beta = torch.full((batch, tokens, heads), 0.5, device=device, dtype=torch.bfloat16)
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
    torch.accelerator.synchronize()

    assert actual.data_ptr() == output_buffer.data_ptr()
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(actual_state, reference_state, atol=0, rtol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 7, 128])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("backend", ["native", "cuda"])
@pytest.mark.usefixtures("default_vllm_config")
def test_output_projection_preserves_per_head_norm(tokens, tp_size, backend):
    """Packed, strided gates must retain per-head normalization and TP layout."""
    torch.manual_seed(42)
    heads, head_dim = 16 // tp_size, 128
    width = heads * head_dim
    x = torch.randn(tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(tokens, 4 * width, device="cuda", dtype=x.dtype)
    z = z[:, 3 * width :].view_as(x)
    norm = RMSNormGated(
        head_dim, eps=1e-6, norm_before_gate=True, device="cuda", dtype=x.dtype
    )
    with torch.no_grad():
        norm.weight.normal_()
    normalize = getattr(norm, f"forward_{backend}")
    weight = torch.randn(32, width, device="cuda", dtype=x.dtype)
    projection = SimpleNamespace(
        norm=normalize,
        out_proj=lambda value: (torch.nn.functional.linear(value, weight), None),
        value_dim=2048,
        tp_size=tp_size,
    )
    with torch.inference_mode():
        expected = normalize(x.reshape(-1, head_dim), z.reshape(-1, head_dim))
        expected = torch.nn.functional.linear(expected.reshape(tokens, width), weight)
        actual = QwenGatedDeltaNetAttention._output_projection(
            cast(QwenGatedDeltaNetAttention, projection), x, z
        )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda()
    or not current_platform.is_device_capability_family(120),
    reason="indexed BF16 GDN prefill requires SM120",
)
@pytest.mark.parametrize("page_stride", [262144, 286720])
@torch.inference_mode()
def test_indexed_prefill_preserves_state_pool(page_stride):
    """Mixed new/cached slots must match packed FP32, including untouched padding."""
    pytest.importorskip("flashinfer.gdn_prefill")
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        _flashinfer_supports_indexed_prefill_state,
        fi_chunk_gated_delta_rule,
    )
    from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
        zero_uninitialized_states,
    )

    if not _flashinfer_supports_indexed_prefill_state():
        pytest.skip("FlashInfer lacks indexed state pool support")
    torch.manual_seed(42)
    storage = torch.randn(5 * page_stride, device="cuda", dtype=torch.bfloat16)
    actual_storage = storage.clone()
    shape, strides = (5, 16, 128, 128), (page_stride, 16384, 128, 1)
    state = storage.as_strided(shape, strides)
    actual_state = actual_storage.as_strided(shape, strides)
    indices = torch.tensor([4, 1, 3], device="cuda", dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1, 64, 560], device="cuda", dtype=torch.int32)
    q = torch.nn.functional.normalize(
        torch.randn(1, 560, 16, 128, device="cuda"), dim=-1
    ).bfloat16()
    k = torch.nn.functional.normalize(torch.randn_like(q).float(), dim=-1).bfloat16()
    v = torch.randn_like(q)
    g = -torch.rand(1, 560, 16, device="cuda")
    beta = torch.rand_like(g)
    output = torch.empty_like(q.squeeze(0))
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    for flags in ([False, True, False], [True, True, True], [True, False, True]):
        has_initial = torch.tensor(flags, device="cuda")
        packed = state[indices]
        packed[~has_initial] = 0
        expected, final = fi_chunk_gated_delta_rule(initial_state=packed, **kwargs)
        state[indices] = final.bfloat16()
        zero_uninitialized_states(actual_state, indices, has_initial)
        actual, actual_final = fi_chunk_gated_delta_rule(
            initial_state=actual_state, state_indices=indices, output=output, **kwargs
        )
        assert actual.data_ptr() == output.data_ptr()
        assert actual_final.data_ptr() == actual_state.data_ptr()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual_storage, storage, atol=0, rtol=0)
