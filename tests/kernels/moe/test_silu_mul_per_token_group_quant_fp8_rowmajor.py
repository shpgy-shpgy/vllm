# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.moe.utils import (
    make_dummy_moe_config,
    make_test_quant_config,
    modular_triton_fused_moe,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    is_deep_gemm_e8m0_used,
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils_moe_fused import (
    silu_mul_per_token_group_quant_fp8_rowmajor,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _get_act_quant_buffers(
    workspace: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_numel = num_tokens * hidden_size
    scale_numel = num_tokens * (hidden_size // group_size)
    scale_offset = (output_numel + 3) & ~3
    workspace_bytes = workspace.view(dtype=torch.uint8).flatten()
    output = workspace_bytes[:output_numel].view(
        dtype=current_platform.fp8_dtype()
    )
    output = output.view(num_tokens, hidden_size)
    scales_end = scale_offset + scale_numel * torch.float32.itemsize
    output_scales = workspace_bytes[scale_offset:scales_end].view(
        dtype=torch.float32
    )
    return output, output_scales.view(
        num_tokens, hidden_size // group_size
    )


@pytest.mark.parametrize("use_ue8m0", [False, True])
@pytest.mark.parametrize("num_tokens", [1, 8, 9])
@torch.inference_mode()
def test_preallocated_rowmajor_act_quant(
    use_ue8m0: bool,
    num_tokens: int,
) -> None:
    torch.manual_seed(20260720)
    packed_size = 512
    hidden_size = packed_size // 2
    group_size = 128

    input = torch.randn(
        (num_tokens, packed_size), dtype=torch.bfloat16, device="cuda"
    )
    act = torch.empty(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    torch.ops._C.silu_and_mul(act, input)
    ref_output, ref_scales = per_token_group_quant_fp8(
        act,
        group_size,
        use_ue8m0=use_ue8m0,
    )

    workspace = torch.empty(
        (num_tokens, 2048), dtype=torch.bfloat16, device="cuda"
    )
    output, output_scales = _get_act_quant_buffers(
        workspace,
        num_tokens,
        hidden_size,
        group_size,
    )
    actual_output, actual_scales = silu_mul_per_token_group_quant_fp8_rowmajor(
        input,
        output=output,
        group_size=group_size,
        use_ue8m0=use_ue8m0,
        output_scales=output_scales,
    )

    assert actual_output.data_ptr() == output.data_ptr()
    assert actual_scales.data_ptr() == output_scales.data_ptr()
    assert actual_output.untyped_storage().data_ptr() == workspace.data_ptr()
    assert actual_scales.untyped_storage().data_ptr() == workspace.data_ptr()
    assert torch.equal(actual_output.view(torch.uint8), ref_output.view(torch.uint8))
    if use_ue8m0:
        assert torch.equal(actual_scales, ref_scales)
    else:
        torch.testing.assert_close(actual_scales, ref_scales, rtol=1e-6, atol=0)



@torch.inference_mode()
def test_strided_scales_and_fused_multiplier() -> None:
    torch.manual_seed(20260720)
    num_tokens = 8
    packed_size = 512
    hidden_size = packed_size // 2
    group_size = 128
    num_groups = hidden_size // group_size

    input = torch.randn(
        (num_tokens, packed_size), dtype=torch.bfloat16, device="cuda"
    )
    multiplier = torch.softmax(
        torch.randn((num_tokens,), dtype=torch.float32, device="cuda"), dim=0
    )
    expected_output, unweighted_scales = (
        silu_mul_per_token_group_quant_fp8_rowmajor(
            input,
            group_size=group_size,
            use_ue8m0=True,
        )
    )
    scale_storage = torch.empty(
        (num_groups, num_tokens), dtype=torch.float32, device="cuda"
    )
    strided_scales = scale_storage.t()
    actual_output = torch.empty_like(expected_output)
    returned_output, returned_scales = (
        silu_mul_per_token_group_quant_fp8_rowmajor(
            input,
            output=actual_output,
            group_size=group_size,
            use_ue8m0=True,
            output_scales=strided_scales,
            scale_multiplier=multiplier,
        )
    )

    assert returned_output.data_ptr() == actual_output.data_ptr()
    assert returned_scales.data_ptr() == strided_scales.data_ptr()
    assert torch.equal(actual_output.view(torch.uint8), expected_output.view(torch.uint8))
    assert torch.equal(
        strided_scales,
        unweighted_scales * multiplier[:, None],
    )
    assert torch.equal(
        scale_storage,
        (unweighted_scales * multiplier[:, None]).t(),
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA Graph test")
@torch.inference_mode()
def test_preallocated_rowmajor_act_quant_cuda_graph() -> None:
    torch.manual_seed(20260720)
    input = torch.randn((8, 512), dtype=torch.bfloat16, device="cuda")
    workspace = torch.empty((8, 2048), dtype=torch.bfloat16, device="cuda")
    output, output_scales = _get_act_quant_buffers(workspace, 8, 256, 128)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        silu_mul_per_token_group_quant_fp8_rowmajor(
            input,
            output=output,
            group_size=128,
            use_ue8m0=True,
            output_scales=output_scales,
        )

    graph.replay()
    torch.cuda.synchronize()
    expected_output = output.clone()
    expected_scales = output_scales.clone()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(output.view(torch.uint8), expected_output.view(torch.uint8))
    assert torch.equal(output_scales, expected_scales)


@torch.inference_mode()
def test_qwen_tp2_modular_e8m0_graph(workspace_init, default_vllm_config) -> None:
    if not is_deep_gemm_e8m0_used():
        pytest.skip("Qwen TP2 case requires UE8M0")

    torch.set_default_device("cuda")
    torch.manual_seed(20260720)
    num_tokens = 1
    hidden_size = 2048
    intermediate_size = 256
    num_experts = 256
    topk = 8

    hidden_states = torch.randn(
        (num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    w1, w2, quant_config = make_test_quant_config(
        num_experts,
        intermediate_size,
        hidden_size,
        torch.bfloat16,
        quant_dtype=torch.float8_e4m3fn,
        per_act_token_quant=False,
        block_shape=[128, 128],
    )
    topk_ids = torch.arange(topk, dtype=torch.int32, device="cuda").view(1, topk)
    topk_weights = torch.softmax(
        torch.randn((num_tokens, topk), dtype=torch.float32, device="cuda"), dim=-1
    )
    kernel = modular_triton_fused_moe(make_dummy_moe_config(), quant_config)

    for _ in range(3):
        output = kernel.apply(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
            global_num_experts=num_experts,
        )
    torch.cuda.synchronize()
    expected = output.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = kernel.apply(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
            global_num_experts=num_experts,
        )
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_output, expected)
    replay_expected = graph_output.clone()
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(graph_output, replay_expected)
