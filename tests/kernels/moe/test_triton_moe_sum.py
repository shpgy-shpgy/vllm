# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import _triton_moe_sum


def _sequential_sum(value: torch.Tensor) -> torch.Tensor:
    expected = torch.zeros(
        (value.shape[0], value.shape[2]),
        dtype=torch.float32,
        device=value.device,
    )
    for expert in range(value.shape[1]):
        expected += value[:, expert].float()
    return expected.to(value.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(1, 8, 2048), (3, 4, 1536)])
def test_triton_moe_sum_matches_sequential_fp32(shape):
    torch.manual_seed(20260720)
    value = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    output = torch.empty((shape[0], shape[2]), dtype=value.dtype, device="cuda")

    _triton_moe_sum(value, output)

    assert torch.equal(output, _sequential_sum(value))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_moe_sum_qwen35_tp2_b1_cuda_graph_replay():
    torch.manual_seed(20260720)
    value = torch.randn((1, 8, 2048), dtype=torch.bfloat16, device="cuda")
    output = torch.empty((1, 2048), dtype=value.dtype, device="cuda")
    expected = _sequential_sum(value)

    _triton_moe_sum(value, output)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _triton_moe_sum(value, output)
    for _ in range(100):
        graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(output, expected)
