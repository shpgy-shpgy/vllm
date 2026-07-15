# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.experts.mxfp6_sm120_moe import (
    Mxfp6Sm120Experts,
    _qwen35_moe_schedule,
    make_mxfp6_sm120_moe_kernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp6_sm120_utils import (
    is_mxfp6_sm120_moe_available,
)

pytestmark = pytest.mark.skipif(
    not is_mxfp6_sm120_moe_available(),
    reason="mxfp6-sm120 MoE backend is unavailable",
)


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (0, "generic"),
        (1, "small_batch"),
        (4, "small_batch"),
        (5, "grouped"),
        (8, "grouped"),
        (9, "grouped"),
        (12, "grouped"),
        (96, "grouped"),
        (97, "generic"),
        (192, "generic"),
    ],
)
def test_qwen35_moe_schedule(num_tokens: int, expected: str) -> None:
    assert _qwen35_moe_schedule(num_tokens) == expected


def _quantize_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import mxfp6

    experts, rows, k = weight.shape
    quantized = mxfp6.quantize_mxfp6(weight.flatten(0, 1))
    values = quantized.values.view(experts, rows, k * 3 // 4)
    scales = quantized.scales.view(experts, -1)
    return values, scales


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mxfp6_sm120_moe_end_to_end(
    dtype: torch.dtype,
    workspace_init,
) -> None:
    del workspace_init
    torch.manual_seed(2035)
    tokens, experts, topk = 7, 8, 2
    hidden_size, intermediate_size = 256, 128
    hidden_states = torch.randn((tokens, hidden_size), device="cuda", dtype=dtype)
    w1_source = torch.randn(
        (experts, 2 * intermediate_size, hidden_size),
        device="cuda",
        dtype=dtype,
    )
    w2_source = torch.randn(
        (experts, hidden_size, intermediate_size),
        device="cuda",
        dtype=dtype,
    )
    w1, w1_scales = _quantize_weight(w1_source)
    w2, w2_scales = _quantize_weight(w2_source)
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=None,
        weight_dtype="mxfp6_e3m2",
        w1_scale=w1_scales,
        w2_scale=w2_scales,
    )
    moe_config = make_dummy_moe_config(
        num_experts=experts,
        experts_per_token=topk,
        hidden_dim=hidden_size,
        intermediate_size=intermediate_size,
        in_dtype=dtype,
        max_num_tokens=tokens,
    )
    kernel = make_mxfp6_sm120_moe_kernel(quant_config, moe_config, routing_tables=None)
    assert isinstance(kernel.fused_experts, Mxfp6Sm120Experts)

    topk_ids = torch.tensor(
        [[0, 1], [7, 2], [2, 5], [1, 0], [6, 4], [3, 7], [5, 2]],
        device="cuda",
        dtype=torch.int32,
    )
    topk_weights = torch.rand((tokens, topk), device="cuda", dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    output = kernel.apply(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=moe_config.activation,
        global_num_experts=experts,
        expert_map=None,
        apply_router_weight_on_input=False,
    )
    torch.accelerator.synchronize()
    assert output.shape == hidden_states.shape
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert output.abs().max() > 0
