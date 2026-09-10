# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bitwise MXFP8 payload/scale checks, including CUTLASS padding bytes."""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.mxfp8_fused_padding import (
    quantize_mxfp8_with_padding,
)
from vllm.platforms import current_platform

mxfp6 = pytest.importorskip("mxfp6")
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(120),
    reason="Native MXFP6 reference requires SM120",
)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2560),
        (5, 2560),
        (16, 2560),
        (80, 4096),
        (80, 9216),
        (160, 2560),
        (320, 2560),
        (17, 96),
    ],
)
@torch.inference_mode()
def test_padding_and_values_survive_graph_input_changes(shape):
    torch.manual_seed(9023)
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quantize_mxfp8_with_padding(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        quantize_mxfp8_with_padding(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, scales = quantize_mxfp8_with_padding(x)
    for exponent in (-20, 0, 20):
        x.normal_().mul_(2.0**exponent)
        x.view(-1)[:32] = 0
        x.view(-1)[32:64] = -0.0
        graph.replay()
        reference = mxfp6.quantize_mxfp8(x)
        assert torch.equal(values.view(torch.uint8), reference.values)
        assert torch.equal(scales, reference.scales)


@torch.inference_mode()
def test_all_bf16_constant_group_bit_patterns():
    codes = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    x = codes.view(torch.bfloat16)[:, None].expand(-1, 32).contiguous()
    x = x.view(128, 16384)
    values, scales = quantize_mxfp8_with_padding(x)
    reference = mxfp6.quantize_mxfp8(x)
    assert torch.equal(values.view(torch.uint8), reference.values)
    assert torch.equal(scales, reference.scales)
