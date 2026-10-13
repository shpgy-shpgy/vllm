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


@pytest.mark.parametrize("k", [2048, 2560])
@pytest.mark.parametrize("rows", [1, 4, 16, 32, 64])
@torch.inference_mode()
def test_fused_norm_preserves_fp32_residual_sum_on_graph_replay(k, rows):
    """Catch premature BF16 rounding before norm; compare valid scale bytes."""
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        fused_add_rms_norm_mxfp8_quant,
        mxfp8_e4m3_quantize,
    )

    torch.manual_seed(20261013)
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    gamma = 1 + torch.randn(k, device="cuda", dtype=torch.bfloat16).float()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused_add_rms_norm_mxfp8_quant(x, residual, gamma, 1e-6)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        values, scales, output_residual = fused_add_rms_norm_mxfp8_quant(
            x, residual, gamma, 1e-6
        )
    row = torch.arange(rows, device="cuda")[:, None]
    group = torch.arange(k // 32, device="cuda")[None, :]
    offsets = (
        ((row // 128 * ((k // 32 + 3) // 4) + group // 4) * 32 + row % 32) * 4
        + row % 128 // 32
    ) * 4 + group % 4
    for exponent in (-10, 0, 10):
        x.normal_().mul_(2.0**exponent)
        residual.normal_().mul_(2.0**exponent)
        graph.replay()
        summed = x.float() + residual.float()
        normalized = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
        reference, reference_scales = mxfp8_e4m3_quantize(
            (normalized * gamma).to(x.dtype), is_sf_swizzled_layout=True
        )
        assert torch.equal(output_residual, summed.to(x.dtype))
        assert torch.equal(values.view(torch.uint8), reference.view(torch.uint8))
        assert torch.equal(
            scales.flatten()[offsets], reference_scales.flatten()[offsets]
        )


@pytest.mark.parametrize("k,n", [(2048, 12288), (2560, 18432)])
@pytest.mark.parametrize("rows", [1, 4, 16, 32, 64, 65, 129])
@torch.inference_mode()
def test_norm_quant_gateup_matches_native_gemm_on_replay(k, n, rows):
    from vllm.mach.mxfp6.norm_quant import register
    from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import (
        load_mxfp6_sm120_mxfp8_libraries,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    load_mxfp6_sm120_mxfp8_libraries()
    register()
    torch.manual_seed(20261013)
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    gamma = 1 + torch.randn(k, device="cuda", dtype=torch.bfloat16).float()
    weight, ws = mxfp8_e4m3_quantize(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16),
        is_sf_swizzled_layout=True,
    )

    def fused():
        return torch.ops.vllm.mach_norm_quant_gateup(
            x, residual, gamma, 1e-6, weight, ws, k == 2048
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, output_residual = fused()
    for _ in range(3):
        x.normal_()
        residual.normal_()
        graph.replay()
        summed = x.float() + residual.float()
        normalized = summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)
        q, sf = mxfp8_e4m3_quantize(
            (normalized * gamma).to(x.dtype), is_sf_swizzled_layout=True
        )
        expected = torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm(q, weight, sf, ws)
        assert torch.equal(output_residual, summed.to(x.dtype))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
