# SPDX-License-Identifier: Apache-2.0
"""Check the upstream producer contract, not just dequantized tolerances."""

import pytest
import torch

from vllm.mach.mxfp6.swiglu_quant import quantize
from vllm.platforms import current_platform


def test_rejects_unsupported_layout():
    with pytest.raises(ValueError):
        quantize(torch.empty((4, 18432), dtype=torch.bfloat16))


def _producer(x):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(120),
    reason="Requires SM120 E8M0 instructions",
)
@pytest.mark.parametrize("width", [12288, 18432])
@pytest.mark.parametrize("rows", [1, 4, 16, 24, 32, 64, 128])
@torch.inference_mode()
def test_compiled_producer_bytes_and_graph_replay(rows, width):
    """Match compiled SiLU/mul + FlashInfer, including changing graph inputs."""
    from flashinfer import mxfp8_quantize

    torch.manual_seed(812)
    x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
    producer = torch.compile(_producer, fullgraph=True)
    producer(x)
    quantize(x)
    k = width // 2
    row = torch.arange(rows, device="cuda")[:, None]
    group = torch.arange(k // 32, device="cuda")[None, :]
    offsets = (
        ((row // 128 * ((k // 32 + 3) // 4) + group // 4) * 32 + row % 32) * 4
        + row % 128 // 32
    ) * 4 + group % 4
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        quantize(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        values, scales = quantize(x)
        consumed = values.float().sum(dim=1)
    for magnitude in (0.0, 0.01, 1.0, 10.0):
        x.normal_().mul_(magnitude)
        x[:, :32] = 0
        graph.replay()
        expected, sf = mxfp8_quantize(producer(x), is_sf_swizzled_layout=True)
        assert torch.equal(values.view(torch.uint8), expected.view(torch.uint8))
        assert torch.equal(scales.flatten()[offsets], sf.flatten()[offsets])
        torch.testing.assert_close(
            consumed, expected.float().sum(dim=1), rtol=0, atol=0
        )


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(120),
    reason="Requires SM120",
)
@pytest.mark.parametrize("width", [12288, 18432])
@pytest.mark.parametrize("rows", [1, 16, 64, 129])
@pytest.mark.parametrize("native", [False, True, "2b", "4b"])
@torch.inference_mode()
def test_down_projection_and_large_m_fallback(rows, width, native):
    from flashinfer import mxfp8_quantize

    from vllm.mach.mxfp6.swiglu_quant import _down_impl
    from vllm.utils.flashinfer import mm_mxfp8

    if native:
        from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import (
            load_mxfp6_sm120_mxfp8_libraries,
        )

        load_mxfp6_sm120_mxfp8_libraries()
    x = torch.randn((rows, width), dtype=torch.bfloat16, device="cuda")
    weight, ws = mxfp8_quantize(
        torch.randn((128, width // 2), dtype=torch.bfloat16, device="cuda"),
        is_sf_swizzled_layout=True,
    )
    producer = torch.compile(_producer, fullgraph=True)
    producer(x)
    backend = f"mach-{native}-native" if native in ("2b", "4b") else "cutlass"
    use_native = native is True
    _down_impl(x, weight, ws, backend, use_native)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _down_impl(x, weight, ws, backend, use_native)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = _down_impl(x, weight, ws, backend, use_native)
    for _ in range(3):
        x.normal_()
        graph.replay()
        values, sf = mxfp8_quantize(producer(x), is_sf_swizzled_layout=True)
        if native in ("2b", "4b"):
            expected = torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm(
                values,
                weight,
                sf,
                ws,
            )
        elif native and rows <= 64:
            expected = torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm_pdl(
                values,
                weight,
                sf,
                ws,
            )
        else:
            expected = mm_mxfp8(
                values, weight.t(), sf, ws, out_dtype=torch.bfloat16, backend="cutlass"
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
