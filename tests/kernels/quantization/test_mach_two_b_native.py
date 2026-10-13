# SPDX-License-Identifier: Apache-2.0
"""Guard model scope and the upstream GEMM-only adapter contract."""

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.mach.mxfp6 import two_b_native as adapter
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "rejected", [None, "4b", "tp", "mtp", "lora", "backend", "dtype", "off"]
)
def test_model_scope(monkeypatch, rejected):
    monkeypatch.setenv("VLLM_MACH_MXFP8_2B_NATIVE", "0" if rejected == "off" else "1")
    monkeypatch.setattr(
        adapter,
        "current_platform",
        NS(is_cuda=lambda: True, is_device_capability=lambda cap: cap == 120),
    )
    config = NS(
        model_config=NS(
            dtype=torch.bfloat16,
            quantization="compressed-tensors",
            hf_text_config=NS(
                model_type="qwen3_5_text",
                hidden_size=2048,
                intermediate_size=6144,
                num_hidden_layers=24,
            ),
        ),
        use_v2_model_runner=True,
        parallel_config=NS(tensor_parallel_size=1),
        lora_config=None,
        speculative_config=None,
        kernel_config=NS(linear_backend="auto"),
    )
    if rejected == "4b":
        config.model_config.hf_text_config.hidden_size = 2560
        config.model_config.hf_text_config.intermediate_size = 9216
    elif rejected == "tp":
        config.parallel_config.tensor_parallel_size = 2
    elif rejected == "mtp":
        config.speculative_config = object()
    elif rejected == "lora":
        config.lora_config = object()
    elif rejected == "backend":
        config.kernel_config.linear_backend = "cutlass"
    elif rejected == "dtype":
        config.model_config.dtype = torch.float16
    assert adapter.eligible_config(config) == (rejected is None)


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(120),
    reason="Requires SM120 native extension",
)
@pytest.mark.parametrize("rows", [1, 4, 16, 24, 32, 64, 129])
@pytest.mark.parametrize(
    "n,k", [(8192, 2048), (5120, 2048), (12288, 2048), (2048, 6144), (2048, 2048)]
)
@torch.inference_mode()
def test_quantization_bias_shape_and_graph(rows, n, k):
    from flashinfer import mxfp8_quantize

    from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import (
        load_mxfp6_sm120_mxfp8_libraries,
    )
    from vllm.utils.flashinfer import mm_mxfp8

    load_mxfp6_sm120_mxfp8_libraries()
    torch.manual_seed(812)
    weight, ws = mxfp8_quantize(
        torch.randn((n, k), device="cuda", dtype=torch.bfloat16),
        is_sf_swizzled_layout=True,
    )
    layer = NS(weight=weight, weight_scale=ws)
    x = torch.randn((1, rows, k), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16)
    adapter.apply_weights(layer, x, bias)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        adapter.apply_weights(layer, x, bias)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = adapter.apply_weights(layer, x, bias)
    for _ in range(3):
        x.normal_()
        graph.replay()
        values, sf = mxfp8_quantize(x.reshape(rows, k), is_sf_swizzled_layout=True)
        expected = adapter.gemm(values, weight, sf, ws) + bias
        torch.testing.assert_close(actual, expected.unsqueeze(0), rtol=0, atol=0)
        reference = (
            mm_mxfp8(
                values, weight.t(), sf, ws, out_dtype=torch.bfloat16, backend="cutlass"
            )
            + bias
        )
        # Different GEMM accumulation schedules can differ by BF16 rounding.
        error = (expected.float() - reference.float()).norm()
        assert error / reference.float().norm() < 0.01
