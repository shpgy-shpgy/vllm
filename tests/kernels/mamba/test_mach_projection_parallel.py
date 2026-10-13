# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check fork/join ordering and allocation lifetime across graph replays."""

from types import SimpleNamespace

import pytest
import torch

from vllm.mach.mxfp6.projection_parallel import project


def test_cpu_fallback_preserves_projection_calls():
    calls = []

    def q(x):
        calls.append("q")
        return x + 1, None

    def b(x):
        calls.append("b")
        return x + 2, None

    layer = SimpleNamespace(_mach_projection_stream=None, in_proj_qkvz=q, in_proj_ba=b)
    x = torch.ones(4, 2560)
    qkvz, ba = project(layer, x)
    assert calls == ["q", "b"]
    torch.testing.assert_close(qkvz, x + 1)
    torch.testing.assert_close(ba, x + 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 4, 16, 32, 48, 56, 64])
@pytest.mark.parametrize("backend", ["bf16", "native", "dual"])
def test_fork_join_graph_replay(rows, backend):
    device = "cuda"
    qweight = torch.randn(12288, 2560, device=device, dtype=torch.bfloat16)
    bweight = torch.randn(64, 2560, device=device, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        _mach_projection_stream=torch.cuda.Stream(),
        in_proj_qkvz=lambda x: (torch.nn.functional.linear(x, qweight), None),
        in_proj_ba=lambda x: (torch.nn.functional.linear(x, bweight), None),
    )
    if backend != "bf16":
        from vllm.mach.mxfp6.dual import _dual_gemm_impl
        from vllm.mach.mxfp6.native import _public_native_impl
        from vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120 import (
            load_mxfp6_sm120_mxfp8_libraries,
        )
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )

        load_mxfp6_sm120_mxfp8_libraries()
        qw, qs = mxfp8_e4m3_quantize(qweight, is_sf_swizzled_layout=True)
        op = _dual_gemm_impl if backend == "dual" else _public_native_impl
        layer.in_proj_qkvz = lambda x: (op(x, qw, qs, "b12x-decode"), None)
    x = torch.randn(rows, 2560, device=device, dtype=torch.bfloat16)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            project(layer, x)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        qkvz, ba = project(layer, x)
        # Consumers on the origin stream must see completed auxiliary output.
        joined = qkvz[:, :64] + ba
    for _ in range(5):
        x.normal_()
        graph.replay()
        expected_q = layer.in_proj_qkvz(x)[0]
        expected_b = torch.nn.functional.linear(x, bweight)
        torch.testing.assert_close(qkvz, expected_q, rtol=0, atol=0)
        torch.testing.assert_close(ba, expected_b, rtol=0, atol=0)
        torch.testing.assert_close(
            joined, expected_q[:, :64] + expected_b, rtol=0, atol=0
        )
