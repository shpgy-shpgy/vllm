# SPDX-License-Identifier: Apache-2.0
"""Exercise the shape-gated 2B BA port and its stock fallback."""

import pytest
import torch

from vllm.mach.mxfp6 import ba
from vllm.platforms import current_platform


def test_m16_4b_keeps_stock_linear(monkeypatch):
    def unexpected(*args):
        raise AssertionError("2B kernel selected for 4B")

    monkeypatch.setattr(ba, "_M16", unexpected)
    x = torch.randn(16, 2560, dtype=torch.bfloat16)
    w = torch.randn(64, 2560, dtype=torch.bfloat16)
    torch.testing.assert_close(
        ba._small_impl(x, w, w.clone()),
        torch.nn.functional.linear(x, w),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(120),
    reason="Requires SM120",
)
@pytest.mark.parametrize("rows", [1, 4, 16, 32, 64])
@torch.inference_mode()
def test_2b_ba_graph_replay(rows, monkeypatch):
    from mxfp6 import mxfp8

    try:
        from vllm.mach.mxfp6 import ba_gemv
    except ModuleNotFoundError as exc:
        if exc.name != "b12x":
            raise
        ba_gemv = None

    mxfp8.load_library()
    monkeypatch.setattr(ba, "_M16", torch.ops.mxfp8_sm120.bf16_ba_m16)
    monkeypatch.setattr(ba, "_GEMV", ba_gemv)
    x = torch.empty((rows, 2048), device="cuda", dtype=torch.bfloat16)
    w = torch.randint(-2, 3, (32, 2048), device="cuda").to(torch.bfloat16)
    private = w.clone()
    if ba_gemv is not None:
        ba_gemv.precompile_bf16_gemv_small_n(private)
    x.random_(-2, 3)
    ba._small_impl(x, w, private)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ba._small_impl(x, w, private)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = ba._small_impl(x, w, private)
    for _ in range(4):
        x.random_(-2, 3)
        graph.replay()
        expected = (x.double() @ w.double().T).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
