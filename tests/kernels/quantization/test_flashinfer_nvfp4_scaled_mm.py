# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    convert_swizzled_to_linear,
    dequantize_nvfp4_to_dtype,
)

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_scaled_fp4_mm,
    has_flashinfer_b12x_gemm,
)
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        reason="Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

DTYPES = [torch.float16, torch.bfloat16]
# m, n, k
SHAPES = [
    (128, 128, 64),
    (128, 128, 128),
    (256, 128, 64),
    (128, 256, 128),
    (1, 128, 128),
]
PAD_SHAPES = [(150, 128, 64), (128, 128, 96), (2, 128, 64), (3, 128, 96)]
SHAPES.extend(PAD_SHAPES)

SEEDS = [42]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("m", [1, 4, 16, 32, 64, 65])
@pytest.mark.parametrize(
    "n,k", [(12288, 2048), (2048, 6144), (5120, 2048), (2048, 2048)]
)
@torch.inference_mode()
def test_nvfp4_decode_routing_preserves_prequantized_activation(m, n, k, monkeypatch):
    """Changing the GEMM consumer must not quantize fused activations twice."""
    if (
        not current_platform.is_device_capability_family(120)
        or not has_flashinfer_b12x_gemm()
    ):
        pytest.skip("requires SM120 with FlashInfer b12x")
    from types import SimpleNamespace

    from vllm.model_executor.kernels.linear.nvfp4 import flashinfer as kernels
    from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
    from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation

    torch.manual_seed(42)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    xs = 2688 / x.float().abs().max()
    ws = 2688 / weight.float().abs().max()
    packed_weight, scale = ops.scaled_fp4_quant(weight, ws)
    kernel = kernels.FlashInferCutlassNvFp4LinearKernel(NvFp4LinearLayerConfig())
    layer = SimpleNamespace(
        output_size_per_partition=n,
        weight=packed_weight,
        weight_scale=scale,
        input_global_scale_inv=xs,
        alpha=1 / (xs * ws),
        nvfp4_b12x_decode=False,
    )
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    expected = kernel.apply_weights(layer, x, bias)
    layer.nvfp4_b12x_decode = True
    actual = kernel.apply_weights(layer, x, bias)
    torch.testing.assert_close(actual, expected, rtol=0.016, atol=0.125)
    data, blockscale = ops.scaled_fp4_quant(x, xs)
    qa = QuantizedActivation(
        data, blockscale, x.dtype, x.shape, kernel.input_quant_key()
    )

    def unexpected_quant(*args, **kwargs):
        raise AssertionError("pre-quantized input must not be quantized again")

    monkeypatch.setattr(kernels, "scaled_fp4_quant", unexpected_quant)
    torch.testing.assert_close(
        kernel.apply_weights(layer, qa, bias), actual, rtol=0, atol=0
    )


@torch.inference_mode()
def test_nvfp4_decode_dispatch_survives_prefill_compile_and_graph_replay(monkeypatch):
    """A prefill trace must not freeze the backend of later decode graphs."""
    if (
        not current_platform.is_device_capability_family(120)
        or not has_flashinfer_b12x_gemm()
    ):
        pytest.skip("requires SM120 with FlashInfer b12x")
    from types import SimpleNamespace

    import flashinfer

    from vllm.model_executor.kernels.linear.nvfp4 import flashinfer as kernels
    from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig

    torch.manual_seed(42)
    w = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    ws = 2688 / w.float().abs().max()
    weight, scale = ops.scaled_fp4_quant(w, ws)
    xs = torch.tensor(512.0, device="cuda")
    layer = SimpleNamespace(
        weight=weight,
        weight_scale=scale,
        input_global_scale_inv=xs,
        alpha=1 / (xs * ws),
        output_size_per_partition=2048,
        nvfp4_b12x_decode=True,
    )
    kernel = kernels.FlashInferCutlassNvFp4LinearKernel(NvFp4LinearLayerConfig())
    calls, graphs = [], []
    original = flashinfer.mm_fp4

    def record_backend(*args, **kwargs):
        calls.append(kwargs["backend"])
        return original(*args, **kwargs)

    def compile_backend(gm, inputs):
        graphs.append(gm)
        return gm.forward

    monkeypatch.setattr(flashinfer, "mm_fp4", record_backend)

    def run(x):
        return kernel.apply_weights(layer, x)

    compiled = torch.compile(run, backend=compile_backend, dynamic=True, fullgraph=True)
    for m in (128, 16, 64, 2, 65):
        x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
        calls.clear()
        actual = compiled(x)
        assert calls == ["b12x" if m <= 64 else "cutlass"]
        torch.testing.assert_close(actual, run(x), rtol=0, atol=0)
    # Quantization buffer shapes may specialize; backend selection must not.
    for gm in graphs:
        mm_nodes = [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.vllm.flashinfer_mm_fp4.default
        ]
        assert mm_nodes
        assert all(node.kwargs["backend"] == "b12x-decode" for node in mm_nodes)

    x = torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)
    compiled(x)
    torch.accelerator.synchronize()
    calls.clear()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled(x)
    assert calls == ["b12x"]
    x.normal_()
    calls.clear()
    graph.replay()
    assert not calls  # Replay uses the captured GPU kernel, without Python dispatch.
    torch.testing.assert_close(actual, run(x), rtol=0, atol=0)


def get_ref_results(
    a_fp4,
    b_fp4,
    a_sf,
    b_sf,
    a_global_scale,
    b_global_scale,
    m,
    n,
    dtype,
    block_size,
    device,
    is_sf_128x4_layout,
):
    _, m_k = a_fp4.shape
    _, n_k = b_fp4.shape
    assert m_k == n_k
    a_in_dtype = dequantize_nvfp4_to_dtype(
        a_fp4,
        a_sf,
        a_global_scale,
        dtype=dtype,
        device=device,
        block_size=block_size,
        is_sf_128x4_layout=is_sf_128x4_layout,
    )
    b_in_dtype = dequantize_nvfp4_to_dtype(
        b_fp4, b_sf, b_global_scale, dtype=dtype, device=device, block_size=block_size
    )
    return torch.matmul(a_in_dtype, b_in_dtype.t())


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("backend", ["cute-dsl", "cutlass", "cudnn", "trtllm", "b12x"])
@pytest.mark.parametrize("autotune", [False, True])
@torch.inference_mode()
def test_flashinfer_nvfp4_gemm(
    dtype: torch.dtype,
    shape: tuple[int, int, int],
    seed: int,
    device: str,
    backend: str,
    autotune: bool,
) -> None:
    if "trtllm" in backend and dtype == torch.float16:
        pytest.skip("Only torch.bfloat16 is supported for TRTLLM FP4 GEMM operations")
    if backend == "cute-dsl" and not current_platform.is_device_capability_family(100):
        pytest.skip("FlashInfer cutedsl backend is only supported on SM10x")
    if backend == "b12x" and not current_platform.has_device_capability(120):
        pytest.skip("b12x FP4 GEMM requires SM120+ (CC 12.0+)")
    if backend == "b12x" and not has_flashinfer_b12x_gemm():
        pytest.skip("b12x FP4 GEMM backend not available in installed FlashInfer")

    set_random_seed(seed)
    m, n, packed_k = shape
    k = packed_k * 2
    block_size = 16
    a_dtype = torch.randn((m, k), dtype=dtype, device=device)
    b_dtype = torch.randn((n, k), dtype=dtype, device=device)

    a_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    b_global_scale = (
        (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(b_dtype.flatten(), dim=-1)
    ).to(torch.float32)
    alpha = 1.0 / (a_global_scale * b_global_scale)

    # ops.scaled_fp4_quant returns swizzled scales, while weights
    # from checkpoints are in linear scales.
    # cutlass and b12x use swizzled scales directly; trtllm needs them unswizzled.
    a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(
        a_dtype, a_global_scale, is_sf_swizzled_layout=True, backend=backend
    )
    is_sf_128x4_layout = not (backend == "trtllm" and m <= 32)

    b_fp4, b_scale_interleaved = ops.scaled_fp4_quant(
        b_dtype, b_global_scale, is_sf_swizzled_layout=True
    )

    # get_ref_results unswizzles the scales internally.
    expected_out = get_ref_results(
        a_fp4,
        b_fp4,
        a_scale_interleaved,
        b_scale_interleaved,
        a_global_scale,
        b_global_scale,
        m,
        n,
        dtype,
        block_size,
        device,
        is_sf_128x4_layout,
    )

    import flashinfer

    if "trtllm" in backend:
        epilogue_tile_m = 128
        b_fp4 = flashinfer.shuffle_matrix_a(b_fp4.view(torch.uint8), epilogue_tile_m)
        b_scale_interleaved = convert_swizzled_to_linear(
            b_scale_interleaved, n, k, block_size
        )
        b_scale_interleaved = (
            flashinfer.shuffle_matrix_sf_a(
                b_scale_interleaved.view(torch.uint8), epilogue_tile_m
            )
            .reshape(b_scale_interleaved.shape)
            .view(torch.float8_e4m3fn)
        )

    with flashinfer.autotune(autotune):
        out = flashinfer_scaled_fp4_mm(
            a_fp4,
            b_fp4,
            a_scale_interleaved,
            b_scale_interleaved,
            alpha,
            dtype,
            backend=backend,
        )

    torch.testing.assert_close(out, expected_out.to(dtype=dtype), atol=1e-1, rtol=1e-1)
