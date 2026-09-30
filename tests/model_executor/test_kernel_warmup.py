# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.warmup.kernel_warmup import (
    _get_mxfp6_sm120_models,
    _get_mxfp6_sm120_warmup_sizes,
)

pytestmark = pytest.mark.cpu_test


def test_mxfp6_warmup_includes_speculator_model() -> None:
    target = torch.nn.Linear(2, 2)
    draft = torch.nn.Linear(2, 2)
    worker = SimpleNamespace(
        get_model=lambda: target,
        model_runner=SimpleNamespace(speculator=SimpleNamespace(model=draft)),
    )

    assert _get_mxfp6_sm120_models(worker) == [target, draft]


def test_mxfp6_warmup_includes_resolved_cudagraph_sizes() -> None:
    manager = SimpleNamespace(get_capture_sizes=lambda: [192, 162, 129])
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(cudagraph_manager=manager),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[128, 160, 192])
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
    )

    assert _get_mxfp6_sm120_warmup_sizes(worker) == [
        1,
        2048,
        128,
        160,
        192,
        192,
        162,
        129,
    ]


def test_mxfp6_warmup_uses_legacy_dispatcher_capture_sizes() -> None:
    dispatcher = SimpleNamespace(
        get_capture_descs=lambda: [
            ("PIECEWISE", [SimpleNamespace(num_tokens=128)]),
            (
                "FULL",
                [
                    SimpleNamespace(num_tokens=129),
                    SimpleNamespace(num_tokens=114),
                ],
            ),
        ]
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(cudagraph_dispatcher=dispatcher),
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[128])
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
    )

    assert _get_mxfp6_sm120_warmup_sizes(worker) == [
        1,
        2048,
        128,
        128,
        129,
        114,
    ]


def test_mxfp8_decode_warmup_covers_dispatch_boundary(monkeypatch) -> None:
    from vllm.model_executor.kernels.linear.mxfp8 import flashinfer
    from vllm.model_executor.warmup.kernel_warmup import (
        _flashinfer_autotune_token_counts,
    )

    monkeypatch.setattr(flashinfer, "use_qwen35_mxfp8_decode", lambda _: True)
    runner = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=2048),
        vllm_config=SimpleNamespace(),
    )
    assert _flashinfer_autotune_token_counts(runner) == (2048, 32)
    runner.scheduler_config.max_num_batched_tokens = 16
    assert _flashinfer_autotune_token_counts(runner) == (16,)


@pytest.mark.parametrize("rows", [33, 36, 40, 48, 56, 64, 128])
def test_mxfp8_decode_high_batch_range_stays_cutlass(rows: int) -> None:
    from vllm.utils.flashinfer import _qwen35_mxfp8_backend

    assert _qwen35_mxfp8_backend(rows, 12288, 2560) == "cutlass"
    assert _qwen35_mxfp8_backend(rows, 2560, 4096) == "cutlass"

    assert _qwen35_mxfp8_backend(32, 12288, 2560) == "b12x"


def test_mxfp8_decode_dispatch_requires_validated_config(monkeypatch) -> None:
    gemm_mm_mxfp8_cute_dsl = pytest.importorskip(
        "flashinfer.gemm.gemm_mm_mxfp8_cute_dsl"
    )
    cpp_ext = pytest.importorskip("flashinfer.jit.cpp_ext")

    from vllm.model_executor.kernels.linear.mxfp8 import flashinfer

    monkeypatch.setattr(flashinfer.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        flashinfer.current_platform, "is_device_capability", lambda _: True
    )
    monkeypatch.setattr(
        gemm_mm_mxfp8_cute_dsl, "_b12x_mxfp8_dsl_supported", lambda: True
    )
    monkeypatch.setattr(cpp_ext, "get_cuda_version", lambda: SimpleNamespace(major=13))
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            quantization="compressed-tensors",
            hf_text_config=SimpleNamespace(
                model_type="qwen3_5_text",
                hidden_size=2560,
                intermediate_size=9216,
            ),
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        kernel_config=SimpleNamespace(linear_backend="auto"),
        lora_config=None,
        speculative_config=None,
    )
    check = flashinfer.use_qwen35_mxfp8_decode
    assert check(config)
    for target, name, value in (
        (config.model_config, "dtype", torch.float16),
        (config.model_config, "quantization", "mxfp8"),
        (config.model_config.hf_text_config, "hidden_size", 4096),
        (config.model_config.hf_text_config, "model_type", "qwen3_5_moe_text"),
        (config.parallel_config, "tensor_parallel_size", 2),
        (config.kernel_config, "linear_backend", "flashinfer_cutlass"),
        (config, "lora_config", object()),
        (config, "speculative_config", object()),
    ):
        original = getattr(target, name)
        setattr(target, name, value)
        assert not check(config)
        setattr(target, name, original)

    def no_cuda_toolkit():
        raise FileNotFoundError("CUDA toolkit unavailable")

    monkeypatch.setattr(cpp_ext, "get_cuda_version", no_cuda_toolkit)
    assert not check(config)
