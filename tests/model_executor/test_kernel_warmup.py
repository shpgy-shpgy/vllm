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
