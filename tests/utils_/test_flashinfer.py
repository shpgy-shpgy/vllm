# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from types import SimpleNamespace

from vllm.utils import flashinfer as flashinfer_utils


def test_autotune_with_torch_cuda_delay_restores_original(monkeypatch) -> None:
    calls: list[object] = []

    def original_delay(microseconds: int) -> None:
        calls.append(("original", microseconds))

    @contextlib.contextmanager
    def fake_autotune(*args, **kwargs):
        calls.append((args, kwargs))
        yield

    fake_module = SimpleNamespace(
        autotune=fake_autotune,
        delay_kernel=original_delay,
    )
    monkeypatch.setattr(flashinfer_utils, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        flashinfer_utils,
        "_get_submodule",
        lambda module_name: fake_module,
    )
    monkeypatch.setattr(
        flashinfer_utils.torch.cuda,
        "_sleep",
        lambda cycles: calls.append(("sleep", cycles)),
    )

    with flashinfer_utils.autotune_with_torch_cuda_delay(tune_mode=True):
        fake_module.delay_kernel(7)

    assert calls == [
        ((), {"tune_mode": True}),
        ("sleep", 7000),
    ]
    assert fake_module.delay_kernel is original_delay


def test_autotune_with_torch_cuda_delay_supports_internal_delay(
    monkeypatch,
) -> None:
    calls: list[object] = []

    def original_delay(microseconds: int) -> None:
        calls.append(("original", microseconds))

    @contextlib.contextmanager
    def fake_autotune(*args, **kwargs):
        calls.append((args, kwargs))
        yield

    fake_autotune.__module__ = "flashinfer.autotuner.autotuner"
    api_module = SimpleNamespace(autotune=fake_autotune)
    implementation_module = SimpleNamespace(delay_kernel=original_delay)

    def fake_get_submodule(module_name: str):
        if module_name == "flashinfer.autotuner":
            return api_module
        if module_name == "flashinfer.autotuner.autotuner":
            return implementation_module
        return None

    monkeypatch.setattr(flashinfer_utils, "has_flashinfer", lambda: True)
    monkeypatch.setattr(flashinfer_utils, "_get_submodule", fake_get_submodule)
    monkeypatch.setattr(
        flashinfer_utils.torch.cuda,
        "_sleep",
        lambda cycles: calls.append(("sleep", cycles)),
    )

    with flashinfer_utils.autotune_with_torch_cuda_delay(tune_mode=True):
        implementation_module.delay_kernel(7)

    assert calls == [
        ((), {"tune_mode": True}),
        ("sleep", 7000),
    ]
    assert not hasattr(api_module, "delay_kernel")
    assert implementation_module.delay_kernel is original_delay
