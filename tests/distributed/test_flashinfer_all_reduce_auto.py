# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.distributed.device_communicators.flashinfer_all_reduce as fi_ar
from vllm.envs import environment_variables
from vllm.platforms.interface import DeviceCapability


def _sm120_platform() -> SimpleNamespace:
    return SimpleNamespace(
        is_cuda=lambda: True,
        get_device_capability=lambda device_id=0: DeviceCapability(12, 0),
    )


def test_flashinfer_allreduce_environment_is_tristate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    getter = environment_variables["VLLM_ALLREDUCE_USE_FLASHINFER"]
    monkeypatch.delenv("VLLM_ALLREDUCE_USE_FLASHINFER", raising=False)
    assert getter() is None

    monkeypatch.setenv("VLLM_ALLREDUCE_USE_FLASHINFER", "0")
    assert getter() is False

    monkeypatch.setenv("VLLM_ALLREDUCE_USE_FLASHINFER", "1")
    assert getter() is True


def test_auto_enable_requires_sm120_tp2_and_validated_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fi_ar, "current_platform", _sm120_platform())
    monkeypatch.setattr(fi_ar, "has_validated_sm120_flashinfer", lambda: True)
    monkeypatch.setattr(fi_ar, "get_node_count", lambda: 1)

    assert fi_ar.should_auto_enable_flashinfer_allreduce(2, torch.device("cuda:0"))
    assert fi_ar._standalone_max_workspace_size_mb(2, torch.device("cuda:0")) == 64
    assert not fi_ar.should_auto_enable_flashinfer_allreduce(4, torch.device("cuda:0"))

    monkeypatch.setattr(
        fi_ar,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True,
            get_device_capability=lambda device_id=0: DeviceCapability(10, 0),
        ),
    )
    assert not fi_ar.should_auto_enable_flashinfer_allreduce(2, torch.device("cuda:0"))


def test_validated_sm120_auto_backend_is_trtllm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fi_ar, "current_platform", _sm120_platform())
    monkeypatch.setattr(fi_ar, "has_validated_sm120_flashinfer", lambda: True)
    monkeypatch.setattr(fi_ar, "get_node_count", lambda: 1)
    monkeypatch.delenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", raising=False)

    assert fi_ar._resolve_fi_ar_backend(2, torch.device("cuda:0")) == (
        "trtllm",
        False,
    )


def test_sm120_tp2_auto_path_is_disabled_for_multi_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fi_ar, "current_platform", _sm120_platform())
    monkeypatch.setattr(fi_ar, "has_validated_sm120_flashinfer", lambda: True)
    monkeypatch.setattr(fi_ar, "get_node_count", lambda: 2)
    monkeypatch.delenv("VLLM_FLASHINFER_ALLREDUCE_BACKEND", raising=False)

    device = torch.device("cuda:1")
    assert not fi_ar.should_auto_enable_flashinfer_allreduce(2, device)
    assert fi_ar._resolve_fi_ar_backend(2, device) == ("mnnvl", False)


def test_validated_trtllm_fast_path_forwards_prevalidated_arguments() -> None:
    calls: list[tuple[object, ...]] = []

    def op(*args: object) -> None:
        calls.append(args)

    communicator = fi_ar.FlashInferAllReduce.__new__(fi_ar.FlashInferAllReduce)
    communicator.world_size = 2
    communicator.rank = 1
    communicator._workspace = SimpleNamespace(workspace_tensor="workspace")
    communicator._trtllm_allreduce_op = op

    input_tensor = torch.ones((3, 2048), dtype=torch.bfloat16)
    output = communicator._all_reduce_trtllm_fast(input_tensor)

    assert output.shape == input_tensor.shape
    assert output.dtype == input_tensor.dtype
    assert len(calls) == 1
    args = calls[0]
    assert args[1:6] == (2, 1, 3, 2048, "workspace")
    assert args[6:11] == (False, True, True, False, 0)


def test_trtllm_fast_path_requires_attached_workspace() -> None:
    attached = SimpleNamespace(
        mem_handles=[SimpleNamespace(mapped=True), SimpleNamespace(mapped=True)]
    )
    detached = SimpleNamespace(
        mem_handles=[SimpleNamespace(mapped=True), SimpleNamespace(mapped=False)]
    )

    assert fi_ar._trtllm_workspace_is_attached(attached)
    assert not fi_ar._trtllm_workspace_is_attached(detached)
    assert not fi_ar._trtllm_workspace_is_attached(SimpleNamespace(mem_handles=[]))
