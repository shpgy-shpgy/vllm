# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from pathlib import Path

import pytest

from vllm._flashinfer_sm120 import prefer_validated_flashinfer


def _make_validated_package(root: Path) -> None:
    metadata = root / "flashinfer_python-0.6.18.dist-info" / "METADATA"
    comm_source = root / "flashinfer" / "jit" / "comm.py"
    mnnvl_source = root / "flashinfer" / "comm" / "mnnvl.py"
    trtllm_ar_source = root / "flashinfer" / "comm" / "trtllm_ar.py"
    trtllm_binding = (
        root / "flashinfer" / "data" / "csrc" / "trtllm_allreduce_fusion.cu"
    )
    trtllm_header = (
        root
        / "flashinfer"
        / "data"
        / "include"
        / "flashinfer"
        / "comm"
        / "trtllm_allreduce_fusion.cuh"
    )
    metadata.parent.mkdir(parents=True)
    comm_source.parent.mkdir(parents=True)
    mnnvl_source.parent.mkdir(parents=True)
    trtllm_ar_source.parent.mkdir(parents=True, exist_ok=True)
    trtllm_binding.parent.mkdir(parents=True)
    trtllm_header.parent.mkdir(parents=True)
    metadata.write_text("Version: 0.6.18\n", encoding="utf-8")
    comm_source.write_text(
        "supported_major_versions=[9, 10, 12]\n",
        encoding="utf-8",
    )
    mnnvl_source.write_text(
        "gpuDirectRDMACapable = int(self._gpu_direct_rdma_capable)\n"
        "self._gpu_direct_rdma_capable = True\n",
        encoding="utf-8",
    )
    trtllm_ar_source.write_text(
        "gpu_direct_rdma_capable=False\n",
        encoding="utf-8",
    )
    trtllm_binding.write_text(
        "params.launch_with_pdl = launch_with_pdl;\n",
        encoding="utf-8",
    )
    trtllm_header.write_text(
        "bool launch_with_pdl = false;\n" + "if (params.launch_with_pdl) {}\n" * 5,
        encoding="utf-8",
    )


def test_vllm_prefers_validated_flashinfer_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "flashinfer"
    _make_validated_package(package_root)
    package_path = str(package_root.resolve())
    monkeypatch.setenv("QWEN35_FLASHINFER_ROOT", package_path)
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    monkeypatch.delenv("QWEN35_USE_VALIDATED_FLASHINFER", raising=False)
    monkeypatch.delitem(sys.modules, "flashinfer", raising=False)

    try:
        prefer_validated_flashinfer()
        assert sys.path[0] == package_path
    finally:
        if package_path in sys.path:
            sys.path.remove(package_path)


def test_vllm_ignores_unvalidated_flashinfer_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "flashinfer"
    _make_validated_package(package_root)
    (package_root / "flashinfer" / "jit" / "comm.py").write_text(
        "supported_major_versions=[9, 10]\n",
        encoding="utf-8",
    )
    package_path = str(package_root.resolve())
    monkeypatch.setenv("QWEN35_FLASHINFER_ROOT", package_path)
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    monkeypatch.delitem(sys.modules, "flashinfer", raising=False)

    prefer_validated_flashinfer()
    assert package_path not in sys.path


def test_vllm_ignores_flashinfer_without_local_ipc_gdr_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "flashinfer"
    _make_validated_package(package_root)
    trtllm_ar_source = (
        package_root
        / "flashinfer"
        / "comm"
        / "trtllm_ar.py"
    )
    trtllm_ar_source.write_text(
        "gpu_direct_rdma_capable=True\n",
        encoding="utf-8",
    )
    package_path = str(package_root.resolve())
    monkeypatch.setenv("QWEN35_FLASHINFER_ROOT", package_path)
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    monkeypatch.delitem(sys.modules, "flashinfer", raising=False)

    prefer_validated_flashinfer()
    assert package_path not in sys.path


def test_vllm_ignores_flashinfer_package_without_pdl_guards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "flashinfer"
    _make_validated_package(package_root)
    trtllm_header = (
        package_root
        / "flashinfer"
        / "data"
        / "include"
        / "flashinfer"
        / "comm"
        / "trtllm_allreduce_fusion.cuh"
    )
    trtllm_header.write_text(
        "bool launch_with_pdl = false;\n",
        encoding="utf-8",
    )
    package_path = str(package_root.resolve())
    monkeypatch.setenv("QWEN35_FLASHINFER_ROOT", package_path)
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    monkeypatch.delitem(sys.modules, "flashinfer", raising=False)

    prefer_validated_flashinfer()
    assert package_path not in sys.path


def test_vllm_requires_flashinfer_version_check_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "flashinfer"
    _make_validated_package(package_root)
    package_path = str(package_root.resolve())
    monkeypatch.setenv("QWEN35_FLASHINFER_ROOT", package_path)
    monkeypatch.delenv("FLASHINFER_DISABLE_VERSION_CHECK", raising=False)
    monkeypatch.delitem(sys.modules, "flashinfer", raising=False)

    prefer_validated_flashinfer()
    assert package_path not in sys.path
