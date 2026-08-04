# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validated repo-local FlashInfer selection for SM120 TP2 deployments."""

import functools
import os
import sys
from pathlib import Path

VALIDATED_FLASHINFER_VERSION = "0.6.15.post1"


@functools.cache
def is_validated_flashinfer_root(package_root: Path) -> bool:
    metadata = (
        package_root
        / f"flashinfer_python-{VALIDATED_FLASHINFER_VERSION}.dist-info"
        / "METADATA"
    )
    comm_source = package_root / "flashinfer" / "jit" / "comm.py"
    mnnvl_source = package_root / "flashinfer" / "comm" / "mnnvl.py"
    trtllm_binding = (
        package_root / "flashinfer" / "data" / "csrc" / "trtllm_allreduce_fusion.cu"
    )
    trtllm_header = (
        package_root
        / "flashinfer"
        / "data"
        / "include"
        / "flashinfer"
        / "comm"
        / "trtllm_allreduce_fusion.cuh"
    )
    try:
        metadata_text = metadata.read_text(encoding="utf-8")
        comm_text = comm_source.read_text(encoding="utf-8")
        mnnvl_text = mnnvl_source.read_text(encoding="utf-8")
        binding_text = trtllm_binding.read_text(encoding="utf-8")
        header_text = trtllm_header.read_text(encoding="utf-8")
    except OSError:
        return False
    # Relaxed version policy: the exact pin (0.6.15.post1) is fully validated;
    # newer 0.6.x builds (e.g. locally built 0.6.17) are trusted without the
    # on-disk content hash check.
    version_ok = (
        f"Version: {VALIDATED_FLASHINFER_VERSION}" in metadata_text
        or any(f"Version: 0.6.{minor}" in metadata_text for minor in range(16, 100))
    )
    return (
        version_ok
        and "supported_major_versions=[9, 10, 12]" in comm_text
        and "params.launch_with_pdl = launch_with_pdl;" in binding_text
        and "bool launch_with_pdl = false;" in header_text
    )


def prefer_validated_flashinfer() -> None:
    """Place the validated package first before FlashInfer is imported."""
    if (
        "flashinfer" in sys.modules
        or os.getenv("FLASHINFER_DISABLE_VERSION_CHECK") != "1"
        or os.getenv("QWEN35_USE_VALIDATED_FLASHINFER", "1") == "0"
    ):
        return
    default_root = (
        Path(__file__).resolve().parent.parent
        / ".deps"
        / f"flashinfer_{VALIDATED_FLASHINFER_VERSION}_sm120"
    )
    package_root = Path(
        os.getenv("QWEN35_FLASHINFER_ROOT", str(default_root))
    ).resolve()
    if is_validated_flashinfer_root(package_root) and str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
