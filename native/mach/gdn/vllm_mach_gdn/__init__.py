# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load packaged TVM-FFI modules without importing the JIT builder."""

import hashlib
import json
import platform
from functools import lru_cache
from importlib import metadata
from pathlib import Path

ABI = "mach-gdn-sm120-v1"


@lru_cache(maxsize=1)
def validate():
    """Validate the binary contract and file digests without requiring a GPU."""
    import torch

    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    actual = {
        "abi": ABI,
        "python": ".".join(platform.python_version_tuple()[:2]),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "cuda": torch.version.cuda,
        "flashinfer-python": metadata.version("flashinfer-python"),
        "apache-tvm-ffi": metadata.version("apache-tvm-ffi"),
    }
    for key, value in actual.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"GDN wheel ABI mismatch: {key} built={manifest.get(key)!r}, "
                f"runtime={value!r}. Install a matching precompiled wheel."
            )
    for entry in manifest["entries"]:
        path = (root / entry["file"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError(f"Invalid GDN library path: {entry['file']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"GDN wheel checksum mismatch: {path.name}")
    return manifest


@lru_cache(maxsize=4)
def load(geometry: tuple, kernel_sha256: str):
    """Load exactly the requested geometry; incompatible wheels never JIT."""
    import tvm_ffi

    manifest = validate()
    if manifest["kernel_sha256"] != kernel_sha256:
        raise RuntimeError(
            "GDN wheel kernel differs from vLLM source; rebuild the wheel"
        )
    for entry in manifest["entries"]:
        if tuple(entry["geometry"]) == geometry:
            return tvm_ffi.load_module(str(Path(__file__).parent / entry["file"]))
    raise RuntimeError(f"GDN precompiled geometry is unavailable: {geometry}")
