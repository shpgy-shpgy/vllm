# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the four validated TVM-FFI GDN modules before deployment."""

import hashlib
import json
import platform
import runpy
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.build_py import build_py


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BuildGDN(build_py):
    def run(self):
        import torch
        from flashinfer.jit import env as jit_env
        from torch.utils.cpp_extension import CUDA_HOME

        if torch.__version__ != "2.13.0+cu130":
            raise RuntimeError("Build with the validated PyTorch 2.13.0+cu130")
        if metadata.version("flashinfer-python") != "0.6.18":
            raise RuntimeError("Build with FlashInfer 0.6.18")
        nvcc = (
            subprocess.check_output(
                [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
            )
            if CUDA_HOME
            else ""
        )
        if "release 13.0," not in nvcc:
            raise RuntimeError("Build with the CUDA 13.0 toolkit")
        super().run()
        repo = Path(__file__).resolve().parents[3]
        source_dir = repo / "vllm/mach/mxfp6/gdn"
        source = source_dir / "gdn_fused_decode_sm120.cu"
        generator = runpy.run_path(str(source_dir / "jit.py"))[
            "gen_gdn_fused_decode_module"
        ]
        package = Path(self.build_lib).resolve() / "vllm_mach_gdn"
        library_dir = package / "lib"
        library_dir.mkdir(parents=True, exist_ok=True)
        # Build from source, never package an unrelated pre-existing AOT cache.
        jit_env.FLASHINFER_JIT_DIR = (
            Path(self.build_lib).resolve().parent / "gdn_compile"
        )
        entries = []
        for model, hidden, n_ba, qkv_dim, hv in (
            ("dense27b", 5120, 48, 5120, 24),
            ("moe35b", 2048, 32, 4096, 16),
        ):
            for bits in (16, 32):
                geometry = (hidden, n_ba, qkv_dim, 8, hv, 128, 4, 3, bits)
                spec = generator(*geometry)
                spec.build(verbose=True)
                filename = f"{model}_fp{bits}.so"
                destination = library_dir / filename
                shutil.copy2(spec.jit_library_path, destination)
                entries.append(
                    {
                        "geometry": geometry,
                        "file": f"lib/{filename}",
                        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                        "cuda_flags": spec.extra_cuda_cflags,
                    }
                )
        manifest = {
            "abi": "mach-gdn-sm120-v1",
            "kernel_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "python": ".".join(platform.python_version_tuple()[:2]),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
            "cuda": torch.version.cuda,
            "flashinfer-python": metadata.version("flashinfer-python"),
            "apache-tvm-ffi": metadata.version("apache-tvm-ffi"),
            "nvcc": nvcc,
            "entries": entries,
        }
        (package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


setup(
    name="vllm-mach-gdn",
    version="0.1.0a1",
    description="Precompiled SM120 TP2 persistent GDN, Dense27B and MoE35B",
    packages=["vllm_mach_gdn"],
    package_data={"vllm_mach_gdn": ["manifest.json", "lib/*.so"]},
    python_requires=">=3.12,<3.13",
    install_requires=["apache-tvm-ffi==0.1.11", "flashinfer-python==0.6.18"],
    license="Apache-2.0",
    license_files=["LICENSE"],
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildGDN},
)
