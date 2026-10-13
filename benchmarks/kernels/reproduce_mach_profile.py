# SPDX-License-Identifier: Apache-2.0
"""Launch an isolated Mach-equivalent experiment; never edit serving defaults.

Print-only by default. The model must be the corresponding original Mach asset;
the 4B KV sidecar must belong to that checkpoint, not an unrelated calibration.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def configuration(size, model, directory, port):
    if size not in ("2b", "4b"):
        raise ValueError("Choose 2b or 4b")
    two = size == "2b"
    captures = (
        [4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 2048]
        if two
        else [1, 2, 4, *range(8, 257, 8), 2048]
    )
    switches = {
        "VLLM_PLUGINS": "",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_MACH_DENSE_AUTO": "0",
        "VLLM_QWEN3_5_FUSED_AR_NORM": "0",
        "VLLM_QWEN3_5_FP16_SSM": "0",
        "VLLM_MACH_MOE_PROJECTION_TUNING": "0",
        "VLLM_VOCAB_PARALLEL_GREEDY": "0",
        "VLLM_MACH_GDN_ORDERED": "0",
        "VLLM_MACH_GDN_PERSISTENT": "0",
        "VLLM_MACH_GDN_BA_OVERLAP": "0",
        "VLLM_MACH_FUSED_SWIGLU_QUANT": "0",
        "VLLM_MACH_FUSED_GDN_QUANT": "0",
        "VLLM_MACH_FUSED_AR_QUANT": "0",
        "VLLM_SM120_OWNER_PREFILL": "0",
        "VLLM_SM120_LOSSLESS_PREFILL": "0",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "VLLM_HYBRID_NVFP4_LM_HEAD": "0",
        "VLLM_HYBRID_MXFP4_LM_HEAD": "0",
        "VLLM_HYBRID_MXFP8_LM_HEAD": "0",
        "VLLM_HYBRID_NVFP4_LM_HEAD_USE_FLASHINFER_TOPK": "1",
        "VLLM_MACH_MXFP8_GREEDY_HEAD": "1",
        "VLLM_MACH_MXFP8_FULL_HEAD": "0" if two else "1",
        "VLLM_MACH_MXFP8_SWIGLU_QUANT": "1",
        "VLLM_MACH_MXFP8_GRAPH_POLICY": "1",
        "VLLM_MACH_MXFP8_BF16_SSM_AUDIT": "1",
        "VLLM_MACH_MXFP8_2B_NATIVE": "1" if two else "0",
        "VLLM_MACH_MXFP8_2B_PDL": "1" if two else "0",
        "VLLM_MACH_MXFP8_2B_BA": "1" if two else "0",
        "VLLM_MACH_MXFP8_4B_NATIVE_POLICY": "" if two else "all",
        "VLLM_MXFP6_SM120_MXFP8_NATIVE": "0" if two else "1",
        "VLLM_MXFP6_SM120_MXFP8_DUAL": "0" if two else "1",
        "VLLM_MACH_MXFP8_DUAL_POLICY": "shpgy" if two else "mach",
        "VLLM_MXFP6_SM120_BA_GEMV": "0" if two else "1",
        "VLLM_MXFP6_SM120_PROJECTION_PARALLEL": "0" if two else "1",
        "VLLM_MACH_MXFP8_COMPILE_CHOICES": "0" if two else "1",
        "VLLM_MACH_MXFP8_MODE": "production",
        "VLLM_USE_AOT_COMPILE": "1",
        "VLLM_DISABLE_COMPILE_CACHE": "0",
        "TORCH_COMPILE_FORCE_DISABLE_CACHES": "0",
        "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "0",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VLLM_CACHE_ROOT": str(directory / "cache/vllm"),
        "TORCHINDUCTOR_CACHE_DIR": str(directory / "cache/inductor"),
        "TRITON_CACHE_DIR": str(directory / "cache/triton"),
    }
    if not two:
        sidecar = model / "mach_profile.json"
        if not sidecar.is_file():
            raise ValueError("4B requires its checkpoint's mach_profile.json")
        switches["VLLM_MACH_MXFP8_KV_SCALE_PROFILE"] = str(sidecar)
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--tokenizer",
        str(model),
        "--served-model-name",
        "mach-reproduction",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--language-model-only",
        "--max-model-len",
        "16384" if two else "8192",
        "--max-num-seqs",
        "160" if two else "128",
        "--max-num-batched-tokens",
        "2048",
        "--kv-cache-memory",
        str(19 * 2**30),
        "--kv-cache-dtype",
        "bfloat16" if two else "fp8_e4m3",
        "--no-enable-prefix-caching",
        "--mamba-ssm-cache-dtype",
        "bfloat16",
        "--attention-backend",
        "TRITON_ATTN" if two else "FLASHINFER",
        "--quantization",
        "compressed-tensors",
        "--linear-backend",
        "flashinfer_cutlass",
        "--seed",
        "20260907",
        "--compilation-config",
        json.dumps(
            {
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": captures,
                "max_cudagraph_capture_size": 2048,
            }
        ),
    ]
    if not two:
        argv += ["--gpu-memory-utilization", "0.85"]
    return switches, argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", choices=("2b", "4b"), required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    directory = args.run_dir.resolve()
    settings, argv = configuration(
        args.size, args.model.resolve(), directory, args.port
    )
    # Do not let unrelated earlier experimental switches alter this profile.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("VLLM_MACH_", "VLLM_MXFP6_SM120_", "VLLM_HYBRID_"))
    }
    env.update(settings)
    contract = dict(
        size=args.size,
        model=str(args.model.resolve()),
        argv=argv,
        environment=settings,
        upstream_commit="05aeea3c30431a8c41616b1503aec9675b837f63",
        status="configuration_only_not_qualified",
    )
    print(json.dumps(contract, indent=2), flush=True)
    if not args.execute:
        return
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "launch.json").write_text(json.dumps(contract, indent=2))
    with (directory / "server.log").open("w") as log:
        result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
