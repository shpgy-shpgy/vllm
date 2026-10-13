# SPDX-License-Identifier: Apache-2.0
"""Same-card ablations using the user's unmodified serving benchmark.

Copy serving argv from an explicitly selected PID. Only GPU/port/interpreter
and per-profile adapter overrides differ. Never signal the source service.
"""

import argparse
import csv
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT.parent / "serving_benchmark/benchmark_dif_bs.py"
FLAGS = {
    "native": "VLLM_MACH_MXFP8_4B_NATIVE_POLICY",
    "parallel": "VLLM_MXFP6_SM120_PROJECTION_PARALLEL",
    "dual": "VLLM_MXFP6_SM120_MXFP8_DUAL",
    "swiglu": "VLLM_MACH_MXFP8_SWIGLU_QUANT",
    "graph": "VLLM_MACH_MXFP8_GRAPH_POLICY",
}
PROFILES = {"all": {}}
PROFILES["norm_off"] = {"VLLM_MACH_MXFP8_NORM_QUANT": "0"}
PROFILES["norm_on"] = {"VLLM_MACH_MXFP8_NORM_QUANT": "1"}
PROFILES.update({"no_" + name: {flag: "0"} for name, flag in FLAGS.items()})
PROFILES["legacy"] = {flag: "0" for flag in FLAGS.values()}
PROFILES["legacy"].update(
    {
        "VLLM_MACH_MXFP8_GREEDY_HEAD": "0",
        "VLLM_MACH_MXFP8_FULL_HEAD": "0",
    }
)
PROFILES["mach_head"] = {
    "VLLM_HYBRID_NVFP4_LM_HEAD": "0",
    "VLLM_HYBRID_MXFP4_LM_HEAD": "0",
    "VLLM_HYBRID_MXFP8_LM_HEAD": "0",
    "VLLM_MACH_MXFP8_GREEDY_HEAD": "1",
    "VLLM_MACH_MXFP8_FULL_HEAD": "1",
}


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def token_metrics(port):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/metrics", timeout=5
    ) as response:
        lines = response.read().decode().splitlines()
    return {
        key: sum(
            float(line.rsplit(" ", 1)[1])
            for line in lines
            if line.startswith(key + "{")
        )
        for key in ("vllm:generation_tokens_total", "vllm:request_success_total")
    }


def stop_owned(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=15)


def main():
    global ROOT
    p = argparse.ArgumentParser(description=__doc__)
    source_group = p.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-pid", type=int)
    source_group.add_argument("--source-manifest", type=Path)
    p.add_argument("--gpu", required=True)
    p.add_argument("--port", type=int, default=18393)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profiles", nargs="+", choices=PROFILES, required=True)
    p.add_argument("--batches", nargs="+", type=int, default=[64])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--source-root", type=Path, default=ROOT)
    p.add_argument("--model", help="Model override for isolated 2B/4B comparison")
    p.add_argument("--tokenizer", default="/data/models/Qwen3.5-4B")
    args = p.parse_args()
    ROOT = args.source_root.resolve()
    if not (ROOT / "vllm/entrypoints/openai/api_server.py").is_file():
        raise ValueError("Source root must contain the vLLM API server")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.source_manifest:
        saved = json.loads(args.source_manifest.read_text())
        argv = saved["server_argv"]
        source_env = saved["environment"]
    else:
        source = Path(f"/proc/{args.source_pid}")
        argv = source.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
        source_env = dict(
            item.split("=", 1)
            for item in source.joinpath("environ").read_bytes().decode().split("\0")
            if "=" in item
        )
    if "vllm.entrypoints.openai.api_server" not in argv:
        raise ValueError("Source PID is not the requested vLLM API service")
    selected = {
        k: v
        for k, v in source_env.items()
        if k.startswith(
            (
                "VLLM_MACH_",
                "VLLM_MXFP6_",
                "VLLM_HYBRID_",
                "TORCHINDUCTOR_",
            )
        )
        or k
        in (
            "VLLM_USE_V2_MODEL_RUNNER",
            "VLLM_USE_DEEP_GEMM",
            "VLLM_USE_FLASHINFER_SAMPLER",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
    }
    env = os.environ.copy()
    # Do not let the orchestrator shell introduce additional optimization flags.
    for key in list(env):
        if key.startswith(("VLLM_MACH_", "VLLM_MXFP6_", "VLLM_HYBRID_")):
            env.pop(key)
    env.update(selected)
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONPATH"] = str(ROOT)
    argv[0] = sys.executable
    argv[argv.index("--port") + 1] = str(args.port)
    if args.model:
        argv[argv.index("--model") + 1] = args.model
    model_name = argv[argv.index("--served-model-name") + 1]
    save(
        args.output / "manifest.json",
        {
            "source_pid": args.source_pid,
            "source_root": str(ROOT),
            "source_manifest": str(args.source_manifest)
            if args.source_manifest
            else None,
            "server_argv": argv,
            "environment": selected,
            "gpu": args.gpu,
            "seed": args.seed,
            "benchmark": str(BENCH),
            "benchmark_sha256": hashlib.sha256(BENCH.read_bytes()).hexdigest(),
            "note": "num-prompts=3000 creates the dataset; actual BS64 requests=320",
            "profiles": args.profiles,
            "repeats": args.repeats,
            "batches": args.batches,
        },
    )
    results = []
    for index, profile in enumerate(args.profiles):
        directory = args.output / f"{index:02d}-{profile}"
        directory.mkdir()
        overrides = PROFILES[profile]
        run_env = dict(env, **overrides)
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", args.port))
        save(directory / "launch.json", {"argv": argv, "overrides": overrides})
        print(f"START {index} {profile}", flush=True)
        with (directory / "server.log").open("w") as server_log:
            server = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=run_env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 900
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(
                            f"{profile} server exited: {server.returncode}"
                        )
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{args.port}/health", timeout=2
                        ) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        pass
                    time.sleep(2)
                else:
                    raise TimeoutError(f"{profile} startup timeout")
                print(f"READY {profile}", flush=True)
                for repeat in range(args.repeats):
                    for bs in args.batches:
                        csv_path = directory / f"r{repeat}-bs{bs}.csv"
                        cmd = [
                            sys.executable,
                            str(BENCH),
                            "--backend",
                            "openai",
                            "--dataset-name",
                            "random",
                            "--tokenizer",
                            args.tokenizer,
                            "--model",
                            model_name,
                            "--random-input-len",
                            "3000",
                            "--random-output-len",
                            "1000",
                            "--base-url",
                            f"http://127.0.0.1:{args.port}/v1/completions",
                            "--endpoint",
                            "",
                            "--ignore-eos",
                            "--request-rate",
                            "inf",
                            "--num-prompts",
                            "3000",
                            "--from-bs",
                            str(bs),
                            "--to-bs",
                            str(bs),
                            "--seed",
                            str(args.seed),
                            "--result-filename",
                            str(csv_path),
                        ]
                        save(directory / f"r{repeat}-bs{bs}-argv.json", cmd)
                        before = token_metrics(args.port)
                        with (directory / f"r{repeat}-bs{bs}.log").open("w") as log:
                            subprocess.run(
                                cmd,
                                cwd=BENCH.parent,
                                env=run_env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                                timeout=900,
                            )
                        with csv_path.open() as f:
                            row = next(csv.DictReader(f))
                        after = token_metrics(args.port)
                        save(
                            directory / f"r{repeat}-bs{bs}-metrics.json",
                            {
                                "before": before,
                                "after": after,
                                "delta": {
                                    key: after[key] - before[key] for key in before
                                },
                            },
                        )
                        expected = (5 if bs > 16 else 10) * bs
                        if int(row["completed"]) != expected:
                            raise RuntimeError(f"Incomplete requests: {row}")
                        results.append(
                            {
                                "profile": profile,
                                "index": index,
                                "repeat": repeat,
                                "bs": bs,
                                **row,
                            }
                        )
                        save(args.output / "results.json", results)
                        print(
                            f"RESULT {profile} r{repeat} bs{bs}: "
                            f"{row['output_throughput']} tok/s, "
                            f"TPOT={row['median_tpot_ms']}",
                            flush=True,
                        )
            finally:
                stop_owned(server)
        print(f"STOPPED {profile}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
