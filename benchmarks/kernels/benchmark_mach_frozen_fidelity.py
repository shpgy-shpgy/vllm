# SPDX-License-Identifier: Apache-2.0
"""Reuse Mach's local frozen256 teacher protocol without copying its corpus.

This diagnostic uses fixed physical decode rows, not serving settings. Fixtures
and anchors are read-only; results belong to a new user-selected directory.
"""

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def install_hidden_trace(worker):
    """Untimed diagnostic to distinguish backbone errors from head errors."""
    import types

    worker = getattr(worker, "worker", worker)
    runner = worker.model_runner
    original = runner.sample
    worker._equivalence_hidden_trace = []

    def sample(self, hidden_states, input_batch, grammar_output):
        worker._equivalence_hidden_trace.append(
            hidden_states[input_batch.logits_indices].detach().cpu()
        )
        return original(hidden_states, input_batch, grammar_output)

    runner.sample = types.MethodType(sample, runner)


def save_hidden_trace(worker, path):
    import torch

    worker = getattr(worker, "worker", worker)
    trace = worker._equivalence_hidden_trace
    torch.save(trace, path)
    return {"steps": len(trace), "path": path}


def upstream_receipt(worker, size):
    worker = getattr(worker, "worker", worker)
    if size == "2b":
        from vllm_mach.mxfp8.two_b.profile import inspect_worker

        return inspect_worker(worker)
    from vllm_mach.mxfp8.profile import write_receipt

    return write_receipt(worker, "quality")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--fixtures", type=Path, required=True)
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, choices=(4, 16, 32, 64), default=32)
    p.add_argument("--ssm-dtype", default="bfloat16")
    p.add_argument("--capture-sizes", type=int, nargs="+")
    p.add_argument("--max-num-seqs", type=int)
    p.add_argument("--mach-size", choices=("2b", "4b"))
    p.add_argument("--implementation", choices=("shpgy", "upstream"), default="shpgy")
    p.add_argument("--hidden-trace", action="store_true")
    p.add_argument("--head-audit", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()
    manifest = args.fixtures / "fidelity-samples.json"
    driver = args.fixtures / "qualify_packaged_profile.py"
    teacher = args.fixtures / "teacher_decode.py"
    for path in (manifest, driver, teacher, args.anchor):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.implementation == "upstream" and not args.mach_size:
        raise ValueError("Upstream requires --mach-size")
    if args.mach_size:
        if args.rows not in (32, 64) or args.ssm_dtype != "bfloat16":
            raise ValueError("Mach fidelity profile requires BF16 state and M32/M64")
        from reproduce_mach_profile import configuration

        settings, _ = configuration(
            args.mach_size, Path(args.model).resolve(), args.output / "runtime", 0
        )
        for name in list(os.environ):
            if name.startswith(("VLLM_MACH_", "VLLM_MXFP6_SM120_", "VLLM_HYBRID_")):
                del os.environ[name]
        settings.update(
            VLLM_MACH_MXFP8_MODE="quality",
            VLLM_MACH_MXFP8_QUALITY_ROWS=str(args.rows),
            VLLM_MACH_MXFP8_GRAPH_POLICY="0",
            # Upstream quality captures have no FULL16/24 PDL launches.
            VLLM_MACH_MXFP8_2B_PDL="0",
            VLLM_MACH_MXFP8_HEAD_AUDIT=str(int(args.head_audit)),
        )
        os.environ.update(settings)
        if args.implementation == "upstream":
            for name in list(os.environ):
                if name.startswith(("VLLM_MACH_", "VLLM_MXFP6_SM120_", "VLLM_HYBRID_")):
                    del os.environ[name]
            module = "vllm_mach.mxfp8."
            if args.mach_size == "2b":
                module += "two_b."
            from importlib import import_module

            extra = (
                {"attention_backend": "TRITON_ATTN"} if args.mach_size == "2b" else {}
            )
            import_module(module + "serve").configure_environment(
                args.output / "runtime",
                quality_rows=args.rows,
                ssm_dtype=args.ssm_dtype,
                hybrid_head=True,
                **extra,
            )
            os.environ["VLLM_MACH_HEAD_AUDIT"] = str(int(args.head_audit))
    sys.path.insert(0, str(args.fixtures.resolve()))
    os.environ["PYTHONPATH"] = (
        str(args.fixtures.resolve()) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    os.environ["K4O_QUALITY_ARM"] = "m32"
    # Callable RPC is confined to this offline diagnostic, never a server.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    spec = importlib.util.spec_from_file_location("qualify_packaged_profile", driver)
    q = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = q
    spec.loader.exec_module(q)
    if q.sha(manifest) != q.MANIFEST_SHA or q.sha(teacher) != q.TEACHER_SHA:
        raise ValueError("Frozen manifest/teacher identity differs from Mach protocol")
    samples = q.read(manifest)["samples"]
    anchor = q.read(args.anchor)
    q.check_records(anchor, samples)
    assert len({s["id"] for s in samples}) == 256
    assert sum(len(s["target_token_ids"]) for s in samples) == 10479
    for sample in samples:
        assert (
            sample["full_token_ids"]
            == sample["prompt_token_ids"] + sample["target_token_ids"]
        )
    import torch

    from vllm import LLM

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    kwargs = dict(
        model=args.model,
        tokenizer=args.model,
        tensor_parallel_size=1,
        max_num_seqs=args.max_num_seqs or args.rows,
        max_model_len=1024,
        max_num_batched_tokens=16384 if args.rows == 64 else 8192,
        kv_cache_memory_bytes=(19 if args.rows == 64 else 4) * 2**30,
        attention_backend="FLASHINFER",
        language_model_only=True,
        seed=20260907,
        logprobs_mode="raw_logprobs",
        enable_prefix_caching=False,
        mamba_ssm_cache_dtype=args.ssm_dtype,
        cudagraph_capture_sizes=args.capture_sizes or sorted({4, args.rows}),
    )
    if args.mach_size:
        two = args.mach_size == "2b"
        kwargs.update(
            max_num_seqs=args.rows,
            kv_cache_memory_bytes=(4 if two or args.rows == 32 else 19) * 2**30,
            attention_backend="TRITON_ATTN" if two else "FLASHINFER",
            kv_cache_dtype="bfloat16" if two else "fp8_e4m3",
            quantization="compressed-tensors",
            kernel_config={"linear_backend": "flashinfer_cutlass"},
            cudagraph_capture_sizes=[m for m in (4, 32, 64) if m <= args.rows],
        )
        kwargs["compilation_config"] = {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "max_cudagraph_capture_size": args.rows,
            "cudagraph_capture_sizes": kwargs.pop("cudagraph_capture_sizes"),
        }
    q.write(
        args.output / "contract.json",
        dict(
            llm_args=kwargs,
            implementation=args.implementation,
            diagnostics={
                "head_audit": args.head_audit,
                "hidden_trace": args.hidden_trace,
            },
            assets={
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (manifest, driver, teacher, args.anchor)
            },
            scope="fixed rows; historical BF16 anchor, not task accuracy",
        ),
    )
    llm = LLM(**kwargs)
    try:
        llm.collective_rpc(functools.partial(q.install_teacher, rows=args.rows))
        if args.hidden_trace:
            llm.collective_rpc(install_hidden_trace)
        records, repeats = q.score(llm, samples, args.rows, kwargs, args.output)
        if args.hidden_trace:
            llm.collective_rpc(
                functools.partial(
                    save_hidden_trace, path=str(args.output / "hidden_trace.pt")
                )
            )
        q.write(
            args.output / "receipt.json",
            llm.collective_rpc(functools.partial(upstream_receipt, size=args.mach_size))
            if args.implementation == "upstream"
            else llm.llm_engine.engine_core.collective_rpc("mach_upstream_receipt"),
        )
    finally:
        llm.llm_engine.engine_core.shutdown()
    errors = [
        abs(a - b)
        for x, y in zip(records, anchor, strict=True)
        for a, b in zip(x["gold_logprobs"], y["gold_logprobs"], strict=True)
    ]
    result = dict(
        queries=len(records),
        gold_tokens=len(errors),
        mae=sum(errors) / len(errors),
        max_abs=max(errors),
        repeat_max_abs=max(r["max_abs"] for r in repeats),
        same_checkpoint_verified=False,
    )
    q.write(args.output / "summary.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
