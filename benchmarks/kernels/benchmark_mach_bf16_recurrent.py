# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen BF16 GDN layouts against stock with rotating state pools.

Only bitwise-correct candidates are timed. Timing covers complete W4 cycles
across six/eight pools, not one repeatedly cached layer or serving throughput.
"""

import argparse
import itertools
import json
import random
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.mach.mxfp6.gdn import bf16_small_gluon as gluon
from vllm.mach.mxfp6.gdn import champion_ordered_allm_triton as materializer
from vllm.mach.mxfp6.gdn import (
    champion_ordered_full_m4_m16_deferred_dispatch as four_b,
)
from vllm.mach.mxfp6.gdn import two_b_ordered as two_b
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode as stock,
)
from vllm.triton_utils import triton


def allocate(rows, heads, bv):
    pages = rows + 3
    storage = (
        torch.randn(pages, heads * 16384 + 128, device="cuda", dtype=torch.bfloat16)
        * 0.1
    )
    base = storage[:, : heads * 16384].view(pages, heads, 128, 128)
    nv = 128 // bv
    shapes = {
        "pending_k": (pages, heads, 4, 128),
        "pending_d": (pages, heads, nv, 4, bv),
        "coeff": (pages, heads, nv, 4),
        "prefix": (pages, heads, nv),
        "age": (pages, heads, nv),
    }
    scratch = SimpleNamespace(
        **{
            key: torch.zeros(
                shape,
                device="cuda",
                dtype=torch.int32 if key == "age" else torch.float32,
            )
            for key, shape in shapes.items()
        }
    )
    return SimpleNamespace(
        storage=storage,
        base=base,
        scratch=scratch,
        out=torch.empty(rows, 1, heads, 128, device="cuda", dtype=torch.bfloat16),
    )


def flush(state, ids, bv):
    s, base = state.scratch, state.base
    heads = base.shape[1]
    materializer._ordered_materialize[(128 // bv, heads, ids.numel())](
        base,
        ids,
        s.pending_k,
        s.pending_d,
        s.coeff,
        s.prefix,
        s.age,
        base.stride(0),
        ids.stride(0),
        base.shape[0],
        4,
        heads,
        128,
        128,
        bv,
        128 // bv,
        num_warps=1,
        num_stages=3,
    )


@torch.inference_mode()
def measure(heads, rows, args):
    torch.manual_seed(731)
    width = 4096 + heads * 128
    qkv = torch.randn(rows, width + heads * 128, device="cuda", dtype=torch.bfloat16)[
        :, :width
    ]
    a = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    a_log = torch.randn(heads, device="cuda") - 2
    bias = torch.randn(heads, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(1, rows + 1, device="cuda", dtype=torch.int32)
    all_ids = torch.arange(1, rows + 3, device="cuda", dtype=torch.int32)
    op = two_b if heads == 16 else four_b

    def reference(state):
        return stock(
            qkv, a, b, a_log, bias, 128**-0.5, state.base, state.out, ids, True
        )

    def launch(state, variant):
        if variant[0] == "stock":
            return reference(state)
        if variant[0] == "current":
            return op.decode(
                qkv, a, b, a_log, bias, state.base, state.out, ids, state.scratch
            )
        _, bv, warps, state_rows, full = variant
        s, base = state.scratch, state.base
        return gluon._ordered_decode[(128 // bv, heads, rows)](
            qkv,
            a,
            b,
            a_log,
            bias,
            state.out,
            base,
            ids,
            s.pending_k,
            s.pending_d,
            s.coeff,
            s.prefix,
            s.age,
            qkv.stride(0),
            a.stride(0),
            b.stride(0),
            base.stride(0),
            ids.stride(0),
            base.shape[0],
            4,
            128**-0.5,
            16,
            heads,
            128,
            128,
            bv,
            128 // bv,
            full,
            state_rows,
            False,
            num_warps=warps,
            num_stages=3,
            launch_pdl=False,
        )

    variants = [("stock", 32, 1, 1, True), ("current", 32, 0, 0, False)]
    variants += [
        ("gluon", bv, warps, state_rows, full)
        for bv, warps, state_rows, full in itertools.product(
            args.bv, (2, 4, 8), (1, 4), (True, False)
        )
    ]
    pools = args.pools or (6 if heads == 16 else 8)
    states = {
        bv: [allocate(rows, heads, bv) for _ in range(pools)] for bv in {32, *args.bv}
    }
    results, admitted = [], []
    for variant in variants:
        name, bv, _, _, _ = variant
        state = states[bv][0]
        expected = allocate(rows, heads, bv)
        expected.storage.copy_(state.storage)
        canaries = state.storage[:, -128:].clone()
        null_slot = state.base[0].clone()
        state.scratch.age.zero_()
        ids.copy_(torch.arange(1, rows + 1, device="cuda"))
        record = dict(
            model="2B" if heads == 16 else "4B",
            rows=rows,
            variant=list(variant),
            correct=True,
        )
        try:
            kernel = launch(state, variant)
            if name == "gluon":
                record.update(registers=kernel.n_regs, spills=kernel.n_spills)
            state.storage.copy_(expected.storage)
            state.scratch.age.zero_()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch(state, variant)
            state.storage.copy_(expected.storage)
            state.scratch.age.zero_()
            for step in range(args.verify_steps):
                qkv.normal_(std=0.1)
                a.normal_()
                b.normal_()
                ids.copy_(torch.arange(1, rows + 1, device="cuda").roll(step))
                if step % 7 == 2:
                    ids[0] = 0
                elif step % 7 == 3:
                    ids[0] = -1
                graph.replay()
                reference(expected)
                torch.testing.assert_close(state.out, expected.out, rtol=0, atol=0)
                if name != "stock" and step % 7 == 4:
                    flush(state, all_ids, bv)
                    torch.testing.assert_close(
                        state.base, expected.base, rtol=0, atol=0
                    )
            if name != "stock":
                flush(state, all_ids, bv)
            torch.testing.assert_close(state.base, expected.base, rtol=0, atol=0)
            torch.testing.assert_close(state.base[0], null_slot, rtol=0, atol=0)
            torch.testing.assert_close(
                state.storage[:, -128:], canaries, rtol=0, atol=0
            )
        except (AssertionError, triton.CompilationError) as exc:
            record.update(correct=False, error=str(exc)[:1000])
        results.append(record)
        print(json.dumps(record), flush=True)
        if record["correct"]:
            admitted.append((variant, record))
        del expected

    ids.copy_(torch.arange(1, rows + 1, device="cuda"))
    rng = random.Random(731)
    for _ in range(args.repeats):
        rng.shuffle(admitted)
        for variant, record in admitted:
            selected = states[variant[1]]
            for state in selected:
                state.scratch.age.zero_()

            def cycle(selected=selected, variant=variant):
                for _ in range(4):
                    for state in selected:
                        launch(state, variant)

            elapsed = triton.testing.do_bench_cudagraph(cycle, rep=args.rep_ms)
            record.setdefault("samples_us", []).append(elapsed * 1000 / (4 * pools))
    for record in results:
        if record["correct"]:
            record["median_us"] = statistics.median(record["samples_us"])
            print(json.dumps(record), flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=("2B", "4B"), default=["2B", "4B"])
    p.add_argument("--rows", nargs="+", type=int, default=[1, 4, 16, 32, 64])
    p.add_argument("--bv", nargs="+", type=int, choices=(16, 32, 64), default=[32])
    p.add_argument("--pools", type=int, default=0)
    p.add_argument("--verify-steps", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--rep-ms", type=int, default=50)
    args = p.parse_args()
    if torch.cuda.get_device_capability() != (12, 0):
        p.error("requires SM120")
    report = dict(
        torch=torch.__version__,
        triton=triton.__version__,
        device=torch.cuda.get_device_name(),
        results=[],
    )
    for model in args.models:
        for rows in args.rows:
            report["results"].extend(measure(16 if model == "2B" else 32, rows, args))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
