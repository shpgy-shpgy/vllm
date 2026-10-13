# SPDX-License-Identifier: Apache-2.0
"""Summarize existing reproduction artifacts; never run or change a service."""

import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def identity(path):
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def performance(ours, upstream):
    left, right = (read(p / "sixpoint/sixpoint.json") for p in (ours, upstream))
    if not left["complete"] or not right["complete"]:
        raise ValueError("Both original six-point runs must finish")
    if left["protocol_sha256"] != right["protocol_sha256"]:
        raise ValueError("Benchmark protocols differ")
    points = []
    for a, b in zip(left["points"], right["points"], strict=True):
        if a["concurrency"] != b["concurrency"]:
            raise ValueError("Different concurrency points")
        for key in ("contract_sha256", "prompt_token_ids_sha256"):
            if a["screen"][key] != b["screen"][key]:
                raise ValueError(f"Different {key}")
        x, y = (
            v["screen"]["aggregate"]["output_throughput_tokens_per_s"] for v in (a, b)
        )
        request_file = f"sixpoint/c{a['concurrency']}_screen.json"
        requests_a, requests_b = (
            read(p / request_file)["requests"] for p in (ours, upstream)
        )
        points.append(
            {
                "bs": a["concurrency"],
                "shpgy_tok_s": x,
                "upstream_tok_s": y,
                "delta_percent": (x / y - 1) * 100,
                "free_generation_text_matches": sum(
                    u["response_text"] == v["response_text"]
                    for u, v in zip(requests_a, requests_b, strict=True)
                ),
                "compared_requests": len(requests_a),
            }
        )
    return {
        "protocol_sha256": left["protocol_sha256"],
        "points": points,
        "artifacts": [identity(p / "sixpoint/sixpoint.json") for p in (ours, upstream)],
        "scope": "single measured run per side; not a confidence interval",
    }


def fidelity(size, rows, ours, upstream):
    a, b = (read(p / "records.json") for p in (ours, upstream))
    errors = []
    for x, y in zip(a, b, strict=True):
        if x["id"] != y["id"]:
            raise ValueError("Fidelity sample ordering differs")
        errors.extend(
            abs(u - v)
            for u, v in zip(x["gold_logprobs"], y["gold_logprobs"], strict=True)
        )
    return {
        "model": size,
        "physical_rows": rows,
        "queries": len(a),
        "tokens": len(errors),
        "entire_records_equal": a == b,
        "different_logprobs": sum(e != 0 for e in errors),
        "mae": sum(errors) / len(errors),
        "max_abs": max(errors),
        "summaries": [read(p / "summary.json") for p in (ours, upstream)],
        "artifacts": [identity(p / "records.json") for p in (ours, upstream)],
        "scope": "frozen teacher-forced logprobs; not task accuracy or free generation",
    }


def single_stream(size, ours, upstream):
    a, b = read(ours), read(upstream)
    if a["contract_sha256"] != b["contract_sha256"]:
        raise ValueError("BS1 contracts differ")
    if a["prompt_token_ids_sha256"] != b["prompt_token_ids_sha256"]:
        raise ValueError("BS1 inputs differ")
    for run in (a, b):
        if run["aggregate"]["completed"] != run["aggregate"]["requested"]:
            raise ValueError("BS1 requests failed")
    x, y = (r["aggregate"]["output_throughput_tokens_per_s"] for r in (a, b))
    return {
        "model": size,
        "shpgy_tok_s": x,
        "upstream_tok_s": y,
        "delta_percent": (x / y - 1) * 100,
        "artifacts": [identity(p) for p in (ours, upstream)],
        "scope": "supplemental 10-request BS1, not the canonical six-point protocol",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for size in ("2b", "4b"):
        for implementation in ("shpgy", "upstream"):
            parser.add_argument(f"--{implementation}-{size}", type=Path, required=True)
    parser.add_argument(
        "--fidelity",
        nargs=4,
        action="append",
        default=[],
        metavar=("MODEL", "ROWS", "SHPGY_DIR", "UPSTREAM_DIR"),
    )
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument(
        "--bs1",
        nargs=3,
        action="append",
        default=[],
        metavar=("MODEL", "SHPGY_JSON", "UPSTREAM_JSON"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "measured_evidence_not_automatic_acceptance",
        "performance": {
            size: performance(
                getattr(args, f"shpgy_{size}"), getattr(args, f"upstream_{size}")
            )
            for size in ("2b", "4b")
        },
        "fidelity": [
            fidelity(size, int(rows), Path(ours), Path(upstream))
            for size, rows, ours, upstream in args.fidelity
        ],
        "notes": args.note,
        "supplemental_bs1": [
            single_stream(size, Path(ours), Path(upstream))
            for size, ours, upstream in args.bs1
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
