# SPDX-License-Identifier: Apache-2.0
"""Offline paired regression; NOT a replacement for the serving benchmark.

Run each configuration in a fresh process with the same GPU and seed. JSON
includes repeat timings, fixed-token prompt logprobs and dispatch receipts.
Timing uses synthetic 3000/1000 requests; precision uses fixed natural text.
"""

import argparse
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ssm-dtype", default="bfloat16")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_num_seqs=64,
        max_model_len=32768,
        max_num_batched_tokens=2048,
        trust_remote_code=True,
        attention_backend="FLASHINFER",
        mamba_ssm_cache_dtype=args.ssm_dtype,
        language_model_only=True,
        cudagraph_capture_sizes=[
            1,
            2,
            4,
            8,
            10,
            12,
            14,
            16,
            20,
            24,
            28,
            32,
            36,
            40,
            48,
            56,
            64,
            72,
            80,
            96,
            112,
            128,
            160,
            192,
            224,
            256,
            320,
            384,
            448,
            512,
            640,
            768,
            784,
            896,
            1024,
            1280,
            1536,
            1792,
            2048,
        ],
    )
    record = dict(
        model=args.model,
        ssm_dtype=args.ssm_dtype,
        timings=[],
        precision=[],
        environment={
            k: v
            for k, v in os.environ.items()
            if k.startswith("VLLM_") or k == "CUDA_VISIBLE_DEVICES"
        },
        versions={
            k: importlib.metadata.version(k)
            for k in ("torch", "triton", "flashinfer-python")
        },
        benchmark_kind="offline_not_serving",
        input_len=3000,
        output_len=1000,
    )

    def save():
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2))

    rng = random.Random(1729)
    prompts = [
        {"prompt_token_ids": [rng.randrange(1000, 20000) for _ in range(3000)]}
        for _ in range(64)
    ]
    tokenizer = llm.get_tokenizer()
    texts = [
        "The capital of France is Paris. Paris is known for its museums.",
        "水在标准大气压下的沸点是一百摄氏度。温度变化会影响物质的状态。",
        "A compiler translates source code into instructions for a computer.",
        "Calculate 17 plus 25. The result is 42, because 17 + 25 = 42.",
    ]
    fixed = [{"prompt_token_ids": tokenizer.encode(texts[i % 4])} for i in range(64)]
    for bs in (1, 4, 16, 32, 64):
        llm.generate(
            prompts[:bs],
            SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
            use_tqdm=False,
        )
        for repeat in range(args.repeats):
            start = time.perf_counter()
            results = llm.generate(
                prompts[:bs],
                SamplingParams(temperature=0, max_tokens=1000, ignore_eos=True),
                use_tqdm=False,
            )
            duration = time.perf_counter() - start
            count = sum(len(r.outputs[0].token_ids) for r in results)
            if count != bs * 1000:
                raise RuntimeError("Incomplete performance requests")
            record["timings"].append(
                dict(
                    bs=bs,
                    repeat=repeat,
                    seconds=duration,
                    output_tokens_per_second=count / duration,
                )
            )
            save()
        for repeat in range(2):
            results = llm.generate(
                fixed[:bs],
                SamplingParams(
                    temperature=0, max_tokens=1, prompt_logprobs=1, logprobs=1
                ),
                use_tqdm=False,
            )
            record["precision"].append(
                dict(
                    bs=bs,
                    repeat=repeat,
                    requests=[
                        dict(
                            tokens=r.prompt_token_ids,
                            next_tokens=r.outputs[0].token_ids,
                            gold_logprobs=[
                                entry[token].logprob
                                for token, entry in zip(
                                    r.prompt_token_ids[1:], r.prompt_logprobs[1:]
                                )
                            ],
                        )
                        for r in results
                    ],
                )
            )
            save()
    record["receipt"] = llm.llm_engine.engine_core.collective_rpc(
        "mach_upstream_receipt"
    )
    save()


if __name__ == "__main__":
    main()
