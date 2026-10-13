# SPDX-License-Identifier: Apache-2.0
"""Opt-in 4B full-vocabulary K2048 refinement, adapted from Mach 05aeea3.

Non-candidate scores remain approximate NVFP4 scores. This is not a lossless
replacement for BF16 scoring. Prepare before memory profiling, attach after
compilation, and keep the ordinary sampler and logits postprocessing intact.
"""

from dataclasses import replace
from types import MethodType

import torch

from vllm import envs


def full_logits(hidden, weight, state):
    coarse = state.coarse_logits(hidden, None)
    indices = torch.topk(coarse, 2048, dim=1, sorted=False).indices
    refined = state.refine_logits(hidden, weight, indices, None)
    return coarse.scatter_(1, indices.long(), refined)


def prepare(worker):
    cfg = worker.vllm_config
    if (
        not cfg.use_v2_model_runner
        or cfg.parallel_config.tensor_parallel_size != 1
        or cfg.parallel_config.pipeline_parallel_size != 1
        or cfg.speculative_config is not None
        or cfg.lora_config is not None
        or cfg.model_config.hf_text_config.model_type != "qwen3_5_text"
    ):
        raise ValueError("Full K2048 head requires Qwen3.5 V2 TP1/PP1 without MTP/LoRA")
    if any(
        (
            envs.VLLM_HYBRID_NVFP4_LM_HEAD,
            envs.VLLM_HYBRID_MXFP4_LM_HEAD,
            envs.VLLM_HYBRID_MXFP8_LM_HEAD,
            envs.VLLM_MACH_MXFP8_HEAD_AUDIT
            and not envs.VLLM_MACH_MXFP8_GREEDY_HEAD,
        )
    ):
        raise ValueError("Full K2048 head is exclusive with compact heads/head audit")
    if hasattr(worker, "_mach_full_head"):
        return
    runner = worker.model_runner
    model = getattr(runner.model, "language_model", runner.model)
    head = model.lm_head
    weight = head.weight
    if (
        runner.batch_sharder is not None
        or tuple(weight.shape) != (248320, 2560)
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
        or weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
        or model.logits_processor.org_vocab_size != 248320
        or model.logits_processor.head_dtype not in (None, torch.bfloat16)
    ):
        raise ValueError("Full K2048 requires a tied contiguous BF16 Qwen3.5-4B head")
    from vllm.model_executor.layers import hybrid_nvfp4_lm_head as nv

    detached = torch.nn.Module()
    detached.register_parameter("weight", weight)
    if not nv.prepare_hybrid_nvfp4_lm_head(detached, candidates=2048):
        raise RuntimeError("Full K2048 NVFP4 preparation unavailable")
    state = replace(detached._hybrid_nvfp4_lm_head_state, candidates=2048, max_rows=0)
    nv.warmup_hybrid_nvfp4_lm_head_kernels(
        replace(state, max_rows=cfg.scheduler_config.max_num_seqs), weight, tp_size=1
    )
    worker._mach_full_head = dict(
        model=model, head=head, weight=weight, state=state, calls=0, installed=False
    )


def install(worker):
    record = worker._mach_full_head
    if record["installed"]:
        raise RuntimeError("Full K2048 head already installed")
    model, head, weight = record["model"], record["head"], record["weight"]

    def apply(self, lm_head, hidden_states, embedding_bias):
        if (
            lm_head is not head
            or lm_head.weight is not weight
            or model.model.embed_tokens.weight.data_ptr() != weight.data_ptr()
        ):
            raise RuntimeError("Full K2048 tied weight changed; reload the worker")
        if embedding_bias is not None:
            raise ValueError("Full K2048 head requires bias-free logits")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Full K2048 sampling must run outside model capture")
        record["calls"] += 1
        return full_logits(hidden_states, weight, record["state"])

    model.logits_processor._apply_head = MethodType(apply, model.logits_processor)
    record["installed"] = True


def inspect(worker):
    record = worker._mach_full_head
    return dict(
        installed=record["installed"],
        calls=record["calls"],
        candidates=2048,
        approximate_non_candidate_scores=True,
    )
