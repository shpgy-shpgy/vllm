# SPDX-License-Identifier: Apache-2.0
"""Opt-in Mach K128 greedy head with model-specific full-score fallback."""

import types
from dataclasses import replace


def _eligible(runner, batch, grammar_output):
    sampler = runner.sampler
    if (
        runner.lora_config is not None
        or runner.speculative_config is not None
        or runner.batch_sharder is not None
        or grammar_output is not None
        or batch.is_prefilling_np.any()
        or sampler is None
        or batch.num_reqs == 0
        or batch.num_draft_tokens != 0
        or sampler.compute_nans
        or sampler.return_sampling_mask
        or sampler.trace_replay_state is not None
    ):
        return False
    indices = batch.idx_mapping_np
    return bool(
        (sampler.sampling_states.temperature.np[indices] == 0.0).all()
        and not sampler.needs_logits_processing[indices].any()
        and sampler.get_logprobs_dims(indices) is None
    )


def _head_eligible(processor, head, hidden, state, bf16):
    shard = head.shard_indices
    active_vocab = shard.org_vocab_end_index - shard.org_vocab_start_index
    return (
        processor.head_dtype in (None, bf16)
        and processor.soft_cap is None
        and processor.scale == 1.0
        and shard.num_added_elements_padded == 0
        and state.candidates <= 1024
        and state.can_use(
            hidden, bf16_weight=head.weight, active_vocab_size=active_vocab, top_k=1
        )
    )


def _wrap_sample(runner, head, processor, state, bf16, argmax, compact_output, stats):
    original = runner.sample

    def sample(self, hidden_states, input_batch, grammar_output):
        stats["calls"] += 1
        if not _eligible(self, input_batch, grammar_output):
            stats["sampling_fallback_calls"] += 1
            return original(hidden_states, input_batch, grammar_output)
        hidden = hidden_states[input_batch.logits_indices]
        if not _head_eligible(processor, head, hidden, state, bf16):
            stats["head_fallback_calls"] += 1
            return original(hidden_states, input_batch, grammar_output)
        # Preserve the exact qualified team operation order, dtype and selector.
        coarse = state.coarse_logits(hidden, None)
        active_vocab = (
            head.shard_indices.org_vocab_end_index
            - head.shard_indices.org_vocab_start_index
        )
        coarse[:, active_vocab:] = -float("inf")
        candidates = state.select_candidates(coarse)
        refined = state.refine_logits(hidden, head.weight, candidates, None)
        _, token_ids = argmax(
            refined, candidates, index_offset=head.shard_indices.org_vocab_start_index
        )
        output = compact_output(self.sampler, token_ids, input_batch)
        rows = int(hidden.shape[0])
        stats["hybrid_success_calls"] += 1
        stats["hybrid_success_rows"] += rows
        by_rows = stats["calls_by_rows"]
        by_rows[str(rows)] = by_rows.get(str(rows), 0) + 1
        return output, output.num_sampled, output.num_rejected

    runner.sample = types.MethodType(sample, runner)


def prepare(worker):
    import torch

    from vllm import envs
    from vllm.model_executor.layers import hybrid_nvfp4_lm_head as nv

    cfg, runner = worker.vllm_config, worker.model_runner
    model = getattr(runner.model, "language_model", runner.model)
    head = model.lm_head
    if any(
        (
            envs.VLLM_HYBRID_NVFP4_LM_HEAD,
            envs.VLLM_HYBRID_MXFP4_LM_HEAD,
            envs.VLLM_HYBRID_MXFP8_LM_HEAD,
        )
    ):
        raise ValueError("Mach greedy head requires generic hybrid heads disabled")
    if (
        not cfg.use_v2_model_runner
        or cfg.parallel_config.tensor_parallel_size != 1
        or cfg.parallel_config.pipeline_parallel_size != 1
        or cfg.speculative_config is not None
        or cfg.lora_config is not None
        or runner.batch_sharder is not None
        or tuple(head.weight.shape) not in ((248320, 2048), (248320, 2560))
        or head.weight.dtype != torch.bfloat16
        or not head.weight.is_cuda
        or not head.weight.is_contiguous()
        or head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    ):
        raise ValueError("Mach greedy head requires tied BF16 Qwen3.5 2B/4B TP1")
    if hasattr(worker, "_mach_greedy_head"):
        return
    detached = None
    if head.weight.shape[1] == 2560:
        if not hasattr(worker, "_mach_full_head"):
            raise ValueError(
                "4B Mach greedy head requires its K2048 full-score fallback"
            )
        state = worker._mach_full_head["state"]
    else:
        detached = torch.nn.Module()
        detached.register_parameter("weight", head.weight)
        if not nv.prepare_hybrid_nvfp4_lm_head(detached, candidates=128):
            raise RuntimeError("Mach K128 head preparation unavailable")
        state = detached._hybrid_nvfp4_lm_head_state
    state = replace(state, candidates=128, max_rows=64, can_use_failure_counts={})
    nv.warmup_hybrid_nvfp4_lm_head_kernels(state, head.weight, tp_size=1)
    worker._mach_greedy_head = dict(
        state=state,
        head=head,
        model=model,
        detached=detached,
        weight_ptr=head.weight.data_ptr(),
        installed=False,
        stats=dict(
            calls=0,
            sampling_fallback_calls=0,
            head_fallback_calls=0,
            hybrid_success_calls=0,
            hybrid_success_rows=0,
            calls_by_rows={},
        ),
    )


def install(worker):
    import torch

    from vllm.model_executor.layers.argmax_triton import indexed_argmax_triton
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    record = worker._mach_greedy_head
    if record["installed"]:
        raise RuntimeError("Mach greedy head already installed")

    def output(sampler, token_ids, batch):
        sampled, rejected = get_num_sampled_and_rejected(
            batch.seq_lens.new_ones(batch.num_reqs),
            batch.seq_lens,
            batch.cu_num_logits,
            batch.idx_mapping,
            sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=token_ids.to(torch.int64).view(-1, 1),
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=sampled,
            num_rejected=rejected,
        )

    runner, model = worker.model_runner, record["model"]
    _wrap_sample(
        runner,
        record["head"],
        model.logits_processor,
        record["state"],
        torch.bfloat16,
        indexed_argmax_triton,
        output,
        record["stats"],
    )
    original = runner.sample

    def checked(self, hidden_states, input_batch, grammar_output):
        if (
            record["head"].weight.data_ptr() != record["weight_ptr"]
            or model.model.embed_tokens.weight.data_ptr() != record["weight_ptr"]
        ):
            raise RuntimeError("Mach greedy head tied weight changed")
        return original(hidden_states, input_batch, grammar_output)

    runner.sample = types.MethodType(checked, runner)
    record["installed"] = True


def inspect(worker):
    record = worker._mach_greedy_head
    return dict(
        installed=record["installed"],
        candidates=128,
        max_rows=64,
        stats={
            k: dict(v) if isinstance(v, dict) else v for k, v in record["stats"].items()
        },
    )
