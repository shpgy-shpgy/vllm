# SPDX-License-Identifier: Apache-2.0
"""Opt-in, untimed NVFP4 candidate audit against the original BF16 head.

Adapted from vllm-mach greedy_head.install_audit. Sampling is not replaced:
shpgy retains its existing compact greedy AND non-greedy implementations.
"""

from types import MethodType

import torch

from vllm.model_executor.layers.argmax_triton import indexed_argmax_triton


def install(worker):
    runner = worker.model_runner
    cfg = worker.vllm_config
    if (
        not cfg.use_v2_model_runner
        or cfg.parallel_config.tensor_parallel_size != 1
        or cfg.speculative_config is not None
        or cfg.lora_config is not None
        or runner.batch_sharder is not None
    ):
        raise ValueError("Head audit requires ordinary V2 TP1 sampling")
    language_model = getattr(runner.model, "language_model", runner.model)
    head = language_model.lm_head
    state = getattr(head, "_hybrid_nvfp4_lm_head_state", None)
    if hasattr(worker, "_mach_greedy_head"):
        state = worker._mach_greedy_head["state"]
    if state is None or head.weight.dtype != torch.bfloat16:
        raise ValueError("Head audit requires an already prepared hybrid NVFP4 head")
    if hasattr(worker, "_mach_head_audit"):
        raise RuntimeError("Head audit already installed")
    counts = torch.zeros(3, device=worker.device, dtype=torch.int64)
    margin = torch.zeros((), device=worker.device, dtype=torch.float32)
    original = runner.sample

    def sample(self, hidden_states, input_batch, grammar_output):
        hidden = hidden_states[input_batch.logits_indices]
        vocab = head.shard_indices.org_vocab_end_index
        if hidden.shape[0] and state.can_use(
            hidden, bf16_weight=head.weight, active_vocab_size=vocab, top_k=1
        ):
            candidates = state.select_candidates(state.coarse_logits(hidden, None))
            refined = state.refine_logits(hidden, head.weight, candidates, None)
            _, proposed = indexed_argmax_triton(refined, candidates, index_offset=0)
            proposed = proposed.flatten().long()
            logits = torch.nn.functional.linear(hidden, head.weight)[:, :vocab]
            target = logits.argmax(-1)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            loss = (
                logits[rows, target].float() - logits[rows, proposed].float()
            ).clamp_min(0)
            counts[0].add_(hidden.shape[0])
            counts[1].add_((target != proposed).sum())
            counts[2].add_((loss > 0).sum())
            margin.copy_(torch.maximum(margin, loss.max()))
        return original(hidden_states, input_batch, grammar_output)

    runner.sample = MethodType(sample, runner)
    worker._mach_head_audit = counts, margin


def inspect(worker):
    counts, margin = worker._mach_head_audit
    rows, different, strict = counts.cpu().tolist()
    return dict(
        rows=rows,
        different_argmax=different,
        strict_score_losses=strict,
        max_logit_loss=float(margin),
        diagnostic_only=True,
    )
