# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import SpeculativeConfig
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.spec_decode.utils import unconditional_to_conditional_rates
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    get_num_sampled_and_rejected,
)
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)


@triton.jit
def _flatten_sampled_kernel(
    # [num_logits]
    flat_sampled_ptr,
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    for i in range(num_sampled):
        token_id = tl.load(sampled_ptr + req_idx * sampled_stride + i)
        tl.store(flat_sampled_ptr + start_idx + i, token_id)


@triton.jit(do_not_specialize=["max_spec_len"])
def _compact_rejection_sample_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    target_draft_probs_ptr,
    bonus_token_ids_ptr,
    recovered_token_ids_ptr,
    uniform_probs_ptr,
    max_spec_len,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            token_idx = start_idx + pos
            draft_token_id = tl.load(draft_token_ids_ptr + token_idx)
            target_prob = tl.load(target_draft_probs_ptr + token_idx)
            uniform_prob = tl.load(uniform_probs_ptr + token_idx)
            accepted = target_prob >= uniform_prob
            token_id = draft_token_id
            if not accepted:
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + token_idx)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


@triton.jit(do_not_specialize=["max_spec_len"])
def _compact_greedy_rejection_sample_kernel(
    output_token_ids_ptr,
    target_token_ids_ptr,
    draft_sampled_ptr,
    cu_num_logits_ptr,
    max_spec_len,
):
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            target_token_id = tl.load(target_token_ids_ptr + start_idx + pos).to(
                tl.int64
            )
            draft_token_id = tl.load(draft_sampled_ptr + start_idx + pos + 1).to(
                tl.int64
            )
            accepted = target_token_id == draft_token_id
            token_id = draft_token_id
            if not accepted:
                rejected = True
                token_id = target_token_id
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                token_id,
            )

    if not rejected:
        bonus_token_id = tl.load(target_token_ids_ptr + end_idx - 1).to(tl.int64)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


@triton.jit
def _prepare_compact_rejection_indices_kernel(
    cu_num_logits_ptr,
    target_indices_ptr,
    bonus_indices_ptr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    target_start_idx = start_idx - req_idx

    for pos in range(num_draft_tokens):
        tl.store(target_indices_ptr + target_start_idx + pos, start_idx + pos)
    tl.store(bonus_indices_ptr + req_idx, end_idx - 1)


class RejectionSampler:
    def __init__(
        self,
        sampler: Sampler,
        spec_config: SpeculativeConfig,
        device: torch.device,
    ):
        self.sampler = sampler
        self.num_speculative_steps = spec_config.num_speculative_tokens
        rejection_sample_method = spec_config.rejection_sample_method
        self.use_block_verification: bool = False
        self.synthetic_conditional_rates: torch.Tensor | None = None
        if rejection_sample_method == "synthetic":
            assert spec_config.synthetic_acceptance_rates is not None
            self.synthetic_conditional_rates = torch.tensor(
                unconditional_to_conditional_rates(
                    spec_config.synthetic_acceptance_rates
                ),
                dtype=torch.float32,
                device=device,
            )
        elif rejection_sample_method == "block":
            self.use_block_verification = True

    @staticmethod
    def _sample_from_candidate_logits(
        candidate_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        valid = torch.isfinite(candidate_logits)
        q = torch.empty_like(candidate_logits)
        q.exponential_()
        scores = (candidate_logits - q.log()).masked_fill(~valid, -float("inf"))
        sample_pos = scores.argmax(dim=-1, keepdim=True)
        return candidate_ids.gather(dim=-1, index=sample_pos).view(-1)

    def sample_from_topk_candidates(
        self,
        candidate_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        """Verify one-hot drafts using compact target top-k candidates."""
        assert candidate_logits.ndim == 2
        assert candidate_ids.shape == candidate_logits.shape
        assert candidate_logits.shape[0] == int(input_batch.cu_num_logits_np[-1])
        assert input_batch.num_draft_tokens_per_req is not None
        assert self.synthetic_conditional_rates is None
        assert not self.use_block_verification

        num_reqs = input_batch.num_reqs
        device = candidate_logits.device
        target_indices = torch.empty(
            (input_batch.num_draft_tokens,), dtype=torch.int64, device=device
        )
        bonus_indices = torch.empty((num_reqs,), dtype=torch.int64, device=device)
        _prepare_compact_rejection_indices_kernel[(num_reqs,)](
            input_batch.cu_num_logits,
            target_indices,
            bonus_indices,
        )

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        draft_token_ids = draft_sampled[target_indices + 1]

        bonus_token_ids = self._sample_from_candidate_logits(
            candidate_logits[bonus_indices], candidate_ids[bonus_indices]
        ).to(draft_token_ids.dtype)
        target_logits = candidate_logits[target_indices]
        target_ids = candidate_ids[target_indices]
        draft_ids = draft_token_ids.to(target_ids.dtype).unsqueeze(-1)

        valid = torch.isfinite(target_logits)
        draft_mask = valid & (target_ids == draft_ids)
        safe_logits = target_logits.masked_fill(~valid, -float("inf"))
        max_logits = safe_logits.max(dim=-1, keepdim=True).values
        weights = torch.exp(safe_logits - max_logits)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        denom = weights.sum(dim=-1)
        draft_weight = torch.where(draft_mask, weights, torch.zeros_like(weights)).sum(
            dim=-1
        )
        target_draft_probs = torch.where(
            denom > 0.0, draft_weight / denom, torch.zeros_like(denom)
        )

        recovered_mask = valid & ~draft_mask
        q = torch.empty_like(target_logits)
        q.exponential_()
        recovered_scores = (target_logits - q.log()).masked_fill(
            ~recovered_mask, -float("inf")
        )
        recovered_pos = recovered_scores.argmax(dim=-1, keepdim=True)
        recovered_token_ids = target_ids.gather(dim=-1, index=recovered_pos).view(-1)
        recovered_token_ids = torch.where(
            recovered_mask.any(dim=-1),
            recovered_token_ids,
            torch.zeros_like(recovered_token_ids),
        ).to(draft_token_ids.dtype)

        uniform_probs = torch.rand(
            (input_batch.num_draft_tokens,),
            dtype=torch.float64,
            device=device,
        )
        cu_num_draft_tokens = input_batch.cu_num_logits[1:] - torch.arange(
            1,
            num_reqs + 1,
            dtype=input_batch.cu_num_logits.dtype,
            device=device,
        )
        sampled = torch.full(
            (num_reqs, self.num_speculative_steps + 1),
            -1,
            dtype=torch.int64,
            device=device,
        )
        _compact_rejection_sample_kernel[(num_reqs,)](
            sampled,
            cu_num_draft_tokens,
            draft_token_ids,
            target_draft_probs,
            bonus_token_ids,
            recovered_token_ids,
            uniform_probs,
            self.num_speculative_steps,
        )
        num_sampled = (sampled != -1).sum(dim=-1, dtype=torch.int32)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def sample_from_greedy_tokens(
        self,
        target_token_ids: torch.Tensor,
        input_batch: InputBatch,
    ) -> SamplerOutput:
        """Verify greedy drafts from compact target top-1 token IDs."""
        assert target_token_ids.ndim == 1
        assert target_token_ids.shape[0] == int(input_batch.cu_num_logits_np[-1])
        assert input_batch.num_draft_tokens_per_req is not None
        assert self.synthetic_conditional_rates is None
        assert not self.use_block_verification

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        sampled = torch.full(
            (input_batch.num_reqs, self.num_speculative_steps + 1),
            -1,
            dtype=torch.int64,
            device=target_token_ids.device,
        )
        _compact_greedy_rejection_sample_kernel[(input_batch.num_reqs,)](
            sampled,
            target_token_ids,
            draft_sampled,
            input_batch.cu_num_logits,
            self.num_speculative_steps,
        )
        num_sampled = (sampled != -1).sum(dim=-1, dtype=torch.int32)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def _get_logprobs_tensors(
        self,
        input_batch: InputBatch,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        logits: torch.Tensor,
    ) -> LogprobsTensors | None:
        max_num_logprobs = self.sampler.sampling_states.max_num_logprobs(
            input_batch.idx_mapping_np
        )
        if max_num_logprobs == NO_LOGPROBS:
            return None

        num_reqs = input_batch.cu_num_logits.shape[0] - 1
        num_logits = logits.shape[0]
        flat_sampled = torch.zeros(
            num_logits, dtype=sampled.dtype, device=sampled.device
        )
        _flatten_sampled_kernel[(num_reqs,)](
            flat_sampled,
            sampled,
            sampled.stride(0),
            num_sampled,
            input_batch.cu_num_logits,
            num_warps=1,
        )
        expanded_logits = num_logits != input_batch.idx_mapping.shape[0]
        return compute_topk_logprobs(
            logits,
            max_num_logprobs,
            flat_sampled,
            input_batch.cu_num_logits_np.tolist() if expanded_logits else None,
        )

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.sampler.compute_nans else None

        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            draft_sampled,
            input_batch.expanded_local_pos,
        )
        sampled, num_sampled = rejection_sample(
            processed_logits,
            draft_logits,
            draft_sampled,
            input_batch.cu_num_logits,
            pos,
            input_batch.idx_mapping,
            input_batch.expanded_idx_mapping,
            input_batch.expanded_local_pos,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            self.num_speculative_steps,
            self.synthetic_conditional_rates,
            use_fp64=self.sampler.use_fp64_gumbel,
            use_block_verification=self.use_block_verification,
        )
        logprobs_tensors = self._get_logprobs_tensors(
            input_batch,
            sampled,
            num_sampled,
            processed_logits
            if self.sampler.logprobs_mode == "processed_logprobs"
            else logits,
        )

        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.sampler.req_states.prefill_len.gpu,
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )
