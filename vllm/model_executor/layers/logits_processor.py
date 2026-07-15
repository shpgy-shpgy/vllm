# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.model_executor.layers.argmax_triton import (
        local_argmax_triton,
        reduce_global_argmax_triton,
    )
    from vllm.v1.sample.ops.topk_topp_triton import (
        pack_topk_pairs_triton,
        sample_from_compact_topk_pairs_triton,
        sample_full_vocab_from_shard_triton,
        select_compact_topk_pairs_triton,
    )
else:
    local_argmax_triton = None  # type: ignore[assignment]
    reduce_global_argmax_triton = None  # type: ignore[assignment]
    pack_topk_pairs_triton = None  # type: ignore[assignment]
    sample_from_compact_topk_pairs_triton = None  # type: ignore[assignment]
    sample_full_vocab_from_shard_triton = None  # type: ignore[assignment]
    select_compact_topk_pairs_triton = None  # type: ignore[assignment]


# --8<-- [start:logits_processor]
@PluggableLayer.register("logits_processor")
class LogitsProcessor(PluggableLayer):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    # --8<-- [end:logits_processor]

    def __init__(
        self,
        vocab_size: int,
        org_vocab_size: int | None = None,
        scale: float = 1.0,
        logits_as_input: bool = False,
        soft_cap: float | None = None,
    ) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_all_gather = current_platform.use_all_gather()
        self._compact_topk_sample_seeds: torch.Tensor | None = None
        self._full_vocab_sample_seeds: torch.Tensor | None = None

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale
        return logits

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """gather/all-gather the logits tensor across model parallel group."""
        if self.use_all_gather:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)
        else:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)

        # Gather logits for TP
        logits = self._gather_logits(logits)

        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    @staticmethod
    def _is_contiguous_org_shard(lm_head: VocabParallelEmbedding) -> bool:
        shard_indices = lm_head.shard_indices
        return getattr(shard_indices, "num_added_elements_padded", 0) == 0

    @staticmethod
    def _get_shard_token_ids(
        lm_head: VocabParallelEmbedding,
        device: torch.device,
    ) -> torch.Tensor:
        """Map local shard logits positions to global token ids.

        VocabParallelEmbedding stores local logits as
        [org shard][org padding][added shard][added padding]. Compact sampling
        communicates token ids directly, so it must use this physical layout
        instead of assuming local_idx + org_vocab_start for every position.
        """
        shard_indices = lm_head.shard_indices
        local_size = getattr(shard_indices, "num_elements_padded", None)
        if local_size is None:
            local_size = lm_head.num_embeddings_per_partition
        token_ids = torch.full((local_size,), -1, dtype=torch.int64, device=device)

        org_start = shard_indices.org_vocab_start_index
        org_elements = getattr(
            shard_indices,
            "num_org_elements",
            local_size - shard_indices.num_org_vocab_padding,
        )
        if org_elements > 0:
            token_ids[:org_elements] = torch.arange(
                org_start,
                org_start + org_elements,
                dtype=torch.int64,
                device=device,
            )

        added_elements = getattr(shard_indices, "num_added_elements", 0)
        if added_elements > 0:
            added_local_start = shard_indices.num_org_elements_padded
            added_start = shard_indices.added_vocab_start_index
            token_ids[added_local_start : added_local_start + added_elements] = (
                torch.arange(
                    added_start,
                    added_start + added_elements,
                    dtype=torch.int64,
                    device=device,
                )
            )
        return token_ids

    @staticmethod
    def _mask_invalid_shard_logits(
        logits: torch.Tensor,
        shard_token_ids: torch.Tensor | None,
        active_vocab_size: int,
    ) -> None:
        if shard_token_ids is None:
            if active_vocab_size < logits.shape[-1]:
                logits[..., active_vocab_size:] = -float("inf")
            return
        invalid = shard_token_ids < 0
        if invalid.any():
            logits.masked_fill_(invalid.unsqueeze(0), -float("inf"))

    @staticmethod
    def _local_indices_to_global(
        local_indices: torch.Tensor,
        *,
        vocab_start: int,
        shard_token_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if shard_token_ids is None:
            return local_indices + vocab_start
        return shard_token_ids.gather(0, local_indices.reshape(-1)).view_as(
            local_indices
        )

    @staticmethod
    def _reduce_global_argmax_pairs(
        gathered_pairs: torch.Tensor,
        tp_size: int,
    ) -> torch.Tensor:
        if reduce_global_argmax_triton is not None and gathered_pairs.is_cuda:
            return reduce_global_argmax_triton(gathered_pairs, tp_size=tp_size)

        pairs = gathered_pairs.view(gathered_pairs.shape[0], tp_size, 2)
        winner_ranks = pairs[..., 0].argmax(dim=-1, keepdim=True)
        return pairs[..., 1].to(torch.int64).gather(1, winner_ranks).view(-1)

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax, then only the (value, index) pairs
        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
        O(batch * vocab_size).
        """
        tp_size = get_tensor_model_parallel_world_size()

        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        shard_indices = lm_head.shard_indices
        num_pad = shard_indices.num_org_vocab_padding
        active_vocab_size = logits.shape[-1] - num_pad
        vocab_start = shard_indices.org_vocab_start_index
        shard_token_ids = None
        if not self._is_contiguous_org_shard(lm_head):
            shard_token_ids = self._get_shard_token_ids(lm_head, logits.device)
            active_vocab_size = logits.shape[-1]

        if (
            shard_token_ids is None
            and local_argmax_triton is not None
            and self.soft_cap is None
            and self.scale > 0.0
            and logits.is_cuda
            and logits.ndim == 2
        ):
            # Greedy argmax is invariant to positive global scaling. Avoid the
            # full-vocab gather and reduce local logits with a small custom path.
            local_max_vals, global_indices = local_argmax_triton(
                logits,
                vocab_start=vocab_start,
                active_vocab_size=active_vocab_size,
            )
        else:
            if self.soft_cap is not None:
                logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
            if self.scale != 1.0:
                logits = logits * self.scale

            self._mask_invalid_shard_logits(logits, shard_token_ids, active_vocab_size)
            local_max_vals, local_max_indices = logits.max(dim=-1)
            global_indices = self._local_indices_to_global(
                local_max_indices,
                vocab_start=vocab_start,
                shard_token_ids=shard_token_ids,
            )

        if tp_size == 1:
            return global_indices

        # All-gather (value, index) pairs, then reduce to global argmax.
        # Use float32 to avoid bf16 precision loss on large vocab indices.
        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        # [batch, 2] -> [batch, 2 * tp_size]
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        return self._reduce_global_argmax_pairs(gathered, tp_size)

    def _get_full_vocab_sample_seeds(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        seeds = self._full_vocab_sample_seeds
        if seeds is None or seeds.device != device or seeds.shape[0] < batch_size:
            seeds = torch.empty((batch_size,), dtype=torch.int64, device=device)
            self._full_vocab_sample_seeds = seeds
        seeds = seeds[:batch_size]
        seeds.random_()
        return seeds

    def _sample_local_full_tokens(
        self,
        logits: torch.Tensor,
        *,
        vocab_start: int,
        active_vocab_size: int,
        temperature: float,
        shard_token_ids: torch.Tensor | None = None,
        exclude_token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            HAS_TRITON
            and sample_full_vocab_from_shard_triton is not None
            and current_platform.is_cuda()
            and logits.is_cuda
            and logits.ndim == 2
        ):
            scale = self.scale
            if self.soft_cap is not None:
                logits = torch.tanh(logits.to(torch.float32) / self.soft_cap)
                logits = logits * self.soft_cap
            seeds = self._get_full_vocab_sample_seeds(logits.shape[0], logits.device)
            return sample_full_vocab_from_shard_triton(
                logits,
                vocab_start=vocab_start,
                active_vocab_size=active_vocab_size,
                seeds=seeds,
                shard_token_ids=shard_token_ids,
                exclude_token_ids=exclude_token_ids,
                scale=scale,
                temperature=temperature,
            )

        logits = logits.to(torch.float32)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale
        if temperature != 1.0:
            logits = logits / temperature
        self._mask_invalid_shard_logits(logits, shard_token_ids, active_vocab_size)
        if shard_token_ids is None:
            logits = logits[..., :active_vocab_size]

        if exclude_token_ids is not None:
            logits = logits.clone()
            if shard_token_ids is None:
                local_exclude = exclude_token_ids.to(torch.int64) - vocab_start
                in_shard = (local_exclude >= 0) & (local_exclude < active_vocab_size)
                if in_shard.any():
                    rows = torch.arange(logits.shape[0], device=logits.device)
                    safe_exclude = local_exclude.clamp(0, active_vocab_size - 1)
                    logits[rows[in_shard], safe_exclude[in_shard]] = -float("inf")
            else:
                exclude_mask = shard_token_ids.unsqueeze(0) == exclude_token_ids.to(
                    torch.int64
                ).unsqueeze(1)
                logits.masked_fill_(exclude_mask, -float("inf"))

        q = torch.empty_like(logits)
        q.exponential_()
        scores = logits - q.log()
        local_scores, local_indices = scores.max(dim=-1)
        global_indices = self._local_indices_to_global(
            local_indices,
            vocab_start=vocab_start,
            shard_token_ids=shard_token_ids,
        )
        return local_scores.to(torch.float64), global_indices.to(torch.int32)

    def sample_full_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        temperature: float,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel random sampling without full-vocab logits gather.

        Uses the Gumbel-max trick locally on each vocab shard, then communicates
        only each row's winning ``(score, token_id)`` pair across TP ranks.
        """
        if temperature <= 0.0:
            raise ValueError(
                f"sample_full_tokens requires positive temperature; got {temperature}"
            )

        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        shard_indices = lm_head.shard_indices
        vocab_start = shard_indices.org_vocab_start_index
        active_vocab_size = logits.shape[-1] - shard_indices.num_org_vocab_padding
        shard_token_ids = None
        if not self._is_contiguous_org_shard(lm_head):
            shard_token_ids = self._get_shard_token_ids(lm_head, logits.device)
            active_vocab_size = logits.shape[-1]
        local_scores, global_indices = self._sample_local_full_tokens(
            logits,
            vocab_start=vocab_start,
            active_vocab_size=active_vocab_size,
            temperature=temperature,
            shard_token_ids=shard_token_ids,
        )

        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return global_indices.to(torch.int64)

        local_pairs = torch.stack(
            [local_scores, global_indices.to(local_scores.dtype)], dim=-1
        )
        tp_group = get_tp_group()
        gathered_pairs = tensor_model_parallel_gather(local_pairs, dst=0, dim=-1)
        if tp_group.rank_in_group == 0:
            assert gathered_pairs is not None
            sampled = self._reduce_global_argmax_pairs(gathered_pairs, tp_size)
        else:
            sampled = torch.empty(
                (logits.shape[0],), dtype=torch.int32, device=logits.device
            )
        tp_group.broadcast(sampled, src=0)
        return sampled.to(torch.int64)

    def get_topk_candidates(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        top_k: int,
        top_p: float,
        temperature: float,
        presence_penalties: torch.Tensor | None = None,
        output_token_ids: torch.Tensor | None = None,
        embedding_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return global compact top-k/top-p candidates without full all-gather.

        The returned tensors have shape ``[batch, top_k]``. Logits masked by
        top-p are set to ``-inf``; token ids are global vocabulary ids.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if temperature <= 0.0:
            raise ValueError(
                f"get_topk_candidates requires positive temperature; got {temperature}"
            )
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        logits = logits.to(torch.float32)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        shard_indices = lm_head.shard_indices
        vocab_start = shard_indices.org_vocab_start_index
        active_vocab_size = logits.shape[-1] - shard_indices.num_org_vocab_padding
        shard_token_ids = None
        if not self._is_contiguous_org_shard(lm_head):
            shard_token_ids = self._get_shard_token_ids(lm_head, logits.device)
            active_vocab_size = logits.shape[-1]
        if presence_penalties is not None:
            assert output_token_ids is not None
            self._apply_sharded_presence_penalty(
                logits,
                presence_penalties,
                output_token_ids,
                shard_indices=shard_indices,
            )
        if temperature != 1.0:
            logits = logits / temperature
        self._mask_invalid_shard_logits(logits, shard_token_ids, active_vocab_size)

        local_top_k = min(top_k, active_vocab_size)
        tp_size = get_tensor_model_parallel_world_size()
        local_vals, local_indices = torch.topk(logits, local_top_k, dim=-1)
        local_pairs = self._pack_topk_pairs(
            local_vals,
            local_indices,
            vocab_start=vocab_start,
            shard_token_ids=shard_token_ids,
        )
        if tp_size == 1:
            gathered_pairs = local_pairs.view(logits.shape[0], local_top_k, 2)
        else:
            gathered_pairs = tensor_model_parallel_all_gather(local_pairs, dim=-1)
            gathered_pairs = gathered_pairs.view(
                logits.shape[0], tp_size * local_top_k, 2
            )

        return self._select_from_compact_topk_pairs(
            gathered_pairs, top_k=top_k, top_p=top_p
        )

    def _pack_topk_pairs(
        self,
        local_vals: torch.Tensor,
        local_indices: torch.Tensor,
        *,
        vocab_start: int,
        shard_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            shard_token_ids is None
            and HAS_TRITON
            and pack_topk_pairs_triton is not None
            and current_platform.is_cuda()
            and local_vals.is_cuda
            and local_vals.dtype == torch.float32
            and local_indices.dtype == torch.int64
            and 0 < local_vals.shape[-1] <= 1024
        ):
            return pack_topk_pairs_triton(
                local_vals, local_indices, vocab_start=vocab_start
            )

        global_indices = self._local_indices_to_global(
            local_indices,
            vocab_start=vocab_start,
            shard_token_ids=shard_token_ids,
        )
        return torch.stack(
            [local_vals, global_indices.to(torch.float32)], dim=-1
        ).flatten(start_dim=-2)

    def _select_from_compact_topk_pairs(
        self, gathered_pairs: torch.Tensor, top_k: int, top_p: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            HAS_TRITON
            and select_compact_topk_pairs_triton is not None
            and current_platform.is_cuda()
            and gathered_pairs.is_cuda
            and gathered_pairs.dtype == torch.float32
            and top_k <= 64
            and top_k <= gathered_pairs.shape[1] <= 1024
        ):
            return select_compact_topk_pairs_triton(
                gathered_pairs, top_k=top_k, top_p=top_p
            )

        candidate_vals = gathered_pairs[..., 0]
        candidate_ids = gathered_pairs[..., 1].to(torch.int64)
        top_vals, top_pos = torch.topk(candidate_vals, top_k, dim=-1)
        top_ids = candidate_ids.gather(dim=-1, index=top_pos)

        if top_p < 1.0:
            probs = top_vals.softmax(dim=-1, dtype=torch.float32)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            remove_mask = cumulative_probs - probs > top_p
            top_vals = top_vals.masked_fill(remove_mask, -float("inf"))

        return top_vals, top_ids

    def _get_compact_topk_sample_seeds(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        seeds = self._compact_topk_sample_seeds
        if seeds is None or seeds.device != device or seeds.shape[0] < batch_size:
            seeds = torch.empty((batch_size,), dtype=torch.int64, device=device)
            self._compact_topk_sample_seeds = seeds
        seeds = seeds[:batch_size]
        seeds.random_()
        return seeds

    def _sample_from_compact_topk_pairs(
        self, gathered_pairs: torch.Tensor, top_k: int, top_p: float
    ) -> torch.Tensor:
        if (
            HAS_TRITON
            and sample_from_compact_topk_pairs_triton is not None
            and current_platform.is_cuda()
            and gathered_pairs.is_cuda
            and gathered_pairs.dtype == torch.float32
            and top_k <= 64
            and top_k <= gathered_pairs.shape[1] <= 1024
        ):
            seeds = self._get_compact_topk_sample_seeds(
                gathered_pairs.shape[0], gathered_pairs.device
            )
            return sample_from_compact_topk_pairs_triton(
                gathered_pairs, top_k=top_k, top_p=top_p, seeds=seeds
            )

        candidate_vals = gathered_pairs[..., 0]
        candidate_ids = gathered_pairs[..., 1].to(torch.int64)
        top_vals, top_pos = torch.topk(candidate_vals, top_k, dim=-1)
        top_ids = candidate_ids.gather(dim=-1, index=top_pos)

        if top_p < 1.0:
            probs = top_vals.softmax(dim=-1, dtype=torch.float32)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            remove_mask = cumulative_probs - probs > top_p
            top_vals = top_vals.masked_fill(remove_mask, -float("inf"))

        probs = top_vals.softmax(dim=-1, dtype=torch.float32)
        q = torch.empty_like(probs)
        q.exponential_()
        sampled_pos = probs.div_(q).argmax(dim=-1, keepdim=True)
        return top_ids.gather(dim=-1, index=sampled_pos).view(-1)

    def sample_topk_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        top_k: int,
        top_p: float,
        temperature: float,
        presence_penalties: torch.Tensor | None = None,
        output_token_ids: torch.Tensor | None = None,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel random sampling for small uniform top-k batches.

        For positive temperature, global top-k can be found from each rank's
        local top-k. We communicate only those candidate (logit, token_id)
        pairs, then apply top-p and random sampling on the compact candidate
        set. This avoids all-gathering full-vocab logits before sampling.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if temperature <= 0.0:
            raise ValueError(
                f"sample_topk_tokens requires positive temperature; got {temperature}"
            )
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        logits = logits.to(torch.float32)
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale

        shard_indices = lm_head.shard_indices
        vocab_start = shard_indices.org_vocab_start_index
        active_vocab_size = logits.shape[-1] - shard_indices.num_org_vocab_padding
        shard_token_ids = None
        if not self._is_contiguous_org_shard(lm_head):
            shard_token_ids = self._get_shard_token_ids(lm_head, logits.device)
            active_vocab_size = logits.shape[-1]
        if presence_penalties is not None:
            assert output_token_ids is not None
            self._apply_sharded_presence_penalty(
                logits,
                presence_penalties,
                output_token_ids,
                shard_indices=shard_indices,
            )
        if temperature != 1.0:
            logits = logits / temperature

        self._mask_invalid_shard_logits(logits, shard_token_ids, active_vocab_size)

        local_top_k = min(top_k, active_vocab_size)
        tp_size = get_tensor_model_parallel_world_size()
        local_vals, local_indices = torch.topk(logits, local_top_k, dim=-1)
        local_pairs = self._pack_topk_pairs(
            local_vals,
            local_indices,
            vocab_start=vocab_start,
            shard_token_ids=shard_token_ids,
        )
        if tp_size == 1:
            gathered_pairs = local_pairs.view(logits.shape[0], local_top_k, 2)
            return self._sample_from_compact_topk_pairs(
                gathered_pairs, top_k=top_k, top_p=top_p
            )

        tp_group = get_tp_group()
        gathered_pairs = tensor_model_parallel_gather(local_pairs, dst=0, dim=-1)
        if tp_group.rank_in_group == 0:
            assert gathered_pairs is not None
            gathered_pairs = gathered_pairs.view(
                logits.shape[0], tp_size * local_top_k, 2
            )
            sampled = self._sample_from_compact_topk_pairs(
                gathered_pairs, top_k=top_k, top_p=top_p
            )
        else:
            total_top_k = tp_size * local_top_k
            if (
                HAS_TRITON
                and sample_from_compact_topk_pairs_triton is not None
                and current_platform.is_cuda()
                and local_pairs.is_cuda
                and local_pairs.dtype == torch.float32
                and top_k <= 64
                and top_k <= total_top_k <= 1024
            ):
                self._get_compact_topk_sample_seeds(logits.shape[0], logits.device)
            else:
                q = torch.empty(
                    (logits.shape[0], top_k),
                    dtype=torch.float32,
                    device=logits.device,
                )
                q.exponential_()
            sampled = torch.empty(
                (logits.shape[0],), dtype=torch.int64, device=logits.device
            )

        tp_group.broadcast(sampled, src=0)
        return sampled

    @staticmethod
    def _apply_sharded_presence_penalty(
        logits: torch.Tensor,
        presence_penalties: torch.Tensor,
        output_token_ids: torch.Tensor,
        *,
        shard_indices,
    ) -> None:
        """Apply presence-only penalty to a vocab-parallel logits shard."""
        if output_token_ids.numel() == 0:
            return

        num_org_padding = shard_indices.num_org_vocab_padding
        org_start = shard_indices.org_vocab_start_index
        org_end = getattr(
            shard_indices,
            "org_vocab_end_index",
            org_start + logits.shape[-1] - num_org_padding,
        )
        org_local = output_token_ids - org_start
        org_valid = (output_token_ids >= org_start) & (output_token_ids < org_end)

        added_start = getattr(shard_indices, "added_vocab_start_index", org_end)
        added_end = getattr(shard_indices, "added_vocab_end_index", added_start)
        added_local_start = getattr(
            shard_indices,
            "num_org_elements_padded",
            org_end - org_start + num_org_padding,
        )
        added_local = output_token_ids - added_start + added_local_start
        added_valid = (output_token_ids >= added_start) & (output_token_ids < added_end)

        pad_id = logits.shape[-1]
        local_token_ids = torch.where(
            org_valid,
            org_local,
            torch.where(added_valid, added_local, pad_id),
        )

        penalty_values = presence_penalties.to(logits.dtype).unsqueeze(1)
        penalty_values = penalty_values.expand_as(output_token_ids)

        penalty_mask = torch.zeros(
            (logits.shape[0], logits.shape[1] + 1),
            dtype=logits.dtype,
            device=logits.device,
        )
        penalty_mask.scatter_(1, local_token_ids, penalty_values)
        logits.sub_(penalty_mask[:, : logits.shape[1]])

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
