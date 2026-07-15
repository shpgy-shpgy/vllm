# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A layer that compute logits from hidden_stats."""

from collections.abc import Callable
from functools import cache

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.utils.flashinfer import has_flashinfer

if HAS_TRITON:
    from vllm.model_executor.layers.argmax_triton import (
        indexed_argmax_triton,
        local_argmax_triton,
        reduce_global_argmax_triton,
    )
    from vllm.model_executor.layers.presence_penalty_triton import (
        apply_presence_penalty_from_counts,
        apply_sparse_presence_penalty,
    )
    from vllm.v1.sample.ops.topk_topp_triton import (
        pack_topk_pairs_triton,
        sample_from_compact_topk_pairs_triton,
        sample_full_vocab_from_shard_triton,
        select_compact_topk_pairs_triton,
    )
else:
    indexed_argmax_triton = None  # type: ignore[assignment]
    local_argmax_triton = None  # type: ignore[assignment]
    reduce_global_argmax_triton = None  # type: ignore[assignment]
    pack_topk_pairs_triton = None  # type: ignore[assignment]
    sample_from_compact_topk_pairs_triton = None  # type: ignore[assignment]
    sample_full_vocab_from_shard_triton = None  # type: ignore[assignment]
    select_compact_topk_pairs_triton = None  # type: ignore[assignment]
    apply_presence_penalty_from_counts = None  # type: ignore[assignment]
    apply_sparse_presence_penalty = None  # type: ignore[assignment]

logger = init_logger(__name__)


@cache
def _flashinfer_topk() -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:
    """FlashInfer's radix top-k, or None for torch.topk.

    The top-k spans the vocabulary, where the radix kernel is about twice
    torch.topk.
    """
    if not current_platform.is_cuda():
        return None
    if not has_flashinfer():
        logger.info_once(
            "flashinfer is unavailable; vocab-parallel top-k uses torch.topk, "
            "at roughly half the speed."
        )
        return None
    from flashinfer import top_k

    return top_k


def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    impl = _flashinfer_topk()
    if impl is None or not scores.is_cuda:
        return torch.topk(scores, k, dim=-1)
    return impl(scores, k, sorted=True, deterministic=True)


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
        # Dtype of the lm_head projection. Defaults to the model dtype; an
        # fp32 head (via `--hf-overrides '{"head_dtype": "float32"}'`) is
        # required for RL training-inference consistency.
        current_config = get_current_vllm_config_or_none()
        model_config = (
            current_config.model_config if current_config is not None else None
        )
        self.head_dtype = model_config.head_dtype if model_config is not None else None
        # Hybrid lm-heads are optimized for decode.  The V2 model runner can
        # temporarily disable the compact path while a batch still contains
        # prompt-prefill requests; this avoids paying the FP4 quantize/GEMM
        # setup cost on the TTFT-critical prefill tail.  Keep the default
        # enabled so existing callers and eager execution are unchanged.
        self.hybrid_lm_head_enabled = True
        self.hybrid_lm_head_row_mask: torch.Tensor | None = None
        self._compact_topk_sample_seeds: torch.Tensor | None = None
        self._full_vocab_sample_seeds: torch.Tensor | None = None

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        if self.logits_as_input:
            logits = hidden_states
        else:
            # Get the logits for the next tokens.
            logits = self._get_logits(
                hidden_states, lm_head, embedding_bias, skip_gather
            )
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

    def _apply_head(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project hidden states through the lm_head, honoring head_dtype."""
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
            return lm_head.quant_method.apply(
                lm_head, hidden_states, bias=embedding_bias
            )

        # A quant config that excludes lm_head hands out UnquantizedLinearMethod
        # rather than UnquantizedEmbeddingMethod, so accept both: either way the
        # weight is plain and `lm_head.weight` can be cast directly.
        if not isinstance(
            lm_head.quant_method, (UnquantizedEmbeddingMethod, UnquantizedLinearMethod)
        ):
            raise ValueError(
                "A head_dtype different from the model dtype is only "
                "supported for an unquantized lm_head."
            )
        if (
            self.head_dtype == torch.float32
            and (current_platform.is_cuda() or current_platform.is_rocm())
            and hidden_states.is_cuda
        ):
            # Accumulate the projection directly into fp32. This avoids
            # materializing an fp32 copy of the lm_head weight on every step,
            # unlike casting both operands. `torch.mm(out_dtype=...)` only
            # supports fp32 output for fp16/bf16 inputs, and is only
            # implemented for CUDA and ROCm (the latter via the non-Lt GEMM
            # path); other platforms fall back to the cast path below.
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.mm(flat, lm_head.weight.t(), out_dtype=self.head_dtype)
            if embedding_bias is not None:
                logits = logits + embedding_bias.to(self.head_dtype)
            return logits.reshape(*hidden_states.shape[:-1], -1)
        return F.linear(
            hidden_states.to(self.head_dtype),
            lm_head.weight.to(self.head_dtype),
            embedding_bias.to(self.head_dtype) if embedding_bias is not None else None,
        )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: torch.Tensor | None,
        skip_gather: bool = False,
    ) -> torch.Tensor | None:
        # Get the logits for the next tokens.
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        if skip_gather:
            return logits

        # Gather logits for TP
        if lm_head.tp_size > 1:
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
    def _get_hybrid_lm_head_state(lm_head: VocabParallelEmbedding):
        """Return the active NVFP4/MXFP4/MXFP8 coarse-search state."""
        state = getattr(lm_head, "_hybrid_nvfp4_lm_head_state", None)
        if state is not None:
            return state
        state = getattr(lm_head, "_hybrid_mxfp4_lm_head_state", None)
        if state is not None:
            return state
        return getattr(lm_head, "_hybrid_mxfp8_lm_head_state", None)

    @staticmethod
    def _select_hybrid_candidates(hybrid_state, coarse_logits, top_k: int):
        """Select candidates while keeping compatibility with old test states."""
        if hasattr(hybrid_state, "candidate_count_for_topk"):
            return hybrid_state.select_candidates(coarse_logits, top_k=top_k)
        return hybrid_state.select_candidates(coarse_logits)

    def _get_hybrid_lm_head_row_mask(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        """Return a valid per-row compact-path mask, if one is installed."""
        mask = self.hybrid_lm_head_row_mask
        if mask is None:
            return None
        if mask.ndim != 1 or mask.shape[0] != hidden_states.shape[0]:
            # The mask is tied to one target sampling call.  Prompt-logprob
            # and draft calls can have a different number of rows, in which
            # case silently using the normal all-row policy is safest.
            return None
        if mask.device != hidden_states.device:
            mask = mask.to(hidden_states.device, non_blocking=True)
        return mask.to(dtype=torch.bool)

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
            local_size = getattr(lm_head, "num_embeddings_per_partition", None)
        if local_size is None:
            local_size = lm_head.weight.shape[0]
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

    def reduce_local_argmax(
        self,
        local_max_vals: torch.Tensor,
        global_indices: torch.Tensor,
        tp_size: int | None = None,
    ) -> torch.Tensor:
        """Reduce TP-local winners using a pair-only all-gather."""
        if tp_size is None:
            tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return global_indices

        local_pair = torch.stack(
            [local_max_vals.float(), global_indices.float()], dim=-1
        )
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        return self._reduce_global_argmax_pairs(gathered, tp_size)

    def get_top_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        row_mask = self._get_hybrid_lm_head_row_mask(hidden_states)
        if row_mask is not None and not bool(row_mask.all()):
            # Mixed prefill/decode batches are ordered by request, while the
            # compact lm-head receives one row per sampled logit.  Compute
            # only decode rows with the hybrid path and preserve the full
            # BF16 path for prompt-tail rows.
            result = torch.empty(
                hidden_states.shape[0], dtype=torch.int32, device=hidden_states.device
            )
            enabled_rows = torch.nonzero(row_mask, as_tuple=True)[0]
            if enabled_rows.numel() > 0:
                result[enabled_rows] = self._get_top_tokens_single(
                    lm_head,
                    hidden_states[enabled_rows].contiguous(),
                    embedding_bias=embedding_bias,
                    _hybrid_enabled=True,
                )
            disabled_rows = torch.nonzero(~row_mask, as_tuple=True)[0]
            if disabled_rows.numel() > 0:
                result[disabled_rows] = self._get_top_tokens_single(
                    lm_head,
                    hidden_states[disabled_rows].contiguous(),
                    embedding_bias=embedding_bias,
                    _hybrid_enabled=False,
                )
            return result
        return self._get_top_tokens_single(
            lm_head,
            hidden_states,
            embedding_bias=embedding_bias,
        )

    def _get_top_tokens_single(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        embedding_bias: torch.Tensor | None = None,
        *,
        _hybrid_enabled: bool | None = None,
    ) -> torch.Tensor:
        """Vocab-parallel argmax without all-gathering full logits.

        Each TP rank computes local argmax, then only the (value, index) pairs
        are gathered and reduced. Communication: O(batch * 2 * tp_size) vs
        O(batch * vocab_size).
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local argmax reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )
        shard_indices = lm_head.shard_indices
        num_pad = shard_indices.num_org_vocab_padding
        local_vocab_size = getattr(shard_indices, "num_elements_padded", None)
        if local_vocab_size is None:
            local_vocab_size = getattr(lm_head, "num_embeddings_per_partition", None)
        if local_vocab_size is None:
            local_vocab_size = lm_head.weight.shape[0]
        active_vocab_size = local_vocab_size - num_pad
        vocab_start = shard_indices.org_vocab_start_index
        shard_token_ids = None
        if not self._is_contiguous_org_shard(lm_head):
            shard_token_ids = self._get_shard_token_ids(lm_head, hidden_states.device)
            active_vocab_size = local_vocab_size

        hybrid_state = self._get_hybrid_lm_head_state(lm_head)
        if (
            hybrid_state is not None
            and shard_token_ids is None
            and self.soft_cap is None
            and self.scale > 0.0
            and self.hybrid_lm_head_enabled
            and _hybrid_enabled is not False
            and hybrid_state.can_use(
                hidden_states,
                bf16_weight=lm_head.weight,
                active_vocab_size=active_vocab_size,
                top_k=1,
            )
        ):
            coarse_logits = hybrid_state.coarse_logits(
                hidden_states,
                embedding_bias,
            )
            self._mask_invalid_shard_logits(
                coarse_logits,
                None,
                active_vocab_size,
            )
            candidate_indices = self._select_hybrid_candidates(
                hybrid_state, coarse_logits, top_k=1
            )
            exact_logits = hybrid_state.refine_logits(
                hidden_states,
                lm_head.weight,
                candidate_indices,
                embedding_bias,
            )
            if (
                indexed_argmax_triton is not None
                and exact_logits.is_cuda
                and 0 < exact_logits.shape[-1] <= 1024
            ):
                local_max_vals, global_indices = indexed_argmax_triton(
                    exact_logits,
                    candidate_indices,
                    index_offset=vocab_start,
                )
            else:
                local_max_vals = exact_logits.max(dim=-1).values
                local_max_indices = (
                    candidate_indices.masked_fill(
                        exact_logits != local_max_vals.unsqueeze(-1),
                        lm_head.weight.shape[0],
                    )
                    .min(dim=-1)
                    .values
                )
                global_indices = local_max_indices.to(torch.int32) + vocab_start

            return self.reduce_local_argmax(
                local_max_vals, global_indices, tp_size=lm_head.tp_size
            )

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
        active_vocab_size = logits.shape[-1] - num_pad
        if shard_token_ids is not None:
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

        return self.reduce_local_argmax(
            local_max_vals, global_indices, tp_size=lm_head.tp_size
        )

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

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)
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

        tp_size = lm_head.tp_size
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
        output_token_counts: torch.Tensor | None = None,
        presence_request_indices: torch.Tensor | None = None,
        output_unique_token_ids: torch.Tensor | None = None,
        num_output_unique_tokens: torch.Tensor | None = None,
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
        local_vals, local_indices, vocab_start, shard_token_ids = self._get_local_topk(
            lm_head,
            hidden_states,
            top_k=top_k,
            temperature=temperature,
            presence_penalties=presence_penalties,
            output_token_ids=output_token_ids,
            output_token_counts=output_token_counts,
            presence_request_indices=presence_request_indices,
            output_unique_token_ids=output_unique_token_ids,
            num_output_unique_tokens=num_output_unique_tokens,
            embedding_bias=embedding_bias,
        )
        local_top_k = local_vals.shape[-1]
        batch_size = hidden_states.shape[0]
        tp_size = lm_head.tp_size
        local_pairs = self._pack_topk_pairs(
            local_vals,
            local_indices,
            vocab_start=vocab_start,
            shard_token_ids=shard_token_ids,
        )
        if tp_size == 1:
            gathered_pairs = local_pairs.view(batch_size, local_top_k, 2)
        else:
            gathered_pairs = tensor_model_parallel_all_gather(local_pairs, dim=-1)
            gathered_pairs = gathered_pairs.view(batch_size, tp_size * local_top_k, 2)

        return self._select_from_compact_topk_pairs(
            gathered_pairs, top_k=top_k, top_p=top_p
        )

    def _get_local_topk(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        *,
        top_k: int,
        temperature: float,
        presence_penalties: torch.Tensor | None = None,
        output_token_ids: torch.Tensor | None = None,
        output_token_counts: torch.Tensor | None = None,
        presence_request_indices: torch.Tensor | None = None,
        output_unique_token_ids: torch.Tensor | None = None,
        num_output_unique_tokens: torch.Tensor | None = None,
        embedding_bias: torch.Tensor | None = None,
        _hybrid_enabled: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor | None]:
        """Compute local top-k, using hybrid MXFP4/MXFP8 when eligible."""
        shard_indices = lm_head.shard_indices
        vocab_start = shard_indices.org_vocab_start_index
        local_vocab_size = getattr(shard_indices, "num_elements_padded", None)
        if local_vocab_size is None:
            local_vocab_size = getattr(lm_head, "num_embeddings_per_partition", None)
        if local_vocab_size is None:
            local_vocab_size = lm_head.weight.shape[0]
        active_vocab_size = local_vocab_size - shard_indices.num_org_vocab_padding
        shard_token_ids = None
        if not self._is_contiguous_org_shard(lm_head):
            shard_token_ids = self._get_shard_token_ids(lm_head, hidden_states.device)
            active_vocab_size = local_vocab_size

        local_top_k = min(top_k, active_vocab_size)

        row_mask = (
            None
            if _hybrid_enabled is not None
            else self._get_hybrid_lm_head_row_mask(hidden_states)
        )
        hybrid_state = self._get_hybrid_lm_head_state(lm_head)
        can_split_rows = (
            row_mask is not None
            and not bool(row_mask.all())
            and hybrid_state is not None
            and self.hybrid_lm_head_enabled
            and shard_token_ids is None
            and self.soft_cap is None
            and self.scale == 1.0
            and presence_penalties is None
            and output_token_ids is None
            and output_token_counts is None
            and presence_request_indices is None
            and output_unique_token_ids is None
            and num_output_unique_tokens is None
            and embedding_bias is None
        )
        if can_split_rows:
            # Presence penalties and non-monotonic processors need request
            # metadata that is not row-aligned, so keep those calls on one
            # full-vocab path.  The common speculative top-k path has no such
            # processors and can safely split decode rows from prompt tails.
            enabled_rows = torch.nonzero(row_mask, as_tuple=True)[0]
            disabled_rows = torch.nonzero(~row_mask, as_tuple=True)[0]
            local_vals = torch.empty(
                (hidden_states.shape[0], local_top_k),
                dtype=torch.float32,
                device=hidden_states.device,
            )
            local_indices = torch.empty(
                (hidden_states.shape[0], local_top_k),
                dtype=torch.int64,
                device=hidden_states.device,
            )
            if enabled_rows.numel() > 0:
                enabled_vals, enabled_indices, _, _ = self._get_local_topk(
                    lm_head,
                    hidden_states[enabled_rows].contiguous(),
                    top_k=top_k,
                    temperature=temperature,
                    embedding_bias=embedding_bias,
                    _hybrid_enabled=True,
                )
                local_vals[enabled_rows] = enabled_vals
                local_indices[enabled_rows] = enabled_indices
            disabled_vals, disabled_indices, _, _ = self._get_local_topk(
                lm_head,
                hidden_states[disabled_rows].contiguous(),
                top_k=top_k,
                temperature=temperature,
                embedding_bias=embedding_bias,
                _hybrid_enabled=False,
            )
            local_vals[disabled_rows] = disabled_vals
            local_indices[disabled_rows] = disabled_indices
            return local_vals, local_indices, vocab_start, None

        hybrid_state = self._get_hybrid_lm_head_state(lm_head)
        if (
            hybrid_state is not None
            and shard_token_ids is None
            and self.hybrid_lm_head_enabled
            and _hybrid_enabled is not False
            and hybrid_state.can_use(
                hidden_states,
                bf16_weight=lm_head.weight,
                active_vocab_size=active_vocab_size,
                top_k=local_top_k,
            )
        ):
            coarse_logits = hybrid_state.coarse_logits(
                hidden_states,
                embedding_bias,
            )

            # With no additive processor, soft-cap, positive scale, and
            # temperature are monotonic and cannot change the candidate set.
            # Keep BF16 here so the selector needs neither an [M, N] FP32 copy
            # nor twice as many radix rounds.
            process_coarse_logits = self.scale <= 0.0 or (
                presence_penalties is not None
                and (self.soft_cap is not None or self.scale != 1.0)
            )
            if process_coarse_logits:
                coarse_logits = coarse_logits.to(torch.float32)
                if self.soft_cap is not None:
                    coarse_logits = (
                        torch.tanh(coarse_logits / self.soft_cap) * self.soft_cap
                    )
                if self.scale != 1.0:
                    coarse_logits = coarse_logits * self.scale

            penalty_mask = None
            if presence_penalties is not None:
                if output_token_counts is not None:
                    assert presence_request_indices is not None
                    sparse_applied = (
                        output_unique_token_ids is not None
                        and num_output_unique_tokens is not None
                        and self._apply_sharded_sparse_presence_penalty(
                            coarse_logits,
                            presence_penalties,
                            output_unique_token_ids,
                            num_output_unique_tokens,
                            presence_request_indices,
                            shard_indices=shard_indices,
                        )
                    )
                    if not sparse_applied:
                        self._apply_sharded_presence_penalty_from_counts(
                            coarse_logits,
                            presence_penalties,
                            output_token_counts,
                            presence_request_indices,
                            shard_indices=shard_indices,
                        )
                else:
                    assert output_token_ids is not None
                    penalty_mask = self._get_sharded_presence_penalty_mask(
                        coarse_logits,
                        presence_penalties,
                        output_token_ids,
                        shard_indices=shard_indices,
                    )
                    if penalty_mask is not None:
                        coarse_logits.sub_(penalty_mask)
            self._mask_invalid_shard_logits(
                coarse_logits,
                None,
                active_vocab_size,
            )
            candidate_indices = self._select_hybrid_candidates(
                hybrid_state, coarse_logits, top_k=local_top_k
            )

            exact_logits = hybrid_state.refine_logits(
                hidden_states,
                lm_head.weight,
                candidate_indices,
                embedding_bias,
            ).to(torch.float32)
            if self.soft_cap is not None:
                exact_logits = torch.tanh(exact_logits / self.soft_cap) * self.soft_cap
            if self.scale != 1.0:
                exact_logits = exact_logits * self.scale
            if penalty_mask is not None:
                exact_logits.sub_(penalty_mask.gather(1, candidate_indices))
            elif presence_penalties is not None and output_token_counts is not None:
                assert presence_request_indices is not None
                self._apply_sharded_presence_penalty_from_counts(
                    exact_logits,
                    presence_penalties,
                    output_token_counts,
                    presence_request_indices,
                    shard_indices=shard_indices,
                    local_token_ids=candidate_indices,
                )
            if temperature != 1.0:
                exact_logits = exact_logits / temperature

            local_vals, candidate_positions = torch.topk(
                exact_logits,
                local_top_k,
                dim=-1,
            )
            local_indices = candidate_indices.gather(1, candidate_positions)

            return local_vals, local_indices, vocab_start, None

        logits = lm_head.quant_method.apply(
            lm_head,
            hidden_states,
            bias=embedding_bias,
        ).to(torch.float32)
        active_vocab_size = logits.shape[-1] - shard_indices.num_org_vocab_padding
        if shard_token_ids is not None:
            active_vocab_size = logits.shape[-1]
        if self.soft_cap is not None:
            logits = torch.tanh(logits / self.soft_cap) * self.soft_cap
        if self.scale != 1.0:
            logits = logits * self.scale
        if presence_penalties is not None:
            if output_token_counts is not None:
                assert presence_request_indices is not None
                sparse_applied = (
                    output_unique_token_ids is not None
                    and num_output_unique_tokens is not None
                    and self._apply_sharded_sparse_presence_penalty(
                        logits,
                        presence_penalties,
                        output_unique_token_ids,
                        num_output_unique_tokens,
                        presence_request_indices,
                        shard_indices=shard_indices,
                    )
                )
                if not sparse_applied:
                    self._apply_sharded_presence_penalty_from_counts(
                        logits,
                        presence_penalties,
                        output_token_counts,
                        presence_request_indices,
                        shard_indices=shard_indices,
                    )
            else:
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
        local_vals, local_indices = torch.topk(logits, local_top_k, dim=-1)
        return local_vals, local_indices, vocab_start, shard_token_ids

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
        output_token_counts: torch.Tensor | None = None,
        presence_request_indices: torch.Tensor | None = None,
        output_unique_token_ids: torch.Tensor | None = None,
        num_output_unique_tokens: torch.Tensor | None = None,
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
        local_vals, local_indices, vocab_start, shard_token_ids = self._get_local_topk(
            lm_head,
            hidden_states,
            top_k=top_k,
            temperature=temperature,
            presence_penalties=presence_penalties,
            output_token_ids=output_token_ids,
            output_token_counts=output_token_counts,
            presence_request_indices=presence_request_indices,
            output_unique_token_ids=output_unique_token_ids,
            num_output_unique_tokens=num_output_unique_tokens,
            embedding_bias=embedding_bias,
        )
        local_top_k = local_vals.shape[-1]
        batch_size = hidden_states.shape[0]
        tp_size = lm_head.tp_size
        local_pairs = self._pack_topk_pairs(
            local_vals,
            local_indices,
            vocab_start=vocab_start,
            shard_token_ids=shard_token_ids,
        )
        if tp_size == 1:
            gathered_pairs = local_pairs.view(batch_size, local_top_k, 2)
            return self._sample_from_compact_topk_pairs(
                gathered_pairs, top_k=top_k, top_p=top_p
            )

        tp_group = get_tp_group()
        gathered_pairs = tensor_model_parallel_gather(local_pairs, dst=0, dim=-1)
        if tp_group.rank_in_group == 0:
            assert gathered_pairs is not None
            gathered_pairs = gathered_pairs.view(batch_size, tp_size * local_top_k, 2)
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
                self._get_compact_topk_sample_seeds(batch_size, hidden_states.device)
            else:
                q = torch.empty(
                    (batch_size, top_k),
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                q.exponential_()
            sampled = torch.empty(
                (batch_size,), dtype=torch.int64, device=hidden_states.device
            )

        tp_group.broadcast(sampled, src=0)
        return sampled

    @staticmethod
    def _apply_sharded_sparse_presence_penalty(
        logits: torch.Tensor,
        presence_penalties: torch.Tensor,
        output_unique_token_ids: torch.Tensor,
        num_output_unique_tokens: torch.Tensor,
        presence_request_indices: torch.Tensor,
        *,
        shard_indices,
    ) -> bool:
        """Apply presence penalties by touching only generated token ids."""
        if not (
            HAS_TRITON
            and apply_sparse_presence_penalty is not None
            and current_platform.is_cuda()
            and logits.shape[0] >= 32
            and logits.is_cuda
            and output_unique_token_ids.is_cuda
            and num_output_unique_tokens.is_cuda
            and presence_request_indices.is_cuda
            and presence_penalties.is_cuda
        ):
            return False

        num_org_elements = getattr(
            shard_indices,
            "num_org_elements",
            logits.shape[-1] - shard_indices.num_org_vocab_padding,
        )
        num_org_elements_padded = getattr(
            shard_indices,
            "num_org_elements_padded",
            num_org_elements + shard_indices.num_org_vocab_padding,
        )
        added_vocab_start = getattr(
            shard_indices,
            "added_vocab_start_index",
            shard_indices.org_vocab_end_index,
        )
        num_added_elements = getattr(shard_indices, "num_added_elements", 0)
        apply_sparse_presence_penalty(
            logits,
            output_unique_token_ids,
            num_output_unique_tokens,
            presence_request_indices,
            presence_penalties,
            org_vocab_start=shard_indices.org_vocab_start_index,
            num_org_elements=num_org_elements,
            num_org_elements_padded=num_org_elements_padded,
            added_vocab_start=added_vocab_start,
            num_added_elements=num_added_elements,
        )
        return True

    @staticmethod
    def _apply_sharded_presence_penalty_from_counts(
        logits: torch.Tensor,
        presence_penalties: torch.Tensor,
        output_token_counts: torch.Tensor,
        presence_request_indices: torch.Tensor,
        *,
        shard_indices,
        local_token_ids: torch.Tensor | None = None,
    ) -> None:
        """Apply V2's persistent output-token counts directly to local logits."""
        num_org_elements = getattr(
            shard_indices,
            "num_org_elements",
            logits.shape[-1] - shard_indices.num_org_vocab_padding,
        )
        num_org_elements_padded = getattr(
            shard_indices,
            "num_org_elements_padded",
            num_org_elements + shard_indices.num_org_vocab_padding,
        )
        added_vocab_start = getattr(
            shard_indices,
            "added_vocab_start_index",
            shard_indices.org_vocab_end_index,
        )
        num_added_elements = getattr(shard_indices, "num_added_elements", 0)

        if (
            HAS_TRITON
            and apply_presence_penalty_from_counts is not None
            and current_platform.is_cuda()
            and logits.is_cuda
            and output_token_counts.is_cuda
            and presence_request_indices.is_cuda
            and presence_penalties.is_cuda
            and (local_token_ids is None or local_token_ids.is_cuda)
        ):
            apply_presence_penalty_from_counts(
                logits,
                output_token_counts,
                presence_request_indices,
                presence_penalties,
                org_vocab_start=shard_indices.org_vocab_start_index,
                num_org_elements=num_org_elements,
                num_org_elements_padded=num_org_elements_padded,
                added_vocab_start=added_vocab_start,
                num_added_elements=num_added_elements,
                local_token_ids=local_token_ids,
            )
            return

        if output_token_counts.ndim != 2:
            raise ValueError("output_token_counts must be rank 2")
        if presence_request_indices.shape != (logits.shape[0],):
            raise ValueError("presence_request_indices must have one entry per row")
        if presence_penalties.shape != (logits.shape[0],):
            raise ValueError("presence_penalties must have one entry per row")
        if local_token_ids is not None and local_token_ids.shape != logits.shape:
            raise ValueError("local_token_ids must match logits")

        if local_token_ids is None:
            local_ids = torch.arange(
                logits.shape[-1], dtype=torch.int64, device=logits.device
            ).expand_as(logits)
        else:
            local_ids = local_token_ids.to(torch.int64)
        is_org = local_ids < num_org_elements
        added_offsets = local_ids - num_org_elements_padded
        is_added = (added_offsets >= 0) & (added_offsets < num_added_elements)
        global_ids = torch.where(
            is_org,
            local_ids + shard_indices.org_vocab_start_index,
            added_offsets + added_vocab_start,
        )
        valid = (is_org | is_added) & (global_ids >= 0)
        valid &= global_ids < output_token_counts.shape[-1]
        safe_ids = global_ids.clamp(0, output_token_counts.shape[-1] - 1)
        request_indices = presence_request_indices.to(torch.int64).unsqueeze(1)
        counts = output_token_counts[request_indices, safe_ids]
        penalty = presence_penalties.to(logits.dtype).unsqueeze(1)
        logits.sub_(torch.where(valid & (counts > 0), penalty, 0.0))

    @staticmethod
    def _get_sharded_presence_penalty_mask(
        logits: torch.Tensor,
        presence_penalties: torch.Tensor,
        output_token_ids: torch.Tensor,
        *,
        shard_indices,
    ) -> torch.Tensor | None:
        """Build the local dense presence mask used before and after refine."""
        if output_token_ids.numel() == 0:
            return None

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
        return penalty_mask[:, : logits.shape[1]]

    @classmethod
    def _apply_sharded_presence_penalty(
        cls,
        logits: torch.Tensor,
        presence_penalties: torch.Tensor,
        output_token_ids: torch.Tensor,
        *,
        shard_indices,
    ) -> None:
        """Apply presence-only penalty to a vocab-parallel logits shard."""
        penalty_mask = cls._get_sharded_presence_penalty_mask(
            logits,
            presence_penalties,
            output_token_ids,
            shard_indices=shard_indices,
        )
        if penalty_mask is not None:
            logits.sub_(penalty_mask)

    def get_top_k_tokens(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        k: int,
        embedding_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vocab-parallel top-k without all-gathering full logits.

        The `get_top_tokens` reduction widened from one token to k, returning
        the values as well as the global ids. Communication is
        O(batch * 2k * tp_size) rather than O(batch * vocab_size).

        Scale and soft cap are applied to the k selected values rather than
        the whole vocabulary; both are monotonic, so the selection is the same
        and only k entries are touched.
        """
        if self.scale <= 0.0 and self.scale != 1.0:
            raise ValueError(
                "The local top-k reduction optimization is not supported for "
                "non-positive logit scaling factors."
            )

        logits = self._apply_head(lm_head, hidden_states, embedding_bias)

        # Mask out padding entries beyond org_vocab_size on this shard.
        num_pad = lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")

        values, ids = _topk(logits, k)
        # Convert shard-local indices to global vocab indices.
        ids = ids.to(torch.int64) + lm_head.shard_indices.org_vocab_start_index

        if lm_head.tp_size > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = _topk(values, k)
            ids = ids.gather(-1, selected)

        values = values.float()
        if self.scale != 1.0:
            values = values * self.scale
        if self.soft_cap is not None:
            values = torch.tanh(values / self.soft_cap) * self.soft_cap
        return ids, values

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", org_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s
