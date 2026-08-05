# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.hybrid_fp4_lm_head import (
    release_hybrid_fp4_lm_head,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor


class _FakeHybridState:
    def __init__(
        self,
        coarse_logits: torch.Tensor,
        exact_logits: torch.Tensor,
        candidates: int,
    ) -> None:
        self._coarse_logits = coarse_logits
        self._exact_logits = exact_logits
        self.candidates = candidates

    def can_use(self, *args, **kwargs) -> bool:
        return True

    def coarse_logits(
        self,
        hidden_states: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        return self._coarse_logits.expand(hidden_states.shape[0], -1).clone()

    def select_candidates(self, coarse_logits: torch.Tensor) -> torch.Tensor:
        return torch.topk(coarse_logits, self.candidates, dim=-1, sorted=False).indices

    def refine_logits(
        self,
        hidden_states: torch.Tensor,
        bf16_weight: torch.Tensor,
        candidate_indices: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        exact = self._exact_logits.expand(hidden_states.shape[0], -1)
        return exact.gather(1, candidate_indices)


def _make_lm_head(hybrid_state: _FakeHybridState) -> SimpleNamespace:
    shard_indices = SimpleNamespace(
        org_vocab_start_index=0,
        org_vocab_end_index=8,
        num_elements_padded=8,
        num_org_vocab_padding=0,
        num_added_elements_padded=0,
    )
    return SimpleNamespace(
        shard_indices=shard_indices,
        num_embeddings_per_partition=8,
        weight=torch.zeros((8, 4), dtype=torch.bfloat16),
        _hybrid_fp4_lm_head_state=hybrid_state,
    )


def test_presence_penalty_is_applied_before_and_after_refinement() -> None:
    logits = torch.tensor(
        [[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]],
        dtype=torch.bfloat16,
    )
    hybrid_state = _FakeHybridState(logits, logits, candidates=2)
    lm_head = _make_lm_head(hybrid_state)
    processor = LogitsProcessor(vocab_size=8)

    values, indices, vocab_start, shard_token_ids = processor._get_local_topk(
        lm_head,
        torch.zeros((1, 4), dtype=torch.bfloat16),
        top_k=2,
        temperature=1.0,
        presence_penalties=torch.tensor([5.0]),
        output_token_ids=torch.tensor([[0]], dtype=torch.int64),
        embedding_bias=None,
    )

    assert values.tolist() == [[9.0, 8.0]]
    assert indices.tolist() == [[1, 2]]
    assert vocab_start == 0
    assert shard_token_ids is None


def test_presence_penalty_can_use_persistent_output_counts() -> None:
    logits = torch.tensor(
        [[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]],
        dtype=torch.bfloat16,
    )
    hybrid_state = _FakeHybridState(logits, logits, candidates=2)
    lm_head = _make_lm_head(hybrid_state)
    processor = LogitsProcessor(vocab_size=8)
    output_token_counts = torch.zeros((3, 8), dtype=torch.int32)
    output_token_counts[2, 0] = 1

    values, indices, vocab_start, shard_token_ids = processor._get_local_topk(
        lm_head,
        torch.zeros((1, 4), dtype=torch.bfloat16),
        top_k=2,
        temperature=1.0,
        presence_penalties=torch.tensor([5.0]),
        output_token_counts=output_token_counts,
        presence_request_indices=torch.tensor([2], dtype=torch.int32),
    )

    assert values.tolist() == [[9.0, 8.0]]
    assert indices.tolist() == [[1, 2]]
    assert vocab_start == 0
    assert shard_token_ids is None


def test_presence_penalty_mask_maps_only_local_tokens() -> None:
    shard_indices = SimpleNamespace(
        org_vocab_start_index=4,
        org_vocab_end_index=8,
        num_org_vocab_padding=0,
    )
    mask = LogitsProcessor._get_sharded_presence_penalty_mask(
        torch.zeros((1, 4), dtype=torch.float32),
        torch.tensor([2.0]),
        torch.tensor([[1, 4, 6, 9]], dtype=torch.int64),
        shard_indices=shard_indices,
    )

    assert mask is not None
    assert mask.tolist() == [[2.0, 0.0, 2.0, 0.0]]


def test_compact_top_p_filters_after_global_top_k() -> None:
    processor = LogitsProcessor(vocab_size=64)
    pairs = torch.tensor(
        [[[4.0, 40.0], [3.0, 30.0], [2.0, 20.0], [1.0, 10.0]]],
        dtype=torch.float32,
    )

    values, indices = processor._select_from_compact_topk_pairs(
        pairs,
        top_k=3,
        top_p=0.8,
    )

    assert values[0, :2].tolist() == [4.0, 3.0]
    assert torch.isneginf(values[0, 2])
    assert indices.tolist() == [[40, 30, 20]]


def test_release_hybrid_fp4_lm_head_drops_registered_buffers() -> None:
    layer = torch.nn.Module()
    weight = torch.empty((8, 2), dtype=torch.uint8)
    scale = torch.empty((128, 4), dtype=torch.float8_e4m3fn)
    input_scale = torch.empty((), dtype=torch.float32)
    alpha = torch.empty((), dtype=torch.float32)
    layer.register_buffer("_hybrid_fp4_lm_head_weight", weight, persistent=False)
    layer.register_buffer("_hybrid_fp4_lm_head_scale", scale, persistent=False)
    layer.register_buffer(
        "_hybrid_fp4_lm_head_input_scale", input_scale, persistent=False
    )
    layer.register_buffer("_hybrid_fp4_lm_head_alpha", alpha, persistent=False)
    layer._hybrid_fp4_lm_head_state = object()

    released = release_hybrid_fp4_lm_head(layer)

    assert released == weight.nbytes + scale.nbytes + input_scale.nbytes + alpha.nbytes
    assert not hasattr(layer, "_hybrid_fp4_lm_head_state")
    assert not hasattr(layer, "_hybrid_fp4_lm_head_weight")
    assert not hasattr(layer, "_hybrid_fp4_lm_head_scale")
    assert not hasattr(layer, "_hybrid_fp4_lm_head_input_scale")
    assert not hasattr(layer, "_hybrid_fp4_lm_head_alpha")
