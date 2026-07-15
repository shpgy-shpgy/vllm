# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.hybrid_mxfp4_lm_head as hybrid_mxfp4
from vllm.model_executor.layers.hybrid_mxfp4_lm_head import (
    HybridMxfp4LmHead,
    prepare_hybrid_mxfp4_lm_head,
    release_hybrid_mxfp4_lm_head,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor


def test_mxfp4_can_use_records_compact_path_failure() -> None:
    state = HybridMxfp4LmHead(
        weight=torch.empty((5, 2), dtype=torch.uint8),
        scale=torch.empty((128, 4), dtype=torch.uint8),
        input_size=4,
        output_size=5,
        candidates=2,
        max_rows=32,
    )
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)
    weight = torch.ones((5, 4), dtype=torch.bfloat16)

    assert not state.can_use(
        hidden,
        bf16_weight=weight,
        active_vocab_size=5,
        top_k=1,
    )
    assert state.can_use_failure_counts == {"hidden_not_cuda": 1}


def test_mxfp4_coarse_gemm_uses_mxfp4_contract(monkeypatch) -> None:
    state = HybridMxfp4LmHead(
        weight=torch.empty((5, 2), dtype=torch.uint8),
        scale=torch.empty((128, 4), dtype=torch.uint8),
        input_size=4,
        output_size=5,
        candidates=2,
        max_rows=32,
    )
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)
    calls: dict[str, object] = {}

    def fake_quantize(value: torch.Tensor):
        calls.setdefault("quantized_shapes", []).append(tuple(value.shape))
        return torch.empty((value.shape[0], 2), dtype=torch.uint8), torch.empty(
            (128, 4), dtype=torch.uint8
        )

    def fake_mm(*args, **kwargs):
        calls["mm_args"] = args
        calls["mm_kwargs"] = kwargs
        return torch.zeros((3, 5), dtype=torch.bfloat16)

    monkeypatch.setattr(hybrid_mxfp4, "flashinfer_mxfp4_quantize", fake_quantize)
    monkeypatch.setattr(hybrid_mxfp4, "flashinfer_scaled_fp4_mm", fake_mm)

    output = state.coarse_logits(hidden, None)

    assert output.shape == (3, 5)
    # MXFP4 pads to FlashInfer's row bucket before cuDNN GEMM and slices the
    # result back to the real M.  This avoids the runtime override-shape
    # workspace query that causes decode tails when M changes.
    assert calls["quantized_shapes"] == [(4, 4)]
    assert calls["mm_kwargs"] == {
        "alpha": None,
        "out_dtype": torch.bfloat16,
        "backend": "auto",
        "block_size": 32,
        "use_nvfp4": False,
    }


def test_mxfp4_topk_expands_transient_candidates_only() -> None:
    state = HybridMxfp4LmHead(
        weight=torch.empty((4096, 16), dtype=torch.uint8),
        scale=torch.empty((4096, 1), dtype=torch.uint8),
        input_size=32,
        output_size=4096,
        candidates=128,
        max_rows=0,
    )

    assert state.candidate_count_for_topk(1) == 128
    assert state.candidate_count_for_topk(20) == 128
    assert state.candidate_count_for_topk(40) == 512
    assert state.candidate_count_for_topk(100) == 1024
    # The persistent quantized tensors and configured width are unchanged.
    assert state.candidates == 128


def test_mxfp4_selector_uses_expanded_topk_width(monkeypatch) -> None:
    state = HybridMxfp4LmHead(
        weight=torch.empty((4096, 16), dtype=torch.uint8),
        scale=torch.empty((4096, 1), dtype=torch.uint8),
        input_size=32,
        output_size=4096,
        candidates=128,
        max_rows=0,
    )
    calls: dict[str, int] = {}

    def fake_select(coarse_logits, candidates, **kwargs):
        calls["candidates"] = candidates
        return torch.zeros(
            (coarse_logits.shape[0], candidates), dtype=torch.int64
        )

    monkeypatch.setattr(hybrid_mxfp4, "select_lm_head_candidates", fake_select)
    output = state.select_candidates(torch.zeros((1, 4096)), top_k=40)

    assert output.shape == (1, 512)
    assert calls["candidates"] == 512


def test_mxfp4_shared_state_attaches_without_requantizing(monkeypatch) -> None:
    weight = torch.nn.Parameter(torch.empty((256, 32), dtype=torch.bfloat16))
    state = HybridMxfp4LmHead(
        weight=torch.empty((256, 16), dtype=torch.uint8),
        scale=torch.empty((256, 4), dtype=torch.uint8),
        input_size=32,
        output_size=256,
        candidates=40,
        max_rows=64,
    )
    setattr(weight, "_hybrid_mxfp4_lm_head_shared_state", state)
    layer = torch.nn.Module()
    layer.weight = weight

    monkeypatch.setattr(hybrid_mxfp4, "has_flashinfer", lambda: False)

    assert prepare_hybrid_mxfp4_lm_head(layer, candidates=40)
    assert layer._hybrid_mxfp4_lm_head_state is state
    assert layer._hybrid_mxfp4_lm_head_weight is state.weight
    assert layer._hybrid_mxfp4_lm_head_scale is state.scale


def test_release_hybrid_mxfp4_lm_head_drops_registered_buffers() -> None:
    layer = torch.nn.Module()
    weight = torch.empty((8, 2), dtype=torch.uint8)
    scale = torch.empty((128, 4), dtype=torch.uint8)
    layer.register_buffer("_hybrid_mxfp4_lm_head_weight", weight, persistent=False)
    layer.register_buffer("_hybrid_mxfp4_lm_head_scale", scale, persistent=False)
    layer._hybrid_mxfp4_lm_head_state = object()

    released = release_hybrid_mxfp4_lm_head(layer)

    assert released == weight.nbytes + scale.nbytes
    assert not hasattr(layer, "_hybrid_mxfp4_lm_head_state")
    assert not hasattr(layer, "_hybrid_mxfp4_lm_head_weight")
    assert not hasattr(layer, "_hybrid_mxfp4_lm_head_scale")


def test_logits_processor_accepts_mxfp4_state() -> None:
    class FakeState:
        candidates = 2

        def can_use(self, *args, **kwargs) -> bool:
            return True

        def coarse_logits(self, hidden_states, bias):
            return torch.tensor(
                [[10.0, 9.0, 8.0, 7.0]], dtype=torch.bfloat16
            ).expand(hidden_states.shape[0], -1).clone()

        def select_candidates(self, coarse_logits, *, top_k=None):
            return torch.topk(coarse_logits, 2, dim=-1, sorted=False).indices

        def refine_logits(
            self, hidden_states, bf16_weight, candidate_indices, bias
        ):
            exact = torch.tensor(
                [[10.0, 9.0, 8.0, 7.0]], dtype=torch.bfloat16
            ).expand(hidden_states.shape[0], -1)
            return exact.gather(1, candidate_indices)

    shard_indices = type(
        "ShardIndices",
        (),
        {
            "org_vocab_start_index": 0,
            "org_vocab_end_index": 4,
            "num_elements_padded": 4,
            "num_org_vocab_padding": 0,
            "num_added_elements_padded": 0,
        },
    )()
    lm_head = type(
        "LmHead",
        (),
        {
            "shard_indices": shard_indices,
            "num_embeddings_per_partition": 4,
            "weight": torch.zeros((4, 4), dtype=torch.bfloat16),
            "_hybrid_mxfp4_lm_head_state": FakeState(),
        },
    )()

    values, indices, vocab_start, shard_token_ids = LogitsProcessor(
        vocab_size=4
    )._get_local_topk(
        lm_head,
        torch.zeros((1, 4), dtype=torch.bfloat16),
        top_k=2,
        temperature=1.0,
    )

    assert values.tolist() == [[10.0, 9.0]]
    assert indices.shape == (1, 2)
    assert vocab_start == 0
    assert shard_token_ids is None
