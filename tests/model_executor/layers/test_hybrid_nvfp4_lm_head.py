# SPDX-License-Identifier: Apache-2.0

import torch

import vllm.model_executor.layers.hybrid_nvfp4_lm_head as hybrid_nvfp4
from vllm.model_executor.layers.hybrid_nvfp4_lm_head import (
    HybridNvfp4LmHead,
    prepare_hybrid_nvfp4_lm_head,
    release_hybrid_nvfp4_lm_head,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor


def _state(**kwargs) -> HybridNvfp4LmHead:
    defaults = dict(
        weight=torch.empty((8, 2), dtype=torch.uint8),
        scale=torch.empty((128, 4), dtype=torch.uint8),
        global_scale=torch.tensor(1.0),
        input_size=4,
        output_size=8,
        candidates=4,
        max_rows=32,
    )
    defaults.update(kwargs)
    return HybridNvfp4LmHead(**defaults)


def test_nvfp4_global_scale_zero_is_finite() -> None:
    scale = hybrid_nvfp4._global_scale(torch.zeros((2, 4), dtype=torch.bfloat16))
    assert torch.isfinite(scale)
    assert scale.item() == 1.0


def test_nvfp4_global_scale_matches_fp32_reference_for_finite_bf16() -> None:
    tensor = torch.tensor(
        [[-3.5, 1.25, 0.0, 2.0], [4.0, -0.5, 1.0, -2.5]],
        dtype=torch.bfloat16,
    )
    expected_max = tensor.float().abs().nan_to_num().amax()
    expected = torch.where(
        expected_max > 0,
        expected_max.clamp_min(1.0e-12).reciprocal() * 448.0 * 6.0,
        torch.ones_like(expected_max),
    )
    torch.testing.assert_close(hybrid_nvfp4._global_scale(tensor), expected)


def test_nvfp4_can_use_records_compact_path_failure() -> None:
    state = _state(candidates=2)
    hidden = torch.ones((3, 4), dtype=torch.bfloat16)
    weight = torch.ones((8, 4), dtype=torch.bfloat16)

    assert not state.can_use(
        hidden,
        bf16_weight=weight,
        active_vocab_size=8,
        top_k=1,
    )
    assert state.can_use_failure_counts == {"hidden_not_cuda": 1}


def test_nvfp4_coarse_gemm_uses_b12x_contract(monkeypatch) -> None:
    state = _state()
    hidden = torch.ones((2, 4), dtype=torch.bfloat16)
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        hybrid_nvfp4,
        "_global_scale",
        lambda value: torch.tensor(2.0),
    )

    def fake_quantize(value: torch.Tensor, global_scale: torch.Tensor):
        calls["quantize"] = (tuple(value.shape), float(global_scale))
        return (
            torch.empty((value.shape[0], 2), dtype=torch.uint8),
            torch.empty((128, 4), dtype=torch.uint8),
        )

    def fake_mm(*args, **kwargs):
        calls["mm_args"] = args
        calls["mm_kwargs"] = kwargs
        return torch.zeros((2, 8), dtype=torch.bfloat16)

    monkeypatch.setattr(
        hybrid_nvfp4,
        "flashinfer_nvfp4_quantize_128x4",
        fake_quantize,
    )
    monkeypatch.setattr(hybrid_nvfp4, "flashinfer_scaled_fp4_mm", fake_mm)

    output = state.coarse_logits(hidden, None)

    assert output.shape == (2, 8)
    assert calls["quantize"] == ((2, 4), 2.0)
    assert calls["mm_kwargs"] == {
        "alpha": torch.tensor(0.5),
        "out_dtype": torch.bfloat16,
        "backend": "b12x",
        "block_size": 16,
        "use_nvfp4": True,
    }


def test_nvfp4_shared_state_attaches_without_requantizing(monkeypatch) -> None:
    weight = torch.nn.Parameter(torch.empty((256, 32), dtype=torch.bfloat16))
    state = _state(
        weight=torch.empty((256, 16), dtype=torch.uint8),
        scale=torch.empty((256, 2), dtype=torch.uint8),
        input_size=32,
        output_size=256,
        candidates=40,
        max_rows=64,
    )
    setattr(weight, "_hybrid_nvfp4_lm_head_shared_state", state)
    layer = torch.nn.Module()
    layer.weight = weight

    monkeypatch.setattr(hybrid_nvfp4, "has_flashinfer", lambda: False)

    assert prepare_hybrid_nvfp4_lm_head(layer, candidates=40)
    assert layer._hybrid_nvfp4_lm_head_state is state
    assert layer._hybrid_nvfp4_lm_head_weight is state.weight
    assert layer._hybrid_nvfp4_lm_head_scale is state.scale


def test_release_nvfp4_lm_head_drops_registered_buffers() -> None:
    layer = torch.nn.Module()
    weight = torch.empty((8, 2), dtype=torch.uint8)
    scale = torch.empty((128, 4), dtype=torch.uint8)
    global_scale = torch.tensor(1.0)
    layer.register_buffer("_hybrid_nvfp4_lm_head_weight", weight, persistent=False)
    layer.register_buffer("_hybrid_nvfp4_lm_head_scale", scale, persistent=False)
    layer.register_buffer(
        "_hybrid_nvfp4_lm_head_global_scale",
        global_scale,
        persistent=False,
    )
    layer._hybrid_nvfp4_lm_head_state = object()

    released = release_hybrid_nvfp4_lm_head(layer)

    assert released == weight.nbytes + scale.nbytes + global_scale.nbytes
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_state")
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_weight")
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_scale")
    assert not hasattr(layer, "_hybrid_nvfp4_lm_head_global_scale")


def test_logits_processor_accepts_nvfp4_state() -> None:
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
            "_hybrid_nvfp4_lm_head_state": FakeState(),
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
