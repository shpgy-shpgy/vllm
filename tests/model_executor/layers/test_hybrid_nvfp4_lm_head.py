# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 3, 4, 7, 16, 31, 32, 64, 103])
@pytest.mark.parametrize("magnitude", [0.0, 1.0e-15, 1.0, 1.0e10])
def test_nvfp4_global_scale_cuda_graph_matches_reference(rows, magnitude):
    # Cover both reduction stages, zero handling, the clamp, and the large
    # tensor fallback. Scaling must not change NVFP4 quantization decisions.
    tensor = torch.randn((rows, 2560), device="cuda", dtype=torch.bfloat16)
    tensor *= magnitude
    maximum = tensor.abs().amax().float()
    expected = torch.where(
        maximum > 0,
        maximum.clamp_min(1.0e-12).reciprocal() * 2688.0,
        torch.ones_like(maximum),
    )
    hybrid_nvfp4._global_scale(tensor)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = hybrid_nvfp4._global_scale(tensor)
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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
    weight._hybrid_nvfp4_lm_head_shared_state = state
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
            return (
                torch.tensor([[10.0, 9.0, 8.0, 7.0]], dtype=torch.bfloat16)
                .expand(hidden_states.shape[0], -1)
                .clone()
            )

        def select_candidates(self, coarse_logits, *, top_k=None):
            return torch.topk(coarse_logits, 2, dim=-1, sorted=False).indices

        def refine_logits(self, hidden_states, bf16_weight, candidate_indices, bias):
            exact = torch.tensor([[10.0, 9.0, 8.0, 7.0]], dtype=torch.bfloat16).expand(
                hidden_states.shape[0], -1
            )
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


@pytest.mark.skipif(
    not hybrid_nvfp4.current_platform.is_device_capability_family(120)
    or not hybrid_nvfp4.has_flashinfer_b12x_gemm(),
    reason="requires SM120 with FlashInfer b12x",
)
@torch.inference_mode()
def test_greedy_graph_replay_preserves_outputs_and_falls_back(monkeypatch):
    """Replay must use new inputs and own outputs across shared-pool replays."""
    from types import SimpleNamespace

    torch.manual_seed(42)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(512, 256, device="cuda", dtype=torch.bfloat16),
        requires_grad=False,
    )
    assert prepare_hybrid_nvfp4_lm_head(layer, candidates=128)
    state = hybrid_nvfp4.get_hybrid_nvfp4_lm_head(layer)
    state.capture_head_graphs(layer.weight, 8)
    processor = LogitsProcessor(vocab_size=512)
    layer.tp_size = 1
    layer.shard_indices = SimpleNamespace(
        org_vocab_start_index=0,
        org_vocab_end_index=512,
        num_elements_padded=512,
        num_org_vocab_padding=0,
        num_added_elements_padded=0,
    )
    retained = []
    for rows in (1, 8, 3, 8, 1, 9):
        hidden = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        graphs = state._head_graphs
        with monkeypatch.context() as patch:
            patch.setattr(state, "_head_graphs", None)
            expected = processor.get_top_tokens(layer, hidden)
        if rows <= 8:
            # Ensure the public path really replays, not an eager fallback.
            with monkeypatch.context() as patch:
                patch.setattr(
                    state, "coarse_logits", lambda *a, **k: pytest.fail("eager head")
                )
                actual = processor.get_top_tokens(layer, hidden)
        else:
            assert state.replay_greedy(hidden, layer.weight) is None
            actual = processor.get_top_tokens(layer, hidden)
        assert state._head_graphs is graphs
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        retained.append((actual, expected))
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    hidden = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    assert state.replay_greedy(hidden, layer.weight.clone()) is None
    with monkeypatch.context() as patch:
        patch.setattr(state, "candidates", 64)
        assert state.replay_greedy(hidden, layer.weight) is None
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        assert state.replay_greedy(hidden, layer.weight) is None
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        assert state.replay_greedy(hidden, layer.weight) is None
    # A concurrent caller must never overwrite the static input buffer.
    with state._head_graphs.lock:
        assert state.replay_greedy(hidden, layer.weight) is None

    # Bias and padded vocab cannot use the full-vocab, bias-free captured head.
    with monkeypatch.context() as patch:
        patch.setattr(state, "replay_greedy", lambda *a: pytest.fail("graph replay"))
        processor.get_top_tokens(layer, hidden, torch.zeros_like(layer.weight[:, 0]))
        layer.shard_indices.num_org_vocab_padding = 1
        processor.get_top_tokens(layer, hidden)


@pytest.mark.skipif(
    not hybrid_nvfp4.current_platform.is_device_capability_family(120)
    or not hybrid_nvfp4.has_flashinfer_b12x_gemm(),
    reason="requires SM120 with FlashInfer b12x",
)
@torch.inference_mode()
def test_head_graph_topk_sampling_and_presence_match_eager(monkeypatch):
    """Candidate graphs preserve processors; replay never freezes sampling RNG."""
    from types import SimpleNamespace

    torch.manual_seed(42)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.randn(512, 256, device="cuda", dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.tp_size = 1
    layer.shard_indices = SimpleNamespace(
        org_vocab_start_index=0,
        org_vocab_end_index=512,
        num_elements_padded=512,
        num_org_vocab_padding=0,
        num_added_elements_padded=0,
    )
    assert prepare_hybrid_nvfp4_lm_head(layer, candidates=128)
    state = hybrid_nvfp4.get_hybrid_nvfp4_lm_head(layer)
    state.capture_head_graphs(layer.weight, 3)
    processor = LogitsProcessor(vocab_size=512)
    for rows, k, scale, soft_cap, penalty in (
        (3, 20, 1.0, None, False),
        (1, 40, 0.7, 10.0, False),
        (3, 128, 1.0, None, False),
        (3, 20, -1.0, 10.0, True),
    ):
        hidden = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)
        processor.scale, processor.soft_cap = scale, soft_cap
        kwargs = dict(top_k=k, top_p=0.9, temperature=0.7)
        if penalty:
            kwargs.update(
                presence_penalties=torch.full((rows,), 1.5, device="cuda"),
                output_token_counts=torch.randint(
                    0, 2, (rows, 512), device="cuda", dtype=torch.int32
                ),
                presence_request_indices=torch.arange(rows, device="cuda"),
                embedding_bias=torch.randn(512, device="cuda", dtype=torch.bfloat16),
            )
        with monkeypatch.context() as patch:
            patch.setattr(state, "_head_graphs", None)
            expected = processor.get_topk_candidates(layer, hidden, **kwargs)
            torch.manual_seed(123)
            expected_sample = processor.sample_topk_tokens(layer, hidden, **kwargs)
        with monkeypatch.context() as patch:
            # Neither fused candidates nor coarse-only replay may launch GEMM eagerly.
            patch.setattr(
                hybrid_nvfp4,
                "flashinfer_scaled_fp4_mm",
                lambda *a, **k: pytest.fail("eager GEMM"),
            )
            actual = processor.get_topk_candidates(layer, hidden, **kwargs)
            torch.manual_seed(123)
            actual_sample = processor.sample_topk_tokens(layer, hidden, **kwargs)

        # FlashInfer returns an unsorted set: tied BF16 logits can exchange
        # slots. Compare the resulting token distribution, not tie ordering.
        def distribution(pair):
            values, ids = pair
            probs = torch.zeros(values.shape[0], 512, device="cuda")
            return probs.scatter_(1, ids.long(), values.softmax(-1))

        torch.testing.assert_close(
            distribution(actual), distribution(expected), rtol=0, atol=0
        )
        assert (
            distribution(expected).gather(1, actual_sample.long().view(-1, 1)) > 0
        ).all()
        assert (
            distribution(expected).gather(1, expected_sample.long().view(-1, 1)) > 0
        ).all()
        rng_before = torch.cuda.get_rng_state()
        processor.sample_topk_tokens(layer, hidden, **kwargs)
        assert not torch.equal(rng_before, torch.cuda.get_rng_state())
