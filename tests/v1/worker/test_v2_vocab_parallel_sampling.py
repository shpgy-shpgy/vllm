# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler


def _make_fast_path_sampler(
    temperatures: list[float],
    *,
    top_k: int = 100,
    top_p: float = 1.0,
    min_p: float = 0.0,
    num_logprobs: int = NO_LOGPROBS,
    num_logprob_token_ids: int = 0,
    use_penalty: bool = False,
    presence_only_penalty: bool = False,
    use_logit_bias: bool = False,
    num_bad_words: int = 0,
    explicit_seed: bool = False,
    use_fp64_gumbel: bool = False,
) -> tuple[Sampler, SimpleNamespace]:
    num_reqs = len(temperatures)
    idx_mapping = np.arange(num_reqs, dtype=np.int32)
    sampling_states = SimpleNamespace(
        vocab_size=100,
        temperature=SimpleNamespace(np=np.asarray(temperatures, dtype=np.float32)),
        top_k=SimpleNamespace(np=np.full(num_reqs, top_k, dtype=np.int32)),
        top_p=SimpleNamespace(np=np.full(num_reqs, top_p, dtype=np.float32)),
        min_p=SimpleNamespace(np=np.full(num_reqs, min_p, dtype=np.float32)),
        max_num_logprobs=lambda _: num_logprobs,
        any_explicit_seed=lambda _: explicit_seed,
    )
    sampler = Sampler.__new__(Sampler)
    sampler.compute_nans = False
    sampler.use_fp64_gumbel = use_fp64_gumbel
    sampler.sampling_states = sampling_states
    any_penalty = use_penalty or presence_only_penalty
    sampler.penalties_state = SimpleNamespace(
        use_penalty=np.full(num_reqs, any_penalty, dtype=np.bool_),
        repetition_penalty=SimpleNamespace(np=np.ones(num_reqs, dtype=np.float32)),
        frequency_penalty=SimpleNamespace(
            np=np.full(num_reqs, 0.1 if use_penalty else 0.0, dtype=np.float32)
        ),
        presence_penalty=SimpleNamespace(
            np=np.full(
                num_reqs,
                0.1 if presence_only_penalty else 0.0,
                dtype=np.float32,
            )
        ),
    )
    sampler.logit_bias_state = SimpleNamespace(
        use_logit_bias=np.full(num_reqs, use_logit_bias, dtype=np.bool_)
    )
    sampler.bad_words_state = SimpleNamespace(
        num_bad_words=SimpleNamespace(
            np=np.full(num_reqs, num_bad_words, dtype=np.int32)
        )
    )
    sampler.logprob_token_ids_state = SimpleNamespace(
        max_num_token_ids=lambda _: num_logprob_token_ids
    )
    return sampler, SimpleNamespace(idx_mapping_np=idx_mapping)


@pytest.mark.parametrize(
    ("temperatures", "top_k", "top_p", "expected_mode"),
    [
        ([0.0], 100, 1.0, "greedy"),
        ([1.0, 1.0], 100, 1.0, "full"),
        ([0.7, 0.7], 20, 0.9, "topk"),
        ([0.7, 0.8], 20, 0.9, None),
    ],
)
def test_vocab_parallel_sampling_modes(
    temperatures: list[float],
    top_k: int,
    top_p: float,
    expected_mode: str | None,
) -> None:
    sampler, input_batch = _make_fast_path_sampler(
        temperatures, top_k=top_k, top_p=top_p
    )
    params = sampler.get_vocab_parallel_sampling_params(input_batch)
    assert (params[0] if params is not None else None) == expected_mode


def test_vocab_parallel_topk_supports_presence_only_penalty() -> None:
    sampler, input_batch = _make_fast_path_sampler(
        [0.7], top_k=20, top_p=0.9, presence_only_penalty=True
    )
    params = sampler.get_vocab_parallel_sampling_params(input_batch)
    assert params is not None
    assert params[0] == "topk"
    assert params[4]


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_vocab_parallel_non_topk_falls_back_for_presence_penalty(
    temperature: float,
) -> None:
    sampler, input_batch = _make_fast_path_sampler(
        [temperature], presence_only_penalty=True
    )
    assert sampler.get_vocab_parallel_sampling_params(input_batch) is None


@pytest.mark.parametrize(
    "unsupported_params",
    [
        {"num_logprobs": 1},
        {"num_logprob_token_ids": 1},
        {"use_penalty": True},
        {"use_logit_bias": True},
        {"num_bad_words": 1},
        {"explicit_seed": True},
        {"use_fp64_gumbel": True},
        {"min_p": 0.1},
        {"top_k": 65},
    ],
)
def test_vocab_parallel_sampling_falls_back_for_unsupported_features(
    unsupported_params: dict[str, Any],
) -> None:
    params: dict[str, Any] = {"top_k": 20, "top_p": 0.9}
    params.update(unsupported_params)
    sampler, input_batch = _make_fast_path_sampler([0.7], **params)
    assert sampler.get_vocab_parallel_sampling_params(input_batch) is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    ("candidate_ids", "expected_tokens", "expected_sampled", "expected_rejected"),
    [
        (
            [[11, 101], [12, 102], [31, 103], [21, 104], [41, 105]],
            [[11, 12, 31], [21, 41, -1]],
            [3, 2],
            [0, 0],
        ),
        (
            [[51, 101], [52, 102], [31, 103], [61, 104], [41, 105]],
            [[51, -1, -1], [61, -1, -1]],
            [1, 1],
            [2, 1],
        ),
    ],
)
def test_compact_rejection_sampler(
    candidate_ids: list[list[int]],
    expected_tokens: list[list[int]],
    expected_sampled: list[int],
    expected_rejected: list[int],
) -> None:
    device = torch.device("cuda:0")
    cu_num_logits = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_draft_tokens=3,
        num_draft_tokens_per_req=np.array([2, 1], dtype=np.int32),
        cu_num_logits=cu_num_logits,
        cu_num_logits_np=np.array([0, 3, 5], dtype=np.int32),
        input_ids=torch.tensor([99, 11, 12, 88, 21], dtype=torch.int32, device=device),
        logits_indices=torch.arange(5, dtype=torch.int64, device=device),
        seq_lens=torch.tensor([10, 10], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device=device),
    )
    candidate_logits = torch.tensor(
        [[0.0, -float("inf")]] * 5,
        dtype=torch.float32,
        device=device,
    )

    rejection_sampler = RejectionSampler.__new__(RejectionSampler)
    rejection_sampler.num_speculative_steps = 2
    rejection_sampler.synthetic_conditional_rates = None
    rejection_sampler.use_block_verification = False
    rejection_sampler.sampler = SimpleNamespace(
        req_states=SimpleNamespace(
            prefill_len=SimpleNamespace(
                gpu=torch.tensor([1, 1], dtype=torch.int32, device=device)
            )
        )
    )
    output = rejection_sampler.sample_from_topk_candidates(
        candidate_logits,
        torch.tensor(candidate_ids, dtype=torch.int64, device=device),
        input_batch,
    )
    torch.accelerator.synchronize()

    assert output.sampled_token_ids.tolist() == expected_tokens
    assert output.num_sampled.tolist() == expected_sampled
    assert output.num_rejected.tolist() == expected_rejected


@pytest.mark.parametrize(
    "hybrid_env",
    ["VLLM_HYBRID_MXFP4_LM_HEAD", "VLLM_HYBRID_MXFP8_LM_HEAD"],
)
def test_greedy_speculative_sampling_uses_compact_tokens(
    monkeypatch, hybrid_env: str
) -> None:
    monkeypatch.setattr(envs, "VLLM_HYBRID_MXFP4_LM_HEAD", False)
    monkeypatch.setattr(envs, "VLLM_HYBRID_MXFP8_LM_HEAD", False)
    monkeypatch.setattr(envs, hybrid_env, True)
    runner = object.__new__(GPUModelRunner)
    runner.sampler = SimpleNamespace(
        get_vocab_parallel_sampling_params=lambda _: (
            "greedy",
            100,
            1.0,
            0.0,
            False,
        )
    )

    captured: dict[str, Any] = {}

    def get_top_tokens(hidden_states: torch.Tensor):
        captured["hidden_states"] = hidden_states
        return torch.tensor([11, 12, 13])

    runner.model = SimpleNamespace(get_top_tokens=get_top_tokens)
    expected_output = SimpleNamespace(num_sampled="sampled", num_rejected="rejected")

    def sample_from_greedy_tokens(token_ids, input_batch):
        captured["target_token_ids"] = token_ids
        captured["input_batch"] = input_batch
        return expected_output

    runner.rejection_sampler = SimpleNamespace(
        synthetic_conditional_rates=None,
        use_block_verification=False,
        sample_from_greedy_tokens=sample_from_greedy_tokens,
    )
    runner.speculator = SimpleNamespace(draft_logits=None)
    input_batch = SimpleNamespace(
        logits_indices=torch.tensor([0, 1, 2]),
        has_structured_output_reqs=False,
        num_draft_tokens=2,
    )
    hidden_states = torch.randn((3, 4))

    output, num_sampled, num_rejected = GPUModelRunner.sample(
        runner,
        hidden_states,
        input_batch,
        grammar_output=None,
    )

    assert output is expected_output
    assert num_sampled == "sampled"
    assert num_rejected == "rejected"
    assert torch.equal(captured["hidden_states"], hidden_states)
    assert captured["target_token_ids"].tolist() == [11, 12, 13]
    assert captured["input_batch"] is input_batch


def test_greedy_speculative_sampling_keeps_bf16_route(monkeypatch) -> None:
    monkeypatch.setattr(envs, "VLLM_HYBRID_MXFP4_LM_HEAD", False)
    monkeypatch.setattr(envs, "VLLM_HYBRID_MXFP8_LM_HEAD", False)
    runner = object.__new__(GPUModelRunner)
    runner.sampler = SimpleNamespace(
        get_vocab_parallel_sampling_params=lambda _: (
            "greedy",
            100,
            1.0,
            0.0,
            False,
        )
    )

    captured: dict[str, Any] = {}
    logits = torch.randn((3, 8))

    def get_top_tokens(_hidden_states: torch.Tensor):
        pytest.fail("BF16 speculative sampling must not enter the hybrid fast path")

    def compute_logits(hidden_states: torch.Tensor):
        captured["hidden_states"] = hidden_states
        return logits

    runner.model = SimpleNamespace(
        get_top_tokens=get_top_tokens,
        compute_logits=compute_logits,
    )
    expected_output = SimpleNamespace(num_sampled="sampled", num_rejected="rejected")

    class FakeRejectionSampler:
        synthetic_conditional_rates = None
        use_block_verification = False

        def __call__(self, target_logits, input_batch, draft_logits):
            captured["target_logits"] = target_logits
            captured["input_batch"] = input_batch
            captured["draft_logits"] = draft_logits
            return expected_output

    runner.rejection_sampler = FakeRejectionSampler()
    runner.speculator = SimpleNamespace(draft_logits=None)
    input_batch = SimpleNamespace(
        logits_indices=torch.tensor([0, 1, 2]),
        has_structured_output_reqs=False,
        num_draft_tokens=2,
    )
    hidden_states = torch.randn((3, 4))

    output, num_sampled, num_rejected = GPUModelRunner.sample(
        runner,
        hidden_states,
        input_batch,
        grammar_output=None,
    )

    assert output is expected_output
    assert num_sampled == "sampled"
    assert num_rejected == "rejected"
    assert torch.equal(captured["hidden_states"], hidden_states)
    assert captured["target_logits"] is logits
    assert captured["input_batch"] is input_batch
    assert captured["draft_logits"] is None


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    ("target_token_ids", "expected_tokens", "expected_sampled", "expected_rejected"),
    [
        ([11, 12, 31, 21, 41], [[11, 12, 31], [21, 41, -1]], [3, 2], [0, 0]),
        ([51, 52, 31, 61, 41], [[51, -1, -1], [61, -1, -1]], [1, 1], [2, 1]),
    ],
)
def test_compact_greedy_rejection_sampler(
    target_token_ids: list[int],
    expected_tokens: list[list[int]],
    expected_sampled: list[int],
    expected_rejected: list[int],
) -> None:
    device = torch.device("cuda:0")
    input_batch = SimpleNamespace(
        num_reqs=2,
        num_draft_tokens=3,
        num_draft_tokens_per_req=np.array([2, 1], dtype=np.int32),
        cu_num_logits=torch.tensor([0, 3, 5], dtype=torch.int32, device=device),
        cu_num_logits_np=np.array([0, 3, 5], dtype=np.int32),
        input_ids=torch.tensor([99, 11, 12, 88, 21], dtype=torch.int32, device=device),
        logits_indices=torch.arange(5, dtype=torch.int64, device=device),
        seq_lens=torch.tensor([10, 10], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([0, 1], dtype=torch.int32, device=device),
    )
    rejection_sampler = RejectionSampler.__new__(RejectionSampler)
    rejection_sampler.num_speculative_steps = 2
    rejection_sampler.synthetic_conditional_rates = None
    rejection_sampler.use_block_verification = False
    rejection_sampler.sampler = SimpleNamespace(
        req_states=SimpleNamespace(
            prefill_len=SimpleNamespace(
                gpu=torch.tensor([1, 1], dtype=torch.int32, device=device)
            )
        )
    )

    output = rejection_sampler.sample_from_greedy_tokens(
        torch.tensor(target_token_ids, dtype=torch.int64, device=device),
        input_batch,
    )
    torch.accelerator.synchronize()

    assert output.sampled_token_ids.tolist() == expected_tokens
    assert output.num_sampled.tolist() == expected_sampled
    assert output.num_rejected.tolist() == expected_rejected
