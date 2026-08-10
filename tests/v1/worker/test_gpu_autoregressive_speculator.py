# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils


class _TestSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError


class _DraftModel(torch.nn.Module):
    def __init__(self, output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]):
        super().__init__()
        self.output = output

    def forward(self, **kwargs):
        return self.output


def _make_speculator(
    monkeypatch,
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> _TestSpeculator:
    monkeypatch.setattr(
        spec_module,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    speculator = object.__new__(_TestSpeculator)
    speculator.supports_mm_inputs = False
    speculator.vllm_config = None
    speculator.input_buffers = SimpleNamespace(
        input_ids=torch.arange(4),
        positions=torch.arange(4),
    )
    speculator.hidden_states = torch.zeros(4, 3)
    speculator.model = _DraftModel(output)
    return speculator


def test_run_model_unpacks_tuple_return_for_mtp(monkeypatch):
    logits_hidden = torch.full((4, 3), 1.0)
    feedback_hidden = torch.full((4, 3), 2.0)
    speculator = _make_speculator(monkeypatch, (logits_hidden, feedback_hidden))

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is logits_hidden
    assert actual_feedback_hidden is feedback_hidden


def test_run_model_reuses_tensor_return_for_mtp(monkeypatch):
    hidden = torch.full((4, 3), 1.0)
    speculator = _make_speculator(monkeypatch, hidden)

    actual_logits_hidden, actual_feedback_hidden = speculator._run_model(
        4,
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_across_dp=None,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert actual_logits_hidden is hidden
    assert actual_feedback_hidden is hidden


def test_eagle_loader_shares_inner_language_model_lm_head(monkeypatch):
    target_language_model = torch.nn.Module()
    target_language_model.model = torch.nn.Module()
    target_language_model.model.embed_tokens = torch.nn.Module()
    target_language_model.lm_head = torch.nn.Module()

    class _TargetWrapper(torch.nn.Module):
        def get_language_model(self):
            return target_language_model

    draft_model = torch.nn.Module()
    draft_model.model = torch.nn.Module()
    draft_model.model.embed_tokens = torch.nn.Module()
    draft_model.lm_head = torch.nn.Module()
    old_draft_lm_head = draft_model.lm_head
    old_draft_lm_head.register_buffer(
        "_hybrid_mxfp8_lm_head_weight",
        torch.empty((8, 4), dtype=torch.float8_e4m3fn),
        persistent=False,
    )
    old_draft_lm_head.register_buffer(
        "_hybrid_mxfp8_lm_head_scale",
        torch.empty((32,), dtype=torch.uint8),
        persistent=False,
    )
    old_draft_lm_head._hybrid_mxfp8_lm_head_state = object()

    monkeypatch.setattr(eagle_utils, "get_model", lambda **kwargs: draft_model)
    monkeypatch.setattr(
        eagle_utils,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=1),
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=object())
    )

    loaded = eagle_utils.load_eagle_model(_TargetWrapper(), vllm_config)

    assert loaded.lm_head is target_language_model.lm_head
    assert loaded.model.embed_tokens is target_language_model.model.embed_tokens
    assert not hasattr(old_draft_lm_head, "_hybrid_mxfp8_lm_head_state")
    assert not hasattr(old_draft_lm_head, "_hybrid_mxfp8_lm_head_weight")
    assert not hasattr(old_draft_lm_head, "_hybrid_mxfp8_lm_head_scale")
