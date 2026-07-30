# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.argmax_triton import local_argmax_triton


@pytest.mark.parametrize(
    ("active_vocab_size", "vocab_start"),
    [
        (248320, 0),
        (124160, 124160),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_local_argmax_all_nan_row_stays_in_active_vocab(
    active_vocab_size: int,
    vocab_start: int,
) -> None:
    logits = torch.full(
        (1, active_vocab_size),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )

    values, token_ids = local_argmax_triton(
        logits,
        vocab_start=vocab_start,
        active_vocab_size=active_vocab_size,
    )

    assert torch.isneginf(values).all()
    assert token_ids.item() == vocab_start
    assert token_ids.item() < vocab_start + active_vocab_size
