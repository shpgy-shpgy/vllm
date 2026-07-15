# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (
    _prepare_compact_rejection_indices_kernel,
)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    "num_draft_tokens_per_req",
    [
        [2],
        [2, 0, 2],
        [4, 1, 0, 3],
        [0, 0],
        [2] * 64,
        [(req_idx * 3) % 5 for req_idx in range(64)],
    ],
)
def test_prepare_compact_rejection_indices(
    num_draft_tokens_per_req: list[int],
):
    num_logits_per_req = torch.tensor(num_draft_tokens_per_req, dtype=torch.int32) + 1
    cu_num_logits = torch.cat(
        (torch.zeros(1, dtype=torch.int32), num_logits_per_req.cumsum(0))
    ).cuda()

    expected_target_indices: list[int] = []
    expected_bonus_indices: list[int] = []
    start = 0
    for num_draft_tokens in num_draft_tokens_per_req:
        expected_target_indices.extend(range(start, start + num_draft_tokens))
        start += num_draft_tokens + 1
        expected_bonus_indices.append(start - 1)

    target_indices = torch.empty(
        sum(num_draft_tokens_per_req), dtype=torch.int64, device="cuda"
    )
    bonus_indices = torch.empty(
        len(num_draft_tokens_per_req), dtype=torch.int64, device="cuda"
    )
    _prepare_compact_rejection_indices_kernel[(len(num_draft_tokens_per_req),)](
        cu_num_logits,
        target_indices,
        bonus_indices,
    )

    assert target_indices.cpu().tolist() == expected_target_indices
    assert bonus_indices.cpu().tolist() == expected_bonus_indices
