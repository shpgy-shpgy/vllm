# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.input_batch import post_update
from vllm.v1.worker.gpu.sample.penalties import bincount


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_unique_output_tokens_are_rebuilt_and_updated() -> None:
    device = torch.device("cuda")
    request_indices = torch.tensor([2, 0], dtype=torch.int32, device=device)
    all_token_ids = torch.zeros((3, 12), dtype=torch.int32, device=device)
    all_token_ids[2, :7] = torch.tensor(
        [1, 2, 3, 3, 5, 7, 5], dtype=torch.int32, device=device
    )
    all_token_ids[0, :4] = torch.tensor([4, 9, 10, 9], dtype=torch.int32, device=device)
    prompt_len = torch.tensor([1, 0, 2], dtype=torch.int32, device=device)
    prefill_len = torch.tensor([4, 0, 7], dtype=torch.int32, device=device)
    prompt_bin_mask = torch.zeros((3, 1), dtype=torch.int32, device=device)
    output_bin_counts = torch.zeros((3, 16), dtype=torch.int32, device=device)
    unique_ids = torch.empty((3, 12), dtype=torch.int32, device=device)
    num_unique = torch.zeros(3, dtype=torch.int32, device=device)

    bincount(
        request_indices,
        all_token_ids,
        prompt_len,
        prefill_len,
        prompt_bin_mask,
        output_bin_counts,
        unique_ids,
        num_unique,
        max_prefill_len=7,
    )
    torch.accelerator.synchronize()

    assert output_bin_counts[2, [3, 5, 7]].tolist() == [2, 2, 1]
    assert output_bin_counts[0, [9, 10]].tolist() == [2, 1]
    assert num_unique[[2, 0]].tolist() == [3, 2]
    assert set(unique_ids[2, :3].tolist()) == {3, 5, 7}
    assert set(unique_ids[0, :2].tolist()) == {9, 10}

    post_update(
        request_indices,
        torch.zeros(3, dtype=torch.int32, device=device),
        torch.zeros((3, 1), dtype=torch.int64, device=device),
        output_bin_counts,
        unique_ids,
        num_unique,
        torch.ones(3, dtype=torch.float32, device=device),
        torch.tensor([[7, 8], [10, 11]], dtype=torch.int64, device=device),
        torch.tensor([2, 2], dtype=torch.int32, device=device),
        torch.zeros(2, dtype=torch.int32, device=device),
        None,
        all_token_ids,
        torch.tensor([4, 0, 7], dtype=torch.int32, device=device),
    )
    torch.accelerator.synchronize()

    assert output_bin_counts[2, [7, 8]].tolist() == [2, 1]
    assert output_bin_counts[0, [10, 11]].tolist() == [2, 1]
    assert num_unique[[2, 0]].tolist() == [4, 3]
    assert set(unique_ids[2, :4].tolist()) == {3, 5, 7, 8}
    assert set(unique_ids[0, :3].tolist()) == {9, 10, 11}

    # Rows without a presence penalty still need dense counts for the normal
    # sampler, but must not pay to maintain the hybrid sparse-id table.
    post_update(
        request_indices,
        torch.zeros(3, dtype=torch.int32, device=device),
        torch.zeros((3, 1), dtype=torch.int64, device=device),
        output_bin_counts,
        unique_ids,
        num_unique,
        torch.zeros(3, dtype=torch.float32, device=device),
        torch.tensor([[12], [13]], dtype=torch.int64, device=device),
        torch.ones(2, dtype=torch.int32, device=device),
        torch.zeros(2, dtype=torch.int32, device=device),
        None,
        all_token_ids,
        torch.tensor([6, 0, 9], dtype=torch.int32, device=device),
    )
    torch.accelerator.synchronize()

    assert output_bin_counts[2, 12].item() == 1
    assert output_bin_counts[0, 13].item() == 1
    assert num_unique[[2, 0]].tolist() == [4, 3]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_post_update_without_sparse_presence_state() -> None:
    device = torch.device("cuda")
    output_bin_counts = torch.zeros((1, 16), dtype=torch.int32, device=device)
    all_token_ids = torch.zeros((1, 4), dtype=torch.int32, device=device)

    post_update(
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros((1, 1), dtype=torch.int64, device=device),
        output_bin_counts,
        None,
        None,
        None,
        torch.tensor([[5]], dtype=torch.int64, device=device),
        torch.ones(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        None,
        all_token_ids,
        torch.zeros(1, dtype=torch.int32, device=device),
    )
    torch.accelerator.synchronize()

    assert output_bin_counts[0, 5].item() == 1
