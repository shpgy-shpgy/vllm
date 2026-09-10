# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental single-request split-KV attention with persistent scratch."""

import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention


class SmallBatchAttention:
    def __init__(self) -> None:
        self.buffers: dict[tuple[int, torch.device], tuple[torch.Tensor, ...]] = {}

    def __call__(
        self,
        q: torch.Tensor,
        kv: tuple[torch.Tensor, ...],
        out: torch.Tensor,
        table: torch.Tensor,
        lengths: torch.Tensor,
        scale: float,
        layout: str = "HND",
    ) -> None:
        assert layout in ("NHD", "HND")
        assert lengths.numel() == 1 and table.shape[0] == 1
        tokens, heads, dim = q.shape
        assert tokens in (1, 5) and heads == 16 and dim == 256
        key = (tokens, q.device)
        if key not in self.buffers:
            self.buffers[key] = (
                torch.tensor([0, tokens], dtype=torch.int32, device=q.device),
                torch.empty(
                    tokens, heads, 32, dim, dtype=torch.float32, device=q.device
                ),
                torch.empty(tokens, heads, 32, dtype=torch.float32, device=q.device),
                torch.empty(tokens, heads, 32, dtype=torch.float32, device=q.device),
            )
        cu, partial, maxima, sums = self.buffers[key]
        k, v = kv if layout == "NHD" else (kv[0].transpose(1, 2), kv[1].transpose(1, 2))
        unified_attention(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu,
            max_seqlen_q=tokens,
            seqused_k=lengths,
            max_seqlen_k=table.shape[1] * k.shape[1],
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=table,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            seq_threshold_3D=1,
            num_par_softmax_segments=32,
            softmax_segm_output=partial,
            softmax_segm_max=maxima,
            softmax_segm_expsum=sums,
            split_kv_query_len_threshold=5,
            split_kv_tile_size=32,
        )
