# SPDX-License-Identifier: Apache-2.0
"""Functional AR/Norm boundary used while compiling Qwen3.5 MoE."""

from __future__ import annotations

import torch


@torch.library.custom_op("vllm::mach_allreduce_gemma_rms_norm", mutates_args=())
def compiled_allreduce_gemma_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compile-safe functional wrapper around FlashInfer all-reduce fusion.

    The eager helper uses an in-place residual buffer, which is correct for the
    serving path but is not a functional boundary for Dynamo.  This custom op
    returns two fresh tensors so the compiler can keep the MoE output and the
    following RMSNorm in the same graph without observing workspace aliases.
    """

    from vllm import ir
    from vllm.compilation.passes.fusion.allreduce_rms_fusion import (
        PDL_ADVANCE_LAUNCH_TOKENS,
        _select_flashinfer_allreduce_use_oneshot,
    )
    from vllm.distributed import tensor_model_parallel_all_reduce
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )
    from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
        _AR_RESIDUAL_RMS_NORM,
        _can_use_flashinfer,
        flashinfer_comm,
        get_fi_ar_workspace,
    )
    from vllm.platforms import current_platform

    tp_size = get_tensor_model_parallel_world_size()
    ok, max_tokens = _can_use_flashinfer(hidden_states, tp_size)
    if not ok or tp_size == 1:
        reduced = (
            tensor_model_parallel_all_reduce(hidden_states)
            if tp_size > 1
            else hidden_states
        )
        return ir.ops.fused_add_rms_norm(
            reduced.clone(), residual.clone(), weight.float() + 1.0, epsilon
        )

    workspace = get_fi_ar_workspace(
        world_size=tp_size,
        rank=get_tensor_model_parallel_rank(),
        max_token_num=max_tokens,
        hidden_dim=hidden_states.shape[-1],
        dtype=hidden_states.dtype,
        group=get_tp_group().cpu_group,
    )
    if workspace is None or flashinfer_comm is None or _AR_RESIDUAL_RMS_NORM is None:
        reduced = tensor_model_parallel_all_reduce(hidden_states)
        return ir.ops.fused_add_rms_norm(
            reduced.clone(), residual.clone(), weight.float() + 1.0, epsilon
        )

    capability = current_platform.get_device_capability()
    oneshot = _select_flashinfer_allreduce_use_oneshot(
        workspace.backend,
        capability.to_int() if capability is not None else None,
        tp_size,
        hidden_states.numel() * hidden_states.element_size(),
    )
    norm_out = torch.empty_like(hidden_states)
    residual_out = torch.empty_like(hidden_states)
    flashinfer_comm.allreduce_fusion(
        input=hidden_states,
        workspace=workspace,
        pattern=_AR_RESIDUAL_RMS_NORM,
        launch_with_pdl=True,
        output=None,
        residual_out=residual_out,
        norm_out=norm_out,
        residual_in=residual,
        rms_gamma=weight,
        rms_eps=epsilon,
        use_oneshot=oneshot,
        fp32_acc=True,
        weight_bias=1.0,
        trigger_completion_at_end=(oneshot is True)
        or hidden_states.shape[0] > PDL_ADVANCE_LAUNCH_TOKENS,
    )
    return norm_out, residual_out


@compiled_allreduce_gemma_rms_norm.register_fake
def _fake_ar_norm(hidden_states, residual, weight, epsilon):
    del weight, epsilon
    return torch.empty_like(hidden_states), torch.empty_like(residual)


__all__ = ["compiled_allreduce_gemma_rms_norm"]
