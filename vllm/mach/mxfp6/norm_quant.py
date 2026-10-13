# SPDX-License-Identifier: Apache-2.0
"""Post-attention norm/quant/GateUp: automatic for eligible 4B, opt-in for 2B."""

import types

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

_REGISTERED = False


@torch.compile(fullgraph=True, dynamic=True)
def _prefill_norm(x, residual, gamma, epsilon):
    values = x.float() + residual.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + epsilon)
    return (normalized * gamma).to(x.dtype), values.to(x.dtype)


def gateup(
    x: torch.Tensor,
    residual: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float,
    weight: torch.Tensor,
    scales: torch.Tensor,
    two_b: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        fused_add_rms_norm_mxfp8_quant,
        mxfp8_e4m3_quantize,
    )

    # Keep the row policy opaque: Dynamo first traces a large prefill and
    # otherwise removes the decode branch from the shared compiled graph.
    if 1 <= x.shape[0] <= 64:
        values, sf, residual = fused_add_rms_norm_mxfp8_quant(
            x, residual, gamma, epsilon
        )
    else:
        normalized, residual = _prefill_norm(x, residual, gamma, epsilon)
        values, sf = mxfp8_e4m3_quantize(normalized, is_sf_swizzled_layout=True)
    if two_b:
        from .two_b_native import gemm

        out = gemm(values, weight, sf, scales)
    else:
        out = torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm(values, weight, sf, scales)
    return out, residual


def register():
    global _REGISTERED
    if _REGISTERED:
        return

    def fake(x, residual, gamma, epsilon, weight, scales, two_b):
        return x.new_empty((x.shape[0], weight.shape[0])), torch.empty_like(residual)

    direct_register_custom_op(
        op_name="mach_norm_quant_gateup",
        op_func=gateup,
        mutates_args=[],
        fake_impl=fake,
    )
    _REGISTERED = True


def prepare_model(model, config):
    from .auto_defaults import model_size

    if (
        not envs.VLLM_MACH_MXFP8_NORM_QUANT
        or model_size(config) is None
        or envs.VLLM_MXFP6_SM120_MXFP8_DUAL
    ):
        return 0
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer

    count = 0
    for layer in model.modules():
        if (
            type(layer) is not Qwen3_5DecoderLayer
            or getattr(layer, "_mach_norm_quant", False)
            or layer.layer_scale
            or layer.use_attn_reduce_scatter_for_moe
            or layer.use_fused_ar_gemma_norm
            or not getattr(layer.mlp, "_mach_swiglu_quant", False)
            or type(layer.post_attention_layernorm) is not GemmaRMSNorm
        ):
            continue
        up, down = layer.mlp.gate_up_proj, layer.mlp.down_proj
        two_b = getattr(up, "_mach_2b_native", False)
        tag = "_mach_2b_native" if two_b else "_mach_4b_native"
        if not getattr(up, tag, False) or not getattr(down, tag, False):
            continue
        register()
        layer.post_attention_layernorm._mach_gamma = (
            layer.post_attention_layernorm.weight.detach().float().add(1).contiguous()
        )
        original = layer.forward

        def forward(
            this,
            hidden_states,
            residual,
            positions=None,
            _original=original,
            _two_b=two_b,
            **kwargs,
        ):
            if (
                hidden_states.ndim != 2
                or hidden_states.dtype != torch.bfloat16
                or not hidden_states.is_cuda
            ):
                return _original(hidden_states, residual, positions, **kwargs)
            if residual is None:
                residual = hidden_states
                hidden_states = this.input_layernorm(hidden_states)
            else:
                hidden_states, residual = this.input_layernorm(hidden_states, residual)
            if this.layer_type == "linear_attention":
                hidden_states = this.linear_attn(hidden_states=hidden_states)
            else:
                hidden_states = this.self_attn(
                    hidden_states=hidden_states, positions=positions
                )
            norm = this.post_attention_layernorm
            up, down = this.mlp.gate_up_proj, this.mlp.down_proj
            hidden_states, residual = torch.ops.vllm.mach_norm_quant_gateup(
                hidden_states,
                residual,
                norm._mach_gamma,
                norm.variance_epsilon,
                up.weight,
                up.weight_scale,
                _two_b,
            )
            hidden_states = torch.ops.vllm.mach_mxfp8_swiglu_down(
                hidden_states,
                down.weight,
                down.weight_scale,
                "mach-2b-native" if _two_b else "mach-4b-native",
                False,
            )
            return hidden_states, residual

        layer.forward = types.MethodType(forward, layer)
        layer._mach_norm_quant = True
        count += 1
    init_logger(__name__).info(
        "Prepared native norm/quant/GateUp fusion: %d layers", count
    )
    return count
