# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed BA/QKVZ fork/join for the SM120 TP1 Qwen3.5-4B profile."""

import torch

from vllm import envs
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

ROWS = (1, 2, 4, 8, 16, 24, 32, 64)
_REGISTERED = False


def project(layer, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stream = layer._mach_projection_stream
    if (
        x.ndim != 2
        or x.shape[0] not in ROWS
        or x.shape[1] != 2560
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
    ):
        return layer.in_proj_qkvz(x)[0], layer.in_proj_ba(x)[0]

    origin = torch.cuda.current_stream(x.device)
    stream.wait_stream(origin)
    # Join even if a projection raises, so the caller can safely release x.
    try:
        with torch.cuda.stream(stream):
            ba = layer.in_proj_ba(x)[0]
        qkvz = layer.in_proj_qkvz(x)[0]
    finally:
        origin.wait_stream(stream)
    # Eager allocations may outlive this call on either stream. Graph capture
    # records the same fork/join edges and owns its allocation pool separately.
    x.record_stream(stream)
    ba.record_stream(origin)
    return qkvz, ba


def _impl(x: torch.Tensor, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    layer = get_forward_context().no_compile_layers[layer_name]
    return project(layer, x)


def _fake(x: torch.Tensor, layer_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty((x.shape[0], 12288)), x.new_empty((x.shape[0], 64))


def prepare_model(model, config) -> int:
    if not envs.VLLM_MXFP6_SM120_PROJECTION_PARALLEL:
        return 0
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    from .native import _eligible_config, _eligible_layer

    if not _eligible_config(model, config):
        return 0
    layers = [m for m in model.modules() if isinstance(m, QwenGatedDeltaNetAttention)]
    if len(layers) != 24:
        return 0
    for layer in layers:
        q, b = layer.in_proj_qkvz, layer.in_proj_ba
        if (
            layer.tp_size != 1
            or not _eligible_layer(q)
            or tuple(q.weight.shape) != (12288, 2560)
            or tuple(b.weight.shape) != (64, 2560)
            or b.weight.dtype != torch.bfloat16
            or b.weight.device != q.weight.device
            or not isinstance(b.quant_method, UnquantizedLinearMethod)
            or q.bias is not None
            or b.bias is not None
        ):
            return 0
    if any(getattr(layer, "_mach_projection_parallel", False) for layer in layers):
        return 0

    global _REGISTERED
    if not _REGISTERED:
        direct_register_custom_op(
            op_name="mach_mxfp8_parallel_projection",
            op_func=_impl,
            mutates_args=[],
            fake_impl=_fake,
        )
        _REGISTERED = True
    device = layers[0].in_proj_ba.weight.device
    stream = torch.cuda.Stream(device=device)
    origin = torch.cuda.current_stream(device)
    stream.wait_stream(origin)
    # Initialize BA kernels and cuBLAS state on the stream used during capture.
    with torch.cuda.stream(stream):
        for rows in ROWS:
            x = torch.zeros((rows, 2560), device=device, dtype=torch.bfloat16)
            for layer in layers[1:]:
                layer.in_proj_ba(x)
    origin.wait_stream(stream)
    for layer in layers[1:]:
        layer._mach_projection_stream = stream
        layer._mach_projection_parallel = True
    init_logger(__name__).info(
        "Prepared BA/QKVZ parallel projections for 23 GDN layers, rows=%s", ROWS
    )
    return 23
