# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 2B/4B SwiGLU quantization ported from vllm-mach e5a444e.

Preserves upstream's single BF16 rounding, CUDA E8M0 conversion and M16/24
PDL policy. Explicit MLP wiring replaces upstream's Inductor source rewriting;
only an unmodified SiluAndMul with a single MXFP8 down-projection is eligible.
"""

import types
from collections import Counter

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)
_REGISTERED = False
_COUNTS = Counter()


def receipt() -> dict:
    """Report eager/capture construction calls, not CUDA Graph replay counts."""
    return {
        "calls": [
            {"capturing": capturing, "rows": rows, "width": width, "count": count}
            for (capturing, rows, width), count in sorted(_COUNTS.items())
        ]
    }


@torch.compile(fullgraph=True, dynamic=True)
def _unfused_producer(x):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate) * up


@triton.jit
def _kernel(X, Y, S, K: tl.constexpr, GROUPS: tl.constexpr, PDL: tl.constexpr):
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0)
    groups = tl.program_id(1) * GROUPS + tl.arange(0, GROUPS)
    cols = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    gate = tl.load(X + row * 2 * K + cols, cols < K, 0).to(tl.float32)
    up = tl.load(X + row * 2 * K + K + cols, cols < K, 0).to(tl.float32)
    values = (gate / (1.0 + tl.extra.cuda.libdevice.exp(-gate))) * up
    values = values.to(tl.bfloat16).to(tl.float32)
    amax = tl.max(tl.abs(values), 1)
    rcp_max = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[tl.full((), 448.0, tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    packed = tl.inline_asm_elementwise(
        "{cvt.rp.satfinite.ue8m0x2.f32 $0, 0f00000000, $1;}",
        constraints="=h,f",
        args=[amax * rcp_max],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    )
    scale = packed.to(tl.uint32) & 255
    descale = (scale << 23).to(tl.float32, bitcast=True)
    reciprocal = tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[descale],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    inverse = tl.where(scale > 0, reciprocal, 0.0)
    quant = tl.minimum(tl.maximum(values * inverse[:, None], -448.0), 448.0)
    tl.store(Y + row * K + cols, quant.to(tl.float8e4nv), cols < K)
    tiles: tl.constexpr = (K // 32 + 3) // 4
    offset = (
        ((row // 128 * tiles + groups // 4) * 32 + row % 32) * 4 + row % 128 // 32
    ) * 4 + groups % 4
    tl.store(S + offset, scale.to(tl.uint8), groups < K // 32)


def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 payload and swizzled scales; padding scales are unspecified."""
    if (
        x.ndim != 2
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.shape[1] not in (12288, 18432)
        or not 1 <= x.shape[0] <= 128
    ):
        raise ValueError("Expected contiguous CUDA BF16 2B/4B gate/up, M=1..128")
    m, width = x.shape
    _COUNTS[(torch.cuda.is_current_stream_capturing(), m, width)] += 1
    k = width // 2
    y = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (triton.cdiv(m, 128) * 128 * triton.cdiv(k // 32, 4) * 4,),
        device=x.device,
        dtype=torch.uint8,
    )
    groups = 8 if m <= 4 else 32
    pdl = m in (16, 24)
    _kernel[(m, triton.cdiv(k // 32, groups))](
        x,
        y,
        scales,
        k,
        groups,
        pdl,
        num_warps=4,
        launch_pdl=pdl,
    )
    return y, scales


def _down_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    backend: str,
    use_native: bool,
) -> torch.Tensor:
    from vllm.utils.flashinfer import mm_mxfp8

    # Runtime M must stay inside the opaque op: Dynamo initially traces prefill.
    if 1 <= x.shape[0] <= 128 and x.is_contiguous():
        values, sf = quantize(x)
    else:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )

        values, sf = mxfp8_e4m3_quantize(
            _unfused_producer(x),
            is_sf_swizzled_layout=True,
        )
    if backend == "mach-2b-native":
        from .two_b_native import gemm

        if envs.VLLM_MACH_MXFP8_2B_PDL:
            from .two_b_pdl import record_swiglu

            record_swiglu(int(x.shape[0]))
        return gemm(values, weight, sf, scales)
    if backend == "mach-4b-native":
        return torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm(
            values, weight, sf, scales
        )
    if use_native and 1 <= x.shape[0] <= 64:
        return torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm_pdl(
            values,
            weight,
            sf,
            scales,
        )
    return mm_mxfp8(values, weight.t(), sf, scales, out_dtype=x.dtype, backend=backend)


def _down_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    backend: str,
    use_native: bool,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def prepare_model(model, config) -> int:
    """Install opt-in fusion without extending the 4B native GEMM policy to 2B."""
    global _REGISTERED
    if not envs.VLLM_MACH_MXFP8_SWIGLU_QUANT:
        return 0
    from vllm.model_executor.kernels.linear.mxfp8.flashinfer import (
        FlashInferCutlassMxfp8LinearKernel,
    )
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

    from .native import _kernel as linear_kernel

    mc = config.model_config
    if (
        mc is None
        or not current_platform.is_cuda()
        or not current_platform.is_device_capability(120)
        or not config.use_v2_model_runner
        or config.lora_config is not None
        or config.speculative_config is not None
        or config.parallel_config.tensor_parallel_size != 1
        or config.kernel_config.linear_backend not in ("auto", "flashinfer_cutlass")
        or mc.dtype != torch.bfloat16
        or mc.quantization != "compressed-tensors"
    ):
        return 0
    hf = mc.hf_text_config
    shape = (getattr(hf, "hidden_size", None), getattr(hf, "intermediate_size", None))
    if getattr(hf, "model_type", None) != "qwen3_5_text" or shape not in (
        (2048, 6144),
        (2560, 9216),
    ):
        return 0
    count = 0
    for mlp in model.modules():
        if (
            type(mlp) is not Qwen2MoeMLP
            or mlp.expert_gate is not None
            or type(mlp.act_fn) is not SiluAndMul
            or getattr(mlp, "_mach_swiglu_quant", False)
        ):
            continue
        up, down = mlp.gate_up_proj, mlp.down_proj
        if (
            type(up) is not MergedColumnParallelLinear
            or type(down) is not RowParallelLinear
            or up.tp_size != 1
            or down.tp_size != 1
            or not down.input_is_parallel
            or up.bias is not None
            or down.bias is not None
            or tuple(up.weight.shape) != (shape[1] * 2, shape[0])
            or tuple(down.weight.shape) != shape
            or type(linear_kernel(down)) is not FlashInferCutlassMxfp8LinearKernel
            or down.weight.dtype != torch.float8_e4m3fn
            or down.weight_scale.dtype != torch.uint8
            or not up.weight.is_cuda
            or not down.weight.is_cuda
            or not down.weight.is_contiguous()
            or down.weight_scale.device != down.weight.device
            or not down.weight_scale.is_contiguous()
            or down.weight_scale.numel() < shape[0] * (shape[1] // 32)
        ):
            continue
        if not _REGISTERED:
            direct_register_custom_op(
                op_name="mach_mxfp8_swiglu_down",
                op_func=_down_impl,
                mutates_args=[],
                fake_impl=_down_fake,
            )
            _REGISTERED = True
        original = mlp.forward

        def forward(this, x, _original=original):
            if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda:
                return _original(x)
            gate_up, _ = this.gate_up_proj(x)
            down = this.down_proj
            return torch.ops.vllm.mach_mxfp8_swiglu_down(
                gate_up,
                down.weight,
                down.weight_scale,
                (
                    "mach-2b-native"
                    if getattr(down, "_mach_2b_native", False)
                    else "mach-4b-native"
                    if getattr(down, "_mach_4b_native", False)
                    else down.mxfp8_backend
                ),
                getattr(this, "_mxfp6_sm120_fused_swiglu_prepared", False),
            )

        mlp.forward = types.MethodType(forward, mlp)
        mlp._mach_swiglu_quant = True
        count += 1
    logger.info(
        "Enabled upstream MXFP8 SwiGLU quantization for %d MLPs (%s)", count, shape
    )
    return count
