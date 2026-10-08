# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
    swizzle_mxfp8_scale,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

# The native MXFP8 path has a fixed activation-quantization and launch cost.
# Keep the measured low-M decode range on the existing FlashInfer dispatcher;
# it selects the tuned B12x kernel for the Qwen3.5 shapes below M=32.
_MXFP6_NATIVE_MIN_M = 17
_MXFP8_NATIVE_CONFIG = (-1, 1, 1, 0)


@dataclass
class _WorkspaceEntry:
    # Retain the stream wrapper as well as the tensor.  PyTorch stream objects
    # own a small amount of lifetime state that must not be recycled while a
    # cached native workspace is still associated with that stream.
    workspace: torch.Tensor | None
    stream: torch.cuda.Stream


_NATIVE_LOCK = threading.RLock()
_NATIVE_WORKSPACES: dict[
    tuple[int, int, int, int, int, tuple[int, int, int, int]], _WorkspaceEntry
] = {}
_NATIVE_COUNTS: Counter[str] = Counter()


def load_mxfp6_sm120_mxfp8_libraries() -> None:
    """Load both native libraries before entering a compiled model graph."""
    import mxfp6.mxfp8 as mxfp8

    mxfp8.load_library()
    from mxfp6 import load_library as load_mxfp6_library

    load_mxfp6_library()

    # Keep the schema check next to the loader.  If another mxfp6/mxfp8
    # binary is accidentally picked up through LD_LIBRARY_PATH, fail before a
    # model graph is compiled instead of falling back to a lazy loader failure
    # from inside Dynamo.
    native = torch.ops.mxfp8_sm120
    gemm_schema = str(native.gemm.default._schema)
    workspace_schema = str(native.allocate_workspace.default._schema)
    required_gemm = (
        "Tensor a",
        "Tensor b",
        "Tensor sa",
        "Tensor sb",
        "Tensor(a!)? out",
        "Tensor? workspace",
        "int tactic",
        "int splits",
        "int swizzle",
        "int sms",
    )
    required_workspace = (
        "Tensor a",
        "Tensor b",
        "Tensor sa",
        "Tensor sb",
        "Tensor out",
        "int tactic",
        "int splits",
        "int swizzle",
        "int sms",
    )
    if not all(fragment in gemm_schema for fragment in required_gemm):
        raise RuntimeError(f"unexpected mxfp8_sm120::gemm schema: {gemm_schema}")
    if not all(fragment in workspace_schema for fragment in required_workspace):
        raise RuntimeError(
            f"unexpected mxfp8_sm120::allocate_workspace schema: {workspace_schema}"
        )


def _native_gemm_impl(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Launch the prequantized native GEMM through an opaque vLLM op.

    The lock, stream lookup and first-use allocation are intentionally inside
    this custom-op implementation.  Dynamo/Inductor only sees the fake
    implementation below, so a Python threading lock can never become part of
    the compiled model graph.
    """
    m, k = (int(value) for value in activation.shape)
    n = int(weight.shape[0])
    device_index = activation.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    stream = torch.cuda.current_stream(device_index)
    key = (device_index, int(stream.cuda_stream), m, n, k, _MXFP8_NATIVE_CONFIG)
    capturing = torch.cuda.is_current_stream_capturing()

    with _NATIVE_LOCK:
        entry = _NATIVE_WORKSPACES.get(key)
        if entry is None:
            if capturing:
                _NATIVE_COUNTS["capture_workspace_miss"] += 1
                raise RuntimeError(
                    "mxfp6-sm120 native workspace was not warmed for this "
                    "CUDA stream and shape before graph capture"
                )

            # Both the native library and the workspace allocator are loaded
            # before entering the compiled graph in the normal serving path.
            # Keeping this defensive call here also makes eager direct calls
            # safe; the call is hidden by the opaque custom-op boundary.
            load_mxfp6_sm120_mxfp8_libraries()
            output = torch.empty((m, n), device=activation.device, dtype=torch.bfloat16)
            workspace = torch.ops.mxfp8_sm120.allocate_workspace(
                activation,
                weight,
                activation_scale,
                weight_scale,
                output,
                *_MXFP8_NATIVE_CONFIG,
            )
            entry = _WorkspaceEntry(workspace=workspace, stream=stream)
            _NATIVE_WORKSPACES[key] = entry
            _NATIVE_COUNTS["workspace_miss"] += 1
        else:
            output = torch.empty((m, n), device=activation.device, dtype=torch.bfloat16)
            _NATIVE_COUNTS["workspace_hit"] += 1

    # Stream ordering makes reuse safe for calls on the same stream.  A
    # different CUDA stream gets a separate cache entry and workspace lane.
    torch.ops.mxfp8_sm120.gemm(
        activation,
        weight,
        activation_scale,
        weight_scale,
        output,
        entry.workspace,
        *_MXFP8_NATIVE_CONFIG,
    )
    with _NATIVE_LOCK:
        _NATIVE_COUNTS["native_calls"] += 1
        _NATIVE_COUNTS[
            "native_capture_calls" if capturing else "native_eager_calls"
        ] += 1
    return output


def _native_gemm_fake(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    del activation_scale, weight_scale
    # Keep M symbolic.  Converting activation.shape[0] to int here would
    # specialize the graph and defeat the runtime workspace key.
    return torch.empty(
        (activation.shape[0], weight.shape[0]),
        device=activation.device,
        dtype=torch.bfloat16,
    )


def _mxfp6_sm120_mxfp8_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    if input.shape[0] < _MXFP6_NATIVE_MIN_M:
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
            mxfp8_e4m3_quantize,
        )
        from vllm.utils import flashinfer as vllm_flashinfer

        input_mxfp8, input_scale = mxfp8_e4m3_quantize(
            input, is_sf_swizzled_layout=True
        )
        return vllm_flashinfer.mm_mxfp8(
            input_mxfp8,
            weight.t(),
            input_scale,
            weight_scale,
            out_dtype=torch.bfloat16,
            backend="b12x-decode",
        )

    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        mxfp8_e4m3_quantize,
    )

    input_mxfp8, input_scale = mxfp8_e4m3_quantize(input, is_sf_swizzled_layout=True)
    return torch.ops.vllm.mxfp6_sm120_native_mxfp8_gemm(
        input_mxfp8,
        weight,
        input_scale,
        weight_scale,
    )


def _mxfp6_sm120_mxfp8_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    del weight_scale
    return torch.empty(
        (input.shape[0], weight.shape[0]),
        dtype=torch.bfloat16,
        device=input.device,
    )


direct_register_custom_op(
    op_name="mxfp6_sm120_native_mxfp8_gemm",
    op_func=_native_gemm_impl,
    mutates_args=[],
    fake_impl=_native_gemm_fake,
)

direct_register_custom_op(
    op_name="mxfp6_sm120_mxfp8_gemm",
    op_func=_mxfp6_sm120_mxfp8_gemm_impl,
    mutates_args=[],
    fake_impl=_mxfp6_sm120_mxfp8_gemm_fake,
)


def warmup_mxfp6_sm120_mxfp8_workspaces(
    model: torch.nn.Module,
    sizes: tuple[int, ...] | list[int] | None = None,
) -> int:
    """Pre-allocate and first-launch native workspaces for model GEMM shapes.

    vLLM's normal cudagraph warmup also populates exact capture shapes.  This
    explicit pass handles the common dispatch-bin sizes before memory
    profiling, and the same cache then serves the later profile/capture pass.
    ``sizes`` is deliberately explicit so deployments can keep startup and
    persistent workspace memory bounded.
    """
    if not torch.cuda.is_available():
        return 0
    if sizes is None:
        sizes = (17, 32, 64, 128, 256, 512, 1024, 2048)
    warmup_sizes = tuple(
        sorted({int(m) for m in sizes if int(m) >= _MXFP6_NATIVE_MIN_M})
    )
    if not warmup_sizes:
        return 0

    by_shape: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in model.modules():
        weight = getattr(layer, "mxfp6_sm120_mxfp8_weight_values", None)
        weight_scale = getattr(layer, "mxfp6_sm120_mxfp8_weight_scales", None)
        if weight is None or weight_scale is None or not weight.is_cuda:
            continue
        n, k = (int(value) for value in weight.shape)
        device_index = weight.device.index
        if device_index is None:
            device_index = torch.accelerator.current_device_index()
        by_shape.setdefault((device_index, n, k), (weight, weight_scale))

    if not by_shape:
        return 0

    load_mxfp6_sm120_mxfp8_libraries()
    launches = 0
    for (device_index, n, k), (weight, weight_scale) in by_shape.items():
        for m in warmup_sizes:
            # The zeros are intentional: they make warmup deterministic and
            # avoid feeding uninitialized values through the quantizer while
            # still forcing both the quantizer and native first launch.
            input_ = torch.zeros(
                (m, k), device=torch.device("cuda", device_index), dtype=torch.bfloat16
            )
            torch.ops.vllm.mxfp6_sm120_mxfp8_gemm(input_, weight, weight_scale)
            launches += 1
    for device_index, _, _ in by_shape:
        torch.accelerator.synchronize(device_index)
    return launches


def native_mxfp8_workspace_stats() -> dict[str, object]:
    """Return lightweight diagnostics for startup receipts and benchmarks."""
    with _NATIVE_LOCK:
        workspaces: list[dict[str, Any]] = [
            {
                "device": key[0],
                "stream": key[1],
                "m": key[2],
                "n": key[3],
                "k": key[4],
                "config": key[5],
                "workspace_bytes": 0
                if entry.workspace is None
                else entry.workspace.numel() * entry.workspace.element_size(),
            }
            for key, entry in _NATIVE_WORKSPACES.items()
        ]
        return {
            "counts": dict(_NATIVE_COUNTS),
            "workspaces": workspaces,
            "workspace_bytes_total": sum(
                int(item["workspace_bytes"]) for item in workspaces
            ),
        }


class Mxfp6Sm120Mxfp8LinearKernel(Mxfp8LinearKernel):
    """SM120 MXFP8 W8A8 GEMM with a low-M FlashInfer fallback."""

    @staticmethod
    def _load_native():
        load_mxfp6_sm120_mxfp8_libraries()
        import mxfp6.mxfp8 as mxfp8

        return mxfp8

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "mxfp6-sm120 MXFP8 requires CUDA"
        if not current_platform.is_device_capability(120):
            return False, "mxfp6-sm120 MXFP8 requires an SM120 device"
        try:
            cls._load_native()
        except (ImportError, OSError, RuntimeError, AttributeError) as e:
            return False, f"mxfp6-sm120 MXFP8 library is unavailable: {e}"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        del c
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._load_native()

        weight = layer.weight.data
        if weight.dtype != MXFP8_VALUE_DTYPE:
            raise ValueError(
                "mxfp6-sm120 MXFP8 requires "
                f"{MXFP8_VALUE_DTYPE} weights, got {weight.dtype}"
            )
        if weight.ndim != 2:
            raise ValueError(
                f"mxfp6-sm120 MXFP8 requires 2D weights, got {weight.ndim}D"
            )

        out_features, in_features = map(int, weight.shape)
        if out_features % 128 or in_features % 128:
            raise ValueError(
                "mxfp6-sm120 MXFP8 requires N/K to be multiples of 128, "
                f"got N={out_features}, K={in_features}"
            )
        if in_features % MXFP8_BLOCK_SIZE:
            raise ValueError(
                "mxfp6-sm120 MXFP8 requires K to be divisible by "
                f"{MXFP8_BLOCK_SIZE}, got K={in_features}"
            )

        weight_scale = layer.weight_scale.data
        if weight_scale.dtype != MXFP8_SCALE_DTYPE or weight_scale.ndim != 2:
            raise ValueError(
                "mxfp6-sm120 MXFP8 requires a 2D uint8 weight_scale, "
                f"got {weight_scale.dtype} {tuple(weight_scale.shape)}"
            )

        scale_k = in_features // MXFP8_BLOCK_SIZE
        weight_scale = weight_scale[:out_features, :scale_k].contiguous()
        weight_scale = swizzle_mxfp8_scale(weight_scale, M=out_features, K=in_features)

        weight = weight.contiguous()
        layer.weight = Parameter(weight, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        # Keep the native operands around for the medium/large-M branch. The
        # low-M branch reuses the same weight and scale tensors with FlashInfer.
        layer.mxfp6_sm120_mxfp8_weight_values = layer.weight.view(torch.float8_e4m3fn)
        layer.mxfp6_sm120_mxfp8_weight_scales = layer.weight_scale

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"mxfp6-sm120 MXFP8 requires FP16/BF16 activations, got {x.dtype}"
            )

        _, in_features = layer.weight.shape
        input_shape = x.shape
        input_2d = x.reshape(-1, in_features).contiguous()
        if input_2d.shape[0] == 0:
            output = input_2d.new_empty((0, layer.weight.shape[0]))
            output = output.view(*input_shape[:-1], layer.weight.shape[0])
        else:
            # Both native libraries were loaded during weight processing. Use a
            # vLLM custom op with a fake implementation so Dynamo never traces
            # the optional package's lazy loader or its threading lock. The op
            # also dispatches low-M decode shapes to FlashInfer B12x based on
            # the runtime shape, which avoids freezing the choice in Dynamo.
            output = torch.ops.vllm.mxfp6_sm120_mxfp8_gemm(
                input_2d,
                layer.mxfp6_sm120_mxfp8_weight_values,
                layer.mxfp6_sm120_mxfp8_weight_scales,
            )
            if output.dtype != x.dtype:
                output = output.to(x.dtype)
            output = output.view(*input_shape[:-1], layer.weight.shape[0])

        if bias is not None:
            output = output + bias
        return output
