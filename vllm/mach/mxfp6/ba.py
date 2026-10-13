# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional Qwen3.5 BA-only BF16 small-M GEMV adapter.

The adapter is deliberately shape restricted to the TP1 Qwen3.5-4B BA
projection ``[64, 2560]`` and opt-in 2B ``[32, 2048]``. It preserves the
original linear method for every
other projection, dtype, bias, and M.  The private weight clone is required by
the raw b12x launch path on deployments where loader-owned parameter storage
is not safe for the CuTe kernel.
"""

from __future__ import annotations

from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_BA_SHAPE = (64, 2560)
_BA_2B_SHAPE = (32, 2048)
_SMALL_M_MAX = 8
_INSTALLED = False
_GEMV: Any = None
_SMALL_OP: Any = None
_ORIGINAL_APPLY: Any = None
_COUNTS: dict[int, Counter[str]] = {}
_M16: Any = None


def _small_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    private_weight: torch.Tensor,
) -> torch.Tensor:
    m = int(x.shape[0])
    bucket = _COUNTS.setdefault(m, Counter())
    # vllm-mach db2131e: only 2B M16 uses the native BF16 BA kernel.
    if (
        m == 16
        and tuple(weight.shape) == _BA_2B_SHAPE
        and _M16 is not None
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    ):
        bucket["native_m16_calls"] += 1
        return _M16(x, weight)
    if 1 <= m <= _SMALL_M_MAX and _GEMV is not None:
        bucket[
            "capture_calls"
            if torch.cuda.is_current_stream_capturing()
            else "eager_calls"
        ] += 1
        return _GEMV.bf16_gemv_small_n(x, private_weight)
    bucket["fallback_calls"] += 1
    return torch.nn.functional.linear(x, weight)


def _small_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    private_weight: torch.Tensor,
) -> torch.Tensor:
    del private_weight
    return x.new_empty((x.shape[0], weight.shape[0]))


def _candidate_layers(
    model: torch.nn.Module, include_2b: bool = False
) -> list[tuple[str, torch.nn.Module]]:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    candidates: list[tuple[str, torch.nn.Module]] = []
    for name, layer in model.named_modules():
        if not name.endswith("in_proj_ba"):
            continue
        weight = getattr(layer, "weight", None)
        shapes = (_BA_SHAPE, _BA_2B_SHAPE) if include_2b else (_BA_SHAPE,)
        if weight is None or tuple(weight.shape) not in shapes:
            continue
        if weight.dtype != torch.bfloat16 or not weight.is_cuda:
            continue
        if getattr(layer, "bias", None) is not None:
            continue
        if not isinstance(
            getattr(layer, "quant_method", None), UnquantizedLinearMethod
        ):
            continue
        candidates.append((name, layer))
    return candidates


def _install_apply(allow_native_only: bool = False) -> None:
    global _INSTALLED, _GEMV, _SMALL_OP, _ORIGINAL_APPLY
    if _INSTALLED:
        return

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    if getattr(UnquantizedLinearMethod, "_mxfp6_sm120_ba_gemv", False):
        raise RuntimeError("a conflicting MXFP6 SM120 BA adapter is already installed")

    try:
        _GEMV = __import__(
            "vllm.mach.mxfp6.ba_gemv", fromlist=["precompile_bf16_gemv_small_n"]
        )
    except ModuleNotFoundError as exc:
        if not allow_native_only or exc.name != "b12x":
            raise
        logger.warning("b12x unavailable: 2B BA M1..8 retain F.linear")
    _SMALL_OP = torch.library.custom_op("mxfp6_sm120_ba::small_n", mutates_args=())(
        _small_impl
    )
    _SMALL_OP.register_fake(_small_fake)
    _ORIGINAL_APPLY = UnquantizedLinearMethod.apply

    def apply(
        self: Any,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        private = getattr(layer, "_mxfp6_sm120_ba_private_weight", None)
        if (
            private is None
            or x.ndim != 2
            or x.shape[1] != private.shape[1]
            or x.dtype != torch.bfloat16
            or not x.is_cuda
            or private.device != x.device
            or bias is not None
        ):
            return _ORIGINAL_APPLY(self, layer, x, bias)
        return _SMALL_OP(x, layer.weight, private)

    UnquantizedLinearMethod.apply = apply  # type: ignore[method-assign]
    UnquantizedLinearMethod._mxfp6_sm120_ba_gemv = True  # type: ignore[attr-defined]
    _INSTALLED = True


def prepare_model(model: torch.nn.Module, include_2b: bool = False) -> int:
    """Clone eligible BA weights, precompile M1..8, and install dispatch."""
    global _M16
    if getattr(model, "_mxfp6_sm120_ba_prepared", False):
        return int(getattr(model, "_mxfp6_sm120_ba_count", 0))

    candidates = _candidate_layers(model, include_2b=include_2b)
    if not candidates:
        model._mxfp6_sm120_ba_prepared = True
        model._mxfp6_sm120_ba_count = 0
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("MXFP6 SM120 BA GEMV requires CUDA")
    for _, layer in candidates:
        if torch.cuda.get_device_capability(layer.weight.device) != (12, 0):
            raise RuntimeError("MXFP6 SM120 BA GEMV requires an SM120 device")

    try:
        installed = version("b12x")
    except PackageNotFoundError:
        installed = "unknown"

    # Do capability validation through the actual imports used by the GEMV
    # module.  The compiler/intrinsics APIs are private and may evolve, so an
    # unsupported version should fail with the missing symbol rather than be
    # silently treated as compatible based on its metadata alone.
    try:
        _install_apply(
            allow_native_only=all(
                tuple(layer.weight.shape) == _BA_2B_SHAPE for _, layer in candidates
            )
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "MXFP6 SM120 BA GEMV could not load the b12x compiler/intrinsics "
            f"API (installed version: {installed}); use a b12x build exposing "
            "KernelCompileSpec, bf16 GEMV intrinsics and runtime stream helpers"
        ) from exc
    clone_count = 0
    if any(tuple(layer.weight.shape) == _BA_2B_SHAPE for _, layer in candidates):
        from mxfp6 import mxfp8

        mxfp8.load_library()
        _M16 = getattr(torch.ops.mxfp8_sm120, "bf16_ba_m16", None)
        if _M16 is None:
            logger.warning("2B native BA M16 unavailable; retaining F.linear at M16")
    for name, layer in candidates:
        weight = layer.weight
        private = getattr(layer, "_mxfp6_sm120_ba_private_weight", None)
        if private is None:
            private = weight.detach().clone().contiguous()
            if not torch.equal(private, weight):
                raise AssertionError(f"BA private weight differs for {name}")
            layer._mxfp6_sm120_ba_private_weight = private
            clone_count += 1
        if _GEMV is not None:
            _GEMV.precompile_bf16_gemv_small_n(private, log=logger)
        if tuple(weight.shape) == _BA_2B_SHAPE and _M16 is not None:
            _M16(
                torch.zeros((16, 2048), device=weight.device, dtype=weight.dtype),
                weight,
            )

    model._mxfp6_sm120_ba_prepared = True
    model._mxfp6_sm120_ba_count = len(candidates)
    logger.info(
        "Prepared MXFP6 SM120 BA small-M GEMV for %d layers "
        "(shape=%s, small_m_max=%d, small_m_available=%s, "
        "2b_native_m16=%s, b12x=%s, cloned=%d)",
        len(candidates),
        sorted({tuple(layer.weight.shape) for _, layer in candidates}),
        _SMALL_M_MAX,
        _GEMV is not None,
        _M16 is not None,
        installed,
        clone_count,
    )
    return len(candidates)


def inspect_model(model: torch.nn.Module) -> dict[str, Any]:
    """Return a small in-process receipt for debugging a worker startup."""
    rows = []
    clone_bytes = 0
    for name, layer in model.named_modules():
        private = getattr(layer, "_mxfp6_sm120_ba_private_weight", None)
        if private is None:
            continue
        if not name.endswith("in_proj_ba") or tuple(private.shape) not in (
            _BA_SHAPE,
            _BA_2B_SHAPE,
        ):
            raise RuntimeError(f"invalid BA private weight: {name}")
        if not torch.equal(private, layer.weight):
            raise RuntimeError(f"BA private weight changed after loading: {name}")
        rows.append(name)
        clone_bytes += private.numel() * private.element_size()
    return {
        "installed": _INSTALLED,
        "ba_layers": sorted(rows),
        "ba_layer_count": len(rows),
        "clone_bytes": clone_bytes,
        "small_m_max": _SMALL_M_MAX,
        "small_m_available": _GEMV is not None,
        "native_2b_m16_available": _M16 is not None,
        "counts_by_m": {str(m): dict(counts) for m, counts in sorted(_COUNTS.items())},
    }


__all__ = ["inspect_model", "prepare_model"]
