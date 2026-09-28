# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bridge Mach producers to the existing Quark W6A8 implementation."""

from vllm.model_executor.layers.quantization.utils.mxfp6_sm120_utils import (
    _import_mxfp6,
)


def is_native_linear(layer) -> bool:
    scheme = getattr(layer, "scheme", None)
    return bool(getattr(scheme, "use_mxfp6_sm120", False))


__all__ = ["is_native_linear", "_import_mxfp6"]
