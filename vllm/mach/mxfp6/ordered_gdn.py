# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in Ordered GDN adapter for Qwen3.5-4B on SM120.

This is the Ordered GDN migration from vllm-mach.  It hooks the existing
``qwen_gdn_attention_core`` opaque op, so the model and scheduler retain the
normal vLLM lifecycle.  Pure decode batches with M1/2/4/8 use full-store
kernels, M16 uses deferred W4 scratch, and larger/mixed calls keep the stock
GDN implementation after draining pending state.
"""

from __future__ import annotations

import functools
from collections import Counter
from typing import Any
from weakref import WeakSet

import torch

from vllm import envs
from vllm.logger import init_logger

from .gdn import ordered_lowm, ordered_m16_deferred

logger = init_logger(__name__)

LOW_ROWS = ordered_lowm.ROWS
DEFERRED_ROW = 16
ROWS = LOW_ROWS + (DEFERRED_ROW,)
_STATS: Counter[str] = Counter()
_ORIGINAL_CORE = None
_ORIGINAL_BIND = None
_DEFERRED_LAYERS: WeakSet[Any] = WeakSet()
_DEFERRED_WARMED: set[tuple[str, int]] = set()
_RUNNER_LIFECYCLE_INSTALLED = False


def stats() -> dict[str, int]:
    """Return host dispatch counts; CUDA graph replays are not counted."""

    return dict(_STATS)


def _metadata(layer):
    from vllm.forward_context import get_forward_context

    raw = get_forward_context().attn_metadata
    return raw.get(layer.prefix) if isinstance(raw, dict) else None


def _eligible_layer(layer) -> bool:
    if type(layer).__name__ != "QwenGatedDeltaNetAttention":
        return False
    if (
        not getattr(layer, "_is_sm120", False)
        or layer.tp_size != 1
        or layer.gqa_interleaved_layout
        or not layer.enable_packed_recurrent_decode
        or layer.num_k_heads != 16
        or layer.num_v_heads != 32
        or layer.head_k_dim != 128
        or layer.head_v_dim != 128
    ):
        return False
    if tuple(layer.in_proj_ba.weight.shape) != (64, 2560):
        return False
    if tuple(layer.conv1d.weight.shape) != (8192, 1, 4):
        return False
    if layer.conv1d.weight.dtype != torch.bfloat16:
        return False
    if layer.A_log.dtype != torch.float32 or layer.dt_bias.dtype != torch.bfloat16:
        return False
    try:
        _, state_dtype = layer.get_state_dtype()
    except Exception:
        return False
    return state_dtype == torch.float32


def _state_key(state: torch.Tensor) -> tuple:
    return (
        state.data_ptr(),
        tuple(state.shape),
        tuple(state.stride()),
        str(state.device),
        str(state.dtype),
    )


def _layer_scratch(layer):
    return getattr(layer, "_mach_ordered_gdn_deferred_scratch", None)


@torch.inference_mode()
def _warm_deferred(layer, state, scratch) -> None:
    """Compile every deferred/materialize signature before graph capture."""

    signature = (str(state.device), int(state.shape[0]))
    if signature in _DEFERRED_WARMED:
        return

    device = state.device
    for rows in LOW_ROWS:
        indices = torch.zeros((rows,), dtype=torch.int32, device=device)
        ordered_m16_deferred.materialize_slots(state, indices, scratch)

    qkv = torch.empty_strided(
        (DEFERRED_ROW, ordered_m16_deferred.QKV),
        (12288, 1),
        dtype=torch.bfloat16,
        device=device,
    )
    qkv.zero_()
    a = torch.zeros(
        (DEFERRED_ROW, ordered_m16_deferred.HV),
        dtype=torch.bfloat16,
        device=device,
    )
    b = torch.zeros_like(a)
    out = torch.empty(
        (DEFERRED_ROW, 1, ordered_m16_deferred.HV, ordered_m16_deferred.V),
        dtype=torch.bfloat16,
        device=device,
    )
    indices = torch.zeros((DEFERRED_ROW,), dtype=torch.int32, device=device)
    ordered_m16_deferred.reset_slots(state, indices, scratch)
    ordered_m16_deferred.decode(
        qkv,
        a,
        b,
        layer.A_log,
        layer.dt_bias,
        state,
        out,
        indices,
        scratch,
    )
    ordered_m16_deferred.materialize_slots(state, indices, scratch)
    _DEFERRED_WARMED.add(signature)


def _bind_deferred_scratch(layer) -> None:
    """Attach one stable W4 scratch bundle to each bound GDN state pool."""

    if not getattr(layer, "_mach_ordered_gdn_prepared", False):
        return
    if not _eligible_layer(layer) or len(layer.kv_cache) != 2:
        return
    state = layer.kv_cache[1]
    if (
        state.ndim != 4
        or tuple(state.shape[1:])
        != (ordered_m16_deferred.HV, ordered_m16_deferred.V, ordered_m16_deferred.K)
        or state.dtype != torch.float32
        or not state.is_cuda
        or state.stride(-1) != 1
        or state.stride(-2) != ordered_m16_deferred.K
        or state.stride(-3) != ordered_m16_deferred.V * ordered_m16_deferred.K
        or state.stride(0)
        < (ordered_m16_deferred.HV * ordered_m16_deferred.V * ordered_m16_deferred.K)
    ):
        return

    key = _state_key(state)
    if getattr(layer, "_mach_ordered_gdn_deferred_state_key", None) == key:
        return

    # A worker may bind a small profiling pool and then the final pool.  The
    # old bundle becomes unreachable as each layer is rebound, so no cache
    # layout or graph pointer is ever reused across the two pools.
    # Reuse the private padding already present in each mamba cache page.
    # This keeps deferred metadata out of the KV-cache memory budget.
    scratch = ordered_m16_deferred.from_state(state)
    scratch.age.zero_()
    scratch.prefix.fill_(1.0)
    layer._mach_ordered_gdn_deferred_state_key = key
    layer._mach_ordered_gdn_deferred_scratch = scratch
    layer._mach_ordered_gdn_deferred_seen = False
    _DEFERRED_LAYERS.add(layer)
    _warm_deferred(layer, state, scratch)


def _install_bind_hook() -> None:
    global _ORIGINAL_BIND
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as GDN,
    )

    if getattr(GDN, "_mach_ordered_gdn_bind_installed", False):
        return
    _ORIGINAL_BIND = GDN.bind_kv_cache

    @functools.wraps(_ORIGINAL_BIND)
    def ordered_bind(self, kv_cache):
        result = _ORIGINAL_BIND(self, kv_cache)
        _bind_deferred_scratch(self)
        return result

    GDN.bind_kv_cache = ordered_bind  # type: ignore[method-assign]
    GDN._mach_ordered_gdn_bind_installed = True


def _target(layer, metadata: Any, mixed_qkv, b, a) -> bool:
    if metadata is None or not getattr(layer, "_mach_ordered_gdn_prepared", False):
        return False
    rows = int(mixed_qkv.shape[0])
    if rows not in ROWS:
        return False
    if (
        metadata.spec_sequence_masks is not None
        or metadata.num_spec_decodes != 0
        or metadata.num_prefills != 0
        or metadata.num_decodes <= 0
        or metadata.num_actual_tokens != rows
        or metadata.non_spec_state_indices_tensor is None
    ):
        return False
    if (
        tuple(mixed_qkv.shape) != (rows, ordered_lowm.QKV)
        or tuple(b.shape) != (rows, ordered_lowm.HV)
        or tuple(a.shape) != (rows, ordered_lowm.HV)
        or mixed_qkv.dtype != torch.bfloat16
        or b.dtype != torch.bfloat16
        or a.dtype != torch.bfloat16
        or mixed_qkv.stride(1) != 1
        or b.stride(1) != 1
        or a.stride(1) != 1
    ):
        return False
    if not layer.kv_cache or len(layer.kv_cache) != 2:
        return False
    conv, state = layer.kv_cache
    if rows == DEFERRED_ROW and _layer_scratch(layer) is None:
        return False
    indices = metadata.non_spec_state_indices_tensor[:rows]
    return (
        conv.dtype == torch.bfloat16
        and state.dtype == torch.float32
        and tuple(state.shape[1:]) == (ordered_lowm.HV, ordered_lowm.V, ordered_lowm.K)
        and state.stride(-1) == 1
        and state.stride(-2) == ordered_lowm.K
        and state.stride(-3) == ordered_lowm.V * ordered_lowm.K
        and indices.ndim == 1
        and indices.shape[0] == rows
        and indices.dtype in (torch.int32, torch.int64)
        and indices.device == state.device
        and indices.stride(0) == 1
    )


def _dispatch(layer, mixed_qkv, b, a, core_attn_out, metadata) -> None:
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    conv, state = layer.kv_cache
    if not is_conv_state_dim_first():
        conv = conv.transpose(-1, -2)
    indices = metadata.non_spec_state_indices_tensor[: mixed_qkv.shape[0]]
    conv_weights = layer.conv1d.weight.view(layer.conv1d.weight.size(0), 4)
    mixed_qkv = causal_conv1d_update(
        mixed_qkv,
        conv,
        conv_weights,
        layer.conv1d.bias,
        layer.activation,
        conv_state_indices=indices,
        validate_data=False,
    )
    out = core_attn_out[: mixed_qkv.shape[0]].unsqueeze(1)
    scratch = _layer_scratch(layer)
    if mixed_qkv.shape[0] == DEFERRED_ROW:
        if scratch is None:
            raise RuntimeError("M16 Ordered GDN scratch was not bound")
        ordered_m16_deferred.decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            base=state,
            out=out,
            indices=indices,
            scratch=scratch,
            scale=layer.head_k_dim**-0.5,
        )
        return

    ordered_lowm.decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        base=state,
        out=out,
        indices=indices,
        scale=layer.head_k_dim**-0.5,
    )


def _materialize_fallback(layer, metadata: Any) -> None:
    """Drain pending M16 terms before the stock path reads a state pool."""

    scratch = _layer_scratch(layer)
    if (
        scratch is None
        or not getattr(layer, "_mach_ordered_gdn_deferred_seen", False)
        or metadata is None
        or not layer.kv_cache
    ):
        return
    state = layer.kv_cache[1]

    non_spec = getattr(metadata, "non_spec_state_indices_tensor", None)
    if non_spec is not None and non_spec.numel():
        # The GDN metadata builder stores one block id per active request and
        # leaves zero sentinels in the graph-padded tail.
        active = int(metadata.num_decodes + metadata.num_prefills)
        if active:
            ordered_m16_deferred.materialize_slots(state, non_spec[:active], scratch)

    spec = getattr(metadata, "spec_state_indices_tensor", None)
    if spec is not None and spec.numel():
        # Speculative state ids are [request, accepted-token]; the first
        # column is the owning state slot.  Passing the full graph buffer is
        # capture-safe; zero-padded ids are ignored by the kernel.
        ordered_m16_deferred.materialize_slots(state, spec[:, 0], scratch)


def before_model_run(model_runner, attn_metadata, batch_desc, dummy_run: bool) -> None:
    """Materialize only when the scheduler changes away from physical M16.

    This hook runs after GDN metadata is built but before a CUDA graph replay.
    Keeping the transition materialize outside the M1/2/4/8 graphs avoids a
    per-layer scratch-check kernel on the common small-batch path.
    """

    if dummy_run:
        return
    if not isinstance(attn_metadata, dict):
        model_runner._mach_ordered_gdn_last_m16 = False
        return

    metadata = next(
        (
            value
            for value in attn_metadata.values()
            if hasattr(value, "num_decodes")
            and hasattr(value, "non_spec_state_indices_tensor")
        ),
        None,
    )
    if metadata is None:
        model_runner._mach_ordered_gdn_last_m16 = False
        return

    rows = int(getattr(batch_desc, "num_tokens", 0))
    current_m16 = (
        rows == DEFERRED_ROW
        and metadata.spec_sequence_masks is None
        and metadata.num_spec_decodes == 0
        and metadata.num_prefills == 0
        and metadata.num_decodes > 0
        and metadata.num_actual_tokens == rows
    )
    previous_m16 = getattr(model_runner, "_mach_ordered_gdn_last_m16", False)
    if previous_m16 and not current_m16:
        for layer in list(_DEFERRED_LAYERS):
            if not getattr(layer, "_mach_ordered_gdn_deferred_seen", False):
                continue
            layer_metadata = attn_metadata.get(layer.prefix)
            if layer_metadata is None:
                continue
            _materialize_fallback(layer, layer_metadata)

    model_runner._mach_ordered_gdn_last_m16 = current_m16


def _flatten_block_ids(blocks) -> list[int]:
    if blocks is None:
        return []
    if not isinstance(blocks, tuple):
        raise TypeError("block_ids must be tuple[list[int], ...]")
    ids: list[int] = []
    for group in blocks:
        if not isinstance(group, list):
            raise TypeError("block_ids group must be a Python list")
        ids.extend(block for block in group if type(block) is int and block > 0)
    return ids


def _reset_allocated(ids) -> None:
    if not ids:
        return
    unique = sorted(set(ids))
    layers = [layer for layer in _DEFERRED_LAYERS if _layer_scratch(layer) is not None]
    if not layers:
        return
    device = layers[0].kv_cache[1].device
    device_ids = torch.tensor(unique, dtype=torch.int32, device=device)
    for layer in layers:
        state = layer.kv_cache[1]
        if unique[-1] >= state.shape[0]:
            raise RuntimeError(
                f"allocated GDN block {unique[-1]} exceeds pool {state.shape[0]}"
            )
        ordered_m16_deferred.reset_slots(state, device_ids, _layer_scratch(layer))


def _install_runner_lifecycle() -> None:
    global _RUNNER_LIFECYCLE_INSTALLED
    if _RUNNER_LIFECYCLE_INSTALLED:
        return
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_mach_ordered_gdn_lifecycle", False):
        return
    old_add = GPUModelRunner.add_requests
    old_update = GPUModelRunner.update_requests
    old_shutdown = GPUModelRunner.shutdown

    @functools.wraps(old_add)
    def ordered_add(self, scheduler_output):
        result = old_add(self, scheduler_output)
        ids = [
            block
            for request in scheduler_output.scheduled_new_reqs
            for block in _flatten_block_ids(request.block_ids)
        ]
        _reset_allocated(ids)
        return result

    @functools.wraps(old_update)
    def ordered_update(self, scheduler_output):
        if scheduler_output.kv_cache_block_copies:
            raise RuntimeError(
                "Ordered deferred GDN does not support KV block copies; "
                "disable prefix caching"
            )
        result = old_update(self, scheduler_output)
        ids = [
            block
            for blocks in scheduler_output.scheduled_cached_reqs.new_block_ids
            for block in _flatten_block_ids(blocks)
        ]
        _reset_allocated(ids)
        return result

    @functools.wraps(old_shutdown)
    def ordered_shutdown(self):
        result = old_shutdown(self)
        for layer in list(_DEFERRED_LAYERS):
            layer._mach_ordered_gdn_deferred_scratch = None
            layer._mach_ordered_gdn_deferred_state_key = None
            layer._mach_ordered_gdn_deferred_seen = False
        _DEFERRED_LAYERS.clear()
        return result

    GPUModelRunner.add_requests = ordered_add  # type: ignore[method-assign]
    GPUModelRunner.update_requests = ordered_update  # type: ignore[method-assign]
    GPUModelRunner.shutdown = ordered_shutdown  # type: ignore[method-assign]
    GPUModelRunner._mach_ordered_gdn_lifecycle = True  # type: ignore[attr-defined]
    _RUNNER_LIFECYCLE_INSTALLED = True


def _install_core_hook() -> None:
    global _ORIGINAL_CORE
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as GDN,
    )

    if getattr(GDN, "_mach_ordered_gdn_installed", False):
        return
    if any(
        getattr(GDN, name, False)
        for name in (
            "_mach_mxfp8_ordered_installed",
            "_mx8_deferred_installed",
            "_mx8_ssm_full_installed",
        )
    ):
        raise RuntimeError("Ordered GDN conflicts with another GDN core hook")

    _ORIGINAL_CORE = GDN._forward_core

    @functools.wraps(_ORIGINAL_CORE)
    def ordered_core(self, mixed_qkv, b, a, core_attn_out, hidden_states=None):
        metadata = _metadata(self)
        if _target(self, metadata, mixed_qkv, b, a):
            key = f"ordered_m{mixed_qkv.shape[0]}"
            if not _STATS[key]:
                logger.info("Ordered GDN selected at physical M%d", mixed_qkv.shape[0])
            _STATS[key] += 1
            if mixed_qkv.shape[0] == DEFERRED_ROW:
                self._mach_ordered_gdn_deferred_seen = True
            _dispatch(self, mixed_qkv, b, a, core_attn_out, metadata)
            return
        _STATS["fallback"] += 1
        _materialize_fallback(self, metadata)
        return _ORIGINAL_CORE(
            self,
            mixed_qkv,
            b,
            a,
            core_attn_out,
            hidden_states,
        )

    GDN._forward_core = ordered_core  # type: ignore[method-assign]
    GDN._mach_ordered_gdn_installed = True


@torch.inference_mode()
def prepare(model) -> int:
    """Validate, warm, and install the first Ordered GDN stage."""

    if getattr(model, "_mach_ordered_gdn_prepared", False):
        return int(_STATS.get("prepared_layers", 0))
    if envs.VLLM_MACH_GDN_PERSISTENT or envs.VLLM_MACH_GDN_BA_OVERLAP:
        raise ValueError(
            "VLLM_MACH_GDN_ORDERED is mutually exclusive with the legacy "
            "Mach persistent/BA-overlap GDN adapters"
        )

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as GDN,
    )

    gdn_layers = [layer for layer in model.modules() if isinstance(layer, GDN)]
    layers = [layer for layer in gdn_layers if _eligible_layer(layer)]
    if not gdn_layers or len(layers) != len(gdn_layers):
        raise ValueError(
            "Ordered GDN requires every GDN layer to be the Qwen3.5-4B "
            "TP1 SM120 FP32-state geometry"
        )
    if not layers:
        raise ValueError("Ordered GDN found no eligible Qwen3.5-4B GDN layers")

    device = layers[0].A_log.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError("Ordered GDN requires an SM120 CUDA device")

    # Keep the real mixed-QKV row stride used by Qwen3.5's [q,k,v,z] split.
    for rows in LOW_ROWS:
        qkv = torch.empty_strided(
            (rows, ordered_lowm.QKV),
            (12288, 1),
            dtype=torch.bfloat16,
            device=device,
        )
        qkv.zero_()
        a = torch.zeros((rows, ordered_lowm.HV), dtype=torch.bfloat16, device=device)
        b = torch.zeros((rows, ordered_lowm.HV), dtype=torch.bfloat16, device=device)
        state = torch.zeros(
            (rows + 1, ordered_lowm.HV, ordered_lowm.V, ordered_lowm.K),
            dtype=torch.float32,
            device=device,
        )
        indices = torch.arange(1, rows + 1, dtype=torch.int32, device=device)
        out = torch.empty(
            (rows, 1, ordered_lowm.HV, ordered_lowm.V),
            dtype=torch.bfloat16,
            device=device,
        )
        ordered_lowm.decode(
            qkv,
            a,
            b,
            layers[0].A_log,
            layers[0].dt_bias,
            state,
            out,
            indices,
        )

    torch.accelerator.synchronize()
    _install_core_hook()
    _install_bind_hook()
    _install_runner_lifecycle()
    for layer in layers:
        layer._mach_ordered_gdn_prepared = True
    model._mach_ordered_gdn_prepared = True
    _STATS["prepared_layers"] = len(layers)
    logger.info(
        "Prepared Ordered GDN for %d Qwen3.5-4B layers: full-store rows=%s, "
        "deferred row=%d",
        len(layers),
        LOW_ROWS,
        DEFERRED_ROW,
    )
    return len(layers)
