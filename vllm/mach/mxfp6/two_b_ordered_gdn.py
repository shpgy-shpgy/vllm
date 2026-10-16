# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Complete ordered W4 state lifecycle inside vLLM's existing opaque GDN op.

Imports are CPU only. The composed profile loads kernels after Worker.init_device.
"""

import contextvars
import functools
import hashlib
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
GDN_SHA = "d9fb8a2c00cf603237263c632c7a46eccf2cda4a433756670c5161d038c2ffbc"
# vllm-shpgy carries the request-deadline/cancellation cleanup in the runner;
# keep the lifecycle guard pinned to that exact local source revision.
RUNNER_SHA = "bb2fe1945e9d6c5f43eb28dc8e94fb7bbe1d02956147ca5518a99a232b0c4951"
FLA_SHA = "fe6f1311014809040497aa0a623e7fa97f4b3457cdc7ee1f1f8aed21f326f755"
W = 4
PINNED_IDS = os.environ.get("VLLM_MACH_2B_PINNED_IDS", "1") == "1"
ROWS = (32, 48, 64, 96, 128, 160)
BF16_ROWS = (1, 2, 4, 8, 16, 24) + ROWS
GDN_LAYERS = 18
STATE_HEADS = 16
QKV_WIDTH = 6144
QKV_STRIDE = 8192
MIN_STATE_PAGE_STRIDE = 16 * 128 * 128
_ELIGIBLE_ROWS: tuple[int, ...] = ()
_MAX_NUM_SEQS = None
_RUN_MODE = None
_QUALITY_ROW = None
_PREPARATION_BYTES: dict[Any, Any] = {}
_COMPONENT = None
_ACTIVE_TARGET: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "two_b_gdn_ordered_target", default=None
)
_ORIG_CORE = None
_ORIG_DECODE = None
_READY = False
_OWNER: Any = None
_CACHE_INITIALIZED = False
_STATE_GROUP_IDS = None
_LAYERS = {}
_VIEW_SCRATCH: dict[Any, Any] = {}
_CAPTURE: dict[Any, Any] = {}
_EAGER: dict[Any, Any] = {}
_FALLBACK: dict[Any, Any] = {}
_PROFILE: dict[Any, Any] = {}
_ALLOCATION_RESETS = {"new": 0, "cached": 0, "ids": 0}


def _component():
    global _COMPONENT
    if _COMPONENT is None:
        _COMPONENT = importlib.import_module(".gdn.two_b_ordered", __package__)
    return _COMPONENT


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _view_key(state):
    return (
        state.data_ptr(),
        tuple(state.shape),
        tuple(state.stride()),
        str(state.device),
        str(state.dtype),
    )


def _pools(layer):
    return layer.kv_cache[0], layer.kv_cache[1]


def _mode_contract():
    cfg = _OWNER.vllm_config if _OWNER is not None else None
    budget = getattr(getattr(cfg, "cache_config", None), "kv_cache_memory_bytes", None)
    return "shpgy", None, budget


def _check_config(worker):
    import torch

    cfg = worker.vllm_config
    cache, model = cfg.cache_config, cfg.model_config
    text = model.hf_text_config
    if (
        not cfg.use_v2_model_runner
        or cfg.parallel_config.tensor_parallel_size != 1
        or cfg.parallel_config.pipeline_parallel_size != 1
        or cfg.speculative_config is not None
        or cfg.lora_config is not None
        or cache.enable_prefix_caching
        or cache.mamba_cache_mode != "none"
        or cache.use_replayssm
        or getattr(cache, "kv_offloading_size", None) not in (None, 0)
        or getattr(cfg, "kv_transfer_config", None) is not None
        or getattr(model, "cpu_offload_gb", 0) not in (None, 0)
        or getattr(model, "enable_sleep_mode", False)
        or model.dtype != torch.bfloat16
        or cache.mamba_ssm_cache_dtype not in ("float32", "bfloat16")
    ):
        raise RuntimeError(
            "2B ordered GDN requires V2 TP1, FP32/BF16 SSM, BF16 model, "
            "no MTP/LoRA/prefix/ReplaySSM/offload/sleep/connector"
        )
    expected = {
        "hidden_size": 2048,
        "num_hidden_layers": 24,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
    }
    if any(getattr(text, k, None) != v for k, v in expected.items()):
        raise RuntimeError("2B ordered GDN requires Qwen3.5-2B geometry")
    if torch.cuda.get_device_capability(worker.device) != (12, 0):
        raise RuntimeError("2B ordered GDN requires SM120")


def _layout(layer, state):
    import torch

    return (
        layer.tp_size == 1
        and not layer.gqa_interleaved_layout
        and layer.enable_packed_recurrent_decode
        and layer._is_sm120
        and layer.num_k_heads == 16
        and layer.num_v_heads == STATE_HEADS
        and layer.head_k_dim == layer.head_v_dim == 128
        and len(layer.kv_cache) == 2
        and state.dtype in (torch.float32, torch.bfloat16)
        and tuple(state.shape[1:]) == (STATE_HEADS, 128, 128)
        and tuple(state.stride()[1:]) == (16384, 128, 1)
        and state.stride(0) >= MIN_STATE_PAGE_STRIDE
        and layer.kv_cache[0].dtype == torch.bfloat16
        and layer.A_log.dtype == torch.float32
        and layer.dt_bias.dtype == torch.bfloat16
    )


def _metadata(layer):
    from vllm.forward_context import get_forward_context

    raw = get_forward_context().attn_metadata
    return raw.get(layer.prefix) if isinstance(raw, dict) else None


def _target(layer, md, qkv, b, a, state):
    import torch

    m = int(qkv.shape[0])
    ids = md.non_spec_state_indices_tensor if md is not None else None
    return (
        _READY
        and layer.prefix in _LAYERS
        and m in _ELIGIBLE_ROWS
        and md is not None
        and md.spec_sequence_masks is None
        and md.num_spec_decodes == 0
        and md.num_prefills == 0
        and md.num_decodes > 0
        and md.num_actual_tokens == m
        and ids is not None
        and ids.numel() >= m
        and ids.dtype in (torch.int32, torch.int64)
        and ids.device == state.device
        and qkv.dtype == b.dtype == a.dtype == torch.bfloat16
        and tuple(qkv.shape) == (m, QKV_WIDTH)
        and tuple(b.shape) == tuple(a.shape) == (m, STATE_HEADS)
        and qkv.stride(1) == b.stride(1) == a.stride(1) == 1
        and qkv.stride(0) == QKV_STRIDE
        and b.stride(0) == a.stride(0) == STATE_HEADS
    )


def _relevant(md):
    # In this frozen no-MTP configuration the builder's non-spec vector is
    # block_table_tensor[:,0], one slot per request; 0 padding may repeat.
    if md.spec_sequence_masks is not None or md.num_spec_decodes:
        raise RuntimeError("spec/MTP route violates deferred contract")
    ids = md.non_spec_state_indices_tensor
    if ids is None:
        if md.num_prefills or md.num_decodes:
            raise RuntimeError("stock route has no non-spec state indices")
        return None
    active_reqs = md.num_decodes + md.num_prefills
    if active_reqs < 0 or ids.numel() < active_reqs:
        raise RuntimeError("non-spec index buffer shorter than active requests")
    if active_reqs == 0:
        return None
    return ids[:active_reqs]


def _register_gdn():
    global _ORIG_CORE, _ORIG_DECODE
    import torch

    component = _component()
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as GDN,
    )

    if getattr(GDN, "_two_b_gdn_ordered_installed", False):
        return
    if getattr(GDN, "_mach_ordered_gdn_installed", False) or getattr(
        GDN, "_mx8_deferred_installed", False
    ):
        raise RuntimeError("old deferred hook conflicts with full-M hook")
    if any(
        getattr(GDN, name, False)
        for name in (
            "_mach_mxfp8_ordered_installed",
            "_mx8_ssm_full_installed",
            "_ssm_lowm_capture",
            "_gdn_packed_capture",
            "_mx8_full_shadow_installed",
        )
    ):
        raise RuntimeError("conflicting GDN hook is installed")
    if _sha(inspect.getsourcefile(GDN)) != GDN_SHA:
        raise RuntimeError("live GDN source SHA drift")
    fla_source = importlib.import_module(
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
    )
    if len(FLA_SHA) != 64 or any(c not in "0123456789abcdef" for c in FLA_SHA):
        raise RuntimeError("invalid frozen packed FLA source SHA")
    if _sha(fla_source.__file__) != FLA_SHA:
        raise RuntimeError("live packed FLA source SHA drift")
    _ORIG_CORE = GDN._forward_core
    _ORIG_DECODE = GDN._forward_core_decode_non_spec

    @functools.wraps(_ORIG_CORE)
    def core(self, mixed_qkv, b, a, core_attn_out, hidden_states=None):
        md = _metadata(self)
        key = (self.prefix, int(mixed_qkv.shape[0]))
        if md is None:
            _PROFILE[key] = _PROFILE.get(key, 0) + 1
            return _ORIG_CORE(self, mixed_qkv, b, a, core_attn_out, hidden_states)
        if not _READY:
            raise RuntimeError(
                "GDN metadata reached before deferred pools are prepared"
            )
        _, state = _pools(self)
        if not _layout(self, state):
            raise RuntimeError(f"GDN live pool layout changed: {self.prefix}")
        if _target(self, md, mixed_qkv, b, a, state):
            capturing = torch.cuda.is_current_stream_capturing()
            counter = _CAPTURE if capturing else _EAGER
            counter[key] = counter.get(key, 0) + 1
            token = _ACTIVE_TARGET.set(self)
            try:
                # Only this eligible path suppresses the FI probe; the captured
                # real parent took packed FLA and returned false from FI.
                return _ORIG_CORE(
                    self, mixed_qkv, b, a, core_attn_out, hidden_states=None
                )
            finally:
                _ACTIVE_TARGET.reset(token)
        _FALLBACK[key] = _FALLBACK.get(key, 0) + 1
        ids = _relevant(md)
        if ids is not None:
            scratch = _VIEW_SCRATCH[_view_key(state)]
            component.materialize_slots(state, ids, scratch)
        # materialize_slots already resets age=0/prefix=1 before stock reads
        # base. Stock updates only base, so a second metadata launch is redundant.
        return _ORIG_CORE(self, mixed_qkv, b, a, core_attn_out, hidden_states)

    @functools.wraps(_ORIG_DECODE)
    def decode(self, mixed_qkv, b, a, core_attn_out, attn_metadata):
        if _ACTIVE_TARGET.get() is not self:
            return _ORIG_DECODE(self, mixed_qkv, b, a, core_attn_out, attn_metadata)
        m = int(attn_metadata.num_actual_tokens)
        ids = attn_metadata.non_spec_state_indices_tensor[:m]
        conv = self.kv_cache[0]
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

        if not is_conv_state_dim_first():
            conv = conv.transpose(-1, -2)
        state = self.kv_cache[1]
        # Exact original convolution from _forward_core_decode_non_spec.
        weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        post = gdn.causal_conv1d_update(
            mixed_qkv[:m],
            conv,
            weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=ids,
            validate_data=False,
        )
        parent_out = core_attn_out[:m].unsqueeze(1)
        live = _VIEW_SCRATCH[_view_key(state)]
        component.decode(
            post,
            a[:m],
            b[:m],
            self.A_log,
            self.dt_bias,
            state,
            parent_out,
            ids,
            live,
            scale=128**-0.5,
        )
        return

    GDN._forward_core = core  # type: ignore[method-assign]
    GDN._forward_core_decode_non_spec = decode  # type: ignore[method-assign]
    GDN._two_b_gdn_ordered_installed = True
    from vllm import envs

    from . import two_b_mixed_gdn as mixed

    if envs.VLLM_MACH_MXFP8_2B_GDN_MIXED:
        mixed.install(sys.modules[__name__], gdn)


def _prepare_layers(worker):
    _claim_worker(worker)
    global _ELIGIBLE_ROWS, _MAX_NUM_SEQS
    import torch

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as GDN,
    )

    _check_config(worker)
    _component()
    maximum = int(worker.vllm_config.scheduler_config.max_num_seqs)
    if maximum <= 0 or _MAX_NUM_SEQS is not None:
        raise RuntimeError("invalid/repeated full-M row preparation")
    _MAX_NUM_SEQS = maximum
    state_dtype = worker.vllm_config.cache_config.mamba_ssm_cache_dtype
    configured_rows = BF16_ROWS if state_dtype == "bfloat16" else ROWS
    _ELIGIBLE_ROWS = tuple(m for m in configured_rows if m <= maximum)
    layers = [x for _, x in worker.get_model().named_modules() if isinstance(x, GDN)]
    if len(layers) != GDN_LAYERS or len({x.prefix for x in layers}) != GDN_LAYERS:
        raise RuntimeError("expected 18 unique GDN layers")
    for i, layer in enumerate(layers):
        _LAYERS[layer.prefix] = i
        if (
            layer.tp_size != 1
            or layer.gqa_interleaved_layout
            or layer.num_k_heads != 16
            or layer.num_v_heads != STATE_HEADS
            or layer.head_k_dim != 128
            or layer.head_v_dim != 128
            or not layer.enable_packed_recurrent_decode
            or layer.A_log.dtype != torch.float32
            or layer.dt_bias.dtype != torch.bfloat16
        ):
            raise RuntimeError(f"unsupported layer {layer.prefix}")


def _prepare_pools(worker):
    _claim_worker(worker)
    if not _CACHE_INITIALIZED:
        raise RuntimeError("initialize ordered GDN cache before pool warmup/capture")
    global _READY
    import torch

    component = _component()
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as GDN,
    )
    from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR

    from .gdn import two_b_metadata_reset as kernels

    if _READY or len(_LAYERS) != GDN_LAYERS:
        raise RuntimeError("bad deferred preparation lifecycle")
    if worker.model_runner.kv_connector is not NO_OP_KV_CONNECTOR:
        raise RuntimeError("KV connector/offload path unsupported")
    layers = [x for _, x in worker.get_model().named_modules() if isinstance(x, GDN)]
    views: dict[tuple, Any] = {}
    for layer in layers:
        _, state = _pools(layer)
        if not _layout(layer, state):
            raise RuntimeError(f"unsupported bound pool {layer.prefix}")
        views.setdefault(_view_key(state), state)
    if not 1 <= len(views) <= GDN_LAYERS:
        raise RuntimeError(f"invalid unique GDN state view count: {len(views)}")
    expected_views = os.environ.get("VLLM_MACH_2B_GDN_EXPECTED_STATE_VIEWS")
    if expected_views is not None and len(views) != int(expected_views):
        raise RuntimeError("live state view count differs from explicit contract")
    if len({tuple(state.stride()) for state in views.values()}) != 1:
        raise RuntimeError("GDN state views have different page strides")
    required = 0
    for state in views.values():
        for name, shape in component.required_shapes(state.shape[0], W).items():
            required += __import__("math").prod(shape) * (4 if name == "age" else 4)
    _PREPARATION_BYTES.update(live_scratch=required, reserve=512 * 2**20)
    free, total = torch.accelerator.get_memory_info(layers[0].A_log.device)
    if free < required + 512 * 2**20:
        raise RuntimeError(
            f"ordered GDN requires {required} B live scratch + 512 MiB reserve; "
            f"GPU free {free}/{total}"
        )
    for key, state in views.items():
        shape = component.required_shapes(state.shape[0], W)
        _VIEW_SCRATCH[key] = component.DeferredScratch(
            **{
                name: torch.zeros(
                    size,
                    dtype=(torch.int32 if name == "age" else torch.float32),
                    device=state.device,
                )
                for name, size in shape.items()
            }
        )
        # Initialize only metadata once. Real new assignments are reset again
        # by the scheduler hooks; this protects positive dummy graph captures.
        all_slots = torch.arange(
            1, state.shape[0], dtype=torch.int32, device=state.device
        )
        kernels.reset_metadata(all_slots, _VIEW_SCRATCH[key], state.shape[0])
    device = layers[0].A_log.device
    # Warm every call shape before CUDA graph capture without writing live
    # base state. Zero indices exercise the invalid-slot guards.
    for layer in layers:
        _, state = _pools(layer)
        ids = torch.zeros(
            (max(_ELIGIBLE_ROWS, default=1),), dtype=torch.int32, device=state.device
        )
        scratch = _VIEW_SCRATCH[_view_key(state)]
        kernels.reset_metadata(ids, scratch, state.shape[0])
        # Fallback materialization also needs warming when no target M fits.
        component.materialize_slots(state, ids, scratch)
        for m in _ELIGIBLE_ROWS:
            idx = ids[:m]
            # The observed 2B split QKV has row stride 8192,
            # which is a Triton constexpr. Warm that exact graph signature.
            qkv = torch.empty_strided(
                (m, QKV_WIDTH),
                (QKV_STRIDE, 1),
                dtype=torch.bfloat16,
                device=state.device,
            )
            qkv.zero_()
            ba = torch.zeros(
                (m, STATE_HEADS), dtype=torch.bfloat16, device=state.device
            )
            out = torch.empty(
                (m, 1, STATE_HEADS, 128), dtype=torch.bfloat16, device=state.device
            )
            component.decode(
                qkv,
                ba,
                ba,
                layer.A_log,
                layer.dt_bias,
                state,
                out,
                idx,
                scratch,
                scale=128**-0.5,
            )
            component.materialize_slots(state, idx, scratch)
    torch.accelerator.synchronize(device)
    _READY = True


def _flatten_new(blocks, group_ids=None):
    if blocks is None:
        return []
    if not isinstance(blocks, tuple):
        raise TypeError("block_ids must be tuple[list[int], ...]")
    ids: list[int] = []
    selected = range(len(blocks)) if group_ids is None else group_ids
    for index in selected:
        group = blocks[index]
        if not isinstance(group, list) or any(type(x) is not int for x in group):
            raise TypeError("block_ids group must contain Python ints")
        ids.extend(x for x in group if x > 0)
    return ids


def _reset_allocated(ids, kind):
    if not ids or not _READY:
        return
    import torch

    from .gdn import two_b_metadata_reset as kernels

    unique = sorted(set(ids))
    # CPU scheduler knows these newly assigned IDs. No device age read or
    # capture-time host slot value is involved.
    ids_by_device = {}
    for key, scratch in _VIEW_SCRATCH.items():
        pages = key[1][0]
        if unique[-1] >= pages:
            raise RuntimeError(
                f"allocated block outside GDN state pool: {unique[-1]}/{pages}"
            )
        dev = torch.device(key[3])
        if key[3] not in ids_by_device:
            if PINNED_IDS:
                # Fresh PyTorch-owned pinned storage; never mutated after .to().
                # CUDA copy records allocator stream use before local ref dies.
                host_ids = torch.tensor(
                    unique, dtype=torch.int32, device="cpu", pin_memory=True
                )
                ids_by_device[key[3]] = host_ids.to(dev, non_blocking=True)
            else:
                ids_by_device[key[3]] = torch.tensor(
                    unique, dtype=torch.int32, device=dev
                )
        device_ids = ids_by_device[key[3]]
        kernels.reset_metadata(device_ids, scratch, pages)
    _ALLOCATION_RESETS[kind] += 1
    _ALLOCATION_RESETS["ids"] += len(unique)


def _install_runner_lifecycle():
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_two_b_gdn_ordered_lifecycle", False):
        return
    if (
        getattr(GPUModelRunner, "_mach_ordered_gdn_lifecycle", False)
        or getattr(GPUModelRunner, "_mx8_deferred_lifecycle", False)
        or getattr(GPUModelRunner, "_mach_mxfp8_ordered_lifecycle", False)
    ):
        raise RuntimeError("old deferred runner lifecycle conflicts with full-M hook")
    if _sha(inspect.getsourcefile(GPUModelRunner)) != RUNNER_SHA:
        raise RuntimeError("GPUModelRunner lifecycle source SHA drift")
    old_add, old_update = GPUModelRunner.add_requests, GPUModelRunner.update_requests

    @functools.wraps(old_add)
    def add(self, scheduler_output):
        if _OWNER is None or self is not _OWNER.model_runner:
            return old_add(self, scheduler_output)
        if any(
            req.num_computed_tokens != 0 for req in scheduler_output.scheduled_new_reqs
        ):
            raise RuntimeError(
                "new/resumed request with retained state requires "
                "an explicit deferred-state transfer"
            )
        result = old_add(self, scheduler_output)
        ids = [
            x
            for req in scheduler_output.scheduled_new_reqs
            for x in _flatten_new(req.block_ids, _STATE_GROUP_IDS)
        ]
        _reset_allocated(ids, "new")
        return result

    @functools.wraps(old_update)
    def update(self, scheduler_output):
        if _OWNER is None or self is not _OWNER.model_runner:
            return old_update(self, scheduler_output)
        if scheduler_output.kv_cache_block_copies:
            raise RuntimeError("deferred GDN does not support block copies")
        result = old_update(self, scheduler_output)
        ids = [
            x
            for block in scheduler_output.scheduled_cached_reqs.new_block_ids
            for x in _flatten_new(block, _STATE_GROUP_IDS)
        ]
        _reset_allocated(ids, "cached")
        return result

    GPUModelRunner.add_requests = add  # type: ignore[method-assign]
    GPUModelRunner.update_requests = update  # type: ignore[method-assign]
    GPUModelRunner._two_b_gdn_ordered_lifecycle = True  # type: ignore[attr-defined]


def inspect_worker(worker):
    _claim_worker(worker)
    if _READY:
        _bound_views(worker)
    component = _component()
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as GDN,
    )

    layers = [x for _, x in worker.get_model().named_modules() if isinstance(x, GDN)]
    sources = {path.name: _sha(path) for path in ROOT.glob("*.py")}
    mode, quality_row, budget = _mode_contract()
    record = {
        "mode": "quality" if mode == "quality" else "normal",
        "profile_mode": mode,
        "quality_rows": quality_row,
        "kv_cache_memory_bytes": budget,
        "production_throughput_profile": False,
        "implementation": "two_b_ordered_highM_w4",
        "configured_rows": list(
            BF16_ROWS
            if _OWNER.vllm_config.cache_config.mamba_ssm_cache_dtype == "bfloat16"
            else ROWS
        ),
        "low_rows_policy": (
            "gluon_on_configured_rows_else_materialize_then_stock"
            if _OWNER.vllm_config.cache_config.mamba_ssm_cache_dtype == "bfloat16"
            else "materialize_relevant_slots_then_stock"
        ),
        "adapted_low_kernels_enabled": False,
        "pinned_ids_enabled": PINNED_IDS,
        "eligible_rows": list(_ELIGIBLE_ROWS),
        "scheduler_max_num_seqs": _MAX_NUM_SEQS,
        "warm_index_capacity": max(_ELIGIBLE_ROWS, default=1),
        "preparation_required_bytes": dict(_PREPARATION_BYTES),
        "state_dtype": _OWNER.vllm_config.cache_config.mamba_ssm_cache_dtype,
        "component_module": component.__name__,
        "component_sha256": _sha(component.__file__),
        "coeff_semantics": "raw_alpha_ordered_replay_highM_W4",
        "ready": _READY,
        "cache_initialized": _CACHE_INITIALIZED,
        "layers": sorted(_LAYERS),
        "layer_count": len(_LAYERS),
        "unique_state_views": len(_VIEW_SCRATCH),
        "view_pages_strides": [
            {"pages": key[1][0], "stride": key[2], "device": key[3]}
            for key in _VIEW_SCRATCH
        ],
        "capture_construction": {f"{p}|m{m}": n for (p, m), n in _CAPTURE.items()},
        "eager_dispatch": {f"{p}|m{m}": n for (p, m), n in _EAGER.items()},
        "fallback_dispatch": {f"{p}|m{m}": n for (p, m), n in _FALLBACK.items()},
        "profile_dispatch": {f"{p}|m{m}": n for (p, m), n in _PROFILE.items()},
        "new_block_metadata_resets": dict(_ALLOCATION_RESETS),
        "scratch_bytes": sum(
            sum(
                getattr(s, n).numel() * getattr(s, n).element_size()
                for n in ("pending_k", "pending_d", "coeff", "prefix", "age")
            )
            for s in _VIEW_SCRATCH.values()
        ),
        "sources": sources,
        "fla_sha256": FLA_SHA,
        "gdn_sha256": GDN_SHA,
        "runner_sha256": RUNNER_SHA,
        "normal_gpu_counter_allocated": False,
        "existing_outer_custom_op_only": True,
        "bound_pool_shapes": [list(x.kv_cache[1].shape) for x in layers],
    }
    from . import two_b_mixed_gdn as mixed

    record["mixed_decode"] = mixed.snapshot()
    return record


def _claim_worker(worker):
    global _OWNER
    # Callable collective_rpc receives WorkerWrapperBase; lifecycle hooks
    # receive its real Worker. Both identities refer to the same owner.
    from vllm.v1.worker.worker_base import WorkerWrapperBase

    if isinstance(worker, WorkerWrapperBase):
        worker = worker.worker
        if worker is None:
            raise RuntimeError("ordered GDN RPC wrapper has no initialized worker")
    if _OWNER is None:
        _OWNER = worker  # Retain the owner, not an id that can be recycled.
    elif _OWNER is not worker:
        raise RuntimeError("ordered GDN state already belongs to another worker")


def install_runtime(worker):
    """After init_device: register the opaque-op replacement and scheduler resets."""
    global _RUN_MODE, _QUALITY_ROW
    _claim_worker(worker)
    _check_config(worker)
    _RUN_MODE, _QUALITY_ROW, _ = _mode_contract()
    _register_gdn()
    _install_runner_lifecycle()


def initialize_cache(worker):
    """Validate the final state binding before allocating scratch."""
    global _CACHE_INITIALIZED, _STATE_GROUP_IDS
    _claim_worker(worker)
    if _READY or _CACHE_INITIALIZED or len(_LAYERS) != GDN_LAYERS:
        raise RuntimeError(
            "ordered GDN requires model load, then a single cache binding"
        )
    _check_config(worker)
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as GDN,
    )

    for _, layer in worker.get_model().named_modules():
        if isinstance(layer, GDN) and not _layout(layer, _pools(layer)[1]):
            raise RuntimeError(f"unsupported initialized GDN state: {layer.prefix}")
    groups = worker.model_runner.kv_cache_config.kv_cache_groups
    _STATE_GROUP_IDS = tuple(
        i for i, group in enumerate(groups) if _LAYERS.keys() & set(group.layer_names)
    )
    if not _STATE_GROUP_IDS:
        raise RuntimeError("ordered GDN layers have no bound state cache group")
    # Attention-only allocations cannot own a live GDN slot in the shared
    # block pool. Reset on GDN allocation (including reuse), not KV growth.
    _CACHE_INITIALIZED = True


def _idle_boundary(worker):
    import torch

    _claim_worker(worker)
    if not _READY or torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "state lifecycle operation requires ready pools outside graph capture"
        )
    if worker.model_runner.req_states.req_id_to_index:
        raise RuntimeError("drain requests before explicit state lifecycle operations")
    torch.accelerator.synchronize(worker.device)


def materialize_state(worker):
    """Drain-boundary RPC: flush positive slots before exporting FP32 state."""
    import torch

    _idle_boundary(worker)
    component = _component()
    views = _bound_views(worker)
    for key, scratch in _VIEW_SCRATCH.items():
        state = views[key]
        ids = torch.arange(1, state.shape[0], dtype=torch.int32, device=state.device)
        component.materialize_slots(state, ids, scratch)
    torch.accelerator.synchronize(worker.device)
    return {"materialized_views": len(_VIEW_SCRATCH), "synchronized": True}


def reset_state(worker, *, zero_base=False):
    """Drain-boundary RPC: invalidate pending terms, optionally clear dense state.

    Scratch/owner storage stays stable for previously captured graphs. Normal new
    request and slot reuse resets happen automatically in the scheduler hooks.
    """
    import torch

    if type(zero_base) is not bool:
        raise TypeError("zero_base must be bool")
    _idle_boundary(worker)
    component = _component()
    views = _bound_views(worker)
    for key, scratch in _VIEW_SCRATCH.items():
        state = views[key]
        ids = torch.arange(1, state.shape[0], dtype=torch.int32, device=state.device)
        component.reset_slots(state, ids, scratch, zero_base=zero_base)
    torch.accelerator.synchronize(worker.device)
    return {
        "reset_views": len(_VIEW_SCRATCH),
        "zero_base": zero_base,
        "scratch_storage_preserved": True,
        "synchronized": True,
    }


def _bound_views(worker):
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention as GDN,
    )

    views = {
        _view_key(layer.kv_cache[1]): layer.kv_cache[1]
        for _, layer in worker.get_model().named_modules()
        if isinstance(layer, GDN)
    }
    if set(views) != set(_VIEW_SCRATCH):
        raise RuntimeError("bound GDN state pool storage changed after capture")
    return views


# Direct lifecycle entrypoints for the complete profile's composed Worker hook.
prepare_layers = _prepare_layers
prepare_pools = _prepare_pools
