# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native GDN routing must preserve unsupported and mixed-batch fallbacks."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.mach.mxfp6.gdn_decode import select_path


def test_gdn_cross_stream_order_uses_integer_device_index(monkeypatch):
    """Switching warmup/capture streams must use the accelerator API contract."""
    from contextlib import nullcontext
    from unittest.mock import Mock

    from vllm.mach.mxfp6.gdn._stream_order import order_after_previous_stream

    device = torch.device("cuda:1")
    previous, current, event = Mock(), Mock(), Mock()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: current)
    monkeypatch.setattr(torch.cuda, "Event", lambda: event)
    context = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch.accelerator, "device_index", context)
    streams = {str(device): previous}
    order_after_previous_stream(streams, device)
    context.assert_called_once_with(1)
    event.record.assert_called_once_with(previous)
    current.wait_event.assert_called_once_with(event)
    assert streams[str(device)] is current


def test_precompiled_gdn_loads_once_without_jit(monkeypatch):
    """Deployment must load the wheel once and reuse it for graph readiness."""
    import sys

    from vllm.mach.mxfp6.gdn import persistent

    calls = []
    module = object()

    def load(geometry, digest):
        calls.append((geometry, digest))
        return module

    monkeypatch.setattr(persistent, "_modules", {})
    monkeypatch.setattr(persistent, "_kernel_digest", lambda: "source-hash")
    monkeypatch.setitem(sys.modules, "vllm_mach_gdn", SimpleNamespace(load=load))
    geometry = (5120, 48, 5120, 8, 24, 128, 4, 3, 32)
    assert persistent._get_module(geometry) is module
    assert persistent._get_module(geometry) is module
    assert calls == [(geometry, "source-hash")]


def test_strict_gdn_missing_wheel_fails_without_jit(monkeypatch):
    import sys

    from vllm.mach.mxfp6.gdn import persistent

    monkeypatch.setattr(persistent, "_modules", {})
    monkeypatch.setitem(sys.modules, "vllm_mach_gdn", None)
    monkeypatch.setenv("VLLM_MACH_GDN_REQUIRE_PRECOMPILED", "1")
    with pytest.raises(RuntimeError, match="deployment forbids GDN JIT"):
        persistent._get_module((5120, 48, 5120, 8, 24, 128, 4, 3, 32))


def test_incompatible_gdn_wheel_does_not_fall_back_to_jit(monkeypatch):
    import sys

    from vllm.mach.mxfp6.gdn import persistent

    def incompatible(*args):
        raise RuntimeError("ABI mismatch")

    monkeypatch.setattr(persistent, "_modules", {})
    monkeypatch.setitem(
        sys.modules, "vllm_mach_gdn", SimpleNamespace(load=incompatible)
    )
    monkeypatch.setenv("VLLM_MACH_GDN_REQUIRE_PRECOMPILED", "0")
    with pytest.raises(RuntimeError, match="ABI mismatch"):
        persistent._get_module((5120, 48, 5120, 8, 24, 128, 4, 3, 32))


def test_default_startup_does_not_touch_baseline_model(monkeypatch):
    """No adapter flag means no model traversal, patching or config access."""
    from vllm.mach.mxfp6.warmup import prepare_model

    for name in (
        "VLLM_MACH_GDN_PERSISTENT",
        "VLLM_MACH_GDN_BA_OVERLAP",
        "VLLM_MACH_FUSED_SWIGLU_QUANT",
        "VLLM_MACH_FUSED_GDN_QUANT",
        "VLLM_MACH_FUSED_AR_QUANT",
        "VLLM_MACH_MOE_PROJECTION_TUNING",
        "VLLM_SM120_OWNER_PREFILL",
        "VLLM_SM120_LOSSLESS_PREFILL",
    ):
        monkeypatch.delenv(name, raising=False)
    prepare_model(object(), object())


@pytest.fixture
def dense_auto_config(monkeypatch):
    from vllm import envs
    from vllm.config import CompilationMode
    from vllm.mach.mxfp6 import warmup

    for name in (
        *warmup.DENSE_FLAGS,
        "VLLM_MACH_DENSE_AUTO",
        "VLLM_USE_BREAKABLE_CUDAGRAPH",
        "VLLM_QWEN3_5_FP16_SSM",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(envs, "_mach_dense_defaults", False)
    monkeypatch.setattr(envs, "_mach_dense_fp16_defaults", False)
    monkeypatch.setattr(warmup, "find_spec", lambda _: object())
    monkeypatch.setattr(warmup.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(warmup.current_platform, "is_device_capability", lambda _: True)
    ns = SimpleNamespace
    return ns(
        model_config=ns(
            dtype=torch.bfloat16,
            enforce_eager=False,
            hf_text_config=ns(
                model_type="qwen3_5_text",
                hidden_size=5120,
                num_hidden_layers=64,
                linear_num_key_heads=16,
                linear_num_value_heads=48,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_conv_kernel_dim=4,
            ),
        ),
        parallel_config=ns(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=False,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
            use_ubatching=False,
        ),
        quant_config=ns(
            get_name=lambda: "quark",
            quant_config={
                "global_quant_config": {
                    "weight": {"dtype": "fp6_e3m2"},
                    "input_tensors": {"dtype": "fp8_e4m3"},
                }
            },
        ),
        cache_config=ns(
            mamba_ssm_cache_dtype="float32",
            mamba_cache_dtype="auto",
            enable_prefix_caching=False,
            mamba_cache_mode="none",
            use_replayssm=False,
        ),
        compilation_config=ns(
            backend="inductor",
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=None,
            cudagraph_capture_sizes=[1, 4, 32],
        ),
        profiler_config=ns(profiler=None),
        speculative_config=None,
        lora_config=None,
        kv_transfer_config=None,
        use_v2_model_runner=True,
        additional_config={},
    )


@pytest.mark.parametrize("state_dtype", ["float32", "float16"])
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_dense_auto_selects_bundle_without_exporting_environment(
    dense_auto_config, state_dtype, prefix_caching
):
    import os

    from vllm import envs
    from vllm.config import CompilationMode, CUDAGraphMode
    from vllm.mach.mxfp6 import warmup

    before = dict(os.environ)
    dense_auto_config.cache_config.mamba_ssm_cache_dtype = state_dtype
    dense_auto_config.cache_config.enable_prefix_caching = prefix_caching
    cache_mode = "align" if prefix_caching else "none"
    dense_auto_config.cache_config.mamba_cache_mode = cache_mode
    warmup.configure_dense_defaults(dense_auto_config)
    assert dense_auto_config.cache_config.enable_prefix_caching == prefix_caching
    assert dense_auto_config.cache_config.mamba_cache_mode == cache_mode
    assert dense_auto_config.cache_config.mamba_ssm_cache_dtype == state_dtype
    assert (state_dtype == "float16") == envs.VLLM_QWEN3_5_FP16_SSM
    assert dense_auto_config.compilation_config.mode == CompilationMode.NONE
    assert (
        dense_auto_config.compilation_config.cudagraph_mode
        == CUDAGraphMode.FULL_DECODE_ONLY
    )
    assert dense_auto_config.compilation_config.cudagraph_capture_sizes == [1, 4, 32]
    for name in warmup.DENSE_FLAGS:
        assert getattr(envs, name) in (True, "1")
    from vllm.mach.mxfp6.gdn_decode import enabled

    assert enabled("PERSISTENT") and enabled("BA_OVERLAP")
    assert dict(os.environ) == before
    envs.set_mach_dense_defaults(False)
    warmup.activate_dense_defaults(dense_auto_config)
    assert envs.VLLM_MACH_GDN_PERSISTENT
    assert (state_dtype == "float16") == envs.VLLM_QWEN3_5_FP16_SSM
    warmup.activate_dense_defaults(SimpleNamespace(additional_config={}))
    assert not envs.VLLM_MACH_GDN_PERSISTENT
    assert not envs.VLLM_QWEN3_5_FP16_SSM


@pytest.mark.parametrize("state_dtype", ["float32", "float16"])
def test_dense_auto_refreshes_cached_worker_defaults(dense_auto_config, state_dtype):
    from vllm import envs
    from vllm.mach.mxfp6 import warmup

    dense_auto_config.cache_config.mamba_ssm_cache_dtype = state_dtype
    envs.enable_envs_cache()
    try:
        assert not envs.VLLM_MACH_GDN_PERSISTENT
        warmup.configure_dense_defaults(dense_auto_config)
        assert envs.VLLM_MACH_GDN_PERSISTENT
        assert (state_dtype == "float16") == envs.VLLM_QWEN3_5_FP16_SSM
        warmup.activate_dense_defaults(SimpleNamespace(additional_config={}))
        assert not envs.VLLM_MACH_GDN_PERSISTENT
        assert not envs.VLLM_QWEN3_5_FP16_SSM
    finally:
        envs.disable_envs_cache()


@pytest.mark.parametrize(
    "flag",
    [
        "VLLM_MACH_DENSE_AUTO",
        "VLLM_QWEN3_5_FUSED_AR_NORM",
        "VLLM_MACH_GDN_PERSISTENT",
        "VLLM_MACH_GDN_BA_OVERLAP",
        "VLLM_MACH_FUSED_SWIGLU_QUANT",
        "VLLM_MACH_FUSED_GDN_QUANT",
        "VLLM_MACH_FUSED_AR_QUANT",
        "VLLM_SM120_OWNER_PREFILL",
        "VLLM_SM120_LOSSLESS_PREFILL",
    ],
)
def test_dense_auto_respects_explicit_disable(monkeypatch, dense_auto_config, flag):
    from vllm import envs
    from vllm.config import CompilationMode
    from vllm.mach.mxfp6.warmup import configure_dense_defaults

    monkeypatch.setenv(flag, "0")
    configure_dense_defaults(dense_auto_config)
    assert dense_auto_config.additional_config == {}
    assert dense_auto_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert envs.mach_dense_default() == "0"


@pytest.mark.parametrize(
    "parent,name,value",
    [
        ("cache_config", "mamba_ssm_cache_dtype", "bfloat16"),
        ("cache_config", "enable_prefix_caching", True),
        ("cache_config", "mamba_cache_mode", "all"),
        ("cache_config", "use_replayssm", True),
        ("parallel_config", "tensor_parallel_size", 1),
        ("model_config", "enforce_eager", True),
        (None, "speculative_config", object()),
        (None, "lora_config", object()),
        (None, "use_v2_model_runner", False),
    ],
)
def test_dense_auto_rejects_unvalidated_config(dense_auto_config, parent, name, value):
    from vllm.config import CompilationMode
    from vllm.mach.mxfp6.warmup import configure_dense_defaults

    obj = getattr(dense_auto_config, parent) if parent else dense_auto_config
    setattr(obj, name, value)
    configure_dense_defaults(dense_auto_config)
    assert dense_auto_config.additional_config == {}
    assert dense_auto_config.compilation_config.mode == CompilationMode.VLLM_COMPILE


def test_dense_auto_fp16_respects_explicit_disable(monkeypatch, dense_auto_config):
    from vllm import envs
    from vllm.config import CompilationMode
    from vllm.mach.mxfp6.warmup import configure_dense_defaults

    dense_auto_config.cache_config.mamba_ssm_cache_dtype = "float16"
    monkeypatch.setenv("VLLM_QWEN3_5_FP16_SSM", "0")
    configure_dense_defaults(dense_auto_config)
    assert dense_auto_config.compilation_config.mode == CompilationMode.VLLM_COMPILE
    assert dense_auto_config.cache_config.mamba_ssm_cache_dtype == "float16"
    assert not envs.VLLM_QWEN3_5_FP16_SSM
    assert not envs.VLLM_MACH_GDN_PERSISTENT


def test_dense_auto_missing_wheel_preserves_compile(monkeypatch, dense_auto_config):
    from vllm.config import CompilationMode
    from vllm.mach.mxfp6 import warmup

    monkeypatch.setattr(warmup, "find_spec", lambda _: None)
    warmup.configure_dense_defaults(dense_auto_config)
    assert dense_auto_config.additional_config == {}
    assert dense_auto_config.compilation_config.mode == CompilationMode.VLLM_COMPILE


def test_adapters_reject_speculative_model_before_mutation(monkeypatch):
    from vllm.mach.mxfp6.warmup import prepare_model

    monkeypatch.setenv("VLLM_MACH_GDN_PERSISTENT", "1")
    with pytest.raises(ValueError, match="non-speculative"):
        prepare_model(object(), SimpleNamespace(speculative_config=object()))


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("hidden,heads", [(5120, 48), (2048, 32), (2560, 32)])
@pytest.mark.parametrize("speculative", [False, True])
def test_fp16_ssm_admission_is_opt_in_and_geometry_limited(
    monkeypatch, enabled, hidden, heads, speculative
):
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    monkeypatch.setenv("VLLM_QWEN3_5_FP16_SSM", str(int(enabled)))
    monkeypatch.setattr(gdn.current_platform, "has_device_capability", lambda _: True)
    monkeypatch.setattr(
        torch.ops, "_C", SimpleNamespace(fused_gdn_decode_post_conv_mtp=object())
    )
    layer: Any = SimpleNamespace(
        get_state_dtype=lambda: (torch.bfloat16, torch.float16),
        _is_sm120=True,
        tp_size=2,
        hidden_size=hidden,
        num_v_heads=heads,
        num_k_heads=16,
        head_k_dim=128,
        head_v_dim=128,
        enable_packed_recurrent_decode=True,
        gqa_interleaved_layout=False,
        norm=SimpleNamespace(activation="silu"),
    )
    config: Any = SimpleNamespace(
        speculative_config=object() if speculative else None,
        lora_config=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    reason = gdn.QwenGatedDeltaNetAttention._fused_gdn_decode_unsupported_reason(
        layer, config
    )
    assert (reason is None) == (enabled and hidden in (5120, 2048) and not speculative)


def metadata(rows, **changes):
    return SimpleNamespace(
        **dict(
            dict(
                spec_sequence_masks=None,
                num_spec_decodes=0,
                num_prefills=0,
                num_decodes=rows,
                num_actual_tokens=rows,
                non_spec_state_indices_tensor=object(),
            ),
            **changes,
        )
    )


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 24, 32, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_paths_are_disjoint_and_dtype_guarded(rows, dtype):
    actual = select_path(rows, dtype, metadata(rows), persistent=True, overlap=True)
    expected = (
        "persistent"
        if rows in (1, 2, 4, 8) and dtype in (torch.float32, torch.float16)
        else "overlap"
        if rows in (16, 24, 32) and dtype in (torch.float32, torch.float16)
        else None
    )
    assert actual == expected


@pytest.mark.parametrize(
    "changes",
    [
        dict(num_prefills=1),
        dict(num_spec_decodes=1),
        dict(spec_sequence_masks=object()),
        dict(num_decodes=0),
        dict(num_actual_tokens=7),
        dict(non_spec_state_indices_tensor=None),
    ],
)
@pytest.mark.parametrize("rows", [4, 32])
def test_unsupported_metadata_falls_back(rows, changes):
    assert (
        select_path(
            rows,
            torch.float32,
            metadata(rows, **changes),
            persistent=True,
            overlap=True,
        )
        is None
    )


def test_disabled_and_missing_metadata():
    assert select_path(4, torch.float32, None, persistent=True, overlap=True) is None
    assert (
        select_path(4, torch.float32, metadata(4), persistent=False, overlap=False)
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        {"tp_size": 1},
        {"num_k_heads": 32},
        {"num_v_heads": 64},
        {"head_k_dim": 64},
        {"gqa_interleaved_layout": True},
        {"disable_tp_for_ba_proj": True},
        {"enable_fused_gdn_decode": False},
        {"enable_packed_recurrent_decode": False},
        {"activation": "relu"},
    ],
)
@pytest.mark.parametrize("hidden, heads", [(5120, 24), (2048, 16)])
def test_layer_geometry_and_native_semantics_guard(change, hidden, heads):
    from vllm.mach.mxfp6.gdn_decode import _eligible_layer

    layer = type("QwenGatedDeltaNetAttention", (), {})()
    vars(layer).update(
        tp_size=2,
        num_k_heads=16,
        num_v_heads=2 * heads,
        head_k_dim=128,
        head_v_dim=128,
        gqa_interleaved_layout=False,
        disable_tp_for_ba_proj=False,
        enable_fused_gdn_decode=True,
        enable_packed_recurrent_decode=True,
        activation="silu",
        in_proj_ba=SimpleNamespace(
            weight=torch.empty(2 * heads, hidden, dtype=torch.bfloat16)
        ),
        conv1d=SimpleNamespace(
            weight=torch.empty((16 + heads) * 128, 1, 4, dtype=torch.bfloat16)
        ),
        A_log=torch.empty(heads),
        dt_bias=torch.empty(heads, dtype=torch.bfloat16),
        norm=SimpleNamespace(weight=torch.empty(128)),
    )
    assert _eligible_layer(layer)
    vars(layer).update(change)
    assert not _eligible_layer(layer)


def test_persistent_state_dtype_is_part_of_jit_and_scratch_key():
    from vllm.mach.mxfp6.gdn import persistent

    x = torch.empty(4, 5120)
    ba = torch.empty(5120, 48)
    qkv = torch.empty(4, 5120)
    conv = torch.empty(8, 5120, 3)
    weight = torch.empty(5120, 4)
    al = torch.empty(24)
    keys = []
    for dtype in (torch.float32, torch.float16):
        state = torch.empty(8, 24, 128, 128, dtype=dtype)
        key = persistent._geometry_from_tensors(x, ba, qkv, weight, conv, al, state)
        keys.append(key)
    assert keys[0][:-1] == keys[1][:-1]
    assert keys[0][-1] == 32 and keys[1][-1] == 16
    assert persistent._scratch_key(keys[0], x, conv) != persistent._scratch_key(
        keys[1], x, conv
    )
    with pytest.raises(ValueError, match="FP16 or FP32"):
        persistent._geometry_from_tensors(
            x, ba, qkv, weight, conv, al, state.bfloat16()
        )


@pytest.mark.parametrize(
    "hidden, total_heads, expected",
    [
        (5120, 48, (5120, 24, 5120)),
        (2048, 32, (2048, 16, 4096)),
    ],
)
def test_tp2_geometry_keeps_hidden_and_qkv_widths_distinct(
    hidden, total_heads, expected
):
    from vllm.mach.mxfp6.gdn_decode import _layer_geometry

    layer = SimpleNamespace(
        in_proj_ba=SimpleNamespace(weight=torch.empty(total_heads, hidden)),
        num_v_heads=total_heads,
        num_k_heads=16,
    )
    assert _layer_geometry(layer) == expected


def test_graph_readiness_does_not_reuse_dense_geometry(monkeypatch):
    from vllm.mach.mxfp6.gdn import persistent

    x = torch.empty(4, 2048)
    conv = torch.empty(6, 4096, 3)
    signature = dict(
        hidden=2048,
        n_ba=32,
        qkv_dim=4096,
        h_q=8,
        hv=16,
        d=128,
        conv_width=4,
        conv_state_len=3,
        state_bits=32,
    )
    key = persistent.geometry_key(signature)
    dense_key = (5120, 48, 5120, 8, 24, 128, 4, 3, 32)
    modules = {dense_key: object()}
    scratch = {persistent._scratch_key(dense_key, x, conv): object()}
    monkeypatch.setattr(persistent, "_modules", modules)
    monkeypatch.setattr(persistent, "_scratch_cache", scratch)
    monkeypatch.setattr(persistent, "_barrier_cache", {"cpu": object()})
    assert not persistent.ready_for_graph_capture(signature, x, conv, 128**-0.5)
    modules[key] = object()
    assert not persistent.ready_for_graph_capture(signature, x, conv, 128**-0.5)
    scratch[persistent._scratch_key(key, x, conv)] = object()
    assert persistent.ready_for_graph_capture(signature, x, conv, 128**-0.5)


@pytest.mark.parametrize("hidden, heads", [(5120, 24), (2048, 16)])
@pytest.mark.parametrize("rows", [4, 16, 24, 32])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.float16])
@torch.inference_mode()
def test_gpu_adapter_matches_native_decode(
    monkeypatch, hidden, heads, rows, state_dtype
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    import vllm.forward_context
    from vllm.mach.mxfp6.gdn_decode import _forward
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    torch.manual_seed(20260917)
    qkv_dim = (16 + heads) * 128

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.1

    x = rand(rows, hidden)
    weight = rand(2 * heads, hidden)
    mixed = rand(rows, qkv_dim + heads * 128)
    cw, cb = rand(qkv_dim, 4), rand(qkv_dim)
    conv = rand(rows + 1, 3, qkv_dim).transpose(1, 2)
    if is_conv_state_dim_first():
        conv = conv.contiguous()
    state = rand(rows + 1, heads, 128, 128).to(state_dtype)
    reference_conv, reference_state = conv.clone(), state.clone()
    indices = torch.arange(1, rows + 1, device="cuda", dtype=torch.int32)
    md = metadata(rows, non_spec_state_indices_tensor=indices)
    monkeypatch.setattr(
        vllm.forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata={"test": md}),
    )

    class BA:
        def __init__(self):
            self.weight = weight

        def __call__(self, value):
            return torch.nn.functional.linear(value, self.weight), None

    # A deterministic gated output makes QKV/Z splitting observable without
    # testing vLLM's unrelated RMSNorm implementation again.
    def norm(core, z, out):
        out.copy_(core * torch.nn.functional.silu(z))

    layer = SimpleNamespace(
        num_k_heads=16,
        num_v_heads=heads * 2,
        in_proj_ba=BA(),
        in_proj_qkvz=lambda _: (mixed, None),
        split_ba=lambda ba: ba.chunk(2, -1),
        conv1d=SimpleNamespace(weight=cw, bias=cb),
        _mach_gdn_ba=weight.T.contiguous(),
        _mach_gdn_bias=cb,
        A_log=torch.zeros(heads, device="cuda"),
        dt_bias=rand(heads),
        activation="silu",
        prefix="test",
        kv_cache=(conv if is_conv_state_dim_first() else conv.transpose(-1, -2), state),
        _rms_norm_gated_cuda=norm,
        out_proj=lambda value: (value, None),
    )
    aux = torch.cuda.Stream() if rows >= 16 else None

    def unexpected_fallback(_):
        pytest.fail("supported geometry unexpectedly fell back")

    # Native convolution overwrites the packed QKV view in place.
    reference_mixed = mixed.clone()
    actual = _forward(layer, unexpected_fallback, True, aux, x)
    ba, _ = layer.in_proj_ba(x)
    b, a = ba.chunk(2, -1)
    qkv, z = reference_mixed.split([qkv_dim, heads * 128], -1)
    cq = causal_conv1d_update(
        qkv,
        reference_conv,
        cw,
        cb,
        "silu",
        conv_state_indices=indices,
        validate_data=False,
    )
    expected = torch.empty(rows, 1, heads, 128, device="cuda", dtype=torch.bfloat16)
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=cq,
        a=a.contiguous(),
        b=b.contiguous(),
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        scale=128**-0.5,
        initial_state=reference_state,
        out=expected,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )
    norm(expected, z.reshape_as(expected), expected)
    torch.accelerator.synchronize()
    assert torch.equal(conv, reference_conv)
    for result, reference in ((actual, expected.flatten(1)), (state, reference_state)):
        relative = (
            result.float() - reference.float()
        ).norm() / reference.float().norm()
        assert relative < 0.02
        if aux is not None:
            assert torch.equal(result, reference)
