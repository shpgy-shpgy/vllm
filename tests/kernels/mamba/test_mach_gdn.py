# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native GDN routing must preserve unsupported and mixed-batch fallbacks."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.mach.mxfp6.gdn_decode import select_path


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
@torch.inference_mode()
def test_gpu_adapter_matches_native_decode(monkeypatch, hidden, heads, rows):
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
    state = rand(rows + 1, heads, 128, 128).float()
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
