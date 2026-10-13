# SPDX-License-Identifier: Apache-2.0
"""Routing, safe defaults and actual mixed-core splice admission."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest

from vllm import envs
from vllm.config import CUDAGraphMode as Mode
from vllm.mach.mxfp6 import graph_policy, two_b_mixed_gdn
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor as Desc


@pytest.fixture
def auto_worker(monkeypatch):
    import torch

    from vllm.mach.mxfp6 import auto_defaults

    for name in envs.environment_variables:
        if name.startswith(("VLLM_MACH_", "VLLM_MXFP6_", "VLLM_HYBRID_")):
            monkeypatch.delenv(name, raising=False)
    envs.set_mach_mxfp8_defaults({})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (12, 0))
    monkeypatch.setattr(auto_defaults, "_available", lambda *_: True)
    cfg = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=None,
        lora_config=None,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        kernel_config=SimpleNamespace(linear_backend="auto"),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            quantization="compressed-tensors",
            hf_config=SimpleNamespace(
                quantization_config={
                    "format": "mxfp8-quantized",
                    "config_groups": {
                        "linear": {
                            "weights": {
                                "num_bits": 8,
                                "group_size": 32,
                                "type": "float",
                            }
                        }
                    },
                }
            ),
            hf_text_config=SimpleNamespace(
                model_type="qwen3_5_text",
                hidden_size=2560,
                intermediate_size=9216,
                num_hidden_layers=32,
            ),
        ),
    )
    yield SimpleNamespace(vllm_config=cfg, device="cuda:0")
    envs.set_mach_mxfp8_defaults({})


@pytest.mark.parametrize("size", ["2b", "4b"])
def test_auto_admits_model_specific_kernels_without_environment(auto_worker, size):
    import os

    from vllm.mach.mxfp6 import auto_defaults

    hf = auto_worker.vllm_config.model_config.hf_text_config
    if size == "2b":
        hf.hidden_size, hf.intermediate_size, hf.num_hidden_layers = 2048, 6144, 24
    before = dict(os.environ)
    auto_defaults.activate(auto_worker)
    assert envs.VLLM_MACH_MXFP8_SWIGLU_QUANT
    assert (size == "2b") == envs.VLLM_MACH_MXFP8_2B_NATIVE
    assert ("all" if size == "4b" else "") == envs.VLLM_MACH_MXFP8_4B_NATIVE_POLICY
    assert not envs.VLLM_MXFP6_SM120_MXFP8_DUAL
    assert (size == "4b") == envs.VLLM_MXFP6_SM120_PROJECTION_PARALLEL
    assert not envs.VLLM_MACH_MXFP8_COMPILE_CHOICES
    assert not envs.VLLM_MACH_MXFP8_KV_SCALE_PROFILE
    assert not envs.VLLM_MACH_MXFP8_2B_PDL
    assert os.environ == before


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "enable_dbo", True),
        ("model_config", "quantization", "fp8"),
        ("model_config", "cpu_offload_gb", 1),
        ("kernel_config", "linear_backend", "mxfp6_sm120"),
        (None, "use_v2_model_runner", False),
        (None, "speculative_config", object()),
        (None, "lora_config", object()),
    ],
)
def test_auto_rejects_unsupported_engine_and_resets_stale_defaults(
    auto_worker, section, key, value
):
    from vllm.mach.mxfp6 import auto_defaults

    auto_defaults.activate(auto_worker)
    cfg = auto_worker.vllm_config
    setattr(getattr(cfg, section) if section else cfg, key, value)
    auto_defaults.activate(auto_worker)
    assert auto_worker._mach_auto_defaults == {}
    assert auto_worker._mach_auto_size is None
    assert not envs.VLLM_MACH_MXFP8_SWIGLU_QUANT


def test_auto_checks_quantization_not_just_model_name(auto_worker):
    from vllm.mach.mxfp6 import auto_defaults

    mc = auto_worker.vllm_config.model_config
    mc.hf_config.quantization_config["format"] = "pack-quantized"
    auto_defaults.activate(auto_worker)
    assert not auto_worker._mach_auto_defaults


def test_auto_native_dependency_failure_keeps_independent_fusion(
    auto_worker, monkeypatch
):
    from vllm.mach.mxfp6 import auto_defaults

    monkeypatch.setattr(auto_defaults, "_available", lambda *_: False)
    auto_defaults.activate(auto_worker)
    assert auto_worker._mach_auto_defaults == {"VLLM_MACH_MXFP8_SWIGLU_QUANT": True}


def test_auto_explicit_disables_survive_cache_and_admission(auto_worker, monkeypatch):
    from vllm.mach.mxfp6 import auto_defaults

    monkeypatch.setenv("VLLM_MXFP6_SM120_MXFP8_NATIVE", "0")
    monkeypatch.setenv("VLLM_MACH_MXFP8_SWIGLU_QUANT", "0")
    cached = envs._is_envs_cache_enabled()
    try:
        envs.enable_envs_cache()
        auto_defaults.activate(auto_worker)
        assert not envs.VLLM_MACH_MXFP8_SWIGLU_QUANT
        assert not envs.VLLM_MACH_MXFP8_4B_NATIVE_POLICY
        assert not envs.VLLM_MXFP6_SM120_MXFP8_DUAL
    finally:
        if not cached:
            envs.disable_envs_cache()


def test_auto_preserves_user_hybrid_head(auto_worker, monkeypatch):
    from vllm.mach.mxfp6 import auto_defaults

    auto_defaults.activate(auto_worker)
    monkeypatch.setenv("VLLM_HYBRID_NVFP4_LM_HEAD", "1")
    # No runner/head access is permitted for this conflicting selection.
    auto_defaults.activate_head(auto_worker)
    assert not envs.VLLM_MACH_MXFP8_GREEDY_HEAD
    assert not envs.VLLM_MACH_MXFP8_FULL_HEAD


def test_auto_four_b_policy_accepts_explicit_zero(auto_worker, monkeypatch):
    from vllm.mach.mxfp6 import auto_defaults

    monkeypatch.setenv("VLLM_MACH_MXFP8_4B_NATIVE_POLICY", "0")
    auto_defaults.activate(auto_worker)
    assert envs.VLLM_MACH_MXFP8_4B_NATIVE_POLICY == ""
    assert not envs.VLLM_MXFP6_SM120_MXFP8_DUAL


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_auto_dual_remains_explicit_opt_in(auto_worker, monkeypatch, enabled):
    from vllm.mach.mxfp6 import auto_defaults

    monkeypatch.setenv("VLLM_MXFP6_SM120_MXFP8_DUAL", enabled)
    auto_defaults.activate(auto_worker)
    assert (enabled == "1") == envs.VLLM_MXFP6_SM120_MXFP8_DUAL
    assert envs.VLLM_MACH_MXFP8_DUAL_POLICY == "mach"
    assert envs.VLLM_MACH_MXFP8_4B_NATIVE_POLICY == "all"
    assert envs.VLLM_MXFP6_SM120_PROJECTION_PARALLEL
    assert envs.VLLM_MACH_MXFP8_SWIGLU_QUANT


def test_auto_graph_without_captured_graph_keeps_stock(auto_worker):
    from vllm.mach.mxfp6 import auto_defaults

    auto_defaults.activate(auto_worker)
    auto_worker.model_runner = SimpleNamespace(cudagraph_manager=None)
    auto_defaults.activate_graph(auto_worker)
    assert not envs.VLLM_MACH_MXFP8_GRAPH_POLICY


def test_auto_effective_defaults_isolate_aot_compile_cache(auto_worker, monkeypatch):
    from vllm.mach.mxfp6 import auto_defaults

    baseline = envs.compile_factors()
    auto_defaults.activate(auto_worker)
    enabled = envs.compile_factors()
    assert enabled != baseline
    assert enabled["VLLM_MACH_MXFP8_4B_NATIVE_POLICY"] == "all"
    assert enabled["VLLM_MACH_MXFP8_SWIGLU_QUANT"] is True
    assert enabled["VLLM_MACH_MXFP8_DUAL_POLICY"] == "mach"
    assert enabled["VLLM_MXFP6_SM120_MXFP8_DUAL"] is False
    monkeypatch.setenv("VLLM_MXFP6_SM120_MXFP8_DUAL", "1")
    envs.set_mach_mxfp8_defaults(auto_worker._mach_auto_defaults)
    explicit_dual = envs.compile_factors()
    assert explicit_dual["VLLM_MXFP6_SM120_MXFP8_DUAL"] is True
    assert explicit_dual != enabled
    monkeypatch.setenv("VLLM_MACH_MXFP8_SWIGLU_QUANT", "0")
    envs.set_mach_mxfp8_defaults(auto_worker._mach_auto_defaults)
    assert envs.compile_factors()["VLLM_MACH_MXFP8_SWIGLU_QUANT"] is False


@pytest.mark.parametrize("disable_full", [False, True])
def test_auto_tied_head_requires_its_full_fallback(
    auto_worker, monkeypatch, disable_full
):
    import torch

    from vllm.mach.mxfp6 import auto_defaults
    from vllm.utils import flashinfer

    auto_defaults.activate(auto_worker)
    monkeypatch.setattr(flashinfer, "has_flashinfer_b12x_gemm", lambda: True)
    weight = SimpleNamespace(
        shape=(248320, 2560),
        dtype=torch.bfloat16,
        is_cuda=True,
        is_contiguous=lambda: True,
        data_ptr=lambda: 123,
    )
    auto_worker.model_runner = SimpleNamespace(
        batch_sharder=None,
        model=SimpleNamespace(
            lm_head=SimpleNamespace(weight=weight),
            model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight)),
            logits_processor=SimpleNamespace(org_vocab_size=248320, head_dtype=None),
        ),
    )
    if disable_full:
        monkeypatch.setenv("VLLM_MACH_MXFP8_FULL_HEAD", "0")
    auto_defaults.activate_head(auto_worker)
    assert envs.VLLM_MACH_MXFP8_GREEDY_HEAD is not disable_full
    assert envs.VLLM_MACH_MXFP8_FULL_HEAD is not disable_full


@pytest.mark.parametrize("captured", [False, True])
def test_auto_graph_requires_real_exact2048_capture(auto_worker, monkeypatch, captured):
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.mach.mxfp6 import auto_defaults

    auto_defaults.activate(auto_worker)
    auto_worker.model_runner = SimpleNamespace(
        cudagraph_manager=SimpleNamespace(
            _graphs_captured=True,
            cudagraph_mode=Mode.FULL_AND_PIECEWISE,
        )
    )
    wrapper = SimpleNamespace(
        vllm_config=auto_worker.vllm_config,
        runtime_mode=Mode.PIECEWISE,
        concrete_cudagraph_entries={
            Desc(Mode.PIECEWISE, 2048, None): SimpleNamespace(
                cudagraph=object() if captured else None
            )
        },
    )
    monkeypatch.setattr(CUDAGraphWrapper, "_all_instances", [wrapper])
    auto_defaults.activate_graph(auto_worker)
    assert envs.VLLM_MACH_MXFP8_GRAPH_POLICY is captured


@pytest.mark.parametrize("m", [1, 4, 16, 24, 32, 64, 128, 2048])
@pytest.mark.parametrize("n", [2560, 12288, 18432])
def test_mach_dual_policy_is_not_shpgy_all_small_rows(m, n):
    from vllm.mach.mxfp6.dual import dual_selected

    assert dual_selected(m, n, mach=True) == (
        (n == 18432 and m == 32) or (n == 12288 and m in (32, 64))
    )


def test_full_head_only_replaces_candidate_scores():
    """Full vocabulary is preserved, including approximate non-candidates."""
    import torch

    from vllm.mach.mxfp6.full_head import full_logits

    coarse = torch.arange(4096, dtype=torch.float32).repeat(2, 1)
    state = SimpleNamespace(
        coarse_logits=lambda *_: coarse.clone(),
        refine_logits=lambda hidden, weight, ids, bias: ids.float() + 10,
    )
    actual = full_logits(None, None, state)
    assert torch.equal(actual[:, :2048], coarse[:, :2048])
    assert torch.equal(actual[:, 2048:], coarse[:, 2048:] + 10)


@pytest.mark.parametrize(
    "policy,n,expected",
    [
        ("all", 2560, True),
        ("all", 18432, True),
        ("wide-only", 2560, False),
        ("wide-only", 18432, True),
    ],
)
def test_four_b_native_projection_policy(policy, n, expected):
    from vllm.mach.mxfp6.four_b_native import selected

    assert selected(policy, n) is expected


def test_four_b_native_policy_rejects_unknown_value():
    from vllm.mach.mxfp6.four_b_native import selected

    with pytest.raises(ValueError, match="all or wide-only"):
        selected("automatic", 2560)


@pytest.mark.parametrize("compact", ["NVFP4", "MXFP4", "MXFP8"])
def test_full_head_rejects_compact_combination_before_allocating(monkeypatch, compact):
    from vllm.mach.mxfp6.full_head import prepare

    for name in ("NVFP4", "MXFP4", "MXFP8"):
        monkeypatch.setattr(envs, f"VLLM_HYBRID_{name}_LM_HEAD", name == compact)
    cfg = SimpleNamespace(
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        speculative_config=None,
        lora_config=None,
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_text")
        ),
    )
    with pytest.raises(ValueError, match="exclusive"):
        prepare(SimpleNamespace(vllm_config=cfg))


@pytest.mark.parametrize("hidden,threshold", [(2048, 160), (2560, 256)])
@pytest.mark.parametrize("tokens", [1, 4, 16, 32, 64, 160, 161, 256, 257, 2047, 2048])
def test_graph_policy_preserves_small_batches_and_unpads_large_tails(
    hidden, threshold, tokens
):
    desc = Desc(Mode.PIECEWISE, 2048, None)
    result = graph_policy.apply_descriptor(desc, tokens, 1, hidden_size=hidden)
    if threshold < tokens < 2048:
        assert result.cg_mode == Mode.NONE
        assert result.num_tokens == tokens
    else:
        assert result is desc


def test_exact2048_requires_real_piecewise_descriptor():
    with pytest.raises(RuntimeError, match="PIECEWISE2048"):
        graph_policy.apply_descriptor(
            Desc(Mode.NONE, 2048, 1), 2048, 1, hidden_size=2048
        )


def test_new_features_default_off(monkeypatch):
    for name in (
        "VLLM_MACH_MXFP8_2B_GDN_ORDERED",
        "VLLM_MACH_MXFP8_2B_GDN_MIXED",
        "VLLM_MACH_MXFP8_4B_GDN_FULL",
        "VLLM_MACH_MXFP8_COMPILE_CHOICES",
        "VLLM_MACH_MXFP8_GRAPH_POLICY",
        "VLLM_MACH_MXFP8_KV_SCALE_PROFILE",
        "VLLM_MACH_MXFP8_BF16_SSM_AUDIT",
        "VLLM_MACH_MXFP8_HEAD_AUDIT",
    ):
        monkeypatch.delenv(name, raising=False)
        assert not envs.environment_variables[name]()


def test_mixed_splice_preserves_current_core_except_recurrent_branch(monkeypatch):
    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn

    original = gdn.QwenGatedDeltaNetAttention._forward_core
    clone = two_b_mixed_gdn._clone_core(original, vars(gdn))
    assert inspect.signature(clone) == inspect.signature(original)
    # Compare the exact current source, not a synthetic shape-compatible core.
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    fn = tree.body[0]
    branches = [
        node
        for node in fn.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "split_non_spec"
    ]
    assert len(branches) == 1
    assert two_b_mixed_gdn._ORIGINAL_SHA != two_b_mixed_gdn._CLONE_SHA


def test_two_b_reset_ids_are_strict_and_keep_zero_sentinel_out():
    from vllm.mach.mxfp6.two_b_ordered_gdn import _flatten_new

    assert _flatten_new(([0, 1, 3], [2])) == [1, 3, 2]
    assert _flatten_new(None) == []
    with pytest.raises(TypeError):
        _flatten_new(([True],))
    with pytest.raises(TypeError):
        _flatten_new([[1]])
