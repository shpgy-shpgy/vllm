# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm import envs
from vllm.mach.mxfp6.upstream_runtime import _four_b_bf16_scratch_enabled


def test_four_b_scratch_requires_prepared_ordered_gdn(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MACH_MXFP8_4B_GDN_FULL", False)
    text = SimpleNamespace(
        model_type="qwen3_5_text",
        hidden_size=2560,
        intermediate_size=9216,
        num_hidden_layers=32,
    )
    model = SimpleNamespace(_mach_ordered_gdn_prepared=False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        cache_config=SimpleNamespace(mamba_ssm_cache_dtype="bfloat16"),
    )
    worker = SimpleNamespace(vllm_config=config, get_model=lambda: model)

    assert not _four_b_bf16_scratch_enabled(worker)
    model._mach_ordered_gdn_prepared = True
    assert _four_b_bf16_scratch_enabled(worker)
    config.parallel_config.tensor_parallel_size = 2
    assert not _four_b_bf16_scratch_enabled(worker)
    config.parallel_config.tensor_parallel_size = 1
    config.cache_config.mamba_ssm_cache_dtype = "float32"
    assert not _four_b_bf16_scratch_enabled(worker)
    monkeypatch.setattr(envs, "VLLM_MACH_MXFP8_4B_GDN_FULL", True)
    assert _four_b_bf16_scratch_enabled(worker)
    model._mach_ordered_gdn_prepared = False
    assert not _four_b_bf16_scratch_enabled(worker)
