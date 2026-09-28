# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock, patch


def test_qwen3_5_lm_head_receives_quant_config():
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.scheduler_config = Mock()
    mock_vllm_config.quant_config = mock_quant_config
    mock_vllm_config.lora_config = None

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5.Qwen3_5Model") as MockModel,
        patch("vllm.model_executor.models.qwen3_5.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        MockModel.return_value.make_empty_intermediate_tensors = Mock()

        Qwen3_5ForCausalLMBase(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_qwen3_5_mtp_lm_head_receives_quant_config():
    from vllm.config import CompilationMode
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP

    mock_quant_config = Mock()

    mock_hf_config = Mock()
    mock_hf_config.tie_word_embeddings = False
    mock_hf_config.vocab_size = 128
    mock_hf_config.hidden_size = 64

    mock_vllm_config = Mock()
    mock_vllm_config.model_config.hf_text_config = mock_hf_config
    mock_vllm_config.cache_config.mamba_cache_mode = "align"
    mock_vllm_config.compilation_config.mode = CompilationMode.NONE
    mock_vllm_config.quant_config = mock_quant_config

    mock_pp_group = Mock()
    mock_pp_group.is_last_rank = True

    with (
        patch("vllm.model_executor.models.qwen3_5_mtp.Qwen3_5MultiTokenPredictor"),
        patch("vllm.model_executor.models.qwen3_5_mtp.ParallelLMHead") as MockLMHead,
        patch("vllm.model_executor.models.qwen3_5_mtp.LogitsProcessor"),
        patch(
            "vllm.model_executor.models.qwen3_5_mtp.get_pp_group",
            return_value=mock_pp_group,
        ),
    ):
        Qwen3_5MTP(vllm_config=mock_vllm_config)

        MockLMHead.assert_called_once()
        call_kwargs = MockLMHead.call_args.kwargs
        assert call_kwargs["quant_config"] is mock_quant_config


def test_manual_ar_norm_rejects_unsupported_execution_modes(monkeypatch):
    """Delayed row reductions must not escape into PP, LoRA, MoE, or Eagle."""
    from types import SimpleNamespace as NS

    import torch

    from vllm.config import CompilationMode
    from vllm.model_executor.models import qwen3_5 as model

    monkeypatch.setenv("VLLM_QWEN3_5_FUSED_AR_NORM", "1")
    cfg = NS(
        model_config=NS(
            dtype=torch.bfloat16,
            hf_text_config=NS(
                model_type="qwen3_5_text", hidden_size=5120, layer_scale=False
            ),
        ),
        parallel_config=NS(tensor_parallel_size=2, pipeline_parallel_size=1),
        compilation_config=NS(mode=CompilationMode.NONE),
        quant_config=NS(get_name=lambda: "quark"),
        speculative_config=None,
        lora_config=None,
    )
    with patch.object(model, "current_platform") as platform:
        platform.is_cuda.return_value = True
        platform.is_device_capability.return_value = True
        assert model._use_mxfp6_ar_norm(cfg)
        for obj, attr, value in (
            (cfg, "speculative_config", object()),
            (cfg, "lora_config", object()),
            (cfg.parallel_config, "pipeline_parallel_size", 2),
            (cfg.model_config.hf_text_config, "model_type", "qwen3_5_moe_text"),
            (cfg.model_config.hf_text_config, "layer_scale", True),
            (cfg.compilation_config, "mode", CompilationMode.VLLM_COMPILE),
        ):
            old = getattr(obj, attr)
            setattr(obj, attr, value)
            assert not model._use_mxfp6_ar_norm(cfg)
            setattr(obj, attr, old)
        monkeypatch.setenv("VLLM_QWEN3_5_FUSED_AR_NORM", "0")
        assert not model._use_mxfp6_ar_norm(cfg)
        monkeypatch.delenv("VLLM_QWEN3_5_FUSED_AR_NORM")
        assert not model._use_mxfp6_ar_norm(cfg)


def test_moe_ar_norm_defaults_on_only_for_supported_models(monkeypatch):
    """The measured MoE path defaults on; overrides and admission still apply."""
    from types import SimpleNamespace as NS

    import torch

    from vllm.config import CompilationMode
    from vllm.mach.mxfp6 import moe_ar_norm as ar

    monkeypatch.delenv("VLLM_QWEN3_5_FUSED_AR_NORM", raising=False)
    config = NS(
        model_config=NS(
            dtype=torch.bfloat16,
            hf_text_config=NS(
                model_type="qwen3_5_moe_text",
                hidden_size=2048,
                num_experts=256,
                num_experts_per_tok=8,
                moe_intermediate_size=512,
                shared_expert_intermediate_size=512,
            ),
        ),
        parallel_config=NS(
            rank=0,
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
        quant_config=NS(
            get_name=lambda: "quark",
            quant_config={
                "global_quant_config": {
                    "weight": {"dtype": "fp6_e3m2"},
                    "input_tensors": {"dtype": "fp8_e4m3"},
                }
            },
        ),
        compilation_config=NS(mode=CompilationMode.VLLM_COMPILE),
        speculative_config=None,
        lora_config=None,
        additional_config={},
    )
    with patch.object(ar, "current_platform") as platform:
        platform.is_cuda.return_value = True
        platform.is_device_capability.return_value = True
        assert ar.use_moe_ar_norm(config)
        ar.prepare_compiled_ar_norm(config)
        assert config.additional_config["mach_compiled_ar_norm"]["rank"] == 0
        monkeypatch.setenv("VLLM_QWEN3_5_FUSED_AR_NORM", "0")
        assert not ar.use_moe_ar_norm(config)
        config.additional_config = {}
        ar.prepare_compiled_ar_norm(config)
        assert config.additional_config == {}
        monkeypatch.delenv("VLLM_QWEN3_5_FUSED_AR_NORM")
        for obj, name, value in (
            (config.parallel_config, "tensor_parallel_size", 1),
            (config.model_config.hf_text_config, "hidden_size", 2560),
            (config, "speculative_config", object()),
            (config, "lora_config", object()),
        ):
            previous = getattr(obj, name)
            setattr(obj, name, value)
            assert not ar.use_moe_ar_norm(config)
            ar.prepare_compiled_ar_norm(config)
            assert config.additional_config == {}
            setattr(obj, name, previous)
        platform.is_device_capability.return_value = False
        assert not ar.use_moe_ar_norm(config)
