# SPDX-License-Identifier: Apache-2.0
"""Model-scoped automatic admission, without changing serving configuration.

Calibration, fixed compiler recipes, PDL and state-pool replacements remain
explicit: a matching model shape alone does not qualify those policies.
"""

import importlib
import os

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def model_size(cfg):
    mc, pc = cfg.model_config, cfg.parallel_config
    if (
        mc is None
        or not cfg.use_v2_model_runner
        or cfg.speculative_config is not None
        or cfg.lora_config is not None
        or any(
            getattr(pc, k, 1) != 1
            for k in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
                "prefill_context_parallel_size",
                "decode_context_parallel_size",
            )
        )
        or getattr(pc, "enable_dbo", False)
        or getattr(pc, "use_ubatching", False)
        or getattr(mc, "cpu_offload_gb", 0)
        or getattr(mc, "enable_sleep_mode", False)
        or cfg.kernel_config.linear_backend not in ("auto", "flashinfer_cutlass")
        or mc.dtype != torch.bfloat16
        or mc.quantization != "compressed-tensors"
    ):
        return None
    hf = mc.hf_text_config
    quant = getattr(mc.hf_config, "quantization_config", None)
    if not isinstance(quant, dict) or quant.get("format") != "mxfp8-quantized":
        return None
    groups = quant.get("config_groups", {})
    if not groups or any(
        group.get("weights", {}).get("num_bits") != 8
        or group.get("weights", {}).get("group_size") != 32
        or group.get("weights", {}).get("type") != "float"
        for group in groups.values()
    ):
        return None
    if getattr(hf, "model_type", None) != "qwen3_5_text":
        return None
    shape = tuple(
        getattr(hf, k, None)
        for k in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
        )
    )
    return {(2048, 6144, 24): "2b", (2560, 9216, 32): "4b"}.get(shape)


def _available(module, function=None):
    try:
        imported = importlib.import_module(module)
        if function:
            getattr(imported, function)()
        return True
    except (ImportError, OSError, RuntimeError, AttributeError) as exc:
        logger.info("Mach auto: skipping unavailable %s: %s", module, exc)
        return False


def activate(worker):
    # Called after selecting the worker GPU, before constructing/loading a model.
    envs.set_mach_mxfp8_defaults({})
    worker._mach_auto_defaults = {}
    worker._mach_auto_size = None
    size = model_size(worker.vllm_config)
    if size is None or torch.cuda.get_device_capability(worker.device) != (12, 0):
        return
    values = {"VLLM_MACH_MXFP8_SWIGLU_QUANT": True}
    native = envs.VLLM_MXFP6_SM120_MXFP8_NATIVE and _available(
        "vllm.model_executor.kernels.linear.mxfp8.mxfp6_sm120",
        "load_mxfp6_sm120_mxfp8_libraries",
    )
    if native:
        if size == "4b":
            values["VLLM_MACH_MXFP8_4B_NATIVE_POLICY"] = "all"
            values["VLLM_MXFP6_SM120_PROJECTION_PARALLEL"] = True
            policy = (
                envs.VLLM_MACH_MXFP8_4B_NATIVE_POLICY
                if "VLLM_MACH_MXFP8_4B_NATIVE_POLICY" in os.environ
                else "all"
            )
            if policy == "all" and _available("vllm.mach.mxfp6.dual", "_load_dual_api"):
                # Same-command 4B BS64 ablation favors native without dual.
                # Keep the Mach policy for explicit opt-in, not auto-enablement.
                values["VLLM_MXFP6_SM120_MXFP8_DUAL"] = False
                values["VLLM_MACH_MXFP8_DUAL_POLICY"] = "mach"
        else:
            values["VLLM_MACH_MXFP8_2B_NATIVE"] = True
    if _available("vllm.mach.mxfp6.ba_gemv"):
        if size == "4b":
            values["VLLM_MXFP6_SM120_BA_GEMV"] = True
        elif native:
            values["VLLM_MACH_MXFP8_2B_BA"] = True
    worker._mach_auto_size = size
    _publish(worker, values)


def _publish(worker, values):
    # Do not export decisions through os.environ into a later engine/worker.
    worker._mach_auto_defaults = {
        name: value for name, value in values.items() if name not in os.environ
    }
    envs.set_mach_mxfp8_defaults(worker._mach_auto_defaults)
    logger.info("Mach automatic defaults: %s", worker._mach_auto_defaults)


def activate_head(worker):
    if not getattr(worker, "_mach_auto_size", None):
        return
    if any(
        (
            envs.VLLM_HYBRID_NVFP4_LM_HEAD,
            envs.VLLM_HYBRID_MXFP4_LM_HEAD,
            envs.VLLM_HYBRID_MXFP8_LM_HEAD,
            envs.VLLM_MACH_MXFP8_HEAD_AUDIT,
        )
    ):
        logger.info("Mach auto head: preserving explicitly selected hybrid/audit path")
        return
    from vllm.utils.flashinfer import has_flashinfer_b12x_gemm

    runner = worker.model_runner
    model = getattr(runner.model, "language_model", runner.model)
    head = getattr(model, "lm_head", None)
    body = getattr(model, "model", None)
    embedding = getattr(body, "embed_tokens", None)
    weight = getattr(head, "weight", None)
    processor = getattr(model, "logits_processor", None)
    if (
        weight is None
        or embedding is None
        or processor is None
        or runner.batch_sharder is not None
        or tuple(weight.shape) not in ((248320, 2048), (248320, 2560))
        or weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
        or getattr(weight, "_vllm_is_uva_offloaded", False)
        or weight.data_ptr() != embedding.weight.data_ptr()
        or processor.org_vocab_size != 248320
        or processor.head_dtype not in (None, torch.bfloat16)
        or envs.VLLM_HYBRID_NVFP4_LM_HEAD_BACKEND not in ("auto", "b12x")
        or not has_flashinfer_b12x_gemm()
    ):
        return
    values = dict(worker._mach_auto_defaults)
    if weight.shape[1] == 2560:
        if os.getenv("VLLM_MACH_MXFP8_FULL_HEAD") == "0":
            return
        values["VLLM_MACH_MXFP8_FULL_HEAD"] = True
    values["VLLM_MACH_MXFP8_GREEDY_HEAD"] = True
    _publish(worker, values)
    logger.info(
        "Mach auto head uses approximate NVFP4 candidate scoring, not BF16 equivalence"
    )


def activate_graph(worker):
    if not getattr(worker, "_mach_auto_size", None):
        return
    if "VLLM_MACH_MXFP8_GRAPH_POLICY" in os.environ:
        return
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode

    cfg = worker.vllm_config
    manager = worker.model_runner.cudagraph_manager
    if (
        manager is None
        or not manager._graphs_captured
        or manager.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
    ):
        return
    entries = [
        entry
        for wrapper in list(CUDAGraphWrapper._all_instances)
        if wrapper.vllm_config is cfg
        and wrapper.runtime_mode == CUDAGraphMode.PIECEWISE
        for desc, entry in wrapper.concrete_cudagraph_entries.items()
        if desc.num_tokens == 2048
    ]
    if entries and all(entry.cudagraph is not None for entry in entries):
        _publish(
            worker, {**worker._mach_auto_defaults, "VLLM_MACH_MXFP8_GRAPH_POLICY": True}
        )
