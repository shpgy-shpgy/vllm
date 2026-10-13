# SPDX-License-Identifier: Apache-2.0
"""Independent lifecycle entrypoints for the remaining Mach ports.

No serving profile, model asset, dependency or capture-size override is applied.
"""

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def reserve_state_scratch(worker, available: int) -> int:
    """Budget separate W4 storage before the engine assigns KV-cache pages.

    Plan on copies with the original budget, then reserve that upper bound.
    The real (smaller) pool cannot require more scratch than this plan.
    """
    two_b = envs.VLLM_MACH_MXFP8_2B_GDN_ORDERED
    four_b = envs.VLLM_MACH_MXFP8_4B_GDN_FULL
    if not (two_b or four_b):
        return available
    from copy import copy, deepcopy

    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    cfg = worker.vllm_config
    parallel = cfg.parallel_config
    if any(
        getattr(parallel, name) != 1
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
        )
    ):
        raise ValueError("Deferred scratch budgeting currently requires TP/PP/DP=1")
    if cfg.cache_config.num_gpu_blocks_override is not None:
        raise ValueError("Fixed KV block count cannot reserve dynamic GDN scratch")
    # Compilation config holds live model parameters, which must never be
    # deep-copied. The planner only mutates model/cache configuration fields.
    planning = copy(cfg)
    planning.model_config = copy(cfg.model_config)
    planning.cache_config = copy(cfg.cache_config)
    plan = get_kv_cache_configs(
        planning, [deepcopy(worker.get_kv_cache_spec())], [available]
    )[0]
    heads, layers = (16, 18) if two_b else (32, 24)
    per_page = heads * (4 * 128 * 2 + 4 * 4 + 4 * 2) * 4
    pages = plan.num_blocks + cfg.scheduler_config.max_num_seqs + 1
    reserved = pages * per_page * layers
    if reserved >= available:
        raise ValueError("Insufficient KV budget for separate ordered GDN scratch")
    worker._mach_state_scratch_reserved = reserved
    logger.info(
        "Reserving %.2f MiB for ordered GDN scratch (KV budget %.2f MiB)",
        reserved / 2**20,
        (available - reserved) / 2**20,
    )
    return available - reserved


def before_model_load(worker):
    if envs.VLLM_MACH_FP8_COMPILE_MODE:
        if envs.VLLM_MACH_MXFP8_COMPILE_CHOICES:
            raise ValueError("Block-FP8 and MXFP8 compiler recipes are exclusive")
        from vllm.mach.fp8 import compile_choices as fp8_choices

        fp8_choices.install(envs.VLLM_MACH_FP8_COMPILE_MODE)
    if envs.VLLM_MACH_MXFP8_COMPILE_CHOICES:
        cfg = worker.vllm_config
        if (
            not cfg.use_v2_model_runner
            or cfg.parallel_config.tensor_parallel_size != 1
            or cfg.model_config.hf_text_config.hidden_size != 2560
            or cfg.speculative_config is not None
            or cfg.lora_config is not None
        ):
            raise ValueError("Mach compile choices require 4B V2 TP1 without MTP/LoRA")
        from . import compile_choices

        compile_choices.install()


def before_profile(worker):
    from .auto_defaults import activate_head

    activate_head(worker)
    if envs.VLLM_MACH_MXFP8_FULL_HEAD:
        from . import full_head

        full_head.prepare(worker)
    if envs.VLLM_MACH_MXFP8_GREEDY_HEAD:
        from . import greedy_head

        greedy_head.prepare(worker)
    if envs.VLLM_MACH_MXFP8_KV_SCALE_PROFILE:
        from . import kv_scales

        if not hasattr(worker, kv_scales._STATE):
            kv_scales.install_scales(worker, envs.VLLM_MACH_MXFP8_KV_SCALE_PROFILE)
    if envs.VLLM_MACH_MXFP8_2B_GDN_MIXED and not envs.VLLM_MACH_MXFP8_2B_GDN_ORDERED:
        raise ValueError("2B mixed GDN requires 2B ordered GDN")


def after_cache(worker):
    if envs.VLLM_MACH_MXFP8_2B_GDN_ORDERED:
        from . import two_b_ordered_gdn as gdn

        # Install after memory profiling: profiling uses the unchanged stock
        # path; only the final state pools acquire W4 scratch ownership.
        gdn.install_runtime(worker)
        gdn.prepare_layers(worker)
        gdn.initialize_cache(worker)
        gdn.prepare_pools(worker)
    if envs.VLLM_MACH_MXFP8_BF16_SSM_AUDIT:
        from . import stock_gdn

        stock_gdn.install_runtime(worker)
        stock_gdn.prepare_layers(worker)
        stock_gdn.initialize_cache(worker)
        stock_gdn.prepare_pools(worker)


def after_compile(worker):
    from .auto_defaults import activate_graph

    activate_graph(worker)
    if envs.VLLM_MACH_MXFP8_FULL_HEAD:
        from . import full_head

        full_head.install(worker)
    if envs.VLLM_MACH_MXFP8_GREEDY_HEAD:
        from . import greedy_head

        greedy_head.install(worker)
    if envs.VLLM_MACH_MXFP8_GRAPH_POLICY:
        from . import graph_policy

        graph_policy.install(worker)
    if envs.VLLM_MACH_MXFP8_HEAD_AUDIT:
        from . import head_audit

        head_audit.install(worker)
    receipt = inspect(worker)
    if receipt:
        logger.info("Mach upstream adapter startup receipt: %s", receipt)


def inspect(worker):
    """Host counters describe dispatch/capture construction, not graph replays."""
    result = {}
    if getattr(worker, "_mach_auto_defaults", None):
        result["automatic_defaults"] = dict(worker._mach_auto_defaults)
    model = worker.model_runner.model
    if envs.VLLM_MACH_MXFP8_2B_NATIVE:
        result["two_b_native"] = dict(
            prepared=getattr(model, "_mach_2b_native_prepared", False),
            layers=getattr(model, "_mach_2b_native_count", 0),
        )
    if envs.VLLM_MACH_MXFP8_2B_PDL:
        from . import two_b_pdl

        result["two_b_pdl"] = two_b_pdl.receipt()
    if envs.VLLM_MACH_MXFP8_SWIGLU_QUANT:
        from . import swiglu_quant

        result["swiglu_quant"] = swiglu_quant.receipt()
    if envs.VLLM_MACH_MXFP8_2B_BA or envs.VLLM_MXFP6_SM120_BA_GEMV:
        from . import ba

        result["ba"] = ba.inspect_model(model)
    if envs.VLLM_MXFP6_SM120_MXFP8_DUAL:
        result["dual"] = dict(
            prepared=getattr(model, "_mxfp6_sm120_dual_prepared", False),
            policy=envs.VLLM_MACH_MXFP8_DUAL_POLICY,
        )
    if envs.VLLM_MXFP6_SM120_PROJECTION_PARALLEL:
        result["parallel_projection"] = dict(
            prepared_layers=sum(
                bool(getattr(layer, "_mach_projection_parallel", False))
                for layer in model.modules()
            )
        )
    if envs.VLLM_MACH_MXFP8_GREEDY_HEAD:
        from . import greedy_head

        result["greedy_head"] = greedy_head.inspect(worker)
    from vllm.mach.fp8.runtime import inspect as inspect_block_fp8

    block_fp8 = inspect_block_fp8(worker)
    if block_fp8:
        result["block_fp8"] = block_fp8
    if envs.VLLM_MACH_FP8_COMPILE_MODE:
        from vllm.mach.fp8 import compile_choices as fp8_choices

        result["block_fp8_compile"] = fp8_choices.verify_coverage(
            require_complete=False
        )
    if envs.VLLM_MACH_MXFP8_4B_NATIVE_POLICY:
        model = worker.model_runner.model
        result["four_b_native"] = dict(
            policy=getattr(model, "_mach_4b_native_policy", None),
            prepared_layers=getattr(model, "_mach_4b_native_layers", 0),
        )
    if envs.VLLM_MACH_MXFP8_FULL_HEAD:
        from . import full_head

        result["full_head"] = full_head.inspect(worker)
    if hasattr(worker, "_mach_state_scratch_reserved"):
        result["state_scratch_reserved_bytes"] = worker._mach_state_scratch_reserved
    if envs.VLLM_MACH_MXFP8_4B_GDN_FULL:
        from . import ordered_gdn

        result["four_b_ordered"] = ordered_gdn.stats()
    if envs.VLLM_MACH_MXFP8_COMPILE_CHOICES:
        from . import compile_choices

        # shpgy intentionally has different fused ops and capture sizes. Never
        # invent coverage for upstream sites that are absent in this graph.
        result["compile_choices"] = compile_choices.verify_coverage(
            require_complete=False
        )
    if envs.VLLM_MACH_MXFP8_2B_GDN_ORDERED:
        from . import two_b_ordered_gdn

        result["ordered_gdn"] = two_b_ordered_gdn.inspect_worker(worker)
    if envs.VLLM_MACH_MXFP8_BF16_SSM_AUDIT:
        from . import stock_gdn

        result["bf16_ssm"] = stock_gdn.inspect_worker(worker)
    if envs.VLLM_MACH_MXFP8_KV_SCALE_PROFILE:
        from . import kv_scales

        result["kv_scales"] = kv_scales.inspect_scales(worker)
    if envs.VLLM_MACH_MXFP8_GRAPH_POLICY:
        result["exact2048_graph_policy"] = bool(
            getattr(worker.model_runner.cudagraph_manager, "_mach_exact2048", False)
        )
    if envs.VLLM_MACH_MXFP8_HEAD_AUDIT:
        from . import head_audit

        result["head_audit"] = head_audit.inspect(worker)
    return result
