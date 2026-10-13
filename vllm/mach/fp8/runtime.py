# SPDX-License-Identifier: Apache-2.0
"""Install independent block-FP8 adapters before model construction."""

from vllm import envs


def install(worker):
    n64 = envs.VLLM_MACH_FP8_N64
    ordered = envs.VLLM_MACH_FP8_ORDERED
    silu = envs.VLLM_MACH_FP8_SILU
    if not (n64 or ordered or silu):
        return
    cfg = worker.vllm_config
    hf = cfg.model_config.hf_text_config
    offload = getattr(cfg, "offload_config", None)
    if (
        hf.model_type != "qwen3_5_text"
        or hf.hidden_size != 2560
        or hf.intermediate_size != 9216
        or hf.num_hidden_layers != 32
        or cfg.parallel_config.tensor_parallel_size != 1
        or cfg.parallel_config.pipeline_parallel_size != 1
        or cfg.speculative_config is not None
        or cfg.lora_config is not None
        or cfg.weight_transfer_config is not None
        or getattr(cfg.model_config, "enable_sleep_mode", False)
        or getattr(getattr(offload, "uva", None), "cpu_offload_gb", 0)
        or getattr(getattr(offload, "prefetch", None), "offload_group_size", 0)
    ):
        raise ValueError("Block-FP8 adapters require stationary Qwen3.5-4B TP1/PP1")
    if n64 or ordered:
        from . import linear

        linear.install(n64=n64, ordered=ordered)
    if silu:
        from . import activation

        activation.install()

    def reject(*args, **kwargs):
        raise RuntimeError(
            "Block-FP8 adapters require a worker restart for weight updates"
        )

    for name in (
        "reload_weights",
        "start_weight_update",
        "update_weights",
        "init_weight_transfer_engine",
    ):
        if hasattr(worker, name):
            setattr(worker, name, reject)


def inspect(worker):
    result = {}
    if envs.VLLM_MACH_FP8_N64 or envs.VLLM_MACH_FP8_ORDERED:
        from . import linear

        result["linear"] = linear.inspect(worker)
    if envs.VLLM_MACH_FP8_SILU:
        from . import activation

        result["activation"] = activation.inspect()
    return result
