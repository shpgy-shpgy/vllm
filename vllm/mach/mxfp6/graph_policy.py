# SPDX-License-Identifier: Apache-2.0
"""Opt-in upstream exact-2048 / unpadded large-tail dispatch, for V2 TP1.

Keep the user's capture sizes. Only the runtime large-tail routing changes.
Thresholds (2B:160, 4B:256) come from vllm-mach 05aeea3.
"""

from functools import wraps
from types import MethodType

from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


def apply_descriptor(desc, num_tokens, num_reqs, *, hidden_size):
    threshold = {2048: 160, 2560: 256}[hidden_size]
    if (
        num_tokens > threshold
        and num_tokens != 2048
        and (hidden_size == 2048 or desc.cg_mode == CUDAGraphMode.PIECEWISE)
    ):
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.NONE,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            num_active_loras=desc.num_active_loras,
        )
    if num_tokens == 2048 and (
        desc.cg_mode != CUDAGraphMode.PIECEWISE or desc.num_tokens != 2048
    ):
        raise RuntimeError("Mach graph policy requires exact PIECEWISE2048")
    return desc


def install(worker):
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    cfg = worker.vllm_config
    manager = worker.model_runner.cudagraph_manager
    hidden = cfg.model_config.hf_text_config.hidden_size
    if (
        hidden not in (2048, 2560)
        or not cfg.use_v2_model_runner
        or cfg.parallel_config.tensor_parallel_size != 1
        or cfg.parallel_config.pipeline_parallel_size != 1
        or cfg.speculative_config is not None
        or cfg.lora_config is not None
        or manager is None
        or not manager._graphs_captured
        or manager.cudagraph_mode != CUDAGraphMode.FULL_AND_PIECEWISE
    ):
        raise RuntimeError("Mach graph policy requires captured Qwen3.5 2B/4B V2 TP1")
    entries = [
        entry
        for wrapper in list(CUDAGraphWrapper._all_instances)
        if wrapper.vllm_config is cfg
        and wrapper.runtime_mode == CUDAGraphMode.PIECEWISE
        for desc, entry in wrapper.concrete_cudagraph_entries.items()
        if desc.num_tokens == 2048
    ]
    if not entries or any(entry.cudagraph is None for entry in entries):
        raise RuntimeError("Mach graph policy cannot find captured PIECEWISE2048")
    if getattr(manager, "_mach_exact2048", False):
        return
    original = manager.dispatch

    @wraps(original)
    def dispatch(self, num_reqs, num_tokens, *args, **kwargs):
        desc = original(num_reqs, num_tokens, *args, **kwargs)
        return apply_descriptor(desc, num_tokens, num_reqs, hidden_size=hidden)

    manager.dispatch = MethodType(dispatch, manager)
    manager._mach_exact2048 = True
