# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backward-compatible import for the generic hybrid lm-head warmup."""

from vllm.model_executor.warmup.hybrid_lm_head_warmup import hybrid_lm_head_warmup

hybrid_mxfp8_lm_head_warmup = hybrid_lm_head_warmup

__all__ = ["hybrid_mxfp8_lm_head_warmup"]
