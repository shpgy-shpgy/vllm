#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Validated SM120 TP2 selects FlashInfer all-reduce with PDL disabled in
# both the launch and compiled kernel.
export QWEN35_MODEL_PATH="${QWEN35_MOE_MODEL_PATH:-}"
if [[ -z "${QWEN35_MODEL_PATH}" ]]; then
    export QWEN35_MODEL_PATH="/data/models/Qwen3.5-35B-A3B-FP8"
fi
export QWEN35_SERVED_NAME="${QWEN35_MOE_SERVED_NAME:-Qwen3.5-35B-A3B-FP8}"
export QWEN35_SAFE_FLASHINFER_AR="${QWEN35_SAFE_FLASHINFER_AR:-auto}"

exec "${script_dir}/serve_tp2.sh" "$@"
