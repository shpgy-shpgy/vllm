#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export QWEN35_MODEL_PATH="${QWEN35_DENSE_MODEL_PATH:-}"
if [[ -z "${QWEN35_MODEL_PATH}" ]]; then
    export QWEN35_MODEL_PATH="/data/models/Qwen3.5-27B-FP8"
fi
export QWEN35_SERVED_NAME="${QWEN35_DENSE_SERVED_NAME:-Qwen3.5-27B-FP8}"
export QWEN35_SAFE_FLASHINFER_AR="${QWEN35_SAFE_FLASHINFER_AR:-auto}"

exec "${script_dir}/serve_tp2.sh" "$@"
