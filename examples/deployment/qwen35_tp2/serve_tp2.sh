#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_vllm_root="$(cd -- "${script_dir}/../../.." && pwd)"
vllm_root="${QWEN35_VLLM_ROOT:-${default_vllm_root}}"
python_bin="${vllm_root}/.venv/bin/python"
model_path="${QWEN35_MODEL_PATH:?QWEN35_MODEL_PATH is required}"
served_name="${QWEN35_SERVED_NAME:?QWEN35_SERVED_NAME is required}"
server_host="${QWEN35_SERVER_HOST:-0.0.0.0}"
server_port="${QWEN35_SERVER_PORT:-8203}"
safe_flashinfer_ar="${QWEN35_SAFE_FLASHINFER_AR:-auto}"
original_pythonpath="${PYTHONPATH:-}"

if [[ ! -x "${python_bin}" ]]; then
    echo "vLLM Python is not executable: ${python_bin}" >&2
    exit 1
fi

# Safe mode inserts the validated FlashInfer package. Auto leaves backend
# selection to vLLM's strict SM120/TP2 runtime gates.
case "${safe_flashinfer_ar}" in
    0)
        export QWEN35_USE_VALIDATED_FLASHINFER=0
        export VLLM_ALLREDUCE_USE_FLASHINFER=0
        export PYTHONPATH="${vllm_root}${original_pythonpath:+:${original_pythonpath}}"
        ;;
    1 | auto)
        flashinfer_root="${QWEN35_FLASHINFER_ROOT:-}"
        if [[ -z "${flashinfer_root}" ]]; then
            flashinfer_root="${vllm_root}/.deps/flashinfer_0.6.15.post1_sm120"
        fi
        if [[ ! -f "${flashinfer_root}/flashinfer/__init__.py" ]]; then
            echo "Validated FlashInfer package is missing: ${flashinfer_root}" >&2
            echo "Run ${script_dir}/prepare_flashinfer.sh first." >&2
            exit 1
        fi
        if ! grep -q "^Version: 0\\.6\\.15\\.post1$" \
            "${flashinfer_root}/flashinfer_python-0.6.15.post1.dist-info/METADATA" ||
            ! grep -q "supported_major_versions=\\[9, 10, 12\\]" \
            "${flashinfer_root}/flashinfer/jit/comm.py" ||
            ! grep -q "gpuDirectRDMACapable = 0" \
                "${flashinfer_root}/flashinfer/comm/mnnvl.py" ||
            ! grep -q "params.launch_with_pdl = launch_with_pdl;" \
                "${flashinfer_root}/flashinfer/data/csrc/trtllm_allreduce_fusion.cu" ||
            ! grep -q "bool launch_with_pdl = false;" \
                "${flashinfer_root}/flashinfer/data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh" ||
            [[ "$(grep -c "if (params.launch_with_pdl)" \
                "${flashinfer_root}/flashinfer/data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh")" \
                -ne 5 ]]; then
            echo "FlashInfer package is missing the validated SM120 patch" >&2
            exit 1
        fi
        if [[ -z "${FLASHINFER_WORKSPACE_BASE:-}" ]]; then
            # Keep generated JIT files under vLLM's existing ignored .deps tree.
            export FLASHINFER_WORKSPACE_BASE="${vllm_root}/.deps/qwen35_flashinfer_workspace"
        fi
        if [[ "${safe_flashinfer_ar}" == "1" ]]; then
            export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
            export VLLM_ALLREDUCE_USE_FLASHINFER=1
        fi
        export PYTHONPATH="${flashinfer_root}:${vllm_root}${original_pythonpath:+:${original_pythonpath}}"
        ;;
    *)
        echo "QWEN35_SAFE_FLASHINFER_AR must be auto, 0, or 1" >&2
        exit 1
        ;;
esac

export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export CUDA_VISIBLE_DEVICES="${QWEN35_CUDA_DEVICES:-4,5}"

exec "${python_bin}" -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --host "${server_host}" \
    --port "${server_port}" \
    --served-model-name "${served_name}" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 64 \
    --max-model-len 32768 \
    --trust-remote-code \
    --cudagraph-capture-sizes \
        1 2 4 8 10 12 14 16 20 24 28 32 36 40 48 56 64 72 80 96 \
        112 128 160 192 224 256 320 384 448 512 640 768 784 896 1024 \
        1280 1536 1792 2048 \
    --enable-logging-iteration-details \
    --attention-backend FLASHINFER \
    "$@"
