#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_vllm_root="$(cd -- "${script_dir}/../../.." && pwd)"
vllm_root="${QWEN35_VLLM_ROOT:-${default_vllm_root}}"
python_bin="${vllm_root}/.venv/bin/python"
target="${QWEN35_FLASHINFER_ROOT:-${vllm_root}/.deps/flashinfer_0.6.15.post1_sm120}"
patch_file="${script_dir}/flashinfer_0615_sm120.patch"

# Keep the validated FlashInfer build isolated from vLLM's global dependency.
is_valid_install() {
    local root="$1"
    [[ -f "${root}/flashinfer/__init__.py" ]] &&
        grep -q "^Version: 0\\.6\\.15\\.post1$" \
            "${root}/flashinfer_python-0.6.15.post1.dist-info/METADATA" &&
        grep -q "supported_major_versions=\\[9, 10, 12\\]" \
            "${root}/flashinfer/jit/comm.py" &&
        grep -q "gpuDirectRDMACapable = 0" \
            "${root}/flashinfer/comm/mnnvl.py" &&
        grep -q "params.launch_with_pdl = launch_with_pdl;" \
            "${root}/flashinfer/data/csrc/trtllm_allreduce_fusion.cu" &&
        grep -q "bool launch_with_pdl = false;" \
            "${root}/flashinfer/data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh" &&
        [[ "$(grep -c "if (params.launch_with_pdl)" \
            "${root}/flashinfer/data/include/flashinfer/comm/trtllm_allreduce_fusion.cuh")" \
            -eq 5 ]]
}

if is_valid_install "${target}"; then
    echo "Validated FlashInfer package already exists: ${target}"
    exit 0
fi

if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite an incomplete FlashInfer tree: ${target}" >&2
    exit 1
fi
if [[ ! -x "${python_bin}" ]]; then
    echo "vLLM Python is not executable: ${python_bin}" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to prepare the FlashInfer package" >&2
    exit 1
fi

mkdir -p "$(dirname -- "${target}")"
temporary="$(mktemp -d "$(dirname -- "${target}")/.flashinfer-0615.XXXXXX")"
cleanup() {
    rm -rf -- "${temporary}"
}
trap cleanup EXIT

uv pip install \
    --python "${python_bin}" \
    --target "${temporary}" \
    --no-deps \
    --only-binary=:all: \
    "flashinfer-python==0.6.15.post1"
patch --batch --forward --strip=1 --directory="${temporary}" < "${patch_file}"
if ! is_valid_install "${temporary}"; then
    echo "Prepared FlashInfer tree failed validation: ${temporary}" >&2
    exit 1
fi
mv -- "${temporary}" "${target}"
trap - EXIT

echo "Prepared validated FlashInfer package: ${target}"
