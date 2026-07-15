// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cub/device/device_topk.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/iterator>
#include <cuda/stream>

#include <cuda_bf16.h>

#include <cstdint>
#include <vector>

namespace {

std::vector<torch::Tensor> cub_device_topk(torch::Tensor input, int64_t k) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(input.dim() == 2, "input must be two-dimensional");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(k > 0 && k <= input.size(1), "k is outside the input width");

  const auto rows = input.size(0);
  const auto width = input.size(1);
  auto output_values = torch::empty({rows, k}, input.options());
  auto output_indices =
      torch::empty({rows, k}, input.options().dtype(torch::kInt32));

  auto requirements =
      cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                               cuda::execution::output_ordering::unsorted);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(input.get_device());
  auto env = cuda::std::execution::env{cuda::stream_ref{stream}, requirements};
  auto counting_indices = cuda::make_counting_iterator<int32_t>(0);

  auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
  auto* value_ptr = reinterpret_cast<__nv_bfloat16*>(output_values.data_ptr());
  auto* index_ptr = output_indices.data_ptr<int32_t>();

  size_t workspace_bytes = 0;
  C10_CUDA_CHECK(cub::DeviceTopK::MaxPairs(nullptr, workspace_bytes, input_ptr,
                                           value_ptr, counting_indices,
                                           index_ptr, width, k, env));
  auto workspace = torch::empty({static_cast<int64_t>(workspace_bytes)},
                                input.options().dtype(torch::kUInt8));

  for (int64_t row = 0; row < rows; ++row) {
    size_t row_workspace_bytes = workspace_bytes;
    C10_CUDA_CHECK(cub::DeviceTopK::MaxPairs(
        workspace.data_ptr(), row_workspace_bytes, input_ptr + row * width,
        value_ptr + row * k, counting_indices, index_ptr + row * k, width, k,
        env));
  }
  return {output_values, output_indices};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("topk", &cub_device_topk, "Batched CUB DeviceTopK by row");
}
