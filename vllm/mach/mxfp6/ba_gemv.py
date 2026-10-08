# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small-M BF16 GEMV used by the Qwen3.5 BA projection.

This is intentionally kept in a lazy-import module.  The CUDA/CuTe and b12x
imports are optional for vLLM as a whole and are only required when the
explicit ``VLLM_MXFP6_SM120_BA_GEMV=1`` adapter is enabled.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from b12x._lib.compiler import KernelCompileSpec
from b12x._lib.compiler import compile as b12x_compile
from b12x._lib.intrinsics import (
    block_reduce,
    get_ptr_as_int64,
    ld_global_v4_u32,
    u32_as_f32,
    warp_reduce,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream
from cutlass import Int64
from cutlass.cutlass_dsl import Int32, Uint32

_THREADS = 128
SMALL_M_MAX = 8

_KERNEL_CACHE: dict[tuple[int, int, int], object] = {}
_PRECOMPILED: set[tuple[int, int]] = set()


def _fadd(a, b):
    return a + b


class SmallNGemvKernel:
    """One CTA per output column; ``y[0:m, n] = x[0:m, :] @ w[n, :]``."""

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        y: cute.Tensor,
        stream: cuda.CUstream,
    ):
        n = w.shape[0]
        self.kernel(x, w, y).launch(
            grid=(n, 1, 1),
            block=[_THREADS, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mW: cute.Tensor, mY: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        m = mX.shape[0]
        n = mW.shape[0]
        k = Int32(mX.shape[1])
        nvec = k // Int32(8)
        w_base = Int64(bidx) * Int64(k)

        smem = cutlass.utils.SmemAllocator()
        red_buf = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((1, _THREADS // 32)),
            byte_alignment=16,
        )

        acc = cute.make_rmem_tensor((m,), cutlass.Float32)
        for r in cutlass.range_constexpr(m):
            acc[r] = cutlass.Float32(0.0)

        # Each thread reads a strided chunk of the weight once and reuses it
        # for all M rows.  Four uint32 loads cover eight BF16 values.
        i = Int32(tidx)
        while i < nvec:
            col = Int64(i * Int32(8))
            w0, w1, w2, w3 = ld_global_v4_u32(get_ptr_as_int64(mW, w_base + col))
            for r in cutlass.range_constexpr(m):
                x0, x1, x2, x3 = ld_global_v4_u32(
                    get_ptr_as_int64(mX, Int64(r) * Int64(k) + col)
                )
                s = acc[r]
                for wv, xv in ((w0, x0), (w1, x1), (w2, x2), (w3, x3)):
                    s = s + u32_as_f32(wv << Uint32(16)) * u32_as_f32(xv << Uint32(16))
                    s = s + u32_as_f32(wv & Uint32(0xFFFF0000)) * u32_as_f32(
                        xv & Uint32(0xFFFF0000)
                    )
                acc[r] = s
            i += Int32(_THREADS)

        # Reuse the reduction scratch between rows.  The barrier prevents the
        # next row's lane-0 write from racing with the previous read.
        for r in cutlass.range_constexpr(m):
            v = warp_reduce(acc[r], _fadd)
            total = block_reduce(v, _fadd, red_buf, cutlass.Float32(0.0))
            if tidx == Int32(0):
                # Use a flat output view; a 2D (m, n) tensor degenerates to a
                # different compact-layout deduction when n == 1.
                mY[Int32(r * n) + bidx] = cutlass.BFloat16(total)
            cute.arch.barrier()


def compile_bf16_gemv_small_n(m: int, n: int, k: int):
    """Compile and cache one ``(M,N,K)`` small-N GEMV launcher."""
    assert 1 <= m <= SMALL_M_MAX, f"small-N GEMV requires m<={SMALL_M_MAX}, got {m}"
    assert k % 8 == 0, f"K must be a multiple of 8, got {k}"
    assert n >= 1
    cache_key = (m, n, k)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    x_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=16
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (n, k), stride_order=(1, 0), assumed_align=16
    )
    y_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m * n,), assumed_align=16
    )
    kernel = SmallNGemvKernel()
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    raw = b12x_compile(
        kernel,
        x_fake,
        w_fake,
        y_fake,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key("gemm.bf16_gemv_small_n", 2, cache_key),
    )

    def launch(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor):
        raw(x, w, y.view(-1), current_cuda_stream())

    _KERNEL_CACHE[cache_key] = launch
    return launch


def get_cached_bf16_gemv_small_n(m: int, n: int, k: int):
    """Cache-only lookup; it never triggers JIT compilation."""
    return _KERNEL_CACHE.get((m, n, k))


def precompile_bf16_gemv_small_n(weight: torch.Tensor, log=None) -> None:
    """Compile and first-launch M1..8 for one private weight storage."""
    if log is None:
        import logging

        log = logging.getLogger("vllm.mach.mxfp6.ba_gemv")
    n, k = (int(value) for value in weight.shape)
    if (
        k % 8 != 0
        or weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or weight.data_ptr() % 16 != 0
    ):
        log.info(
            "BA BF16 GEMV precompile skipped for unsupported weight n=%d k=%d",
            n,
            k,
        )
        return
    if (n, k) in _PRECOMPILED:
        return

    log.info("BA BF16 GEMV precompile n=%d k=%d, m=1..%d", n, k, SMALL_M_MAX)
    for m in range(1, SMALL_M_MAX + 1):
        launch = compile_bf16_gemv_small_n(m, n, k)
        x = torch.zeros(m, k, dtype=torch.bfloat16, device=weight.device)
        y = torch.empty(m, n, dtype=torch.bfloat16, device=weight.device)

        # Loader-backed parameter storage has triggered raw-launch faults on
        # some deployments.  A dummy allocation separates a bad kernel from a
        # bad parameter storage; the adapter always passes the private clone.
        w_dummy = torch.zeros_like(weight)
        launch(x, w_dummy, y)
        torch.accelerator.synchronize()
        del w_dummy
        launch(x, weight, y)
        torch.accelerator.synchronize()
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    _PRECOMPILED.add((n, k))


def _use_kernel(x: torch.Tensor, weight: torch.Tensor) -> bool:
    m, k = x.shape
    return (
        1 <= m <= SMALL_M_MAX
        and k % 8 == 0
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
    )


@torch.library.custom_op("b12x::bf16_gemv_small_n", mutates_args=())
def bf16_gemv_small_n(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``y = x @ weight.T`` for BF16 x ``(M,K)`` and weight ``(N,K)``."""
    if not _use_kernel(x, weight):
        return torch.nn.functional.linear(x, weight)
    if not x.is_contiguous() or x.data_ptr() % 16 != 0:
        x = x.contiguous()
        if x.data_ptr() % 16 != 0:
            return torch.nn.functional.linear(x, weight)

    m, k = (int(value) for value in x.shape)
    n = int(weight.shape[0])
    launch = get_cached_bf16_gemv_small_n(m, n, k)
    if launch is None:
        # Never JIT while capturing.  The vLLM adapter precompiles M1..8
        # before profiling and graph capture; this is only a safe fallback for
        # an unexpected shape in eager execution.
        if torch.cuda.is_current_stream_capturing():
            return torch.nn.functional.linear(x, weight)
        launch = compile_bf16_gemv_small_n(m, n, k)
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    launch(x, weight, y)
    return y


@bf16_gemv_small_n.register_fake
def _bf16_gemv_small_n_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.bfloat16)


__all__ = [
    "SMALL_M_MAX",
    "bf16_gemv_small_n",
    "compile_bf16_gemv_small_n",
    "get_cached_bf16_gemv_small_n",
    "precompile_bf16_gemv_small_n",
]
