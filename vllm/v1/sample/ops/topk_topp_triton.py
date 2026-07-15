# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Combined Top-K and Top-P Triton kernels.

Based on the paper "Qrita: High-performance Top-k and Top-p Algorithm for GPUs
using Pivot-based Truncation and Selection" By Park et al.
(https://arxiv.org/abs/2602.01518)

"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.platform_utils import num_compute_units

_TRITON_TABLE_CACHE: dict[tuple[torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
_TRITON_BUFFER_CACHE: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}
_TRITON_SMALL_TOPK_SAMPLE_CACHE: dict[
    tuple[torch.device, torch.dtype, int, int],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}

# fmt: off
_NORMAL_CDF_TO_SIGMA_TABLE = [
  3.656,  3.650,  3.650,  3.650,  3.626,  3.626,  3.626,  3.514,  3.514,  3.503, 
  3.503,  3.434,  3.434,  3.428,  3.428,  3.387,  3.380,  3.380,  3.376,  3.373, 
  3.373,  3.356,  3.354,  3.354,  3.291,  3.249,  3.234,  3.214,  3.198,  3.198, 
  3.185,  3.177,  3.177,  3.165,  3.164,  3.161,  3.138,  3.120,  3.115,  3.113, 
  3.093,  3.066,  3.054,  3.043,  3.037,  3.023,  2.993,  2.991,  2.976,  2.970, 
  2.952,  2.946,  2.932,  2.908,  2.902,  2.895,  2.886,  2.874,  2.861,  2.844, 
  2.836,  2.810,  2.801,  2.790,  2.784,  2.779,  2.767,  2.757,  2.745,  2.733, 
  2.723,  2.716,  2.693,  2.678,  2.671,  2.656,  2.649,  2.629,  2.611,  2.595, 
  2.592,  2.585,  2.574,  2.550,  2.543,  2.534,  2.521,  2.518,  2.497,  2.485, 
  2.468,  2.450,  2.441,  2.430,  2.412,  2.402,  2.389,  2.383,  2.377,  2.364, 
  2.349,  2.338,  2.332,  2.319,  2.310,  2.301,  2.282,  2.274,  2.266,  2.250, 
  2.242,  2.236,  2.226,  2.215,  2.207,  2.196,  2.179,  2.171,  2.162,  2.147, 
  2.135,  2.121,  2.109,  2.095,  2.085,  2.073,  2.063,  2.045,  2.030,  2.016, 
  2.003,  1.992,  1.983,  1.972,  1.960,  1.949,  1.940,  1.928,  1.912,  1.897, 
  1.881,  1.869,  1.854,  1.838,  1.824,  1.807,  1.792,  1.779,  1.764,  1.751, 
  1.739,  1.726,  1.711,  1.697,  1.685,  1.668,  1.652,  1.636,  1.622,  1.603, 
  1.585,  1.568,  1.551,  1.534,  1.513,  1.499,  1.480,  1.464,  1.441,  1.422, 
  1.394,  1.373,  1.347,  1.320,  1.296,  1.270,  1.246,  1.219,  1.190,  1.163, 
  1.135,  1.104,  1.073,  1.041,  1.006,  0.969,  0.931,  0.894,  0.851,  0.806, 
  0.757,  0.702,  0.643,  0.574,  0.498,  0.405,  0.288,  0.134, -0.110, -3.813 
]

_PERCENTILE_TO_STD_TABLE = [
  2.576,  2.319,  2.178,  2.064,  1.968,  1.892,  1.819,  1.757,  1.708,  1.659, 
  1.616,  1.568,  1.526,  1.492,  1.456,  1.420,  1.382,  1.342,  1.309,  1.280, 
  1.249,  1.221,  1.193,  1.169,  1.145,  1.121,  1.095,  1.073,  1.050,  1.030, 
  1.008,  0.987,  0.966,  0.945,  0.926,  0.910,  0.891,  0.871,  0.854,  0.837, 
  0.819,  0.803,  0.784,  0.767,  0.753,  0.734,  0.719,  0.702,  0.690,  0.675, 
  0.658,  0.640,  0.625,  0.609,  0.595,  0.578,  0.564,  0.550,  0.537,  0.521, 
  0.509,  0.495,  0.481,  0.466,  0.453,  0.439,  0.424,  0.410,  0.397,  0.383, 
  0.370,  0.356,  0.343,  0.330,  0.316,  0.302,  0.289,  0.274,  0.261,  0.247, 
  0.235,  0.223,  0.209,  0.196,  0.184,  0.172,  0.159,  0.149,  0.137,  0.124, 
  0.112,  0.100,  0.086,  0.074,  0.062,  0.050,  0.035,  0.023,  0.009, -0.003, 
 -0.015, -0.027, -0.039, -0.052, -0.063, -0.074, -0.085, -0.097, -0.109, -0.122, 
 -0.134, -0.147, -0.158, -0.171, -0.184, -0.196, -0.210, -0.223, -0.235, -0.248, 
 -0.261, -0.275, -0.289, -0.302, -0.317, -0.328, -0.341, -0.353, -0.368, -0.382, 
 -0.396, -0.410, -0.426, -0.439, -0.452, -0.465, -0.480, -0.493, -0.507, -0.521, 
 -0.537, -0.551, -0.568, -0.582, -0.597, -0.614, -0.628, -0.643, -0.658, -0.673, 
 -0.691, -0.706, -0.721, -0.738, -0.754, -0.769, -0.789, -0.808, -0.824, -0.838, 
 -0.857, -0.877, -0.893, -0.912, -0.929, -0.947, -0.965, -0.983, -1.003, -1.027, 
 -1.050, -1.070, -1.092, -1.117, -1.139, -1.162, -1.189, -1.216, -1.241, -1.272, 
 -1.300, -1.330, -1.367, -1.404, -1.441, -1.485, -1.523, -1.564, -1.607, -1.658, 
 -1.710, -1.778, -1.832, -1.901, -1.978, -2.068, -2.174, -2.325, -2.577, -3.813 
]
# fmt: on


@triton.jit
def _update_min_larger_stats(data, above_mask, min_larger, num_min_larger, sentinel):
    """Update running (min, count) of values above a pivot across tiles.

    Tracks the smallest value strictly above a pivot and how many times
    it occurs.  Called once per tile per pivot; the running state is
    carried across tiles via `min_larger` / `num_min_larger`.

    Merge rule:
      - tile min < running min  → replace both
      - tile min == running min → accumulate count
      - tile min > running min  → keep running values
    """
    tile_min = tl.min(tl.where(above_mask, data, sentinel))
    tile_eq = above_mask & (tl.abs(data - tile_min) < 1e-9)
    tile_cnt = tl.sum(tile_eq)
    is_new = tile_min < min_larger
    is_same = tl.abs(tile_min - min_larger) < 1e-9
    num_min_larger = tl.where(is_new, tile_cnt, num_min_larger + tile_cnt * is_same)
    min_larger = tl.minimum(min_larger, tile_min)
    return min_larger, num_min_larger


@triton.jit
def _topk_topp_kernel(
    LOGITS,
    LOGITS_STRIDE_0,
    BUFFER,
    PERCENTILE_TO_STD_TABLE,
    NORMAL_CDF_TO_SIGMA_TABLE,
    K,
    P,
    PIVOTS,
    COUNTS,
    BATCH_SIZE,
    VOCAB_SIZE: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_TRUNC: tl.constexpr,
    TOPK_ENABLED: tl.constexpr,
    TOPP_ENABLED: tl.constexpr,
    APPLY_FINAL_MASK: tl.constexpr,
    OUTPUT_PIVOT: tl.constexpr,
    RESET_COUNTS: tl.constexpr,
):
    NUM_TILES: tl.constexpr = (VOCAB_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for row_id in tl.range(pid, BATCH_SIZE, num_programs):
        LOGITS_ROW = LOGITS + row_id.to(tl.int64) * LOGITS_STRIDE_0
        BUFFER_ROW = BUFFER + pid * VOCAB_SIZE

        final_pivot = -float("inf")
        duplicate_logit = float("inf")
        num_duplicate_logit = tl.zeros((), dtype=tl.uint32)
        num_keep = tl.zeros((), dtype=tl.uint32)
        num_kept = tl.zeros((), dtype=tl.uint32)

        max_logit = -float("inf")
        min_logit = float("inf")

        if TOPK_ENABLED:
            k = tl.load(K + row_id)
            if k < VOCAB_SIZE:
                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(
                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                )
                # Exclude -inf values (e.g. from grammar bitmasks) from
                # statistics to avoid NaN in pivot computation.
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(finite_mask)
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(
                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                )
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(
                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                )

                # Calculate outlier pivot t for Gaussian sigma-truncation
                percentile = tl.cast(k / VOCAB_SIZE * 200, tl.uint32)
                percentile = tl.minimum(percentile, 199)
                sigma = tl.load(PERCENTILE_TO_STD_TABLE + percentile)
                sigma = sigma + tl.abs(sigma) * -0.15
                outlier_pivot = avg_logit + std_logit * sigma
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # First pass: compute max and min logits and gather outliers
                num_finite_total = tl.zeros((), dtype=tl.uint32)
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )

                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # Exclude -inf from min to keep binary search bounds
                    # finite (avoids NaN pivots).
                    finite_blk_mask = logits_blk > -float("inf")
                    finite_blk = tl.where(finite_blk_mask, logits_blk, float("inf"))
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))
                    num_finite_total += tl.sum(finite_blk_mask & mask_n)

                    outlier_mask = (logits_blk > outlier_pivot) & mask_n
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                    )
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, logits_blk, mask=outlier_mask)

                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                # Second passes: Ternary search for pivot
                num_iters = 0
                k_pivot = float("inf")
                k_pivots_num = tl.zeros((), dtype=tl.uint32)
                min_larger = float("inf")
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                if num_outliers > k:
                    max_range = max_logit
                    min_range = outlier_pivot
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )
                    found_pivot = 0
                    while found_pivot == 0:
                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute k_pivots_num,
                        # min_larger, and num_min_larger together to avoid
                        # a second data scan. See _update_min_larger_stats
                        # for the tile-level merge logic.
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            logits_blk2 = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=-float("inf")
                            )

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(above_0)
                            k_pivots_num_1 += tl.sum(above_1)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # Check if any of the pivots satisfy termination condition
                        if (
                            k_pivots_num_0 >= k
                            and k_pivots_num_0 - num_min_larger_0 < k
                        ):
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if (
                            k_pivots_num_1 >= k
                            and k_pivots_num_1 - num_min_larger_1 < k
                        ):
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # Update range
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                else:
                    # If top-k outlier gathering failed, search whole logit space
                    max_range = max_logit
                    min_range = min_logit
                    found_pivot = 0
                    while found_pivot == 0:
                        k_pivot_0 = (max_range - min_range) * 1.0 / 3.0 + min_range
                        k_pivots_num_0 = tl.zeros((), dtype=tl.uint32)
                        min_larger_0 = float("inf")
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        k_pivot_1 = (max_range - min_range) * 2.0 / 3.0 + min_range
                        k_pivots_num_1 = tl.zeros((), dtype=tl.uint32)
                        min_larger_1 = float("inf")
                        num_min_larger_1 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass over full vocab (same approach
                        # as the buffer path above).
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            logits_blk2 = tl.load(
                                LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                            )

                            above_0 = logits_blk2 > k_pivot_0
                            above_1 = logits_blk2 > k_pivot_1
                            k_pivots_num_0 += tl.sum(above_0)
                            k_pivots_num_1 += tl.sum(above_1)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                logits_blk2,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                float("inf"),
                            )
                            min_larger_1, num_min_larger_1 = _update_min_larger_stats(
                                logits_blk2,
                                above_1,
                                min_larger_1,
                                num_min_larger_1,
                                float("inf"),
                            )

                        # Check if any of the pivots satisfy termination condition
                        if (
                            k_pivots_num_0 >= k
                            and k_pivots_num_0 - num_min_larger_0 < k
                        ):
                            k_pivot = k_pivot_0
                            k_pivots_num = k_pivots_num_0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1
                        if (
                            k_pivots_num_1 >= k
                            and k_pivots_num_1 - num_min_larger_1 < k
                        ):
                            k_pivot = k_pivot_1
                            k_pivots_num = k_pivots_num_1
                            min_larger = min_larger_1
                            num_min_larger = num_min_larger_1
                            found_pivot = 1

                        # Update range
                        if k_pivots_num_1 > k:
                            min_range = k_pivot_1
                        elif k_pivots_num_0 > k:
                            min_range = k_pivot_0

                        if k_pivots_num_0 < k:
                            max_range = k_pivot_0
                        elif k_pivots_num_1 < k:
                            max_range = k_pivot_1

                        num_iters += 1
                        if num_iters >= 18 or tl.abs(min_range - max_range) < 1e-9:
                            k_pivot = (max_range + min_range) / 2.0
                            min_larger = min_larger_0
                            num_min_larger = num_min_larger_0
                            found_pivot = 1

                duplicate_logit = min_larger
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - (k_pivots_num - k)
                num_kept = tl.zeros((), dtype=tl.uint32)

                # Top-k only path.  If there are fewer finite values
                # than k (e.g. grammar mask), keep everything.
                final_pivot = k_pivot if num_finite_total > k else -float("inf")

                if TOPP_ENABLED and num_finite_total > k:
                    #### TOP-P SAMPLING AFTER TOP-K ####
                    p = tl.load(P + row_id)
                    if p < 1.0:
                        min_logit = k_pivot
                        sum_exp_logits = 0.0
                        num_outliers_2 = tl.zeros((), dtype=tl.uint32)
                        search_range = tl.cast(num_outliers, tl.int32)
                        search_iters = tl.cast(
                            (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                            tl.int32,
                        )

                        # Third pass: Calculate exp logits and sum, gather outliers
                        if num_outliers > k:
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=-float("inf"),
                                )

                                outlier_mask = (probs_blk > min_logit) & mask_n_2

                                # Duplicate logit handling for Top-k
                                if num_keep < num_duplicate_logit:
                                    duplicate_mask = (
                                        tl.abs(probs_blk - duplicate_logit) < 1e-9
                                    )
                                    duplicate_count = (
                                        tl.cumsum(duplicate_mask) + num_kept
                                    )
                                    duplicate_keep_mask = (
                                        duplicate_count <= num_keep
                                    ) & duplicate_mask
                                    duplicate_remove_mask = (
                                        duplicate_mask & ~duplicate_keep_mask
                                    )
                                    outlier_mask = outlier_mask & (
                                        ~duplicate_remove_mask
                                    )
                                    num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(
                                    outlier_mask, probs_blk, -float("inf")
                                )
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                            # Fourth pass: Calculate BUFFER and get outliers
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n,
                                    mask=mask_n_2,
                                    other=-float("inf"),
                                )

                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)
                        else:
                            # If top-k outlier gathering failed,
                            # retry gathering using top-k pivot
                            for i in range(0, NUM_TILES):
                                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                                mask_n = offs_n < VOCAB_SIZE

                                probs_blk = tl.load(
                                    LOGITS_ROW + offs_n,
                                    mask=mask_n,
                                    other=-float("inf"),
                                )

                                outlier_mask = (probs_blk > min_logit) & mask_n

                                # Duplicate logit handling for Top-k
                                duplicate_mask = (
                                    tl.abs(probs_blk - duplicate_logit) < 1e-9
                                )
                                duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                                duplicate_keep_mask = (
                                    duplicate_count <= num_keep
                                ) & duplicate_mask
                                duplicate_remove_mask = (
                                    duplicate_mask & ~duplicate_keep_mask
                                )
                                outlier_mask = outlier_mask & (~duplicate_remove_mask)
                                num_kept += tl.sum(duplicate_keep_mask)

                                probs_blk = tl.where(
                                    outlier_mask, probs_blk, -float("inf")
                                )
                                probs_blk = probs_blk - max_logit
                                probs_blk = tl.exp(probs_blk)
                                sum_exp_logits += tl.sum(probs_blk)

                                cumulative_pos = tl.cast(
                                    tl.cumsum(outlier_mask) - 1 + num_outliers_2,
                                    tl.int32,
                                )
                                num_outliers_2 += tl.sum(outlier_mask)
                                write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                                tl.store(
                                    BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask
                                )

                            search_range = tl.cast(num_outliers_2, tl.int32)
                            search_iters = tl.cast(
                                (num_outliers_2 + BLOCK_SIZE_TRUNC - 1)
                                // BLOCK_SIZE_TRUNC,
                                tl.int32,
                            )

                            # Fourth pass: Calculate BUFFER and get outliers
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range

                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )
                                probs_blk = probs_blk / sum_exp_logits
                                tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n_2)

                        max_range = tl.exp(max_logit - max_logit) / sum_exp_logits
                        min_range = tl.exp(min_logit - max_logit) / sum_exp_logits

                        p_pivot = 1.0
                        num_iters = 0
                        min_larger_prob = 1.0
                        num_min_larger = tl.zeros((), dtype=tl.uint32)
                        p_pivots_sum = 0.0

                        # Fifth passes: Search for p_pivot
                        found_pivot = 0
                        while found_pivot == 0:
                            p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                            p_pivots_sum_0 = 0.0
                            min_larger_0 = 1.0
                            num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                            # Single fused pass: compute p_pivots_sum,
                            # min_larger, and num_min_larger together.
                            # See _update_min_larger_stats for the
                            # tile-level merge logic.
                            for i in range(0, search_iters):
                                offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                    0, BLOCK_SIZE_TRUNC
                                )
                                mask_n_2 = offs_n < search_range
                                probs_blk = tl.load(
                                    BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                                )

                                above_0 = probs_blk > p_pivot_0
                                p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                                min_larger_0, num_min_larger_0 = (
                                    _update_min_larger_stats(
                                        probs_blk,
                                        above_0,
                                        min_larger_0,
                                        num_min_larger_0,
                                        1.0,
                                    )
                                )

                            # Check if the pivot satisfies termination condition
                            if p_pivots_sum_0 >= p and (
                                p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                            ):
                                p_pivot = p_pivot_0
                                min_larger_prob = min_larger_0
                                num_min_larger = num_min_larger_0
                                p_pivots_sum = p_pivots_sum_0
                                found_pivot = 1

                            # Update range
                            if p_pivots_sum_0 > p:
                                min_range = p_pivot_0
                            elif p_pivots_sum_0 < p:
                                max_range = p_pivot_0

                            num_iters += 1
                            if (max_range - min_range) < 1e-9 or num_iters >= 18:
                                p_pivot = (max_range + min_range) / 2.0
                                min_larger_prob = min_larger_0
                                num_min_larger = num_min_larger_0
                                p_pivots_sum = p_pivots_sum_0
                                found_pivot = 1

                        duplicate_logit = (
                            tl.log(min_larger_prob * sum_exp_logits) + max_logit
                        )
                        num_duplicate_logit = num_min_larger
                        num_keep = num_duplicate_logit - tl.cast(
                            (p_pivots_sum - p) / min_larger_prob, tl.uint32
                        )
                        num_kept = tl.zeros((), dtype=tl.uint32)

                        # Top-k + Top-p path
                        final_pivot = tl.log(p_pivot * sum_exp_logits) + max_logit

        if TOPP_ENABLED and final_pivot == -float("inf"):
            #### STANDALONE TOP-P SAMPLING ####
            p = tl.load(P + row_id)
            if p < 1.0:
                # Zeroth pass: Compute avg and std from a sample block
                offs = tl.arange(0, BLOCK_SIZE)
                mask_n = offs < VOCAB_SIZE
                logits_blk0 = tl.load(
                    LOGITS_ROW + offs, mask=mask_n, other=-float("inf")
                )
                # Exclude -inf values (e.g. from grammar bitmasks) from
                # statistics to avoid NaN in pivot computation.
                finite_mask = (logits_blk0 > -float("inf")) & mask_n
                num_finite = tl.sum(finite_mask)
                finite_logits = tl.where(finite_mask, logits_blk0, 0.0)
                avg_logit = tl.where(
                    num_finite > 0, tl.sum(finite_logits) / num_finite, 0.0
                )
                sq_avg_logit = tl.where(
                    num_finite > 0,
                    tl.sum(finite_logits * finite_logits) / num_finite,
                    0.0,
                )
                std_logit = tl.sqrt(
                    tl.maximum(sq_avg_logit - avg_logit * avg_logit, 0.0)
                )
                max_sample = avg_logit + std_logit * 10.0
                sum_exp_logits = 0.0

                # First pass: compute max and min logits and sum_exp_logits
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE
                    logits_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    max_logit = tl.maximum(max_logit, tl.max(logits_blk))
                    # Exclude -inf from min to keep binary search bounds
                    # finite (avoids NaN pivots).
                    finite_blk = tl.where(
                        logits_blk > -float("inf"), logits_blk, float("inf")
                    )
                    min_logit = tl.minimum(min_logit, tl.min(finite_blk))

                    probs_blk = tl.exp(logits_blk - max_sample)
                    probs_blk = tl.where(mask_n, probs_blk, 0.0)
                    sum_exp_logits += tl.sum(probs_blk)

                # If no finite logits exist (all -inf), clamp min to
                # max so the search converges to -inf (no masking).
                min_logit = tl.minimum(min_logit, max_logit)

                idx = tl.cast(p * 200, tl.int32)
                idx = tl.maximum(0, tl.minimum(idx, 199))
                sigma = tl.load(NORMAL_CDF_TO_SIGMA_TABLE + idx)
                sigma = sigma + tl.abs(sigma) * -0.25
                outlier_pivot = avg_logit + std_logit * sigma

                outlier_prob = tl.exp(outlier_pivot - max_sample) / sum_exp_logits
                sum_outlier_probs = 0.0
                num_outliers = tl.zeros((), dtype=tl.uint32)

                # Second pass: Calculate softmax and gather outliers
                for i in range(0, NUM_TILES):
                    offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    mask_n = offs_n < VOCAB_SIZE

                    probs_blk = tl.load(
                        LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                    )
                    probs_blk = tl.exp(probs_blk - max_sample)
                    probs_blk = probs_blk / sum_exp_logits

                    outlier_mask = (probs_blk > outlier_prob) & mask_n
                    sum_outlier_probs += tl.sum(outlier_mask * probs_blk)
                    cumulative_pos = tl.cast(
                        tl.cumsum(outlier_mask) - 1 + num_outliers, tl.int32
                    )
                    num_outliers += tl.sum(outlier_mask)
                    write_pos = tl.where(outlier_mask, cumulative_pos, -1)
                    tl.store(BUFFER_ROW + write_pos, probs_blk, mask=outlier_mask)

                max_range = tl.exp(max_logit - max_sample) / sum_exp_logits
                min_range = tl.exp(min_logit - max_sample) / sum_exp_logits

                p_pivot = 1.0
                num_iters = 0
                min_larger_prob = 1.0
                num_min_larger = tl.zeros((), dtype=tl.uint32)
                p_pivots_sum = 0.0

                # Third pass: Search for p_pivot
                if sum_outlier_probs > p:
                    min_range = outlier_prob
                    search_range = tl.cast(num_outliers, tl.int32)
                    search_iters = tl.cast(
                        (num_outliers + BLOCK_SIZE_TRUNC - 1) // BLOCK_SIZE_TRUNC,
                        tl.int32,
                    )

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(0, search_iters):
                            offs_n = i * BLOCK_SIZE_TRUNC + tl.arange(
                                0, BLOCK_SIZE_TRUNC
                            )
                            mask_n_2 = offs_n < search_range
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n_2, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1
                else:
                    # Re-populate the buffer with full softmax probabilities
                    for i in range(0, NUM_TILES):
                        offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                        mask_n = offs_n < VOCAB_SIZE

                        probs_blk = tl.load(
                            LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                        )
                        probs_blk = tl.exp(probs_blk - max_sample)
                        probs_blk = probs_blk / sum_exp_logits
                        tl.store(BUFFER_ROW + offs_n, probs_blk, mask=mask_n)

                    found_pivot = 0
                    while found_pivot == 0:
                        p_pivot_0 = (max_range - min_range) * 0.5 + min_range
                        p_pivots_sum_0 = 0.0
                        min_larger_0 = 1.0
                        num_min_larger_0 = tl.zeros((), dtype=tl.uint32)

                        # Single fused pass: compute p_pivots_sum,
                        # min_larger, and num_min_larger together.
                        # See _update_min_larger_stats for the
                        # tile-level merge logic.
                        for i in range(0, NUM_TILES):
                            offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                            mask_n = offs_n < VOCAB_SIZE
                            probs_blk = tl.load(
                                BUFFER_ROW + offs_n, mask=mask_n, other=0.0
                            )

                            above_0 = probs_blk > p_pivot_0
                            p_pivots_sum_0 += tl.sum(probs_blk * above_0)

                            min_larger_0, num_min_larger_0 = _update_min_larger_stats(
                                probs_blk,
                                above_0,
                                min_larger_0,
                                num_min_larger_0,
                                1.0,
                            )

                        # Check if the pivot satisfies termination condition
                        if (
                            p_pivots_sum_0 >= p
                            and p_pivots_sum_0 - (min_larger_0 * num_min_larger_0) < p
                        ):
                            p_pivot = p_pivot_0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                        # Update range
                        if p_pivots_sum_0 > p:
                            min_range = p_pivot_0
                        elif p_pivots_sum_0 < p:
                            max_range = p_pivot_0

                        num_iters += 1
                        if (max_range - min_range) < 1e-9 or num_iters >= 18:
                            p_pivot = (max_range + min_range) / 2.0
                            min_larger_prob = min_larger_0
                            num_min_larger = num_min_larger_0
                            p_pivots_sum = p_pivots_sum_0
                            found_pivot = 1

                duplicate_logit = tl.log(min_larger_prob * sum_exp_logits) + max_sample
                num_duplicate_logit = num_min_larger
                num_keep = num_duplicate_logit - tl.cast(
                    (p_pivots_sum - p) / min_larger_prob, tl.uint32
                )
                num_kept = tl.zeros((), dtype=tl.uint32)

                # Top-p only path
                final_pivot = tl.log(p_pivot * sum_exp_logits) + max_sample

        # Sixth pass: Apply mask and store final output.
        # If the pivot >= max logit (or is NaN), no token would
        # survive the strict `>` keep_mask.  Skip masking.
        # Using `not <` instead of `>=` so that NaN is also caught.
        if not (final_pivot < max_logit):
            final_pivot = -float("inf")

        if OUTPUT_PIVOT:
            tl.store(PIVOTS + row_id, final_pivot)
        if RESET_COUNTS:
            tl.store(COUNTS + row_id, 0)

        if APPLY_FINAL_MASK and final_pivot != -float("inf"):
            for i in range(0, NUM_TILES):
                offs_n = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask_n = offs_n < VOCAB_SIZE
                logits_blk = tl.load(
                    LOGITS_ROW + offs_n, mask=mask_n, other=-float("inf")
                )
                keep_mask = (logits_blk > final_pivot) & mask_n

                # Duplicate logit handling
                if num_keep < num_duplicate_logit:
                    duplicate_mask = (
                        tl.abs(logits_blk - duplicate_logit) < 1e-9
                    ) & mask_n
                    duplicate_count = tl.cumsum(duplicate_mask) + num_kept
                    duplicate_keep_mask = (duplicate_count <= num_keep) & duplicate_mask
                    duplicate_remove_mask = duplicate_mask & ~duplicate_keep_mask
                    num_kept += tl.sum(duplicate_keep_mask)
                    keep_mask = keep_mask & (~duplicate_remove_mask)

                logits_blk = tl.where(keep_mask, logits_blk, MASK_VALUE)
                tl.store(LOGITS_ROW + offs_n, logits_blk, mask=mask_n)


def apply_top_k_top_p_triton(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """
    Apply combined top-k and top-p masking using Triton.

    Top-k is applied first (by logit value), then top-p is applied
    to the remaining k values (by probability).

    Args:
        logits: [batch_size, vocab_size] float32 tensor. The returned tensor
            may alias this input or be a new contiguous tensor for unsupported
            layouts.
        k: [batch_size] int32 tensor of top-k values per row, or None to disable top-k
        p: [batch_size] float32 tensor of top-p values per row (0 to 1),
            or None to disable top-p
        mask_value: Value for masked positions (default: -inf)

    Returns:
        The masked logits tensor. It may or may not be modified in-place.
    """
    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    batch_size, vocab_size = logits.shape
    topk_enabled = k is not None
    topp_enabled = p is not None

    if batch_size == 0 or not (topk_enabled or topp_enabled):
        return logits

    # The Triton kernel supports arbitrary row strides, but it still assumes
    # the vocab dimension is laid out contiguously within each row.
    if logits.stride(1) != 1:
        logits = logits.contiguous()

    if k is not None:
        assert k.ndim == 1 and k.shape[0] == batch_size
        k_ptr = k.to(torch.int32)
    else:
        k_ptr = logits  # Dummy pointer (won't be read)

    if p is not None:
        assert p.ndim == 1 and p.shape[0] == batch_size
        p_ptr = p.to(torch.float32)
    else:
        p_ptr = logits  # Dummy pointer (won't be read)

    num_sm = num_compute_units(logits.device.index)
    NUM_PROGRAMS = min(num_sm, batch_size)

    # Cache per-Triton Program buffer on each device.
    buf_key = (logits.device, logits.dtype, vocab_size)
    buffer = _TRITON_BUFFER_CACHE.get(buf_key)
    if buffer is None or buffer.shape[0] < NUM_PROGRAMS:
        size = min(next_power_of_2(NUM_PROGRAMS), num_sm)
        buffer = logits.new_empty((size, vocab_size))
        _TRITON_BUFFER_CACHE[buf_key] = buffer
    if buffer.shape[0] > NUM_PROGRAMS:
        buffer = buffer[:NUM_PROGRAMS]

    # Cache lookup table entries on each device.
    tables = _TRITON_TABLE_CACHE.get(logits.device)
    if tables is None:
        with gpu_sync_allowed():
            normal_cdf_to_sigma_table = logits.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)
            percentile_to_std_table = logits.new_tensor(_PERCENTILE_TO_STD_TABLE)
            _TRITON_TABLE_CACHE[logits.device] = (
                normal_cdf_to_sigma_table,
                percentile_to_std_table,
            )
    else:
        normal_cdf_to_sigma_table, percentile_to_std_table = tables

    # Smaller tiles compile and run faster on CPU; GPU benefits from larger tiles.
    # On XPU, large BLOCK_SIZE causes precision loss in the single-pass pivot
    # approximation; use smaller tiles for accurate top-p results.
    launch_kwargs = {}
    if logits.device.type == "cpu":
        block_size, block_size_trunc = 256, 128
    elif logits.device.type == "xpu":
        block_size, block_size_trunc = 4096, 2048
    else:
        block_size, block_size_trunc = 8192, 4096
        # Each program serially sweeps the vocab row in BLOCK_SIZE tiles, so
        # per-tile latency bounds kernel latency, and Triton's default of 4
        # warps leaves an 8192-wide tile at 16 elements per lane. 8 warps is
        # faster on every arch measured (SM90, SM100, SM120, gfx950); 16 is not.
        launch_kwargs["num_warps"] = 8

    _topk_topp_kernel[(NUM_PROGRAMS,)](
        logits,
        logits.stride(0),
        buffer,
        percentile_to_std_table,
        normal_cdf_to_sigma_table,
        k_ptr,
        p_ptr,
        logits,
        logits,
        BATCH_SIZE=batch_size,
        MASK_VALUE=mask_value,
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=block_size,
        BLOCK_SIZE_TRUNC=block_size_trunc,
        TOPK_ENABLED=topk_enabled,
        TOPP_ENABLED=topp_enabled,
        APPLY_FINAL_MASK=True,
        OUTPUT_PIVOT=False,
        RESET_COUNTS=False,
        **launch_kwargs,
    )

    return logits


@triton.jit
def _rand64(seed, offset, includes_zero: tl.constexpr):
    lo, hi, _, _ = tl.randint4x(seed, offset)
    lo = lo.to(tl.uint32, bitcast=True).to(tl.uint64)
    hi = hi.to(tl.uint32, bitcast=True).to(tl.uint64)
    r = (hi << 32) | lo

    # 1 / 2**64
    scale = 5.421010862427522170037e-20
    u = r.to(tl.float64) * scale
    u = tl.minimum(u, 0.9999999999999999)
    if not includes_zero:
        u = tl.maximum(u, 2.2250738585072014e-308)  # float64 tiny
    return u


@triton.jit
def _compact_masked_topk_kernel(
    LOGITS,
    LOGITS_STRIDE,
    CANDIDATE_IDS,
    CANDIDATE_LOGITS,
    COUNTS,
    VOCAB_SIZE: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < VOCAB_SIZE

    logits = tl.load(
        LOGITS + row_id * LOGITS_STRIDE + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    keep = (logits > -float("inf")) & mask
    keep_i32 = keep.to(tl.int32)
    num_keep = tl.sum(keep_i32, axis=0)
    local_pos = tl.cumsum(keep_i32, axis=0) - 1
    base = tl.atomic_add(COUNTS + row_id, num_keep, sem="relaxed")
    write_pos = base + local_pos
    write_mask = keep & (write_pos < MAX_CANDIDATES)

    tl.store(
        CANDIDATE_LOGITS + row_id * MAX_CANDIDATES + write_pos,
        logits,
        mask=write_mask,
    )
    tl.store(
        CANDIDATE_IDS + row_id * MAX_CANDIDATES + write_pos,
        offs,
        mask=write_mask,
    )


def apply_top_k_top_p_and_sample_small_topk_parallel_triton(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor | None,
    seeds: torch.Tensor,
    max_top_k: int,
    mask_value: float = float("-inf"),
) -> torch.Tensor:
    """Compute top-k/top-p pivots, then parallel mask+compact and sample."""
    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    assert k.ndim == 1 and k.shape[0] == logits.shape[0]
    assert seeds.ndim == 1 and seeds.shape[0] == logits.shape[0]
    if p is not None:
        assert p.ndim == 1 and p.shape[0] == logits.shape[0]

    batch_size, vocab_size = logits.shape
    if batch_size == 0:
        return torch.empty((0,), dtype=torch.int64, device=logits.device)
    if logits.stride(1) != 1:
        logits = logits.contiguous()

    max_candidates = min(64, next_power_of_2(max(1, max_top_k)))
    candidate_ids, candidate_logits, counts, sampled = _get_small_topk_sample_buffers(
        logits, batch_size, max_candidates
    )
    k_ptr = k.to(torch.int32)
    if p is not None:
        p_ptr = p.to(torch.float32)
        topp_enabled = True
    else:
        p_ptr = logits
        topp_enabled = False

    num_sm = num_compute_units(logits.device.index)
    NUM_PROGRAMS = min(num_sm, batch_size)

    buf_key = (logits.device, logits.dtype, vocab_size)
    buffer = _TRITON_BUFFER_CACHE.get(buf_key)
    if buffer is None or buffer.shape[0] < NUM_PROGRAMS:
        size = min(next_power_of_2(NUM_PROGRAMS), num_sm)
        buffer = logits.new_empty((size, vocab_size))
        _TRITON_BUFFER_CACHE[buf_key] = buffer
    if buffer.shape[0] > NUM_PROGRAMS:
        buffer = buffer[:NUM_PROGRAMS]

    tables = _TRITON_TABLE_CACHE.get(logits.device)
    if tables is None:
        normal_cdf_to_sigma_table = logits.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)
        percentile_to_std_table = logits.new_tensor(_PERCENTILE_TO_STD_TABLE)
        _TRITON_TABLE_CACHE[logits.device] = (
            normal_cdf_to_sigma_table,
            percentile_to_std_table,
        )
    else:
        normal_cdf_to_sigma_table, percentile_to_std_table = tables

    _topk_topp_kernel[(NUM_PROGRAMS,)](
        logits,
        logits.stride(0),
        buffer,
        percentile_to_std_table,
        normal_cdf_to_sigma_table,
        k_ptr,
        p_ptr,
        logits,
        counts,
        BATCH_SIZE=batch_size,
        MASK_VALUE=mask_value,
        VOCAB_SIZE=vocab_size,
        BLOCK_SIZE=8192,
        BLOCK_SIZE_TRUNC=4096,
        TOPK_ENABLED=True,
        TOPP_ENABLED=topp_enabled,
        APPLY_FINAL_MASK=True,
        OUTPUT_PIVOT=False,
        RESET_COUNTS=False,
    )

    counts.zero_()
    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    _compact_masked_topk_kernel[(batch_size, num_blocks)](
        logits,
        logits.stride(0),
        candidate_ids,
        candidate_logits,
        counts,
        VOCAB_SIZE=vocab_size,
        MAX_CANDIDATES=max_candidates,
        BLOCK_SIZE=block_size,
    )

    _sample_compacted_topk_kernel[(batch_size,)](
        candidate_ids,
        candidate_logits,
        counts,
        seeds,
        sampled,
        MAX_CANDIDATES=max_candidates,
        BLOCK_SIZE=max_candidates,
    )
    return sampled


@triton.jit
def _sample_compacted_topk_kernel(
    CANDIDATE_IDS,
    CANDIDATE_LOGITS,
    COUNTS,
    SEEDS,
    SAMPLED,
    MAX_CANDIDATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    count = tl.minimum(tl.load(COUNTS + row_id), MAX_CANDIDATES)
    mask = offs < count

    logits = tl.load(
        CANDIDATE_LOGITS + row_id * MAX_CANDIDATES + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float64)

    candidate_ids = tl.load(
        CANDIDATE_IDS + row_id * MAX_CANDIDATES + offs,
        mask=mask,
        other=0,
    )
    seed = tl.load(SEEDS + row_id)
    # Key random values by token id rather than compact slot. Compaction uses
    # atomics across vocab tiles, whose completion order is not deterministic.
    u = _rand64(seed, candidate_ids, includes_zero=False)
    gumbel_noise = -tl.log(-tl.log(u))
    scores = tl.where(mask, logits + gumbel_noise, -float("inf"))
    _, idx = tl.max(scores, axis=0, return_indices=True)

    token_id = tl.sum(tl.where((offs == idx) & mask, candidate_ids, 0), axis=0)
    tl.store(SAMPLED + row_id, token_id)


@triton.jit
def _full_vocab_sample_block_kernel(
    LOGITS,
    SEEDS,
    EXCLUDE_TOKEN_IDS,
    SHARD_TOKEN_IDS,
    BLOCK_VALUES,
    BLOCK_INDICES,
    VOCAB_START: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    ACTIVE_VOCAB_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    SCALE_OVER_TEMPERATURE: tl.constexpr,
    HAS_EXCLUDE: tl.constexpr,
    HAS_SHARD_TOKEN_IDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    lane = tl.arange(0, BLOCK_SIZE)
    vocab_offsets = block_id * BLOCK_SIZE + lane
    mask = vocab_offsets < ACTIVE_VOCAB_SIZE
    if HAS_SHARD_TOKEN_IDS:
        global_offsets = tl.load(
            SHARD_TOKEN_IDS + vocab_offsets,
            mask=vocab_offsets < VOCAB_SIZE,
            other=-1,
        )
        mask = mask & (global_offsets >= 0)
    else:
        global_offsets = VOCAB_START + vocab_offsets
    if HAS_EXCLUDE:
        exclude_token_id = tl.load(EXCLUDE_TOKEN_IDS + row)
        mask = mask & (global_offsets != exclude_token_id)

    logits = tl.load(
        LOGITS + row * VOCAB_SIZE + vocab_offsets,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)
    logits = logits * SCALE_OVER_TEMPERATURE

    seed = tl.load(SEEDS + row)
    random_offsets = tl.where(mask, global_offsets, 0)
    u = _rand64(seed, random_offsets, includes_zero=False)
    gumbel_noise = -tl.log(-tl.log(u))
    scores = logits.to(tl.float64) + gumbel_noise
    scores = tl.where(mask, scores, -float("inf"))

    value, lane_idx = tl.max(scores, axis=0, return_indices=True)
    token_id = tl.max(tl.where((lane == lane_idx) & mask, global_offsets, 0), axis=0)
    out_offset = row * NUM_BLOCKS + block_id
    tl.store(BLOCK_VALUES + out_offset, value)
    tl.store(BLOCK_INDICES + out_offset, token_id)


@triton.jit
def _merge_full_vocab_sample_blocks_kernel(
    BLOCK_VALUES,
    BLOCK_INDICES,
    OUT_VALUES,
    OUT_INDICES,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_NUM_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK_NUM_BLOCKS)
    mask = lane < NUM_BLOCKS
    in_offset = row * NUM_BLOCKS + lane

    values = tl.load(BLOCK_VALUES + in_offset, mask=mask, other=-float("inf"))
    indices = tl.load(BLOCK_INDICES + in_offset, mask=mask, other=0)
    _, lane_idx = tl.max(values, axis=0, return_indices=True)
    token_id = tl.max(tl.where(lane == lane_idx, indices, 0), axis=0)

    tl.store(OUT_VALUES + row, tl.max(values, axis=0))
    tl.store(OUT_INDICES + row, token_id)


def sample_full_vocab_from_shard_triton(
    logits: torch.Tensor,
    *,
    vocab_start: int,
    active_vocab_size: int,
    seeds: torch.Tensor,
    shard_token_ids: torch.Tensor | None = None,
    exclude_token_ids: torch.Tensor | None = None,
    scale: float = 1.0,
    temperature: float = 1.0,
    block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample local vocab-shard winners with Gumbel-max.

    Returns per-row ``(score, global_token_id)`` winners. Reducing those winners
    across TP ranks is equivalent to sampling from the full global softmax,
    while communicating only one pair per row per rank.
    """
    assert logits.ndim == 2
    assert logits.is_cuda
    assert seeds.ndim == 1 and seeds.shape[0] == logits.shape[0]
    assert seeds.dtype == torch.int64
    if exclude_token_ids is not None:
        assert exclude_token_ids.ndim == 1
        assert exclude_token_ids.shape[0] == logits.shape[0]
        assert exclude_token_ids.dtype in (torch.int32, torch.int64)
    assert 0 < active_vocab_size <= logits.shape[1]
    assert temperature > 0.0
    if shard_token_ids is not None:
        assert shard_token_ids.ndim == 1
        assert shard_token_ids.shape[0] == logits.shape[1]
        assert shard_token_ids.dtype == torch.int64

    batch_size, vocab_size = logits.shape
    out_values = torch.empty((batch_size,), device=logits.device, dtype=torch.float64)
    out_indices = torch.empty((batch_size,), device=logits.device, dtype=torch.int32)
    if batch_size == 0:
        return out_values, out_indices

    if not logits.is_contiguous():
        logits = logits.contiguous()
    if exclude_token_ids is None:
        exclude_token_ids = seeds
        has_exclude = False
    else:
        if not exclude_token_ids.is_contiguous():
            exclude_token_ids = exclude_token_ids.contiguous()
        has_exclude = True
    if shard_token_ids is None:
        shard_token_ids = seeds
        has_shard_token_ids = False
    else:
        if not shard_token_ids.is_contiguous():
            shard_token_ids = shard_token_ids.contiguous()
        has_shard_token_ids = True

    num_blocks = triton.cdiv(active_vocab_size, block_size)
    block_values = torch.empty(
        (batch_size, num_blocks), device=logits.device, dtype=torch.float64
    )
    block_indices = torch.empty(
        (batch_size, num_blocks), device=logits.device, dtype=torch.int32
    )
    _full_vocab_sample_block_kernel[(batch_size, num_blocks)](
        logits,
        seeds,
        exclude_token_ids,
        shard_token_ids,
        block_values,
        block_indices,
        VOCAB_START=int(vocab_start),
        VOCAB_SIZE=vocab_size,
        ACTIVE_VOCAB_SIZE=active_vocab_size,
        NUM_BLOCKS=num_blocks,
        SCALE_OVER_TEMPERATURE=float(scale) / float(temperature),
        HAS_EXCLUDE=has_exclude,
        HAS_SHARD_TOKEN_IDS=has_shard_token_ids,
        BLOCK_SIZE=block_size,
    )
    _merge_full_vocab_sample_blocks_kernel[(batch_size,)](
        block_values,
        block_indices,
        out_values,
        out_indices,
        NUM_BLOCKS=num_blocks,
        BLOCK_NUM_BLOCKS=next_power_of_2(num_blocks),
    )
    return out_values, out_indices


@triton.jit
def _pack_topk_pairs_kernel(
    LOCAL_VALS,
    LOCAL_INDICES,
    LOCAL_PAIRS,
    VOCAB_START: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < TOP_K

    input_base = row_id * TOP_K
    vals = tl.load(LOCAL_VALS + input_base + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    indices = tl.load(LOCAL_INDICES + input_base + offsets, mask=mask, other=0).to(
        tl.int64
    )
    global_indices = indices + VOCAB_START

    output_base = row_id * TOP_K * 2
    tl.store(LOCAL_PAIRS + output_base + offsets * 2, vals, mask=mask)
    tl.store(
        LOCAL_PAIRS + output_base + offsets * 2 + 1,
        global_indices.to(tl.float32),
        mask=mask,
    )


def pack_topk_pairs_triton(
    local_vals: torch.Tensor,
    local_indices: torch.Tensor,
    vocab_start: int,
) -> torch.Tensor:
    """Pack local top-k values and local ids into compact gather pairs.

    The output shape is ``[batch, top_k * 2]`` with interleaved
    ``(logit, global_token_id_as_float)`` entries, matching the distributed
    gather send buffer consumed by compact top-k sampling.
    """
    assert local_vals.ndim == 2
    assert local_indices.shape == local_vals.shape
    assert local_vals.dtype == torch.float32
    assert local_indices.dtype == torch.int64

    batch_size, top_k = local_vals.shape
    local_pairs = torch.empty(
        (batch_size, top_k * 2), dtype=torch.float32, device=local_vals.device
    )
    if batch_size == 0 or top_k == 0:
        return local_pairs

    if not local_vals.is_contiguous():
        local_vals = local_vals.contiguous()
    if not local_indices.is_contiguous():
        local_indices = local_indices.contiguous()

    block = next_power_of_2(top_k)
    _pack_topk_pairs_kernel[(batch_size,)](
        local_vals,
        local_indices,
        local_pairs,
        VOCAB_START=int(vocab_start),
        TOP_K=top_k,
        BLOCK=block,
    )
    return local_pairs


@triton.jit
def _select_from_compact_topk_pairs_kernel(
    GATHERED_PAIRS,
    TOP_VALS_OUT,
    TOP_IDS_OUT,
    NUM_CANDIDATES: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_P: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)

    cand_offsets = tl.arange(0, CANDIDATE_BLOCK)
    cand_mask = cand_offsets < NUM_CANDIDATES
    pair_base = row_id * NUM_CANDIDATES * 2
    vals = tl.load(
        GATHERED_PAIRS + pair_base + cand_offsets * 2,
        mask=cand_mask,
        other=-float("inf"),
    ).to(tl.float32)
    ids = tl.load(
        GATHERED_PAIRS + pair_base + cand_offsets * 2 + 1,
        mask=cand_mask,
        other=0.0,
    ).to(tl.float32)

    top_offsets = tl.arange(0, TOPK_BLOCK)
    top_vals = tl.full((TOPK_BLOCK,), -float("inf"), tl.float32)
    top_ids = tl.full((TOPK_BLOCK,), 0.0, tl.float32)
    work_vals = vals

    for i in tl.static_range(0, TOP_K):
        max_val, max_idx = tl.max(work_vals, axis=0, return_indices=True)
        is_max = cand_offsets == max_idx
        token_id = tl.sum(tl.where(is_max, ids, 0.0), axis=0)
        top_vals = tl.where(top_offsets == i, max_val, top_vals)
        top_ids = tl.where(top_offsets == i, token_id, top_ids)
        work_vals = tl.where(is_max, -float("inf"), work_vals)

    valid_top = top_offsets < TOP_K
    if TOP_P < 1.0:
        max_top_val = tl.max(tl.where(valid_top, top_vals, -float("inf")), axis=0)
        weights = tl.exp(top_vals - max_top_val)
        weights = tl.where(valid_top, weights, 0.0)
        denom = tl.sum(weights, axis=0)
        probs = tl.where(denom > 0.0, weights / denom, 0.0)
        prev_cum_probs = tl.cumsum(probs, axis=0) - probs
        valid_top = valid_top & (prev_cum_probs <= TOP_P)

    out_base = row_id * TOP_K + top_offsets
    out_mask = top_offsets < TOP_K
    top_vals = tl.where(valid_top, top_vals, -float("inf"))
    tl.store(TOP_VALS_OUT + out_base, top_vals, mask=out_mask)
    tl.store(TOP_IDS_OUT + out_base, top_ids.to(tl.int64), mask=out_mask)


def select_compact_topk_pairs_triton(
    gathered_pairs: torch.Tensor,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse global compact top-k merge and top-p filtering.

    ``gathered_pairs`` is the TP-gathered compact candidate tensor with shape
    ``[batch, num_candidates, 2]`` where the last dimension stores
    ``(logit, token_id_as_float)``. The outputs have shape ``[batch, top_k]``.
    """
    assert gathered_pairs.ndim == 3
    assert gathered_pairs.shape[-1] == 2
    assert gathered_pairs.dtype == torch.float32
    assert 0 < top_k <= 64

    batch_size, num_candidates, _ = gathered_pairs.shape
    top_vals = torch.empty(
        (batch_size, top_k), dtype=torch.float32, device=gathered_pairs.device
    )
    top_ids = torch.empty(
        (batch_size, top_k), dtype=torch.int64, device=gathered_pairs.device
    )
    if batch_size == 0:
        return top_vals, top_ids
    if num_candidates < top_k:
        raise ValueError(
            f"num_candidates ({num_candidates}) must be >= top_k ({top_k})"
        )
    if num_candidates > 2048:
        raise ValueError(
            "compact top-k fused select only supports up to 2048 "
            f"candidates, got {num_candidates}"
        )

    if not gathered_pairs.is_contiguous():
        gathered_pairs = gathered_pairs.contiguous()

    candidate_block = next_power_of_2(num_candidates)
    topk_block = next_power_of_2(top_k)
    _select_from_compact_topk_pairs_kernel[(batch_size,)](
        gathered_pairs,
        top_vals,
        top_ids,
        NUM_CANDIDATES=num_candidates,
        TOP_K=top_k,
        TOP_P=float(top_p),
        CANDIDATE_BLOCK=candidate_block,
        TOPK_BLOCK=topk_block,
    )
    return top_vals, top_ids


@triton.jit
def _sample_from_compact_topk_pairs_kernel(
    GATHERED_PAIRS,
    SEEDS,
    SAMPLED,
    NUM_CANDIDATES: tl.constexpr,
    TOP_K: tl.constexpr,
    TOP_P: tl.constexpr,
    CANDIDATE_BLOCK: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)

    cand_offsets = tl.arange(0, CANDIDATE_BLOCK)
    cand_mask = cand_offsets < NUM_CANDIDATES
    pair_base = row_id * NUM_CANDIDATES * 2
    vals = tl.load(
        GATHERED_PAIRS + pair_base + cand_offsets * 2,
        mask=cand_mask,
        other=-float("inf"),
    ).to(tl.float32)
    ids = tl.load(
        GATHERED_PAIRS + pair_base + cand_offsets * 2 + 1,
        mask=cand_mask,
        other=0.0,
    ).to(tl.float32)

    top_offsets = tl.arange(0, TOPK_BLOCK)
    top_vals = tl.full((TOPK_BLOCK,), -float("inf"), tl.float32)
    top_ids = tl.full((TOPK_BLOCK,), 0.0, tl.float32)
    work_vals = vals

    for i in tl.static_range(0, TOP_K):
        max_val, max_idx = tl.max(work_vals, axis=0, return_indices=True)
        is_max = cand_offsets == max_idx
        token_id = tl.sum(tl.where(is_max, ids, 0.0), axis=0)
        top_vals = tl.where(top_offsets == i, max_val, top_vals)
        top_ids = tl.where(top_offsets == i, token_id, top_ids)
        work_vals = tl.where(is_max, -float("inf"), work_vals)

    valid_top = top_offsets < TOP_K
    if TOP_P >= 1.0:
        keep = valid_top
    else:
        max_top_val = tl.max(tl.where(valid_top, top_vals, -float("inf")), axis=0)
        weights = tl.exp(top_vals - max_top_val)
        weights = tl.where(valid_top, weights, 0.0)
        denom = tl.sum(weights, axis=0)
        probs = tl.where(denom > 0.0, weights / denom, 0.0)
        prev_cum_probs = tl.cumsum(probs, axis=0) - probs
        keep = valid_top & (prev_cum_probs <= TOP_P)

    seed = tl.load(SEEDS + row_id)
    u = _rand64(seed, top_offsets, includes_zero=False)
    gumbel_noise = -tl.log(-tl.log(u))
    scores = tl.where(keep, top_vals.to(tl.float64) + gumbel_noise, -float("inf"))
    _, sampled_idx = tl.max(scores, axis=0, return_indices=True)
    sampled_id = tl.sum(tl.where(top_offsets == sampled_idx, top_ids, 0.0), axis=0)
    tl.store(SAMPLED + row_id, sampled_id.to(tl.int64))


def sample_from_compact_topk_pairs_triton(
    gathered_pairs: torch.Tensor,
    top_k: int,
    top_p: float,
    seeds: torch.Tensor,
) -> torch.Tensor:
    """Fuse global compact top-k merge, top-p filtering, and sampling.

    ``gathered_pairs`` is the TP-gathered compact candidate tensor with shape
    ``[batch, num_candidates, 2]`` where the last dimension stores
    ``(logit, token_id_as_float)``. The output is one sampled global token id
    per row.
    """
    assert gathered_pairs.ndim == 3
    assert gathered_pairs.shape[-1] == 2
    assert gathered_pairs.dtype == torch.float32
    assert seeds.ndim == 1 and seeds.shape[0] == gathered_pairs.shape[0]
    assert seeds.dtype == torch.int64
    assert 0 < top_k <= 64

    batch_size, num_candidates, _ = gathered_pairs.shape
    if batch_size == 0:
        return torch.empty((0,), dtype=torch.int64, device=gathered_pairs.device)
    if num_candidates < top_k:
        raise ValueError(
            f"num_candidates ({num_candidates}) must be >= top_k ({top_k})"
        )
    if num_candidates > 2048:
        raise ValueError(
            "compact top-k fused sampling only supports up to 2048 "
            f"candidates, got {num_candidates}"
        )

    if not gathered_pairs.is_contiguous():
        gathered_pairs = gathered_pairs.contiguous()

    sampled = torch.empty(
        (batch_size,), dtype=torch.int64, device=gathered_pairs.device
    )
    candidate_block = next_power_of_2(num_candidates)
    topk_block = next_power_of_2(top_k)
    _sample_from_compact_topk_pairs_kernel[(batch_size,)](
        gathered_pairs,
        seeds,
        sampled,
        NUM_CANDIDATES=num_candidates,
        TOP_K=top_k,
        TOP_P=float(top_p),
        CANDIDATE_BLOCK=candidate_block,
        TOPK_BLOCK=topk_block,
    )
    return sampled


@triton.jit
def _sample_recovered_compacted_topk_kernel(
    CANDIDATE_IDS,
    CANDIDATE_LOGITS,
    COUNTS,
    DRAFT_TOKEN_IDS,
    SEEDS,
    RECOVERED_TOKEN_IDS,
    TARGET_DRAFT_PROBS,
    MAX_CANDIDATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    count = tl.minimum(tl.load(COUNTS + row_id), MAX_CANDIDATES)
    mask = offs < count

    candidate_ids = tl.load(
        CANDIDATE_IDS + row_id * MAX_CANDIDATES + offs,
        mask=mask,
        other=-1,
    )
    logits = tl.load(
        CANDIDATE_LOGITS + row_id * MAX_CANDIDATES + offs,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float64)

    draft_token_id = tl.load(DRAFT_TOKEN_IDS + row_id)
    is_draft = mask & (candidate_ids == draft_token_id)
    recovered_mask = mask & (candidate_ids != draft_token_id)

    max_logit = tl.max(tl.where(mask, logits, -float("inf")), axis=0)
    weights = tl.exp(logits - max_logit)
    denom = tl.sum(tl.where(mask, weights, 0.0), axis=0)
    draft_weight = tl.sum(tl.where(is_draft, weights, 0.0), axis=0)
    target_draft_prob = tl.where(denom > 0.0, draft_weight / denom, 0.0)

    seed = tl.load(SEEDS + row_id)
    random_offsets = tl.where(mask, candidate_ids, 0)
    u = _rand64(seed, random_offsets, includes_zero=False)
    gumbel_noise = -tl.log(-tl.log(u))
    scores = tl.where(recovered_mask, logits + gumbel_noise, -float("inf"))
    _, idx = tl.max(scores, axis=0, return_indices=True)

    has_recovered = tl.sum(recovered_mask.to(tl.int32), axis=0) > 0
    recovered_token_id = tl.load(
        CANDIDATE_IDS + row_id * MAX_CANDIDATES + idx,
        mask=has_recovered,
        other=0,
    )
    tl.store(RECOVERED_TOKEN_IDS + row_id, recovered_token_id)
    tl.store(TARGET_DRAFT_PROBS + row_id, target_draft_prob)


def _get_small_topk_sample_buffers(
    logits: torch.Tensor,
    batch_size: int,
    max_candidates: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_bucket = next_power_of_2(batch_size)
    key = (logits.device, logits.dtype, batch_bucket, max_candidates)
    buffers = _TRITON_SMALL_TOPK_SAMPLE_CACHE.get(key)
    if buffers is None:
        candidate_ids = torch.empty(
            (batch_bucket, max_candidates), dtype=torch.int64, device=logits.device
        )
        candidate_logits = torch.empty(
            (batch_bucket, max_candidates), dtype=logits.dtype, device=logits.device
        )
        counts = torch.empty((batch_bucket,), dtype=torch.int32, device=logits.device)
        sampled = torch.empty((batch_bucket,), dtype=torch.int64, device=logits.device)
        buffers = (candidate_ids, candidate_logits, counts, sampled)
        _TRITON_SMALL_TOPK_SAMPLE_CACHE[key] = buffers

    candidate_ids, candidate_logits, counts, sampled = buffers
    return (
        candidate_ids[:batch_size],
        candidate_logits[:batch_size],
        counts[:batch_size],
        sampled[:batch_size],
    )


def sample_masked_small_topk_triton(
    logits: torch.Tensor,
    seeds: torch.Tensor,
    max_top_k: int,
) -> torch.Tensor:
    """
    Sample from logits after top-k/top-p masking when top_k is small.

    `apply_top_k_top_p_triton` has already set masked logits to -inf. For
    small top_k, only a handful of finite entries remain per row, so compact
    those entries and run Gumbel sampling over the compact candidate set instead
    of materializing full-vocab probabilities and exponential noise.
    """
    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    assert seeds.ndim == 1 and seeds.shape[0] == logits.shape[0]

    batch_size, vocab_size = logits.shape
    if batch_size == 0:
        return torch.empty((0,), dtype=torch.int64, device=logits.device)
    if logits.stride(1) != 1:
        logits = logits.contiguous()

    max_candidates = min(64, next_power_of_2(max(1, max_top_k)))
    candidate_ids, candidate_logits, counts, sampled = _get_small_topk_sample_buffers(
        logits, batch_size, max_candidates
    )
    counts.zero_()

    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    _compact_masked_topk_kernel[(batch_size, num_blocks)](
        logits,
        logits.stride(0),
        candidate_ids,
        candidate_logits,
        counts,
        VOCAB_SIZE=vocab_size,
        MAX_CANDIDATES=max_candidates,
        BLOCK_SIZE=block_size,
    )

    _sample_compacted_topk_kernel[(batch_size,)](
        candidate_ids,
        candidate_logits,
        counts,
        seeds,
        sampled,
        MAX_CANDIDATES=max_candidates,
        BLOCK_SIZE=max_candidates,
    )
    return sampled


def sample_recovered_no_draft_probs_triton(
    logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    seeds: torch.Tensor,
    max_top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compact masked top-k logits, sample recovered tokens excluding each draft
    # token, and return the target probability of each draft token.
    assert logits.ndim == 2
    assert logits.dtype == torch.float32
    assert draft_token_ids.ndim == 1 and draft_token_ids.shape[0] == logits.shape[0]
    assert seeds.ndim == 1 and seeds.shape[0] == logits.shape[0]

    batch_size, vocab_size = logits.shape
    recovered = torch.empty_like(draft_token_ids)
    target_draft_probs = torch.empty(
        (batch_size,), dtype=torch.float32, device=logits.device
    )
    if batch_size == 0:
        return recovered, target_draft_probs
    if logits.stride(1) != 1:
        logits = logits.contiguous()

    max_candidates = min(64, next_power_of_2(max(1, max_top_k)))
    candidate_ids, candidate_logits, counts, _ = _get_small_topk_sample_buffers(
        logits, batch_size, max_candidates
    )
    counts.zero_()

    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    _compact_masked_topk_kernel[(batch_size, num_blocks)](
        logits,
        logits.stride(0),
        candidate_ids,
        candidate_logits,
        counts,
        VOCAB_SIZE=vocab_size,
        MAX_CANDIDATES=max_candidates,
        BLOCK_SIZE=block_size,
    )

    _sample_recovered_compacted_topk_kernel[(batch_size,)](
        candidate_ids,
        candidate_logits,
        counts,
        draft_token_ids,
        seeds,
        recovered,
        target_draft_probs,
        MAX_CANDIDATES=max_candidates,
        BLOCK_SIZE=max_candidates,
    )
    return recovered, target_draft_probs


def reset_buffer_cache():
    _TRITON_BUFFER_CACHE.clear()
    _TRITON_TABLE_CACHE.clear()
    _TRITON_SMALL_TOPK_SAMPLE_CACHE.clear()
    torch.accelerator.empty_cache()
