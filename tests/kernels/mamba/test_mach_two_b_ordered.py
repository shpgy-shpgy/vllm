# SPDX-License-Identifier: Apache-2.0
"""W4 must preserve stock recurrence across flush, recycle and graph replay.

Only the kernel contract is tested here, not scheduler/model integration.
"""

import pytest
import torch

from vllm.mach.mxfp6.gdn import two_b_ordered as ordered
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule_packed_decode as stock,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(120),
    reason="Requires SM120",
)


def inputs(rows, padded):
    torch.manual_seed(71)
    pages = rows + 3
    storage = torch.randn(pages, 16 * 128 * 128 + padded, device="cuda")
    base = storage[:, : 16 * 128 * 128].view(pages, 16, 128, 128)
    scratch = ordered.DeferredScratch(
        **{
            name: torch.zeros(
                shape,
                dtype=torch.int32 if name == "age" else torch.float32,
                device="cuda",
            )
            for name, shape in ordered.required_shapes(pages, 4).items()
        }
    )
    scratch.prefix.fill_(1)
    # Match packed QKVZ storage: post-conv QKV can have a wider row pitch.
    qkv = torch.randn(rows, 8192, device="cuda", dtype=torch.bfloat16)[:, :6144]
    ab = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
    a, b = ab[:, :16], ab[:, 16:]
    alog = torch.randn(16, device="cuda") - 2
    bias = torch.randn(16, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(rows, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    # Also exercise a noncontiguous ID vector.
    indices = torch.arange(1, rows * 2 + 1, device="cuda", dtype=torch.int32)[::2]
    indices.copy_(torch.arange(1, rows + 1, device="cuda"))
    return base, scratch, qkv, a, b, alog, bias, out, indices


def reference(qkv, a, b, alog, bias, base, out, indices):
    stock(qkv, a, b, alog, bias, 128**-0.5, base, out, indices, True)


@pytest.mark.parametrize("rows", [32, 48, 64, 96, 128, 160])
@pytest.mark.parametrize("padded", [0, 320])
@torch.inference_mode()
def test_ordered_matches_stock_across_flush_and_stock_transition(rows, padded):
    base, scratch, qkv, a, b, alog, bias, out, ids = inputs(rows, padded)
    expected = base.clone()
    expected_out = torch.empty_like(out)
    sentinel = base[0].clone()
    for step in range(11):
        qkv.normal_()
        a.normal_()
        b.normal_()
        # Rotate requests so pending state is looked up by slot, not batch row.
        ids.copy_(ids.roll(1))
        ordered.decode(qkv, a, b, alog, bias, base, out, ids, scratch)
        reference(qkv, a, b, alog, bias, expected, expected_out, ids)
        torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
        if step in (2, 7, 10):
            ordered.materialize_slots(base, ids, scratch)
            torch.testing.assert_close(base, expected, rtol=0, atol=0)
            # Stock may now update dense state, e.g. low-M decode/prefill.
            reference(qkv[:4], a[:4], b[:4], alog, bias, base, out[:4], ids[:4])
            reference(
                qkv[:4],
                a[:4],
                b[:4],
                alog,
                bias,
                expected,
                expected_out[:4],
                ids[:4],
            )
    torch.testing.assert_close(base[0], sentinel, rtol=0, atol=0)


@torch.inference_mode()
def test_recycled_slots_discard_pending_and_invalid_slots_do_not_write():
    base, scratch, qkv, a, b, alog, bias, out, ids = inputs(32, 320)
    ids[:3] = torch.tensor([0, -1, base.shape[0]], device="cuda")
    original = base.clone()
    ordered.decode(qkv, a, b, alog, bias, base, out, ids, scratch)
    assert torch.count_nonzero(out[:3]).item() == 0
    torch.testing.assert_close(base, original, rtol=0, atol=0)
    ordered.reset_slots(base, ids, scratch, zero_base=True)
    assert torch.count_nonzero(scratch.age).item() == 0
    assert torch.count_nonzero(base[4:33]).item() == 0
    torch.testing.assert_close(base[:4], original[:4], rtol=0, atol=0)
    expected = base.clone()
    ordered.materialize_slots(base, ids, scratch)
    torch.testing.assert_close(base, expected, rtol=0, atol=0)
    # Empty lifecycle events are valid and should not launch a zero-size grid.
    ordered.materialize_slots(base, ids[:0], scratch)
    ordered.reset_slots(base, ids[:0], scratch)


@torch.inference_mode()
def test_graph_replay_reads_new_inputs_and_slot_ids():
    base, scratch, qkv, a, b, alog, bias, out, ids = inputs(64, 320)
    expected_out = torch.empty_like(out)
    ordered.decode(qkv, a, b, alog, bias, base, out, ids, scratch)
    ordered.materialize_slots(base, ids, scratch)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        ordered.decode(qkv, a, b, alog, bias, base, out, ids, scratch)
    torch.cuda.current_stream().wait_stream(stream)
    expected = base.clone()
    for _ in range(9):
        qkv.normal_()
        a.normal_()
        ids.copy_(ids.roll(3))
        graph.replay()
        reference(qkv, a, b, alog, bias, expected, expected_out, ids)
        torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    ordered.materialize_slots(base, ids, scratch)
    torch.testing.assert_close(base, expected, rtol=0, atol=0)


@torch.inference_mode()
def test_reject_bf16_state_and_4b_geometry():
    base, scratch, *_ = inputs(32, 0)
    with pytest.raises(ValueError, match="FP32"):
        ordered.validate(base.bfloat16(), scratch)
    with pytest.raises(ValueError, match="base must be"):
        ordered.validate(torch.empty(2, 32, 128, 128, device="cuda"), scratch)


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 16, 24, 32, 64])
@torch.inference_mode()
def test_four_b_full_dispatch_preserves_stock_recurrence(rows):
    from vllm.mach.mxfp6.gdn import (
        champion_ordered_full_m4_m16_deferred_dispatch as full,
    )
    from vllm.mach.mxfp6.gdn.ordered_m16_deferred import allocate, materialize_slots

    torch.manual_seed(81)
    base = torch.randn(rows + 2, 32, 128, 128, device="cuda")
    expected = base.clone()
    scratch = allocate(rows + 2, base.device)
    qkv = torch.randn(rows, 12288, device="cuda", dtype=torch.bfloat16)[:, :8192]
    a = torch.randn(rows, 32, device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    alog = torch.randn(32, device="cuda") - 2
    bias = torch.randn(32, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(rows, 1, 32, 128, device="cuda", dtype=torch.bfloat16)
    expected_out = torch.empty_like(out)
    ids = torch.arange(1, rows + 1, device="cuda", dtype=torch.int32)
    for _ in range(9):
        qkv.normal_()
        full.decode(qkv, a, b, alog, bias, base, out, ids, scratch)
        reference(qkv, a, b, alog, bias, expected, expected_out, ids)
        torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    materialize_slots(base, ids, scratch)
    torch.testing.assert_close(base, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 4, 16, 32, 64])
@torch.inference_mode()
def test_mixed_kernel_preserves_stock_mixed_normalization(rows):
    from vllm.mach.mxfp6.gdn import two_b_mixed_kernel as mixed
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    base, scratch, qkv, a, b, alog, bias, out, ids = inputs(rows, 320)
    expected = base.clone()
    cu = torch.arange(rows + 1, device="cuda", dtype=torch.int32)
    for _ in range(9):
        qkv.normal_()
        mixed._ordered_decode[(mixed.NV, mixed.HV, rows)](
            qkv,
            a,
            b,
            alog,
            bias,
            out,
            base,
            ids,
            scratch.pending_k,
            scratch.pending_d,
            scratch.coeff,
            scratch.prefix,
            scratch.age,
            qkv.stride(0),
            a.stride(0),
            b.stride(0),
            base.stride(0),
            ids.stride(0),
            base.shape[0],
            4,
            128**-0.5,
            16,
            16,
            128,
            128,
            32,
            4,
            num_warps=4,
            num_stages=3,
        )
        q, k, v = (t.view(1, rows, 16, 128) for t in qkv.split(2048, dim=-1))
        expected_out, _ = fused_sigmoid_gating_delta_rule_update(
            alog,
            a,
            b,
            bias,
            q,
            k,
            v,
            initial_state=expected,
            inplace_final_state=True,
            cu_seqlens=cu,
            ssm_state_indices=ids,
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(out, expected_out.transpose(0, 1), rtol=0, atol=0)
    ordered.materialize_slots(base, ids, scratch)
    torch.testing.assert_close(base, expected, rtol=0, atol=0)
