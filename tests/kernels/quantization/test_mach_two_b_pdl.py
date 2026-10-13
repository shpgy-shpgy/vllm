# SPDX-License-Identifier: Apache-2.0
"""Validate qualified producer math and the capture-only scheduling boundary."""

import copy
from types import SimpleNamespace

import pytest
import torch

from vllm.mach.mxfp6 import two_b_pdl as pdl
from vllm.platforms import current_platform


def test_norm_policy_matches_and_rejects_changed_math():
    specs = [s for s in pdl.policy()["kernels"] if pdl._target(s["name"])]
    assert len({s["semantic_key"] for s in specs}) == 3
    for spec in specs:
        assert pdl._descriptor(spec["name"], spec["original_source"])["spec"]
        changed = spec["original_source"].replace("1e-06", "1e-03")
        assert changed != spec["original_source"]
        assert pdl._descriptor(spec["name"], changed)["spec"] is None


@pytest.mark.parametrize("rows", [1, 4, 16, 24, 32, 64])
@pytest.mark.parametrize(
    "full,warmup,capturing",
    [
        (True, False, True),
        (True, True, False),
        (False, False, True),
        (True, False, False),
    ],
)
def test_capture_selection(monkeypatch, rows, full, warmup, capturing):
    monkeypatch.setattr(pdl, "_INSTALLED", True)
    monkeypatch.setattr(pdl, "_FULL_MODE", "full")
    monkeypatch.setattr(pdl, "_CAPTURING", lambda: capturing)
    monkeypatch.setattr(
        pdl,
        "_DESCRIPTOR_LOCAL",
        SimpleNamespace(
            stack=[
                {
                    "mode": "full" if full else "piecewise",
                    "rows": rows,
                    "warmup": warmup,
                }
            ]
        ),
    )
    assert pdl.selected_full(rows) == (
        rows in (16, 24) and full and not warmup and capturing
    )
    assert pdl.warming_full(rows) == (
        rows in (16, 24) and full and warmup and not capturing
    )


@pytest.mark.parametrize("missing", ["norm", "native", "swiglu"])
def test_capture_receipt_rejects_incomplete_chain(monkeypatch, missing):
    monkeypatch.setattr(pdl, "_INSTALLED", True)
    monkeypatch.setattr(pdl, "_READY", True)
    selected = {
        s["semantic_key"]: {"semantic_key": s["semantic_key"], "rows": 16}
        for s in pdl.policy()["kernels"]
        if pdl._target(s["name"])
    }
    counts = dict(native_capture_m16=1, producer_capture_m16=1, swiglu_capture_m16=1)
    if missing == "norm":
        selected.pop(next(iter(selected)))
    else:
        counts[f"{missing}_capture_m16"] = 0
    monkeypatch.setattr(pdl, "_SELECTED", selected)
    monkeypatch.setattr(pdl, "_COUNTS", counts)
    with pytest.raises(RuntimeError, match="Incomplete 2B FULL16"):
        pdl.verify_capture([16])


@pytest.mark.skipif(
    not current_platform.is_cuda() or not current_platform.is_device_capability(120),
    reason="Requires qualified SM120/Torch/Triton runtime",
)
@torch.inference_mode()
def test_precompiled_norm_clones_preserve_math():
    import triton
    from torch._inductor.codecache import PyCodeCache

    pdl.prepare()
    assert pdl.receipt()["precompiled_variants"] == 20
    for spec in pdl.policy()["kernels"]:
        if not pdl._target(spec["name"]):
            continue
        for layout in spec["layouts"]:
            for config in spec["allowed_configs"]:
                original = getattr(
                    PyCodeCache.load(
                        spec["original_source"]
                        + f"\n# qualification {layout['rows']} {config}\n"
                    ),
                    spec["name"],
                )
                # A fresh tuner per config, as in upstream's qualification.
                original.configs = [
                    triton.Config(
                        config["kwargs"],
                        num_warps=config["num_warps"],
                        num_stages=config["num_stages"],
                        num_ctas=config["num_ctas"],
                    )
                ]
                for key in (
                    "dynamic_scale_rblock",
                    "coordinate_descent_tuning",
                    "dynamic_disable_pipelining",
                ):
                    original.inductor_meta[key] = False
                original.save_cache_hook = None
                original.precompile()
                args = pdl._allocate(layout)
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        arg.normal_()
                reference = copy.deepcopy(args)
                original.run(*reference, stream=torch.cuda.current_stream().cuda_stream)
                clone = pdl._CLONES[pdl._kernel_key(spec, layout["rows"], config)]
                clone.run(*args, stream=torch.cuda.current_stream().cuda_stream)
                for actual, expected in zip(args, reference):
                    if isinstance(actual, torch.Tensor):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
