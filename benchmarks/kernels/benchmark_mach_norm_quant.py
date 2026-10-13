# SPDX-License-Identifier: Apache-2.0
"""Compare compiled BF16 residual norm + quant against the fused producer.

Reports CUDA Graph latency and payload differences, not serving throughput.
"""

import json

import torch
from benchmark_mach_two_b_native import graph_ms

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    fused_add_rms_norm_mxfp8_quant,
    mxfp8_e4m3_quantize,
)


def reference_norm(x, residual, gamma):
    values = x.float() + residual.float()
    normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    return (normalized * gamma).to(x.dtype), values.to(x.dtype)


@torch.inference_mode()
def main():
    torch.manual_seed(20261013)
    norm = torch.compile(reference_norm, fullgraph=True)
    for k in (2048, 2560):
        for m in (1, 4, 16, 32, 64):
            x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            gamma = 1 + torch.randn(k, device="cuda", dtype=torch.bfloat16).float()

            def reference(x=x, residual=residual, gamma=gamma):
                y, r = norm(x, residual, gamma)
                q, s = mxfp8_e4m3_quantize(y, is_sf_swizzled_layout=True)
                return q, s, r

            def fused(x=x, residual=residual, gamma=gamma):
                return fused_add_rms_norm_mxfp8_quant(x, residual, gamma, 1e-6)

            q, s, r = reference()
            fq, fs, fr = fused()
            # Padding bytes are not part of the scale payload.
            row = torch.arange(m, device="cuda")[:, None]
            group = torch.arange(k // 32, device="cuda")[None, :]
            offsets = (
                ((row // 128 * ((k // 32 + 3) // 4) + group // 4) * 32 + row % 32) * 4
                + row % 128 // 32
            ) * 4 + group % 4
            print(
                json.dumps(
                    dict(
                        m=m,
                        k=k,
                        payload_mismatch=int(
                            (q.view(torch.uint8) != fq.view(torch.uint8)).sum()
                        ),
                        scale_mismatch=int(
                            (s.flatten()[offsets] != fs.flatten()[offsets]).sum()
                        ),
                        residual_equal=torch.equal(r, fr),
                        reference_us=graph_ms(reference) * 1000,
                        fused_us=graph_ms(fused) * 1000,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
