"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import itertools

import numpy as np
import torch

from flashinfer.cute_dsl.moe_grouped_gemm_fp8 import moe_grouped_gemm_fp8
from flashinfer.gemm import group_gemm_fp8_nt_groupwise
from flashinfer.testing.utils import bench_gpu_time


def bench_groupwise_grouped_gemm_fp8_blackwell(
    batch_size, m, n, k, in_dtype, out_dtype, idx, total
):
    print(
        f"  [{idx}/{total}] E={batch_size} m={m} N={n} K={k} ... ",
        end="",
        flush=True,
    )
    torch.random.manual_seed(0)
    total_M = batch_size * m
    a = torch.randn(total_M, k, device="cuda:0").to(in_dtype)
    b = torch.randn(batch_size, n, k, device="cuda:0").to(in_dtype)

    a_scale = torch.randn(
        (k // 128, total_M), dtype=torch.float32, device="cuda:0"
    )
    # b_scale: [E, N//128, K//128]
    b_scale = torch.randn(
        (batch_size, n // 128, k // 128), dtype=torch.float32, device="cuda:0"
    )

    segment_offsets = torch.arange(
        0, (batch_size + 1) * m, m, device="cuda:0", dtype=torch.int32
    )

    flops = 2 * batch_size * m * n * k

    # --- Benchmark CuTeDSL kernel ---
    out_cutedsl = torch.empty(total_M, n, device="cuda:0", dtype=out_dtype)
    measurements_cutedsl = bench_gpu_time(
        lambda: moe_grouped_gemm_fp8(
            a, b, a_scale, b_scale, segment_offsets,
            out_dtype=out_dtype, out=out_cutedsl,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms_cutedsl = np.median(measurements_cutedsl)
    tflops_cutedsl = flops * 1e-9 / ms_cutedsl

    # --- Benchmark trtllm cubin kernel (group_gemm_fp8_nt_groupwise) ---
    # Requires m to be multiple of 4
    tflops_trtllm = float("nan")
    if m % 4 == 0:
        out_trtllm = torch.empty(total_M, n, device="cuda:0", dtype=out_dtype)
        try:
            measurements_trtllm = bench_gpu_time(
                lambda: group_gemm_fp8_nt_groupwise(
                    a, b, a_scale, b_scale, segment_offsets, out=out_trtllm, mma_sm=2
                ),
                dry_run_time_ms=100,
                repeat_time_ms=1000,
            )
            ms_trtllm = np.median(measurements_trtllm)
            tflops_trtllm = flops * 1e-9 / ms_trtllm
        except Exception as e:
            tflops_trtllm = float("nan")

    ratio = tflops_cutedsl / tflops_trtllm if not np.isnan(tflops_trtllm) else float("nan")
    ratio_str = f"{ratio:.2f}x" if not np.isnan(ratio) else "N/A"

    print(
        f"CuTeDSL={tflops_cutedsl:.1f}  trtllm={tflops_trtllm:.1f}  ratio={ratio_str}",
        flush=True,
    )

    return {
        "E": batch_size, "m": m, "N": n, "K": k,
        "cutedsl": tflops_cutedsl, "trtllm": tflops_trtllm, "ratio": ratio,
    }


if __name__ == "__main__":
    batch_sizes = [1, 4, 8]
    ms = [64, 128, 512]
    ns = [2048, 4096, 7168]
    ks = [2048, 4096, 7168]

    combos = list(itertools.product(batch_sizes, ms, ns, ks))
    total = len(combos)

    print(f"Running {total} configurations: CuTeDSL vs trtllm cubin\n", flush=True)
    print(
        f"{'E':>3} {'m':>5} {'N':>5} {'K':>5} | "
        f"{'CuTeDSL':>10} {'trtllm':>10} {'ratio':>8}",
        flush=True,
    )
    print("-" * 60, flush=True)

    results = []
    for idx, (batch_size, m, n, k) in enumerate(combos, 1):
        r = bench_groupwise_grouped_gemm_fp8_blackwell(
            batch_size, m, n, k, torch.float8_e4m3fn, torch.bfloat16, idx, total
        )
        results.append(r)

    # Summary
    valid = [r for r in results if not np.isnan(r["ratio"])]
    if valid:
        ratios = [r["ratio"] for r in valid]
        print(f"\n--- Summary ({len(valid)} configs) ---", flush=True)
        print(f"  CuTeDSL / trtllm ratio: min={min(ratios):.2f}x  median={np.median(ratios):.2f}x  max={max(ratios):.2f}x", flush=True)
