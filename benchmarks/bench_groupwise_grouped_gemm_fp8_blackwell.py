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
import sys

import numpy as np
import torch

import flashinfer
from flashinfer.cute_dsl.moe_grouped_gemm_fp8 import moe_grouped_gemm_fp8
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
    a = torch.randn(batch_size * m, k, device="cuda:0").to(in_dtype)
    b = torch.randn(batch_size, n, k, device="cuda:0").to(in_dtype)

    a_scale = torch.randn(
        (k // 128, batch_size * m), dtype=torch.float32, device="cuda:0"
    )
    # b_scale: [E, N//128, K//128] for CuTeDSL / moe_grouped_gemm_fp8
    b_scale = torch.randn(
        (batch_size, n // 128, k // 128), dtype=torch.float32, device="cuda:0"
    )

    segment_offsets = torch.arange(
        0, (batch_size + 1) * m, m, device="cuda:0", dtype=torch.int32
    )

    # --- Benchmark CuTeDSL kernel ---
    out_cutedsl = torch.empty(batch_size * m, n, device="cuda:0", dtype=out_dtype)
    measurements_cutedsl = bench_gpu_time(
        lambda: moe_grouped_gemm_fp8(
            a, b, a_scale, b_scale, segment_offsets,
            out_dtype=out_dtype, out=out_cutedsl,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
    )
    ms_cutedsl = np.median(measurements_cutedsl)
    tflops_cutedsl = 2 * batch_size * m * n * k * 1e-9 / ms_cutedsl

    print(f"CuTeDSL={tflops_cutedsl:.1f} TFLOPs/s", flush=True)


if __name__ == "__main__":
    batch_sizes = [1, 4, 8]
    ms = [64, 128, 512]
    ns = [2048, 4096, 7168]
    ks = [2048, 4096, 7168]

    combos = list(itertools.product(batch_sizes, ms, ns, ks))
    total = len(combos)

    print(f"Running {total} configurations...\n", flush=True)
    print(f"{'E':>3} {'m':>5} {'N':>5} {'K':>5} | {'CuTeDSL TFLOPs/s':>18}", flush=True)
    print("-" * 50, flush=True)

    for idx, (batch_size, m, n, k) in enumerate(combos, 1):
        bench_groupwise_grouped_gemm_fp8_blackwell(
            batch_size, m, n, k, torch.float8_e4m3fn, torch.bfloat16, idx, total
        )
