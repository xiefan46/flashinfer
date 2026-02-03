"""Benchmark CuTeDSL FP8 MoE pipeline against trtllm and triton baselines.

Usage:
    python benchmarks/bench_cute_dsl_moe_fp8.py [--seq_len 256] [--intermediate_size 2048]
"""

import argparse

import torch

from flashinfer.cute_dsl.moe_pipeline import cutedsl_fp8_moe
from flashinfer.testing.utils import bench_gpu_time


def _generate_inputs(seq_len, H=7168, I=2048, E_global=256, E_local=32, device="cuda"):
    """Generate benchmark inputs."""
    import pathlib

    # Ensure tests/ is importable as a package so relative imports inside work
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    import sys
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from tests.moe.test_dpsk_fused_moe_fp8 import generate_random_inputs_moe

    inputs = generate_random_inputs_moe(
        seq_len,
        num_experts_global=E_global,
        num_local_experts=E_local,
        hidden_size=H,
        intermediate_size=I,
        device=device,
    )

    # Force tokens to route to local experts
    inputs["routing_logits"][:, :E_local] += 10.0
    inputs["routing_logits"][:, E_local:] -= 10.0

    return inputs


def bench_pipeline(seq_len, intermediate_size=2048, num_iters=20, warmup_iters=5):
    """Benchmark full CuTeDSL MoE pipeline."""
    device = "cuda"
    torch.manual_seed(42)

    H = 7168
    I = intermediate_size
    E_GLOBAL = 256
    E_LOCAL = 32

    inputs = _generate_inputs(seq_len, H, I, E_GLOBAL, E_LOCAL, device)

    def run_cutedsl():
        return cutedsl_fp8_moe(
            inputs["routing_logits"],
            inputs["routing_bias"],
            inputs["hidden_states"],
            inputs["hidden_states_scale"],
            inputs["gemm1_weights"],
            inputs["gemm1_weights_scale"].to(torch.float32),
            inputs["gemm2_weights"],
            inputs["gemm2_weights_scale"].to(torch.float32),
            num_experts_global=E_GLOBAL,
            num_local_experts=E_LOCAL,
            local_expert_offset=0,
            top_k=8,
            intermediate_size=I,
        )

    # Warmup
    for _ in range(warmup_iters):
        run_cutedsl()
    torch.cuda.synchronize()

    # Benchmark
    try:
        times = bench_gpu_time(
            run_cutedsl,
            enable_cupti=True,
            cold_l2_cache=True,
        )
        median_us = sorted(times)[len(times) // 2] * 1e6
        print(f"CuTeDSL MoE (T={seq_len}, I={I}): {median_us:.1f} us (median, CUPTI)")
    except Exception as e:
        print(f"CUPTI benchmark failed ({e}), falling back to events")
        # Manual benchmark with CUDA events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(num_iters):
            start.record()
            run_cutedsl()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        median_ms = sorted(times)[len(times) // 2]
        print(f"CuTeDSL MoE (T={seq_len}, I={I}): {median_ms*1e3:.1f} us (median, events)")

    return times


def bench_per_stage(seq_len, intermediate_size=2048):
    """Benchmark each pipeline stage individually."""
    device = "cuda"
    torch.manual_seed(42)

    H = 7168
    I = intermediate_size
    E_LOCAL = 32

    inputs = _generate_inputs(seq_len, H, I, 256, E_LOCAL, device)

    print(f"\n--- Per-Stage Breakdown (T={seq_len}, I={I}) ---")

    # Stage 1: Routing
    from flashinfer.cute_dsl.moe_routing import moe_routing_deepseek

    def run_routing():
        return moe_routing_deepseek(
            inputs["routing_logits"],
            inputs["routing_bias"],
            num_local_experts=E_LOCAL,
            pad_to=4,
        )

    for _ in range(3):
        routing_result = run_routing()
    torch.cuda.synchronize()

    try:
        times = bench_gpu_time(run_routing, enable_cupti=True)
        median = sorted(times)[len(times) // 2] * 1e6
        print(f"  Routing: {median:.1f} us")
    except Exception:
        print("  Routing: (CUPTI failed)")

    max_padded = routing_result.max_padded_tokens
    if max_padded == 0:
        print("  No tokens routed to local experts, skipping remaining stages")
        return

    # Stage 2: GEMM1
    from flashinfer.cute_dsl.moe_grouped_gemm_fp8 import moe_gemm1_fp8

    def run_gemm1():
        return moe_gemm1_fp8(
            inputs["hidden_states"],
            inputs["gemm1_weights"],
            inputs["hidden_states_scale"],
            inputs["gemm1_weights_scale"].to(torch.float32),
            routing_result.m_indptr,
            routing_result.permuted_idx_to_token_idx,
        )

    for _ in range(3):
        gemm1_out, gemm1_scale = run_gemm1()
    torch.cuda.synchronize()

    try:
        times = bench_gpu_time(run_gemm1, enable_cupti=True)
        median = sorted(times)[len(times) // 2] * 1e6
        print(f"  GEMM1: {median:.1f} us")
    except Exception:
        print("  GEMM1: (CUPTI failed)")

    # Stage 3: SwiGLU + Requant
    from flashinfer.cute_dsl.moe_activation import moe_swiglu_fp8_requant

    def run_activation():
        return moe_swiglu_fp8_requant(gemm1_out, gemm1_scale)

    for _ in range(3):
        act_out, act_scale = run_activation()
    torch.cuda.synchronize()

    try:
        times = bench_gpu_time(run_activation, enable_cupti=True)
        median = sorted(times)[len(times) // 2] * 1e6
        print(f"  SwiGLU+Requant: {median:.1f} us")
    except Exception:
        print("  SwiGLU+Requant: (CUPTI failed)")

    # Stage 4: GEMM2
    from flashinfer.cute_dsl.moe_grouped_gemm_fp8 import moe_gemm2_fp8

    def run_gemm2():
        return moe_gemm2_fp8(
            act_out,
            inputs["gemm2_weights"],
            act_scale,
            inputs["gemm2_weights_scale"].to(torch.float32),
            routing_result.m_indptr,
        )

    for _ in range(3):
        gemm2_out = run_gemm2()
    torch.cuda.synchronize()

    try:
        times = bench_gpu_time(run_gemm2, enable_cupti=True)
        median = sorted(times)[len(times) // 2] * 1e6
        print(f"  GEMM2: {median:.1f} us")
    except Exception:
        print("  GEMM2: (CUPTI failed)")

    # Stage 5: Finalize
    from flashinfer.cute_dsl.moe_finalize import moe_finalize

    def run_finalize():
        return moe_finalize(
            gemm2_out,
            routing_result.topk_values,
            routing_result.topk_indices,
            routing_result.expanded_idx_to_permuted_idx,
            E_LOCAL, 0, H,
        )

    for _ in range(3):
        run_finalize()
    torch.cuda.synchronize()

    try:
        times = bench_gpu_time(run_finalize, enable_cupti=True)
        median = sorted(times)[len(times) // 2] * 1e6
        print(f"  Finalize: {median:.1f} us")
    except Exception:
        print("  Finalize: (CUPTI failed)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark CuTeDSL FP8 MoE")
    parser.add_argument("--seq_len", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024])
    parser.add_argument("--intermediate_size", type=int, default=2048)
    parser.add_argument("--per_stage", action="store_true", help="Show per-stage breakdown")
    args = parser.parse_args()

    for seq_len in args.seq_len:
        bench_pipeline(seq_len, args.intermediate_size)
        if args.per_stage:
            bench_per_stage(seq_len, args.intermediate_size)


if __name__ == "__main__":
    main()
