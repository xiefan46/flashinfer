"""Performance benchmark for CuTeDSL MoE v3 vs v1/v2.

Measures end-to-end pipeline latency and per-stage timing.
Run on B200 GPU with:
    python tests/moe/bench_cute_dsl_moe_v3.py [--seq_lens 1,4,16,64,256,1024]

AI-assisted implementation (Claude).
"""

import argparse
import sys
import time

import torch


def _check_sm100():
    if not torch.cuda.is_available():
        return False
    try:
        from flashinfer.utils import get_compute_capability

        cc = get_compute_capability(torch.device("cuda"))
        return cc[0] >= 10
    except Exception:
        return False


def _make_inputs(seq_len, device="cuda"):
    torch.manual_seed(42)

    H = 7168
    I = 2048
    E_GLOBAL = 256
    E_LOCAL = 32
    BLOCK = 128

    routing_logits = torch.randn(seq_len, E_GLOBAL, dtype=torch.float32, device=device)
    routing_logits[:, :E_LOCAL] += 20.0
    routing_logits[:, E_LOCAL:] -= 20.0
    routing_bias = torch.randn(E_GLOBAL, dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn(seq_len, H, device=device).to(torch.float8_e4m3fn)
    hs_scale = (
        torch.randn(H // BLOCK, seq_len, dtype=torch.float32, device=device).abs()
        + 0.01
    )
    g1w = torch.randn(E_LOCAL, 2 * I, H, device=device).to(torch.float8_e4m3fn)
    g1ws = (
        torch.randn(
            E_LOCAL, 2 * I // BLOCK, H // BLOCK, dtype=torch.float32, device=device
        ).abs()
        + 0.01
    )
    g2w = torch.randn(E_LOCAL, H, I, device=device).to(torch.float8_e4m3fn)
    g2ws = (
        torch.randn(
            E_LOCAL, H // BLOCK, I // BLOCK, dtype=torch.float32, device=device
        ).abs()
        + 0.01
    )

    common_kwargs = dict(
        num_experts_global=E_GLOBAL,
        num_local_experts=E_LOCAL,
        local_expert_offset=0,
        top_k=8,
        n_group=8,
        topk_group=4,
        intermediate_size=I,
        routed_scaling_factor=2.5,
    )

    return (
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, common_kwargs,
    )


def bench_cuda_events(fn, warmup=5, iters=20):
    """Benchmark using CUDA events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    # Use median of middle 50%
    q1 = len(times) // 4
    q3 = 3 * len(times) // 4
    trimmed = times[q1:q3] if q3 > q1 else times
    median = times[len(times) // 2]
    mean_trimmed = sum(trimmed) / len(trimmed)

    return median, mean_trimmed, min(times), max(times)


def bench_cupti(fn, warmup=5, iters=20):
    """Benchmark using CUPTI if available, fallback to CUDA events."""
    try:
        from flashinfer.testing import bench_gpu_time

        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        median_ms, std_ms = bench_gpu_time(
            fn, args=(), enable_cupti=True, num_iters=iters
        )
        return median_ms, median_ms, median_ms - std_ms, median_ms + std_ms
    except Exception:
        return bench_cuda_events(fn, warmup=warmup, iters=iters)


def run_benchmarks(seq_lens, use_cupti=False):
    from flashinfer.cute_dsl.moe_pipeline_v2 import cutedsl_fp8_moe_v2
    from flashinfer.cute_dsl.moe_pipeline_v3 import cutedsl_fp8_moe_v3

    bench_fn = bench_cupti if use_cupti else bench_cuda_events

    print("=" * 80)
    print("CuTeDSL MoE Pipeline Benchmark: v3 (flat) vs v2 (vectorized batched)")
    print(f"  Timing: {'CUPTI' if use_cupti else 'CUDA events'}")
    print("=" * 80)

    header = f"{'seq_len':>8} | {'v2 (ms)':>10} | {'v3 (ms)':>10} | {'v3/v2':>8} | {'cos_sim':>8}"
    print(header)
    print("-" * len(header))

    results = []

    for seq_len in seq_lens:
        (
            routing_logits, routing_bias, hidden_states, hs_scale,
            g1w, g1ws, g2w, g2ws, common_kwargs,
        ) = _make_inputs(seq_len)

        def run_v2():
            return cutedsl_fp8_moe_v2(
                routing_logits, routing_bias, hidden_states, hs_scale,
                g1w, g1ws, g2w, g2ws, **common_kwargs,
            )

        def run_v3():
            return cutedsl_fp8_moe_v3(
                routing_logits, routing_bias, hidden_states, hs_scale,
                g1w, g1ws, g2w, g2ws, **common_kwargs,
            )

        # Correctness check (v3 vs v2)
        v2_out = run_v2()
        v3_out = run_v3()
        cos_sim = torch.nn.functional.cosine_similarity(
            v2_out.float().flatten(), v3_out.float().flatten(), dim=0
        ).item()

        # Benchmark v2
        v2_med, v2_mean, v2_min, v2_max = bench_fn(run_v2)

        # Benchmark v3
        v3_med, v3_mean, v3_min, v3_max = bench_fn(run_v3)

        ratio = v3_med / v2_med if v2_med > 0 else float('inf')

        row = f"{seq_len:>8} | {v2_med:>10.3f} | {v3_med:>10.3f} | {ratio:>8.3f} | {cos_sim:>8.4f}"
        print(row)

        results.append({
            "seq_len": seq_len,
            "v2_ms": v2_med,
            "v3_ms": v3_med,
            "ratio": ratio,
            "cos_sim": cos_sim,
        })

    print("-" * len(header))
    print("\nv3/v2 < 1.0 means v3 is faster")
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark CuTeDSL MoE v3")
    parser.add_argument(
        "--seq_lens", type=str, default="1,4,16,64,256,1024",
        help="Comma-separated list of sequence lengths",
    )
    parser.add_argument(
        "--cupti", action="store_true",
        help="Use CUPTI timing (more accurate, requires cupti-python)",
    )
    args = parser.parse_args()

    if not _check_sm100():
        print("ERROR: SM100+ GPU required. Exiting.")
        sys.exit(1)

    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",")]
    run_benchmarks(seq_lens, use_cupti=args.cupti)


if __name__ == "__main__":
    main()
