"""Tests for CuTeDSL DeepSeek FP8 MoE pipeline v3.2 (FP8 output epilogue).

Validates that v3.2 produces the same output as v3.1 (v3).
"""

import pytest
import torch


def _check_sm100():
    """Check if SM100+ is available."""
    if not torch.cuda.is_available():
        return False
    try:
        from flashinfer.utils import get_compute_capability

        cc = get_compute_capability(torch.device("cuda"))
        return cc[0] >= 10
    except Exception:
        return False


def _check_pipeline_available():
    """Check if all pipeline components are available."""
    try:
        from flashinfer.cute_dsl.moe_pipeline_v3 import cutedsl_fp8_moe_v3
        from flashinfer.cute_dsl.moe_pipeline_v3_2 import cutedsl_fp8_moe_v3_2

        return True
    except Exception:
        return False


def _make_test_inputs(seq_len, device="cuda"):
    """Generate test inputs for the MoE pipeline."""
    torch.manual_seed(42)

    H = 7168
    I = 2048
    E_GLOBAL = 256
    E_LOCAL = 32
    BLOCK = 128

    # Bias routing toward local experts
    routing_logits = torch.randn(seq_len, E_GLOBAL, dtype=torch.float32, device=device)
    routing_logits[:, :E_LOCAL] += 10.0
    routing_logits[:, E_LOCAL:] -= 10.0
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


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
@pytest.mark.parametrize("seq_len", [1, 4, 16, 64, 256, 1024])
def test_v3_2_vs_v3(seq_len):
    """Test CuTeDSL v3.2 pipeline output matches v3."""
    from flashinfer.cute_dsl.moe_pipeline_v3 import cutedsl_fp8_moe_v3
    from flashinfer.cute_dsl.moe_pipeline_v3_2 import cutedsl_fp8_moe_v3_2

    (
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, common_kwargs,
    ) = _make_test_inputs(seq_len)

    # Run v3 (baseline)
    v3_out = cutedsl_fp8_moe_v3(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, **common_kwargs,
    )

    # Run v3.2
    v3_2_out = cutedsl_fp8_moe_v3_2(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, **common_kwargs,
    )

    # Compare
    v3_f32 = v3_out.float()
    v3_2_f32 = v3_2_out.float()

    cos_sim = torch.nn.functional.cosine_similarity(
        v3_f32.flatten(), v3_2_f32.flatten(), dim=0
    ).item()

    abs_diff = (v3_f32 - v3_2_f32).abs()

    print(f"\n[v3.2 vs v3] seq_len={seq_len}")
    print(f"  v3 max: {v3_f32.abs().max().item():.6e}")
    print(f"  v3.2 max: {v3_2_f32.abs().max().item():.6e}")
    print(f"  Cosine sim: {cos_sim:.6f} (need >= 0.999)")
    print(f"  Max abs diff: {abs_diff.max().item():.6e}")
    print(f"  Mean abs diff: {abs_diff.mean().item():.6e}")

    assert v3_f32.abs().max().item() > 0, "v3 output is all zeros"
    assert v3_2_f32.abs().max().item() > 0, "v3.2 output is all zeros"
    # v3.2 uses rcp_approx_ftz for scale (vs exact division in v3),
    # so slightly lower threshold than v3-vs-v1 comparison
    assert cos_sim >= 0.999, f"Cosine similarity {cos_sim:.6f} < 0.999"


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
def test_v3_2_small_batch():
    """Test v3.2 pipeline with very small batch (edge case)."""
    from flashinfer.cute_dsl.moe_pipeline_v3_2 import cutedsl_fp8_moe_v3_2

    (
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, common_kwargs,
    ) = _make_test_inputs(1)

    out = cutedsl_fp8_moe_v3_2(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, **common_kwargs,
    )

    H = 7168
    assert out.shape == (1, H)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
    assert out.abs().max().item() > 0, "Output is all zeros"


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
def test_v3_2_output_preallocated():
    """Test v3.2 pipeline with pre-allocated output tensor (destination passing)."""
    from flashinfer.cute_dsl.moe_pipeline_v3 import cutedsl_fp8_moe_v3
    from flashinfer.cute_dsl.moe_pipeline_v3_2 import cutedsl_fp8_moe_v3_2

    seq_len = 32
    H = 7168
    device = "cuda"

    (
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, common_kwargs,
    ) = _make_test_inputs(seq_len)

    # Pre-allocate output
    output = torch.zeros(seq_len, H, dtype=torch.bfloat16, device=device)

    v3_2_out = cutedsl_fp8_moe_v3_2(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, **common_kwargs, output=output,
    )

    # Verify output was written in-place
    assert v3_2_out.data_ptr() == output.data_ptr(), "Output not written in-place"
    assert v3_2_out.abs().max().item() > 0, "Output is all zeros"
    assert not torch.isnan(v3_2_out).any(), "Output contains NaN"

    # Verify correctness against v3
    v3_out = cutedsl_fp8_moe_v3(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, **common_kwargs,
    )
    cos_sim = torch.nn.functional.cosine_similarity(
        v3_out.float().flatten(), v3_2_out.float().flatten(), dim=0
    ).item()
    print(f"\n[v3.2 dest-passing vs v3] Cosine sim: {cos_sim:.6f}")
    assert cos_sim >= 0.999, f"Cosine similarity {cos_sim:.6f} < 0.999"


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
def test_v3_2_gemm1_fp8_output_correctness():
    """Test GEMM1 FP8 output directly: v3.2 vs v3 at the GEMM1 level.

    Compares the dequantized GEMM1 outputs to verify the FP8 quantization
    in the epilogue matches the separate quantization kernel.
    """
    from flashinfer.cute_dsl.moe_grouped_gemm_fp8_v3 import (
        compute_aligned_m_indptr,
        moe_gemm1_fp8_v3,
        scatter_to_flat_padded,
    )
    from flashinfer.cute_dsl.moe_grouped_gemm_fp8_v3_2 import moe_gemm1_fp8_v3_2
    from flashinfer.cute_dsl.moe_routing import moe_routing_sglang

    seq_len = 64
    device = "cuda"
    torch.manual_seed(42)

    H = 7168
    I = 2048
    E_GLOBAL = 256
    E_LOCAL = 32
    BLOCK = 128

    routing_logits = torch.randn(seq_len, E_GLOBAL, dtype=torch.float32, device=device)
    routing_logits[:, :E_LOCAL] += 10.0
    routing_logits[:, E_LOCAL:] -= 10.0
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

    routing_result = moe_routing_sglang(
        routing_logits, routing_bias,
        num_local_experts=E_LOCAL, local_expert_offset=0,
        n_group=8, topk_group=4, top_k=8,
        routed_scaling_factor=2.5,
        intermediate_size=I, hidden_size=H,
    )

    # Run GEMM1 with v3 (BF16 → separate quant)
    v3_out, v3_scale, _, _, _, _ = moe_gemm1_fp8_v3(
        hidden_states, g1w, hs_scale, g1ws,
        routing_result.m_indptr,
        routing_result.permuted_idx_to_token_idx,
    )

    # Run GEMM1 with v3.2 (direct FP8 output)
    v3_2_out, v3_2_scale, _, _, _, _ = moe_gemm1_fp8_v3_2(
        hidden_states, g1w, hs_scale, g1ws,
        routing_result.m_indptr,
        routing_result.permuted_idx_to_token_idx,
    )

    # Dequantize both to compare
    total_M = v3_out.shape[0]
    two_I = v3_out.shape[1]

    v3_deq = v3_out.to(torch.float32) * v3_scale.t().repeat_interleave(BLOCK, dim=1)[:total_M]
    v3_2_deq = v3_2_out.to(torch.float32) * v3_2_scale.t().repeat_interleave(BLOCK, dim=1)[:total_M]

    cos_sim = torch.nn.functional.cosine_similarity(
        v3_deq.flatten(), v3_2_deq.flatten(), dim=0
    ).item()

    print(f"\n[GEMM1 v3.2 vs v3] total_M={total_M}")
    print(f"  v3 deq max: {v3_deq.abs().max().item():.6e}")
    print(f"  v3.2 deq max: {v3_2_deq.abs().max().item():.6e}")
    print(f"  Cosine sim: {cos_sim:.6f}")

    assert v3_deq.abs().max().item() > 0, "v3 GEMM1 output is all zeros"
    assert v3_2_deq.abs().max().item() > 0, "v3.2 GEMM1 output is all zeros"
    assert cos_sim >= 0.999, f"GEMM1 cosine similarity {cos_sim:.6f} < 0.999"


if __name__ == "__main__":
    if _check_sm100() and _check_pipeline_available():
        test_v3_2_vs_v3(16)
        print("test_v3_2_vs_v3 passed")
        test_v3_2_small_batch()
        print("test_v3_2_small_batch passed")
        test_v3_2_output_preallocated()
        print("test_v3_2_output_preallocated passed")
        test_v3_2_gemm1_fp8_output_correctness()
        print("test_v3_2_gemm1_fp8_output_correctness passed")
    else:
        print("Skipping: SM100+ or pipeline components not available")
