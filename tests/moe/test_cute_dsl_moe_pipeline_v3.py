"""Tests for CuTeDSL DeepSeek FP8 MoE pipeline v3 (flat layout).

Validates that v3 produces the same output as v1.
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
        from flashinfer.cute_dsl.moe_pipeline import cutedsl_fp8_moe
        from flashinfer.cute_dsl.moe_pipeline_v3 import cutedsl_fp8_moe_v3

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
def test_v3_vs_v1(seq_len):
    """Test CuTeDSL v3 pipeline output matches v1."""
    from flashinfer.cute_dsl.moe_pipeline import cutedsl_fp8_moe
    from flashinfer.cute_dsl.moe_pipeline_v3 import cutedsl_fp8_moe_v3

    (
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, common_kwargs,
    ) = _make_test_inputs(seq_len)

    # Run v1
    v1_out = cutedsl_fp8_moe(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, **common_kwargs,
    )

    # Run v3
    v3_out = cutedsl_fp8_moe_v3(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, **common_kwargs,
    )

    # Compare
    v1_f32 = v1_out.float()
    v3_f32 = v3_out.float()

    cos_sim = torch.nn.functional.cosine_similarity(
        v1_f32.flatten(), v3_f32.flatten(), dim=0
    ).item()

    abs_diff = (v1_f32 - v3_f32).abs()

    print(f"\n[v3 vs v1] seq_len={seq_len}")
    print(f"  v1 max: {v1_f32.abs().max().item():.6e}")
    print(f"  v3 max: {v3_f32.abs().max().item():.6e}")
    print(f"  Cosine sim: {cos_sim:.6f} (need >= 0.99)")
    print(f"  Max abs diff: {abs_diff.max().item():.6e}")
    print(f"  Mean abs diff: {abs_diff.mean().item():.6e}")

    assert v1_f32.abs().max().item() > 0, "v1 output is all zeros"
    assert v3_f32.abs().max().item() > 0, "v3 output is all zeros"
    assert cos_sim >= 0.99, f"Cosine similarity {cos_sim:.6f} < 0.99"


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
def test_v3_small_batch():
    """Test v3 pipeline with very small batch (edge case)."""
    from flashinfer.cute_dsl.moe_pipeline_v3 import cutedsl_fp8_moe_v3

    (
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, common_kwargs,
    ) = _make_test_inputs(1)

    out = cutedsl_fp8_moe_v3(
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
def test_v3_output_preallocated():
    """Test v3 pipeline with pre-allocated output tensor (destination passing)."""
    from flashinfer.cute_dsl.moe_pipeline import cutedsl_fp8_moe
    from flashinfer.cute_dsl.moe_pipeline_v3 import cutedsl_fp8_moe_v3

    seq_len = 32
    H = 7168
    device = "cuda"

    (
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, common_kwargs,
    ) = _make_test_inputs(seq_len)

    # Pre-allocate output
    output = torch.zeros(seq_len, H, dtype=torch.bfloat16, device=device)

    v3_out = cutedsl_fp8_moe_v3(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, **common_kwargs, output=output,
    )

    # Verify output was written in-place
    assert v3_out.data_ptr() == output.data_ptr(), "Output not written in-place"
    assert v3_out.abs().max().item() > 0, "Output is all zeros"
    assert not torch.isnan(v3_out).any(), "Output contains NaN"

    # Verify correctness against v1
    v1_out = cutedsl_fp8_moe(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, **common_kwargs,
    )
    cos_sim = torch.nn.functional.cosine_similarity(
        v1_out.float().flatten(), v3_out.float().flatten(), dim=0
    ).item()
    print(f"\n[v3 dest-passing vs v1] Cosine sim: {cos_sim:.6f}")
    assert cos_sim >= 0.99, f"Cosine similarity {cos_sim:.6f} < 0.99"


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
def test_v3_sparse_routing():
    """Test v3 with sparse routing: only a few experts active (strong bias).

    This tests the key v3 advantage: memory savings for skewed routing.
    With very strong routing bias, only ~8 of 32 local experts get tokens,
    so v3's flat layout should use much less memory than v2's batched layout.
    """
    from flashinfer.cute_dsl.moe_pipeline import cutedsl_fp8_moe
    from flashinfer.cute_dsl.moe_pipeline_v3 import cutedsl_fp8_moe_v3

    seq_len = 64
    device = "cuda"
    H = 7168
    I = 2048
    E_GLOBAL = 256
    E_LOCAL = 32
    BLOCK = 128

    torch.manual_seed(123)

    # Very strong bias: only first 8 local experts get tokens
    routing_logits = torch.randn(seq_len, E_GLOBAL, dtype=torch.float32, device=device)
    routing_logits[:, :8] += 20.0     # Strong bias to first 8 experts
    routing_logits[:, 8:] -= 20.0     # Suppress remaining experts
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

    v1_out = cutedsl_fp8_moe(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, **common_kwargs,
    )
    v3_out = cutedsl_fp8_moe_v3(
        routing_logits, routing_bias, hidden_states, hs_scale,
        g1w, g1ws, g2w, g2ws, **common_kwargs,
    )

    v1_f32 = v1_out.float()
    v3_f32 = v3_out.float()
    cos_sim = torch.nn.functional.cosine_similarity(
        v1_f32.flatten(), v3_f32.flatten(), dim=0
    ).item()

    print(f"\n[v3 sparse routing vs v1] seq_len={seq_len}")
    print(f"  Cosine sim: {cos_sim:.6f}")
    assert v3_f32.abs().max().item() > 0, "v3 output is all zeros"
    assert cos_sim >= 0.99, f"Cosine similarity {cos_sim:.6f} < 0.99"


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
def test_v3_memory_savings():
    """Verify v3 uses less memory than v2 for skewed routing."""
    from flashinfer.cute_dsl.moe_grouped_gemm_fp8_v3 import compute_aligned_m_indptr

    device = "cuda"
    E = 32

    # Simulate skewed routing: 8 experts active with 128 tokens, rest 0
    masked_m = torch.zeros(E, dtype=torch.int32, device=device)
    masked_m[:8] = 128

    m_indptr = torch.zeros(E + 1, dtype=torch.int32, device=device)
    m_indptr[1:] = torch.cumsum(masked_m, dim=0)

    _, _, _, _, total_padded_M = compute_aligned_m_indptr(m_indptr)

    # v2 would pad all experts to max_M = 128: 32 * 128 = 4096
    v2_total = E * 128  # all experts padded to max_M

    # v3 only pads active experts: 8 * 128 = 1024
    print(f"\n[Memory] Skewed routing (8/32 active):")
    print(f"  v2 total rows: {v2_total}")
    print(f"  v3 total rows: {total_padded_M}")
    print(f"  Savings: {100 * (1 - total_padded_M / v2_total):.1f}%")

    assert total_padded_M == 8 * 128, f"Expected {8*128}, got {total_padded_M}"
    assert total_padded_M < v2_total, "v3 should use less memory than v2"


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
def test_v3_alignment_edge_cases():
    """Test compute_aligned_m_indptr with various alignment patterns."""
    from flashinfer.cute_dsl.moe_grouped_gemm_fp8_v3 import compute_aligned_m_indptr

    device = "cuda"

    # Case 1: Mixed alignment - some experts need padding, some don't
    m_indptr = torch.tensor([0, 1, 129, 256, 256], dtype=torch.int32, device=device)
    masked_m, aligned_m, m_indptr_aligned, m_indptr_tiles, total = (
        compute_aligned_m_indptr(m_indptr)
    )
    assert masked_m.tolist() == [1, 128, 127, 0]
    assert aligned_m.tolist() == [128, 128, 128, 0]
    assert m_indptr_aligned.tolist() == [0, 128, 256, 384, 384]
    assert m_indptr_tiles.tolist() == [0, 1, 2, 3, 3]
    assert total == 384

    # Case 2: All experts empty
    m_indptr = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=device)
    _, _, _, _, total = compute_aligned_m_indptr(m_indptr)
    assert total == 0

    # Case 3: Single expert with exactly 128 tokens
    m_indptr = torch.tensor([0, 128], dtype=torch.int32, device=device)
    _, aligned_m, _, _, total = compute_aligned_m_indptr(m_indptr)
    assert aligned_m.tolist() == [128]
    assert total == 128

    # Case 4: Single expert with 1 token (minimum padding)
    m_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    _, aligned_m, _, _, total = compute_aligned_m_indptr(m_indptr)
    assert aligned_m.tolist() == [128]
    assert total == 128

    print("\n[Alignment] All edge cases passed")


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
def test_v3_scatter_gather_roundtrip():
    """Test scatter_to_flat_padded and gather_from_flat_padded are inverses."""
    from flashinfer.cute_dsl.moe_grouped_gemm_fp8_v3 import (
        compute_aligned_m_indptr,
        scatter_to_flat_padded,
        gather_from_flat_padded,
    )

    device = "cuda"
    K = 256  # smaller K for test speed
    E = 4

    # Varying tokens per expert
    m_indptr = torch.tensor([0, 10, 10, 30, 55], dtype=torch.int32, device=device)
    max_padded = int(m_indptr[-1].item())  # 55
    masked_m, aligned_m, m_indptr_aligned, m_indptr_tiles, total_padded_M = (
        compute_aligned_m_indptr(m_indptr)
    )

    # Create test data
    torch.manual_seed(42)
    a_grouped = torch.randn(max_padded, K, dtype=torch.float32, device=device)
    a_scale_grouped = torch.randn(K // 128, max_padded, dtype=torch.float32, device=device)

    # Scatter
    a_flat, a_scale_flat, dst_row = scatter_to_flat_padded(
        a_grouped, a_scale_grouped, m_indptr, m_indptr_aligned, total_padded_M, K
    )

    # Verify flat shape
    assert a_flat.shape == (total_padded_M, K)
    assert a_scale_flat.shape == (K // 128, total_padded_M)

    # Gather back
    a_roundtrip = gather_from_flat_padded(a_flat, dst_row, max_padded)

    # Must be exact (no computation, just index operations)
    assert torch.equal(a_grouped, a_roundtrip), "Scatter/gather roundtrip failed"

    # Also verify scale roundtrip
    a_scale_roundtrip = a_scale_flat[:, dst_row]
    assert torch.equal(a_scale_grouped, a_scale_roundtrip), "Scale scatter/gather roundtrip failed"

    print(f"\n[Roundtrip] max_padded={max_padded}, total_padded_M={total_padded_M}")
    print(f"  Expert counts: {masked_m.tolist()}")
    print(f"  Aligned counts: {aligned_m.tolist()}")


if __name__ == "__main__":
    if _check_sm100() and _check_pipeline_available():
        test_v3_vs_v1(16)
        print("test_v3_vs_v1 passed")
        test_v3_small_batch()
        print("test_v3_small_batch passed")
        test_v3_output_preallocated()
        print("test_v3_output_preallocated passed")
        test_v3_sparse_routing()
        print("test_v3_sparse_routing passed")
        test_v3_memory_savings()
        print("test_v3_memory_savings passed")
        test_v3_alignment_edge_cases()
        print("test_v3_alignment_edge_cases passed")
        test_v3_scatter_gather_roundtrip()
        print("test_v3_scatter_gather_roundtrip passed")
    else:
        print("Skipping: SM100+ or pipeline components not available")
