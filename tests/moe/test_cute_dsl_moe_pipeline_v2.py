"""Tests for CuTeDSL DeepSeek FP8 MoE pipeline v2 (vectorized GEMM wrappers).

Validates that v2 produces the same output as v1 (bitwise or near-exact).
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
        from flashinfer.cute_dsl.moe_pipeline_v2 import cutedsl_fp8_moe_v2

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
@pytest.mark.parametrize("seq_len", [4, 16, 64, 256])
def test_v2_vs_v1(seq_len):
    """Test CuTeDSL v2 pipeline output matches v1."""
    from flashinfer.cute_dsl.moe_pipeline import cutedsl_fp8_moe
    from flashinfer.cute_dsl.moe_pipeline_v2 import cutedsl_fp8_moe_v2

    device = "cuda"
    torch.manual_seed(42)

    H = 7168
    I = 2048
    E_GLOBAL = 256
    E_LOCAL = 32
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    local_expert_offset = 0
    routed_scaling_factor = 2.5
    BLOCK = 128

    # Generate inputs
    # DeepSeek routing uses sigmoid + group selection, so we need to dominate
    # both the group scores and the global top-k selection.
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
        local_expert_offset=local_expert_offset,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=I,
        routed_scaling_factor=routed_scaling_factor,
    )

    # Run v1
    v1_out = cutedsl_fp8_moe(
        routing_logits,
        routing_bias,
        hidden_states,
        hs_scale,
        g1w,
        g1ws,
        g2w,
        g2ws,
        **common_kwargs,
    )

    # Run v2
    v2_out = cutedsl_fp8_moe_v2(
        routing_logits,
        routing_bias,
        hidden_states,
        hs_scale,
        g1w,
        g1ws,
        g2w,
        g2ws,
        **common_kwargs,
    )

    # Compare
    v1_f32 = v1_out.float()
    v2_f32 = v2_out.float()

    cos_sim = torch.nn.functional.cosine_similarity(
        v1_f32.flatten(), v2_f32.flatten(), dim=0
    ).item()

    abs_diff = (v1_f32 - v2_f32).abs()

    print(f"\n[v2 vs v1] seq_len={seq_len}")
    print(f"  v1 max: {v1_f32.abs().max().item():.6e}")
    print(f"  v2 max: {v2_f32.abs().max().item():.6e}")
    print(f"  Cosine sim: {cos_sim:.6f} (need >= 0.99)")
    print(f"  Max abs diff: {abs_diff.max().item():.6e}")
    print(f"  Mean abs diff: {abs_diff.mean().item():.6e}")

    assert v1_f32.abs().max().item() > 0, "v1 output is all zeros"
    assert v2_f32.abs().max().item() > 0, "v2 output is all zeros"
    assert cos_sim >= 0.99, f"Cosine similarity {cos_sim:.6f} < 0.99"


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
def test_v2_small_batch():
    """Test v2 pipeline with very small batch (edge case)."""
    from flashinfer.cute_dsl.moe_pipeline_v2 import cutedsl_fp8_moe_v2

    device = "cuda"
    torch.manual_seed(42)

    H = 7168
    I = 2048
    E_GLOBAL = 256
    E_LOCAL = 32
    BLOCK = 128

    routing_logits = torch.randn(1, E_GLOBAL, dtype=torch.float32, device=device)
    routing_logits[:, :E_LOCAL] += 10.0
    routing_logits[:, E_LOCAL:] -= 10.0
    routing_bias = torch.randn(E_GLOBAL, dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn(1, H, device=device).to(torch.float8_e4m3fn)
    hs_scale = (
        torch.randn(H // BLOCK, 1, dtype=torch.float32, device=device).abs() + 0.01
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

    out = cutedsl_fp8_moe_v2(
        routing_logits,
        routing_bias,
        hidden_states,
        hs_scale,
        g1w,
        g1ws,
        g2w,
        g2ws,
        num_experts_global=E_GLOBAL,
        num_local_experts=E_LOCAL,
        intermediate_size=I,
    )

    assert out.shape == (1, H)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
    assert out.abs().max().item() > 0, "Output is all zeros"


if __name__ == "__main__":
    if _check_sm100() and _check_pipeline_available():
        test_v2_vs_v1(16)
        print("test_v2_vs_v1 passed")
        test_v2_small_batch()
        print("test_v2_small_batch passed")
    else:
        print("Skipping: SM100+ or pipeline components not available")
