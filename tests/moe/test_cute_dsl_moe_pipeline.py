"""End-to-end tests for CuTeDSL DeepSeek FP8 MoE pipeline."""

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
        from flashinfer.fused_moe.fused_routing_dsv3 import fused_topk_deepseek
        from flashinfer.gemm import group_gemm_fp8_nt_groupwise

        return True
    except Exception:
        return False


def _check_trtllm_available():
    """Check if trtllm fused MoE kernel is available."""
    try:
        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
@pytest.mark.parametrize("seq_len", [4, 16, 64, 256])
@pytest.mark.parametrize("intermediate_size", [512, 2048])
def test_pipeline_vs_reference(seq_len, intermediate_size):
    """Test CuTeDSL MoE pipeline against Python reference."""
    from flashinfer.cute_dsl.moe_pipeline import cutedsl_fp8_moe
    from .test_dpsk_fused_moe_fp8 import (
        generate_random_inputs_moe,
        run_fp8_block_scale_moe_reference,
    )

    device = "cuda"
    torch.manual_seed(42)

    H = 7168
    I = intermediate_size
    E_GLOBAL = 256
    E_LOCAL = 32
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    local_expert_offset = 0
    routed_scaling_factor = 2.5

    # Generate inputs
    inputs = generate_random_inputs_moe(
        seq_len,
        num_experts_global=E_GLOBAL,
        num_local_experts=E_LOCAL,
        hidden_size=H,
        intermediate_size=I,
        use_bias=True,
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=routed_scaling_factor,
        device=device,
    )

    # Force tokens to route to local experts:
    # Set local expert logits high, non-local logits very low.
    # DeepSeek routing uses sigmoid + group selection, so we need to dominate
    # both the group scores and the global top-k selection.
    inputs["routing_logits"][:, local_expert_offset:local_expert_offset + E_LOCAL] += 10.0
    inputs["routing_logits"][:, local_expert_offset + E_LOCAL:] -= 10.0

    # Run reference
    ref_out = run_fp8_block_scale_moe_reference(
        routing_logits=inputs["routing_logits"],
        routing_bias=inputs["routing_bias"],
        hidden_states=inputs["hidden_states"],
        hidden_states_scale=inputs["hidden_states_scale"],
        gemm1_weights=inputs["gemm1_weights"],
        gemm1_weights_scale=inputs["gemm1_weights_scale"],
        gemm2_weights=inputs["gemm2_weights"],
        gemm2_weights_scale=inputs["gemm2_weights_scale"],
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=routed_scaling_factor,
        hidden_size=H,
        intermediate_size=I,
        num_experts_global=E_GLOBAL,
        num_local_experts=E_LOCAL,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
    )

    # Run CuTeDSL pipeline
    cute_out = cutedsl_fp8_moe(
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
        local_expert_offset=local_expert_offset,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=I,
        routed_scaling_factor=routed_scaling_factor,
    )

    # The Python reference uses exact float32 matmul while CuTeDSL uses FP8 GEMM.
    # With random inputs (K=7168), output magnitudes reach millions, making fixed
    # atol/rtol hit-ratio checks unreliable. Cosine similarity is the correct
    # metric for comparing FP8 vs float32 at arbitrary scale.
    ref_f32 = ref_out.float()
    cute_f32 = cute_out.float()

    # Cosine similarity (primary metric)
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), cute_f32.flatten(), dim=0
    ).item()

    # Normalized RMSE (informational)
    abs_diff = (ref_f32 - cute_f32).abs()
    nrmse = abs_diff.pow(2).mean().sqrt().item() / (ref_f32.abs().mean().item() + 1e-8)

    # Ensure outputs are not all zeros (i.e., tokens were actually routed)
    ref_nonzero = ref_f32.abs().max().item()
    cute_nonzero = cute_f32.abs().max().item()

    print(f"\n[vs reference] seq_len={seq_len}, I={I}")
    print(f"  Ref max: {ref_nonzero:.6e}, Cute max: {cute_nonzero:.6e}")
    print(f"  Cosine sim: {cos_sim:.6f} (need >= 0.98)")
    print(f"  Normalized RMSE: {nrmse:.6f}")
    print(f"  Max abs diff: {abs_diff.max():.6e}")
    print(f"  Mean abs diff: {abs_diff.mean():.6e}")

    assert ref_nonzero > 0, "Reference output is all zeros — no tokens routed to local experts"
    assert cute_nonzero > 0, "CuTeDSL output is all zeros — no tokens routed to local experts"
    assert cos_sim >= 0.98, (
        f"Cosine similarity {cos_sim:.6f} < 0.98"
    )


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
def test_pipeline_small_batch():
    """Test pipeline with very small batch (edge case)."""
    from flashinfer.cute_dsl.moe_pipeline import cutedsl_fp8_moe
    from .test_dpsk_fused_moe_fp8 import generate_random_inputs_moe

    device = "cuda"
    torch.manual_seed(42)

    inputs = generate_random_inputs_moe(
        1,
        num_experts_global=256,
        num_local_experts=32,
        hidden_size=7168,
        intermediate_size=2048,
        device=device,
    )

    # Force token to route to local experts
    inputs["routing_logits"][:, :32] += 10.0
    inputs["routing_logits"][:, 32:] -= 10.0

    out = cutedsl_fp8_moe(
        inputs["routing_logits"],
        inputs["routing_bias"],
        inputs["hidden_states"],
        inputs["hidden_states_scale"],
        inputs["gemm1_weights"],
        inputs["gemm1_weights_scale"].to(torch.float32),
        inputs["gemm2_weights"],
        inputs["gemm2_weights_scale"].to(torch.float32),
        num_experts_global=256,
        num_local_experts=32,
        intermediate_size=2048,
    )

    assert out.shape == (1, 7168)
    assert out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
    assert out.abs().max().item() > 0, "Output is all zeros — no tokens routed"


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.skipif(not _check_pipeline_available(), reason="Pipeline not available")
@pytest.mark.skipif(not _check_trtllm_available(), reason="trtllm MoE not available")
@pytest.mark.parametrize("seq_len", [4, 16, 64, 256])
@pytest.mark.parametrize("intermediate_size", [512, 2048])
def test_pipeline_vs_trtllm(seq_len, intermediate_size):
    """Test CuTeDSL MoE pipeline against trtllm CUTLASS kernel."""
    from flashinfer.cute_dsl.moe_pipeline import cutedsl_fp8_moe
    from flashinfer.fused_moe import trtllm_fp8_block_scale_moe, WeightLayout
    from flashinfer.autotuner import autotune
    from .test_dpsk_fused_moe_fp8 import generate_random_inputs_moe

    device = "cuda"
    torch.manual_seed(42)

    H = 7168
    I = intermediate_size
    E_GLOBAL = 256
    E_LOCAL = 32
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    local_expert_offset = 0
    routed_scaling_factor = 2.5

    # Generate inputs
    inputs = generate_random_inputs_moe(
        seq_len,
        num_experts_global=E_GLOBAL,
        num_local_experts=E_LOCAL,
        hidden_size=H,
        intermediate_size=I,
        use_bias=True,
        local_expert_offset=local_expert_offset,
        routed_scaling_factor=routed_scaling_factor,
        device=device,
    )

    # Force tokens to route to local experts
    inputs["routing_logits"][:, local_expert_offset:local_expert_offset + E_LOCAL] += 10.0
    inputs["routing_logits"][:, local_expert_offset + E_LOCAL:] -= 10.0

    # Run trtllm kernel (autotune off to avoid slow first-run tuning)
    with autotune(False):
        trtllm_out = trtllm_fp8_block_scale_moe(
            inputs["routing_logits"].to(torch.float32),
            inputs["routing_bias"],
            inputs["hidden_states"],
            inputs["hidden_states_scale"],
            inputs["gemm1_weights"],
            inputs["gemm1_weights_scale"].to(torch.float32),
            inputs["gemm2_weights"],
            inputs["gemm2_weights_scale"].to(torch.float32),
            E_GLOBAL,
            TOP_K,
            N_GROUP,
            TOPK_GROUP,
            I,
            local_expert_offset,
            E_LOCAL,
            routed_scaling_factor,
            routing_method_type=2,  # DeepSeek-styled
            use_shuffled_weight=False,
            weight_layout=WeightLayout.MajorK,
            enable_pdl=True,
            tune_max_num_tokens=4096,
        )

    # Run CuTeDSL pipeline
    cute_out = cutedsl_fp8_moe(
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
        local_expert_offset=local_expert_offset,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=I,
        routed_scaling_factor=routed_scaling_factor,
    )

    # Both are FP8 kernels with different GEMM implementations.
    # Cosine similarity is the primary metric; both should agree well since
    # they operate on the same quantized inputs.
    trtllm_f32 = trtllm_out.float()
    cute_f32 = cute_out.float()

    # Cosine similarity (primary metric)
    cos_sim = torch.nn.functional.cosine_similarity(
        trtllm_f32.flatten(), cute_f32.flatten(), dim=0
    ).item()

    # Normalized RMSE (informational)
    abs_diff = (trtllm_f32 - cute_f32).abs()
    nrmse = abs_diff.pow(2).mean().sqrt().item() / (trtllm_f32.abs().mean().item() + 1e-8)

    # Non-zero checks
    trtllm_nonzero = trtllm_f32.abs().max().item()
    cute_nonzero = cute_f32.abs().max().item()

    print(f"\n[vs trtllm] seq_len={seq_len}, I={I}")
    print(f"  trtllm max: {trtllm_nonzero:.6e}, Cute max: {cute_nonzero:.6e}")
    print(f"  Cosine sim: {cos_sim:.6f} (need >= 0.98)")
    print(f"  Normalized RMSE: {nrmse:.6f}")
    print(f"  Max abs diff: {abs_diff.max():.6e}")
    print(f"  Mean abs diff: {abs_diff.mean():.6e}")

    assert trtllm_nonzero > 0, "trtllm output is all zeros"
    assert cute_nonzero > 0, "CuTeDSL output is all zeros"
    assert cos_sim >= 0.98, (
        f"Cosine similarity {cos_sim:.6f} < 0.98"
    )


if __name__ == "__main__":
    if _check_sm100() and _check_pipeline_available():
        test_pipeline_vs_reference(16, 2048)
        print("test_pipeline_vs_reference passed")
        test_pipeline_small_batch()
        print("test_pipeline_small_batch passed")
        if _check_trtllm_available():
            test_pipeline_vs_trtllm(16, 2048)
            print("test_pipeline_vs_trtllm passed")
        else:
            print("Skipping trtllm test: not available")
    else:
        print("Skipping: SM100+ or pipeline components not available")
