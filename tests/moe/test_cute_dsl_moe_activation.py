"""Tests for CuTeDSL MoE SwiGLU + FP8 Requantization."""

import pytest
import torch

from flashinfer.cute_dsl.moe_activation import (
    moe_swiglu_fp8_requant,
    moe_swiglu_fp8_requant_reference,
)


def _make_fp8_gemm1_output(padded, two_I, device="cuda"):
    """Generate synthetic GEMM1 output in FP8 with per-block scales."""
    BLOCK = 128
    num_blocks = two_I // BLOCK

    # Generate random float32 data, then quantize
    x_f32 = torch.randn(padded, two_I, dtype=torch.float32, device=device)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    x_blocks = x_f32.reshape(padded, num_blocks, BLOCK)
    block_amax = x_blocks.abs().amax(dim=2)  # [padded, num_blocks]
    scale = block_amax / fp8_max
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    x_scaled = x_blocks / scale.unsqueeze(2)
    x_fp8 = x_scaled.reshape(padded, two_I).to(torch.float8_e4m3fn)

    # MN-major scale: [two_I//128, padded]
    scale_mn = scale.t().contiguous()

    return x_fp8, scale_mn, x_f32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("padded", [16, 64, 256])
@pytest.mark.parametrize("intermediate_size", [256, 512, 1024, 2048])
def test_swiglu_fp8_requant_shape(padded, intermediate_size):
    """Test output shapes are correct."""
    device = "cuda"
    two_I = 2 * intermediate_size
    I = intermediate_size
    BLOCK = 128

    gemm1_out, gemm1_scale, _ = _make_fp8_gemm1_output(padded, two_I, device)

    act_out, act_scale = moe_swiglu_fp8_requant(gemm1_out, gemm1_scale)

    assert act_out.shape == (padded, I), f"act_out shape: {act_out.shape}"
    assert act_out.dtype == torch.float8_e4m3fn
    assert act_scale.shape == (I // BLOCK, padded), f"act_scale shape: {act_scale.shape}"
    assert act_scale.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("padded", [16, 64, 128])
@pytest.mark.parametrize("intermediate_size", [256, 512, 2048])
def test_swiglu_fp8_requant_accuracy(padded, intermediate_size):
    """Test SwiGLU + FP8 requant accuracy against reference."""
    device = "cuda"
    torch.manual_seed(42)
    two_I = 2 * intermediate_size
    I = intermediate_size
    BLOCK = 128

    gemm1_out, gemm1_scale, _ = _make_fp8_gemm1_output(padded, two_I, device)

    # Run kernel
    act_out, act_scale = moe_swiglu_fp8_requant(gemm1_out, gemm1_scale)

    # Dequantize output for comparison
    scale_t = act_scale.t().contiguous()  # [padded, I//128]
    scale_expanded = scale_t.repeat_interleave(BLOCK, dim=1)  # [padded, I]
    act_dequant = act_out.to(torch.float32) * scale_expanded

    # Reference: dequant input -> SwiGLU -> compare
    y_ref, _, _ = moe_swiglu_fp8_requant_reference(gemm1_out, gemm1_scale)

    # Check accuracy (accumulates dequant + activation + requant error)
    cos_sim = torch.nn.functional.cosine_similarity(
        y_ref.flatten(), act_dequant.flatten(), dim=0
    ).item()
    assert cos_sim > 0.99, f"Cosine similarity {cos_sim:.6f} too low"

    abs_diff = (y_ref - act_dequant).abs()
    assert abs_diff.mean() < 3e-2, f"Mean abs diff {abs_diff.mean():.6e} too high"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_swiglu_fp8_requant_preallocated():
    """Test with pre-allocated output buffers."""
    device = "cuda"
    torch.manual_seed(42)
    padded = 64
    I = 512
    two_I = 2 * I
    BLOCK = 128

    gemm1_out, gemm1_scale, _ = _make_fp8_gemm1_output(padded, two_I, device)

    act_out = torch.empty(padded, I, dtype=torch.float8_e4m3fn, device=device)
    act_scale = torch.empty(I // BLOCK, padded, dtype=torch.float32, device=device)

    result_out, result_scale = moe_swiglu_fp8_requant(
        gemm1_out, gemm1_scale, act_out=act_out, act_scale=act_scale
    )

    # Should be the same tensors
    assert result_out.data_ptr() == act_out.data_ptr()
    assert result_scale.data_ptr() == act_scale.data_ptr()


if __name__ == "__main__":
    test_swiglu_fp8_requant_shape(64, 2048)
    print("test_swiglu_fp8_requant_shape passed")
    test_swiglu_fp8_requant_accuracy(64, 2048)
    print("test_swiglu_fp8_requant_accuracy passed")
    test_swiglu_fp8_requant_preallocated()
    print("test_swiglu_fp8_requant_preallocated passed")
    print("All activation tests passed!")
