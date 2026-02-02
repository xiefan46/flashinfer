"""Tests for CuTeDSL FP8 Grouped GEMM for MoE."""

import pytest
import torch

from flashinfer.cute_dsl.moe_grouped_gemm_fp8 import (
    moe_grouped_gemm_fp8,
    moe_gemm1_fp8,
    moe_gemm2_fp8,
    _quantize_output_fp8,
)


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


def _fp8_block_quant_1d(x_bf16, block=128):
    """Quantize [M, N] to FP8 with per-block scales."""
    M, N = x_bf16.shape
    assert N % block == 0
    nb = N // block
    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max

    x_f32 = x_bf16.to(torch.float32)
    x_fp8 = torch.empty_like(x_bf16, dtype=torch.float8_e4m3fn)
    scales = torch.empty(M, nb, dtype=torch.float32, device=x_bf16.device)

    for j in range(nb):
        sl = slice(j * block, (j + 1) * block)
        blk = x_f32[:, sl]
        amax = torch.amax(torch.abs(blk), dim=1)
        s = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))
        q = (blk / s.unsqueeze(1)).to(torch.float8_e4m3fn)
        x_fp8[:, sl] = q
        scales[:, j] = s

    return x_fp8, scales


def _fp8_block_quant_2d(w_bf16, block=128):
    """Quantize [E, R, C] to FP8 with per-block 2D scales."""
    E, R, C = w_bf16.shape
    assert R % block == 0 and C % block == 0
    nb_r = R // block
    nb_c = C // block
    finfo = torch.finfo(torch.float8_e4m3fn)
    max_fp8 = finfo.max

    w_f32 = w_bf16.to(torch.float32)
    reshaped = w_f32.reshape(E, nb_r, block, nb_c, block)
    blocks = reshaped.permute(0, 1, 3, 2, 4).contiguous()

    amax = torch.amax(torch.abs(blocks), dim=(-1, -2))
    scales = torch.where(amax > 0, amax / max_fp8, torch.ones_like(amax))

    q_blocks = (blocks / scales.unsqueeze(-1).unsqueeze(-1)).to(torch.float8_e4m3fn)
    w_fp8 = q_blocks.permute(0, 1, 3, 2, 4).reshape(E, R, C)

    return w_fp8, scales


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.parametrize("N,K", [(4096, 7168), (7168, 2048)])
@pytest.mark.parametrize("E", [4, 8])
@pytest.mark.parametrize("m_per_expert", [16, 64, 128])
def test_grouped_gemm_fp8_bf16_output(N, K, E, m_per_expert):
    """Test grouped GEMM FP8 with BF16 output."""
    device = "cuda"
    torch.manual_seed(42)

    total_M = E * m_per_expert

    # Generate random FP8 data
    a_bf16 = torch.randn(total_M, K, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device=device)

    a_fp8, a_scales = _fp8_block_quant_1d(a_bf16)
    b_fp8, b_scales = _fp8_block_quant_2d(b_bf16)

    # MN-major scales
    a_scale_mn = a_scales.t().contiguous()  # [K//128, total_M]
    # b_scales is already [E, N//128, K//128] for MN-major

    # m_indptr (uniform distribution)
    m_indptr = torch.arange(0, (E + 1) * m_per_expert, m_per_expert, dtype=torch.int32, device=device)

    # Run grouped GEMM
    out = moe_grouped_gemm_fp8(
        a_fp8, b_fp8, a_scale_mn, b_scales, m_indptr, out_dtype=torch.bfloat16
    )

    assert out.shape == (total_M, N)
    assert out.dtype == torch.bfloat16

    # Reference: dequant(a) @ dequant(b)^T per expert
    a_dequant = a_fp8.to(torch.float32) * a_scales.repeat_interleave(128, dim=1)

    for e in range(E):
        start = e * m_per_expert
        end = (e + 1) * m_per_expert
        b_e_dequant = b_fp8[e].to(torch.float32) * b_scales[e].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        ref_e = a_dequant[start:end] @ b_e_dequant.t()
        out_e = out[start:end].to(torch.float32)

        # Tolerance for FP8 GEMM: atol=1e-1 is typical
        cos_sim = torch.nn.functional.cosine_similarity(
            ref_e.flatten(), out_e.flatten(), dim=0
        )
        assert cos_sim > 0.95, f"Expert {e}: cosine similarity {cos_sim:.4f} too low"


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
def test_quantize_output_fp8():
    """Test FP8 output quantization roundtrip."""
    device = "cuda"
    torch.manual_seed(42)

    M, N = 64, 4096
    x = torch.randn(M, N, dtype=torch.bfloat16, device=device)

    x_fp8, scale = _quantize_output_fp8(x)

    assert x_fp8.shape == (M, N)
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scale.shape == (N // 128, M)  # MN-major

    # Dequant and check roundtrip
    scale_expanded = scale.t().repeat_interleave(128, dim=1)
    x_roundtrip = x_fp8.to(torch.float32) * scale_expanded
    x_f32 = x.to(torch.float32)

    cos_sim = torch.nn.functional.cosine_similarity(
        x_f32.flatten(), x_roundtrip.flatten(), dim=0
    )
    assert cos_sim > 0.99, f"FP8 roundtrip cosine similarity {cos_sim:.4f} too low"


if __name__ == "__main__":
    if _check_sm100():
        test_grouped_gemm_fp8_bf16_output(4096, 7168, 4, 64)
        print("test_grouped_gemm_fp8_bf16_output passed")
        test_quantize_output_fp8()
        print("test_quantize_output_fp8 passed")
    else:
        print("Skipping: SM100+ not available")
