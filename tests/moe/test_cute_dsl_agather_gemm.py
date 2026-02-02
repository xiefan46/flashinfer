"""Tests for A-Gather GEMM (scatter fused into GEMM1)."""

import pytest
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


def _fp8_block_quant_1d(x_bf16, block=128):
    """Quantize [M, N] to FP8."""
    M, N = x_bf16.shape
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
    """Quantize [E, R, C] to FP8."""
    E, R, C = w_bf16.shape
    nb_r, nb_c = R // block, C // block
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
@pytest.mark.parametrize("m_per_expert", [16, 64])
def test_agather_identity_permutation(m_per_expert):
    """With identity permutation, A-gather GEMM should match standard GEMM."""
    from flashinfer.cute_dsl.moe_grouped_gemm_fp8 import moe_gemm1_fp8
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise

    device = "cuda"
    torch.manual_seed(42)

    E = 4
    H = 7168
    two_I = 4096
    T = E * m_per_expert

    # Generate data
    a_bf16 = torch.randn(T, H, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(E, two_I, H, dtype=torch.bfloat16, device=device)

    a_fp8, a_scales = _fp8_block_quant_1d(a_bf16)
    b_fp8, b_scales = _fp8_block_quant_2d(b_bf16)

    a_scale_mn = a_scales.t().contiguous()

    m_indptr = torch.arange(0, (E + 1) * m_per_expert, m_per_expert, dtype=torch.int32, device=device)

    # Identity permutation — each token maps to itself
    permuted_idx = torch.arange(T, dtype=torch.int32, device=device)

    # Run with A-gather (identity)
    gemm1_out, gemm1_scale = moe_gemm1_fp8(
        a_fp8, b_fp8, a_scale_mn, b_scales, m_indptr, permuted_idx,
    )

    # Run standard GEMM for reference
    ref_out = group_gemm_fp8_nt_groupwise(
        a_fp8, b_fp8, a_scale_mn, b_scales, m_indptr,
        scale_major_mode="MN", out_dtype=torch.bfloat16,
    )

    # Dequant GEMM1 output
    scale_t = gemm1_scale.t().contiguous()
    scale_expanded = scale_t.repeat_interleave(128, dim=1)
    gemm1_dequant = gemm1_out.to(torch.float32) * scale_expanded

    ref_f32 = ref_out.to(torch.float32)

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), gemm1_dequant.flatten(), dim=0
    ).item()

    assert cos_sim > 0.95, f"Identity A-gather cosine similarity {cos_sim:.4f} too low"


@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
@pytest.mark.parametrize("m_per_expert", [16, 64])
def test_agather_random_permutation(m_per_expert):
    """Random permutation: A-gather(perm) should match pre-scattered A + standard GEMM."""
    from flashinfer.cute_dsl.moe_grouped_gemm_fp8 import moe_gemm1_fp8
    from flashinfer.gemm import group_gemm_fp8_nt_groupwise

    device = "cuda"
    torch.manual_seed(42)

    E = 4
    H = 7168
    two_I = 4096
    T_original = 128  # More tokens than needed
    T_permuted = E * m_per_expert

    # Generate data
    a_bf16_original = torch.randn(T_original, H, dtype=torch.bfloat16, device=device)
    b_bf16 = torch.randn(E, two_I, H, dtype=torch.bfloat16, device=device)

    a_fp8_original, a_scales_original = _fp8_block_quant_1d(a_bf16_original)
    b_fp8, b_scales = _fp8_block_quant_2d(b_bf16)

    a_scale_mn_original = a_scales_original.t().contiguous()  # [H//128, T_original]

    # Random permutation (gather indices)
    perm = torch.randint(0, T_original, (T_permuted,), dtype=torch.int32, device=device)

    m_indptr = torch.arange(0, (E + 1) * m_per_expert, m_per_expert, dtype=torch.int32, device=device)

    # Method 1: A-gather (let kernel do the gather)
    gemm1_out, gemm1_scale = moe_gemm1_fp8(
        a_fp8_original, b_fp8, a_scale_mn_original, b_scales, m_indptr, perm,
    )

    # Method 2: Pre-scatter A, then standard GEMM
    a_fp8_scattered = a_fp8_original[perm.long()]
    a_scale_scattered = a_scale_mn_original[:, perm.long()]

    ref_out = group_gemm_fp8_nt_groupwise(
        a_fp8_scattered, b_fp8, a_scale_scattered, b_scales, m_indptr,
        scale_major_mode="MN", out_dtype=torch.bfloat16,
    )

    # Compare: dequant GEMM1 output vs ref_out
    scale_t = gemm1_scale.t().contiguous()
    scale_expanded = scale_t.repeat_interleave(128, dim=1)
    gemm1_dequant = gemm1_out.to(torch.float32) * scale_expanded

    ref_f32 = ref_out.to(torch.float32)

    cos_sim = torch.nn.functional.cosine_similarity(
        ref_f32.flatten(), gemm1_dequant.flatten(), dim=0
    ).item()

    assert cos_sim > 0.95, f"Random permutation A-gather cosine similarity {cos_sim:.4f} too low"


if __name__ == "__main__":
    if _check_sm100():
        test_agather_identity_permutation(16)
        print("test_agather_identity_permutation passed")
        test_agather_random_permutation(16)
        print("test_agather_random_permutation passed")
    else:
        print("Skipping: SM100+ not available")
