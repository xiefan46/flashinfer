"""Tests for CuTeDSL MoE Finalize (Gather + Weighted Reduce)."""

import pytest
import torch

from flashinfer.cute_dsl.moe_finalize import moe_finalize, moe_finalize_reference


def _setup_finalize_inputs(T, top_k, H, E_local, local_expert_offset, device="cuda"):
    """Create synthetic inputs for finalize testing."""
    E_global = max(local_expert_offset + E_local, 256)
    local_end = local_expert_offset + E_local

    # Random expert selection (pick from local experts mostly)
    topk_indices = torch.randint(
        local_expert_offset,
        local_end,
        (T, top_k),
        dtype=torch.int64,
        device=device,
    )

    # Routing weights
    expert_weights = torch.randn(T, top_k, dtype=torch.float32, device=device).abs()
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)

    # Build permutation indices
    masked_m = torch.zeros(E_local, dtype=torch.int32, device=device)
    for e in range(E_local):
        ge = local_expert_offset + e
        masked_m[e] = (topk_indices == ge).sum().item()

    m_indptr = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    m_indptr[1:] = torch.cumsum(masked_m, dim=0)
    max_padded = int(m_indptr[-1].item())

    permuted_idx_to_token_idx = torch.full(
        (max(max_padded, 1),), -1, dtype=torch.int32, device=device
    )
    expanded_idx_to_permuted_idx = torch.full(
        (T * top_k,), -1, dtype=torch.int32, device=device
    )

    expert_fill_count = torch.zeros(E_local, dtype=torch.int32, device=device)
    topk_flat = topk_indices.view(-1)

    for idx in range(T * top_k):
        ge = int(topk_flat[idx].item())
        if ge < local_expert_offset or ge >= local_end:
            continue
        le = ge - local_expert_offset
        token_id = idx // top_k
        pos = int(m_indptr[le].item()) + int(expert_fill_count[le].item())
        permuted_idx_to_token_idx[pos] = token_id
        expanded_idx_to_permuted_idx[idx] = pos
        expert_fill_count[le] += 1

    # GEMM2 output in permuted order
    gemm2_out = torch.randn(max(max_padded, 1), H, dtype=torch.bfloat16, device=device)

    return (
        gemm2_out,
        expert_weights,
        topk_indices,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_token_idx,
        max_padded,
    )


def _reference_finalize(
    gemm2_out, expert_weights, topk_indices,
    expanded_idx_to_permuted_idx, E_local, local_expert_offset, H
):
    """Pure Python reference finalize."""
    T, top_k = expert_weights.shape
    local_end = local_expert_offset + E_local
    device = gemm2_out.device

    acc = torch.zeros(T, H, dtype=torch.float32, device=device)

    for t in range(T):
        for k in range(top_k):
            idx = t * top_k + k
            perm_idx = expanded_idx_to_permuted_idx[idx].item()
            if perm_idx < 0:
                continue
            ge = topk_indices[t, k].item()
            if ge < local_expert_offset or ge >= local_end:
                continue
            w = expert_weights[t, k].item()
            acc[t] += w * gemm2_out[perm_idx].to(torch.float32)

    return acc.to(torch.bfloat16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("T", [4, 16, 64])
@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize("H", [256, 1024])
def test_finalize_correctness(T, top_k, H):
    """Test CuTeDSL finalize kernel against Python reference."""
    device = "cuda"
    torch.manual_seed(42)
    E_local = 32
    local_expert_offset = 0

    (
        gemm2_out,
        expert_weights,
        topk_indices,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_token_idx,
        max_padded,
    ) = _setup_finalize_inputs(T, top_k, H, E_local, local_expert_offset, device)

    # Run CuTeDSL kernel
    output = moe_finalize(
        gemm2_out,
        expert_weights,
        topk_indices,
        expanded_idx_to_permuted_idx,
        E_local,
        local_expert_offset,
        H,
    )

    # Reference
    ref = _reference_finalize(
        gemm2_out, expert_weights, topk_indices,
        expanded_idx_to_permuted_idx, E_local, local_expert_offset, H,
    )

    assert output.shape == (T, H)
    assert output.dtype == torch.bfloat16

    assert torch.allclose(output.float(), ref.float(), atol=1e-3), (
        f"Max diff: {(output.float() - ref.float()).abs().max():.6e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("T", [4, 16, 64])
def test_finalize_matches_reference(T):
    """Test CuTeDSL kernel matches moe_finalize_reference."""
    device = "cuda"
    torch.manual_seed(42)
    top_k = 8
    H = 512
    E_local = 32
    local_expert_offset = 0

    (
        gemm2_out,
        expert_weights,
        topk_indices,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_token_idx,
        max_padded,
    ) = _setup_finalize_inputs(T, top_k, H, E_local, local_expert_offset, device)

    out_ref = moe_finalize_reference(
        gemm2_out, expert_weights, topk_indices,
        expanded_idx_to_permuted_idx, E_local, local_expert_offset, H,
    )

    out_kernel = moe_finalize(
        gemm2_out, expert_weights, topk_indices,
        expanded_idx_to_permuted_idx, E_local, local_expert_offset, H,
    )

    assert torch.allclose(out_ref.float(), out_kernel.float(), atol=1e-3), (
        f"Max diff: {(out_ref.float() - out_kernel.float()).abs().max():.6e}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_finalize_no_local_experts():
    """Test finalize when no tokens are routed to local experts."""
    device = "cuda"
    T, top_k, H = 8, 4, 256
    E_local = 32
    local_expert_offset = 200  # Far from typical routing

    # All experts routed to non-local range
    topk_indices = torch.randint(0, 32, (T, top_k), dtype=torch.int64, device=device)
    expert_weights = torch.randn(T, top_k, dtype=torch.float32, device=device).abs()
    expanded_idx_to_permuted_idx = torch.full(
        (T * top_k,), -1, dtype=torch.int32, device=device
    )
    gemm2_out = torch.randn(1, H, dtype=torch.bfloat16, device=device)

    output = moe_finalize(
        gemm2_out, expert_weights, topk_indices,
        expanded_idx_to_permuted_idx, E_local, local_expert_offset, H,
    )

    # All zeros since no local experts
    assert output.abs().sum() == 0


if __name__ == "__main__":
    test_finalize_correctness(16, 8, 512)
    print("test_finalize_correctness passed")
    test_finalize_matches_reference(16)
    print("test_finalize_matches_reference passed")
    test_finalize_no_local_experts()
    print("test_finalize_no_local_experts passed")
    print("All finalize tests passed!")
