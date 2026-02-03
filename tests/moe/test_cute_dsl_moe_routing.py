"""Tests for CuTeDSL MoE routing wrapper."""

import pytest
import torch
from flashinfer.cute_dsl.moe_routing import (
    moe_routing_deepseek,
    moe_routing_reference,
    _compute_permutation_indices,
)


def _check_routing_available():
    """Check if fused routing kernel is available."""
    try:
        from flashinfer.fused_moe.fused_routing_dsv3 import fused_topk_deepseek

        return True
    except Exception:
        return False


@pytest.mark.parametrize("seq_len", [4, 16, 64, 256])
@pytest.mark.parametrize(
    "n_group, topk_group, top_k, num_experts",
    [
        (8, 4, 8, 256),  # DeepSeek-V3
        (1, 1, 8, 384),  # Kimi-K2
    ],
)
def test_routing_reference_properties(seq_len, n_group, topk_group, top_k, num_experts):
    """Test that reference routing produces valid outputs."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    torch.manual_seed(42)

    E_local = 32
    local_expert_offset = 0

    routing_logits = torch.randn(seq_len, num_experts, dtype=torch.float32, device=device)
    routing_bias = torch.randn(num_experts, dtype=torch.bfloat16, device=device)

    result = moe_routing_reference(
        routing_logits,
        routing_bias,
        num_local_experts=E_local,
        local_expert_offset=local_expert_offset,
        n_group=n_group,
        topk_group=topk_group,
        top_k=top_k,
        routed_scaling_factor=2.5,
    )

    # Check shapes
    assert result.topk_values.shape == (seq_len, top_k)
    assert result.topk_indices.shape == (seq_len, top_k)
    assert result.masked_m.shape == (E_local,)
    assert result.m_indptr.shape == (E_local + 1,)

    # Check that topk_values are non-negative (normalized sigmoid * scale)
    assert (result.topk_values >= 0).all()

    # Check that topk_indices are valid expert IDs
    assert (result.topk_indices >= 0).all()
    assert (result.topk_indices < num_experts).all()

    # Check that each token has exactly top_k experts selected
    assert result.topk_indices.shape[1] == top_k

    # Check masked_m sums to <= T * top_k (some experts may be non-local)
    total_local = result.masked_m.sum().item()
    assert total_local <= seq_len * top_k

    # Check m_indptr is consistent with masked_m
    assert result.m_indptr[0] == 0
    assert result.max_padded_tokens == result.m_indptr[-1].item()

    # Check permuted_idx_to_token_idx has valid entries
    valid_mask = result.permuted_idx_to_token_idx >= 0
    valid_entries = result.permuted_idx_to_token_idx[valid_mask]
    assert (valid_entries < seq_len).all()

    # Check expanded_idx_to_permuted_idx: valid entries should map within range
    valid_expanded = result.expanded_idx_to_permuted_idx >= 0
    valid_perm_idx = result.expanded_idx_to_permuted_idx[valid_expanded]
    if valid_perm_idx.numel() > 0:
        assert (valid_perm_idx < result.max_padded_tokens).all()


@pytest.mark.parametrize("seq_len", [4, 16, 64])
def test_permutation_roundtrip(seq_len):
    """Test that permutation indices form a valid bijection for local experts."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = "cuda"
    torch.manual_seed(42)

    E_local = 32
    E_global = 256
    top_k = 8
    local_expert_offset = 0

    routing_logits = torch.randn(seq_len, E_global, dtype=torch.float32, device=device)
    routing_bias = torch.randn(E_global, dtype=torch.bfloat16, device=device)

    result = moe_routing_reference(
        routing_logits,
        routing_bias,
        num_local_experts=E_local,
        local_expert_offset=local_expert_offset,
        top_k=top_k,
    )

    # For each valid entry in expanded_idx_to_permuted_idx,
    # permuted_idx_to_token_idx[perm_idx] should equal the token_id
    for idx in range(seq_len * top_k):
        perm_idx = result.expanded_idx_to_permuted_idx[idx].item()
        if perm_idx < 0:
            continue
        token_id = idx // top_k
        mapped_token = result.permuted_idx_to_token_idx[perm_idx].item()
        assert mapped_token == token_id, (
            f"Roundtrip failed: expanded[{idx}]={perm_idx}, "
            f"permuted[{perm_idx}]={mapped_token}, expected token={token_id}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("seq_len", [4, 16, 64])
def test_routing_kernel_vs_reference(seq_len):
    """Compare fused routing kernel output against Python reference."""
    if not _check_routing_available():
        pytest.skip("Fused routing kernel not available")

    from flashinfer.utils import get_compute_capability

    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [8, 9, 10, 12]:
        pytest.skip(f"Unsupported compute capability {cc}")

    device = "cuda"
    torch.manual_seed(42)

    E_global = 256
    E_local = 32
    top_k = 8
    n_group = 8
    topk_group = 4
    local_expert_offset = 0
    routed_scaling_factor = 2.5

    routing_logits = torch.randn(seq_len, E_global, dtype=torch.float32, device=device)
    routing_bias = torch.randn(E_global, dtype=torch.bfloat16, device=device)

    # Run kernel
    kernel_result = moe_routing_deepseek(
        routing_logits,
        routing_bias,
        num_local_experts=E_local,
        local_expert_offset=local_expert_offset,
        n_group=n_group,
        topk_group=topk_group,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
    )

    # Run reference
    ref_result = moe_routing_reference(
        routing_logits,
        routing_bias,
        num_local_experts=E_local,
        local_expert_offset=local_expert_offset,
        n_group=n_group,
        topk_group=topk_group,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
    )

    # Compare: both should select the same experts per token
    # Sort indices per token for comparison (order within top_k may differ)
    kernel_sorted, _ = kernel_result.topk_indices.sort(dim=1)
    ref_sorted, _ = ref_result.topk_indices.sort(dim=1)
    assert torch.equal(kernel_sorted, ref_sorted), (
        f"Expert selection mismatch:\nkernel={kernel_sorted}\nref={ref_sorted}"
    )

    # Compare weights: gather kernel weights in same order as ref
    # For each token, match experts and compare weights
    for t in range(seq_len):
        for k in range(top_k):
            expert_id = ref_result.topk_indices[t, k].item()
            # Find this expert in kernel result
            kernel_k = (kernel_result.topk_indices[t] == expert_id).nonzero()
            assert kernel_k.numel() > 0, f"Expert {expert_id} not found in kernel output for token {t}"
            kernel_weight = kernel_result.topk_values[t, kernel_k[0, 0]].item()
            ref_weight = ref_result.topk_values[t, k].item()
            assert abs(kernel_weight - ref_weight) < 1e-3, (
                f"Weight mismatch for token {t}, expert {expert_id}: "
                f"kernel={kernel_weight:.6f}, ref={ref_weight:.6f}"
            )

    # Compare permutation structure
    assert kernel_result.masked_m.sum() == ref_result.masked_m.sum()

    # Compare permutation indices exactly (both use the same vectorized
    # _compute_permutation_indices, so they must match given same topk_indices)
    assert torch.equal(kernel_result.masked_m, ref_result.masked_m), (
        f"masked_m mismatch:\nkernel={kernel_result.masked_m}\nref={ref_result.masked_m}"
    )
    assert torch.equal(kernel_result.m_indptr, ref_result.m_indptr), (
        f"m_indptr mismatch:\nkernel={kernel_result.m_indptr}\nref={ref_result.m_indptr}"
    )
    assert torch.equal(
        kernel_result.permuted_idx_to_token_idx,
        ref_result.permuted_idx_to_token_idx,
    ), "permuted_idx_to_token_idx mismatch"
    assert torch.equal(
        kernel_result.expanded_idx_to_permuted_idx,
        ref_result.expanded_idx_to_permuted_idx,
    ), "expanded_idx_to_permuted_idx mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("seq_len", [4, 16, 64, 256])
@pytest.mark.parametrize("pad_to", [1, 4])
def test_kernel_permutation_roundtrip(seq_len, pad_to):
    """Verify permutation roundtrip for moe_routing_deepseek (vectorized path)."""
    if not _check_routing_available():
        pytest.skip("Fused routing kernel not available")

    from flashinfer.utils import get_compute_capability

    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] not in [8, 9, 10, 12]:
        pytest.skip(f"Unsupported compute capability {cc}")

    device = "cuda"
    torch.manual_seed(42)

    E_global = 256
    E_local = 32
    top_k = 8
    local_expert_offset = 0

    routing_logits = torch.randn(seq_len, E_global, dtype=torch.float32, device=device)
    routing_bias = torch.randn(E_global, dtype=torch.bfloat16, device=device)

    result = moe_routing_deepseek(
        routing_logits,
        routing_bias,
        num_local_experts=E_local,
        local_expert_offset=local_expert_offset,
        top_k=top_k,
        pad_to=pad_to,
    )

    # Roundtrip: expanded_idx -> permuted_idx -> token_id must match idx // top_k
    exp_to_perm = result.expanded_idx_to_permuted_idx.cpu()
    perm_to_tok = result.permuted_idx_to_token_idx.cpu()
    topk_idx = result.topk_indices.cpu()
    local_end = local_expert_offset + E_local

    for idx in range(seq_len * top_k):
        perm_idx = exp_to_perm[idx].item()
        global_e = topk_idx[idx // top_k, idx % top_k].item()
        is_local = local_expert_offset <= global_e < local_end

        if is_local:
            assert perm_idx >= 0, f"Local expert entry {idx} has perm_idx={perm_idx}"
            mapped_token = perm_to_tok[perm_idx].item()
            expected_token = idx // top_k
            assert mapped_token == expected_token, (
                f"Roundtrip failed at idx={idx}: perm_idx={perm_idx}, "
                f"mapped_token={mapped_token}, expected={expected_token}"
            )
        else:
            assert perm_idx == -1, (
                f"Non-local expert entry {idx} (expert={global_e}) has perm_idx={perm_idx}"
            )

    # m_indptr consistency: last element = max_padded_tokens
    assert result.m_indptr[-1].item() == result.max_padded_tokens

    # padded_m elements must be multiples of pad_to
    padded_m = result.m_indptr[1:] - result.m_indptr[:-1]
    if pad_to > 1:
        assert (padded_m % pad_to == 0).all(), f"padded_m not aligned to {pad_to}: {padded_m}"


if __name__ == "__main__":
    test_routing_reference_properties(16, 8, 4, 8, 256)
    print("test_routing_reference_properties passed")
    test_permutation_roundtrip(16)
    print("test_permutation_roundtrip passed")
    test_kernel_permutation_roundtrip(16, 4)
    print("test_kernel_permutation_roundtrip passed")
    print("All routing tests passed!")
