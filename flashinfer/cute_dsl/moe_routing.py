"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""MoE Routing for DeepSeek-V3 FP8 MoE pipeline.

Wraps the existing fused_topk_deepseek() kernel and generates index tensors
needed by downstream GEMM and finalize stages:
  - masked_m [E_local]: per-expert token count
  - permuted_idx_to_token_idx [max_padded]: GEMM1 A-gather scatter index
  - expanded_idx_to_permuted_idx [T*top_k]: Finalize gather index
  - expert_weights [T, top_k]: routing weights
"""

from typing import NamedTuple

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.fused_moe.fused_routing_dsv3 import fused_topk_deepseek


class MoERoutingOutput(NamedTuple):
    """Output of MoE routing for downstream pipeline stages."""

    topk_values: torch.Tensor  # [T, top_k] float32 — routing weights
    topk_indices: torch.Tensor  # [T, top_k] int64 — selected expert IDs (global)
    masked_m: torch.Tensor  # [E_local] int32 — per-expert token count
    permuted_idx_to_token_idx: torch.Tensor  # [max_padded] int32 — scatter index
    expanded_idx_to_permuted_idx: torch.Tensor  # [T*top_k] int32 — gather index
    max_padded_tokens: int  # total padded token count across all experts
    m_indptr: torch.Tensor  # [E_local+1] int32 — expert token offset (cumsum)


def _compute_permutation_indices(
    topk_indices: torch.Tensor,
    num_local_experts: int,
    local_expert_offset: int,
    top_k: int,
    pad_to: int = 1,
) -> MoERoutingOutput:
    """Compute permutation indices from routing results.

    Given topk_indices [T, top_k] with global expert IDs, compute:
    1. masked_m: how many tokens each local expert received
    2. permuted_idx_to_token_idx: for each slot in the permuted layout,
       which original token it maps to (for A-gather in GEMM1)
    3. expanded_idx_to_permuted_idx: for each (token, k) pair,
       where to find its result in the permuted layout (for Finalize)
    """
    T = topk_indices.shape[0]
    device = topk_indices.device

    E_local = num_local_experts
    local_end = local_expert_offset + E_local

    # Count tokens per local expert
    masked_m = torch.zeros(E_local, dtype=torch.int32, device=device)
    for e in range(E_local):
        global_e = local_expert_offset + e
        masked_m[e] = (topk_indices == global_e).sum().item()

    # Pad each expert's count to multiple of pad_to
    if pad_to > 1:
        padded_m = ((masked_m + pad_to - 1) // pad_to) * pad_to
    else:
        padded_m = masked_m.clone()

    # Compute cumulative offsets (m_indptr)
    m_indptr = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    m_indptr[1:] = torch.cumsum(padded_m, dim=0)
    max_padded_tokens = int(m_indptr[-1].item())

    # Build permuted_idx_to_token_idx and expanded_idx_to_permuted_idx
    permuted_idx_to_token_idx = torch.full(
        (max(max_padded_tokens, 1),), -1, dtype=torch.int32, device=device
    )
    expanded_idx_to_permuted_idx = torch.full(
        (T * top_k,), -1, dtype=torch.int32, device=device
    )

    # Per-expert counters for filling positions
    expert_fill_count = torch.zeros(E_local, dtype=torch.int32, device=device)

    # Iterate over all (token, k) pairs
    topk_flat = topk_indices.view(-1)  # [T * top_k]
    for idx in range(T * top_k):
        global_e = int(topk_flat[idx].item())
        if global_e < local_expert_offset or global_e >= local_end:
            continue
        local_e = global_e - local_expert_offset
        token_id = idx // top_k
        pos_in_expert = int(expert_fill_count[local_e].item())
        permuted_pos = int(m_indptr[local_e].item()) + pos_in_expert

        permuted_idx_to_token_idx[permuted_pos] = token_id
        expanded_idx_to_permuted_idx[idx] = permuted_pos
        expert_fill_count[local_e] += 1

    return masked_m, padded_m, m_indptr, max_padded_tokens, permuted_idx_to_token_idx, expanded_idx_to_permuted_idx


@flashinfer_api
def moe_routing_deepseek(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    num_local_experts: int,
    local_expert_offset: int = 0,
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
    routed_scaling_factor: float = 2.5,
    pad_to: int = 1,
) -> MoERoutingOutput:
    """Run DeepSeek-V3 routing and compute permutation indices.

    Args:
        routing_logits: [T, E_global] float32 routing scores.
        routing_bias: [E_global] routing bias (bf16 or f32).
        num_local_experts: number of local experts on this rank.
        local_expert_offset: global expert ID offset for local experts.
        n_group: number of expert groups (DeepSeek-V3: 8).
        topk_group: number of top groups (DeepSeek-V3: 4).
        top_k: number of experts per token (DeepSeek-V3: 8).
        routed_scaling_factor: scaling factor for routing weights.
        pad_to: pad each expert's token count to this multiple.

    Returns:
        MoERoutingOutput with all index tensors needed by downstream stages.
    """
    T = routing_logits.shape[0]
    device = routing_logits.device

    # Allocate output tensors for fused routing
    topk_values = torch.empty(T, top_k, dtype=torch.float32, device=device)
    topk_indices = torch.empty(T, top_k, dtype=torch.int32, device=device)

    # Run fused DeepSeek-V3 routing kernel
    fused_topk_deepseek(
        routing_logits.to(torch.float32),
        routing_bias,
        n_group,
        topk_group,
        top_k,
        routed_scaling_factor,
        topk_values,
        topk_indices,
    )

    # Compute permutation indices for downstream stages
    (
        masked_m,
        padded_m,
        m_indptr,
        max_padded_tokens,
        permuted_idx_to_token_idx,
        expanded_idx_to_permuted_idx,
    ) = _compute_permutation_indices(
        topk_indices,
        num_local_experts,
        local_expert_offset,
        top_k,
        pad_to=pad_to,
    )

    return MoERoutingOutput(
        topk_values=topk_values,
        topk_indices=topk_indices,
        masked_m=masked_m,
        permuted_idx_to_token_idx=permuted_idx_to_token_idx,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        max_padded_tokens=max_padded_tokens,
        m_indptr=m_indptr,
    )


def moe_routing_reference(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    num_local_experts: int,
    local_expert_offset: int = 0,
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
    routed_scaling_factor: float = 2.5,
    pad_to: int = 1,
) -> MoERoutingOutput:
    """Python reference routing for validation (no GPU kernel dependency).

    Implements the same DeepSeek-V3 no-aux routing logic as the CUDA kernel:
    1. sigmoid(scores) + bias
    2. Group experts, take top-2 sum per group -> select top groups
    3. From selected groups, take global top-k experts
    4. Normalize weights: sigmoid / sum(sigmoid) * scale
    """
    T, E_global = routing_logits.shape
    device = routing_logits.device

    logits = routing_logits.to(torch.float32)
    bias = routing_bias.to(torch.float32).reshape(-1)

    # Sigmoid
    s = torch.sigmoid(logits)
    s_with_bias = s + bias

    # Grouping
    group_size = E_global // n_group
    s_wb_grouped = s_with_bias.view(T, n_group, group_size)

    # Group scores = sum of top-2 within each group
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)

    # Select top groups
    _, group_idx = torch.topk(
        group_scores, k=topk_group, dim=1, largest=True, sorted=False
    )
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(2).expand(T, n_group, group_size).reshape(T, E_global)
    )

    # Global top-k within kept groups
    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=top_k, dim=1, largest=True, sorted=False)

    # Normalize weights
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * routed_scaling_factor

    # Extract topk values
    topk_values = weights.gather(1, topk_idx)

    # Compute permutation indices
    (
        masked_m,
        padded_m,
        m_indptr,
        max_padded_tokens,
        permuted_idx_to_token_idx,
        expanded_idx_to_permuted_idx,
    ) = _compute_permutation_indices(
        topk_idx,
        num_local_experts,
        local_expert_offset,
        top_k,
        pad_to=pad_to,
    )

    return MoERoutingOutput(
        topk_values=topk_values,
        topk_indices=topk_idx,
        masked_m=masked_m,
        permuted_idx_to_token_idx=permuted_idx_to_token_idx,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        max_padded_tokens=max_padded_tokens,
        m_indptr=m_indptr,
    )
