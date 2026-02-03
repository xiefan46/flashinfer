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

Two backends:
  - moe_routing_deepseek: fused_topk_deepseek + vectorized PyTorch permutation
  - moe_routing_sglang: sgl_kernel moe_fused_gate + prepare_moe_input (faster)
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
):
    """Compute permutation indices from routing results (vectorized).

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

    # Flatten to [T * top_k]
    topk_flat = topk_indices.view(-1)

    # Identify entries routed to local experts
    local_mask = (topk_flat >= local_expert_offset) & (topk_flat < local_end)
    local_e_all = (topk_flat - local_expert_offset).to(torch.int32)

    # Count tokens per local expert via bincount
    # Mask non-local entries to 0 then subtract their contribution
    masked_local_e = local_e_all.clamp(0, E_local - 1)
    masked_local_e = torch.where(local_mask, masked_local_e, torch.zeros_like(masked_local_e))
    masked_m = torch.bincount(
        masked_local_e[local_mask].long(), minlength=E_local
    ).to(torch.int32)

    # Pad each expert's count to multiple of pad_to
    if pad_to > 1:
        padded_m = ((masked_m + pad_to - 1) // pad_to) * pad_to
    else:
        padded_m = masked_m.clone()

    # Compute cumulative offsets (m_indptr)
    m_indptr = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    m_indptr[1:] = torch.cumsum(padded_m, dim=0)
    max_padded_tokens = int(m_indptr[-1].item())

    # Output tensors
    permuted_idx_to_token_idx = torch.full(
        (max(max_padded_tokens, 1),), -1, dtype=torch.int32, device=device
    )
    expanded_idx_to_permuted_idx = torch.full(
        (T * top_k,), -1, dtype=torch.int32, device=device
    )

    # Get flat indices of local entries
    local_indices = torch.where(local_mask)[0]  # indices into topk_flat
    if local_indices.numel() == 0:
        return masked_m, padded_m, m_indptr, max_padded_tokens, permuted_idx_to_token_idx, expanded_idx_to_permuted_idx

    local_experts = local_e_all[local_indices]  # [num_local_entries] int32
    local_token_ids = (local_indices // top_k).to(torch.int32)  # token id for each entry

    # Stable sort by expert — preserves original idx order within each expert
    sorted_order = torch.argsort(local_experts.long(), stable=True)
    sorted_experts = local_experts[sorted_order]
    sorted_flat_indices = local_indices[sorted_order]
    sorted_token_ids = local_token_ids[sorted_order]

    # Compute within-expert position using unpadded cumsum offsets
    expert_start = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    expert_start[1:] = torch.cumsum(masked_m, dim=0)
    # Each element's position-in-expert = its index in sorted array - expert_start[its expert]
    pos_in_sorted = torch.arange(local_indices.numel(), dtype=torch.int32, device=device)
    pos_in_expert = pos_in_sorted - expert_start[sorted_experts.long()]

    # Permuted position = m_indptr[expert] + pos_in_expert (using padded offsets)
    permuted_pos = m_indptr[sorted_experts.long()] + pos_in_expert

    # Scatter into output tensors
    permuted_idx_to_token_idx[permuted_pos.long()] = sorted_token_ids
    expanded_idx_to_permuted_idx[sorted_flat_indices.long()] = permuted_pos

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


@flashinfer_api
def moe_routing_sglang(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    num_local_experts: int,
    local_expert_offset: int = 0,
    n_group: int = 8,
    topk_group: int = 4,
    top_k: int = 8,
    routed_scaling_factor: float = 2.5,
    intermediate_size: int = 2048,
    hidden_size: int = 7168,
    pad_to: int = 1,
) -> MoERoutingOutput:
    """Run DeepSeek-V3 routing using sgl_kernel (moe_fused_gate + prepare_moe_input).

    Faster than moe_routing_deepseek by using fused CUDA kernels from sgl_kernel
    instead of Python-level torch ops for permutation index computation.

    Args:
        routing_logits: [T, E_global] float32 routing scores.
        routing_bias: [E_global] routing bias (bf16 or f32).
        num_local_experts: number of local experts on this rank.
        local_expert_offset: global expert ID offset for local experts.
        n_group: number of expert groups (DeepSeek-V3: 8).
        topk_group: number of top groups (DeepSeek-V3: 4).
        top_k: number of experts per token (DeepSeek-V3: 8).
        routed_scaling_factor: scaling factor for routing weights.
        intermediate_size: I dimension (needed by prepare_moe_input).
        hidden_size: H dimension (needed by prepare_moe_input).
        pad_to: unused (kept for API compat; sgl_kernel doesn't pad).

    Returns:
        MoERoutingOutput with all index tensors needed by downstream stages.
    """
    from sgl_kernel import moe_fused_gate, prepare_moe_input

    T, E_global = routing_logits.shape
    E_local = num_local_experts
    device = routing_logits.device

    # --- Step 1: DeepSeek-V3 routing via moe_fused_gate ---
    topk_weights, topk_ids = moe_fused_gate(
        routing_logits.to(torch.float32).contiguous(),
        routing_bias.to(torch.float32).contiguous(),
        n_group,
        topk_group,
        top_k,
        num_fused_shared_experts=0,
        routed_scaling_factor=float(routed_scaling_factor),
        apply_routed_scaling_factor_on_output=False,
    )
    # Apply scaling factor (moe_fused_gate normalizes but doesn't scale when
    # apply_routed_scaling_factor_on_output=False)
    topk_weights = topk_weights * routed_scaling_factor

    # Save original global expert IDs for finalize stage
    topk_indices_global = topk_ids.clone()

    # --- Step 2: Map global expert IDs to local [0, E_local) ---
    # Non-local experts get mapped to E_local (out of range) so
    # prepare_moe_input won't count them.
    local_end = local_expert_offset + E_local
    local_mask = (topk_ids >= local_expert_offset) & (topk_ids < local_end)
    local_ids = (topk_ids - local_expert_offset).to(torch.int32)
    # Set non-local to E_local (will be ignored by prepare_moe_input)
    local_ids = torch.where(local_mask, local_ids, torch.full_like(local_ids, E_local))
    topk_ids_local = local_ids.to(torch.int32)

    # --- Step 3: prepare_moe_input ---
    m = T
    expert_offsets = torch.empty((E_local + 1,), dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty((E_local, 3), dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty((E_local, 3), dtype=torch.int32, device=device)
    a_map = torch.empty((m * top_k,), dtype=torch.int32, device=device)
    # Pre-fill c_map with -1 so non-local entries stay -1
    c_map = torch.full((m * top_k,), -1, dtype=torch.int32, device=device)

    prepare_moe_input(
        topk_ids_local,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        E_local,
        intermediate_size,
        hidden_size,
    )

    # --- Step 4: Map to MoERoutingOutput ---
    max_padded_tokens = int(expert_offsets[-1].item())
    masked_m = (expert_offsets[1:] - expert_offsets[:-1]).to(torch.int32)

    # a_map = permuted_idx_to_token_idx (token ID for each permuted slot)
    # Only first max_padded_tokens entries are valid
    if max_padded_tokens > 0:
        permuted_idx_to_token_idx = a_map[:max_padded_tokens]
    else:
        permuted_idx_to_token_idx = torch.full(
            (1,), -1, dtype=torch.int32, device=device
        )

    # c_map = expanded_idx_to_permuted_idx (-1 for non-local entries)
    expanded_idx_to_permuted_idx = c_map

    return MoERoutingOutput(
        topk_values=topk_weights.to(torch.float32),
        topk_indices=topk_indices_global,
        masked_m=masked_m,
        permuted_idx_to_token_idx=permuted_idx_to_token_idx,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        max_padded_tokens=max_padded_tokens,
        m_indptr=expert_offsets,
    )
