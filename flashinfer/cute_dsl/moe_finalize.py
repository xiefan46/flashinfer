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

MoE Finalize: Gather + Weighted Reduce
=======================================

Gathers expert outputs by token and computes weighted sum:
  output[t, :] = sum_{k=0}^{top_k-1} expert_weights[t,k] * gemm2_out[permuted_idx[t*top_k+k], :]

Pipeline position: GEMM2 output -> [this kernel] -> final MoE output
"""

import torch

from ..api_logging import flashinfer_api


@flashinfer_api
def moe_finalize(
    gemm2_out: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    num_local_experts: int,
    local_expert_offset: int,
    hidden_size: int,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Finalize MoE: gather expert outputs and weighted reduce.

    Args:
        gemm2_out: [max_padded, H] bfloat16 — GEMM2 output in permuted order.
        expert_weights: [T, top_k] float32 — per-token per-expert routing weights.
        topk_indices: [T, top_k] int64 — global expert IDs selected per token.
        expanded_idx_to_permuted_idx: [T*top_k] int32 — maps (t*top_k+k) to
            position in permuted layout.
        num_local_experts: number of local experts on this rank.
        local_expert_offset: global expert ID offset for local experts.
        hidden_size: hidden dimension H.
        output: [T, H] bfloat16 — pre-allocated output (optional).

    Returns:
        output: [T, H] bfloat16
    """
    T, top_k = expert_weights.shape
    device = gemm2_out.device
    local_end = local_expert_offset + num_local_experts

    if output is None:
        output = torch.zeros(T, hidden_size, dtype=torch.bfloat16, device=device)
    else:
        output.zero_()

    # Accumulate weighted expert outputs
    acc = torch.zeros(T, hidden_size, dtype=torch.float32, device=device)

    for t in range(T):
        for k in range(top_k):
            expanded_idx = t * top_k + k
            perm_idx = expanded_idx_to_permuted_idx[expanded_idx].item()
            if perm_idx < 0:
                continue

            global_e = topk_indices[t, k].item()
            if global_e < local_expert_offset or global_e >= local_end:
                continue

            w = expert_weights[t, k].item()
            acc[t] += w * gemm2_out[perm_idx].to(torch.float32)

    output.copy_(acc.to(torch.bfloat16))
    return output


@flashinfer_api
def moe_finalize_vectorized(
    gemm2_out: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    num_local_experts: int,
    local_expert_offset: int,
    hidden_size: int,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Vectorized finalize using torch operations (faster than loop version).

    Same interface as moe_finalize but uses tensor indexing for speed.
    """
    T, top_k = expert_weights.shape
    device = gemm2_out.device
    local_end = local_expert_offset + num_local_experts

    if output is None:
        output = torch.zeros(T, hidden_size, dtype=torch.bfloat16, device=device)

    acc = torch.zeros(T, hidden_size, dtype=torch.float32, device=device)

    for k in range(top_k):
        # Get permutation indices for this k-th expert slot
        perm_indices = expanded_idx_to_permuted_idx[k::top_k]  # [T]
        global_expert_ids = topk_indices[:, k]  # [T]
        weights_k = expert_weights[:, k]  # [T]

        # Mask: valid local experts
        valid = (
            (perm_indices >= 0)
            & (global_expert_ids >= local_expert_offset)
            & (global_expert_ids < local_end)
        )

        if not valid.any():
            continue

        valid_tokens = valid.nonzero(as_tuple=True)[0]
        valid_perm = perm_indices[valid_tokens].long()
        valid_weights = weights_k[valid_tokens]

        # Gather expert outputs and accumulate
        gathered = gemm2_out[valid_perm].to(torch.float32)  # [num_valid, H]
        acc[valid_tokens] += gathered * valid_weights.unsqueeze(1)

    output.copy_(acc.to(torch.bfloat16))
    return output
