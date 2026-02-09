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

FP8 Grouped GEMM v3.3 — Cluster Shape (2,1) Optimization
==========================================================

Based on v3.2 with cluster_shape_mn=(2,1) for TMA multicast.
B matrix loads are shared across 2 CTAs processing different M tiles.

GEMM2 still outputs BF16 (reuses v3 GEMM2 unchanged).

AI-assisted implementation (Claude).
"""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from .moe_grouped_gemm_cutedsl_v3_3 import moe_grouped_gemm_fp8_cutedsl_v3_3
from .moe_grouped_gemm_fp8_v3 import (
    compute_aligned_m_indptr,
    gather_from_flat_padded,
    moe_gemm2_fp8_v3,
    scatter_to_flat_padded,
)


@flashinfer_api
def moe_gemm1_fp8_v3_3(
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    permuted_idx_to_token_idx: torch.Tensor,
    gemm1_out: Optional[torch.Tensor] = None,
    gemm1_out_scale: Optional[torch.Tensor] = None,
) -> tuple:
    """GEMM1 for MoE v3.2: hidden_states @ W13^T -> FP8 output (flat layout).

    Uses v3.2 kernel which directly outputs FP8 + scales from the epilogue,
    eliminating the separate _quantize_output_fp8() call.

    Args:
        hidden_states: [T, H] float8_e4m3fn
        gemm1_weights: [E, 2*I, H] float8_e4m3fn
        hidden_states_scale: [H//128, T] float32 (MN-major)
        gemm1_weights_scale: [E, 2*I//128, H//128] float32 (MN-major)
        m_indptr: [E+1] int32
        permuted_idx_to_token_idx: [max_padded] int32
        gemm1_out: optional [total_padded_M, 2*I] float8_e4m3fn
        gemm1_out_scale: optional [2*I//128, total_padded_M] float32

    Returns:
        (gemm1_out, gemm1_out_scale, m_indptr_aligned, m_indptr_tiles, masked_m, dst_row)
    """
    E = gemm1_weights.shape[0]
    H = hidden_states.shape[1]
    two_I = gemm1_weights.shape[1]

    # Gather A using permuted indices (A-gather)
    valid_mask = permuted_idx_to_token_idx >= 0
    safe_indices = permuted_idx_to_token_idx.clone()
    safe_indices[~valid_mask] = 0
    a_gathered = hidden_states[safe_indices.long()]
    a_scale_gathered = hidden_states_scale[:, safe_indices.long()]

    max_padded = a_gathered.shape[0]

    # Compute 128-aligned expert boundaries
    masked_m, aligned_m, m_indptr_aligned, m_indptr_tiles, total_padded_M = (
        compute_aligned_m_indptr(m_indptr)
    )

    if total_padded_M == 0:
        empty_out = torch.empty(0, two_I, dtype=torch.float8_e4m3fn, device=hidden_states.device)
        empty_scale = torch.empty(two_I // 128, 0, dtype=torch.float32, device=hidden_states.device)
        return empty_out, empty_scale, m_indptr_aligned, m_indptr_tiles, masked_m, torch.empty(0, dtype=torch.int64, device=hidden_states.device)

    # Scatter to flat padded layout
    a_flat, a_scale_flat, dst_row = scatter_to_flat_padded(
        a_gathered, a_scale_gathered, m_indptr, m_indptr_aligned, total_padded_M, H
    )

    # Run v3.2 flat grouped GEMM -> direct FP8 + scales output
    gemm1_out_fp8, gemm1_out_sc = moe_grouped_gemm_fp8_cutedsl_v3_3(
        a_flat,
        gemm1_weights,
        a_scale_flat,
        gemm1_weights_scale,
        masked_m,
        m_indptr_tiles,
        out_fp8=gemm1_out,
        out_scale=gemm1_out_scale,
    )

    return gemm1_out_fp8, gemm1_out_sc, m_indptr_aligned, m_indptr_tiles, masked_m, dst_row


# GEMM2 reuses v3 unchanged (outputs BF16 for downstream finalize)
moe_gemm2_fp8_v3_3 = moe_gemm2_fp8_v3
