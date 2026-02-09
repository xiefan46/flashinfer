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

CuTeDSL DeepSeek FP8 MoE Pipeline v3.3 — Cluster Shape (2,1) Optimization
=========================================================================

Based on v3.2 with cluster_shape_mn=(2,1) for TMA multicast.
B matrix loads are shared across 2 CTAs processing different M tiles.

Pipeline: Routing -> Scatter -> GEMM1(flat,FP8 out,2x1 cluster) -> SwiGLU -> GEMM2(flat,2x1 cluster) -> Unsatter -> Finalize

AI-assisted implementation (Claude).
"""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from .moe_routing import moe_routing_sglang
from .moe_grouped_gemm_fp8_v3_3 import (
    moe_gemm1_fp8_v3_3,
    moe_gemm2_fp8_v3_3,
)
from .moe_grouped_gemm_fp8_v3 import gather_from_flat_padded
from .moe_activation import moe_swiglu_fp8_requant
from .moe_finalize import moe_finalize


@flashinfer_api
def cutedsl_fp8_moe_v3_3(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    num_experts_global: int,
    num_local_experts: int,
    local_expert_offset: int = 0,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
    intermediate_size: int = 2048,
    routed_scaling_factor: float = 2.5,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """CuTeDSL DeepSeek FP8 MoE full pipeline v3.2.

    Same as v3 but GEMM1 outputs FP8 directly (no separate quantization kernel).

    Pipeline: Routing -> Scatter -> GEMM1(flat,FP8 out) -> SwiGLU -> GEMM2(flat) -> Unsatter -> Finalize

    Args:
        routing_logits: [T, E_global] float32
        routing_bias: [E_global] bf16
        hidden_states: [T, H] float8_e4m3fn
        hidden_states_scale: [H//128, T] float32 (MN-major)
        gemm1_weights: [E_local, 2*I, H] float8_e4m3fn
        gemm1_weights_scale: [E_local, 2*I//128, H//128] float32
        gemm2_weights: [E_local, H, I] float8_e4m3fn
        gemm2_weights_scale: [E_local, H//128, I//128] float32
        num_experts_global: total number of experts globally
        num_local_experts: number of local experts on this rank
        local_expert_offset: global expert ID offset
        top_k: number of experts per token
        n_group: number of expert groups
        topk_group: number of top groups
        intermediate_size: I dimension
        routed_scaling_factor: routing weight scaling factor
        output: optional [T, H] bfloat16 pre-allocated output

    Returns:
        output: [T, H] bfloat16
    """
    T, H = hidden_states.shape
    I = intermediate_size
    device = hidden_states.device

    # --- Stage 1: Routing (sgl_kernel backend) ---
    routing_result = moe_routing_sglang(
        routing_logits,
        routing_bias,
        num_local_experts=num_local_experts,
        local_expert_offset=local_expert_offset,
        n_group=n_group,
        topk_group=topk_group,
        top_k=top_k,
        routed_scaling_factor=routed_scaling_factor,
        intermediate_size=I,
        hidden_size=H,
    )

    max_padded = routing_result.max_padded_tokens
    if max_padded == 0:
        if output is None:
            return torch.zeros(T, H, dtype=torch.bfloat16, device=device)
        output.zero_()
        return output

    # --- Stage 2: GEMM1 (A-gather + scatter to flat + direct FP8 output) ---
    gemm1_out, gemm1_scale, m_indptr_aligned, m_indptr_tiles, masked_m, dst_row = (
        moe_gemm1_fp8_v3_3(
            hidden_states,
            gemm1_weights,
            hidden_states_scale,
            gemm1_weights_scale,
            routing_result.m_indptr,
            routing_result.permuted_idx_to_token_idx,
        )
    )

    total_padded_M = gemm1_out.shape[0]

    # --- Stage 3: SwiGLU + FP8 Requantization ---
    act_out, act_scale = moe_swiglu_fp8_requant(
        gemm1_out,
        gemm1_scale,
    )

    # --- Stage 4: GEMM2 (FP8->BF16, flat layout, reuses v3 kernel) ---
    gemm2_out_flat = moe_gemm2_fp8_v3_3(
        act_out,
        gemm2_weights,
        act_scale,
        gemm2_weights_scale,
        masked_m,
        m_indptr_tiles,
    )

    # --- Stage 4.5: Unsatter flat padded -> expert-grouped layout ---
    gemm2_out = gather_from_flat_padded(gemm2_out_flat, dst_row, max_padded)

    # --- Stage 5: Finalize (Gather + Weighted Reduce) ---
    output = moe_finalize(
        gemm2_out,
        routing_result.topk_values,
        routing_result.topk_indices,
        routing_result.expanded_idx_to_permuted_idx,
        num_local_experts,
        local_expert_offset,
        H,
        output=output,
    )

    return output
