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

CuTeDSL DeepSeek FP8 MoE Pipeline v2
======================================

Same pipeline as moe_pipeline.py but uses vectorized GEMM wrappers
(moe_grouped_gemm_fp8_v2) to eliminate Python for-loops and GPU->CPU
synchronization in the padding/unpadding logic.

Pipeline: Routing -> GEMM1(FP8->FP8, A-gather) -> SwiGLU+FP8Requant -> GEMM2(FP8->BF16) -> Finalize

AI-assisted implementation (Claude).
"""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from .moe_routing import moe_routing_sglang
from .moe_grouped_gemm_fp8_v2 import moe_gemm1_fp8_v2, moe_gemm2_fp8_v2
from .moe_activation import moe_swiglu_fp8_requant
from .moe_finalize import moe_finalize
from .moe_pipeline import MoEWorkspace, allocate_moe_workspace


@flashinfer_api
def cutedsl_fp8_moe_v2(
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
    """CuTeDSL DeepSeek FP8 MoE full pipeline v2 (vectorized GEMM wrappers).

    Same interface as cutedsl_fp8_moe but uses vectorized padding/unpadding.

    Pipeline: Routing -> GEMM1(A-gather) -> SwiGLU+FP8Requant -> GEMM2 -> Finalize

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

    # --- Allocate workspace ---
    ws = allocate_moe_workspace(max_padded, H, I, device)

    # --- Stage 2: GEMM1 (A-gather + FP8->FP8) — vectorized v2 ---
    gemm1_out, gemm1_scale = moe_gemm1_fp8_v2(
        hidden_states,
        gemm1_weights,
        hidden_states_scale,
        gemm1_weights_scale,
        routing_result.m_indptr,
        routing_result.permuted_idx_to_token_idx,
        gemm1_out=ws.gemm1_out[:max_padded],
        gemm1_out_scale=ws.gemm1_scale[:, :max_padded],
    )

    # --- Stage 3: SwiGLU + FP8 Requantization ---
    act_out, act_scale = moe_swiglu_fp8_requant(
        gemm1_out,
        gemm1_scale,
        act_out=ws.act_out[:max_padded],
        act_scale=ws.act_scale[:, :max_padded],
    )

    # --- Stage 4: GEMM2 (FP8->BF16) — vectorized v2 ---
    gemm2_out = moe_gemm2_fp8_v2(
        act_out,
        gemm2_weights,
        act_scale,
        gemm2_weights_scale,
        routing_result.m_indptr,
        gemm2_out=ws.gemm2_out[:max_padded],
    )

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
