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

FP8 Grouped GEMM with Float32 Per-128-Block Scales
===================================================

Wraps group_gemm_fp8_nt_groupwise for MoE pipeline usage with:
- Per-expert variable M (masked_m)
- Float32 per-128-block scales (MN-major layout)
- Support for both BF16 and FP8 output types
- Optional A-gather (Phase 2): scatter indices fused into input A

Scale layout convention (MN-major, matching trtllm):
  a_scale: [K//128, total_M]  float32
  b_scale: [E, N//128, K//128]  float32

Pipeline positions:
  GEMM1: hidden_states[T, H] @ W13[E, 2I, H]^T -> [padded, 2I] FP8
  GEMM2: act_out[padded, I] @ W2[E, H, I]^T -> [padded, H] BF16
"""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..gemm import group_gemm_fp8_nt_groupwise
from .moe_grouped_gemm_cutedsl import moe_grouped_gemm_fp8_cutedsl


def _m_indptr_to_masked_m(m_indptr: torch.Tensor) -> torch.Tensor:
    """Convert m_indptr [E+1] to masked_m [E] (per-expert token counts)."""
    return (m_indptr[1:] - m_indptr[:-1]).to(torch.int32)


def _pad_m_indptr(m_indptr: torch.Tensor, pad_to: int = 4) -> torch.Tensor:
    """Pad m_indptr values to multiples of pad_to.

    group_gemm_fp8_nt_groupwise requires each element in m_indptr
    to be a multiple of 4.
    """
    if pad_to <= 1:
        return m_indptr
    # Recompute from deltas
    deltas = m_indptr[1:] - m_indptr[:-1]
    padded_deltas = ((deltas + pad_to - 1) // pad_to) * pad_to
    result = torch.zeros_like(m_indptr)
    result[1:] = torch.cumsum(padded_deltas, dim=0)
    return result


@flashinfer_api
def moe_grouped_gemm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: Optional[torch.Tensor] = None,
    use_a_gather: bool = False,
    gather_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 grouped GEMM for MoE pipeline.

    Computes: out[i] = dequant(a[i]) @ dequant(b[expert(i)])^T
    where expert assignment is determined by m_indptr.

    Args:
        a: [total_M, K] float8_e4m3fn — input activations (permuted by expert).
        b: [E, N, K] float8_e4m3fn — expert weights.
        a_scale: [K//128, total_M] float32 — activation block scales (MN-major).
        b_scale: [E, N//128, K//128] float32 — weight block scales (MN-major).
        m_indptr: [E+1] int32 — cumulative token count per expert.
            Each value must be a multiple of 4.
        out_dtype: output dtype (torch.bfloat16 or torch.float8_e4m3fn).
        out: optional pre-allocated output [total_M, N].
        use_a_gather: if True, use gather_indices to scatter A rows.
        gather_indices: [total_M] int32 — original token index for each row in A.
            Only used when use_a_gather=True.

    Returns:
        out: [total_M, N] with out_dtype
    """
    total_M = a.shape[0]
    E = b.shape[0]
    N = b.shape[1]
    K = b.shape[2]

    if use_a_gather:
        assert gather_indices is not None, "gather_indices required when use_a_gather=True"
        # Phase 2: A-gather — reorder A rows using scatter indices
        # This is done host-side before the GEMM call.
        # The actual A data has shape [T_original, K] and we need to
        # permute it to [total_M, K] using gather_indices.
        a_gathered = a[gather_indices.long()]
        # Also gather A scales: a_scale is [K//128, T_original], need [K//128, total_M]
        a_scale_gathered = a_scale[:, gather_indices.long()]
        return _run_gemm(
            a_gathered, b, a_scale_gathered, b_scale, m_indptr, out_dtype, out
        )

    return _run_gemm(a, b, a_scale, b_scale, m_indptr, out_dtype, out)


def _run_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
) -> torch.Tensor:
    """Internal GEMM dispatch using CuTeDSL kernel."""
    total_M = a.shape[0]
    N = b.shape[1]

    masked_m = _m_indptr_to_masked_m(m_indptr)

    if out is None:
        out = torch.empty(total_M, N, dtype=out_dtype, device=a.device)

    if out_dtype == torch.float8_e4m3fn:
        # For FP8 output (GEMM1), run GEMM to BF16 then quantize
        bf16_out = moe_grouped_gemm_fp8_cutedsl(
            a, b, a_scale, b_scale, masked_m,
            out_dtype=torch.bfloat16,
        )
        # Quantize BF16 output to FP8 with per-block scales
        out_fp8, out_scale = _quantize_output_fp8(bf16_out, block_size=128)
        return out_fp8, out_scale

    # Standard BF16 output path
    out = moe_grouped_gemm_fp8_cutedsl(
        a, b, a_scale, b_scale, masked_m,
        out_dtype=out_dtype,
        out=out,
    )
    return out


def _quantize_output_fp8(
    x_bf16: torch.Tensor, block_size: int = 128
) -> tuple:
    """Quantize BF16 tensor to FP8 with per-block float32 scales.

    Args:
        x_bf16: [M, N] bfloat16
        block_size: quantization block size (128)

    Returns:
        (x_fp8, scale): x_fp8 is [M, N] float8_e4m3fn,
                        scale is [N//block_size, M] float32 (MN-major)
    """
    M, N = x_bf16.shape
    assert N % block_size == 0
    num_blocks = N // block_size

    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    x_f32 = x_bf16.to(torch.float32)
    x_blocks = x_f32.reshape(M, num_blocks, block_size)

    block_amax = x_blocks.abs().amax(dim=2)  # [M, num_blocks]
    scale = block_amax / fp8_max
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    x_scaled = x_blocks / scale.unsqueeze(2)
    x_fp8 = x_scaled.reshape(M, N).to(torch.float8_e4m3fn)

    # MN-major: [N//128, M]
    scale_mn = scale.t().contiguous()

    return x_fp8, scale_mn


@flashinfer_api
def moe_gemm1_fp8(
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    permuted_idx_to_token_idx: torch.Tensor,
    gemm1_out: Optional[torch.Tensor] = None,
    gemm1_out_scale: Optional[torch.Tensor] = None,
) -> tuple:
    """GEMM1 for MoE: hidden_states @ W13^T -> FP8 output.

    Implements A-gather: uses permuted_idx_to_token_idx to scatter
    hidden_states rows into expert-grouped order.

    Args:
        hidden_states: [T, H] float8_e4m3fn
        gemm1_weights: [E, 2*I, H] float8_e4m3fn
        hidden_states_scale: [H//128, T] float32 (MN-major)
        gemm1_weights_scale: [E, 2*I//128, H//128] float32 (MN-major)
        m_indptr: [E+1] int32
        permuted_idx_to_token_idx: [max_padded] int32
        gemm1_out: optional [max_padded, 2*I] float8_e4m3fn
        gemm1_out_scale: optional [2*I//128, max_padded] float32

    Returns:
        (gemm1_out, gemm1_out_scale)
    """
    max_padded = int(m_indptr[-1].item())
    T, H = hidden_states.shape
    E = gemm1_weights.shape[0]
    two_I = gemm1_weights.shape[1]

    # Gather A using permuted indices (A-gather)
    valid_mask = permuted_idx_to_token_idx >= 0
    # Create gathered A: for invalid indices, use row 0 (will be masked out)
    safe_indices = permuted_idx_to_token_idx.clone()
    safe_indices[~valid_mask] = 0
    a_gathered = hidden_states[safe_indices.long()]  # [max_padded, H]

    # Gather A scales: hidden_states_scale is [H//128, T] -> [H//128, max_padded]
    a_scale_gathered = hidden_states_scale[:, safe_indices.long()]

    # b_scale for MN-major needs to be [E, N//128, K//128]
    # gemm1_weights_scale is already [E, 2I//128, H//128] which is [E, N//128, K//128]
    b_scale_mn = gemm1_weights_scale

    # Convert m_indptr to masked_m for CuTeDSL kernel
    masked_m = _m_indptr_to_masked_m(m_indptr)

    # Run grouped GEMM -> BF16 via CuTeDSL
    bf16_out = moe_grouped_gemm_fp8_cutedsl(
        a_gathered,
        gemm1_weights,
        a_scale_gathered,
        b_scale_mn,
        masked_m,
        out_dtype=torch.bfloat16,
    )

    # Quantize output to FP8
    gemm1_out_fp8, gemm1_out_sc = _quantize_output_fp8(bf16_out, block_size=128)

    if gemm1_out is not None:
        gemm1_out.copy_(gemm1_out_fp8)
    else:
        gemm1_out = gemm1_out_fp8

    if gemm1_out_scale is not None:
        gemm1_out_scale.copy_(gemm1_out_sc)
    else:
        gemm1_out_scale = gemm1_out_sc

    return gemm1_out, gemm1_out_scale


@flashinfer_api
def moe_gemm2_fp8(
    act_out: torch.Tensor,
    gemm2_weights: torch.Tensor,
    act_scale: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    gemm2_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GEMM2 for MoE: activation_output @ W2^T -> BF16.

    Args:
        act_out: [max_padded, I] float8_e4m3fn
        gemm2_weights: [E, H, I] float8_e4m3fn
        act_scale: [I//128, max_padded] float32 (MN-major)
        gemm2_weights_scale: [E, H//128, I//128] float32 (MN-major)
        m_indptr: [E+1] int32
        gemm2_out: optional [max_padded, H] bfloat16

    Returns:
        gemm2_out: [max_padded, H] bfloat16
    """
    # Convert m_indptr to masked_m for CuTeDSL kernel
    masked_m = _m_indptr_to_masked_m(m_indptr)

    gemm2_out = moe_grouped_gemm_fp8_cutedsl(
        act_out,
        gemm2_weights,
        act_scale,
        gemm2_weights_scale,
        masked_m,
        out_dtype=torch.bfloat16,
        out=gemm2_out,
    )
    return gemm2_out
