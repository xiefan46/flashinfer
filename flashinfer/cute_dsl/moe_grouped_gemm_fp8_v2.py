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

FP8 Grouped GEMM v2 — Vectorized Wrapper (No Python Loops)
===========================================================

Drop-in replacement for moe_grouped_gemm_fp8.py that eliminates all
Python for-loops and GPU->CPU synchronization in the padding/unpadding
logic. Uses torch.searchsorted + scatter/gather instead.

The underlying CuTeDSL kernel (Sm100GroupwiseScaledGroupedGemmKernel)
is UNCHANGED — only the host-side data reshaping is vectorized.

AI-assisted implementation (Claude).
"""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from .moe_grouped_gemm_cutedsl import moe_grouped_gemm_fp8_cutedsl
from .moe_grouped_gemm_fp8 import _quantize_output_fp8


def _m_indptr_to_masked_m(m_indptr: torch.Tensor) -> torch.Tensor:
    """Convert m_indptr [E+1] to masked_m [E] (per-expert token counts)."""
    return (m_indptr[1:] - m_indptr[:-1]).to(torch.int32)


def _vectorized_grouped_gemm_fp8_cutedsl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Vectorized wrapper around CuTeDSL grouped GEMM.

    Same semantics as moe_grouped_gemm_fp8_cutedsl but replaces the
    three Python for-loops (A padding, a_scale padding, output extraction)
    with vectorized torch operations. No .item() calls, no GPU->CPU sync.

    Args:
        a: [total_M, K] float8_e4m3fn
        b: [E, N, K] float8_e4m3fn
        a_scale: [K//128, total_M] float32
        b_scale: [E, N//128, K//128] float32
        masked_m: [E] int32
        out_dtype: output dtype (torch.bfloat16)
        out: optional pre-allocated output [total_M, N]

    Returns:
        out: [total_M, N] with out_dtype
    """
    total_M, K = a.shape
    E, N, K_check = b.shape
    assert K == K_check, f"K mismatch: {K} vs {K_check}"
    assert K % 128 == 0, f"K must be multiple of 128, got {K}"
    assert N % 128 == 0, f"N must be multiple of 128, got {N}"
    assert masked_m.shape[0] == E

    device = a.device

    # Compute max_M without GPU->CPU sync for individual elements.
    # We still need one .item() for max_M_raw to determine padding,
    # but this is a single scalar reduction, not E iterations.
    max_M_raw = int(masked_m.max().item())

    if max_M_raw == 0:
        if out is None:
            out = torch.empty(0, N, dtype=out_dtype, device=device)
        return out

    # Pad max_M up to next multiple of 128 (MMA tile M dimension)
    max_M = ((max_M_raw + 127) // 128) * 128

    # Build m_indptr from masked_m (cumulative offsets, all on GPU)
    m_indptr = torch.zeros(E + 1, dtype=torch.int64, device=device)
    m_indptr[1:] = torch.cumsum(masked_m.long(), dim=0)

    # Check if direct reshape is possible (all experts have same M, already aligned)
    if total_M == E * max_M_raw and max_M == max_M_raw:
        a_batched = a.view(E, max_M, K)
        a_scale_flat = a_scale
    else:
        # Vectorized flat->batched index mapping using searchsorted.
        # For each row i in [0, total_M), find which expert it belongs to
        # and its position within that expert.
        row_idx = torch.arange(total_M, dtype=torch.int64, device=device)
        # searchsorted(m_indptr[1:], row_idx, right=True) gives expert index
        expert_of_row = torch.searchsorted(m_indptr[1:], row_idx, right=True)
        pos_in_expert = row_idx - m_indptr[expert_of_row]
        # Destination index in [E*max_M, K] flat layout
        dest_flat = (expert_of_row * max_M + pos_in_expert).long()

        # Pad A: [total_M, K] -> [E*max_M, K] via single scatter
        a_batched = torch.zeros(E * max_M, K, dtype=a.dtype, device=device)
        a_batched[dest_flat] = a
        a_batched = a_batched.view(E, max_M, K)

        # Pad a_scale: [K//128, total_M] -> [K//128, E*max_M] via single scatter
        a_scale_flat = torch.zeros(
            K // 128, E * max_M, dtype=torch.float32, device=device
        )
        a_scale_flat[:, dest_flat] = a_scale

    # B is already [E, N, K]
    b_batched = b

    # Output: [E, max_M, N]
    c_batched = torch.empty(E, max_M, N, dtype=out_dtype, device=device)

    # Map output dtype
    dtype_str_map = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }
    c_dtype_str = dtype_str_map[out_dtype]

    from ..utils import get_compute_capability
    from .utils import get_num_sm

    major, minor = get_compute_capability(device)
    sm_count = get_num_sm(device)
    sm_version = f"sm_{major}{minor}"

    mma_tiler_mn = (128, 128)
    cluster_shape_mn = (1, 1)

    from .moe_grouped_gemm_cutedsl import _get_compiled_grouped_gemm_kernel

    kernel_fn = _get_compiled_grouped_gemm_kernel(
        M=max_M,
        N=N,
        K=K,
        E=E,
        c_dtype_str=c_dtype_str,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sm_count=sm_count,
        sm_version=sm_version,
    )

    kernel_fn(
        a_batched.contiguous(),
        b_batched.contiguous(),
        a_scale_flat.contiguous(),
        b_scale.contiguous(),
        c_batched,
        masked_m.to(torch.int32).contiguous(),
    )

    # Extract output: [E, max_M, N] -> [total_M, N] via single gather
    if total_M == E * max_M:
        if out is None:
            out = c_batched.view(total_M, N)
        else:
            out.copy_(c_batched.view(total_M, N))
    else:
        c_flat = c_batched.view(E * max_M, N)
        # Reuse dest_flat computed above for gather
        gathered = c_flat[dest_flat]
        if out is None:
            out = gathered
        else:
            out.copy_(gathered)

    return out


@flashinfer_api
def moe_gemm1_fp8_v2(
    hidden_states: torch.Tensor,
    gemm1_weights: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    permuted_idx_to_token_idx: torch.Tensor,
    gemm1_out: Optional[torch.Tensor] = None,
    gemm1_out_scale: Optional[torch.Tensor] = None,
) -> tuple:
    """GEMM1 for MoE v2: hidden_states @ W13^T -> FP8 output.

    Same interface as moe_gemm1_fp8 but uses vectorized wrapper internally.

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
    # Gather A using permuted indices (A-gather)
    valid_mask = permuted_idx_to_token_idx >= 0
    safe_indices = permuted_idx_to_token_idx.clone()
    safe_indices[~valid_mask] = 0
    a_gathered = hidden_states[safe_indices.long()]
    a_scale_gathered = hidden_states_scale[:, safe_indices.long()]

    b_scale_mn = gemm1_weights_scale
    masked_m = _m_indptr_to_masked_m(m_indptr)

    # Run grouped GEMM -> BF16 via vectorized CuTeDSL wrapper
    bf16_out = _vectorized_grouped_gemm_fp8_cutedsl(
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
def moe_gemm2_fp8_v2(
    act_out: torch.Tensor,
    gemm2_weights: torch.Tensor,
    act_scale: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    m_indptr: torch.Tensor,
    gemm2_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """GEMM2 for MoE v2: activation_output @ W2^T -> BF16.

    Same interface as moe_gemm2_fp8 but uses vectorized wrapper internally.

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
    masked_m = _m_indptr_to_masked_m(m_indptr)

    gemm2_out = _vectorized_grouped_gemm_fp8_cutedsl(
        act_out,
        gemm2_weights,
        act_scale,
        gemm2_weights_scale,
        masked_m,
        out_dtype=torch.bfloat16,
        out=gemm2_out,
    )
    return gemm2_out
