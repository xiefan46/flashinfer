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

MoE Finalize: Gather + Weighted Reduce (CuTeDSL Kernel)
========================================================

Gathers expert outputs by token and computes weighted sum:
  output[t, :] = sum_{k=0}^{top_k-1} expert_weights[t,k] * gemm2_out[permuted_idx[t*top_k+k], :]

Pipeline position: GEMM2 output -> [this kernel] -> final MoE output

CuTeDSL kernel design:
  Grid:  (T, 1, 1)   — one CTA per token
  Block: (256, 1, 1)  — 256 threads cooperatively process the H dimension

  Each thread accumulates top_k weighted rows into f32, then writes BF16 output.
  BF16 tensors are viewed as Uint32 (bfloat16x2 pairs) with H//2 columns.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32

from ..api_logging import flashinfer_api
from .fp4_common import (
    cvt_bf16_to_f32,
    cvt_f32_to_bf16,
    pack_two_bf16,
    unpack_bf16_hi,
    unpack_bf16_lo,
)


# =============================================================================
# Part 1: CuTeDSL Kernel Class
# =============================================================================


class MoEFinalizeKernel:
    """Gather + weighted reduce kernel for MoE finalize.

    Grid:  (T, 1, 1)   — one CTA per token
    Block: (256, 1, 1)  — 256 threads process H//2 Uint32 columns

    Each thread handles COLS_PER_THREAD = ceil(H/2 / 256) bfloat16x2 pairs.
    Adjacent threads read adjacent columns for coalesced memory access.

    For H=7168: H//2 = 3584 Uint32 columns, COLS_PER_THREAD = 14.
    Each Uint32 holds 2 BF16 values → 28 BF16 elements per thread.
    """

    BLOCK_SIZE = 256

    def __init__(self, H: int, top_k: int):
        assert H % 2 == 0, f"H must be even, got {H}"
        self.H = H
        self.H_half = H // 2  # Number of Uint32 columns
        self.top_k = top_k
        # Ceiling division for columns per thread
        self.cols_per_thread = (self.H_half + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

    @cute.jit
    def __call__(
        self,
        mGemm2Out: cute.Tensor,
        mExpertWeights: cute.Tensor,
        mTopkIndices: cute.Tensor,
        mExpandedIdxToPermutedIdx: cute.Tensor,
        num_tokens: Int32,
        num_local_experts: Int32,
        local_expert_offset: Int32,
        mOutput: cute.Tensor,
        stream,
    ):
        self.kernel(
            mGemm2Out,
            mExpertWeights,
            mTopkIndices,
            mExpandedIdxToPermutedIdx,
            num_tokens,
            num_local_experts,
            local_expert_offset,
            mOutput,
        ).launch(
            grid=[num_tokens, 1, 1],
            block=[self.BLOCK_SIZE, 1, 1],
            smem=0,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mGemm2Out: cute.Tensor,
        mExpertWeights: cute.Tensor,
        mTopkIndices: cute.Tensor,
        mExpandedIdxToPermutedIdx: cute.Tensor,
        num_tokens: Int32,
        num_local_experts: Int32,
        local_expert_offset: Int32,
        mOutput: cute.Tensor,
    ):
        tid = cute.arch.thread_idx()[0]
        t = cute.arch.block_idx()[0]  # token index

        H_half = self.H_half
        top_k = self.top_k
        COLS_PER_THREAD = self.cols_per_thread
        BLOCK = self.BLOCK_SIZE

        local_end = local_expert_offset + num_local_experts

        # f32 accumulators for each bfloat16x2 pair (lo and hi)
        acc_lo = cute.make_rmem_tensor((COLS_PER_THREAD,), Float32)
        acc_hi = cute.make_rmem_tensor((COLS_PER_THREAD,), Float32)
        for c in cutlass.range_constexpr(COLS_PER_THREAD):
            acc_lo[c] = Float32(0.0)
            acc_hi[c] = Float32(0.0)

        # Loop over top_k experts
        for k in cutlass.range_constexpr(top_k):
            expanded_idx = t * top_k + k

            perm_idx = mExpandedIdxToPermutedIdx[expanded_idx]
            # Skip invalid entries (perm_idx < 0)
            if perm_idx >= Int32(0):
                global_e = mTopkIndices[expanded_idx]
                # Skip non-local experts
                if global_e >= local_expert_offset:
                    if global_e < local_end:
                        w = mExpertWeights[expanded_idx]

                        # Gather and accumulate
                        for c in cutlass.range_constexpr(COLS_PER_THREAD):
                            col = tid + c * BLOCK
                            if col < H_half:
                                packed = mGemm2Out[perm_idx, col]
                                lo_bf16 = unpack_bf16_lo(packed)
                                hi_bf16 = unpack_bf16_hi(packed)
                                lo_f32 = cvt_bf16_to_f32(lo_bf16)
                                hi_f32 = cvt_bf16_to_f32(hi_bf16)
                                acc_lo[c] = acc_lo[c] + w * lo_f32
                                acc_hi[c] = acc_hi[c] + w * hi_f32

        # Write output as bfloat16x2
        for c in cutlass.range_constexpr(COLS_PER_THREAD):
            col = tid + c * BLOCK
            if col < H_half:
                out_lo = cvt_f32_to_bf16(acc_lo[c])
                out_hi = cvt_f32_to_bf16(acc_hi[c])
                mOutput[t, col] = pack_two_bf16(out_lo, out_hi)


# =============================================================================
# Part 2: Compilation Cache
# =============================================================================


@functools.cache
def _get_compiled_finalize_kernel(H: int, top_k: int):
    """Get compiled MoE finalize kernel for given (H, top_k)."""
    kernel_obj = MoEFinalizeKernel(H=H, top_k=top_k)

    sym_T = cute.sym_int()
    sym_padded = cute.sym_int()
    H_half = H // 2

    # gemm2_out: [padded, H//2] as Uint32 (bfloat16x2 pairs)
    gemm2_out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint32, (sym_padded, H_half), stride_order=(1, 0), assumed_align=128
    )
    # expert_weights: [T*top_k] as Float32 (flattened)
    expert_weights_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (sym_T * top_k,), stride_order=(0,), assumed_align=4
    )
    # topk_indices: [T*top_k] as Int32 (flattened, converted from i64)
    topk_indices_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_T * top_k,), stride_order=(0,), assumed_align=4
    )
    # expanded_idx_to_permuted_idx: [T*top_k] as Int32
    expanded_perm_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_T * top_k,), stride_order=(0,), assumed_align=4
    )
    # output: [T, H//2] as Uint32 (bfloat16x2 pairs)
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint32, (sym_T, H_half), stride_order=(1, 0), assumed_align=128
    )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled = cute.compile(
        kernel_obj,
        gemm2_out_fake,
        expert_weights_fake,
        topk_indices_fake,
        expanded_perm_fake,
        Int32(1),  # dummy num_tokens
        Int32(1),  # dummy num_local_experts
        Int32(0),  # dummy local_expert_offset
        output_fake,
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        gemm2_out: torch.Tensor,
        expert_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        num_tokens: int,
        num_local_experts: int,
        local_expert_offset: int,
        output: torch.Tensor,
    ) -> None:
        compiled(
            gemm2_out.view(torch.int32),  # BF16 -> view as int32 (2 bf16 per i32)
            expert_weights,
            topk_indices,
            expanded_idx_to_permuted_idx,
            Int32(num_tokens),
            Int32(num_local_experts),
            Int32(local_expert_offset),
            output.view(torch.int32),  # BF16 -> view as int32
        )

    return tensor_api


# =============================================================================
# Part 3: Public API
# =============================================================================


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
    """Finalize MoE: gather expert outputs and weighted reduce (CuTeDSL kernel).

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
    H = hidden_size
    device = gemm2_out.device

    assert H % 2 == 0, f"hidden_size must be even, got {H}"

    if output is None:
        output = torch.zeros(T, H, dtype=torch.bfloat16, device=device)
    else:
        output.zero_()

    if T == 0:
        return output

    # Convert topk_indices from i64 to i32 (expert IDs fit in i32)
    topk_indices_i32 = topk_indices.to(torch.int32)

    # Flatten expert_weights and topk_indices to 1D for kernel
    expert_weights_flat = expert_weights.contiguous().view(-1)
    topk_indices_flat = topk_indices_i32.contiguous().view(-1)

    kernel = _get_compiled_finalize_kernel(H, top_k)
    kernel(
        gemm2_out,
        expert_weights_flat,
        topk_indices_flat,
        expanded_idx_to_permuted_idx,
        T,
        num_local_experts,
        local_expert_offset,
        output,
    )

    return output


# =============================================================================
# Reference Implementation (for testing only)
# =============================================================================


def moe_finalize_reference(
    gemm2_out: torch.Tensor,
    expert_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    num_local_experts: int,
    local_expert_offset: int,
    hidden_size: int,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Reference finalize using Python double loop (for testing only).

    Same interface as moe_finalize.
    """
    T, top_k = expert_weights.shape
    device = gemm2_out.device
    local_end = local_expert_offset + num_local_experts

    if output is None:
        output = torch.zeros(T, hidden_size, dtype=torch.bfloat16, device=device)
    else:
        output.zero_()

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
