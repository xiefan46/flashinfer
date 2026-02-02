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

SwiGLU + FP8 Requantization for MoE Pipeline (CuTeDSL Kernel)
==============================================================

Single fused GPU kernel that takes GEMM1 FP8 output, applies SwiGLU activation,
and requantizes to FP8 with per-128-block float32 scales.

Replaces a ~6 kernel PyTorch implementation with one CuTeDSL kernel.

Pipeline position: GEMM1 output -> [this kernel] -> GEMM2 input

Input:  gemm1_out [padded, 2*I] FP8, gemm1_scale [2I//128, padded] f32
Output: act_out [padded, I] FP8, act_scale [I//128, padded] f32
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint8

from ..api_logging import flashinfer_api
from .fp4_common import (
    FLOAT8_E4M3_MAX,
    cvt_e4m3_to_f32,
    cvt_f32_to_e4m3,
    fabs_f32,
    fmax_f32,
    rcp_approx_ftz,
    silu_f32,
    warp_reduce,
)

# FP8 E4M3 max representable value (for PyTorch reference)
_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0

# =============================================================================
# Part 1: CuTeDSL Kernel Class
# =============================================================================


class MoESwiGLUFP8RequantKernel:
    """Fused SwiGLU + FP8 requantization kernel.

    Grid:  (I // 128, padded, 1)
    Block: (128, 1, 1)

    Each block of 128 threads processes one 128-element output tile:
    - Reads gate[row, blk*128 .. blk*128+127] and up[row, blk*128 .. blk*128+127]
    - Dequantizes from FP8 using per-128-block scales
    - Applies SwiGLU: silu(up) * gate
    - Finds block max via warp shuffle + smem reduction
    - Requantizes to FP8 and writes output + scale
    """

    BLOCK_SIZE = 128
    NUM_WARPS = 4  # 128 threads / 32 threads per warp

    def __init__(self, I: int):
        self.I = I
        self.num_out_blocks = I // self.BLOCK_SIZE
        assert I % self.BLOCK_SIZE == 0

    @cute.jit
    def __call__(
        self,
        mGemm1Out: cute.Tensor,
        mGemm1Scale: cute.Tensor,
        mActOut: cute.Tensor,
        mActScale: cute.Tensor,
        padded: Int32,
        stream,
    ):
        self.kernel(mGemm1Out, mGemm1Scale, mActOut, mActScale, padded).launch(
            grid=[self.num_out_blocks, padded, 1],
            block=[self.BLOCK_SIZE, 1, 1],
            smem=self.NUM_WARPS * 4,  # 4 floats for warp reduction
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mGemm1Out: cute.Tensor,
        mGemm1Scale: cute.Tensor,
        mActOut: cute.Tensor,
        mActScale: cute.Tensor,
        padded: Int32,
    ):
        tid = cute.arch.thread_idx()[0]
        blk_x = cute.arch.block_idx()[0]  # output block index [0, I//128)
        blk_y = cute.arch.block_idx()[1]  # row index [0, padded)

        I = self.I
        BLOCK = self.BLOCK_SIZE

        # Precompute constant: 1/448
        fp8_max_rcp = rcp_approx_ftz(Float32(FLOAT8_E4M3_MAX))

        out_col = blk_x * BLOCK + tid

        # --- Allocate shared memory for warp reduction ---
        smem = cutlass.utils.SmemAllocator()
        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.NUM_WARPS,)),
            byte_alignment=4,
        )

        # --- Read gate (first half) and up (second half) as FP8 bytes ---
        # gemm1_out is [padded, 2*I] stored as Uint8 (FP8 E4M3)
        gate_fp8 = mGemm1Out[blk_y, out_col]
        up_fp8 = mGemm1Out[blk_y, I + out_col]

        # --- Read scales (MN-major: [2*I//128, padded]) ---
        # gate scale: block index = blk_x, row = blk_y
        gate_scale = mGemm1Scale[blk_x, blk_y]
        # up scale: block index = (I//128) + blk_x, row = blk_y
        up_scale = mGemm1Scale[self.num_out_blocks + blk_x, blk_y]

        # --- Dequantize FP8 -> f32 ---
        gate_f32 = cvt_e4m3_to_f32(gate_fp8) * gate_scale
        up_f32 = cvt_e4m3_to_f32(up_fp8) * up_scale

        # --- SwiGLU: silu(up) * gate (DeepSeek convention) ---
        y_f32 = silu_f32(up_f32) * gate_f32

        # --- Block-level max(|y|) reduction ---
        abs_y = fabs_f32(y_f32)

        # Warp-level reduction first
        warp_max = warp_reduce(abs_y, cute.arch.fmax)

        # Cross-warp reduction via shared memory
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.warp_idx()

        if lane_idx == 0:
            reduction_buffer[warp_idx] = warp_max

        cute.arch.barrier()

        # Thread 0 of each warp reads all warp maxes; only need result in lane 0
        block_max = Float32(0.0)
        if lane_idx < self.NUM_WARPS:
            block_max = reduction_buffer[lane_idx]
        block_max = warp_reduce(block_max, cute.arch.fmax)

        # --- Compute output scale ---
        # scale = max(block_max / 448.0, 1e-12)
        out_scale = fmax_f32(block_max * fp8_max_rcp, Float32(1e-12))

        # --- Requantize to FP8 ---
        inv_scale = rcp_approx_ftz(out_scale)
        y_scaled = y_f32 * inv_scale
        y_e4m3_u32 = cvt_f32_to_e4m3(y_scaled)
        y_fp8 = Uint8(y_e4m3_u32 & cutlass.Uint32(0xFF))

        # --- Write output ---
        mActOut[blk_y, out_col] = y_fp8

        # --- Write scale (only thread 0) ---
        if tid == 0:
            mActScale[blk_x, blk_y] = out_scale


# =============================================================================
# Part 2: Compilation Cache
# =============================================================================


@functools.cache
def _get_compiled_swiglu_kernel(I: int):
    """Get compiled SwiGLU+FP8 requant kernel for given intermediate size I."""
    kernel_obj = MoESwiGLUFP8RequantKernel(I=I)

    sym_padded = cute.sym_int()

    # gemm1_out: [padded, 2*I] as Uint8 (FP8 E4M3 bytes)
    gemm1_out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_padded, 2 * I), stride_order=(1, 0), assumed_align=128
    )
    # gemm1_scale: [2*I//128, padded] as Float32 (MN-major)
    gemm1_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (2 * I // 128, sym_padded), stride_order=(1, 0), assumed_align=4
    )
    # act_out: [padded, I] as Uint8 (FP8 E4M3 bytes)
    act_out_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_padded, I), stride_order=(1, 0), assumed_align=128
    )
    # act_scale: [I//128, padded] as Float32 (MN-major)
    act_scale_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (I // 128, sym_padded), stride_order=(1, 0), assumed_align=4
    )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled = cute.compile(
        kernel_obj,
        gemm1_out_fake,
        gemm1_scale_fake,
        act_out_fake,
        act_scale_fake,
        Int32(1),  # dummy padded
        stream_fake,
        options="--enable-tvm-ffi",
    )

    def tensor_api(
        gemm1_out: torch.Tensor,
        gemm1_scale: torch.Tensor,
        act_out: torch.Tensor,
        act_scale: torch.Tensor,
        padded: int,
    ) -> None:
        compiled(
            gemm1_out.view(torch.uint8),
            gemm1_scale,
            act_out.view(torch.uint8),
            act_scale,
            Int32(padded),
        )

    return tensor_api


# =============================================================================
# Part 3: Public API
# =============================================================================


@flashinfer_api
def moe_swiglu_fp8_requant(
    gemm1_out: torch.Tensor,
    gemm1_scale: torch.Tensor,
    act_out: torch.Tensor = None,
    act_scale: torch.Tensor = None,
) -> tuple:
    """SwiGLU activation + FP8 requantization.

    Args:
        gemm1_out: [padded, 2*I] float8_e4m3fn — GEMM1 output.
        gemm1_scale: [2*I//128, padded] float32 — GEMM1 output block scales (MN-major).
        act_out: [padded, I] float8_e4m3fn — pre-allocated output (optional).
        act_scale: [I//128, padded] float32 — pre-allocated output scales (optional).

    Returns:
        (act_out, act_scale) tuple.
    """
    padded, two_I = gemm1_out.shape
    assert two_I % 2 == 0, f"GEMM1 output dim must be even, got {two_I}"
    I = two_I // 2
    BLOCK = 128
    assert I % BLOCK == 0, f"intermediate_size must be multiple of {BLOCK}"
    device = gemm1_out.device

    num_out_blocks = I // BLOCK

    # Allocate outputs if not provided
    if act_out is None:
        act_out = torch.empty(padded, I, dtype=torch.float8_e4m3fn, device=device)
    if act_scale is None:
        act_scale = torch.empty(num_out_blocks, padded, dtype=torch.float32, device=device)

    # Get cached compiled kernel and run
    kernel = _get_compiled_swiglu_kernel(I)
    kernel(gemm1_out, gemm1_scale, act_out, act_scale, padded)

    return act_out, act_scale


def moe_swiglu_fp8_requant_reference(
    gemm1_out: torch.Tensor,
    gemm1_scale: torch.Tensor,
) -> tuple:
    """Pure Python reference for validation.

    Returns (y_f32, act_out_fp8, act_scale) where y_f32 is the exact SwiGLU
    output before requantization (for tolerance checking).
    """
    padded, two_I = gemm1_out.shape
    I = two_I // 2
    BLOCK = 128

    # Dequantize
    x_fp8_f32 = gemm1_out.to(torch.float32)
    scale_transposed = gemm1_scale.t().contiguous()  # [padded, 2I//128]
    scale_expanded = scale_transposed.repeat_interleave(BLOCK, dim=1)  # [padded, 2*I]
    x_f32 = x_fp8_f32 * scale_expanded

    # SwiGLU (DeepSeek-V3: silu on second half)
    gate = x_f32[:, :I]
    up = x_f32[:, I:]
    y_f32 = torch.nn.functional.silu(up) * gate

    # Requant
    act_out, act_scale = moe_swiglu_fp8_requant(gemm1_out, gemm1_scale)

    return y_f32, act_out, act_scale
