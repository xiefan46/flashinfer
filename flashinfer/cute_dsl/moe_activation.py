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

SwiGLU + FP8 Requantization for MoE Pipeline
=============================================

Element-wise kernel that takes GEMM1 FP8 output, applies SwiGLU activation,
and requantizes to FP8 with per-128-block float32 scales.

Pipeline position: GEMM1 output -> [this kernel] -> GEMM2 input

Input:  gemm1_out [padded, 2*I] FP8, gemm1_scale [2I//128, padded] f32
Output: act_out [padded, I] FP8, act_scale [I//128, padded] f32
"""

import torch

from ..api_logging import flashinfer_api

# FP8 E4M3 max representable value
_FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


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

    # Dequantize GEMM1 output: x_f32 = x_fp8 * scale
    # gemm1_scale is [2I//128, padded] (MN-major), need to expand to [padded, 2I]
    x_fp8_f32 = gemm1_out.to(torch.float32)  # [padded, 2*I]
    scale_transposed = gemm1_scale.t().contiguous()  # [padded, 2I//128]
    scale_expanded = scale_transposed.repeat_interleave(BLOCK, dim=1)  # [padded, 2*I]
    x_f32 = x_fp8_f32 * scale_expanded  # [padded, 2*I]

    # Split gate and up projections
    gate = x_f32[:, :I]  # [padded, I]
    up = x_f32[:, I:]  # [padded, I]

    # SwiGLU: silu(gate) * up
    y = torch.nn.functional.silu(gate) * up  # [padded, I]

    # Per-128-block FP8 requantization
    y_blocks = y.reshape(padded, num_out_blocks, BLOCK)  # [padded, num_out_blocks, BLOCK]
    block_amax = y_blocks.abs().amax(dim=2)  # [padded, num_out_blocks]
    block_scale = block_amax / _FP8_E4M3_MAX  # [padded, num_out_blocks]
    block_scale = torch.where(
        block_scale > 0, block_scale, torch.ones_like(block_scale)
    )

    # Quantize
    y_scaled = y_blocks / block_scale.unsqueeze(2)  # [padded, num_out_blocks, BLOCK]
    y_fp8 = y_scaled.reshape(padded, I).to(torch.float8_e4m3fn)

    act_out.copy_(y_fp8)
    # Store scale in MN-major format: [I//128, padded]
    act_scale.copy_(block_scale.t().contiguous())

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

    # SwiGLU
    gate = x_f32[:, :I]
    up = x_f32[:, I:]
    y_f32 = torch.nn.functional.silu(gate) * up

    # Requant
    act_out, act_scale = moe_swiglu_fp8_requant(gemm1_out, gemm1_scale)

    return y_f32, act_out, act_scale
