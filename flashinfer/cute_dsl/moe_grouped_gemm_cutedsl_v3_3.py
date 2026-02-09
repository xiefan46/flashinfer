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

CuTeDSL FP8 Flat Grouped GEMM v3.3 — Cluster Shape (2,1) Optimization
======================================================================

Based on v3.2 with cluster_shape_mn=(2,1) for TMA multicast optimization.
With 2×1 clustering, B matrix loads are shared across 2 CTAs processing
different M tiles, reducing memory bandwidth by ~1.5x for B operand.

v3.2: cluster_shape_mn=(1,1), no TMA multicast
v3.3: cluster_shape_mn=(2,1), B multicast across 2 CTAs in M dimension

AI-assisted implementation (Claude).
"""

import functools
from typing import Callable, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Float32, Uint8
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import Int32

from ..api_logging import flashinfer_api
from ..utils import get_compute_capability
from .blockscaled_gemm import MaskedScheduler, MaskedSchedulerParams
from .fp4_common import (
    FLOAT8_E4M3_MAX,
    cvt_f32_to_e4m3,
    fabs_f32,
    fmax_f32,
    rcp_approx_ftz,
)
from .utils import get_num_sm, make_ptr

# =============================================================================
# Part 1: Kernel Class
# =============================================================================


class Sm100FlatGroupedGemmFP8OutputKernel:
    """Persistent flat grouped GEMM with direct FP8 output + per-128-block scales.

    Based on Sm100FlatGroupedGemmKernel (v3) with modified epilogue:
    - No TMA store for C (no C SMEM needed)
    - F32 accumulator → per-128-block absmax → scale → FP8 quantize → global store
    - Each thread independently quantizes its M row's N elements (no cross-thread reduction)

    Architecture: Blackwell SM100/SM103
    MMA: tcgen05 dense (no hardware block scales)
    Scales: float32 per-128-block, applied per-K-tile in software
    Scheduling: Persistent tile with MaskedScheduler (per-expert masked M)
    """

    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        sm_version: str,
    ):
        supported_sm_versions = ["sm_100", "sm_103"]
        assert sm_version in supported_sm_versions, (
            f"Only {supported_sm_versions} supported, got {sm_version}"
        )

        self.acc_dtype = cutlass.Float32
        # Internal c_dtype is BF16 for epi_tile computation (T2R partition shape)
        self.c_dtype_init = cutlass.BFloat16
        self.use_2cta_instrs = False
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.smem_capacity = utils.get_smem_capacity_in_bytes(sm_version)

    def _setup_attributes(self):
        """Set up configurations dependent on GEMM inputs."""
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Epi tile still needed for TMEM→register partition shape
        self.use_tma_store = False
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Stage counts — no C SMEM (FP8 direct global store)
        self.num_acc_stage = 1
        self.num_ab_stage = self._compute_ab_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage,
        )
        # No c_smem_layout_staged — direct FP8 output

        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,        # logical (M_total, K, 1)
        b: cute.Tensor,        # logical (N, K, L)
        a_scale: cute.Tensor,  # (K//128, M_total) float32
        b_scale: cute.Tensor,  # (L, N//128, K//128) float32
        c_dummy: cute.Tensor,  # logical (M_total, N, 1) BF16 — for partition shape only
        c_fp8: cute.Tensor,    # (M_total, N) Uint8 — FP8 output
        c_scale: cute.Tensor,  # (N//128, M_total) float32 — output scales
        c_sched: cute.Tensor,  # logical (M_dummy, N, L) for MaskedScheduler
        masked_m: cute.Tensor, # (L,) int32
        m_indptr_tiles: cute.Tensor,  # (L+1,) int32
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Execute the flat grouped GEMM with direct FP8 output."""
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_dummy.element_type  # BF16
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_dummy)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode,
            self.acc_dtype, self.cta_group, self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA load A
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # No TMA store C — direct FP8 global store

        # Grid via MaskedScheduler
        self.tile_sched_params, grid = self._compute_grid(
            masked_m, c_sched, self.cta_tile_shape_mnk,
            self.cluster_shape_mn, max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # No sC — FP8 output goes directly to GMEM

        self.shared_storage = SharedStorage

        # Launch kernel
        self.kernel(
            tiled_mma,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            c_dummy,   # BF16 for T2R partition shape
            c_fp8,     # Uint8 for FP8 output
            c_scale,   # Float32 for output scales
            a_scale, b_scale,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            m_indptr_tiles,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,    # BF16 dummy for T2R partition shape
        mC_fp8: cute.Tensor,    # Uint8 (M_total, N) — FP8 output
        mCScale: cute.Tensor,   # Float32 (N//128, M_total) — output scales
        mAScale: cute.Tensor,   # (K//128, M_total) float32
        mBScale: cute.Tensor,   # (E, N//128, K//128) float32
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: MaskedSchedulerParams,
        m_indptr_tiles: cute.Tensor,
    ):
        """GPU kernel: persistent flat grouped GEMM with FP8 output epilogue."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Prefetch TMA descriptors (A and B only, no C TMA)
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # Shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        tmem_holding_buf = storage.tmem_holding_buf

        # AB pipeline
        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # ACC pipeline (1 stage, cycled per-K-tile)
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        # Cluster barrier sync
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_arrive_relaxed()
        cute.arch.mbarrier_init_fence()

        # SMEM tensors (A and B only)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )

        # Multicast masks
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # Local tile partition for A, B, C (C uses dummy BF16 tensor for shape)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # MMA fragment partitioning
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # TMA partition A
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )

        # TMA partition B
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # MMA fragments
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        # Cluster sync
        if cute.size(self.cluster_shape_mn) > 1:
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier(
                barrier_id=self.cta_sync_bar_id,
                number_of_threads=self.threads_per_cta,
            )

        # ===================================================================
        # TMA warp: Load A, B from GMEM → SMEM (identical to v3)
        # ===================================================================
        if warp_idx == self.tma_warp_id:
            tile_sched = MaskedScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                expert_idx = mma_tile_coord_mnl[2]
                local_m_tile = mma_tile_coord_mnl[0]
                global_m_tile = m_indptr_tiles[expert_idx] + local_m_tile

                tAgA_slice = tAgA[
                    (None, global_m_tile, None, 0)
                ]
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(
                            ab_producer_state
                        ),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(
                            ab_producer_state
                        ),
                        mcast_mask=b_full_mcast_mask,
                    )
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                tile_sched.advance_to_next_work()
                work_tile, _ = tile_sched.get_current_work()

            ab_pipeline.producer_tail(ab_producer_state)

        # ===================================================================
        # MMA warp: Compute per-K-tile partials (identical to v3)
        # ===================================================================
        if warp_idx == self.mma_warp_id:
            tmem_ptr_read_threads = 32 * len(
                (self.mma_warp_id, *self.epilog_warp_id)
            )
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            tile_sched = MaskedScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                ab_consumer_state.reset_count()

                for k_tile in range(k_tile_cnt):
                    tCtAcc = tCtAcc_base[(None, None, None, 0)]

                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state)

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    if is_leader_cta:
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(
                            num_kblocks, unroll_full=True
                        ):
                            kblock_coord = (
                                None, None, kblock_idx,
                                ab_consumer_state.index,
                            )
                            cute.gemm(
                                tiled_mma, tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()

                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile, _ = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # ===================================================================
        # Epilogue warps: TMEM→register, scale, accumulate, FP8 quantize+store
        # v3.2: replaces BF16 TMA store with direct FP8 global store
        # ===================================================================
        if warp_idx < self.mma_warp_id:
            # Alloc tensor memory
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.alloc_tmem(
                    self.num_tmem_alloc_cols,
                    tmem_holding_buf,
                    is_two_cta=use_2cta_instrs,
                )
            tmem_ptr_read_threads = 32 * len(
                (self.mma_warp_id, *self.epilog_warp_id)
            )
            cute.arch.barrier(
                barrier_id=self.tmem_ptr_sync_bar_id,
                number_of_threads=tmem_ptr_read_threads,
            )
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # Setup T2R epilogue copy (same as v3, but no R2S or S2G)
            epi_tidx = tidx
            (
                tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )

            # Compute subtile count from TMEM partition layout
            tTR_tAcc_staged0 = tTR_tAcc_base[
                (None, None, None, None, None, 0)
            ]
            tTR_tAcc_grouped_init = cute.group_modes(
                tTR_tAcc_staged0, 3, cute.rank(tTR_tAcc_staged0)
            )
            subtile_cnt = cute.size(
                tTR_tAcc_grouped_init.shape, mode=[3]
            )
            subtile_size = cute.size(tTR_rAcc)

            # Allocate register accumulators for ALL epi subtiles
            total_accum_elems = subtile_size * subtile_cnt
            tTR_rAcc_accum = cute.make_fragment(
                cute.make_layout((total_accum_elems,)), self.acc_dtype
            )

            # FP8 max reciprocal (precomputed constant)
            fp8_max_rcp = rcp_approx_ftz(Float32(FLOAT8_E4M3_MAX))

            # Persistent tile loop
            tile_sched = MaskedScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            epilog_threads = 32 * len(self.epilog_warp_id)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                expert_idx = mma_tile_coord_mnl[2]
                n_tile_idx = mma_tile_coord_mnl[1]
                local_m_tile = mma_tile_coord_mnl[0]
                global_m_tile = m_indptr_tiles[expert_idx] + local_m_tile

                # Zero the register accumulators
                tTR_rAcc_accum.fill(cutlass.Float32(0.0))

                # M coordinate for this thread
                m_tile_start = global_m_tile * Int32(self.mma_tiler_mn[0])
                m_flat = m_tile_start + Int32(epi_tidx)

                # --- Per-K-tile accumulation with scale application (same as v3) ---
                for k_tile in range(k_tile_cnt):
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, 0)
                    ]

                    acc_pipeline.consumer_wait(acc_consumer_state)

                    b_scale_val = mBScale[expert_idx, n_tile_idx, k_tile]
                    a_scale_val = mAScale[k_tile, m_flat]
                    combined_scale = a_scale_val * b_scale_val

                    tTR_tAcc_grouped = cute.group_modes(
                        tTR_tAcc, 3, cute.rank(tTR_tAcc)
                    )
                    for subtile_idx in cutlass.range(subtile_cnt):
                        tTR_tAcc_mn = tTR_tAcc_grouped[
                            (None, None, None, subtile_idx)
                        ]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        accum_base = subtile_idx * subtile_size
                        for i in cutlass.range(subtile_size):
                            val = tTR_rAcc[i]
                            acc_idx = accum_base + i
                            tTR_rAcc_accum[acc_idx] = (
                                tTR_rAcc_accum[acc_idx] + val * combined_scale
                            )

                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                # --- v3.2: FP8 quantize and direct global store ---
                # Pass 1: Compute per-128-block absmax over all N elements
                # (N tile = 128 = one quantization block, no cross-thread reduction)
                block_absmax = Float32(0.0)
                for subtile_idx in cutlass.range(subtile_cnt):
                    accum_base = subtile_idx * subtile_size
                    for i in cutlass.range(subtile_size):
                        block_absmax = fmax_f32(
                            block_absmax,
                            fabs_f32(tTR_rAcc_accum[accum_base + i]),
                        )

                # Compute scale: max(absmax / 448.0, 1e-12)
                out_scale = fmax_f32(
                    block_absmax * fp8_max_rcp, Float32(1e-12)
                )
                inv_scale = rcp_approx_ftz(out_scale)

                # Pass 2: Quantize F32 → FP8 and global store
                n_tile_start = n_tile_idx * Int32(self.mma_tiler_mn[1])
                for subtile_idx in cutlass.range(subtile_cnt):
                    accum_base = subtile_idx * subtile_size
                    for i in cutlass.range(subtile_size):
                        n_col = n_tile_start + Int32(
                            subtile_idx * subtile_size + i
                        )
                        val = tTR_rAcc_accum[accum_base + i]
                        fp8_val = cvt_f32_to_e4m3(val * inv_scale)
                        mC_fp8[m_flat, n_col] = Uint8(
                            fp8_val & cutlass.Uint32(0xFF)
                        )

                # Write scale: one per (n_block, m_row)
                mCScale[n_tile_idx, m_flat] = out_scale

                tile_sched.advance_to_next_work()
                work_tile, _ = tile_sched.get_current_work()

            # Dealloc tensor memory
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=use_2cta_instrs)
            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id,
                number_of_threads=epilog_threads,
            )
            if warp_idx == self.epilog_warp_id[0]:
                cute.arch.dealloc_tmem(
                    tmem_ptr, self.num_tmem_alloc_cols,
                    is_two_cta=use_2cta_instrs,
                )

    # --- Epilogue helper (T2R only, no R2S or S2G) ---

    def epilog_tmem_copy_and_partition(
        self, tidx, tAcc, gC_mnl, epi_tile, use_2cta_instrs,
    ):
        """TMEM → register copy setup (identical to v3)."""
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk, self.c_layout, self.c_dtype,
            self.acc_dtype, epi_tile, use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(
            tAcc[((None, None), 0, 0, None)], epi_tile,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_fragment(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    @staticmethod
    def _compute_ab_stages(
        tiled_mma, mma_tiler_mnk, a_dtype, b_dtype,
        smem_capacity, occupancy,
    ):
        """Compute AB stage count without C SMEM reservation."""
        a_smem_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1
        )
        b_smem_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_one)
            + cute.size_in_bytes(b_dtype, b_smem_one)
        )
        mbar_helpers_bytes = 1024

        num_ab_stage = (
            smem_capacity // occupancy - mbar_helpers_bytes
        ) // ab_bytes_per_stage

        return num_ab_stage

    @staticmethod
    def _compute_grid(
        masked_m_tensor, c_sched, cta_tile_shape_mnk, cluster_shape_mn,
        max_active_clusters,
    ):
        """Compute grid size using MaskedScheduler (identical to v3)."""
        c_tiler = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        tile_sched_params = MaskedSchedulerParams(
            masked_m_tensor, None, c_sched, c_tiler, cluster_shape_mnl
        )
        grid = MaskedScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma, mma_tiler, num_acc_stage):
        """Compute tensor memory allocation columns (identical to v3)."""
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, num_acc_stage)
        )
        return tcgen05.find_tmem_tensor_col_offset(tCtAcc_fake)


# =============================================================================
# Part 2: Compilation Cache and Host Wrapper
# =============================================================================


class GroupwiseScaledGroupedGemmCuteDSL_V3_2:
    """Host wrapper for flat grouped GEMM with FP8 output.

    Tensor format:
      A: physical [1, M_total, K], logical (M_total, K, 1)
      B: physical [E, N, K], logical (N, K, E)
      c_dummy: logical (M_total, N, 1) BF16 — for partition only
      c_fp8: physical [M_total, N] Uint8 — FP8 output
      c_scale: physical [N//128, M_total] Float32 — output scales
    """

    def __init__(
        self,
        M_total: int,
        N: int,
        K: int,
        E: int,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        sm_count: int,
        sm_version: str,
    ):
        self._M_total = M_total
        self._N = N
        self._K = K
        self._E = E
        self._mma_tiler_mn = mma_tiler_mn
        self._cluster_shape_mn = cluster_shape_mn

        hardware_info = cutlass.utils.HardwareInfo()
        self._max_active_clusters = min(
            hardware_info.get_max_active_clusters(
                cluster_shape_mn[0] * cluster_shape_mn[1]
            ),
            sm_count,
        )
        self._sm_version = sm_version

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,         # FP8
        b_ptr: cute.Pointer,         # FP8
        a_scale_ptr: cute.Pointer,   # F32
        b_scale_ptr: cute.Pointer,   # F32
        c_dummy_ptr: cute.Pointer,   # BF16 (for partition shape)
        c_fp8_ptr: cute.Pointer,     # Uint8 (FP8 output)
        c_scale_ptr: cute.Pointer,   # F32 (output scales)
        masked_m_ptr: cute.Pointer,  # Int32
        m_indptr_tiles_ptr: cute.Pointer,  # Int32
        c_sched_ptr: cute.Pointer,   # BF16 (dummy for scheduler)
        current_stream: cuda.CUstream,
    ):
        # A: logical (M_total, K, 1)
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout(
                (self._M_total, self._K, 1),
                order=(1, 0, 2),
            ),
        )
        # B: logical (N, K, E)
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (self._N, self._K, self._E),
                order=(1, 0, 2),
            ),
        )
        # c_dummy: logical (M_total, N, 1) BF16 — for T2R partition only
        c_dummy_tensor = cute.make_tensor(
            c_dummy_ptr,
            layout=cute.make_ordered_layout(
                (self._M_total, self._N, 1),
                order=(1, 0, 2),
            ),
        )
        # c_fp8: (M_total, N) Uint8 — FP8 output
        c_fp8_tensor = cute.make_tensor(
            c_fp8_ptr,
            layout=cute.make_ordered_layout(
                (self._M_total, self._N),
                order=(1, 0),
            ),
        )
        # c_scale: (N//128, M_total) Float32 — output scales
        c_scale_tensor = cute.make_tensor(
            c_scale_ptr,
            layout=cute.make_ordered_layout(
                (self._N // 128, self._M_total),
                order=(1, 0),
            ),
        )
        # c_sched: dummy 3D for MaskedScheduler
        c_sched = cute.make_tensor(
            c_sched_ptr,
            layout=cute.make_ordered_layout(
                (self._M_total, self._N, self._E),
                order=(1, 0, 2),
            ),
        )
        # a_scale: (K//128, M_total) float32
        a_scale_tensor = cute.make_tensor(
            a_scale_ptr,
            layout=cute.make_ordered_layout(
                (self._K // 128, self._M_total),
                order=(1, 0),
            ),
        )
        # b_scale: (E, N//128, K//128) float32
        b_scale_tensor = cute.make_tensor(
            b_scale_ptr,
            layout=cute.make_ordered_layout(
                (self._E, self._N // 128, self._K // 128),
                order=(2, 1, 0),
            ),
        )
        # masked_m: (E,) int32
        masked_m_tensor = cute.make_tensor(
            masked_m_ptr,
            layout=cute.make_ordered_layout((self._E,), order=(0,)),
        )
        # m_indptr_tiles: (E+1,) int32
        m_indptr_tiles_tensor = cute.make_tensor(
            m_indptr_tiles_ptr,
            layout=cute.make_ordered_layout((self._E + 1,), order=(0,)),
        )

        Sm100FlatGroupedGemmFP8OutputKernel(
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            sm_version=self._sm_version,
        )(
            a_tensor, b_tensor, a_scale_tensor, b_scale_tensor,
            c_dummy_tensor, c_fp8_tensor, c_scale_tensor,
            c_sched, masked_m_tensor, m_indptr_tiles_tensor,
            self._max_active_clusters,
            current_stream,
        )


@functools.cache
def _get_compiled_flat_grouped_gemm_fp8out_kernel(
    M_total: int,
    N: int,
    K: int,
    E: int,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
    sm_version: str,
) -> Callable:
    """Compile and cache the flat grouped GEMM kernel with FP8 output."""
    c_dtype = cutlass.BFloat16  # Internal layout computation only
    ab_dtype = cutlass.Float8E4M3FN

    def get_cute_pointers(input_tensors=None):
        if input_tensors is None:
            # a, b, a_scale, b_scale, c_dummy, c_fp8, c_scale,
            # masked_m, m_indptr_tiles, c_sched
            ptrs = [16] * 10
        else:
            # input_tensors: [a, b, a_scale, b_scale, c_fp8, c_scale,
            #                  masked_m, m_indptr_tiles]
            ptrs = [
                input_tensors[0].data_ptr(),   # a
                input_tensors[1].data_ptr(),   # b
                input_tensors[2].data_ptr(),   # a_scale
                input_tensors[3].data_ptr(),   # b_scale
                input_tensors[4].data_ptr(),   # c_dummy (reuse c_fp8 addr)
                input_tensors[4].data_ptr(),   # c_fp8
                input_tensors[5].data_ptr(),   # c_scale
                input_tensors[6].data_ptr(),   # masked_m
                input_tensors[7].data_ptr(),   # m_indptr_tiles
                16,                             # c_sched (dummy)
            ]

        return [
            make_ptr(ab_dtype, ptrs[0], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(ab_dtype, ptrs[1], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, ptrs[2], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, ptrs[3], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(c_dtype, ptrs[4], cute.AddressSpace.gmem, assumed_align=16),       # c_dummy BF16
            make_ptr(cutlass.Uint8, ptrs[5], cute.AddressSpace.gmem, assumed_align=16), # c_fp8
            make_ptr(cutlass.Float32, ptrs[6], cute.AddressSpace.gmem, assumed_align=16),  # c_scale
            make_ptr(cutlass.Int32, ptrs[7], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, ptrs[8], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(c_dtype, ptrs[9], cute.AddressSpace.gmem, assumed_align=16),       # c_sched BF16
        ]

    wrapper = GroupwiseScaledGroupedGemmCuteDSL_V3_2(
        M_total=M_total, N=N, K=K, E=E,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sm_count=sm_count,
        sm_version=sm_version,
    )

    kernel = cute.compile(
        wrapper,
        *get_cute_pointers(None),
        cutlass_torch.current_stream(),
    )

    def tensor_api(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scale: torch.Tensor,
        b_scale: torch.Tensor,
        c_fp8: torch.Tensor,
        c_scale: torch.Tensor,
        masked_m: torch.Tensor,
        m_indptr_tiles: torch.Tensor,
    ):
        nonlocal kernel
        current_stream = cutlass_torch.current_stream()
        kernel(
            *get_cute_pointers([a, b, a_scale, b_scale, c_fp8, c_scale, masked_m, m_indptr_tiles]),
            current_stream,
        )
        return c_fp8, c_scale

    return tensor_api


# =============================================================================
# Part 3: Public API
# =============================================================================


@flashinfer_api
def moe_grouped_gemm_fp8_cutedsl_v3_3(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    m_indptr_tiles: torch.Tensor,
    out_fp8: torch.Tensor = None,
    out_scale: torch.Tensor = None,
) -> tuple:
    """CuTeDSL flat grouped GEMM v3.3 with cluster_shape=(2,1) optimization.

    Based on v3.2 with 2×1 clustering for TMA multicast. B matrix loads
    are shared across 2 CTAs processing different M tiles.

    Args:
        a: [total_padded_M, K] float8_e4m3fn — flat padded input.
        b: [E, N, K] float8_e4m3fn — expert weights.
        a_scale: [K//128, total_padded_M] float32 — activation block scales.
        b_scale: [E, N//128, K//128] float32 — weight block scales.
        masked_m: [E] int32 — valid M per expert (unaligned).
        m_indptr_tiles: [E+1] int32 — expert boundaries in tile units.
        out_fp8: optional pre-allocated [total_padded_M, N] float8_e4m3fn.
        out_scale: optional pre-allocated [N//128, total_padded_M] float32.

    Returns:
        (out_fp8, out_scale): FP8 output and per-128-block scales.
    """
    total_padded_M, K = a.shape
    E, N, K_check = b.shape
    assert K == K_check, f"K mismatch: {K} vs {K_check}"
    assert K % 128 == 0, f"K must be multiple of 128, got {K}"
    assert N % 128 == 0, f"N must be multiple of 128, got {N}"
    assert masked_m.shape[0] == E
    assert m_indptr_tiles.shape[0] == E + 1

    device = a.device

    if total_padded_M == 0:
        if out_fp8 is None:
            out_fp8 = torch.empty(0, N, dtype=torch.float8_e4m3fn, device=device)
        if out_scale is None:
            out_scale = torch.empty(N // 128, 0, dtype=torch.float32, device=device)
        return out_fp8, out_scale

    if out_fp8 is None:
        out_fp8 = torch.empty(total_padded_M, N, dtype=torch.float8_e4m3fn, device=device)
    if out_scale is None:
        out_scale = torch.empty(N // 128, total_padded_M, dtype=torch.float32, device=device)

    major, minor = get_compute_capability(device)
    sm_count = get_num_sm(device)
    sm_version = f"sm_{major}{minor}"

    mma_tiler_mn = (128, 128)
    cluster_shape_mn = (2, 1)  # v3.3: B multicast across 2 CTAs

    kernel_fn = _get_compiled_flat_grouped_gemm_fp8out_kernel(
        M_total=total_padded_M,
        N=N,
        K=K,
        E=E,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=cluster_shape_mn,
        sm_count=sm_count,
        sm_version=sm_version,
    )

    kernel_fn(
        a.contiguous(),
        b.contiguous(),
        a_scale.contiguous(),
        b_scale.contiguous(),
        out_fp8.view(torch.uint8),
        out_scale,
        masked_m.to(torch.int32).contiguous(),
        m_indptr_tiles.to(torch.int32).contiguous(),
    )

    return out_fp8, out_scale
