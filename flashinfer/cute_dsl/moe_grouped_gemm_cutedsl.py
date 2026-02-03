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

CuTeDSL FP8 Grouped GEMM with Float32 Per-128-Block Scales
===========================================================

Persistent grouped GEMM kernel for MoE pipeline on Blackwell SM100/SM103.

Architecture:
  - MMA: tcgen05 dense mode (no hardware block scales) via make_trivial_tiled_mma()
  - Scales: float32 per-128-block, applied per-K-tile in software
  - Scheduling: Persistent tile with MaskedScheduler (per-expert masked M)
  - Warp specialization: TMA warp / MMA warp / Epilogue warps

Key design: Since hardware block_scale MMA only supports Float8 scales (E8M0FNU),
we use the non-scaled dense MMA and apply float32 scales per-K-tile in software.
Each K-tile (128 elements) of MMA produces an unscaled partial sum in TMEM,
which the epilogue warps read, scale, and accumulate in registers.

Tensor layout convention (matching blockscaled_gemm.py):
  A: logical (M, K, L), physical (L, M, K) — K contiguous, L=E is batch
  B: logical (N, K, L), physical (L, N, K) — K contiguous
  C: logical (M, N, L), physical (L, M, N) — N contiguous

Scale layout convention (MN-major, matching trtllm):
  a_scale: [K//128, total_M]  float32
  b_scale: [E, N//128, K//128]  float32

Based on:
  - gemm_allreduce_two_shot.py: PersistentDenseGemmKernel (non-scaled MMA)
  - blockscaled_gemm.py: MaskedScheduler for grouped GEMM masking

AI-assisted implementation (Claude).
"""

import functools
from typing import Callable, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import Int32

from ..api_logging import flashinfer_api
from ..utils import get_compute_capability
from .blockscaled_gemm import MaskedScheduler, MaskedSchedulerParams
from .utils import cutlass_to_torch_dtype, get_cutlass_dtype, get_num_sm, make_ptr

# =============================================================================
# Part 1: Kernel Class
# =============================================================================


class Sm100GroupwiseScaledGroupedGemmKernel:
    """Persistent grouped GEMM with float32 per-128-block scales.

    Architecture: Blackwell SM100/SM103
    MMA: tcgen05 dense (no hardware block scales)
    Scales: float32 per-128-block, applied per-K-tile in software
    Scheduling: Persistent tile with MaskedScheduler (per-expert masked M)

    Warp specialization:
      - Warps 0-3 (epilogue): TMEM→register, scale application, accumulation, store
      - Warp 4 (MMA): Matrix multiply-accumulate
      - Warp 5 (TMA): Global→shared memory loads for A, B
    """

    def __init__(
        self,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        sm_version: str,
        c_dtype: Type[cutlass.Numeric],
    ):
        supported_sm_versions = ["sm_100", "sm_103"]
        assert sm_version in supported_sm_versions, (
            f"Only {supported_sm_versions} supported, got {sm_version}"
        )

        self.acc_dtype = cutlass.Float32
        self.c_dtype_init = c_dtype
        self.use_2cta_instrs = False  # Only single CTA mode for grouped GEMM
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.ONE

        self.occupancy = 1
        # Warp specialization IDs
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )
        # Barrier IDs
        self.cta_sync_bar_id = 0
        self.epilog_sync_bar_id = 1
        self.tmem_ptr_sync_bar_id = 2
        self.smem_capacity = utils.get_smem_capacity_in_bytes(sm_version)

    def _setup_attributes(self):
        """Set up configurations dependent on GEMM inputs."""
        # Configure tiled MMA (non-scaled, dense mode)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        # Compute mma/cluster/tile shapes
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

        # Cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Multicast CTAs
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # TMA store for epilogue
        self.use_tma_store = True
        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Stage counts — 1 acc stage for per-K-tile cycling
        self.num_acc_stage = 1
        _, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
        )

        # Shared memory layouts for A, B, C
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage,
        )

        # Tensor memory allocation columns
        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma, self.mma_tiler, self.num_acc_stage
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,        # logical (M, K, L), physical (L, M, K)
        b: cute.Tensor,        # logical (N, K, L), physical (L, N, K)
        a_scale: cute.Tensor,  # (K//128, M*L) float32 — flat scale
        b_scale: cute.Tensor,  # (L, N//128, K//128) float32
        c: cute.Tensor,        # logical (M, N, L), physical (L, M, N)
        masked_m: cute.Tensor, # (L,) int32
        max_active_clusters: cutlass.Constexpr,
        m_per_expert: Int32,   # M dimension per expert (for a_scale indexing)
        stream: cuda.CUstream,
    ):
        """Execute the grouped GEMM operation."""
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

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

        # TMA store C
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c, epi_smem_layout, self.epi_tile,
        )

        # Grid via MaskedScheduler
        self.tile_sched_params, grid = self._compute_grid(
            masked_m, c, self.cta_tile_shape_mnk,
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
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
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

        self.shared_storage = SharedStorage

        # Launch kernel
        self.kernel(
            tiled_mma,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_c, tma_tensor_c,
            a_scale, b_scale,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            m_per_expert,
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
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mAScale: cute.Tensor,  # (K//128, total_M) float32
        mBScale: cute.Tensor,  # (E, N//128, K//128) float32
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: MaskedSchedulerParams,
        m_per_expert: Int32,
    ):
        """GPU kernel: persistent grouped GEMM with per-K-tile scaling."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        # Prefetch TMA descriptors
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

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

        # SMEM tensors
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
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

        # Local tile partition for A, B, C
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
        # TMA warp: Load A, B from GMEM → SMEM
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
                tAgA_slice = tAgA[
                    (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
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
        # MMA warp: Compute per-K-tile partials
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
                    # Single acc stage
                    tCtAcc = tCtAcc_base[(None, None, None, 0)]

                    # Wait for AB data
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(
                            ab_consumer_state, peek_ab_full_status
                        )

                    # Wait for acc empty
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state)

                    # Reset accumulate for each K-tile
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

                    # Signal partial ready
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile, _ = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # ===================================================================
        # Epilogue warps: TMEM→register, scale, accumulate, store
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

            # Setup epilogue copies
            epi_tidx = tidx
            (
                tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_base, tCgC, epi_tile, use_2cta_instrs
            )
            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(
                tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            tma_atom_c_final, bSG_sC, bSG_gC = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC, epi_tile, sC
            )

            # Compute subtile count from TMEM partition layout.
            # subtile_cnt > 1 when epi_tile is smaller than the full MMA tile.
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

            # Allocate register accumulators for ALL epi subtiles across
            # K-tiles. We use a 1D flat buffer of size subtile_cnt * subtile_size
            # so that each subtile has its own accumulation region.
            total_accum_elems = subtile_size * subtile_cnt
            tTR_rAcc_accum = cute.make_fragment(
                cute.make_layout((total_accum_elems,)), self.acc_dtype
            )

            # Persistent tile loop
            tile_sched = MaskedScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            epilog_threads = 32 * len(self.epilog_warp_id)
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread, epilog_threads, epilog_threads,
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                expert_idx = mma_tile_coord_mnl[2]
                n_tile_idx = mma_tile_coord_mnl[1]

                bSG_gC_tile = bSG_gC[(None, None, None, *mma_tile_coord_mnl)]

                # Zero the register accumulators (all subtiles)
                tTR_rAcc_accum.fill(cutlass.Float32(0.0))

                # M coordinate for a_scale indexing.
                # Assumption: epilogue thread tidx maps to M-row tidx within
                # the tile. This holds for standard Blackwell tmem load ops
                # when epi_tile M == CTA tile M (128 threads, 128 M rows).
                # If the tmem load uses a different thread-to-row permutation,
                # this needs adjustment (verify on hardware).
                m_tile_start = mma_tile_coord_mnl[0] * Int32(self.mma_tiler[0])
                m_flat = expert_idx * m_per_expert + m_tile_start + Int32(epi_tidx)

                # --- Per-K-tile accumulation with scale application ---
                for k_tile in range(k_tile_cnt):
                    tTR_tAcc = tTR_tAcc_base[
                        (None, None, None, None, None, 0)
                    ]

                    acc_pipeline.consumer_wait(acc_consumer_state)

                    # Load float32 scales for this K-tile.
                    # b_scale: one scalar per (expert, N-block, K-tile)
                    b_scale_val = mBScale[expert_idx, n_tile_idx, k_tile]
                    # a_scale: one value per (K-tile, M-row)
                    a_scale_val = mAScale[k_tile, m_flat]
                    combined_scale = a_scale_val * b_scale_val

                    # Read each epi subtile from TMEM, scale, and accumulate.
                    # Each K-tile's full MMA partial is in TMEM; we must read
                    # ALL subtiles before releasing back to the MMA warp.
                    tTR_tAcc_grouped = cute.group_modes(
                        tTR_tAcc, 3, cute.rank(tTR_tAcc)
                    )
                    for subtile_idx in cutlass.range(subtile_cnt):
                        tTR_tAcc_mn = tTR_tAcc_grouped[
                            (None, None, None, subtile_idx)
                        ]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        # Scale the partial and accumulate into this subtile's
                        # region of the flat accumulator buffer.
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

                # --- Store accumulated result per subtile ---
                bSG_gC_tile_grouped = cute.group_modes(
                    bSG_gC_tile, 1, cute.rank(bSG_gC_tile)
                )
                store_subtile_cnt = cute.size(
                    bSG_gC_tile_grouped.shape, mode=[1]
                )

                for subtile_idx in cutlass.range(store_subtile_cnt):
                    # Convert accumulated F32 to output dtype from
                    # this subtile's region of the accumulator.
                    accum_base = subtile_idx * subtile_size
                    for i in cutlass.range(subtile_size):
                        tTR_rC[i] = self.c_dtype(
                            tTR_rAcc_accum[accum_base + i]
                        )

                    # Copy register → SMEM
                    c_buffer = subtile_idx % self.num_c_stage
                    cute.copy(
                        tiled_copy_r2s, tRS_rC,
                        tRS_sC[(None, None, None, c_buffer)],
                    )
                    cute.arch.fence_proxy(
                        cute.arch.ProxyKind.async_shared,
                        space=cute.arch.SharedSpace.shared_cta,
                    )
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

                    # TMA store SMEM → GMEM
                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c_final,
                            bSG_sC[(None, c_buffer)],
                            bSG_gC_tile_grouped[(None, subtile_idx)],
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=epilog_threads,
                    )

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
            c_pipeline.producer_tail()

    # --- Epilogue helpers (matching blockscaled_gemm.py pattern) ---

    def epilog_tmem_copy_and_partition(
        self, tidx, tAcc, gC_mnl, epi_tile, use_2cta_instrs,
    ):
        """TMEM → register copy setup."""
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

    def epilog_smem_copy_and_partition(
        self, tiled_copy_t2r, tTR_rC, tidx, sC,
    ):
        """Register → SMEM copy setup."""
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self, tidx, atom, gC_mnl, epi_tile, sC,
    ):
        """SMEM → GMEM TMA store setup."""
        gC_epi = cute.flat_divide(
            gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile
        )
        tma_atom_c = atom
        sC_for_tma = cute.group_modes(sC, 0, 2)
        gC_for_tma = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c, 0, cute.make_layout(1),
            sC_for_tma, gC_for_tma,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(
        tiled_mma, mma_tiler_mnk, a_dtype, b_dtype, epi_tile,
        c_dtype, c_layout, smem_capacity, occupancy,
    ):
        """Compute stage counts for A/B/C operands."""
        num_acc_stage = 1  # Per-K-tile cycling
        num_c_stage = 2    # TMA store double-buffered

        a_smem_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1
        )
        b_smem_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1
        )
        c_smem_one = sm100_utils.make_smem_layout_epi(
            c_dtype, c_layout, epi_tile, 1
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_one)
            + cute.size_in_bytes(b_dtype, b_smem_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    @staticmethod
    def _compute_grid(
        masked_m_tensor, c, cta_tile_shape_mnk, cluster_shape_mn,
        max_active_clusters,
    ):
        """Compute grid size using MaskedScheduler."""
        c_tiler = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        tile_sched_params = MaskedSchedulerParams(
            masked_m_tensor, None, c, c_tiler, cluster_shape_mnl
        )
        grid = MaskedScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma, mma_tiler, num_acc_stage):
        """Compute tensor memory allocation columns."""
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, num_acc_stage)
        )
        return tcgen05.find_tmem_tensor_col_offset(tCtAcc_fake)


# =============================================================================
# Part 2: Compilation Cache and Host Wrapper
# =============================================================================


class GroupwiseScaledGroupedGemmCuteDSL:
    """Host wrapper that creates CuTe tensors and launches the kernel.

    Tensor format (matching blockscaled_gemm.py convention):
      A: physical [L, M, K], logical (M, K, L) with order=(1, 0, 2)
      B: physical [L, N, K], logical (N, K, L) with order=(1, 0, 2)
      C: physical [L, M, N], logical (M, N, L) with order=(1, 0, 2)

    Where L=E (number of experts), M=max_padded_per_expert.
    """

    def __init__(
        self,
        M: int,   # max padded M per expert
        N: int,
        K: int,
        E: int,   # number of experts (= L batch dimension)
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        sm_count: int,
        sm_version: str,
    ):
        self._M = M
        self._N = N
        self._K = K
        self._E = E
        self._c_dtype = c_dtype
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
        a_ptr: cute.Pointer,       # FP8
        b_ptr: cute.Pointer,       # FP8
        a_scale_ptr: cute.Pointer, # F32
        b_scale_ptr: cute.Pointer, # F32
        c_ptr: cute.Pointer,       # output dtype
        masked_m_ptr: cute.Pointer, # Int32
        current_stream: cuda.CUstream,
    ):
        # A: logical (M, K, L), physical (L, M, K)
        a_tensor = cute.make_tensor(
            a_ptr,
            layout=cute.make_ordered_layout(
                (self._M, self._K, self._E),
                order=(1, 0, 2),  # K contiguous, M next, L slowest
            ),
        )
        # B: logical (N, K, L), physical (L, N, K)
        b_tensor = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout(
                (self._N, self._K, self._E),
                order=(1, 0, 2),
            ),
        )
        # C: logical (M, N, L), physical (L, M, N)
        c_tensor = cute.make_tensor(
            c_ptr,
            layout=cute.make_ordered_layout(
                (self._M, self._N, self._E),
                order=(1, 0, 2),
            ),
        )
        # a_scale: (K//128, M*E) flat float32
        a_scale_tensor = cute.make_tensor(
            a_scale_ptr,
            layout=cute.make_ordered_layout(
                (self._K // 128, self._M * self._E),
                order=(1, 0),  # M*E contiguous
            ),
        )
        # b_scale: (E, N//128, K//128) float32
        b_scale_tensor = cute.make_tensor(
            b_scale_ptr,
            layout=cute.make_ordered_layout(
                (self._E, self._N // 128, self._K // 128),
                order=(2, 1, 0),  # K//128 contiguous
            ),
        )
        # masked_m: (E,) int32
        masked_m_tensor = cute.make_tensor(
            masked_m_ptr,
            layout=cute.make_ordered_layout((self._E,), order=(0,)),
        )

        Sm100GroupwiseScaledGroupedGemmKernel(
            mma_tiler_mn=self._mma_tiler_mn,
            cluster_shape_mn=self._cluster_shape_mn,
            sm_version=self._sm_version,
            c_dtype=self._c_dtype,
        )(
            a_tensor, b_tensor, a_scale_tensor, b_scale_tensor,
            c_tensor, masked_m_tensor,
            self._max_active_clusters,
            Int32(self._M),  # m_per_expert for a_scale indexing
            current_stream,
        )


@functools.cache
def _get_compiled_grouped_gemm_kernel(
    M: int,    # max padded M per expert
    N: int,
    K: int,
    E: int,
    c_dtype_str: str,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    sm_count: int,
    sm_version: str,
) -> Callable:
    """Compile and cache the grouped GEMM kernel."""
    c_dtype = get_cutlass_dtype(c_dtype_str)
    ab_dtype = cutlass.Float8E4M3FN

    def get_cute_pointers(input_tensors=None):
        if input_tensors is None:
            ptrs = [16] * 6
        else:
            ptrs = [t.data_ptr() for t in input_tensors]

        return [
            make_ptr(ab_dtype, ptrs[0], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(ab_dtype, ptrs[1], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, ptrs[2], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, ptrs[3], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(c_dtype, ptrs[4], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, ptrs[5], cute.AddressSpace.gmem, assumed_align=16),
        ]

    wrapper = GroupwiseScaledGroupedGemmCuteDSL(
        M=M, N=N, K=K, E=E,
        c_dtype=c_dtype,
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
        a: torch.Tensor,       # [E, M, K] physical
        b: torch.Tensor,       # [E, N, K] physical
        a_scale: torch.Tensor, # [K//128, M*E] flat
        b_scale: torch.Tensor, # [E, N//128, K//128]
        c: torch.Tensor,       # [E, M, N] physical
        masked_m: torch.Tensor, # [E]
    ):
        nonlocal kernel
        current_stream = cutlass_torch.current_stream()
        kernel(
            *get_cute_pointers([a, b, a_scale, b_scale, c, masked_m]),
            current_stream,
        )
        return c

    return tensor_api


# =============================================================================
# Part 3: Public API
# =============================================================================


@flashinfer_api
def moe_grouped_gemm_fp8_cutedsl(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    masked_m: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """CuTeDSL grouped GEMM with float32 per-128-block scales.

    Computes: out = dequant(a) @ dequant(b)^T with per-expert masking.

    The input A is [total_M, K] concatenated across experts, where total_M = sum(masked_m).
    This function reshapes to [E, max_M, K] batched format for the CuTeDSL kernel.

    Args:
        a: [total_M, K] float8_e4m3fn — input activations (grouped by expert).
        b: [E, N, K] float8_e4m3fn — expert weights.
        a_scale: [K//128, total_M] float32 — activation block scales (MN-major).
        b_scale: [E, N//128, K//128] float32 — weight block scales (MN-major).
        masked_m: [E] int32 — valid M per expert.
        out_dtype: output dtype (torch.bfloat16).
        out: optional pre-allocated output [total_M, N].

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
    max_M_raw = int(masked_m.max().item())

    # Early exit: no tokens routed to any expert
    if max_M_raw == 0:
        if out is None:
            out = torch.empty(0, N, dtype=out_dtype, device=device)
        return out

    # Pad max_M up to next multiple of 128 (MMA tile M dimension).
    # MaskedScheduler uses masked_m to skip padded rows.
    max_M = ((max_M_raw + 127) // 128) * 128

    # Reshape [total_M, K] → [E, max_M, K] with padding
    if total_M == E * max_M_raw and max_M == max_M_raw:
        # Direct reshape: no padding needed
        a_batched = a.view(E, max_M, K)
    else:
        # Pad each expert's rows to max_M
        a_batched = torch.zeros(E, max_M, K, dtype=a.dtype, device=device)
        offset = 0
        for e in range(E):
            m_e = int(masked_m[e].item())
            a_batched[e, :m_e, :] = a[offset:offset + m_e, :]
            offset += m_e

    # B is already [E, N, K] — matches physical layout (L, N, K)
    b_batched = b

    # Output: [E, max_M, N]
    c_batched = torch.empty(E, max_M, N, dtype=out_dtype, device=device)

    # a_scale: [K//128, total_M] → [K//128, E*max_M] with padding
    if total_M == E * max_M_raw and max_M == max_M_raw:
        a_scale_flat = a_scale  # already correct
    else:
        a_scale_flat = torch.zeros(
            K // 128, E * max_M, dtype=torch.float32, device=device
        )
        offset = 0
        for e in range(E):
            m_e = int(masked_m[e].item())
            a_scale_flat[:, e * max_M:e * max_M + m_e] = a_scale[:, offset:offset + m_e]
            offset += m_e

    # Map output dtype
    dtype_str_map = {
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
        torch.float32: "float32",
    }
    c_dtype_str = dtype_str_map[out_dtype]

    major, minor = get_compute_capability(device)
    sm_count = get_num_sm(device)
    sm_version = f"sm_{major}{minor}"

    mma_tiler_mn = (128, 128)
    cluster_shape_mn = (1, 1)

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

    # Flatten output back to [total_M, N]
    if total_M == E * max_M:
        if out is None:
            out = c_batched.view(total_M, N)
        else:
            out.copy_(c_batched.view(total_M, N))
    else:
        if out is None:
            out = torch.empty(total_M, N, dtype=out_dtype, device=device)
        offset = 0
        for e in range(E):
            m_e = int(masked_m[e].item())
            out[offset:offset + m_e, :] = c_batched[e, :m_e, :]
            offset += m_e

    return out
