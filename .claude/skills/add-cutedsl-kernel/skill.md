---
name: add-cutedsl-kernel
description: Step-by-step tutorial for adding new CuTeDSL kernels to FlashInfer
---

# Tutorial: Adding a New CuTeDSL Kernel to FlashInfer

This tutorial documents the pattern established during the CuTeDSL FP8 MoE pipeline development (Steps 1-3). CuTeDSL kernels live entirely in Python (`flashinfer/cute_dsl/`) and use NVIDIA's CuTe DSL for GPU kernel authoring.

## When to Use CuTeDSL vs Traditional C++/CUDA

| Aspect | CuTeDSL (Python) | Traditional (C++/CUDA) |
|--------|-------------------|------------------------|
| Language | Python (`@cute.kernel`) | C++ in `include/`, bindings in `csrc/` |
| Compilation | `cute.compile()` → MLIR → PTX | Ninja + nvcc JIT |
| SM Target | SM100+ (Blackwell) primarily | All architectures |
| Use case | Tensor-core heavy, persistent kernels | General CUDA, older architectures |
| Iteration speed | Faster (no C++ rebuild) | Slower |

## File Structure: The 3-Part Pattern

Every CuTeDSL kernel file follows this structure:

```
flashinfer/cute_dsl/my_kernel.py
├── Part 1: Kernel Class (GPU logic)
├── Part 2: Compilation Cache (@functools.cache)
└── Part 3: Public API (@flashinfer_api)
```

### Part 1: Kernel Class

```python
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint8

class MyKernel:
    """Docstring describing what the kernel computes.

    Grid:  (dim_x, dim_y, 1)
    Block: (threads_per_block, 1, 1)
    """

    BLOCK_SIZE = 128

    def __init__(self, fixed_param: int):
        # Store compile-time constants
        self.fixed_param = fixed_param

    @cute.jit
    def __call__(self, mInput: cute.Tensor, mOutput: cute.Tensor,
                 dynamic_param: Int32, stream):
        """JIT entry point — launches the kernel."""
        self.kernel(mInput, mOutput, dynamic_param).launch(
            grid=[grid_x, grid_y, 1],
            block=[self.BLOCK_SIZE, 1, 1],
            smem=0,  # shared memory bytes
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mInput: cute.Tensor, mOutput: cute.Tensor,
               dynamic_param: Int32):
        """The actual GPU kernel."""
        tid = cute.arch.thread_idx()[0]
        blk_x = cute.arch.block_idx()[0]
        # ... kernel logic ...
```

**Key rules:**
- `__init__` params are compile-time constants (baked into compiled binary)
- `__call__` params with `cute.Tensor` type use TMA descriptors
- `Int32`/`Float32` scalar params are passed as kernel arguments
- `stream` is always the last parameter of `__call__`
- Use `cute.arch.*` for thread/block indices, barriers, warp shuffles

### Part 2: Compilation Cache

```python
import functools

@functools.cache
def _get_compiled_kernel(fixed_param: int):
    """Compile kernel once, cache forever (keyed on fixed_param)."""
    kernel_obj = MyKernel(fixed_param=fixed_param)

    # Create symbolic dimension for dynamic sizes
    sym_dynamic = cute.sym_int()

    # Make fake tensors describing the layout (NOT real data)
    input_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,                           # element type
        (sym_dynamic, fixed_param),              # shape (can mix symbolic + concrete)
        stride_order=(1, 0),                     # row-major
        assumed_align=128,                       # alignment hint
    )
    output_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_dynamic, fixed_param),
        stride_order=(1, 0),
        assumed_align=128,
    )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # Compile to binary
    compiled = cute.compile(
        kernel_obj,
        input_fake,
        output_fake,
        Int32(1),           # dummy for dynamic_param
        stream_fake,
        options="--enable-tvm-ffi",
    )

    # Return a closure that accepts real PyTorch tensors
    def tensor_api(input_tensor: torch.Tensor, output_tensor: torch.Tensor,
                   dynamic_param: int) -> None:
        compiled(
            input_tensor.view(torch.uint8),  # FP8 tensors must be viewed as uint8
            output_tensor,
            Int32(dynamic_param),
        )

    return tensor_api
```

**Key rules:**
- `@functools.cache` key must include ALL compile-time constants
- Symbolic dimensions (`cute.sym_int()`) allow dynamic shapes without recompilation
- FP8 tensors (`float8_e4m3fn`) must use `cutlass.Uint8` type and `.view(torch.uint8)` at call time
- `options="--enable-tvm-ffi"` is required for PyTorch interop
- The `tensor_api` closure hides compilation details from callers

### Part 3: Public API

```python
from ..api_logging import flashinfer_api

@flashinfer_api
def my_operation(
    input: torch.Tensor,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Public API with shape validation and output allocation.

    Args:
        input: [M, N] float8_e4m3fn
        output: optional [M, N] bfloat16

    Returns:
        output: [M, N] bfloat16
    """
    M, N = input.shape
    assert N % 128 == 0, f"N must be multiple of 128, got {N}"

    if output is None:
        output = torch.empty(M, N, dtype=torch.bfloat16, device=input.device)

    kernel = _get_compiled_kernel(fixed_param=N)
    kernel(input, output, M)

    return output
```

**Key rules:**
- Always use `@flashinfer_api` decorator for logging/debugging support
- Validate shapes before calling kernel
- Allocate output if not provided
- Never expose compilation details to the caller

## FP8 Handling Patterns

### FP8 Tensor Convention
CuTeDSL treats FP8 as raw bytes. Always use `Uint8` in fake tensors and `.view(torch.uint8)` when passing to compiled kernel:

```python
# In fake tensor creation
fake = cute.runtime.make_fake_compact_tensor(
    cutlass.Uint8,  # NOT Float8
    shape, stride_order=(1, 0), assumed_align=128,
)

# In tensor_api closure
compiled(tensor.view(torch.uint8), ...)
```

### FP8 ↔ Float32 Conversion in Kernels
Use the helper functions from `fp4_common.py`:

```python
from .fp4_common import cvt_e4m3_to_f32, cvt_f32_to_e4m3

# In @cute.kernel:
val_u8 = mInput[row, col]           # Uint8
val_f32 = cvt_e4m3_to_f32(val_u8)   # Float32
# ... compute ...
out_u32 = cvt_f32_to_e4m3(result_f32)
mOutput[row, col] = Uint8(out_u32 & cutlass.Uint32(0xFF))
```

### Per-128-Block Scale Layout (MN-major)
FlashInfer uses MN-major scale layout matching trtllm convention:

```
a_scale: [K//128, M]          float32   (K blocks × M rows)
b_scale: [E, N//128, K//128]  float32   (per-expert, N blocks × K blocks)
```

## Warp-Level Patterns

### Warp Shuffle Reduction
```python
from .fp4_common import warp_reduce

warp_max = warp_reduce(abs_val, cute.arch.fmax)  # max across 32 threads
```

### Cross-Warp Reduction via SMEM
```python
smem = cutlass.utils.SmemAllocator()
buf = smem.allocate_tensor(Float32, cute.make_layout((NUM_WARPS,)), byte_alignment=4)

lane_idx = cute.arch.lane_idx()
warp_idx = cute.arch.warp_idx()

if lane_idx == 0:
    buf[warp_idx] = warp_max

cute.arch.barrier()

if lane_idx < NUM_WARPS:
    block_max = buf[lane_idx]
block_max = warp_reduce(block_max, cute.arch.fmax)
```

## Persistent GEMM Kernel Pattern (Advanced)

For GEMM kernels using tcgen05 MMA on SM100+, the pattern is more complex. Key references:

| Component | Reference File |
|-----------|---------------|
| Dense MMA (no hardware scales) | `gemm_allreduce_two_shot.py` |
| MaskedScheduler (grouped GEMM) | `blockscaled_gemm.py` |
| Block-scaled MMA (FP8 scales) | `blockscaled_gemm.py` |
| Our groupwise float32 scales | `moe_grouped_gemm_cutedsl.py` |

Key concepts:
- **Warp specialization**: TMA warp (loads), MMA warp (compute), Epilogue warps (store)
- **Pipeline stages**: `AB pipeline` (A/B data), `acc pipeline` (accumulator)
- **MaskedScheduler**: Handles per-expert variable M with masked rows
- **`make_trivial_tiled_mma()`**: FP8×FP8→F32 dense mode (no hardware block scales)

## Testing Pattern

### Correctness Testing for FP8 Kernels

FP8 precision limits mean standard `torch.allclose()` is unreliable. Use **cosine similarity**:

```python
@pytest.mark.skipif(not _check_sm100(), reason="Requires SM100+")
def test_my_kernel():
    # ... run kernel and reference ...

    cos_sim = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), out.float().flatten(), dim=0
    )
    assert cos_sim > 0.95, f"cosine similarity {cos_sim:.4f} too low"
```

**Tolerance guidelines:**
- Single FP8 GEMM: cosine sim > 0.95
- FP8 quantize roundtrip: cosine sim > 0.99
- Full MoE pipeline (multiple FP8 stages): cosine sim > 0.98

### SM100+ Check Helper
```python
def _check_sm100():
    if not torch.cuda.is_available():
        return False
    try:
        from flashinfer.utils import get_compute_capability
        return get_compute_capability(torch.device("cuda"))[0] >= 10
    except Exception:
        return False
```

## Common Pitfalls

1. **Forgetting `.view(torch.uint8)`** for FP8 tensors → crash or silent wrong results
2. **Cache key missing a compile-time param** → wrong kernel dispatched for different configs
3. **Symbolic vs concrete dimensions** — use `cute.sym_int()` for any dimension that varies at runtime; concrete values get baked in
4. **SMEM allocation** — must be declared inside `@cute.kernel`, not `@cute.jit`
5. **`make_fragment` deprecated** → use `make_rmem_tensor` instead (warning only for now)
6. **MMA tile alignment** — M, N, K must be multiples of 128 for tcgen05 MMA; pad inputs if needed and use MaskedScheduler to skip padded rows

## Example Files (Increasing Complexity)

1. **Simple (element-wise)**: `moe_activation.py` — SwiGLU + FP8 requant, 128 threads per block
2. **Medium (reduction)**: `moe_finalize.py` — gather + weighted reduce with atomics
3. **Complex (persistent GEMM)**: `moe_grouped_gemm_cutedsl.py` — warp-specialized grouped GEMM with TMA
