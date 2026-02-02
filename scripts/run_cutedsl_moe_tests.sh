#!/bin/bash
# Run CuTeDSL FP8 MoE tests and benchmarks on a remote Blackwell (SM100+) machine.
#
# Usage:
#   bash scripts/run_cutedsl_moe_tests.sh              # run all tests
#   bash scripts/run_cutedsl_moe_tests.sh --bench       # run tests + benchmarks
#   bash scripts/run_cutedsl_moe_tests.sh --bench-only  # benchmarks only
#   SKIP_INSTALL=1 bash scripts/run_cutedsl_moe_tests.sh  # skip pip install

set -eo pipefail
set -x

: ${CUDA_VISIBLE_DEVICES:=0}
: ${SKIP_INSTALL:=0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- Source test environment (handles TVM-FFI overrides) ---
if [ -f "$SCRIPT_DIR/setup_test_env.sh" ]; then
    source "$SCRIPT_DIR/setup_test_env.sh"
fi

# --- Install flashinfer in editable mode ---
if [ "$SKIP_INSTALL" = "0" ]; then
    echo "=== Installing flashinfer (editable) ==="
    pip install --no-build-isolation -e . -v
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Parse args ---
RUN_TESTS=1
RUN_BENCH=0
for arg in "$@"; do
    case "$arg" in
        --bench)
            RUN_BENCH=1
            ;;
        --bench-only)
            RUN_TESTS=0
            RUN_BENCH=1
            ;;
    esac
done

FAILED=0

run_test() {
    local test_file="$1"
    echo ""
    echo "========================================="
    echo "  Running: $test_file"
    echo "========================================="
    if pytest -s "$test_file"; then
        echo "  PASSED: $test_file"
    else
        echo "  FAILED: $test_file"
        FAILED=1
    fi
}

if [ "$RUN_TESTS" = "1" ]; then
    echo ""
    echo "============================================"
    echo "  CuTeDSL FP8 MoE — Test Suite"
    echo "============================================"

    # Phase 0: Routing (CPU-only tests, no SM100 required)
    run_test tests/moe/test_cute_dsl_moe_routing.py

    # Phase 3: SwiGLU + FP8 Requant (CPU-only tests)
    run_test tests/moe/test_cute_dsl_moe_activation.py

    # Phase 4: Finalize (CPU-only tests)
    run_test tests/moe/test_cute_dsl_moe_finalize.py

    # Phase 1: Grouped GEMM FP8 (SM100+ required)
    run_test tests/moe/test_cute_dsl_grouped_gemm_fp8.py

    # Phase 2: A-Gather GEMM (SM100+ required)
    run_test tests/moe/test_cute_dsl_agather_gemm.py

    # Phase 5: Full pipeline E2E (SM100+ required)
    run_test tests/moe/test_cute_dsl_moe_pipeline.py
fi

if [ "$RUN_BENCH" = "1" ]; then
    echo ""
    echo "============================================"
    echo "  CuTeDSL FP8 MoE — Benchmarks"
    echo "============================================"

    # Full pipeline benchmark
    echo "--- Pipeline benchmark ---"
    python benchmarks/bench_cute_dsl_moe_fp8.py --seq_len 1 4 16 64 256 1024

    # Per-stage breakdown
    echo ""
    echo "--- Per-stage breakdown ---"
    python benchmarks/bench_cute_dsl_moe_fp8.py --seq_len 16 64 256 --per_stage
fi

echo ""
echo "============================================"
if [ "$FAILED" = "0" ]; then
    echo "  All tests passed!"
else
    echo "  Some tests FAILED (see above)"
    exit 1
fi
