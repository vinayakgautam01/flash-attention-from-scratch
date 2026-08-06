#!/usr/bin/env bash
# bench.sh — run the M5 benchmark sweep and refresh the plots.
#
# Requires a CUDA host with the flash_from_scratch._C extension installed:
#   pip install -e .            # scikit-build-core builds the extension
# or, from a pre-built CMake tree:
#   BUILD_PY_EXT=ON scripts/build.sh && cp build/_C.*.so flash_from_scratch/
#
# Outputs:
#   benchmarks/results/all.csv
#   docs/plots/runtime_vs_N.png
#   docs/plots/speedup_vs_naive.png
#   docs/plots/error_histogram.png
#
# Overrides:
#   BENCH_ARGS="--N 128,256"   # forward extra CLI args to bench_attention
#   PLOT_ARGS="--D 32"         # forward extra CLI args to plot
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

echo "→ python -m benchmarks.bench_attention"
python -m benchmarks.bench_attention ${BENCH_ARGS:-}

echo
echo "→ python -m benchmarks.plot"
python -m benchmarks.plot ${PLOT_ARGS:-}

echo
echo "Done. See benchmarks/results/all.csv and docs/plots/."
