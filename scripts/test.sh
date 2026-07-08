#!/usr/bin/env bash
# test.sh — run the pytest suite.
# Real tests land in M1 (CPU reference) and M2+ (CUDA kernels via a pybind11 shim).
set -euo pipefail

if ! ls tests/*.py >/dev/null 2>&1; then
    echo "no tests yet (added in M1). skipping."
    exit 0
fi

pytest tests/ "$@"
