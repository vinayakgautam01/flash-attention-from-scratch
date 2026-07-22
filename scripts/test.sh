#!/usr/bin/env bash
# test.sh — run the full test suite: pytest (Python) + ctest (C++).
#
# Python: always runs (M1+).
# C++:    runs iff the build tree exists and has a C++ test binary. To bootstrap,
#         first run `scripts/build.sh` — that fetches GoogleTest and builds the
#         `test_attention_cpu_ref` binary.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"

# ── Python: pytest ───────────────────────────────────────────────────────────
if ls "${REPO_ROOT}"/tests/*.py >/dev/null 2>&1; then
    echo "→ pytest"
    (cd "${REPO_ROOT}" && pytest tests/ "$@")
else
    echo "→ pytest: no Python tests yet, skipping."
fi

# ── C++: ctest ───────────────────────────────────────────────────────────────
if [[ -d "${BUILD_DIR}" ]] && [[ -f "${BUILD_DIR}/CTestTestfile.cmake" ]]; then
    echo
    echo "→ ctest (${BUILD_DIR})"
    (cd "${BUILD_DIR}" && ctest --output-on-failure)
else
    echo
    echo "→ ctest: no build tree at ${BUILD_DIR}. Run scripts/build.sh first to enable C++ tests."
fi
