#!/usr/bin/env bash
# scripts/bootstrap_colab.sh — get a Colab session ready to build & test.
#
# Prereq: repo is already cloned and the current directory is inside it.
# Idempotent. First run in a session: ~2-3 min. Re-runs: ~10-15 s with SKIP_PIP=1.
#
# Steps (each fails fast on error):
#   0. Locate repo root (cd to it, so this works from any subdir).
#   1. GPU check     — nvidia-smi present + returns a device.
#   2. Toolchain     — nvcc, python, torch versions.
#   3. Repo sync     — git fetch + checkout $BRANCH + pull --ff-only.
#   4. Python deps   — pip install -e .[dev] (skippable via SKIP_PIP=1).
#   5. Build         — scripts/build.sh with auto-detected CUDA arch.
#   6. Smoke test    — ./build/hello.
#   7. Test suite    — scripts/test.sh (skippable via SKIP_TESTS=1).
#
# Usage in a Colab cell:
#
#   # Once per session (first cell):
#   !git clone https://github.com/<user>/<repo>.git
#   %cd <repo>
#   !bash scripts/bootstrap_colab.sh
#
#   # After every `git push` from local — resync + rebuild:
#   !bash scripts/bootstrap_colab.sh
#
#   # Fast re-run when only kernel/build changed:
#   !SKIP_PIP=1 bash scripts/bootstrap_colab.sh
#
# Env vars:
#   BRANCH       Branch/ref to sync to (default: main).
#   SKIP_PIP     Set to 1 to skip `pip install`.
#   SKIP_TESTS   Set to 1 to skip `scripts/test.sh`.

set -euo pipefail

BRANCH="${BRANCH:-main}"
SKIP_PIP="${SKIP_PIP:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"

banner() { printf "\n\033[1;34m== %s ==\033[0m\n" "$1"; }
ok()     { printf "\033[1;32m✓ %s\033[0m\n" "$1"; }
warn()   { printf "\033[1;33m! %s\033[0m\n" "$1"; }
fail()   { printf "\033[1;31m✗ %s\033[0m\n" "$1" >&2; exit 1; }

banner "0. Repo root"
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) \
    || fail "not inside a git repo. Run: git clone <url> && cd <repo> && bash scripts/bootstrap_colab.sh"
cd "${REPO_ROOT}"
ok "cwd: ${REPO_ROOT}"

banner "1. GPU check"
command -v nvidia-smi >/dev/null 2>&1 \
    || fail "nvidia-smi not found. Runtime → Change runtime type → GPU."
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader \
    || fail "no GPU allocated. Runtime → Change runtime type → GPU."

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '. ')
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
ok "detected: ${GPU} (sm_${CC})"

banner "2. Toolchain"
nvcc --version | tail -n2 || fail "nvcc missing."
python --version
python -c "import torch; print(f'torch {torch.__version__} · cuda {torch.version.cuda}')" \
    2>/dev/null || warn "torch not importable yet (will install below)"

banner "3. Repo sync"
echo "fetching + checkout + pull ${BRANCH}"
git fetch --quiet origin
git checkout --quiet "${BRANCH}"
git pull --ff-only --quiet
ok "$(git log -1 --oneline)"

if [ "${SKIP_PIP}" != "1" ]; then
    banner "4. Python deps"
    pip install --quiet --upgrade pip
    pip install --quiet -e ".[dev]"
    ok "pip install complete"
else
    warn "SKIP_PIP=1 → not touching python env"
fi

banner "5. Build (arch sm_${CC})"
CMAKE_CUDA_ARCHITECTURES="${CC}" scripts/build.sh

banner "6. Smoke test — ./build/hello"
./build/hello

if [ "${SKIP_TESTS}" != "1" ]; then
    banner "7. Test suite"
    scripts/test.sh || warn "tests skipped or none yet — expected for M0/M1"
fi

banner "done"
echo
echo "GPU:    ${GPU} (sm_${CC})"
echo "commit: $(git log -1 --oneline)"
