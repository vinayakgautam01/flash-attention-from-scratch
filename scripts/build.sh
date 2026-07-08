#!/usr/bin/env bash
# build.sh — configure and build all C++/CUDA artifacts.
# Overrides (env vars):
#   CMAKE_CUDA_ARCHITECTURES  e.g. `CMAKE_CUDA_ARCHITECTURES=89 scripts/build.sh`
#   CMAKE_BUILD_TYPE          e.g. `CMAKE_BUILD_TYPE=Debug scripts/build.sh`
#   BUILD_DIR                 defaults to `build`
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
ARCH="${CMAKE_CUDA_ARCHITECTURES:-86}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_CUDA_ARCHITECTURES="${ARCH}"

cmake --build "${BUILD_DIR}" -j

echo
echo "build OK. try: ./${BUILD_DIR}/hello"
