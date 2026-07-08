#!/usr/bin/env bash
# reproduce.sh — single-command reproduction of the hero result.
# For M0 this is the documented substitute for GPU CI: build + test + bench (both stubs at first).
# End-to-end hero-plot reproduction lands in M10.
set -euo pipefail

echo "== build =="
scripts/build.sh

echo
echo "== test =="
scripts/test.sh

echo
echo "== bench =="
scripts/bench.sh

echo
echo "reproduce complete. hero plot at docs/plots/hero.png lands in M10."
