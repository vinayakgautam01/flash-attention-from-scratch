"""M5 correctness matrix — every registered variant × every shape in the grid.

Diffs each variant against ``torch_ref`` (the ``VariantSpec.is_reference``
entry — PyTorch's own SDPA), with the per-variant Python-side parity
tolerance from ``benchmarks.variants.VariantSpec.tolerance_abs``. The
tighter M1-style oracle check (variant vs the CPU reference in
``tests/reference.py``) still lives in the M1/M2/M3/M4 GoogleTest binaries
under ``tests/cpp/`` — this file is the Python-side parity net.

Skips cleanly (via ``pytest.skip``) when:
* no CUDA device is present (Mac-local),
* the compiled ``_C`` extension is missing,
* the variant does not support the shape (e.g. ``D=128`` for M5 kernels).

See ``theory/M5.md`` §7 for the correctness-policy rationale.
"""

from __future__ import annotations

import types

import pytest

# Shape grid — mirrors ``benchmarks.bench_attention`` defaults. Keeping them
# in sync manually is fine for M5's minimal grid; M7 will factor these into a
# shared module when the grid expands.
SHAPES = [
    (B, H, N, D, causal)
    for B in (1,)
    for H in (1,)
    for N in (128, 256, 512, 1024, 2048)
    for D in (32, 64)
    for causal in (False,)
]


def _torch() -> types.ModuleType:
    """Import torch or skip the test cleanly on hosts without it."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    return torch


def _require_cuda_ext():
    """Import the harness or skip cleanly. Returns (VARIANTS, variant_names)."""
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for the variant matrix")
    try:
        import flash_from_scratch  # noqa: F401
    except ImportError as e:
        pytest.skip(f"flash_from_scratch not importable: {e}")
    if not flash_from_scratch.HAS_CUDA_EXT:
        pytest.skip("flash_from_scratch._C extension not built on this host")
    from benchmarks.variants import VARIANTS, supports, variant_names
    return VARIANTS(), variant_names(), supports


def _variant_names_lazy() -> list[str]:
    """Best-effort registered variant names for pytest parametrization.

    Called at *collection time*, so we cannot rely on CUDA. Fall back to a
    hard-coded list (matching :mod:`benchmarks.variants`) so ``pytest
    --collect-only`` still enumerates the matrix on Mac-local.
    """
    try:
        from benchmarks.variants import VARIANTS as _V
        return list(_V().keys())
    except Exception:
        return [
            "torch_ref",
            "attention_naive",
            "attention_online_ref",
            "flash_fwd_v1",
            "flash_fwd_v2",
        ]


@pytest.mark.parametrize("variant", _variant_names_lazy())
@pytest.mark.parametrize("B,H,N,D,causal", SHAPES)
def test_variant_matches_reference(variant: str, B: int, H: int, N: int, D: int,
                                   causal: bool) -> None:
    """Each variant must match ``torch_ref`` within its declared tolerance."""
    VARIANTS, _, supports = _require_cuda_ext()
    if not supports(variant, D, causal, "fp32"):
        pytest.skip(f"{variant} does not support D={D}, causal={causal}, fp32")

    from tests.util_tensors import make_qkv
    Q, K, V = make_qkv(B, H, N, D, device="cuda", seed=0)

    spec = VARIANTS[variant]
    out = spec.fn(Q, K, V, causal)

    ref_spec = VARIANTS["torch_ref"]
    out_ref = ref_spec.fn(Q, K, V, causal)

    abs_err = (out.float() - out_ref.float()).abs().max().item()
    assert abs_err <= spec.tolerance_abs, (
        f"{variant} @ B={B} H={H} N={N} D={D} causal={causal}: "
        f"max abs err {abs_err:.3e} > tol {spec.tolerance_abs:.3e}"
    )


@pytest.mark.parametrize("N,D", [(128, 32), (512, 64), (2048, 64)])
def test_flash_v2_matches_v1(N: int, D: int) -> None:
    """M6 entry condition: v2 is a drop-in replacement for v1.

    MILESTONES §M6 asks for "correctness parity with v1". The C++ suite
    (``tests/cpp/test_flash_fwd_v2.cu``) owns the authoritative check against
    the CPU reference; this is the Python-side mirror so a parity break shows
    up in ``scripts/test.sh`` too.

    Bound is 1e-5, tighter than the 5e-4 oracle tolerance: both kernels
    accumulate in the same order, so they should differ only by FMA-contraction
    round-off, not by algebra.
    """
    VARIANTS, _, supports = _require_cuda_ext()
    if not (supports("flash_fwd_v1", D, False, "fp32")
            and supports("flash_fwd_v2", D, False, "fp32")):
        pytest.skip(f"v1/v2 do not both support D={D}")

    from tests.util_tensors import make_qkv
    Q, K, V = make_qkv(1, 1, N, D, device="cuda", seed=0)

    out_v1 = VARIANTS["flash_fwd_v1"].fn(Q, K, V, False)
    out_v2 = VARIANTS["flash_fwd_v2"].fn(Q, K, V, False)

    abs_err = (out_v2.float() - out_v1.float()).abs().max().item()
    assert abs_err <= 1e-5, (
        f"flash_fwd_v2 diverged from flash_fwd_v1 @ N={N} D={D}: "
        f"max abs err {abs_err:.3e}"
    )


def test_harness_sanity_timing() -> None:
    """M5 verification bullet: ``time_kernel`` returns positive medians.

    Runs a trivial no-op-ish CUDA launch to make sure the timing plumbing is
    wired correctly and does not silently return zero (the failure mode from
    forgetting ``torch.cuda.synchronize()`` — see ``theory/M5.md`` §3.1).
    """
    torch = _torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for timing sanity")
    from benchmarks.timing import time_kernel

    x = torch.randn(1024, 1024, device="cuda")

    def _launch():
        # Any real kernel; matmul is deterministic-enough for a sanity check.
        return x @ x

    result = time_kernel(_launch, warmup_iters=2, timed_iters=5)
    assert result.median_ms > 0.0, "timing loop reported non-positive median"
    assert result.p95_ms >= result.median_ms


def test_perf_model_hbm_estimates_are_positive() -> None:
    """Pure-Python sanity for ``benchmarks.perf_model``. No CUDA required."""
    from benchmarks.perf_model import (
        hbm_bytes_est,
        hbm_bytes_flash_v1,
        hbm_bytes_naive,
        hbm_bytes_online_ref,
        tflops_effective,
    )

    B, H, N, D = 1, 1, 2048, 64
    assert hbm_bytes_naive(B, H, N, D) > 0
    assert hbm_bytes_online_ref(B, H, N, D) > 0
    assert hbm_bytes_flash_v1(B, H, N, D, Br=32) > 0

    # theory/M5.md §4.1 predicts online_ref >> naive at large N.
    assert (hbm_bytes_online_ref(B, H, N, D)
            > hbm_bytes_naive(B, H, N, D))

    for variant in ("attention_naive", "attention_online_ref",
                    "flash_fwd_v1", "flash_fwd_v2", "torch_ref"):
        assert hbm_bytes_est(variant, B, H, N, D) > 0

    # M6 kept Br=32 and the FA-1 loop order, so v2's analytical traffic is
    # identical to v1's by construction (theory/M6.md §10). If this ever
    # diverges, the "mechanical changes only" claim needs revisiting.
    assert (hbm_bytes_est("flash_fwd_v2", B, H, N, D)
            == hbm_bytes_est("flash_fwd_v1", B, H, N, D))

    assert tflops_effective(B, H, N, D, runtime_ms=1.0) > 0.0
    assert tflops_effective(B, H, N, D, runtime_ms=0.0) == 0.0
