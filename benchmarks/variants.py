"""The variant registry — one entry per benchmarkable kernel.

Adding a new kernel (e.g. Flash v2 in M6) is one entry here. Both the
correctness matrix (``tests/test_all_variants.py``) and the benchmark sweep
(``benchmarks/bench_attention.py``) iterate over :data:`VARIANTS`.

Every callable shares the same signature::

    fn(Q, K, V, is_causal) -> O

See ``theory/M5.md`` §5 for the design rationale.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class VariantSpec:
    """A single benchmarkable kernel.

    Attributes:
        fn: the Python callable taking ``(Q, K, V, is_causal)`` and returning
            ``O``. All callables live in :mod:`flash_from_scratch`.
        is_reference: true only for ``torch_ref`` — the oracle that every other
            variant is diffed against.
        supports_causal: whether the kernel implements the causal mask (M4 says
            "kernel supports it, tests skip it" — this flag matches the tests).
        supported_D: head dimensions the kernel accepts.
        supported_dtypes: dtype strings (``"fp32"``, ``"fp16"``) the kernel
            accepts. M5 is fp32-only; fp16 lands in M8.
        tolerance_abs: max absolute error allowed vs the ``torch_ref`` output
            in ``tests/test_all_variants.py``. This is the Python-side
            *parity* threshold, not the primary M1-style oracle tolerance —
            the tighter algebraic-equivalence check for M3 (1e-5 vs CPU ref)
            still lives in the ``test_attention_online_ref`` GoogleTest.
    """

    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]
    is_reference: bool
    supports_causal: bool
    supported_D: frozenset[int]
    supported_dtypes: frozenset[str]
    tolerance_abs: float


def _lazy_variants() -> dict[str, VariantSpec]:
    """Build the registry on first access.

    We import :mod:`flash_from_scratch` lazily so that plain ``import
    benchmarks.variants`` on Mac-local (no CUDA extension) still succeeds; the
    callables just raise a clean :class:`RuntimeError` when *called* without
    the extension, per ``flash_from_scratch/__init__.py``.
    """
    from flash_from_scratch import (
        attention_naive_forward,
        attention_online_ref_forward,
        flash_fwd_v1_forward,
        torch_ref_forward,
    )

    return {
        "torch_ref": VariantSpec(
            fn=torch_ref_forward,
            is_reference=True,
            supports_causal=True,
            supported_D=frozenset({32, 64, 128}),
            supported_dtypes=frozenset({"fp32"}),
            # torch_ref is compared against CPU ref (double precision NumPy) in
            # M1 tests at 1e-5. Reused here.
            tolerance_abs=1e-5,
        ),
        "attention_naive": VariantSpec(
            fn=attention_naive_forward,
            is_reference=False,
            supports_causal=True,
            supported_D=frozenset({32, 64}),
            supported_dtypes=frozenset({"fp32"}),
            tolerance_abs=5e-4,
        ),
        "attention_online_ref": VariantSpec(
            fn=attention_online_ref_forward,
            is_reference=False,
            # M3 kernel plumbs the flag but its tests keep it false; M5 keeps
            # the same convention so tolerance/regime is unchanged.
            supports_causal=False,
            supported_D=frozenset({32, 64}),
            supported_dtypes=frozenset({"fp32"}),
            # M5 pytest diffs vs `torch_ref` (not the CPU ref), whose matmul
            # order differs from online-softmax's; the tight 1e-5 algebraic-
            # equivalence bound lives in the M3 GoogleTest against the CPU
            # ref. Here we use the wider fp32-CUDA parity threshold from
            # `docs/AGENTS.md` §9.
            tolerance_abs=5e-4,
        ),
        "flash_fwd_v1": VariantSpec(
            fn=flash_fwd_v1_forward,
            is_reference=False,
            # Kernel implements causal (see flash_fwd_v1.cuh), but M5's shape
            # grid keeps is_causal=False; causal correctness lands in M7.
            supports_causal=False,
            supported_D=frozenset({32, 64}),
            supported_dtypes=frozenset({"fp32"}),
            tolerance_abs=5e-4,
        ),
    }


# Public, cached view.
_VARIANTS: dict[str, VariantSpec] | None = None


def VARIANTS() -> dict[str, VariantSpec]:
    """Return the (cached) variant registry."""
    global _VARIANTS
    if _VARIANTS is None:
        _VARIANTS = _lazy_variants()
    return _VARIANTS


def variant_names() -> list[str]:
    """Registered variant names in a stable order (reference first)."""
    v = VARIANTS()
    # Reference-first ordering makes error columns fill in a single pass.
    return sorted(v.keys(), key=lambda k: (0 if v[k].is_reference else 1, k))


def supports(variant: str, D: int, causal: bool, dtype: str = "fp32") -> bool:
    """Cheap predicate the harness uses to skip unsupported combinations."""
    spec = VARIANTS()[variant]
    if D not in spec.supported_D:
        return False
    if causal and not spec.supports_causal:
        return False
    if dtype not in spec.supported_dtypes:
        return False
    return True
