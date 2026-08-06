"""Analytical performance model for the M5 benchmark harness.

Pure functions — no CUDA, no torch. Runs on Mac-local, importable by both the
sweep driver (`benchmarks/bench_attention.py`) and the pytest suite.

Every formula here is derived in ``theory/M5.md`` §4:

* :func:`flops_attention_forward` — the standard ``4 * B * H * N^2 * D`` count
  used across the FlashAttention / SDPA literature (§4.2).
* :func:`hbm_bytes_naive` — three-kernel materialize-S-and-P baseline (§4.1).
* :func:`hbm_bytes_online_ref` — one-CTA-per-row streaming (K/V re-loaded per
  row) (§4.1).
* :func:`hbm_bytes_flash_v1` — one-CTA-per-Q-tile tiled kernel (§4.1).
* :func:`hbm_bytes_est` — the dispatcher a benchmark row calls, so adding a
  new variant is one branch here.

Sizes are returned in bytes; the harness formats them for display and writes
integers into the CSV.
"""

from __future__ import annotations

from typing import Final

# Element size in bytes. M5 is fp32-only; fp16 arrives in M8.
FP32_BYTES: Final[int] = 4


# --------------------------------------------------------------------------- #
# FLOPs — algorithmic count, identical across variants (theory/M5.md §4.2).   #
# --------------------------------------------------------------------------- #

def flops_attention_forward(B: int, H: int, N: int, D: int) -> int:
    """Return ``4 * B * H * N^2 * D`` — the algorithmic FLOP count of exact SDPA
    forward.

    Two matmuls (QK^T and PV), each ``2 * N^2 * D`` per head, times ``B * H``
    heads. The softmax term is ``O(B H N^2)`` (no D factor) and is dropped by
    convention — it is ~ ``D`` times smaller than the matmuls (~1600× at
    ``D=64``). Every FlashAttention / SDPA paper uses the same convention.
    """
    return 4 * B * H * N * N * D


def tflops_effective(B: int, H: int, N: int, D: int, runtime_ms: float) -> float:
    """Effective (algorithmic) TFLOPs given a measured median runtime in ms.

    ``TFLOPs_eff = FLOPs / (runtime_s * 1e12) = FLOPs / (runtime_ms * 1e9)``.

    "Effective" because we normalize by the algorithmic FLOP budget, not the
    actual FLOPs executed — otherwise Flash v1 (which does *more* actual FLOPs
    than naive because of partial-dot-product recomputation) would score lower
    than naive despite being faster. See ``theory/M5.md`` §4.2.
    """
    if runtime_ms <= 0.0:
        return 0.0
    flops = flops_attention_forward(B, H, N, D)
    return flops / (runtime_ms * 1.0e9)


# --------------------------------------------------------------------------- #
# HBM traffic — per-variant closed forms (theory/M5.md §4.1).                 #
# --------------------------------------------------------------------------- #

def _T_bytes(B: int, H: int, N: int, D: int, dtype_bytes: int) -> int:
    """Size of one of Q, K, V, O in bytes: ``4 * B * H * N * D`` (fp32)."""
    return B * H * N * D * dtype_bytes


def _S_bytes(B: int, H: int, N: int, dtype_bytes: int) -> int:
    """Size of the transient ``[B, H, N, N]`` matrix in bytes."""
    return B * H * N * N * dtype_bytes


def hbm_bytes_naive(B: int, H: int, N: int, D: int,
                    dtype_bytes: int = FP32_BYTES) -> int:
    """M2 baseline: three kernels each round-trip through HBM.

    ``qk_matmul``  reads Q + K, writes S  →  ``2T + S``
    ``row_softmax`` reads S, writes P      →  ``2S``
    ``pv_matmul``  reads P + V, writes O  →  ``S + 2T``

    Total: ``4T + 4S``. See ``theory/M5.md`` §4.1 and
    ``benchmarks/naive_memory.py`` (M2) for the original derivation.
    """
    T = _T_bytes(B, H, N, D, dtype_bytes)
    S = _S_bytes(B, H, N, dtype_bytes)
    return 4 * T + 4 * S


def hbm_bytes_online_ref(B: int, H: int, N: int, D: int,
                         dtype_bytes: int = FP32_BYTES) -> int:
    """M3 online-softmax reference: one CTA per query row.

    Each of the ``B * H * N`` CTAs independently loads all N K-columns and V-
    columns from HBM (no sharing across rows). Q is loaded once per row;
    O is written once per row.

        Q reads  =  T
        K reads  =  N * T   (each row re-loads all N K-columns)
        V reads  =  N * T
        O writes =  T

    Total: ``(2 + 2N) * T``. See ``theory/M5.md`` §4.1.
    """
    T = _T_bytes(B, H, N, D, dtype_bytes)
    return (2 + 2 * N) * T


def hbm_bytes_flash_v1(B: int, H: int, N: int, D: int,
                       Br: int, dtype_bytes: int = FP32_BYTES) -> int:
    """M4 Flash v1: one CTA per Q-tile of Br rows.

    Q is loaded once (each Q-tile by its owning CTA). K/V are re-loaded once
    per Q-tile — i.e. ``ceil(N / Br)`` full copies of each.

        Q reads  =  T
        K reads  =  ceil(N / Br) * T
        V reads  =  ceil(N / Br) * T
        O writes =  T

    Total: ``(2 + 2 * ceil(N / Br)) * T``. See ``theory/M5.md`` §4.1 and
    ``theory/M4.md`` §10 for the K/V re-load argument.
    """
    T = _T_bytes(B, H, N, D, dtype_bytes)
    tiles = (N + Br - 1) // Br
    return (2 + 2 * tiles) * T


# --------------------------------------------------------------------------- #
# Dispatcher used by the sweep driver.                                        #
# --------------------------------------------------------------------------- #

# Import Br from the extension when available; fall back to the header value
# so pure-Python callers (Mac-local) still get the right number.
try:
    from flash_from_scratch import kFlashV1Br as _FLASH_V1_BR  # type: ignore[no-redef]
except ImportError:
    _FLASH_V1_BR = 32


def hbm_bytes_est(variant: str, B: int, H: int, N: int, D: int,
                  dtype_bytes: int = FP32_BYTES) -> int:
    """Return the analytical HBM-bytes estimate for ``variant`` at the given
    shape. Called by the benchmark driver to fill the ``hbm_bytes_est`` column.

    ``torch_ref`` is treated as-if-it-were Flash v1 for the purpose of this
    estimate — a rough proxy that captures the "tiled kernel" regime PyTorch's
    SDPA fastpath uses. This is an *estimate*, not a measurement; the M9 Nsight
    cross-check is where measured values enter the record.
    """
    if variant == "attention_naive":
        return hbm_bytes_naive(B, H, N, D, dtype_bytes)
    if variant == "attention_online_ref":
        return hbm_bytes_online_ref(B, H, N, D, dtype_bytes)
    if variant in ("flash_fwd_v1", "torch_ref"):
        return hbm_bytes_flash_v1(B, H, N, D, _FLASH_V1_BR, dtype_bytes)
    raise KeyError(f"unknown variant for HBM estimate: {variant!r}")


# --------------------------------------------------------------------------- #
# Byte-size pretty-printer (shared with `benchmarks/naive_memory.py` output). #
# --------------------------------------------------------------------------- #

def fmt_bytes(n: int) -> str:
    """Human-readable byte count. E.g. ``264 MB`` for 277 M."""
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0:
            return f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.1f} PB"
