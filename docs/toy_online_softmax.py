"""M3 toy example — run the online-softmax recurrence for N=8 by hand.

Runs the recurrence in pure NumPy on the same tiny inputs the theory doc
uses, prints every intermediate, and cross-checks that:

  1. The block-wise online recurrence (`Bc = 2`, 4 blocks) matches the
     stable two-pass softmax + PV baseline to `< 1e-12` (fp64).
  2. Skipping the `alpha = exp(m_old - m_new)` rescale visibly diverges
     from the baseline (this is the figure that lands in
     `docs/online_softmax_derivation.md`).

Runs on Mac-local — no GPU, no torch, no cuda. Just NumPy.

Usage:
    python docs/toy_online_softmax.py

References:
    - theory/M3.md  §9 (this script's derivation)
    - theory/M3.md §10 (what breaks without alpha)
    - docs/MILESTONES.md §M3 TODO 4
"""

from __future__ import annotations

import numpy as np

# Inputs from theory/M3.md §9 — scores already scaled, D=1 values.
# We skip the Q @ K^T / sqrt(D) step so numbers are hand-checkable.
S = np.array([1.0, 3.0, 2.0, 5.0, 0.0, 4.0, 1.5, 2.5], dtype=np.float64)
V = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
BC = 2
N = S.shape[0]
NBLOCKS = (N + BC - 1) // BC


def stable_softmax_baseline(
    scores: np.ndarray, values: np.ndarray,
) -> tuple[float, float, float]:
    """Two-pass numerically-stable softmax + PV. The ground truth."""
    m = float(scores.max())
    exps = np.exp(scores - m)
    ell = float(exps.sum())
    o = float((exps * values).sum() / ell)
    return m, ell, o


def run_online(
    scores: np.ndarray, values: np.ndarray, *, with_alpha: bool,
) -> tuple[float, float, float, list[dict]]:
    """Block-wise online softmax + PV. Returns final (m, ell, O) plus a per-block trace."""
    m = -np.inf
    ell = 0.0
    o_tilde = 0.0
    trace: list[dict] = []

    for b in range(NBLOCKS):
        j0 = b * BC
        block_s = scores[j0:j0 + BC]
        block_v = values[j0:j0 + BC]

        m_local = float(block_s.max())
        m_new = max(m, m_local)
        # exp(-inf - m_new) = 0 for the very first block (m = -inf).
        # numpy handles this naturally; guard explicitly to keep the print tidy.
        alpha = float(np.exp(m - m_new)) if np.isfinite(m) else 0.0

        p_tilde = np.exp(block_s - m_new)          # shape (Bc,), each in (0, 1]
        ell_local = float(p_tilde.sum())
        o_local = float((p_tilde * block_v).sum())  # a single scalar since D=1

        # The whole M3 story is these two lines.
        if with_alpha:
            ell_new = alpha * ell + ell_local
            o_tilde_new = alpha * o_tilde + o_local
        else:
            # Broken variant: skip the rescale — should visibly diverge.
            ell_new = ell + ell_local
            o_tilde_new = o_tilde + o_local

        trace.append({
            "b": b,
            "j": f"{j0}..{j0 + BC - 1}",
            "block_s": block_s.copy(),
            "m_old": m,
            "m_local": m_local,
            "m_new": m_new,
            "alpha": alpha,
            "ell_local": ell_local,
            "ell_new": ell_new,
            "o_tilde_new": o_tilde_new,
        })

        m, ell, o_tilde = m_new, ell_new, o_tilde_new

    o_final = o_tilde / ell if ell != 0 else float("nan")
    return m, ell, o_final, trace


def print_trace(trace: list[dict], label: str) -> None:
    print(f"\n─── Trace: {label} " + "─" * (60 - len(label)))
    header = (
        f"{'b':>2}  {'j':<7}  {'m_old':>8}  {'m_new':>8}  "
        f"{'alpha':>10}  {'ell_local':>10}  {'ell_new':>10}  {'O_tilde':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in trace:
        m_old_str = "-inf" if not np.isfinite(r["m_old"]) else f"{r['m_old']:8.4f}"
        print(
            f"{r['b']:>2}  "
            f"{r['j']:<7}  "
            f"{m_old_str:>8}  "
            f"{r['m_new']:8.4f}  "
            f"{r['alpha']:10.6f}  "
            f"{r['ell_local']:10.6f}  "
            f"{r['ell_new']:10.6f}  "
            f"{r['o_tilde_new']:10.6f}"
        )


def main() -> None:
    bar = "═" * 75
    print(bar)
    print(" M3 toy: online softmax + PV over N=8, Bc=2 (D=1 for hand-checkability)")
    print(bar)
    print(f"scores  s = {S.tolist()}")
    print(f"values  V = {V.tolist()}")

    m_ref, ell_ref, o_ref = stable_softmax_baseline(S, V)
    print("\n─── Baseline (stable two-pass softmax + PV, ground truth) " + "─" * 15)
    print(f"  m  = {m_ref:.6f}")
    print(f"  ℓ  = {ell_ref:.6f}")
    print(f"  O* = {o_ref:.6f}")

    # ── Correct online recurrence ───────────────────────────────────────────
    m_ok, ell_ok, o_ok, trace_ok = run_online(S, V, with_alpha=True)
    print_trace(trace_ok, "with alpha (correct)")
    print(f"\n  Final: m = {m_ok:.6f}, ℓ = {ell_ok:.6f}, O = {o_ok:.6f}")
    diff_ok = abs(o_ok - o_ref)
    print(f"  |O_online − O*| = {diff_ok:.2e}   (fp64 target: < 1e-12)")
    assert diff_ok < 1e-12, (
        f"online-with-alpha diverged from baseline: {diff_ok}"
    )

    # ── Broken variant: without the rescale ─────────────────────────────────
    m_bad, ell_bad, o_bad, trace_bad = run_online(S, V, with_alpha=False)
    print_trace(trace_bad, "without alpha (broken — apples + oranges)")
    print(f"\n  Final: m = {m_bad:.6f}, ℓ = {ell_bad:.6f}, O = {o_bad:.6f}")
    diff_bad = abs(o_bad - o_ref)
    rel_bad = diff_bad / abs(o_ref) if o_ref != 0 else float("inf")
    print(f"  |O_broken − O*| = {diff_bad:.4f}   (relative {rel_bad * 100:.2f}%)")
    assert diff_bad > 1e-3, (
        f"broken variant should diverge, got |diff| = {diff_bad}"
    )

    print("\n" + bar)
    print(" All assertions passed.")
    print(
        f"   with-α: O = {o_ok:.6f}  matches baseline {o_ref:.6f} "
        f"within {diff_ok:.1e}."
    )
    print(
        f"   no-α  : O = {o_bad:.6f}  differs from baseline by "
        f"{rel_bad * 100:.2f}%."
    )
    print(bar)


if __name__ == "__main__":
    main()
