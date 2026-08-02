"""Analytical HBM memory accounting for the M2 naive baseline.

Not a benchmark (that's M5). This script computes — without touching a GPU —
the two things `docs/memory_analysis.md` needs to make concrete:

  1. Peak transient device memory (the S and P allocations that live for the
     duration of one attention forward pass).
  2. Total HBM traffic (bytes read + bytes written across all three kernels).

Both are pure functions of (B, H, N, D, dtype). We log a small grid over
`N ∈ {128, 256, 512, 1024, 2048, 4096}` at the canonical `B=H=1, D=64, fp32`,
matching the MILESTONES §M2 verification plan.

See `theory/M2.md` §9 for the derivation of these formulas.

Usage:
    python benchmarks/naive_memory.py
    # → writes benchmarks/results/naive_memory.csv
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = REPO_ROOT / "benchmarks" / "results" / "naive_memory.csv"

# fp32 in M2 — matches docs/AGENTS.md §9.
BYTES_PER_ELEM = 4

# Verification-grid N values, from MILESTONES §M2 TODO 4.
N_VALUES = [128, 256, 512, 1024, 2048, 4096]

# Canonical shape: B=H=1, D=64. Matches the shape the M9 Nsight report will
# use (see MILESTONES §M9 TODO 1). Kept configurable at module level so callers
# can sweep if they want.
B_DEFAULT = 1
H_DEFAULT = 1
D_DEFAULT = 64


@dataclass(frozen=True)
class NaiveMemoryRow:
    """One row of the memory-analysis CSV. See theory/M2.md §9.1 for formulas."""

    B: int
    H: int
    N: int
    D: int
    dtype_bytes: int

    @property
    def s_bytes(self) -> int:
        # S has shape [B, H, N, N].
        return self.B * self.H * self.N * self.N * self.dtype_bytes

    @property
    def p_bytes(self) -> int:
        # P has shape [B, H, N, N] — same size as S.
        return self.s_bytes

    @property
    def transient_bytes_total(self) -> int:
        # Peak transient device allocation: S and P live simultaneously
        # between the qk_matmul and row_softmax launches. Q, K, V, O are
        # caller-provided and not counted here.
        return self.s_bytes + self.p_bytes

    @property
    def qk_bytes(self) -> int:
        # qk_matmul reads Q + K, writes S.
        bh_nd = self.B * self.H * self.N * self.D
        bh_nn = self.B * self.H * self.N * self.N
        return (2 * bh_nd + bh_nn) * self.dtype_bytes

    @property
    def softmax_bytes(self) -> int:
        # row_softmax reads S, writes P.
        return 2 * self.B * self.H * self.N * self.N * self.dtype_bytes

    @property
    def pv_bytes(self) -> int:
        # pv_matmul reads P + V, writes O.
        bh_nn = self.B * self.H * self.N * self.N
        bh_nd = self.B * self.H * self.N * self.D
        return (bh_nn + 2 * bh_nd) * self.dtype_bytes

    @property
    def hbm_traffic_bytes(self) -> int:
        return self.qk_bytes + self.softmax_bytes + self.pv_bytes


def rows(
    N_values: list[int] = N_VALUES,
    B: int = B_DEFAULT,
    H: int = H_DEFAULT,
    D: int = D_DEFAULT,
    dtype_bytes: int = BYTES_PER_ELEM,
) -> list[NaiveMemoryRow]:
    return [NaiveMemoryRow(B, H, N, D, dtype_bytes) for N in N_values]


def write_csv(path: Path, rows_: list[NaiveMemoryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "B", "H", "N", "D", "dtype_bytes",
            "s_bytes", "p_bytes", "transient_bytes_total",
            "qk_bytes", "softmax_bytes", "pv_bytes",
            "hbm_traffic_bytes",
        ])
        for r in rows_:
            w.writerow([
                r.B, r.H, r.N, r.D, r.dtype_bytes,
                r.s_bytes, r.p_bytes, r.transient_bytes_total,
                r.qk_bytes, r.softmax_bytes, r.pv_bytes,
                r.hbm_traffic_bytes,
            ])


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def main() -> None:
    rs = rows()
    write_csv(OUTPUT_CSV, rs)

    # Human-readable summary — makes the log useful even without opening the CSV.
    print(f"→ wrote {OUTPUT_CSV.relative_to(REPO_ROOT)}")
    print(f"  B=H={B_DEFAULT}, D={D_DEFAULT}, fp32")
    print()
    print(f"  {'N':>6}  {'S':>10}  {'transient':>12}  {'HBM traffic':>14}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*14}")
    for r in rs:
        print(
            f"  {r.N:>6}  {_fmt_bytes(r.s_bytes):>10}  "
            f"{_fmt_bytes(r.transient_bytes_total):>12}  "
            f"{_fmt_bytes(r.hbm_traffic_bytes):>14}"
        )


if __name__ == "__main__":
    main()
