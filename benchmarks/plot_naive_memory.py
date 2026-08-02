"""Log-log HBM memory plot for the M2 naive baseline.

Reads `benchmarks/results/naive_memory.csv` (produced by `naive_memory.py`) and
renders `docs/plots/naive_memory.png` — a two-line plot showing:

  1. Peak transient device allocation (S + P).
  2. Total HBM traffic per forward pass.

Both are plotted against N on log-log axes; both should show a clean slope of 2
(quadratic in N). The visual is what MILESTONES §M2 verification calls the
"HBM footprint vs N" plot.

See `theory/M2.md` §9 for the theory.

Usage:
    python benchmarks/plot_naive_memory.py
    # → writes docs/plots/naive_memory.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "benchmarks" / "results" / "naive_memory.csv"
OUTPUT_PNG = REPO_ROOT / "docs" / "plots" / "naive_memory.png"


def read_csv(path: Path) -> tuple[list[int], list[int], list[int]]:
    Ns: list[int] = []
    transient: list[int] = []
    hbm: list[int] = []
    with path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            Ns.append(int(row["N"]))
            transient.append(int(row["transient_bytes_total"]))
            hbm.append(int(row["hbm_traffic_bytes"]))
    return Ns, transient, hbm


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"missing {CSV_PATH.relative_to(REPO_ROOT)} — "
            f"run `python benchmarks/naive_memory.py` first."
        )

    Ns, transient, hbm = read_csv(CSV_PATH)

    # Scale bytes → MB for readability. Log-log preserves the slope regardless
    # of unit, so the slope-2 claim is unchanged.
    to_mb = 1.0 / (1024 * 1024)
    transient_mb = [b * to_mb for b in transient]
    hbm_mb = [b * to_mb for b in hbm]

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=120)

    ax.loglog(Ns, transient_mb, "o-", label="peak transient (S + P)", linewidth=1.8)
    ax.loglog(Ns, hbm_mb, "s--", label="HBM traffic (per forward)", linewidth=1.8)

    # Slope-2 reference line, anchored at the first point of the transient curve.
    # Purpose: make the "quadratic in N" claim visually falsifiable, not asserted.
    ref_scale = transient_mb[0] / (Ns[0] ** 2)
    slope2 = [ref_scale * (n ** 2) for n in Ns]
    ax.loglog(Ns, slope2, ":", color="gray", linewidth=1.0,
              label="slope 2 reference")

    ax.set_xlabel("N (sequence length)")
    ax.set_ylabel("bytes (MB)")
    ax.set_title("Naive attention: HBM footprint vs N  (B=H=1, D=64, fp32)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)

    # Annotate the rightmost point of transient so the reader sees the number.
    ax.annotate(
        f"{transient_mb[-1]:.0f} MB",
        xy=(Ns[-1], transient_mb[-1]),
        xytext=(6, -4),
        textcoords="offset points",
        fontsize=9,
        color="tab:blue",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG)
    print(f"→ wrote {OUTPUT_PNG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
