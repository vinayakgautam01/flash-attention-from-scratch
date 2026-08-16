"""M5 plot script — consumes ``benchmarks/results/all.csv`` and writes:

* ``docs/plots/runtime_vs_N.png``  — runtime (ms) vs N, one line per variant.
* ``docs/plots/speedup_vs_naive.png`` — variant runtime / naive runtime vs N.
* ``docs/plots/error_histogram.png`` — histogram of ``max_abs_err`` per variant.

Usage::

    python -m benchmarks.plot
    # or
    python benchmarks/plot.py --csv path/to/all.csv

Kept intentionally minimal — matplotlib only, one figure per file, no seaborn.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "benchmarks" / "results" / "all.csv"
PLOT_DIR = REPO_ROOT / "docs" / "plots"

# Fixed rendering order so plots are stable across runs (reference first).
# v1 and v2 are adjacent on purpose: MILESTONES §M6 asks for the two Flash
# variants to read side-by-side in the legend.
VARIANT_ORDER = [
    "torch_ref",
    "attention_naive",
    "attention_online_ref",
    "flash_fwd_v1",
    "flash_fwd_v2",
]


def _load(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _group_by_variant(rows: list[dict], D_filter: int | None = 64):
    """Return {variant: sorted list of (N, row)} at fixed D (default 64)."""
    out: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for r in rows:
        if D_filter is not None and int(r["D"]) != D_filter:
            continue
        out[r["variant"]].append((int(r["N"]), r))
    for v in out:
        out[v].sort(key=lambda t: t[0])
    return out


def _order(variants: list[str]) -> list[str]:
    known = [v for v in VARIANT_ORDER if v in variants]
    unknown = sorted(set(variants) - set(VARIANT_ORDER))
    return known + unknown


def plot_runtime_vs_N(rows: list[dict], out_path: Path, D: int = 64) -> None:
    grouped = _group_by_variant(rows, D_filter=D)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for variant in _order(list(grouped.keys())):
        pts = grouped[variant]
        xs = [n for n, _ in pts]
        ys = [float(r["runtime_ms_median"]) for _, r in pts]
        ax.plot(xs, ys, marker="o", label=variant)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(f"N (sequence length, D={D})")
    ax.set_ylabel("runtime — median of 20 (ms)")
    ax.set_title("Attention forward runtime vs N")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_speedup_vs_naive(rows: list[dict], out_path: Path, D: int = 64) -> None:
    grouped = _group_by_variant(rows, D_filter=D)
    naive = {n: float(r["runtime_ms_median"])
             for n, r in grouped.get("attention_naive", [])}
    if not naive:
        # Nothing to compare against — skip cleanly rather than error, but say
        # so on stderr so the missing plot has a documented reason.
        print(f"[plot] skipping speedup plot at D={D}: "
              "no `attention_naive` rows in the CSV.",
              file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for variant in _order(list(grouped.keys())):
        if variant == "attention_naive":
            continue
        pts = grouped[variant]
        xs = [n for n, _ in pts if n in naive]
        ys = [naive[n] / float(r["runtime_ms_median"])
              for n, r in pts if n in naive]
        if xs:
            ax.plot(xs, ys, marker="o", label=variant)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(f"N (sequence length, D={D})")
    ax.set_ylabel("speedup vs attention_naive (higher is better)")
    ax.set_title("Speedup over the naive baseline")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_histogram(rows: list[dict], out_path: Path) -> None:
    per_variant: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        per_variant[r["variant"]].append(float(r["max_abs_err_vs_ref"]))

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for variant in _order(list(per_variant.keys())):
        errs = [e for e in per_variant[variant] if e > 0]
        if not errs:
            continue
        ax.hist(errs, bins=20, alpha=0.5, label=variant, edgecolor="black")
    ax.set_xscale("log")
    ax.set_xlabel("max abs error vs reference")
    ax.set_ylabel("count of shapes")
    ax.set_title("Correctness distribution across the shape grid")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M5 plot script.")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-dir", type=Path, default=PLOT_DIR)
    p.add_argument("--D", type=int, default=64,
                   help="Head dim to plot for runtime/speedup (default 64).")
    args = p.parse_args(argv)

    if not args.csv.exists():
        raise FileNotFoundError(
            f"{args.csv} not found — run `python -m benchmarks.bench_attention` "
            "on a GPU host first."
        )
    rows = _load(args.csv)
    if not rows:
        raise RuntimeError(f"{args.csv} is empty")

    plot_runtime_vs_N(rows, args.out_dir / "runtime_vs_N.png", D=args.D)
    plot_speedup_vs_naive(rows, args.out_dir / "speedup_vs_naive.png", D=args.D)
    plot_error_histogram(rows, args.out_dir / "error_histogram.png")
    print(f"→ wrote plots into {args.out_dir.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
