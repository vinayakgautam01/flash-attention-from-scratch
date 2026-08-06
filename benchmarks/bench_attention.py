"""M5 benchmark sweep driver — writes ``benchmarks/results/all.csv``.

Iterates over the (variant × shape) grid, times each launch with
:func:`benchmarks.timing.time_kernel`, records analytical HBM bytes and
effective TFLOPs from :mod:`benchmarks.perf_model`, and diffs the output
against the reference variant to fill the ``max_abs_err`` / ``max_rel_err``
columns. Every claim in the README traces back to a row here.

CSV schema (frozen per MILESTONES §M5):

    variant, B, H, N, D, causal, dtype,
    runtime_ms_median, runtime_ms_p95,
    hbm_bytes_est, peak_alloc_bytes, tflops_effective,
    max_abs_err_vs_ref, max_rel_err_vs_ref,
    git_sha, gpu_name

Usage::

    python -m benchmarks.bench_attention

(Invoke via ``-m`` from the repo root; direct-script invocation fails because
``benchmarks.perf_model`` / ``benchmarks.variants`` need ``benchmarks`` on
``sys.path``.)

See ``theory/M5.md`` §4 for the per-column derivations.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from benchmarks.perf_model import hbm_bytes_est, tflops_effective
from benchmarks.timing import time_kernel
from benchmarks.variants import VARIANTS, supports, variant_names

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_CSV = REPO_ROOT / "benchmarks" / "results" / "all.csv"

# Shape grid — minimal per user decision in the M5 brainstorm. B/H/causal
# expand in M7; fp16 expands in M8.
DEFAULT_B: tuple[int, ...] = (1,)
DEFAULT_H: tuple[int, ...] = (1,)
DEFAULT_N: tuple[int, ...] = (128, 256, 512, 1024, 2048)
DEFAULT_D: tuple[int, ...] = (32, 64)
DEFAULT_CAUSAL: tuple[bool, ...] = (False,)
DEFAULT_DTYPE: tuple[str, ...] = ("fp32",)

CSV_COLUMNS = [
    "variant", "B", "H", "N", "D", "causal", "dtype",
    "runtime_ms_median", "runtime_ms_p95",
    "hbm_bytes_est", "peak_alloc_bytes", "tflops_effective",
    "max_abs_err_vs_ref", "max_rel_err_vs_ref",
    "git_sha", "gpu_name",
]

# The reference variant every other variant is diffed against. Kept as a
# module-level constant so both the driver and its tests can agree.
REFERENCE_VARIANT = "torch_ref"

# Fixed RNG seed for reproducible inputs across runs. Matches `util_tensors.py`.
RNG_SEED = 0


@dataclass(frozen=True)
class Shape:
    B: int
    H: int
    N: int
    D: int
    causal: bool
    dtype: str


def git_sha() -> str:
    """Short git SHA for the ``git_sha`` column; ``""`` outside a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def gpu_name() -> str:
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def _make_inputs(shape: Shape) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fresh random Q/K/V tensors on the current CUDA device.

    Reuses :func:`tests.util_tensors.make_qkv` so bench and tests share
    identical inputs at every shape (see user rule "Reuse existing patterns").
    """
    from tests.util_tensors import make_qkv
    return make_qkv(shape.B, shape.H, shape.N, shape.D,
                    device="cuda", seed=RNG_SEED)


def _max_abs_rel(a: torch.Tensor, b: torch.Tensor,
                 eps: float = 1e-12) -> tuple[float, float]:
    """Return ``(max_abs_err, max_rel_err)`` between two same-shape tensors."""
    diff = (a.float() - b.float()).abs()
    abs_err = float(diff.max().item())
    rel_err = float((diff / (b.float().abs() + eps)).max().item())
    return abs_err, rel_err


@dataclass(frozen=True)
class RunContext:
    """Sweep-wide constants pinned once so every row shares them."""

    git_sha: str
    gpu_name: str


def _measure_row(variant: str, shape: Shape,
                 ref_O: torch.Tensor | None,
                 ctx: RunContext) -> dict[str, object]:
    """Time a single (variant, shape) pair and return the CSV row dict."""
    import torch
    Q, K, V = _make_inputs(shape)
    spec = VARIANTS()[variant]

    def _launch() -> torch.Tensor:
        return spec.fn(Q, K, V, shape.causal)

    # Peak-alloc bracket. `reset_peak_memory_stats` gives us a per-launch delta
    # over the current baseline. `baseline` is measured *after* Q/K/V are
    # already resident, so `peak_delta` captures allocations made *inside*
    # `_launch` — i.e. `O` (always ≈ T bytes) plus any transient
    # intermediates the kernel allocates (`S` and `P` for naive, none for the
    # Flash / online-ref variants). See `theory/M5.md` §4.3.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    out = _launch()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    peak_delta = max(0, peak - baseline)

    # Timing loop (median-of-20 with 10 warmups).
    timing = time_kernel(_launch)

    # Error vs reference. Reference variant diffs against itself → zero.
    if ref_O is None:
        abs_err, rel_err = 0.0, 0.0
    else:
        abs_err, rel_err = _max_abs_rel(out, ref_O)

    return {
        "variant": variant,
        "B": shape.B, "H": shape.H, "N": shape.N, "D": shape.D,
        "causal": shape.causal, "dtype": shape.dtype,
        "runtime_ms_median": f"{timing.median_ms:.6f}",
        "runtime_ms_p95": f"{timing.p95_ms:.6f}",
        "hbm_bytes_est": hbm_bytes_est(variant, shape.B, shape.H, shape.N, shape.D),
        "peak_alloc_bytes": peak_delta,
        "tflops_effective":
            f"{tflops_effective(shape.B, shape.H, shape.N, shape.D, timing.median_ms):.4f}",
        "max_abs_err_vs_ref": f"{abs_err:.3e}",
        "max_rel_err_vs_ref": f"{rel_err:.3e}",
        "git_sha": ctx.git_sha,
        "gpu_name": ctx.gpu_name,
    }


def sweep(shapes: list[Shape], variants: list[str] | None = None) -> list[dict]:
    """Run the full sweep and return all rows (also emit progress to stderr)."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("bench_attention requires CUDA; no device visible")

    # Pin `git_sha` and `gpu_name` once per sweep — they're constants across
    # all rows, and `git_sha()` spawns a subprocess so avoiding 40× calls is
    # a cheap cleanup.
    ctx = RunContext(git_sha=git_sha(), gpu_name=gpu_name())

    variant_order = variants or variant_names()
    if REFERENCE_VARIANT not in variant_order:
        raise ValueError(
            f"reference variant {REFERENCE_VARIANT!r} must be in the sweep "
            "so error columns can be filled"
        )
    # Reference first so downstream variants have `ref_O` cached.
    variant_order = ([REFERENCE_VARIANT] +
                     [v for v in variant_order if v != REFERENCE_VARIANT])

    rows: list[dict] = []
    for shape in shapes:
        # Skip shapes no variant supports (defensive; shouldn't happen at M5's grid).
        if not any(supports(v, shape.D, shape.causal, shape.dtype)
                   for v in variant_order):
            continue

        # Compute the reference output once per shape (used for every other
        # variant's error columns).
        Qr, Kr, Vr = _make_inputs(shape)
        ref_spec = VARIANTS()[REFERENCE_VARIANT]
        ref_O = ref_spec.fn(Qr, Kr, Vr, shape.causal).detach().clone()

        for variant in variant_order:
            if not supports(variant, shape.D, shape.causal, shape.dtype):
                print(f"[skip] {variant} does not support "
                      f"D={shape.D}, causal={shape.causal}, dtype={shape.dtype}",
                      file=sys.stderr)
                continue

            ref_for_row = None if variant == REFERENCE_VARIANT else ref_O
            row = _measure_row(variant, shape, ref_for_row, ctx)
            rows.append(row)
            print(
                f"[ok]   {variant:>22s}  "
                f"B={shape.B} H={shape.H} N={shape.N:>5d} D={shape.D:>3d}  "
                f"med={row['runtime_ms_median']} ms  "
                f"p95={row['runtime_ms_p95']} ms  "
                f"abs_err={row['max_abs_err_vs_ref']}",
                file=sys.stderr,
            )
    return rows


def write_csv(rows: list[dict], out_path: Path = OUTPUT_CSV) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_grid(args: argparse.Namespace) -> list[Shape]:
    """Cartesian product of the CLI-supplied axes → list of Shape."""
    shapes: list[Shape] = []
    for B in args.B:
        for H in args.H:
            for N in args.N:
                for D in args.D:
                    for causal in args.causal:
                        for dtype in args.dtype:
                            shapes.append(Shape(B, H, N, D, causal, dtype))
    return shapes


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _bool_list(s: str) -> list[bool]:
    return [x.lower() in ("1", "true", "yes") for x in s.split(",") if x]


def _str_list(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M5 benchmark sweep.")
    p.add_argument("--B",      type=_int_list,  default=list(DEFAULT_B))
    p.add_argument("--H",      type=_int_list,  default=list(DEFAULT_H))
    p.add_argument("--N",      type=_int_list,  default=list(DEFAULT_N))
    p.add_argument("--D",      type=_int_list,  default=list(DEFAULT_D))
    p.add_argument("--causal", type=_bool_list, default=list(DEFAULT_CAUSAL))
    p.add_argument("--dtype",  type=_str_list,  default=list(DEFAULT_DTYPE))
    p.add_argument("--out",    type=Path,       default=OUTPUT_CSV)
    p.add_argument("--variants", type=_str_list, default=None,
                   help="Comma-separated variant names; default = all registered.")
    args = p.parse_args(argv)

    shapes = build_grid(args)
    rows = sweep(shapes, args.variants)
    write_csv(rows, args.out)
    print(f"→ wrote {args.out.relative_to(REPO_ROOT)} "
          f"({len(rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
