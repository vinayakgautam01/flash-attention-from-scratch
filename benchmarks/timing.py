"""CUDA-event-based timing helpers for the M5 harness.

One public entry point — :func:`time_kernel` — implements the fixed
measurement policy justified in ``theory/M5.md`` §3:

1. ``WARMUP_ITERS`` untimed launches (kernel + module cache priming, clock
   ramp-up).
2. ``TIMED_ITERS`` timed launches, each measured with a pair of
   :class:`torch.cuda.Event` timestamps.
3. Report the ``median`` and ``p95`` of the sample. Never the mean.

Failure modes we defend against:

* Forgetting ``torch.cuda.synchronize()`` before reading ``elapsed_time`` —
  we sync inside the timing loop.
* Skewed distributions from OS interference — we return the median, and the
  caller flags any row where ``(p95 - med) / med > CV_MAX``.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# Fixed for the whole project. Chosen in `theory/M5.md` §3.2 / §3.3:
# * WARMUP_ITERS=10 is enough to absorb CUDA context init + clock ramp-up.
# * TIMED_ITERS=20 makes the median well-defined (avg of ranks 10, 11) and
#   keeps a full sweep under 2 min on Colab T4.
WARMUP_ITERS: int = 10
TIMED_ITERS: int = 20

# Rows exceeding this coefficient of variation are flagged as unstable
# ((p95 - median) / median > 0.25). See `theory/M5.md` §3.4.
CV_MAX: float = 0.25


@dataclass(frozen=True)
class TimingResult:
    """Aggregate timing statistics from :func:`time_kernel`."""

    median_ms: float
    p95_ms: float
    n_samples: int

    @property
    def cv(self) -> float:
        """``(p95 - median) / median`` — the health metric flagged in plots."""
        if self.median_ms <= 0.0:
            return float("inf")
        return (self.p95_ms - self.median_ms) / self.median_ms

    @property
    def is_stable(self) -> bool:
        return self.cv < CV_MAX


def _percentile(sorted_samples: list[float], q: float) -> float:
    """Cheap percentile (no numpy dependency in the timing hot path).

    ``q`` is a fraction in [0, 1]. Uses the "nearest-rank" method: returns
    the sample at 1-indexed rank ``ceil(q * n)``, i.e. 0-indexed
    ``ceil(q * n) - 1``. At ``N = 20, q = 0.95`` this is 1-indexed rank 19 =
    ``sorted[18]`` — matching ``theory/M5.md`` §3.3. ``q = 0`` returns the
    minimum; ``q = 1`` returns the maximum.
    """
    if not sorted_samples:
        raise ValueError("cannot take percentile of empty sample")
    n = len(sorted_samples)
    idx = max(0, min(n - 1, int(round(q * n)) - 1 if q > 0 else 0))
    return sorted_samples[idx]


def time_kernel(
    fn: Callable[[], torch.Tensor],
    *,
    warmup_iters: int = WARMUP_ITERS,
    timed_iters: int = TIMED_ITERS,
) -> TimingResult:
    """Measure the runtime of ``fn`` using CUDA events + median-of-N.

    Args:
        fn: a zero-argument callable that launches one attention forward pass.
            The caller wraps their kernel + inputs in a lambda so the timing
            loop only sees a nullary launcher.
        warmup_iters: untimed launches before the timed run.
        timed_iters: launches whose runtime is recorded.

    Returns:
        :class:`TimingResult` with ``median_ms`` and ``p95_ms``.

    Raises:
        RuntimeError: if CUDA is unavailable.
    """
    import torch  # local import — this module is imported on Mac-local too

    if not torch.cuda.is_available():
        raise RuntimeError("time_kernel requires CUDA; no CUDA device visible")

    # Warm-up: kernel launches, context init, module load, clock ramp-up.
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(timed_iters):
        start.record()
        fn()
        end.record()
        # This is the sync a naive user forgets — see theory/M5.md §3.1.
        torch.cuda.synchronize()
        samples_ms.append(start.elapsed_time(end))

    samples_ms.sort()
    median_ms = statistics.median(samples_ms)
    p95_ms = _percentile(samples_ms, 0.95)
    return TimingResult(median_ms=median_ms, p95_ms=p95_ms,
                        n_samples=timed_iters)
