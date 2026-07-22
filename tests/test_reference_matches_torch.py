"""M1 gate: the NumPy reference matches ``torch.nn.functional.scaled_dot_product_attention``.

This is the file that must be green for M1 to be "done." If PyTorch's SDPA and our
independently-written NumPy reference agree to ``1e-5`` on the M1 shape grid, our
oracle is trustworthy and downstream CUDA kernels can be diffed against it.

Tolerance comes from ``docs/AGENTS.md`` §9 (CPU ref vs PyTorch, fp32).

Shape grid: ``N x D x causal = 3 x 3 x 2 = 18`` cases (see ``theory/M1.md`` §10).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.reference import attention_reference
from tests.util_tensors import make_qkv

# Kept tiny on purpose — see theory/M1.md §10. NumPy ref runs on the whole grid in <1s.
N_VALUES = (8, 16, 32)
D_VALUES = (8, 16, 32)
CAUSAL_VALUES = (False, True)

# Fixed at 1e-5 per docs/AGENTS.md §9 for the "CPU ref vs PyTorch, fp32" regime.
ABS_TOL = 1e-5

# One (B, H) is enough for correctness — see theory/M1.md §10 ("big sizes buy nothing
# for correctness at this stage"). Batch/head axes get exercised in M7.
B = 2
H = 4


@pytest.mark.parametrize("is_causal", CAUSAL_VALUES, ids=lambda c: f"causal={c}")
@pytest.mark.parametrize("d", D_VALUES, ids=lambda d: f"D={d}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda n: f"N={n}")
def test_reference_matches_torch_sdpa(n: int, d: int, is_causal: bool) -> None:
    q, k, v = make_qkv(B, H, n, d, dtype=torch.float32, seed=0)

    o_ref = attention_reference(q, k, v, is_causal=is_causal)
    o_torch = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    assert o_ref.shape == o_torch.shape == (B, H, n, d)
    max_abs_err = (o_ref - o_torch).abs().max().item()
    assert max_abs_err < ABS_TOL, (
        f"NumPy ref disagrees with torch.SDPA at N={n}, D={d}, causal={is_causal}: "
        f"max_abs_err={max_abs_err:.3e} (tol={ABS_TOL:.0e})"
    )


def test_causal_zeroes_future_positions() -> None:
    """Sanity check: with is_causal=True, query 0's output is a linear combination
    of value row 0 alone (all other positions are masked to -inf)."""
    q, k, v = make_qkv(1, 1, 4, 8, dtype=torch.float32, seed=1)
    out = attention_reference(q, k, v, is_causal=True)
    # For i=0, only j=0 is unmasked, so softmax is [1, 0, 0, 0] and O[0] == V[0].
    torch.testing.assert_close(out[0, 0, 0], v[0, 0, 0], rtol=0, atol=1e-6)
