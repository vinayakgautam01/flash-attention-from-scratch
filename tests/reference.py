"""Pure-NumPy reference implementation of scaled dot-product attention.

This is the **oracle**: a stupid, slow, obviously-correct implementation that every
downstream CUDA kernel gets diffed against. Its whole job is to be numerically
trustworthy — not fast.

Decisions (see chat log 2026-07-22 and ``theory/M1.md``):

* Computes internally in ``float64`` regardless of input dtype. fp64 rounding error
  (~1e-16 per op) is effectively zero from fp32's perspective (~1e-7 per op), so any
  disagreement with the reference is the kernel's fault.
* Uses ``np.einsum`` for the two matmuls (**pragmatic** style — see ``theory/M1.md`` §9.2).
  The tautology risk is mitigated by the fp64 internal precision and by the fact that
  PyTorch's SDPA oracle uses a totally independent code path.
* Numerically stable softmax: subtract the row max before ``exp``.
* Causal mask uses ``-inf`` (not ``-1e9``). ``exp(-inf) = 0`` gives exact zero weight.

Contract mirrors the C++ header ``csrc/attention_cpu_ref.hpp`` — same algorithm, same
argument order, same causal semantics.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["attention_reference"]


def attention_reference(
    q: torch.Tensor | np.ndarray,
    k: torch.Tensor | np.ndarray,
    v: torch.Tensor | np.ndarray,
    is_causal: bool = False,
) -> torch.Tensor:
    """Reference forward pass of scaled dot-product attention.

    Parameters
    ----------
    q, k, v:
        Query / key / value tensors, shape ``[B, H, N, D]``. Accepts either
        ``torch.Tensor`` or ``np.ndarray``; the return type is always ``torch.Tensor``
        matching the input dtype/device of ``q``.
    is_causal:
        If ``True``, mask entries ``S[i, j] = -inf`` for ``j > i`` before softmax so
        each query position only attends to itself and prior positions.

    Returns
    -------
    Output ``O`` of shape ``[B, H, N, D]``.

    Notes
    -----
    Computes as:

    .. math::
        S = \\frac{Q K^{\\top}}{\\sqrt{D}}, \\quad
        P = \\mathrm{softmax}(S), \\quad
        O = P V

    All internal math is in ``float64``.
    """
    # Capture return-type metadata before dropping into NumPy.
    return_torch = isinstance(q, torch.Tensor)
    out_dtype = q.dtype if return_torch else None
    out_device = q.device if return_torch else None

    q64 = _to_float64_ndarray(q)
    k64 = _to_float64_ndarray(k)
    v64 = _to_float64_ndarray(v)

    if q64.shape != k64.shape or q64.shape != v64.shape:
        raise ValueError(
            f"Q/K/V must share shape [B, H, N, D]; got {q64.shape}, {k64.shape}, {v64.shape}"
        )
    if q64.ndim != 4:
        raise ValueError(f"expected 4-D tensors [B, H, N, D]; got ndim={q64.ndim}")

    _, _, n, d = q64.shape
    scale = 1.0 / np.sqrt(d)

    # S = Q @ K^T * scale, per (b, h). einsum keeps (b, h) implicit and does the
    # inner product over the head-dim axis. Shape: [B, H, N_q, N_k].
    scores = np.einsum("bhid,bhjd->bhij", q64, k64) * scale

    if is_causal:
        # Mask strict upper triangle to -inf (see theory/M1.md §5).
        # ``np.triu(..., k=1)`` gives a [N, N] bool mask of positions where j > i.
        causal_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        scores = np.where(causal_mask, -np.inf, scores)

    # Numerically stable softmax across the last dim (keys).
    #
    # Subtracting the row max before exp keeps every exponent in (-inf, 0], so exp
    # is in (0, 1]. See theory/M1.md §4 for why this is exact math + safe fp.
    row_max = scores.max(axis=-1, keepdims=True)
    # For a fully-masked row (would only happen if N=0 or a pathological all-inf
    # input — neither occurs in this project), row_max would be -inf. We don't
    # guard against it since it can't happen given our shape grid.
    stable = scores - row_max
    exp_scores = np.exp(stable)
    denom = exp_scores.sum(axis=-1, keepdims=True)
    probs = exp_scores / denom  # shape [B, H, N, N]

    # O = P @ V, per (b, h).
    out64 = np.einsum("bhij,bhjd->bhid", probs, v64)

    if return_torch:
        return torch.from_numpy(out64).to(dtype=out_dtype, device=out_device)
    return torch.from_numpy(out64)  # dtype float64 by default


def _to_float64_ndarray(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert any input to a contiguous ``float64`` NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float64).numpy()
    return np.ascontiguousarray(x, dtype=np.float64)
