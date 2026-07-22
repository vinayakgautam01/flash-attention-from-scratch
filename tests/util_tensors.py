"""Fixed-seed tensor generation shared across the whole test/bench suite.

Every test in the project imports :func:`make_qkv` so that the same ``(B, H, N, D, seed)``
tuple always yields byte-identical ``(Q, K, V)``. This is what lets M2's CUDA kernel and
M1's CPU reference compare answers on the *exact same* inputs.

Decisions (see chat log 2026-07-22):

* Uses ``torch.Generator`` + ``manual_seed`` (not ``numpy.random.default_rng``) — outputs
  ``torch.Tensor`` directly, which is what downstream CUDA tests need on ``.cuda()``.
* Tensors are drawn from :math:`\\mathcal{N}(0, 1)`. Matches the "unit-variance Q/K"
  assumption behind the ``1/sqrt(D)`` scale in scaled-dot-product attention.
"""

from __future__ import annotations

import torch

__all__ = ["make_qkv"]


def make_qkv(
    B: int,
    H: int,
    N: int,
    D: int,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a reproducible ``(Q, K, V)`` triple shaped ``[B, H, N, D]``.

    Parameters
    ----------
    B, H, N, D:
        Batch / heads / sequence length / head dim, per ``docs/AGENTS.md`` §5.1.
    dtype:
        Element dtype. Defaults to ``float32`` — the M1 test grid.
    device:
        Where to allocate. Defaults to ``"cpu"``.
    seed:
        Seed for the private ``torch.Generator``. Every ``(shape, seed)`` maps to one
        specific triple across the whole project.

    Returns
    -------
    ``(Q, K, V)`` — three tensors of shape ``[B, H, N, D]`` from :math:`\\mathcal{N}(0, 1)`.

    Notes
    -----
    We create a fresh :class:`torch.Generator` each call rather than seeding the global
    RNG. Global-RNG seeding leaks state between tests and is a classic source of
    "green locally, red in CI" flakiness.
    """
    if min(B, H, N, D) <= 0:
        raise ValueError(f"all of B, H, N, D must be positive; got ({B}, {H}, {N}, {D})")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    shape = (B, H, N, D)
    q = torch.randn(shape, generator=gen, dtype=dtype, device=device)
    k = torch.randn(shape, generator=gen, dtype=dtype, device=device)
    v = torch.randn(shape, generator=gen, dtype=dtype, device=device)
    return q, k, v
