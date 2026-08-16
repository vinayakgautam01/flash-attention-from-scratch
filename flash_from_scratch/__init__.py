"""flash_from_scratch — Python entry point for the M5 harness.

Exposes every CUDA kernel behind a common ``fn(Q, K, V, is_causal) -> O``
signature, layered on top of the pybind11 shim in ``csrc/bindings.cpp``.

Design contract (see ``theory/M5.md`` §6.2):

* ``Q, K, V`` are contiguous ``[B, H, N, D]`` ``torch.float32`` CUDA tensors
  living on the same device.
* Output ``O`` is a fresh tensor with the same shape and dtype as ``Q``.
* Each variant may impose further constraints (e.g. ``D ∈ {32, 64}`` for
  ``flash_fwd_v1``); those are enforced by the shim and re-checked here.

Mac-local behaviour: importing the module succeeds even when the compiled
``_C`` extension is missing (there is no CUDA toolkit on the developer laptop).
The module-level flag :data:`HAS_CUDA_EXT` reports which regime we are in,
and every ``*_forward`` function raises a clear :class:`RuntimeError` when
called without the extension. The pure-Python benchmark utilities
(``benchmarks.perf_model``, ``benchmarks.timing``) don't need the extension
and remain fully functional.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

__all__ = [
    "HAS_CUDA_EXT",
    "attention_naive_forward",
    "attention_online_ref_forward",
    "flash_fwd_v1_forward",
    "flash_fwd_v2_forward",
    "torch_ref_forward",
    "kFlashV1Br",
    "kFlashV1Bc",
    "kFlashV2Br",
    "kFlashV2Bc",
    "kFlashV2RowsPerThread",
    "kOnlineRefBc",
]

# Populated on the failure branch; initialised to None so `_require_ext` can
# always reference it safely regardless of which branch ran.
_import_error: ImportError | None = None

try:
    from . import _C  # type: ignore[attr-defined]

    HAS_CUDA_EXT: bool = True
    kFlashV1Br: int = _C.kFlashV1Br
    kFlashV1Bc: int = _C.kFlashV1Bc
    kFlashV2Br: int = _C.kFlashV2Br
    kFlashV2Bc: int = _C.kFlashV2Bc
    kFlashV2RowsPerThread: int = _C.kFlashV2RowsPerThread
    kOnlineRefBc: int = _C.kOnlineRefBc
except ImportError as _e:
    _import_error = _e
    HAS_CUDA_EXT = False
    # Fallbacks so downstream code can still reference these constants when
    # the extension is missing. Values mirror the .cuh headers verbatim.
    kFlashV1Br = 32
    kFlashV1Bc = 32
    kFlashV2Br = 32
    kFlashV2Bc = 32
    kFlashV2RowsPerThread = 2
    kOnlineRefBc = 64


def _require_ext() -> None:
    """Raise if the ``_C`` extension was not built (Mac-local, no CUDA)."""
    if not HAS_CUDA_EXT:
        raise RuntimeError(
            "flash_from_scratch._C is not available. Rebuild the extension "
            "on a CUDA host (Colab T4 or Modal A10G) via `pip install -e .` "
            "or `BUILD_PY_EXT=ON scripts/build.sh`. Original ImportError: "
            f"{_import_error!r}"
        )


# --------------------------------------------------------------------------- #
# Tensor-contract validation (theory/M5.md §6.2).                             #
# --------------------------------------------------------------------------- #

def _check_qkv(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
               variant: str, supported_D: frozenset[int] | set[int]) -> None:
    import torch  # local import so `import flash_from_scratch` works CPU-only

    if not (Q.is_cuda and K.is_cuda and V.is_cuda):
        raise TypeError(f"{variant}: Q, K, V must all be CUDA tensors")
    if not (Q.device == K.device == V.device):
        raise ValueError(f"{variant}: Q, K, V must live on the same CUDA device")
    if not (Q.dtype == K.dtype == V.dtype == torch.float32):
        raise TypeError(f"{variant}: only fp32 is supported in M5 "
                        f"(got Q={Q.dtype}, K={K.dtype}, V={V.dtype})")
    if not (Q.dim() == K.dim() == V.dim() == 4):
        raise ValueError(f"{variant}: Q, K, V must be 4-D [B, H, N, D] tensors "
                         f"(got dims {Q.dim()}, {K.dim()}, {V.dim()})")
    if not (Q.shape == K.shape == V.shape):
        raise ValueError(f"{variant}: Q, K, V must share the same shape "
                         f"(got {tuple(Q.shape)}, {tuple(K.shape)}, {tuple(V.shape)})")
    if not (Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()):
        raise ValueError(f"{variant}: Q, K, V must be contiguous in "
                         "[B, H, N, D] layout")
    D = int(Q.size(-1))
    if D not in supported_D:
        raise ValueError(f"{variant}: D={D} not in supported set "
                         f"{sorted(supported_D)}")


# --------------------------------------------------------------------------- #
# Variant entry points (all share the same signature).                        #
# --------------------------------------------------------------------------- #

def attention_naive_forward(Q: torch.Tensor, K: torch.Tensor,
                            V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    """M2 three-kernel naive baseline. See ``csrc/attention_naive.cuh``."""
    _check_qkv(Q, K, V, "attention_naive_forward", frozenset({32, 64}))
    _require_ext()
    return _C.attention_naive_forward(Q, K, V, is_causal)


def attention_online_ref_forward(Q: torch.Tensor, K: torch.Tensor,
                                 V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    """M3 online-softmax reference. See ``csrc/attention_online_ref.cuh``."""
    _check_qkv(Q, K, V, "attention_online_ref_forward", frozenset({32, 64}))
    _require_ext()
    return _C.attention_online_ref_forward(Q, K, V, is_causal)


def flash_fwd_v1_forward(Q: torch.Tensor, K: torch.Tensor,
                         V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    """M4 FlashAttention v1 tiled forward. See ``csrc/flash_fwd_v1.cuh``."""
    _check_qkv(Q, K, V, "flash_fwd_v1_forward", frozenset({32, 64}))
    _require_ext()
    return _C.flash_fwd_v1_forward(Q, K, V, is_causal)


def flash_fwd_v2_forward(Q: torch.Tensor, K: torch.Tensor,
                         V: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    """M6 Flash v2 tiled forward. See ``csrc/flash_fwd_v2_shared_kv.cuh``.

    Same algebra as :func:`flash_fwd_v1_forward`; differs only in memory
    layout, thread mapping, and load width. Outputs agree with v1 to ~1e-5.
    """
    _check_qkv(Q, K, V, "flash_fwd_v2_forward", frozenset({32, 64}))
    _require_ext()
    return _C.flash_fwd_v2_forward(Q, K, V, is_causal)


def torch_ref_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                      is_causal: bool = False) -> torch.Tensor:
    """PyTorch's own SDPA (the oracle every CUDA variant is diffed against).

    This does not need the ``_C`` extension. Included in the same namespace
    so the variant-registry (``benchmarks/variants.py``) can treat all four
    callables uniformly.
    """
    import torch
    return torch.nn.functional.scaled_dot_product_attention(
        Q, K, V, is_causal=is_causal
    )
