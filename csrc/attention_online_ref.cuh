// attention_online_ref.cuh — public API for the M3 online-softmax reference.
//
// This kernel is the *standalone artifact* proving that we can compute exact
// scaled dot-product attention while streaming K/V in blocks and keeping only
// (m, ℓ, Õ) — (D + 2) floats — as running state per query row. It is not
// tiled over Q (that's M4), not fused across query rows, and not shared-memory
// optimal (that's M6). Correctness of the streaming recurrence is the whole
// deliverable.
//
// See `theory/M3.md` (theory + block diagrams) and
// `docs/online_softmax_derivation.md` (compressed derivation) for the math.
// Layout: row-major `[B, H, N, D]`, per `docs/AGENTS.md` §5.1.
// Tolerance for correctness vs `attention_cpu_ref`: 1e-5 abs (fp32), tighter
// than M2's 5e-4 because M3 is algebraic equivalence, not accumulation drift.
//
// Constraints: D must be <= Bc (M3 hard-codes Bc = 64), since the kernel's Õ
// update parallelizes over D by using one thread per output element.

#pragma once

#include <cuda_runtime.h>

namespace flash_from_scratch {

// Block width (number of K/V columns processed per online-softmax step).
// Fixed at 64 in M3 — chosen so shared memory fits in Colab T4's default
// 48 KB per-CTA budget for D ∈ {32, 64} (see theory/M3.md §15.4).
constexpr int kOnlineRefBc = 64;

// Device-pointer overload. Inputs and outputs already reside on the device.
// Allocates no transient buffers on HBM (that's the whole point) — only
// shared memory inside the kernel. Synchronizes on `stream` before returning.
//
// Args:
//   Q_d, K_d, V_d : device pointers to row-major [B, H, N, D] fp32 tensors.
//   O_d           : device output pointer, same shape.
//   B, H, N, D    : dims. All positive. D must be <= kOnlineRefBc (= 64).
//   is_causal     : if true, masks S[i, j] for j > i (applied inside the score
//                   step). M3 tests keep this false; M7 formalizes causal
//                   correctness testing across the full grid.
//   stream        : CUDA stream (default: legacy default stream).
//
// Errors: aborts via ::exit(1) on any CUDA error or dimension violation, with
// a printed diagnostic; matches the check() style already used in the M2 code.
void attention_online_ref_forward(
    const float* Q_d, const float* K_d, const float* V_d, float* O_d,
    int B, int H, int N, int D, bool is_causal,
    cudaStream_t stream = 0);

// Host-pointer overload. Convenience wrapper for tests: allocates Q/K/V/O on
// the device, memcpys inputs H2D, calls the device-pointer overload, memcpys
// the output D2H, and frees everything before returning.
void attention_online_ref_forward_host(
    const float* Q_h, const float* K_h, const float* V_h, float* O_h,
    int B, int H, int N, int D, bool is_causal);

}  // namespace flash_from_scratch
