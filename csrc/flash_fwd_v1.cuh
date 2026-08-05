// flash_fwd_v1.cuh — public API for the M4 FlashAttention v1 forward kernel.
//
// The first *real* Flash kernel: one CTA per Q-tile of Br rows, streaming K/V
// in Bc-column tiles, applying the online-softmax recurrence in lock-step
// across all Br rows. S and P never touch HBM — the whole point.
//
// This is M3's recurrence (see attention_online_ref.cuh) run in parallel for
// Br query rows so each K/V tile loaded from HBM amortizes across Br rows
// instead of one. Correctness follows M3 rowwise; performance follows from
// the Br-fold reduction in K/V re-reads. See theory/M4.md for the derivation.
//
// Deliberate non-choices in v1 (see docs/flash_attention_notes.md):
//   - No float4 / vectorized loads              — M6.
//   - No shared-memory swizzling / conflict fix — M6.
//   - No register-resident Õ                    — M6.
//   - No async copies                           — M6/M7.
//   - No causal correctness testing             — M7 (kernel implements it).
//   - No batch/multi-head testing               — M7 (kernel supports any B, H).
//   - No fp16                                   — M8.
//
// Layout: row-major `[B, H, N, D]`, per `docs/AGENTS.md` §5.1.
// Tolerance for correctness vs `attention_cpu_ref`: 5e-4 abs (fp32), per
// `docs/AGENTS.md` §9 (looser than M3's 1e-5 because M4 introduces a second
// axis of accumulation via the PV matmul across Bc columns).

#pragma once

#include <cuda_runtime.h>

namespace flash_from_scratch {

// Tile shapes chosen so shared memory + threads-per-CTA both fit Colab T4's
// default 48 KB per-CTA smem budget AND CUDA's 1024 max-threads-per-CTA limit
// for D ∈ {32, 64}. See theory/M4.md §7 for the calculation.
//
// smem(Br=32, Bc=32, D=64) = 37 120 B ≈ 36.25 KB  (fits 48 KB default)
// smem(Br=32, Bc=32, D=32) = 20 736 B ≈ 20.25 KB
// threads/CTA = Br * Bc = 1024                     (hits the CUDA max)
//
// Larger tiles (Br=Bc=64) blow past both budgets; that unlock waits for
// register-resident Õ (M6) or D-specialized paths (M7).
constexpr int kFlashV1Br = 32;
constexpr int kFlashV1Bc = 32;

// Device-pointer overload. Inputs and outputs already reside on the device.
// Allocates NO transient HBM buffers (that's the whole point) — only shared
// memory inside the kernel. Synchronizes on `stream` before returning.
//
// Args:
//   Q_d, K_d, V_d : device pointers to row-major [B, H, N, D] fp32 tensors.
//   O_d           : device output pointer, same shape.
//   B, H, N, D    : dims. All positive. D must be in {32, 64} for v1.
//   is_causal     : if true, masks S[i, j] for j > i inside the tile loop.
//                   M4 tests keep this false; M7 formalizes causal testing.
//   stream        : CUDA stream (default: legacy default stream).
//
// Errors: aborts via ::exit(1) on any CUDA error or dimension violation, with
// a printed diagnostic; matches the check() style already used in M2/M3.
void flash_fwd_v1_forward(
    const float* Q_d, const float* K_d, const float* V_d, float* O_d,
    int B, int H, int N, int D, bool is_causal,
    cudaStream_t stream = 0);

// Host-pointer overload. Convenience wrapper for tests: allocates Q/K/V/O on
// the device, memcpys inputs H2D, calls the device-pointer overload, memcpys
// the output D2H, and frees everything before returning.
void flash_fwd_v1_forward_host(
    const float* Q_h, const float* K_h, const float* V_h, float* O_h,
    int B, int H, int N, int D, bool is_causal);

}  // namespace flash_from_scratch
