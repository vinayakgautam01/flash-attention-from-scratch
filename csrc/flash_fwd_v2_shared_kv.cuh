// flash_fwd_v2_shared_kv.cuh — public API for the M6 "Flash v2" forward kernel.
//
// IMPORTANT NAMING NOTE: this is *not* FlashAttention-2 the paper. The paper's
// v2 reverses the loop order (outer = KV-tile) and adds sequence-parallel work
// partitioning across CTAs. This kernel keeps M4's FA-1 loop order and the exact
// same algebra; only the *mechanical* implementation changes. See theory/M6.md
// §2 for the distinction, and docs/flash_attention_notes.md §6 for why the
// real FA-2 loop reversal stays out of scope for v1.0.0.
//
// Identical math to flash_fwd_v1 (same online-softmax recurrence, same tile
// shapes, same tolerance). Four mechanical changes, each derived in theory/M6.md:
//
//   1. sK padded to a [Bc][D+1] row stride      — kills the 32-way bank
//      conflict on the STEP-2 dot-product read. §5.
//   2. Õ (the running output accumulator) moved  — frees Br*D floats of smem
//      from shared memory into registers.          and removes 2*L smem
//                                                  round-trips per tile. §6.
//   3. Half the threads (512, not 1024), two Q   — two CTAs per SM instead of
//      rows per thread.                            one, so barrier stalls in
//                                                  one CTA are hidden by the
//                                                  other. §7.
//   4. float4 vectorized global loads for Q/K/V  — 4x fewer load instructions
//      via a linearized (Br==Bc-free) loop.        and less address math. §8.
//
// Plus one change that falls out of (2): the Br x Bc probability tile P never
// reaches shared memory at all. Row `i` of P is written and read entirely
// within one warp, so it travels by __shfl_sync instead. That removes both the
// sS buffer and one __syncthreads() per tile iteration. §9.
//
// Deliberate non-choices still standing after v2 (see docs/flash_attention_notes.md):
//   - No 2-D register blocking of the S matmul  — v3 (breaks warp-per-row).
//   - No async copies (cp.async)                — M7 (Ampere-only; T4 is Turing).
//   - No FA-2 loop reversal                     — out of scope for v1.0.0.
//   - No D = 128                                — M7 (needs the 64 KB opt-in).
//   - No batch/multi-head or causal testing     — M7 (kernel supports both).
//   - No fp16                                   — M8.
//
// Layout: row-major `[B, H, N, D]`, per `docs/AGENTS.md` §5.1.
// Tolerance vs `attention_cpu_ref`: 5e-4 abs (fp32), same as v1 per
// `docs/AGENTS.md` §9. v2 must hit *parity* with v1 on the whole M4 grid —
// that is the entry condition for quoting any performance number.

#pragma once

#include <cuda_runtime.h>

namespace flash_from_scratch {

// Tile shapes. Unchanged from v1 (Br = Bc = 32) so the HBM traffic model from
// theory/M4.md §10 carries over verbatim and the v1-vs-v2 comparison isolates
// *mechanical* effects only. See theory/M6.md §10 for why we did not take the
// Br = 64 option on T4 by default (it halves HBM traffic but forfeits the
// second CTA per SM — and HBM is only ~1.8% of v1's runtime).
constexpr int kFlashV2Br = 32;
constexpr int kFlashV2Bc = 32;

// Q rows owned by each thread. Threads/CTA = Bc * (Br / kFlashV2RowsPerThread)
//                                          = 32 * 16 = 512.
// This is the occupancy lever: 512 threads/CTA lets two CTAs be resident on a
// Turing SM (1024 max threads/SM), where v1's 1024-thread CTA allowed exactly
// one. See theory/M6.md §7.
constexpr int kFlashV2RowsPerThread = 2;

// Shared-memory bytes per CTA for a given D. Exposed so tests and the
// occupancy notes in docs/ptxas_v1_vs_v2.md can assert against one source of
// truth instead of re-deriving the layout.
//
//   sQ [Br][D]      + sK [Bc][D+1]  + sV [Bc][D]
//   = Br*D          + Bc*(D+1)      + Bc*D          floats
//
// D = 64 -> 24,704 B   (v1: 37,120 B)
// D = 32 -> 12,416 B   (v1: 20,736 B)
constexpr int flash_v2_smem_bytes(int Br, int Bc, int D) {
    return static_cast<int>(sizeof(float)) *
           (Br * D + Bc * (D + 1) + Bc * D);
}

// Device-pointer overload. Inputs and outputs already reside on the device.
// Allocates NO transient HBM buffers — S and P never leave the SM. Synchronizes
// on `stream` before returning (matches v1's contract so the M5 harness times
// the two variants identically).
//
// Args:
//   Q_d, K_d, V_d : device pointers to row-major [B, H, N, D] fp32 tensors.
//                   Must be 16-byte aligned (guaranteed by cudaMalloc and by
//                   torch's caching allocator) for the float4 load path.
//   O_d           : device output pointer, same shape.
//   B, H, N, D    : dims. All positive. D must be in {32, 64} for v2.
//   is_causal     : if true, masks S[i, j] for j > i inside the tile loop.
//                   Plumbed and implemented; formal causal testing is M7.
//   stream        : CUDA stream (default: legacy default stream).
//
// Errors: aborts via ::exit(1) on any CUDA error or dimension violation, with
// a printed diagnostic; matches the check() style used in M2/M3/M4.
void flash_fwd_v2_forward(
    const float* Q_d, const float* K_d, const float* V_d, float* O_d,
    int B, int H, int N, int D, bool is_causal,
    cudaStream_t stream = 0);

// Host-pointer overload. Convenience wrapper for the GoogleTest binary:
// allocates Q/K/V/O on the device, memcpys inputs H2D, calls the device-pointer
// overload, memcpys the output D2H, and frees everything before returning.
void flash_fwd_v2_forward_host(
    const float* Q_h, const float* K_h, const float* V_h, float* O_h,
    int B, int H, int N, int D, bool is_causal);

}  // namespace flash_from_scratch
