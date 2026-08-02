// attention_naive.cuh — public API for the M2 naive multi-kernel baseline.
//
// The naive baseline materializes S = QK^T/sqrt(D), then P = softmax(S), then
// O = PV as three separate CUDA kernel launches, each writing its intermediate
// to HBM. This is the "straw man" whose memory cost M4's Flash kernel demolishes.
//
// See `theory/M2.md` for the theory, byte accounting, and design rationale.
// Layout: row-major `[B, H, N, D]`, per `docs/AGENTS.md` §5.1.
// Tolerance for correctness vs `attention_cpu_ref`: 5e-4 abs (fp32), per
// `docs/AGENTS.md` §9.

#pragma once

#include <cuda_runtime.h>

namespace flash_from_scratch {

// Device-pointer overload. Inputs and outputs already reside on the device.
// Allocates transient S and P buffers on the device, launches the three
// kernels in order, frees them. Synchronizes on `stream` before returning.
//
// Args:
//   Q_d, K_d, V_d : device pointers to row-major [B, H, N, D] fp32 tensors.
//   O_d           : device output pointer, same shape.
//   B, H, N, D    : dims (all must be positive).
//   is_causal     : if true, masks S[i, j] for j > i (see theory/M2.md §6).
//   stream        : CUDA stream (default: legacy default stream).
//
// Errors: aborts via ::exit(1) with a printed message on any CUDA error, matching
// the check() style already used in csrc/hello.cu.
void attention_naive_forward(
    const float* Q_d, const float* K_d, const float* V_d, float* O_d,
    int B, int H, int N, int D, bool is_causal,
    cudaStream_t stream = 0);

// Host-pointer overload. Convenience wrapper for tests: allocates Q/K/V/O on the
// device, memcpys inputs H2D, calls the device-pointer overload, memcpys the
// output D2H, and frees everything before returning.
//
// Prefer the device-pointer overload for benchmarks (avoids the memcpy tax).
void attention_naive_forward_host(
    const float* Q_h, const float* K_h, const float* V_h, float* O_h,
    int B, int H, int N, int D, bool is_causal);

}  // namespace flash_from_scratch
