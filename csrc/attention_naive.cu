// attention_naive.cu — the M2 straw man.
//
// Three kernels launched back-to-back, each materializing its output to HBM:
//   1. qk_matmul_kernel   :  S = Q K^T / sqrt(D)         (writes N*N floats)
//   2. row_softmax_kernel :  P = softmax(S)  [+ causal]  (reads S, writes P)
//   3. pv_matmul_kernel   :  O = P V                     (reads P and V)
//
// Every kernel is deliberately naive:
//   - One thread per output element for both matmuls, no shared-memory tiling.
//   - Row-per-block softmax with a shared-memory tree reduction for max & sum.
//   - No float4, no bank-conflict tricks, no fused epilogues.
//
// The design is chosen so the byte accounting in `theory/M2.md` §9 lands
// unambiguously: transient S/P allocations grow quadratically with N, and the
// M4 Flash kernel gets to erase every one of them.
//
// See `theory/M2.md` for the full walkthrough (esp. §5–§7 for per-kernel design
// and §8 for the CUDA idioms used below).

#include "attention_naive.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace flash_from_scratch {

namespace {

// Mirror the check() helper from csrc/hello.cu. Aborts on any CUDA error with
// a printable location tag; tests will surface the message via the GoogleTest
// binary's stderr.
inline void cuda_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        std::exit(1);
    }
}

// Row-major offset for a [B, H, N, D] tensor. Mirrors the host-side helper in
// csrc/attention_cpu_ref.hpp so index arithmetic stays consistent across CPU/GPU.
__device__ __forceinline__ int bhnd_offset(int b, int h, int i, int d,
                                           int H, int N, int D) {
    return ((b * H + h) * N + i) * D + d;
}

// Row-major offset for a [B, H, N, N] tensor (S and P).
__device__ __forceinline__ int bhnn_offset(int b, int h, int i, int j,
                                           int H, int N) {
    return ((b * H + h) * N + i) * N + j;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 — qk_matmul: S[b, h, i, j] = (1/sqrt(D)) * sum_d Q[b,h,i,d] * K[b,h,j,d]
//
// Grid : (ceil(N/16), ceil(N/16), B*H)
// Block: (16, 16, 1) — 256 threads/block
// One thread owns one (i, j) element of S in one (b, h). See theory/M2.md §5.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void qk_matmul_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int B, int H, int N, int D,
    float scale)
{
    const int j  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int bh = blockIdx.z;

    if (i >= N || j >= N) return;

    const int b = bh / H;
    const int h = bh % H;

    float acc = 0.0f;
    for (int d = 0; d < D; ++d) {
        const float qv = Q[bhnd_offset(b, h, i, d, H, N, D)];
        const float kv = K[bhnd_offset(b, h, j, d, H, N, D)];
        acc += qv * kv;
    }
    S[bhnn_offset(b, h, i, j, H, N)] = acc * scale;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 — row_softmax: P[b, h, i, :] = softmax_row(S[b, h, i, :])
//                         with optional causal mask (S[i, j] treated as -inf
//                         for j > i, without materializing it into S).
//
// Grid : (N, B*H, 1)          — one block per (i, b, h)
// Block: (T)                  — T = blockDim.x threads, e.g. 128 or 256
// Shared mem: T * sizeof(float) — used for the block reduction
//
// Three passes over the N scores of one row:
//   Pass 1: block-reduce row max.
//   Pass 2: block-reduce sum of exp(score - max).
//   Pass 3: element-wise write P[i, j] = exp(score - max) / denom.
//
// Masked positions (causal + j > i) are skipped in pass 1/2 and written as
// exact 0 in pass 3, so downstream pv_matmul reads exact zeros.
// See theory/M2.md §6 for the derivation and reduction idiom.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void row_softmax_kernel(
    const float* __restrict__ S,
    float* __restrict__ P,
    int B, int H, int N,
    int is_causal)   // int (0 or 1) — bool works too but int is one less warning
{
    extern __shared__ float smem[];

    const int i  = blockIdx.x;
    const int bh = blockIdx.y;

    if (i >= N) return;

    const int b = bh / H;
    const int h = bh % H;

    const int tid = threadIdx.x;
    const int T   = blockDim.x;

    const int row_off = bhnn_offset(b, h, i, 0, H, N);

    // ── Pass 1: block-reduce row max ────────────────────────────────────────
    float thread_max = -CUDART_INF_F;
    for (int j = tid; j < N; j += T) {
        if (is_causal && j > i) continue;  // masked: contributes nothing to max
        const float s = S[row_off + j];
        if (s > thread_max) thread_max = s;
    }
    smem[tid] = thread_max;
    __syncthreads();

    // Tree reduction on smem — max
    for (int stride = T >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = smem[tid + stride];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }
    const float row_max = smem[0];
    __syncthreads();

    // If every position is masked (impossible on our grid but defensive),
    // row_max stays at -inf. Guard by short-circuiting to all-zero output.
    if (!isfinite(row_max)) {
        for (int j = tid; j < N; j += T) {
            P[row_off + j] = 0.0f;
        }
        return;
    }

    // ── Pass 2: block-reduce sum of exp(score - row_max) ────────────────────
    float thread_sum = 0.0f;
    for (int j = tid; j < N; j += T) {
        if (is_causal && j > i) continue;
        thread_sum += expf(S[row_off + j] - row_max);
    }
    smem[tid] = thread_sum;
    __syncthreads();

    // Tree reduction on smem — sum
    for (int stride = T >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    const float row_denom = smem[0];
    __syncthreads();

    const float inv_denom = 1.0f / row_denom;

    // ── Pass 3: normalize + write P ─────────────────────────────────────────
    for (int j = tid; j < N; j += T) {
        if (is_causal && j > i) {
            P[row_off + j] = 0.0f;
        } else {
            P[row_off + j] = expf(S[row_off + j] - row_max) * inv_denom;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3 — pv_matmul: O[b, h, i, d] = sum_j P[b, h, i, j] * V[b, h, j, d]
//
// Grid : (ceil(D/16), ceil(N/16), B*H)
// Block: (16, 16, 1)
// One thread per (i, d) output element in one (b, h). See theory/M2.md §7.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void pv_matmul_kernel(
    const float* __restrict__ P,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D)
{
    const int d  = blockIdx.x * blockDim.x + threadIdx.x;
    const int i  = blockIdx.y * blockDim.y + threadIdx.y;
    const int bh = blockIdx.z;

    if (i >= N || d >= D) return;

    const int b = bh / H;
    const int h = bh % H;

    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
        const float pv = P[bhnn_offset(b, h, i, j, H, N)];
        const float vv = V[bhnd_offset(b, h, j, d, H, N, D)];
        acc += pv * vv;
    }
    O[bhnd_offset(b, h, i, d, H, N, D)] = acc;
}

// Choose a softmax block width in a small, well-known set. Larger is better up
// to N, then more threads would be idle. We want a power of two ≤ N, capped at
// 256 so smem stays tiny (256 floats = 1 KB).
inline int pick_softmax_block(int N) {
    if (N >= 256) return 256;
    if (N >= 128) return 128;
    if (N >=  64) return  64;
    if (N >=  32) return  32;
    return 16;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API — device-pointer overload
// ─────────────────────────────────────────────────────────────────────────────
void attention_naive_forward(
    const float* Q_d, const float* K_d, const float* V_d, float* O_d,
    int B, int H, int N, int D, bool is_causal,
    cudaStream_t stream)
{
    if (B <= 0 || H <= 0 || N <= 0 || D <= 0) {
        std::fprintf(stderr, "attention_naive_forward: B/H/N/D must be positive "
                             "(got B=%d, H=%d, N=%d, D=%d)\n", B, H, N, D);
        std::exit(1);
    }

    const size_t sp_elems = static_cast<size_t>(B) * H * N * N;
    const size_t sp_bytes = sp_elems * sizeof(float);

    float* S_d = nullptr;
    float* P_d = nullptr;
    cuda_check(cudaMallocAsync(&S_d, sp_bytes, stream), "cudaMallocAsync S");
    cuda_check(cudaMallocAsync(&P_d, sp_bytes, stream), "cudaMallocAsync P");

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // ── Launch 1: qk_matmul ─────────────────────────────────────────────────
    {
        dim3 block(16, 16, 1);
        dim3 grid((N + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y,
                  static_cast<unsigned>(B * H));
        qk_matmul_kernel<<<grid, block, 0, stream>>>(Q_d, K_d, S_d, B, H, N, D, scale);
        cuda_check(cudaGetLastError(), "qk_matmul_kernel launch");
    }

    // ── Launch 2: row_softmax ───────────────────────────────────────────────
    {
        const int T = pick_softmax_block(N);
        dim3 block(static_cast<unsigned>(T), 1, 1);
        dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(B * H), 1);
        const size_t smem_bytes = static_cast<size_t>(T) * sizeof(float);
        row_softmax_kernel<<<grid, block, smem_bytes, stream>>>(
            S_d, P_d, B, H, N, is_causal ? 1 : 0);
        cuda_check(cudaGetLastError(), "row_softmax_kernel launch");
    }

    // ── Launch 3: pv_matmul ─────────────────────────────────────────────────
    {
        dim3 block(16, 16, 1);
        dim3 grid((D + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y,
                  static_cast<unsigned>(B * H));
        pv_matmul_kernel<<<grid, block, 0, stream>>>(P_d, V_d, O_d, B, H, N, D);
        cuda_check(cudaGetLastError(), "pv_matmul_kernel launch");
    }

    // Free the transient scratch before returning. cudaFreeAsync is stream-
    // ordered, so the frees only occur after the kernels above finish.
    cuda_check(cudaFreeAsync(S_d, stream), "cudaFreeAsync S");
    cuda_check(cudaFreeAsync(P_d, stream), "cudaFreeAsync P");

    // Synchronize so the caller can immediately observe results (or free the
    // input tensors) without worrying about stream state. Tests rely on this.
    cuda_check(cudaStreamSynchronize(stream), "attention_naive_forward sync");
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API — host-pointer overload
// ─────────────────────────────────────────────────────────────────────────────
void attention_naive_forward_host(
    const float* Q_h, const float* K_h, const float* V_h, float* O_h,
    int B, int H, int N, int D, bool is_causal)
{
    const size_t nd_elems = static_cast<size_t>(B) * H * N * D;
    const size_t nd_bytes = nd_elems * sizeof(float);

    float* Q_d = nullptr;
    float* K_d = nullptr;
    float* V_d = nullptr;
    float* O_d = nullptr;
    cuda_check(cudaMalloc(&Q_d, nd_bytes), "cudaMalloc Q");
    cuda_check(cudaMalloc(&K_d, nd_bytes), "cudaMalloc K");
    cuda_check(cudaMalloc(&V_d, nd_bytes), "cudaMalloc V");
    cuda_check(cudaMalloc(&O_d, nd_bytes), "cudaMalloc O");

    cuda_check(cudaMemcpy(Q_d, Q_h, nd_bytes, cudaMemcpyHostToDevice), "H2D Q");
    cuda_check(cudaMemcpy(K_d, K_h, nd_bytes, cudaMemcpyHostToDevice), "H2D K");
    cuda_check(cudaMemcpy(V_d, V_h, nd_bytes, cudaMemcpyHostToDevice), "H2D V");

    attention_naive_forward(Q_d, K_d, V_d, O_d, B, H, N, D, is_causal, /*stream=*/0);

    cuda_check(cudaMemcpy(O_h, O_d, nd_bytes, cudaMemcpyDeviceToHost), "D2H O");

    cuda_check(cudaFree(Q_d), "cudaFree Q");
    cuda_check(cudaFree(K_d), "cudaFree K");
    cuda_check(cudaFree(V_d), "cudaFree V");
    cuda_check(cudaFree(O_d), "cudaFree O");
}

}  // namespace flash_from_scratch
