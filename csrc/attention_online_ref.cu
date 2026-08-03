// attention_online_ref.cu — the M3 online-softmax reference.
//
// One CUDA thread block per query row (grid: (N, B*H, 1); block: (Bc, 1, 1),
// with Bc = kOnlineRefBc = 64). The block streams over K/V in chunks of Bc
// columns, applying the online-softmax recurrence:
//
//   m_new = max(m_old, m̃_local)
//   α     = exp(m_old − m_new)              ∈ (0, 1]
//   p̃_c  = exp(s_c − m_new)                (per-column, in the block)
//   ℓ_new = α · ℓ_old  + Σ_c p̃_c
//   Õ_new = α · Õ_old + Σ_c p̃_c · V_c
//
// After all blocks: O[i, :] = Õ / ℓ.
//
// Per-row persistent state is (m, ℓ, Õ[D]) = (D + 2) floats, independent of N.
// This is what the M4 tiled forward will inherit almost verbatim; M3's job is
// to prove the recurrence works before adding Q-tiling.
//
// Deliberate non-choices (all deferred to later milestones):
//   - No Q tiling                         — M4.
//   - No float4 vectorized loads          — M6.
//   - No shared-memory swizzling          — M6.
//   - No causal correctness testing       — M7 (kernel implements it though).
//   - No batch/multi-head testing         — M7 (kernel supports any B, H).
//
// See `theory/M3.md` for the full walkthrough (esp. §7 for the block picture,
// §11 for numerical stability, §15.4 for the shared-memory layout).

#include "attention_online_ref.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace flash_from_scratch {

namespace {

// Mirror the check() helper from csrc/hello.cu / attention_naive.cu.
inline void cuda_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        std::exit(1);
    }
}

// Row-major offset for a [B, H, N, D] tensor. Duplicated from attention_naive.cu
// rather than shared through a header, because both files live in an anonymous
// namespace — hoisting to a shared header is refactor work for M4/M5.
__device__ __forceinline__ int bhnd_offset(int b, int h, int i, int d,
                                           int H, int N, int D) {
    return ((b * H + h) * N + i) * D + d;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel — attention_online_ref
//
// Grid : (N, B*H, 1)
// Block: (Bc, 1, 1)  — Bc threads cooperating on one query row
//
// Shared-memory layout (extern __shared__ float smem[], offsets in floats):
//   [0, 4)                    : sm_stats   = {m, ℓ, α, m_new}   (persistent)
//   [4, 4+D)                  : sO_tilde                          (persistent)
//   [4+D, 4+2D)               : sQ (cached Q[b, h, i, :])
//   [4+2D, 4+2D+Bc*D)         : sK (current block's K slice)
//   [.., + Bc*D)              : sV (current block's V slice)
//   [.., + Bc)                : s_scores  (per-block p̃)
//   [.., + Bc)                : s_reduce  (reduction scratch)
//
// Constraints: D <= Bc so the Õ update (parallel over D) uses one thread per
// output element with no per-thread loop over D.
// ─────────────────────────────────────────────────────────────────────────────
template <int Bc>
__global__ void attention_online_ref_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D,
    int is_causal, float scale)
{
    extern __shared__ float smem[];

    float* sm_stats = smem;
    float* sO_tilde = sm_stats + 4;
    float* sQ       = sO_tilde + D;
    float* sK       = sQ + D;
    float* sV       = sK + Bc * D;
    float* s_scores = sV + Bc * D;
    float* s_reduce = s_scores + Bc;

    const int i  = blockIdx.x;   // query row
    const int bh = blockIdx.y;   // (batch, head)
    if (i >= N) return;

    const int b = bh / H;
    const int h = bh % H;
    const int t = threadIdx.x;   // t in [0, Bc)

    // ── One-time init ────────────────────────────────────────────────────────
    // Cache Q[b, h, i, :] in shared memory: reused every block-loop iteration.
    if (t < D) {
        sQ[t] = Q[bhnd_offset(b, h, i, t, H, N, D)];
    }
    // Persistent state: (m, ℓ, Õ) = (−∞, 0, 0). See theory/M3.md §6.
    if (t == 0) {
        sm_stats[0] = -CUDART_INF_F;  // m
        sm_stats[1] = 0.0f;            // ℓ
    }
    if (t < D) {
        sO_tilde[t] = 0.0f;
    }
    __syncthreads();

    // ── Block loop over K/V columns ──────────────────────────────────────────
    const int nblocks = (N + Bc - 1) / Bc;
    (void)B;  // kept in signature for symmetry with attention_naive; unused inside.

    for (int b_idx = 0; b_idx < nblocks; ++b_idx) {
        const int j = b_idx * Bc + t;
        const bool j_in_range = (j < N);
        const bool j_causal_ok = (!is_causal) || (j <= i);
        const bool j_active = j_in_range && j_causal_ok;

        // ── Step 1: cooperatively load this block's K and V slices. ─────────
        // Thread t loads the D floats of K[b, h, j, :] and V[b, h, j, :].
        // Out-of-range slots are zeroed; they won't be read (p̃ = 0 by mask),
        // but zeroing avoids reading uninitialized memory in the Õ update.
        if (j_in_range) {
            for (int d = 0; d < D; ++d) {
                sK[t * D + d] = K[bhnd_offset(b, h, j, d, H, N, D)];
                sV[t * D + d] = V[bhnd_offset(b, h, j, d, H, N, D)];
            }
        } else {
            for (int d = 0; d < D; ++d) {
                sK[t * D + d] = 0.0f;
                sV[t * D + d] = 0.0f;
            }
        }
        __syncthreads();

        // ── Step 2: score s_local = (sQ · sK[t]) / sqrt(D). ────────────────
        float s_local;
        if (j_active) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) {
                acc += sQ[d] * sK[t * D + d];
            }
            s_local = acc * scale;
        } else {
            s_local = -CUDART_INF_F;
        }

        // ── Step 3: block reduction — local max. ────────────────────────────
        s_reduce[t] = s_local;
        __syncthreads();
        #pragma unroll
        for (int stride = Bc >> 1; stride > 0; stride >>= 1) {
            if (t < stride) {
                s_reduce[t] = fmaxf(s_reduce[t], s_reduce[t + stride]);
            }
            __syncthreads();
        }
        const float m_local = s_reduce[0];

        // ── Step 4: compute new max and α; broadcast via sm_stats. ──────────
        // Guard against −∞ propagation:
        //   * If m_new is −∞ (entire block is out-of-range or causal-masked
        //     with no prior state), leave state untouched by setting α = 1
        //     and letting Σ p̃ = 0 handle the rest.
        //   * If m_old is −∞ (first block on real data), α = 0 correctly
        //     annihilates the initial zero state.
        //   * Otherwise α = exp(m_old − m_new) ∈ (0, 1], never overflows.
        if (t == 0) {
            const float m_old = sm_stats[0];
            const float m_new = fmaxf(m_old, m_local);
            float alpha;
            if (!isfinite(m_new)) {
                alpha = 1.0f;
            } else if (!isfinite(m_old)) {
                alpha = 0.0f;
            } else {
                alpha = expf(m_old - m_new);
            }
            sm_stats[2] = alpha;
            sm_stats[3] = m_new;
        }
        __syncthreads();

        const float alpha = sm_stats[2];
        const float m_new = sm_stats[3];

        // ── Step 5: p̃_t = exp(s − m_new); stash in s_scores. ───────────────
        float p_local;
        if (j_active && isfinite(m_new)) {
            p_local = expf(s_local - m_new);
        } else {
            p_local = 0.0f;
        }
        s_scores[t] = p_local;

        // ── Step 5b: block reduction — local ℓ. ────────────────────────────
        s_reduce[t] = p_local;
        __syncthreads();
        #pragma unroll
        for (int stride = Bc >> 1; stride > 0; stride >>= 1) {
            if (t < stride) {
                s_reduce[t] += s_reduce[t + stride];
            }
            __syncthreads();
        }
        const float ell_local = s_reduce[0];

        // ── Step 6: merge into running (m, ℓ). ─────────────────────────────
        if (t == 0) {
            sm_stats[1] = alpha * sm_stats[1] + ell_local;
            sm_stats[0] = m_new;
        }
        // No sync needed here: step 7 doesn't read sm_stats. s_scores was made
        // visible by the sync at the end of step 5b's last reduction stride.

        // ── Step 7: Õ update — parallel over D. ─────────────────────────────
        // Each thread t < D produces one element of the new Õ vector.
        // s_scores[c] is p̃_c (visible via the reduction's sync); sV[c*D + t]
        // is V_{b_idx*Bc+c, t} (visible via the top-of-iter sync).
        if (t < D) {
            float acc = alpha * sO_tilde[t];
            #pragma unroll 4
            for (int c = 0; c < Bc; ++c) {
                acc += s_scores[c] * sV[c * D + t];
            }
            sO_tilde[t] = acc;
        }
        __syncthreads();
    }

    // ── Terminal: normalize and emit O[i, :] = Õ / ℓ. ───────────────────────
    // Guard against ℓ = 0 (pathological all-masked / underflow row); write 0
    // rather than NaN. See theory/M3.md §11 footgun note.
    const float ell_final = sm_stats[1];
    if (t < D) {
        const float o_val = (ell_final > 0.0f) ? (sO_tilde[t] / ell_final) : 0.0f;
        O[bhnd_offset(b, h, i, t, H, N, D)] = o_val;
    }
}

// Total shared-memory bytes required for the kernel above, given (Bc, D).
constexpr size_t online_ref_smem_bytes(int Bc, int D) {
    return static_cast<size_t>(4 + 2 * D + 2 * Bc * D + 2 * Bc) * sizeof(float);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API — device-pointer overload
// ─────────────────────────────────────────────────────────────────────────────
void attention_online_ref_forward(
    const float* Q_d, const float* K_d, const float* V_d, float* O_d,
    int B, int H, int N, int D, bool is_causal,
    cudaStream_t stream)
{
    if (B <= 0 || H <= 0 || N <= 0 || D <= 0) {
        std::fprintf(stderr, "attention_online_ref_forward: B/H/N/D must be positive "
                             "(got B=%d, H=%d, N=%d, D=%d)\n", B, H, N, D);
        std::exit(1);
    }
    if (D > kOnlineRefBc) {
        std::fprintf(stderr, "attention_online_ref_forward: D (=%d) must be <= "
                             "kOnlineRefBc (=%d); Õ update requires one thread "
                             "per output dim. Raise Bc if you need larger D.\n",
                             D, kOnlineRefBc);
        std::exit(1);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    const size_t smem = online_ref_smem_bytes(kOnlineRefBc, D);

    dim3 block(static_cast<unsigned>(kOnlineRefBc), 1, 1);
    dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(B * H), 1);

    attention_online_ref_kernel<kOnlineRefBc><<<grid, block, smem, stream>>>(
        Q_d, K_d, V_d, O_d,
        B, H, N, D,
        is_causal ? 1 : 0, scale);
    cuda_check(cudaGetLastError(), "attention_online_ref_kernel launch");

    cuda_check(cudaStreamSynchronize(stream), "attention_online_ref_forward sync");
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API — host-pointer overload
// ─────────────────────────────────────────────────────────────────────────────
void attention_online_ref_forward_host(
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

    attention_online_ref_forward(Q_d, K_d, V_d, O_d, B, H, N, D, is_causal, /*stream=*/0);

    cuda_check(cudaMemcpy(O_h, O_d, nd_bytes, cudaMemcpyDeviceToHost), "D2H O");

    cuda_check(cudaFree(Q_d), "cudaFree Q");
    cuda_check(cudaFree(K_d), "cudaFree K");
    cuda_check(cudaFree(V_d), "cudaFree V");
    cuda_check(cudaFree(O_d), "cudaFree O");
}

}  // namespace flash_from_scratch
