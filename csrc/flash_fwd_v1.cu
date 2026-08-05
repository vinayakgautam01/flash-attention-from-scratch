// flash_fwd_v1.cu — the M4 FlashAttention v1 forward kernel.
//
// One CTA per Q-tile of Br rows; the block loop streams K/V in Bc-column tiles.
// Inside each tile iteration, Br independent copies of M3's online-softmax
// recurrence execute in lock-step, sharing the loaded K/V slice. S and P never
// touch HBM — they live and die inside the CTA's shared-memory `sS` scratch.
//
// Per-row recurrence (identical to M3, one copy per row of the Q-tile):
//
//   m_new = max(m_old, m̃_local)                  [Br × ]
//   α     = exp(m_old − m_new)                    ∈ (0, 1]
//   p̃     = exp(sS − m_new)                       [Br × Bc]
//   ℓ_new = α · ℓ_old  + rowsum(p̃)                [Br × ]
//   Õ_new = α · Õ_old + p̃ · sV                    [Br × D]
//
// After all Tc tiles: O[i, :] = Õ / ℓ.
//
// Deliberate non-choices in v1 (all listed in docs/flash_attention_notes.md):
//   - No float4 vectorized loads               — M6.
//   - No shared-memory swizzling / padding     — M6 (sK/sV bank-conflict on
//                                                    dot-product read; accepted
//                                                    in v1 for code clarity).
//   - No register-resident Õ                   — M6.
//   - No async copies (cp.async)               — M6/M7.
//   - No batch/multi-head testing              — M7 (grid already carries B·H).
//   - No causal correctness testing            — M7 (kernel implements it).
//   - No fp16                                  — M8.
//
// See `theory/M4.md` for the full walkthrough (esp. §7 for the smem budget,
// §8 for the launch config, §9 for the per-step thread mapping).

#include "flash_fwd_v1.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace flash_from_scratch {

namespace {

// Mirror check() helpers from attention_naive.cu / attention_online_ref.cu.
inline void cuda_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        std::exit(1);
    }
}

// Row-major offset for a [B, H, N, D] tensor. Same helper pattern as M2/M3;
// duplicated here rather than shared through a header because all three
// kernels live in anonymous namespaces (refactor is M5/M6 work).
__device__ __forceinline__ int bhnd_offset(int b, int h, int i, int d,
                                           int H, int N, int D) {
    return ((b * H + h) * N + i) * D + d;
}

// Warp-level butterfly reduction (max). Bc = warpSize = 32, so one warp
// carries one Q row across the K/V columns of a tile.
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, stride));
    }
    return v;
}

// Warp-level butterfly reduction (sum).
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, stride);
    }
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel — flash_fwd_v1
//
// Grid : (Tr = ⌈N/Br⌉,   B*H,   1)
// Block: (Bc,             Br,    1)      — 1024 threads at Br = Bc = 32
//
// Shared-memory layout (extern __shared__ float smem[], offsets in floats):
//   [0,        Br*D)             : sQ                       (persistent, one load)
//   [Br*D,     Br*D + Bc*D)      : sK  (this tile's slice)  (reloaded per tile)
//   [+Bc*D,    +2*Bc*D)          : sV  (this tile's slice)  (reloaded per tile)
//   [+Br*D)                      : sO  (running Õ)          (persistent)
//   [+Br*Bc)                     : sS  (scores → p̃ scratch) (transient per tile)
//   [+Br)                        : sm  (running max per row)  (persistent)
//   [+Br)                        : sl  (running ℓ per row)    (persistent)
//
// Total smem = 4 * (2*Br*D + 2*Bc*D + Br*Bc + 2*Br)  bytes (fp32).
//
// Constraints for v1:
//   Br * Bc <= 1024                          (CUDA per-CTA thread cap)
//   Bc == warpSize == 32                     (row-reductions use __shfl)
//   D in {32, 64}                            (dims we test in M4)
// ─────────────────────────────────────────────────────────────────────────────
template <int Br, int Bc>
__global__ void flash_fwd_v1_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D,
    int is_causal, float scale)
{
    // Compile-time sanity: Bc == 32 is required for the __shfl_xor row reductions.
    static_assert(Bc == 32, "flash_fwd_v1_kernel: Bc must equal warpSize=32.");
    static_assert(Br * Bc <= 1024, "flash_fwd_v1_kernel: Br*Bc must fit CUDA's 1024 threads/CTA cap.");

    // The K/V load loop (STEP 1) maps ty → KV-row-within-tile, so it fills
    // exactly Br rows of sK/sV. Since sK/sV are shaped [Bc × D], we need
    // Br == Bc for every KV row of the tile to be covered exactly once:
    //   - Br < Bc  → rows [Br, Bc) of sK/sV go unwritten → STEP 2 reads garbage.
    //   - Br > Bc  → threads with ty ≥ Bc write past sK into sV → silent corruption.
    // Same coupling applies to the Q load in the init block (ty → Q-row-within-tile
    // fills Br rows of sQ, which has [Br × D] so that side is fine by construction —
    // the K/V side is the load that constrains this).
    // Generalizing to asymmetric tiles (Br ≠ Bc) requires switching to a linearized
    // load loop; see docs/flash_attention_notes.md (M6 backlog).
    static_assert(Br == Bc,
        "flash_fwd_v1_kernel: STEP 1's K/V load assumes Br == Bc (one thread row "
        "per KV-tile row). Generalize the load loop before relaxing this.");

    extern __shared__ float smem[];

    float* sQ = smem;
    float* sK = sQ + Br * D;
    float* sV = sK + Bc * D;
    float* sO = sV + Bc * D;
    float* sS = sO + Br * D;
    float* sm = sS + Br * Bc;
    float* sl = sm + Br;

    const int i_tile = blockIdx.x;
    const int bh    = blockIdx.y;

    const int b = bh / H;
    const int h = bh % H;

    const int tx = threadIdx.x;   // 0..Bc-1
    const int ty = threadIdx.y;   // 0..Br-1

    // Global Q row this thread's *ty axis* speaks for. When i >= N, this thread's
    // row is a tail-padding row: we still let it participate in reductions (with
    // -inf scores that contribute nothing) but never write its output to HBM.
    const int  i           = i_tile * Br + ty;
    const bool i_in_range  = (i < N);

    // ── Init: load sQ once, zero sO, seed (m, ℓ) = (-∞, 0) ──────────────────
    // Thread (tx, ty) loads sQ[ty, d] for d = tx, tx + Bc, ..., < D.
    // Warp (fixed ty, tx=0..Bc-1) reads Q[b, h, i, d=varying_tx] → coalesced.
    #pragma unroll 2
    for (int d = tx; d < D; d += Bc) {
        sQ[ty * D + d] = i_in_range ? Q[bhnd_offset(b, h, i, d, H, N, D)] : 0.0f;
        sO[ty * D + d] = 0.0f;
    }
    if (tx == 0) {
        sm[ty] = -CUDART_INF_F;
        sl[ty] = 0.0f;
    }
    __syncthreads();

    // ── Block loop over KV tiles ────────────────────────────────────────────
    const int Tc = (N + Bc - 1) / Bc;
    (void)B;  // signature symmetry with attention_naive; unused inside kernel.

    for (int b_idx = 0; b_idx < Tc; ++b_idx) {
        // ── STEP 1: cooperative K/V load ─────────────────────────────────────
        // Thread (tx, ty) loads sK[ty, d] and sV[ty, d] for d = tx, tx+Bc, ...
        // Here ty transiently means "which KV row within this tile" (j_load).
        // The warp (fixed ty, tx=0..Bc-1) reads K[..., j_load, d=varying_tx]
        // — consecutive threads → consecutive HBM addresses → coalesced.
        const int  j_load           = b_idx * Bc + ty;
        const bool j_load_in_range  = (j_load < N);

        #pragma unroll 2
        for (int d = tx; d < D; d += Bc) {
            if (j_load_in_range) {
                sK[ty * D + d] = K[bhnd_offset(b, h, j_load, d, H, N, D)];
                sV[ty * D + d] = V[bhnd_offset(b, h, j_load, d, H, N, D)];
            } else {
                sK[ty * D + d] = 0.0f;
                sV[ty * D + d] = 0.0f;
            }
        }
        __syncthreads();

        // ── STEP 2: score matmul sS[ty, tx] = (sQ[ty, :] · sK[tx, :]) / √D ──
        // Now (tx, ty) means (KV col j inside tile, Q row inside tile).
        // Thread (tx, ty) computes ONE element of the Br × Bc score tile.
        const int  j_score  = b_idx * Bc + tx;
        const bool j_in_row = (j_score < N);
        const bool j_causal = (is_causal == 0) || (j_score <= i);
        const bool active   = i_in_range && j_in_row && j_causal;

        float s_val;
        if (active) {
            float acc = 0.0f;
            #pragma unroll 4
            for (int d = 0; d < D; ++d) {
                acc += sQ[ty * D + d] * sK[tx * D + d];
            }
            s_val = acc * scale;
        } else {
            s_val = -CUDART_INF_F;
        }
        // No __syncthreads() needed after this store because step 3 reads only
        // from registers (via __shfl) and step 5 overwrites sS after another
        // barrier below.
        sS[ty * Bc + tx] = s_val;

        // ── STEP 3: row-max reduction m̃[ty] via warp shuffle ────────────────
        // One warp per row (Bc = 32 = warpSize; threads with same ty form one
        // warp). Butterfly reduction — every lane ends with the row max.
        const float m_local = warp_reduce_max(s_val);

        // ── STEP 4: per-row α, m_new (register-resident, warp-broadcast) ─────
        // Every lane in the warp reads sm[ty] (broadcast — same value for all)
        // and computes m_new / alpha in registers. No smem for α or m_new.
        // sm[ty] is visible thanks to __syncthreads at the end of the previous
        // iteration (or the init barrier for iteration 0).
        const float m_old_row = sm[ty];

        float m_new_row;
        if (!isfinite(m_old_row) && !isfinite(m_local)) {
            m_new_row = -CUDART_INF_F;  // whole row masked so far
        } else {
            m_new_row = fmaxf(m_old_row, m_local);
        }

        float alpha_row;
        if (!isfinite(m_new_row)) {
            alpha_row = 1.0f;   // no state yet; α = 1 leaves ℓ, Õ untouched.
        } else if (!isfinite(m_old_row)) {
            alpha_row = 0.0f;   // first real block — annihilate the zero init.
        } else {
            alpha_row = expf(m_old_row - m_new_row);   // ∈ (0, 1]
        }

        // ── STEP 5: p̃ tile + row-sum → ℓ̃[ty] ───────────────────────────────
        float p_val;
        if (active && isfinite(m_new_row)) {
            p_val = expf(s_val - m_new_row);
        } else {
            p_val = 0.0f;
        }
        // Overwrite sS with p̃; step 7 will consume these values.
        sS[ty * Bc + tx] = p_val;

        // Row-sum via the same warp butterfly.
        const float ell_local = warp_reduce_sum(p_val);

        // Make sS's p̃ values visible cross-warp for step 7 (which reads
        // sS[ty, c] across all c). We could not skip this sync because step 7
        // reads elements of sS that other warps' lanes just wrote.
        __syncthreads();

        // ── STEP 6: merge (m, ℓ) — one lane per row writes ──────────────────
        if (tx == 0) {
            sl[ty] = alpha_row * sl[ty] + ell_local;
            sm[ty] = m_new_row;
        }
        // No sync needed here — step 7 doesn't touch sm/sl. The next-iteration
        // read of sm[ty] in step 4 is guarded by the sync at the end of the
        // block-loop body.

        // ── STEP 7: Õ update — sO[ty, d] ← α · sO[ty, d] + Σ_c sS[ty, c] · sV[c, d] ─
        // Now tx swings meaning to "D-index". Thread (tx, ty) produces one or
        // more elements of sO[ty, d] for d = tx, tx+Bc, ..., < D.
        // At D=64, Bc=32 → each thread produces 2 output elements. At D=32 → 1.
        #pragma unroll 2
        for (int d = tx; d < D; d += Bc) {
            float acc = alpha_row * sO[ty * D + d];
            #pragma unroll 8
            for (int c = 0; c < Bc; ++c) {
                acc += sS[ty * Bc + c] * sV[c * D + d];
            }
            sO[ty * D + d] = acc;
        }
        __syncthreads();  // sK/sV/sS will be overwritten by next iteration.
    }

    // ── Terminal: normalize and emit O[i, :] = sO / sl ───────────────────────
    // Tail-padding rows (i >= N) don't write. ℓ ≤ 0 (pathological all-masked
    // / underflow row) writes zeros instead of NaN — matches M3's guard.
    if (i_in_range) {
        const float ell_final = sl[ty];
        const bool  ok        = ell_final > 0.0f;
        #pragma unroll 2
        for (int d = tx; d < D; d += Bc) {
            const float o_val = ok ? (sO[ty * D + d] / ell_final) : 0.0f;
            O[bhnd_offset(b, h, i, d, H, N, D)] = o_val;
        }
    }
}

// Total shared-memory bytes required for the kernel above, given (Br, Bc, D).
constexpr size_t flash_v1_smem_bytes(int Br, int Bc, int D) {
    // sQ + sK + sV + sO + sS + sm + sl (all fp32)
    return static_cast<size_t>(
        2 * Br * D  +  // sQ, sO
        2 * Bc * D  +  // sK, sV
        Br * Bc     +  // sS
        2 * Br         // sm, sl
    ) * sizeof(float);
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API — device-pointer overload
// ─────────────────────────────────────────────────────────────────────────────
void flash_fwd_v1_forward(
    const float* Q_d, const float* K_d, const float* V_d, float* O_d,
    int B, int H, int N, int D, bool is_causal,
    cudaStream_t stream)
{
    if (B <= 0 || H <= 0 || N <= 0 || D <= 0) {
        std::fprintf(stderr, "flash_fwd_v1_forward: B/H/N/D must be positive "
                             "(got B=%d, H=%d, N=%d, D=%d)\n", B, H, N, D);
        std::exit(1);
    }
    // v1 supports D ∈ {32, 64}. Wider D needs 64 KB smem opt-in and/or a
    // register-resident Õ; both are M6/M7 work. Fail loudly here rather
    // than silently mis-launching.
    if (D != 32 && D != 64) {
        std::fprintf(stderr, "flash_fwd_v1_forward: D must be 32 or 64 in v1 "
                             "(got D=%d). Wider D is deferred to M7.\n", D);
        std::exit(1);
    }

    constexpr int Br = kFlashV1Br;
    constexpr int Bc = kFlashV1Bc;

    const float  scale = 1.0f / std::sqrt(static_cast<float>(D));
    const size_t smem  = flash_v1_smem_bytes(Br, Bc, D);

    // Sanity: fits Colab T4's default 48 KB per-CTA budget for D ∈ {32, 64}.
    // (See theory/M4.md §7 for the table of tile-size-vs-smem calculations.)
    if (smem > 48 * 1024) {
        std::fprintf(stderr, "flash_fwd_v1_forward: smem %zu B exceeds default "
                             "48 KB per-CTA cap. Enable dynamic smem via "
                             "cudaFuncSetAttribute before landing wider D.\n",
                     smem);
        std::exit(1);
    }

    dim3 block(static_cast<unsigned>(Bc),
               static_cast<unsigned>(Br), 1);
    dim3 grid(static_cast<unsigned>((N + Br - 1) / Br),
              static_cast<unsigned>(B * H), 1);

    flash_fwd_v1_kernel<Br, Bc><<<grid, block, smem, stream>>>(
        Q_d, K_d, V_d, O_d,
        B, H, N, D,
        is_causal ? 1 : 0, scale);
    cuda_check(cudaGetLastError(), "flash_fwd_v1_kernel launch");

    cuda_check(cudaStreamSynchronize(stream), "flash_fwd_v1_forward sync");
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API — host-pointer overload
// ─────────────────────────────────────────────────────────────────────────────
void flash_fwd_v1_forward_host(
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

    flash_fwd_v1_forward(Q_d, K_d, V_d, O_d, B, H, N, D, is_causal, /*stream=*/0);

    cuda_check(cudaMemcpy(O_h, O_d, nd_bytes, cudaMemcpyDeviceToHost), "D2H O");

    cuda_check(cudaFree(Q_d), "cudaFree Q");
    cuda_check(cudaFree(K_d), "cudaFree K");
    cuda_check(cudaFree(V_d), "cudaFree V");
    cuda_check(cudaFree(O_d), "cudaFree O");
}

}  // namespace flash_from_scratch
