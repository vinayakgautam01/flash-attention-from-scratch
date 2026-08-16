// flash_fwd_v2_shared_kv.cu — the M6 "Flash v2" forward kernel.
//
// Same algebra as flash_fwd_v1.cu, byte for byte. Same recurrence, same tile
// shapes, same tolerance. What changes is where operands live and how many
// hardware cycles it costs to fetch them. theory/M6.md has the full derivation;
// the short version of *why* each change exists is inline at each step below.
//
// Per-row recurrence (unchanged from M3/M4, one copy per Q row):
//
//   m_new = max(m_old, m̃_local)
//   α     = exp(m_old − m_new)                    ∈ (0, 1]
//   p̃     = exp(s − m_new)
//   ℓ_new = α · ℓ_old + rowsum(p̃)
//   Õ_new = α · Õ_old + p̃ · V_tile
//
// After all Tc tiles: O[i, :] = Õ / ℓ.
//
// ─────────────────────────────────────────────────────────────────────────────
// WHAT MOVED, AND WHY (measured baseline: v1 at N=2048, D=64 on T4 = 11.65 ms,
// which is 1.1% of peak fp32 and 1.8% of peak HBM bandwidth — i.e. v1 is
// bound by neither compute nor DRAM, but by on-chip stalls):
//
//  [1] sK row stride D -> D+1.
//      v1's STEP 2 reads sK[tx * D + d]: within a warp `tx` is the lane and
//      `d` is loop-invariant, so the lane stride is D. Bank = addr % 32, and
//      with D ∈ {32, 64} every lane lands on the SAME bank:
//          conflict degree = gcd(D, 32) = 32.
//      That one line is ~91% of all shared-memory wavefronts in v1 (2048 of
//      2252 per warp per tile at D=64; the full audit is in theory/M6.md §5.4).
//      Padding the stride to D+1 makes it odd, so
//          gcd(D + 1, 32) = 1  ->  conflict-free.
//      Cost: Bc extra floats = 128 B. See theory/M6.md §5.
//
//      NOTE (correction to docs/flash_attention_notes.md §2, which claimed sV
//      had the same problem): it does not. In STEP 7 `c` is the loop counter
//      (uniform across lanes) and `d = tx + k*Bc` carries the lane, so the lane
//      stride there is 1 — already conflict-free in v1 and still so here.
//
//  [2] Õ: shared memory -> registers.
//      v1's sO[ty*D + d] is only ever touched by the one thread that owns
//      (ty, d) — it is thread-private state sitting in a shared buffer. Lifting
//      it to `acc_o[RowsPerThread][L]` frees Br*D floats of smem (8 KB at
//      D = 64) and removes 2*L smem round-trips per tile. Requires D to be a
//      *template* parameter so L = D/Bc is compile-time and ptxas allocates
//      real registers rather than spilling to local memory. See theory/M6.md §6.
//
//  [3] 1024 threads/CTA -> 512, two Q rows per thread.
//      Turing caps 1024 threads/SM, so v1's 1024-thread CTA was exactly one CTA
//      per SM: at every __syncthreads() all 32 warps froze together with no
//      other CTA's warps for the schedulers to issue. 512 threads makes two
//      CTAs resident, so one CTA's barrier is covered by the other's work.
//      Bonus: sK[tx] is loaded once per `d` and reused by both of the thread's
//      Q rows, halving sK traffic per Q row. See theory/M6.md §7.
//
//  [4] Scalar global loads -> float4, via a linearized loop.
//      v1's loads were already perfectly coalesced, so this does not reduce
//      HBM *bytes* — it reduces load instructions 4x and the address arithmetic
//      that goes with them. Honest expectation: the smallest of the four wins,
//      because HBM is only ~1.8% of v1's runtime. The linearized form also
//      removes v1's hidden `Br == Bc` coupling (flash_attention_notes.md §11).
//      See theory/M6.md §8.
//
//  [5] The Br x Bc probability tile P never reaches shared memory.
//      Falls out of [3]: a warp is 32 consecutive tx at fixed ty, so row i of P
//      is written AND read entirely within one warp. __shfl_sync moves it
//      lane-to-lane. Removes the sS buffer (4 KB) and one __syncthreads() per
//      tile. This is strictly safer than v1's sS: with no shared array there is
//      no cross-warp hazard to reason about. See theory/M6.md §9.
// ─────────────────────────────────────────────────────────────────────────────

#include "flash_fwd_v2_shared_kv.cuh"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace flash_from_scratch {

namespace {

// Mirror check() helpers from attention_naive.cu / flash_fwd_v1.cu.
inline void cuda_check(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        std::exit(1);
    }
}

// Row-major offset for a [B, H, N, D] tensor.
__device__ __forceinline__ int bhnd_offset(int b, int h, int i, int d,
                                           int H, int N, int D) {
    return ((b * H + h) * N + i) * D + d;
}

// Warp-level butterfly reduction (max). Every lane ends with the row max.
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, stride));
    }
    return v;
}

// Warp-level butterfly reduction (sum). Every lane ends with the row sum.
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int stride = 16; stride > 0; stride >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, stride);
    }
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel — flash_fwd_v2
//
// Grid : (Tr = ⌈N/Br⌉,   B*H,   1)
// Block: (Bc,             Br/RowsPerThread,   1)   — 512 threads at Br=Bc=32, R=2
//
// Shared-memory layout (extern __shared__ float smem[], offsets in floats):
//   [0,                     Br*D)              : sQ  [Br][D]     persistent
//   [Br*D,        +Bc*(D+1))                   : sK  [Bc][D+1]   per-tile, PADDED
//   [+Bc*(D+1),   +Bc*D)                       : sV  [Bc][D]     per-tile
//
// Total smem = 4 * (Br*D + Bc*(D+1) + Bc*D) bytes (fp32):
//   D = 64 -> 24,704 B     (v1: 37,120 B)
//   D = 32 -> 12,416 B     (v1: 20,736 B)
// Two CTAs therefore fit in T4's 64 KB/SM at both D (49,408 / 24,832 B).
//
// Note there is no sS, no sm, no sl: P travels by shuffle, and (m, ℓ) live in
// registers replicated across the warp (both warp reductions broadcast their
// result to every lane, so all 32 lanes compute bit-identical state).
//
// Compile-time constraints:
//   Bc == warpSize == 32                     (row reductions use __shfl)
//   Br % RowsPerThread == 0                  (warps tile the Q rows evenly)
//   D % Bc == 0                              (L = D/Bc accumulator registers)
//   D % 4  == 0                              (float4 load path)
//   Bc * (Br / RowsPerThread) <= 1024        (CUDA per-CTA thread cap)
//
// Note there is NO Br == Bc requirement (v1 had one, via a static_assert): the
// linearized load loop below covers any [Bc][D] tile with any thread count.
// ─────────────────────────────────────────────────────────────────────────────
// __launch_bounds__(maxThreadsPerBlock=512, minBlocksPerMultiprocessor=2)
// states the occupancy intent from [3] directly to ptxas: fit two CTAs of 512
// threads on one SM. That implies a 65,536 / 1024 = 64 register/thread ceiling.
// Rough demand here is ~40-45 registers at D=64, so this should be comfortable
// — but `ptxas -v` is the arbiter. If it reports spill stores, drop the second
// argument to 1 (one CTA, up to 128 registers, no spills) and re-measure; that
// fork is exactly what docs/ptxas_v1_vs_v2.md is for.
template <int Br, int Bc, int D, int RowsPerThread>
__global__ void __launch_bounds__(Bc * (Br / RowsPerThread), 2)
flash_fwd_v2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N,
    int is_causal, float scale)
{
    static_assert(Bc == 32, "flash_fwd_v2_kernel: Bc must equal warpSize=32.");
    static_assert(Br % RowsPerThread == 0,
                  "flash_fwd_v2_kernel: RowsPerThread must divide Br.");
    static_assert(D % Bc == 0,
                  "flash_fwd_v2_kernel: D must be a multiple of Bc (L = D/Bc regs).");
    static_assert(D % 4 == 0,
                  "flash_fwd_v2_kernel: D must be a multiple of 4 for the float4 path.");
    static_assert(Bc * (Br / RowsPerThread) <= 1024,
                  "flash_fwd_v2_kernel: threads/CTA must fit CUDA's 1024 cap.");

    // Number of warps in the CTA == number of Q rows handled per RowsPerThread slot.
    constexpr int kWarps    = Br / RowsPerThread;      // 16
    constexpr int kThreads  = Bc * kWarps;             // 512
    constexpr int kSKStride = D + 1;                   // [1] the bank-conflict fix
    constexpr int L         = D / Bc;                  // [2] accumulator regs per row
    constexpr int D4        = D / 4;                   // [4] float4 per tensor row

    // 16-byte alignment is required for the float4 accesses into sQ / sV below.
    // sV's offset (Br*D + Bc*(D+1)) is a multiple of 4 floats for all supported
    // (Br, Bc, D), so a 16-byte-aligned base keeps every float4 view aligned.
    extern __shared__ __align__(16) float smem[];
    float* sQ = smem;                    // [Br][D]
    float* sK = sQ + Br * D;             // [Bc][D+1]   <- padded stride
    float* sV = sK + Bc * kSKStride;     // [Bc][D]

    const int i_tile = blockIdx.x;
    const int bh     = blockIdx.y;
    const int b      = bh / H;
    const int h      = bh % H;

    const int tx  = threadIdx.x;          // lane within the warp, 0..Bc-1
    const int ty  = threadIdx.y;          // warp id,              0..kWarps-1
    const int tid = ty * Bc + tx;         // linear thread id,     0..kThreads-1

    (void)B;  // signature symmetry with v1; grid.y already carries B*H.

    // ── Which Q rows does this thread own? ───────────────────────────────────
    // Warp `ty` owns local rows {ty, ty + kWarps, ...}. Splitting by kWarps
    // (rather than 2*ty, 2*ty+1) keeps each warp's Q reads on one sQ row at a
    // time, which stays a broadcast in STEP 2.
    int  q_local[RowsPerThread];
    int  i_glob[RowsPerThread];
    bool i_ok[RowsPerThread];
    #pragma unroll
    for (int rr = 0; rr < RowsPerThread; ++rr) {
        q_local[rr] = ty + rr * kWarps;
        i_glob[rr]  = i_tile * Br + q_local[rr];
        i_ok[rr]    = (i_glob[rr] < N);
    }

    // ── Running state, entirely in registers ─────────────────────────────────
    // Both warp reductions broadcast to all lanes, so every lane of a warp
    // holds bit-identical (m, ℓ) for its rows. No shared memory, no barrier.
    float acc_o[RowsPerThread][L];
    float m_run[RowsPerThread];
    float l_run[RowsPerThread];
    #pragma unroll
    for (int rr = 0; rr < RowsPerThread; ++rr) {
        m_run[rr] = -CUDART_INF_F;
        l_run[rr] = 0.0f;
        #pragma unroll
        for (int k = 0; k < L; ++k) acc_o[rr][k] = 0.0f;
    }

    // ── Init: load the Q-tile once, vectorized ───────────────────────────────
    // Linearized over the tile's float4 count so the mapping is independent of
    // (Br, Bc, thread count). Consecutive tids cover consecutive float4s within
    // a row, so the global side stays fully coalesced.
    for (int v4 = tid; v4 < Br * D4; v4 += kThreads) {
        const int r  = v4 / D4;
        const int c4 = v4 % D4;
        const int i  = i_tile * Br + r;
        float4 q = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (i < N) {
            q = reinterpret_cast<const float4*>(
                    Q + bhnd_offset(b, h, i, 0, H, N, D))[c4];
        }
        reinterpret_cast<float4*>(sQ + r * D)[c4] = q;
    }

    const int Tc = (N + Bc - 1) / Bc;

    for (int b_idx = 0; b_idx < Tc; ++b_idx) {
        const int j0 = b_idx * Bc;

        // Guard the sK/sV overwrite against the previous iteration's readers
        // (STEP 7 below). Paired with the barrier after the load; those two are
        // the only barriers in the loop — v1 needed three.
        __syncthreads();

        // ── STEP 1: cooperative, vectorized K/V tile load ────────────────────
        for (int v4 = tid; v4 < Bc * D4; v4 += kThreads) {
            const int r  = v4 / D4;
            const int c4 = v4 % D4;
            const int j  = j0 + r;
            float4 kk = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 vv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (j < N) {
                const int off = bhnd_offset(b, h, j, 0, H, N, D);
                kk = reinterpret_cast<const float4*>(K + off)[c4];
                vv = reinterpret_cast<const float4*>(V + off)[c4];
            }
            // sK's padded stride (D+1) is odd, so a float4 store into it would
            // be misaligned; scatter the four scalars instead. This costs a
            // 2-way store conflict on ~8 stores per warp per tile, against the
            // 2048-wavefront read conflict the padding removes. Worth it.
            float* dst_k = sK + r * kSKStride + 4 * c4;
            dst_k[0] = kk.x;
            dst_k[1] = kk.y;
            dst_k[2] = kk.z;
            dst_k[3] = kk.w;
            // sV keeps the natural stride D, so the float4 store stands.
            reinterpret_cast<float4*>(sV + r * D)[c4] = vv;
        }
        __syncthreads();

        // ── STEP 2: scores. sK[tx] is read ONCE per d and reused by both of
        // this thread's Q rows — that reuse is the payoff of [3]. sQ is a
        // broadcast (no tx in the index); sK is conflict-free thanks to [1].
        //
        // The dot product runs unconditionally: out-of-range Q rows and KV
        // columns were zero-filled at load time, so the accumulator is finite
        // and gets masked to -inf immediately below. Cheaper than branching
        // inside the hot loop, and it keeps the warp convergent for STEP 3.
        float s_acc[RowsPerThread];
        #pragma unroll
        for (int rr = 0; rr < RowsPerThread; ++rr) s_acc[rr] = 0.0f;

        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            const float k_d = sK[tx * kSKStride + d];
            #pragma unroll
            for (int rr = 0; rr < RowsPerThread; ++rr) {
                s_acc[rr] += sQ[q_local[rr] * D + d] * k_d;
            }
        }

        const int  j    = j0 + tx;        // this lane's KV column
        const bool j_ok = (j < N);

        float p_val[RowsPerThread];
        float alpha[RowsPerThread];

        #pragma unroll
        for (int rr = 0; rr < RowsPerThread; ++rr) {
            const bool causal_ok = (is_causal == 0) || (j <= i_glob[rr]);
            const bool active    = i_ok[rr] && j_ok && causal_ok;
            const float s_val    = active ? (s_acc[rr] * scale) : -CUDART_INF_F;

            // ── STEP 3: row-max across the tile. All 32 lanes participate
            // unconditionally (inactive lanes carry -inf, which cannot win).
            const float m_local = warp_reduce_max(s_val);

            // ── STEP 4: per-row α and m_new, register-resident. Identical
            // guards to v1 so the numerics match exactly.
            const float m_old = m_run[rr];
            float m_new;
            if (!isfinite(m_old) && !isfinite(m_local)) {
                m_new = -CUDART_INF_F;   // whole row masked so far
            } else {
                m_new = fmaxf(m_old, m_local);
            }

            float a;
            if (!isfinite(m_new)) {
                a = 1.0f;   // no state yet; α = 1 leaves ℓ, Õ untouched.
            } else if (!isfinite(m_old)) {
                a = 0.0f;   // first real block — annihilate the zero init.
            } else {
                a = expf(m_old - m_new);   // ∈ (0, 1]
            }
            alpha[rr] = a;

            // ── STEP 5: p̃ and the row-sum. p̃ stays in a register — it is
            // consumed by this lane's own warp in STEP 7 (see [5]).
            p_val[rr] = (active && isfinite(m_new)) ? expf(s_val - m_new) : 0.0f;
            const float ell_local = warp_reduce_sum(p_val[rr]);

            // ── STEP 6: merge (m, ℓ). Every lane performs the same update on
            // its own copy, so no write-back or broadcast is needed.
            l_run[rr] = a * l_run[rr] + ell_local;
            m_run[rr] = m_new;
        }

        // ── STEP 7: Õ update — acc_o ← α·acc_o + p̃ · sV ─────────────────────
        // `c` is the outer loop so each sV element is fetched once and reused
        // across both Q rows (the second half of [3]'s payoff). p̃ arrives by
        // shuffle: lane `c` of this warp holds p̃[row, c], and every lane needs
        // it, so __shfl_sync with a uniform `c` broadcasts it. All 32 lanes
        // reach every shuffle — the loop bounds are compile-time constants and
        // there is no divergence here.
        #pragma unroll
        for (int rr = 0; rr < RowsPerThread; ++rr) {
            #pragma unroll
            for (int k = 0; k < L; ++k) acc_o[rr][k] *= alpha[rr];
        }

        #pragma unroll 4
        for (int c = 0; c < Bc; ++c) {
            float v_reg[L];
            #pragma unroll
            for (int k = 0; k < L; ++k) {
                // Lane stride is 1 (tx is the only lane-varying term), so this
                // is conflict-free — as it already was in v1.
                v_reg[k] = sV[c * D + tx + k * Bc];
            }
            #pragma unroll
            for (int rr = 0; rr < RowsPerThread; ++rr) {
                const float p_c = __shfl_sync(0xffffffffu, p_val[rr], c);
                #pragma unroll
                for (int k = 0; k < L; ++k) {
                    acc_o[rr][k] += p_c * v_reg[k];
                }
            }
        }
    }

    // ── Terminal: normalize and emit O[i, :] = Õ / ℓ ──────────────────────────
    // Tail-padding rows (i >= N) never write. ℓ <= 0 (a pathologically
    // all-masked or fully-underflowed row) writes zeros rather than NaN —
    // same guard as M3/M4.
    //
    // Writes stay scalar: thread (tx, ty) owns d = tx, tx+Bc, ..., so lanes
    // cover consecutive d and the store is coalesced. A float4 store would
    // need each thread to own 4 *consecutive* d, which would in turn make the
    // STEP 7 sV read lane-strided by 4 — trading a conflict-free read for a
    // 4-way conflict to speed up Br*D writes that happen once per CTA. Not worth it.
    #pragma unroll
    for (int rr = 0; rr < RowsPerThread; ++rr) {
        if (!i_ok[rr]) continue;
        const float ell = l_run[rr];
        const bool  ok  = ell > 0.0f;
        #pragma unroll
        for (int k = 0; k < L; ++k) {
            const int d = tx + k * Bc;
            O[bhnd_offset(b, h, i_glob[rr], d, H, N, D)] =
                ok ? (acc_o[rr][k] / ell) : 0.0f;
        }
    }
}

// Launch helper — one instantiation per compile-time D. The smem figure comes
// from flash_v2_smem_bytes() in the header so the layout has exactly one
// source of truth (the GoogleTest asserts against the same function).
template <int D>
void launch_v2(const float* Q_d, const float* K_d, const float* V_d, float* O_d,
               int B, int H, int N, bool is_causal, cudaStream_t stream) {
    constexpr int Br = kFlashV2Br;
    constexpr int Bc = kFlashV2Bc;
    constexpr int R  = kFlashV2RowsPerThread;

    const float  scale = 1.0f / std::sqrt(static_cast<float>(D));
    const size_t smem  = static_cast<size_t>(flash_v2_smem_bytes(Br, Bc, D));

    // Same guard as v1: fits T4's default 48 KB per-CTA budget without a
    // cudaFuncSetAttribute opt-in. 24,704 B at D=64, 12,416 B at D=32.
    //
    // Note this is the *per-CTA* cap, not the per-SM one. Reaching the 2 CTAs
    // that __launch_bounds__ asks for also needs the driver to pick a 64 KB
    // shared-memory carveout on Turing (2 * 24,704 = 49,408 B > the 32 KB
    // carveout tier). It does that automatically from the launch's smem
    // request; if the occupancy ever comes up short of 2, this is the first
    // thing to check. See docs/ptxas_v1_vs_v2.md §2.2.
    if (smem > static_cast<size_t>(48 * 1024)) {
        std::fprintf(stderr, "flash_fwd_v2_forward: smem %zu B exceeds the default "
                             "48 KB per-CTA cap. Enable dynamic smem via "
                             "cudaFuncSetAttribute before landing wider D.\n", smem);
        std::exit(1);
    }

    dim3 block(static_cast<unsigned>(Bc),
               static_cast<unsigned>(Br / R), 1);
    dim3 grid(static_cast<unsigned>((N + Br - 1) / Br),
              static_cast<unsigned>(B * H), 1);

    flash_fwd_v2_kernel<Br, Bc, D, R><<<grid, block, smem, stream>>>(
        Q_d, K_d, V_d, O_d,
        B, H, N,
        is_causal ? 1 : 0, scale);
    cuda_check(cudaGetLastError(), "flash_fwd_v2_kernel launch");
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API — device-pointer overload
// ─────────────────────────────────────────────────────────────────────────────
void flash_fwd_v2_forward(
    const float* Q_d, const float* K_d, const float* V_d, float* O_d,
    int B, int H, int N, int D, bool is_causal,
    cudaStream_t stream)
{
    if (B <= 0 || H <= 0 || N <= 0 || D <= 0) {
        std::fprintf(stderr, "flash_fwd_v2_forward: B/H/N/D must be positive "
                             "(got B=%d, H=%d, N=%d, D=%d)\n", B, H, N, D);
        std::exit(1);
    }

    // D is a *template* parameter here (unlike v1, where it was a runtime arg)
    // because the register-resident Õ array needs a compile-time length —
    // otherwise ptxas puts it in local memory and the optimization inverts into
    // a pessimization. That forces an explicit dispatch. D=128 waits for M7:
    // its smem (49,280 B) exceeds the 48 KB default per-CTA cap and needs a
    // cudaFuncSetAttribute opt-in.
    switch (D) {
        case 32:
            launch_v2<32>(Q_d, K_d, V_d, O_d, B, H, N, is_causal, stream);
            break;
        case 64:
            launch_v2<64>(Q_d, K_d, V_d, O_d, B, H, N, is_causal, stream);
            break;
        default:
            std::fprintf(stderr, "flash_fwd_v2_forward: D must be 32 or 64 in v2 "
                                 "(got D=%d). Wider D is deferred to M7.\n", D);
            std::exit(1);
    }

    cuda_check(cudaStreamSynchronize(stream), "flash_fwd_v2_forward sync");
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API — host-pointer overload
// ─────────────────────────────────────────────────────────────────────────────
void flash_fwd_v2_forward_host(
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

    flash_fwd_v2_forward(Q_d, K_d, V_d, O_d, B, H, N, D, is_causal, /*stream=*/0);

    cuda_check(cudaMemcpy(O_h, O_d, nd_bytes, cudaMemcpyDeviceToHost), "D2H O");

    cuda_check(cudaFree(Q_d), "cudaFree Q");
    cuda_check(cudaFree(K_d), "cudaFree K");
    cuda_check(cudaFree(V_d), "cudaFree V");
    cuda_check(cudaFree(O_d), "cudaFree O");
}

}  // namespace flash_from_scratch
