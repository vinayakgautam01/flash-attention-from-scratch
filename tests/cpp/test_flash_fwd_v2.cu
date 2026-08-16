// GoogleTest for csrc/flash_fwd_v2_shared_kv.cu — the M6 "Flash v2" kernel.
//
// M6's premise is "same algebra, better mechanical sympathy", so the bar is
// higher than "matches the CPU reference": v2 must be a drop-in replacement for
// v1 on every shape v1 supports. This file therefore tests three layers:
//
//   1) Parametrized correctness grid — the SAME 10 cases as the M4 grid
//      (N ∈ {128, 256, 512, 1024, 2048}, D ∈ {32, 64}, causal=false), diffed
//      against attention_cpu_ref<float>. MILESTONES §M6: "correctness passes on
//      the same grid as v1."
//
//   2) v1-parity — v2 vs v1 on the same inputs. This is the check that would
//      catch a v2-only regression that happens to stay inside the CPU-ref
//      tolerance. Bound is tighter than the CPU-ref tolerance (see kParityTol
//      below) because both kernels accumulate in the same order.
//
//   3) Edge cases — the M4 set (N=1, uniform, N=100, N=257, peak key,
//      determinism) plus three that are specific to what M6 changed:
//      - N=17    → exercises the RowsPerThread=2 split: warp 0's second row
//                  (local row 16) is valid while warps 1..15's second rows are
//                  tail padding. v1 had no such split, so this path is new.
//      - Causal smoke → v2 restructured the mask (it moved out of the
//                  dot-product branch and is now applied to the accumulated
//                  score). Formal causal testing is M7, but leaving a
//                  restructured mask completely untested until then would be
//                  inheriting an untested path — same reasoning M4 used for
//                  its N=257 case.
//      - Config guard → asserts the smem budget and thread count that the
//                  occupancy argument in theory/M6.md §7 depends on. If someone
//                  retunes kFlashV2RowsPerThread, this fails loudly rather
//                  than silently invalidating the writeup.
//
// Tolerance: 5e-4 abs vs CPU ref — identical to M4, per docs/AGENTS.md §9.
// v2 introduces no new axis of accumulation, so the bound does not move.

#include <cmath>
#include <cstdio>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "attention_cpu_ref.hpp"
#include "flash_fwd_v1.cuh"
#include "flash_fwd_v2_shared_kv.cuh"

namespace {

using flash_from_scratch::attention_cpu_ref;
using flash_from_scratch::bhnd_index;
using flash_from_scratch::flash_fwd_v1_forward_host;
using flash_from_scratch::flash_fwd_v2_forward_host;
using flash_from_scratch::flash_v2_smem_bytes;
using flash_from_scratch::kFlashV2Bc;
using flash_from_scratch::kFlashV2Br;
using flash_from_scratch::kFlashV2RowsPerThread;

// Fixed-seed Gaussian tensor generator (mirrors tests/cpp/test_flash_fwd_v1.cu).
std::vector<float> gaussian_tensor(std::size_t n, std::uint64_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> out(n);
    for (auto& x : out) x = dist(gen);
    return out;
}

// CUDA-vs-CPU-ref fp32 tolerance. docs/AGENTS.md §9, row "CUDA kernels, fp32".
constexpr float kAbsTolFp32 = 5e-4f;

// v1-vs-v2 parity bound. Deliberately tighter than kAbsTolFp32: the two kernels
// sum the D-axis and the Bc-axis in the same order, so they should agree to
// near round-off. They are NOT expected to be bit-identical — v2 reassociates
// the two Q rows' dot products into one loop and reads V through registers, so
// FMA contraction can differ. 1e-5 is loose enough for that and tight enough to
// catch a genuine algebraic divergence.
constexpr float kParityTol = 1e-5f;

float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ── Parametrized correctness grid ────────────────────────────────────────────
class FlashFwdV2Grid
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(FlashFwdV2Grid, MatchesCpuRefWithinTolerance) {
    const auto& p = GetParam();
    const int  N = std::get<0>(p);
    const int  D = std::get<1>(p);
    constexpr int B = 1;
    constexpr int H = 1;
    constexpr bool is_causal = false;
    const std::size_t nelems = static_cast<std::size_t>(B) * H * N * D;

    const auto Q = gaussian_tensor(nelems, /*seed=*/1000 + N + D);
    const auto K = gaussian_tensor(nelems, /*seed=*/2000 + N + D);
    const auto V = gaussian_tensor(nelems, /*seed=*/3000 + N + D);

    std::vector<float> O_cpu(nelems, 0.0f);
    std::vector<float> O_gpu(nelems, 0.0f);

    attention_cpu_ref<float>(Q.data(), K.data(), V.data(), O_cpu.data(),
                             B, H, N, D, is_causal);
    flash_fwd_v2_forward_host(Q.data(), K.data(), V.data(), O_gpu.data(),
                              B, H, N, D, is_causal);

    for (float x : O_gpu) {
        ASSERT_TRUE(std::isfinite(x))
            << "non-finite entry in O_gpu at N=" << N << " D=" << D;
    }

    const float err = max_abs_error(O_gpu, O_cpu);
    EXPECT_LT(err, kAbsTolFp32)
        << "N=" << N << " D=" << D << " · max_abs_error=" << err;
}

INSTANTIATE_TEST_SUITE_P(
    M6Grid, FlashFwdV2Grid,
    ::testing::Combine(
        ::testing::Values(128, 256, 512, 1024, 2048),   // same grid as M4
        ::testing::Values(32, 64)),                     // v2 supports {32, 64}
    [](const ::testing::TestParamInfo<FlashFwdV2Grid::ParamType>& info) {
        const int N = std::get<0>(info.param);
        const int D = std::get<1>(info.param);
        return "N" + std::to_string(N) + "_D" + std::to_string(D);
    });

// ── v1 parity — the entry condition for quoting any M6 speedup ───────────────
class FlashFwdV2Parity
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(FlashFwdV2Parity, MatchesV1) {
    const auto& p = GetParam();
    const int  N = std::get<0>(p);
    const int  D = std::get<1>(p);
    constexpr int B = 1;
    constexpr int H = 1;
    const std::size_t nelems = static_cast<std::size_t>(B) * H * N * D;

    const auto Q = gaussian_tensor(nelems, /*seed=*/4100 + N + D);
    const auto K = gaussian_tensor(nelems, /*seed=*/4200 + N + D);
    const auto V = gaussian_tensor(nelems, /*seed=*/4300 + N + D);

    std::vector<float> O_v1(nelems, 0.0f);
    std::vector<float> O_v2(nelems, 0.0f);

    flash_fwd_v1_forward_host(Q.data(), K.data(), V.data(), O_v1.data(),
                              B, H, N, D, /*is_causal=*/false);
    flash_fwd_v2_forward_host(Q.data(), K.data(), V.data(), O_v2.data(),
                              B, H, N, D, /*is_causal=*/false);

    const float err = max_abs_error(O_v2, O_v1);
    EXPECT_LT(err, kParityTol)
        << "v1-vs-v2 divergence at N=" << N << " D=" << D
        << " · max_abs_error=" << err;
}

INSTANTIATE_TEST_SUITE_P(
    M6Parity, FlashFwdV2Parity,
    ::testing::Combine(
        ::testing::Values(128, 512, 2048),
        ::testing::Values(32, 64)),
    [](const ::testing::TestParamInfo<FlashFwdV2Parity::ParamType>& info) {
        const int N = std::get<0>(info.param);
        const int D = std::get<1>(info.param);
        return "N" + std::to_string(N) + "_D" + std::to_string(D);
    });

// ── Config guard — the numbers theory/M6.md §7's occupancy claim rests on ────
// Not a kernel test: a guard so a future retune of the tile/thread config
// can't silently invalidate the writeup's occupancy arithmetic.
TEST(FlashFwdV2Config, LaunchConfigMatchesOccupancyClaim) {
    EXPECT_EQ(kFlashV2Br, 32);
    EXPECT_EQ(kFlashV2Bc, 32);
    EXPECT_EQ(kFlashV2RowsPerThread, 2);

    const int threads_per_cta = kFlashV2Bc * (kFlashV2Br / kFlashV2RowsPerThread);
    EXPECT_EQ(threads_per_cta, 512)
        << "two CTAs of this size must fit T4's 1024 threads/SM";
    EXPECT_LE(2 * threads_per_cta, 1024);

    // Shared memory: two CTAs must fit T4's 64 KB/SM, and one must fit the
    // 48 KB default per-CTA cap (no cudaFuncSetAttribute opt-in in M6).
    for (int D : {32, 64}) {
        const int smem = flash_v2_smem_bytes(kFlashV2Br, kFlashV2Bc, D);
        EXPECT_LE(smem, 48 * 1024) << "D=" << D << " smem=" << smem;
        EXPECT_LE(2 * smem, 64 * 1024) << "D=" << D << " smem=" << smem;
    }
    // Exact figures quoted in theory/M6.md §7 and docs/ptxas_v1_vs_v2.md.
    EXPECT_EQ(flash_v2_smem_bytes(32, 32, 64), 24704);
    EXPECT_EQ(flash_v2_smem_bytes(32, 32, 32), 12416);
}

// ── Edge case: N=1 → single key → O == V. ────────────────────────────────────
TEST(FlashFwdV2Edge, SinglePositionOutputEqualsValue) {
    constexpr int B = 1, H = 1, N = 1, D = 32;
    const auto Q = gaussian_tensor(B * H * N * D, /*seed=*/5000);
    const auto K = gaussian_tensor(B * H * N * D, /*seed=*/5001);
    const auto V = gaussian_tensor(B * H * N * D, /*seed=*/5002);
    std::vector<float> O(B * H * N * D, 0.0f);

    flash_fwd_v2_forward_host(Q.data(), K.data(), V.data(), O.data(),
                              B, H, N, D, /*is_causal=*/false);
    for (int d = 0; d < D; ++d) {
        EXPECT_NEAR(O[d], V[d], kAbsTolFp32) << "d=" << d;
    }
}

// ── Edge case: uniform Q, uniform K → uniform attention → O[i] == mean(V). ──
TEST(FlashFwdV2Edge, UniformQKGivesMeanOfV) {
    constexpr int B = 1, H = 1, N = 8, D = 32;
    std::vector<float> Q(B * H * N * D, 0.3f);
    std::vector<float> K(B * H * N * D, 0.7f);
    const auto V = gaussian_tensor(B * H * N * D, /*seed=*/7000);
    std::vector<float> O(B * H * N * D, 0.0f);

    flash_fwd_v2_forward_host(Q.data(), K.data(), V.data(), O.data(),
                              B, H, N, D, /*is_causal=*/false);

    std::vector<float> mean_v(D, 0.0f);
    for (int j = 0; j < N; ++j) {
        for (int d = 0; d < D; ++d) {
            mean_v[d] += V[bhnd_index(0, 0, j, d, 1, N, D)];
        }
    }
    for (int d = 0; d < D; ++d) mean_v[d] /= static_cast<float>(N);

    for (int i = 0; i < N; ++i) {
        for (int d = 0; d < D; ++d) {
            const auto o = O[bhnd_index(0, 0, i, d, 1, N, D)];
            EXPECT_NEAR(o, mean_v[d], kAbsTolFp32) << "i=" << i << " d=" << d;
        }
    }
}

// ── Edge case: N=17 → the RowsPerThread=2 row split, straddled. ──────────────
// M6-specific. With Br=32 and RowsPerThread=2 the CTA has 16 warps; warp `ty`
// owns local rows `ty` (rr=0) and `ty + 16` (rr=1). At N=17 exactly one rr=1
// row is live (warp 0's local row 16) while the other fifteen are tail padding.
// v1 had one row per warp and could not exercise this; a bug in the q_local /
// i_glob split would show up here and nowhere else in the grid.
TEST(FlashFwdV2Edge, RowsPerThreadSplitStraddlesTail) {
    constexpr int B = 1, H = 1, N = 17, D = 32;
    const std::size_t n = B * H * N * D;
    const auto Q = gaussian_tensor(n, /*seed=*/9200);
    const auto K = gaussian_tensor(n, /*seed=*/9201);
    const auto V = gaussian_tensor(n, /*seed=*/9202);

    std::vector<float> O_cpu(n, 0.0f), O_gpu(n, 0.0f);
    attention_cpu_ref<float>(Q.data(), K.data(), V.data(), O_cpu.data(),
                             B, H, N, D, /*is_causal=*/false);
    flash_fwd_v2_forward_host(Q.data(), K.data(), V.data(), O_gpu.data(),
                              B, H, N, D, /*is_causal=*/false);

    for (float x : O_gpu) ASSERT_TRUE(std::isfinite(x));
    const float err = max_abs_error(O_gpu, O_cpu);
    EXPECT_LT(err, kAbsTolFp32) << "N=17 row-split max_abs_error=" << err;
}

// ── Edge case: N=100 (non-multiple of Bc=32) → partial-KV-tile correctness. ──
TEST(FlashFwdV2Edge, PartialKvTailBlock) {
    constexpr int B = 1, H = 1, N = 100, D = 32;
    const std::size_t n = B * H * N * D;
    const auto Q = gaussian_tensor(n, /*seed=*/9000);
    const auto K = gaussian_tensor(n, /*seed=*/9001);
    const auto V = gaussian_tensor(n, /*seed=*/9002);

    std::vector<float> O_cpu(n, 0.0f), O_gpu(n, 0.0f);
    attention_cpu_ref<float>(Q.data(), K.data(), V.data(), O_cpu.data(),
                             B, H, N, D, /*is_causal=*/false);
    flash_fwd_v2_forward_host(Q.data(), K.data(), V.data(), O_gpu.data(),
                              B, H, N, D, /*is_causal=*/false);

    for (float x : O_gpu) ASSERT_TRUE(std::isfinite(x));
    const float err = max_abs_error(O_gpu, O_cpu);
    EXPECT_LT(err, kAbsTolFp32) << "partial-KV-tail max_abs_error=" << err;
}

// ── Edge case: N=257 → partial Q-tile AND partial KV-tile tails. ─────────────
TEST(FlashFwdV2Edge, PartialQAndKvTailBlocks) {
    constexpr int B = 1, H = 1, N = 257, D = 32;
    const std::size_t n = B * H * N * D;
    const auto Q = gaussian_tensor(n, /*seed=*/9100);
    const auto K = gaussian_tensor(n, /*seed=*/9101);
    const auto V = gaussian_tensor(n, /*seed=*/9102);

    std::vector<float> O_cpu(n, 0.0f), O_gpu(n, 0.0f);
    attention_cpu_ref<float>(Q.data(), K.data(), V.data(), O_cpu.data(),
                             B, H, N, D, /*is_causal=*/false);
    flash_fwd_v2_forward_host(Q.data(), K.data(), V.data(), O_gpu.data(),
                              B, H, N, D, /*is_causal=*/false);

    for (float x : O_gpu) ASSERT_TRUE(std::isfinite(x));
    const float err = max_abs_error(O_gpu, O_cpu);
    EXPECT_LT(err, kAbsTolFp32) << "partial-Q+KV-tail max_abs_error=" << err;
}

// ── Edge case: one score dominates → O ≈ V[j*] for the peak key j*. ─────────
TEST(FlashFwdV2Edge, PeakKeyDominates) {
    constexpr int B = 1, H = 1, N = 128, D = 32;
    constexpr int j_star = 77;    // middle of KV-tile 2 (Bc=32 → covers 64..95)
    const std::size_t n = B * H * N * D;

    std::vector<float> Q(n, 0.0f);
    std::vector<float> K(n, 0.0f);
    std::vector<float> V(n, 0.0f);

    Q[bhnd_index(0, 0, 0, 0, 1, N, D)] = 1.0f;
    K[bhnd_index(0, 0, j_star, 0, 1, N, D)] = 100.0f;
    for (int d = 0; d < D; ++d) {
        V[bhnd_index(0, 0, j_star, d, 1, N, D)] = static_cast<float>(d + 1);
    }

    std::vector<float> O(n, 0.0f);
    flash_fwd_v2_forward_host(Q.data(), K.data(), V.data(), O.data(),
                              B, H, N, D, /*is_causal=*/false);

    // Same 1e-4 bound as the M4 test: the denominator carries (N-1) small-exp
    // terms that don't quite vanish in fp32.
    for (int d = 0; d < D; ++d) {
        const float expected = static_cast<float>(d + 1);
        const float got = O[bhnd_index(0, 0, 0, d, 1, N, D)];
        EXPECT_NEAR(got, expected, 1e-4f)
            << "peak-key: d=" << d << " got=" << got << " expected=" << expected;
    }
}

// ── Causal smoke test — guards the restructured mask until M7 formalizes it. ─
// v2 applies the causal predicate to the accumulated score rather than gating
// the dot-product loop (v1's shape). That restructure is exactly the kind of
// change that can silently invert a comparison, so we pin one shape where the
// diagonal tile is genuinely mixed (N=100 → the last KV-tile is partial AND
// the diagonal crosses inside several tiles). Full causal grid lands in M7.
TEST(FlashFwdV2Edge, CausalSmokeMatchesCpuRef) {
    constexpr int B = 1, H = 1, N = 100, D = 32;
    const std::size_t n = B * H * N * D;
    const auto Q = gaussian_tensor(n, /*seed=*/9300);
    const auto K = gaussian_tensor(n, /*seed=*/9301);
    const auto V = gaussian_tensor(n, /*seed=*/9302);

    std::vector<float> O_cpu(n, 0.0f), O_gpu(n, 0.0f);
    attention_cpu_ref<float>(Q.data(), K.data(), V.data(), O_cpu.data(),
                             B, H, N, D, /*is_causal=*/true);
    flash_fwd_v2_forward_host(Q.data(), K.data(), V.data(), O_gpu.data(),
                              B, H, N, D, /*is_causal=*/true);

    for (float x : O_gpu) ASSERT_TRUE(std::isfinite(x));
    const float err = max_abs_error(O_gpu, O_cpu);
    EXPECT_LT(err, kAbsTolFp32) << "causal max_abs_error=" << err;
}

// ── Sanity: two independent runs on identical inputs are bit-identical. ─────
TEST(FlashFwdV2Edge, DeterministicAcrossRuns) {
    constexpr int B = 1, H = 1, N = 128, D = 32;
    const std::size_t n = B * H * N * D;
    const auto Q = gaussian_tensor(n, /*seed=*/8000);
    const auto K = gaussian_tensor(n, /*seed=*/8001);
    const auto V = gaussian_tensor(n, /*seed=*/8002);

    std::vector<float> O1(n, 0.0f), O2(n, 0.0f);
    flash_fwd_v2_forward_host(Q.data(), K.data(), V.data(), O1.data(),
                              B, H, N, D, /*is_causal=*/false);
    flash_fwd_v2_forward_host(Q.data(), K.data(), V.data(), O2.data(),
                              B, H, N, D, /*is_causal=*/false);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(O1[i], O2[i]) << "diverge at i=" << i;
    }
}

}  // namespace
