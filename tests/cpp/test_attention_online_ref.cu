// GoogleTest for csrc/attention_online_ref.cu — the M3 online-softmax kernel.
//
// The kernel is validated in two layers, mirroring the M2 test file:
//
//   1) Parametrized correctness grid — 6 cases covering MILESTONES §M3
//      verification plan (N ∈ {128, 512, 2048}, D ∈ {32, 64}, causal=false),
//      diffed against attention_cpu_ref<float>.
//
//   2) Edge-case tests — N=1, uniform Q/K, N=100 (non-Bc-multiple tail),
//      peak-key row, and determinism across two runs.
//
// Tolerance: 1e-5 abs — TIGHTER than M2's 5e-4. Reason: the online recurrence
// is algebraic equivalence, not floating-point matmul accumulation drift.
// Recorded in `docs/AGENTS.md` §9 alongside the M2 row.

#include <cmath>
#include <cstdio>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "attention_cpu_ref.hpp"
#include "attention_online_ref.cuh"

namespace {

using flash_from_scratch::attention_cpu_ref;
using flash_from_scratch::attention_online_ref_forward_host;
using flash_from_scratch::bhnd_index;

// Fixed-seed Gaussian tensor generator (mirrors tests/cpp/test_attention_naive.cu).
std::vector<float> gaussian_tensor(std::size_t n, std::uint64_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> out(n);
    for (auto& x : out) x = dist(gen);
    return out;
}

// M3 tolerance: algebraic equivalence, not accumulation drift. See
// docs/AGENTS.md §9 (row: "M3 online-softmax vs CPU ref, fp32").
constexpr float kAbsTolFp32 = 1e-5f;

float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ── Parametrized correctness grid ────────────────────────────────────────────
// (N, D) pairs — matches MILESTONES §M3 verification plan. Causal deferred to
// M7 (kernel implements causal via one -inf line but M3 tests keep it off).
class AttentionOnlineRefGrid
    : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(AttentionOnlineRefGrid, MatchesCpuRefWithinTolerance) {
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
    attention_online_ref_forward_host(Q.data(), K.data(), V.data(), O_gpu.data(),
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
    M3Grid, AttentionOnlineRefGrid,
    ::testing::Combine(
        ::testing::Values(128, 512, 2048),   // N — MILESTONES §M3 verification plan
        ::testing::Values(32, 64)),          // D — capped at kOnlineRefBc = 64
    [](const ::testing::TestParamInfo<AttentionOnlineRefGrid::ParamType>& info) {
        const int N = std::get<0>(info.param);
        const int D = std::get<1>(info.param);
        std::string s = "N" + std::to_string(N) + "_D" + std::to_string(D);
        return s;
    });

// ── Edge case: N=1 → single key → O == V. ────────────────────────────────────
TEST(AttentionOnlineRefEdge, SinglePositionOutputEqualsValue) {
    constexpr int B = 1, H = 1, N = 1, D = 8;
    const auto Q = gaussian_tensor(B * H * N * D, /*seed=*/5000);
    const auto K = gaussian_tensor(B * H * N * D, /*seed=*/5001);
    const auto V = gaussian_tensor(B * H * N * D, /*seed=*/5002);
    std::vector<float> O(B * H * N * D, 0.0f);

    attention_online_ref_forward_host(Q.data(), K.data(), V.data(), O.data(),
                                      B, H, N, D, /*is_causal=*/false);
    for (int d = 0; d < D; ++d) {
        EXPECT_NEAR(O[d], V[d], kAbsTolFp32) << "d=" << d;
    }
}

// ── Edge case: uniform Q, uniform K → uniform attention → O[i] == mean(V). ──
TEST(AttentionOnlineRefEdge, UniformQKGivesMeanOfV) {
    constexpr int B = 1, H = 1, N = 8, D = 4;
    std::vector<float> Q(B * H * N * D, 0.3f);
    std::vector<float> K(B * H * N * D, 0.7f);
    const auto V = gaussian_tensor(B * H * N * D, /*seed=*/7000);
    std::vector<float> O(B * H * N * D, 0.0f);

    attention_online_ref_forward_host(Q.data(), K.data(), V.data(), O.data(),
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

// ── Edge case: N=100 (non-multiple of Bc=64) → tail-block correctness. ──────
// This exercises the last-block guard where j >= N for some threads in the
// CTA. If the mask on j_in_range leaks, we'll see garbage in the output.
TEST(AttentionOnlineRefEdge, PartialTailBlock) {
    constexpr int B = 1, H = 1, N = 100, D = 32;
    const std::size_t n = B * H * N * D;
    const auto Q = gaussian_tensor(n, /*seed=*/9000);
    const auto K = gaussian_tensor(n, /*seed=*/9001);
    const auto V = gaussian_tensor(n, /*seed=*/9002);

    std::vector<float> O_cpu(n, 0.0f), O_gpu(n, 0.0f);
    attention_cpu_ref<float>(Q.data(), K.data(), V.data(), O_cpu.data(),
                             B, H, N, D, /*is_causal=*/false);
    attention_online_ref_forward_host(Q.data(), K.data(), V.data(), O_gpu.data(),
                                      B, H, N, D, /*is_causal=*/false);

    for (float x : O_gpu) {
        ASSERT_TRUE(std::isfinite(x));
    }
    const float err = max_abs_error(O_gpu, O_cpu);
    EXPECT_LT(err, kAbsTolFp32) << "partial-tail max_abs_error=" << err;
}

// ── Edge case: one score dominates → O ≈ V[j*] for the peak key j*. ─────────
// Constructs Q, K such that Q · K[j*] is very large and Q · K[j != j*] is small.
// The recurrence must handle the mid-stream jump in the running max cleanly.
TEST(AttentionOnlineRefEdge, PeakKeyDominates) {
    constexpr int B = 1, H = 1, N = 128, D = 32;
    constexpr int j_star = 77;    // in the middle of block 1 (Bc=64)
    const std::size_t n = B * H * N * D;

    // K[j*] = e_0 * big_val; K[other] = small noise. Q = e_0 * moderate.
    std::vector<float> Q(n, 0.0f);
    std::vector<float> K(n, 0.0f);
    std::vector<float> V(n, 0.0f);

    // Q[0, 0, 0, :] = [1, 0, 0, ...]
    Q[bhnd_index(0, 0, 0, 0, 1, N, D)] = 1.0f;

    // K[0, 0, j*, :] = [100, 0, 0, ...]; every other K[j, :] left at zero.
    K[bhnd_index(0, 0, j_star, 0, 1, N, D)] = 100.0f;

    // V[0, 0, j*, :] = [1, 2, 3, ...] (distinguishable pattern).
    for (int d = 0; d < D; ++d) {
        V[bhnd_index(0, 0, j_star, d, 1, N, D)] = static_cast<float>(d + 1);
    }

    std::vector<float> O(n, 0.0f);
    attention_online_ref_forward_host(Q.data(), K.data(), V.data(), O.data(),
                                      B, H, N, D, /*is_causal=*/false);

    // For query row 0: softmax weight on j* is ~1, weights elsewhere ~0.
    // So O[0, :] ≈ V[j*, :] = [1, 2, ..., D].
    // Tolerance loosened here (1e-4) because the softmax denominator has (N-1)
    // small-exp terms that don't quite vanish in fp32.
    for (int d = 0; d < D; ++d) {
        const float expected = static_cast<float>(d + 1);
        const float got = O[bhnd_index(0, 0, 0, d, 1, N, D)];
        EXPECT_NEAR(got, expected, 1e-4f)
            << "peak-key: d=" << d << " got=" << got << " expected=" << expected;
    }
}

// ── Sanity: two independent runs on identical inputs are bit-identical. ─────
TEST(AttentionOnlineRefEdge, DeterministicAcrossRuns) {
    constexpr int B = 1, H = 1, N = 128, D = 32;
    const std::size_t n = B * H * N * D;
    const auto Q = gaussian_tensor(n, /*seed=*/8000);
    const auto K = gaussian_tensor(n, /*seed=*/8001);
    const auto V = gaussian_tensor(n, /*seed=*/8002);

    std::vector<float> O1(n, 0.0f), O2(n, 0.0f);
    attention_online_ref_forward_host(Q.data(), K.data(), V.data(), O1.data(),
                                      B, H, N, D, /*is_causal=*/false);
    attention_online_ref_forward_host(Q.data(), K.data(), V.data(), O2.data(),
                                      B, H, N, D, /*is_causal=*/false);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(O1[i], O2[i]) << "diverge at i=" << i;
    }
}

}  // namespace
