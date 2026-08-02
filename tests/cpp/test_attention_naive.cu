// GoogleTest for csrc/attention_naive.cu — the M2 naive CUDA baseline.
//
// The naive kernel is validated in two layers, following the same pattern as
// the M1 CPU-ref tests:
//
//   1) Parametrized correctness matrix — 16 cases covering the MILESTONES
//      verification grid (B=1, H=1, N in {64, 128, 512, 1024}, D in {32, 64},
//      causal in {false, true}), diffed against attention_cpu_ref<float>.
//
//   2) Edge-case tests — N=1 (O == V), causal-first-row (O[0] == V[0]),
//      uniform Q/K (O[i] == mean(V)). Same invariants exercised by the M1
//      cpu-ref suite; if the CUDA kernel breaks any of them the CPU ref
//      already gave us the known-good baseline.
//
// Tolerance: 5e-4 abs (per docs/AGENTS.md §9). The CPU ref runs in fp64
// internally and casts back to fp32, so all measured error is on the GPU side.

#include <cmath>
#include <cstdio>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "attention_cpu_ref.hpp"
#include "attention_naive.cuh"

namespace {

using flash_from_scratch::attention_cpu_ref;
using flash_from_scratch::attention_naive_forward_host;
using flash_from_scratch::bhnd_index;

// Fixed-seed Gaussian tensor generator (mirrors tests/cpp/test_attention_cpu_ref.cpp).
// std::mt19937 keeps this test independent of both PyTorch's Philox and the
// pytest util_tensors — different backends by design.
std::vector<float> gaussian_tensor(std::size_t n, std::uint64_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> out(n);
    for (auto& x : out) x = dist(gen);
    return out;
}

// Absolute tolerance for the CUDA-vs-CPU-ref diff, per docs/AGENTS.md §9.
constexpr float kAbsTolFp32 = 5e-4f;

// Compute max abs error between two flat buffers.
float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ── Parametrized correctness grid ────────────────────────────────────────────
// (N, D, is_causal) triples — matches MILESTONES §M2 verification plan.
class AttentionNaiveGrid
    : public ::testing::TestWithParam<std::tuple<int, int, bool>> {};

TEST_P(AttentionNaiveGrid, MatchesCpuRefWithinTolerance) {
    // Note: std::tie/std::get instead of C++17 structured bindings to keep
    // this file portable across nvcc's C++14 host-compile fallback path.
    const auto& p = GetParam();
    const int  N         = std::get<0>(p);
    const int  D         = std::get<1>(p);
    const bool is_causal = std::get<2>(p);
    constexpr int B = 1;
    constexpr int H = 1;
    const std::size_t nelems = static_cast<std::size_t>(B) * H * N * D;

    const auto Q = gaussian_tensor(nelems, /*seed=*/1000 + N + D);
    const auto K = gaussian_tensor(nelems, /*seed=*/2000 + N + D);
    const auto V = gaussian_tensor(nelems, /*seed=*/3000 + N + D);

    std::vector<float> O_cpu(nelems, 0.0f);
    std::vector<float> O_gpu(nelems, 0.0f);

    attention_cpu_ref<float>(Q.data(), K.data(), V.data(), O_cpu.data(),
                             B, H, N, D, is_causal);
    attention_naive_forward_host(Q.data(), K.data(), V.data(), O_gpu.data(),
                                 B, H, N, D, is_causal);

    // Every output must be finite (catches NaN/Inf from a softmax stability bug).
    for (float x : O_gpu) {
        ASSERT_TRUE(std::isfinite(x))
            << "non-finite entry in O_gpu at N=" << N << " D=" << D
            << " causal=" << is_causal;
    }

    const float err = max_abs_error(O_gpu, O_cpu);
    EXPECT_LT(err, kAbsTolFp32)
        << "N=" << N << " D=" << D << " causal=" << is_causal
        << " · max_abs_error=" << err;
}

INSTANTIATE_TEST_SUITE_P(
    M2Grid, AttentionNaiveGrid,
    ::testing::Combine(
        ::testing::Values(64, 128, 512, 1024),   // N
        ::testing::Values(32, 64),               // D
        ::testing::Values(false, true)),         // causal
    [](const ::testing::TestParamInfo<AttentionNaiveGrid::ParamType>& info) {
        const int  N         = std::get<0>(info.param);
        const int  D         = std::get<1>(info.param);
        const bool is_causal = std::get<2>(info.param);
        std::string s = "N" + std::to_string(N) + "_D" + std::to_string(D)
                      + (is_causal ? "_causal" : "_noncausal");
        return s;
    });

// ── Edge case: N=1 → softmax over a single element = [1.0], so O == V ───────
TEST(AttentionNaiveEdge, SinglePositionOutputEqualsValue) {
    constexpr int B = 1, H = 1, N = 1, D = 8;
    const auto Q = gaussian_tensor(B * H * N * D, /*seed=*/5000);
    const auto K = gaussian_tensor(B * H * N * D, /*seed=*/5001);
    const auto V = gaussian_tensor(B * H * N * D, /*seed=*/5002);
    std::vector<float> O(B * H * N * D, 0.0f);

    attention_naive_forward_host(Q.data(), K.data(), V.data(), O.data(),
                                 B, H, N, D, /*is_causal=*/false);
    for (int d = 0; d < D; ++d) {
        EXPECT_NEAR(O[d], V[d], kAbsTolFp32) << "d=" << d;
    }
}

// ── Edge case: causal at i=0 → only j=0 is visible → O[0, :] == V[0, :] ──────
TEST(AttentionNaiveEdge, CausalFirstRowEqualsFirstValue) {
    constexpr int B = 1, H = 1, N = 8, D = 4;
    const auto Q = gaussian_tensor(B * H * N * D, /*seed=*/6000);
    const auto K = gaussian_tensor(B * H * N * D, /*seed=*/6001);
    const auto V = gaussian_tensor(B * H * N * D, /*seed=*/6002);
    std::vector<float> O(B * H * N * D, 0.0f);

    attention_naive_forward_host(Q.data(), K.data(), V.data(), O.data(),
                                 B, H, N, D, /*is_causal=*/true);

    for (int d = 0; d < D; ++d) {
        const auto o = O[bhnd_index(0, 0, 0, d, 1, N, D)];
        const auto v = V[bhnd_index(0, 0, 0, d, 1, N, D)];
        EXPECT_NEAR(o, v, kAbsTolFp32) << "d=" << d;
    }
}

// ── Edge case: uniform Q, uniform K → uniform attention → O[i] == mean(V) ────
TEST(AttentionNaiveEdge, UniformQKGivesMeanOfV) {
    constexpr int B = 1, H = 1, N = 8, D = 4;
    std::vector<float> Q(B * H * N * D, 0.3f);  // all rows identical
    std::vector<float> K(B * H * N * D, 0.7f);  // all rows identical
    const auto V = gaussian_tensor(B * H * N * D, /*seed=*/7000);
    std::vector<float> O(B * H * N * D, 0.0f);

    attention_naive_forward_host(Q.data(), K.data(), V.data(), O.data(),
                                 B, H, N, D, /*is_causal=*/false);

    // Expected: mean over j of V[0, 0, j, d], per d.
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

// ── Sanity: two independent runs on the same inputs are bit-identical. ───────
// The naive kernel is deterministic (no atomics, no reduction reordering by
// scheduler), so this should hold — flags any accidental nondeterminism early.
TEST(AttentionNaiveEdge, DeterministicAcrossRuns) {
    constexpr int B = 1, H = 1, N = 64, D = 32;
    const std::size_t n = B * H * N * D;
    const auto Q = gaussian_tensor(n, /*seed=*/8000);
    const auto K = gaussian_tensor(n, /*seed=*/8001);
    const auto V = gaussian_tensor(n, /*seed=*/8002);

    std::vector<float> O1(n, 0.0f), O2(n, 0.0f);
    attention_naive_forward_host(Q.data(), K.data(), V.data(), O1.data(),
                                 B, H, N, D, /*is_causal=*/false);
    attention_naive_forward_host(Q.data(), K.data(), V.data(), O2.data(),
                                 B, H, N, D, /*is_causal=*/false);
    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_EQ(O1[i], O2[i]) << "diverge at i=" << i;
    }
}

}  // namespace
