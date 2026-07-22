// GoogleTest for csrc/attention_cpu_ref.hpp.
//
// The C++ reference is validated at two layers:
//   1) Direct unit tests here — catch C++-specific bugs (index arithmetic,
//      template instantiation, pointer aliasing, etc).
//   2) Transitively via the NumPy mirror + PyTorch SDPA in
//      `tests/test_reference_matches_torch.py`.
//
// Both mirror the same algorithm on the same shape grid; if the C++ header
// drifts from the NumPy ref, one of these fails first.

#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "attention_cpu_ref.hpp"

namespace {

using flash_from_scratch::attention_cpu_ref;
using flash_from_scratch::bhnd_index;

// Reproducible RNG mirroring the M1 pytest grid's philosophy: fixed seed,
// unit-variance Gaussians. Different backend than torch.Generator (std::mt19937
// vs Philox), so numerical values won't match tests/util_tensors.py — but that's
// fine: this test only checks internal invariants (softmax normalization,
// causal masking, boundary cases), not cross-implementation equivalence.
std::vector<float> gaussian_tensor(std::size_t n, std::uint64_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> out(n);
    for (auto& x : out) x = dist(gen);
    return out;
}

// Assert every row of P (implicit, via O = P @ V) has a valid softmax normalization.
// We verify by checking O = P @ V with P having row-sum 1 non-negative entries —
// which requires reconstructing P. Rather than that, we test the invariants
// that are equivalent and easier: `causal` zeros future positions; a uniform-Q
// case gives uniform attention; a single-position N=1 case gives O == V.

// ── Test: shapes are honored, small case runs ────────────────────────────────
TEST(AttentionCpuRef, RunsOnSmallShapeWithoutThrowing) {
    constexpr int B = 1, H = 1, N = 8, D = 4;
    const auto Q = gaussian_tensor(B * H * N * D, /*seed=*/1);
    const auto K = gaussian_tensor(B * H * N * D, /*seed=*/2);
    const auto V = gaussian_tensor(B * H * N * D, /*seed=*/3);
    std::vector<float> O(B * H * N * D, 0.0f);

    ASSERT_NO_THROW(attention_cpu_ref<float>(
        Q.data(), K.data(), V.data(), O.data(), B, H, N, D, /*is_causal=*/false));

    // Every output must be finite.
    for (float x : O) {
        EXPECT_TRUE(std::isfinite(x)) << "non-finite entry in O";
    }
}

// ── Test: N=1 edge case, O = V (softmax over 1 element = [1.0]) ─────────────
TEST(AttentionCpuRef, SinglePositionOutputEqualsValue) {
    constexpr int B = 1, H = 1, N = 1, D = 4;
    const std::vector<float> Q = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::vector<float> K = {0.5f, 0.5f, 0.5f, 0.5f};
    const std::vector<float> V = {10.0f, 20.0f, 30.0f, 40.0f};
    std::vector<float> O(D, 0.0f);

    attention_cpu_ref<float>(Q.data(), K.data(), V.data(), O.data(),
                             B, H, N, D, /*is_causal=*/false);
    for (int d = 0; d < D; ++d) {
        EXPECT_FLOAT_EQ(O[d], V[d]) << "d=" << d;
    }
}

// ── Test: causal mask makes query-0 output exactly V[0] ─────────────────────
// With is_causal=true, position i=0 only sees j=0 (softmax [1, 0, 0, ...]), so
// O[0] == V[0]. Same numerical property as the pytest sanity check.
TEST(AttentionCpuRef, CausalFirstRowEqualsFirstValue) {
    constexpr int B = 1, H = 1, N = 4, D = 3;
    const auto Q = gaussian_tensor(B * H * N * D, /*seed=*/10);
    const auto K = gaussian_tensor(B * H * N * D, /*seed=*/11);
    const auto V = gaussian_tensor(B * H * N * D, /*seed=*/12);
    std::vector<float> O(B * H * N * D, 0.0f);

    attention_cpu_ref<float>(Q.data(), K.data(), V.data(), O.data(),
                             B, H, N, D, /*is_causal=*/true);
    for (int d = 0; d < D; ++d) {
        // O[b=0, h=0, i=0, d] == V[b=0, h=0, j=0, d]
        const auto o = O[bhnd_index(0, 0, 0, d, 1, N, D)];
        const auto v = V[bhnd_index(0, 0, 0, d, 1, N, D)];
        EXPECT_FLOAT_EQ(o, v) << "d=" << d;
    }
}

// ── Test: uniform-Q, uniform-K gives uniform attention → O[i] == mean(V) ────
// When every Q row equals every K row (constant), all S entries are equal, so
// softmax gives uniform 1/N weights and O[i, :] is the mean of V over N.
TEST(AttentionCpuRef, UniformQKGivesMeanOfV) {
    constexpr int B = 1, H = 1, N = 5, D = 2;
    std::vector<float> Q(B * H * N * D, 0.3f);  // all rows identical
    std::vector<float> K(B * H * N * D, 0.7f);  // all rows identical
    const auto V = gaussian_tensor(B * H * N * D, /*seed=*/42);
    std::vector<float> O(B * H * N * D, 0.0f);

    attention_cpu_ref<float>(Q.data(), K.data(), V.data(), O.data(),
                             B, H, N, D, /*is_causal=*/false);

    // Compute expected mean(V) over the N axis, per d.
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
            EXPECT_NEAR(o, mean_v[d], 1e-5f) << "i=" << i << " d=" << d;
        }
    }
}

// ── Test: double and float instantiations both compile & run ────────────────
TEST(AttentionCpuRef, WorksForDoubleAndFloat) {
    constexpr int B = 1, H = 2, N = 4, D = 4;
    const auto Qf = gaussian_tensor(B * H * N * D, /*seed=*/100);
    const auto Kf = gaussian_tensor(B * H * N * D, /*seed=*/101);
    const auto Vf = gaussian_tensor(B * H * N * D, /*seed=*/102);
    std::vector<double> Qd(Qf.begin(), Qf.end());
    std::vector<double> Kd(Kf.begin(), Kf.end());
    std::vector<double> Vd(Vf.begin(), Vf.end());

    std::vector<float> Of(B * H * N * D, 0.0f);
    std::vector<double> Od(B * H * N * D, 0.0);

    attention_cpu_ref<float>(Qf.data(), Kf.data(), Vf.data(), Of.data(),
                             B, H, N, D, false);
    attention_cpu_ref<double>(Qd.data(), Kd.data(), Vd.data(), Od.data(),
                              B, H, N, D, false);

    // fp32 output should be within a few ULPs of fp64 output cast down.
    for (std::size_t i = 0; i < Of.size(); ++i) {
        EXPECT_NEAR(Of[i], static_cast<float>(Od[i]), 1e-5f) << "i=" << i;
    }
}

// ── Test: invalid dims throw ─────────────────────────────────────────────────
TEST(AttentionCpuRef, ThrowsOnInvalidDims) {
    float dummy = 0.0f;
    EXPECT_THROW(attention_cpu_ref<float>(&dummy, &dummy, &dummy, &dummy,
                                          0, 1, 1, 1, false),
                 std::invalid_argument);
    EXPECT_THROW(attention_cpu_ref<float>(&dummy, &dummy, &dummy, &dummy,
                                          1, 1, 1, -1, false),
                 std::invalid_argument);
}

// ── Test: null pointers throw ────────────────────────────────────────────────
TEST(AttentionCpuRef, ThrowsOnNullPointers) {
    float dummy = 0.0f;
    EXPECT_THROW(attention_cpu_ref<float>(nullptr, &dummy, &dummy, &dummy,
                                          1, 1, 1, 1, false),
                 std::invalid_argument);
}

}  // namespace
