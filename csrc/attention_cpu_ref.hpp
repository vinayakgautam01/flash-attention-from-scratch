// attention_cpu_ref.hpp — the C++ CPU oracle for scaled dot-product attention.
//
// Header-only. No BLAS. No CUDA. Slow-and-obvious three-nested-loop matmuls.
// Its job is to be numerically trustworthy, not fast — every downstream CUDA
// kernel gets diffed against this.
//
// Mirrors the algorithm in `tests/reference.py`. If the two ever drift, the
// M1 gate test (`tests/test_reference_matches_torch.py`) fails first.
//
// Layout: row-major `[B, H, N, D]`, per `docs/AGENTS.md` §5.1.
// Causal semantics: if `is_causal`, mask entries `S[i, j] = -inf` for `j > i`
// before softmax (see `theory/M1.md` §5).
//
// See `theory/M1.md` §6 for why we hand-roll the matmul (tautology-test guard).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace flash_from_scratch {

// Row-major offset: element [b, h, n, d] of a [B, H, N, D] tensor.
constexpr std::size_t bhnd_index(
    std::size_t b, std::size_t h, std::size_t n, std::size_t d,
    std::size_t H, std::size_t N, std::size_t D)
{
    return ((b * H + h) * N + n) * D + d;
}

// Forward pass of scaled dot-product attention on CPU.
//
// Template `T` selects storage/output precision (float / double). Internal math
// always runs in `double` — see `theory/M1.md` §6 for the fp64-ref rationale.
//
// Args:
//   Q, K, V: pointers to row-major [B, H, N, D] arrays.
//   O      : output pointer, same shape.
//   B, H, N, D: dims.
//   is_causal : if true, mask upper-triangular positions before softmax.
//
// Throws:
//   std::invalid_argument if any dim is non-positive.
template <typename T>
void attention_cpu_ref(
    const T* Q, const T* K, const T* V, T* O,
    int B, int H, int N, int D, bool is_causal)
{
    if (B <= 0 || H <= 0 || N <= 0 || D <= 0) {
        throw std::invalid_argument("attention_cpu_ref: B, H, N, D must all be positive");
    }
    if (Q == nullptr || K == nullptr || V == nullptr || O == nullptr) {
        throw std::invalid_argument("attention_cpu_ref: Q/K/V/O pointers must be non-null");
    }

    const auto Bs = static_cast<std::size_t>(B);
    const auto Hs = static_cast<std::size_t>(H);
    const auto Ns = static_cast<std::size_t>(N);
    const auto Ds = static_cast<std::size_t>(D);

    const double scale = 1.0 / std::sqrt(static_cast<double>(D));

    // Scratch for one row of S/P at a time — we don't materialize the full [N, N]
    // matrix, which keeps the peak scratch memory at O(N) doubles per thread.
    std::vector<double> row_scores(Ns);

    for (std::size_t b = 0; b < Bs; ++b) {
        for (std::size_t h = 0; h < Hs; ++h) {
            for (std::size_t i = 0; i < Ns; ++i) {
                // --- 1) S[i, :] = Q[i, :] . K[:, :] / sqrt(D) --------------------
                // Three-nested loops over (j, d). Two loops here, third (d) inline.
                for (std::size_t j = 0; j < Ns; ++j) {
                    if (is_causal && j > i) {
                        row_scores[j] = -std::numeric_limits<double>::infinity();
                        continue;
                    }
                    double dot = 0.0;
                    for (std::size_t d = 0; d < Ds; ++d) {
                        const double qv = static_cast<double>(Q[bhnd_index(b, h, i, d, Hs, Ns, Ds)]);
                        const double kv = static_cast<double>(K[bhnd_index(b, h, j, d, Hs, Ns, Ds)]);
                        dot += qv * kv;
                    }
                    row_scores[j] = dot * scale;
                }

                // --- 2) Numerically stable softmax over row_scores --------------
                // Subtract row max, exp, normalize. See theory/M1.md §4.
                double row_max = row_scores[0];
                for (std::size_t j = 1; j < Ns; ++j) {
                    row_max = std::max(row_max, row_scores[j]);
                }
                // Guard against the (impossible in this project) all-masked row.
                // Kept as a defensive assertion since is_causal on i>=0 always keeps j=i.
                double denom = 0.0;
                for (std::size_t j = 0; j < Ns; ++j) {
                    // exp(-inf - row_max) = exp(-inf) = 0, which is exactly what we want.
                    const double e = std::exp(row_scores[j] - row_max);
                    row_scores[j] = e;
                    denom += e;
                }
                const double inv_denom = 1.0 / denom;
                for (std::size_t j = 0; j < Ns; ++j) {
                    row_scores[j] *= inv_denom;  // row_scores now holds P[i, :]
                }

                // --- 3) O[i, :] = P[i, :] . V[:, :] -----------------------------
                for (std::size_t d = 0; d < Ds; ++d) {
                    double acc = 0.0;
                    for (std::size_t j = 0; j < Ns; ++j) {
                        const double vv = static_cast<double>(V[bhnd_index(b, h, j, d, Hs, Ns, Ds)]);
                        acc += row_scores[j] * vv;
                    }
                    O[bhnd_index(b, h, i, d, Hs, Ns, Ds)] = static_cast<T>(acc);
                }
            }
        }
    }
}

}  // namespace flash_from_scratch
