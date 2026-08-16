// bindings.cpp — M5 pybind11 shim exposing every CUDA kernel to Python.
//
// One entry point per variant, all sharing the same signature:
//   Tensor <variant>_forward(Tensor Q, Tensor K, Tensor V, bool is_causal)
// returning a freshly allocated output tensor with the same shape/dtype as Q.
//
// The Python-side wrapper in `flash_from_scratch/__init__.py` performs
// user-facing validation with clean error messages; the TORCH_CHECK calls here
// are defense-in-depth and catch anything a downstream C++ caller might miss.
//
// Design notes:
//  - <torch/extension.h> gives us `torch::Tensor`, TORCH_CHECK, and the
//    pybind11 caster for Tensor in one include. We keep the raw C++ kernel
//    APIs (float*-based) untouched so the M2/M3/M4 GoogleTest binaries still
//    compile against the same static libs.
//  - Allocation of `O` happens here via `torch::empty_like(Q)`. The kernels
//    themselves never allocate output HBM — they only allocate transient S/P
//    (naive) or nothing (online_ref / flash_v1).
//  - The kernels use the current CUDA device / stream implicitly (device 0,
//    legacy default stream). PyTorch's stream tracking is not plumbed through
//    in v1; M9 or M10 can add it if benchmarks require concurrent streams.
//
// See `theory/M5.md` §6 for the design rationale, `docs/AGENTS.md` §5.1 for
// the [B, H, N, D] layout contract, and each kernel's .cuh for its per-variant
// constraints (D range, causal support, etc.).

#include <torch/extension.h>

#include "attention_naive.cuh"
#include "attention_online_ref.cuh"
#include "flash_fwd_v1.cuh"
#include "flash_fwd_v2_shared_kv.cuh"

namespace flash_from_scratch {

namespace {

// Common precondition checks. Raises Python-visible errors via TORCH_CHECK
// (which throws c10::Error, translated to RuntimeError by pybind11).
void check_qkv_contract(const torch::Tensor& Q, const torch::Tensor& K,
                        const torch::Tensor& V, const char* variant_name) {
    TORCH_CHECK(Q.is_cuda(),        variant_name, ": Q must be on a CUDA device");
    TORCH_CHECK(K.is_cuda(),        variant_name, ": K must be on a CUDA device");
    TORCH_CHECK(V.is_cuda(),        variant_name, ": V must be on a CUDA device");
    TORCH_CHECK(Q.device() == K.device() && Q.device() == V.device(),
                variant_name, ": Q, K, V must be on the same CUDA device");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32,
                variant_name, ": only fp32 is supported in M5 (got ", Q.scalar_type(), ")");
    TORCH_CHECK(K.scalar_type() == torch::kFloat32 && V.scalar_type() == torch::kFloat32,
                variant_name, ": K and V must also be fp32");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                variant_name, ": Q, K, V must be 4-D [B, H, N, D] tensors");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                variant_name, ": Q, K, V must all share the same shape");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                variant_name, ": Q, K, V must be contiguous in [B, H, N, D] layout");
}

// Extract (B, H, N, D) from Q; kernels take them as ints.
struct Dims {
    int B, H, N, D;
};

Dims dims_from(const torch::Tensor& Q) {
    return Dims{static_cast<int>(Q.size(0)), static_cast<int>(Q.size(1)),
                static_cast<int>(Q.size(2)), static_cast<int>(Q.size(3))};
}

torch::Tensor attention_naive_forward_py(torch::Tensor Q, torch::Tensor K,
                                         torch::Tensor V, bool is_causal) {
    check_qkv_contract(Q, K, V, "attention_naive_forward");
    const auto d = dims_from(Q);
    TORCH_CHECK(d.D == 32 || d.D == 64,
                "attention_naive_forward: D must be 32 or 64 in M5 (got ", d.D, ")");
    auto O = torch::empty_like(Q);
    attention_naive_forward(Q.data_ptr<float>(), K.data_ptr<float>(),
                            V.data_ptr<float>(), O.data_ptr<float>(),
                            d.B, d.H, d.N, d.D, is_causal, /*stream=*/0);
    return O;
}

torch::Tensor attention_online_ref_forward_py(torch::Tensor Q, torch::Tensor K,
                                              torch::Tensor V, bool is_causal) {
    check_qkv_contract(Q, K, V, "attention_online_ref_forward");
    const auto d = dims_from(Q);
    TORCH_CHECK(d.D <= kOnlineRefBc,
                "attention_online_ref_forward: D must be <= ", kOnlineRefBc,
                " (got ", d.D, ")");
    TORCH_CHECK(d.D == 32 || d.D == 64,
                "attention_online_ref_forward: D must be 32 or 64 in M5 (got ", d.D, ")");
    auto O = torch::empty_like(Q);
    attention_online_ref_forward(Q.data_ptr<float>(), K.data_ptr<float>(),
                                 V.data_ptr<float>(), O.data_ptr<float>(),
                                 d.B, d.H, d.N, d.D, is_causal, /*stream=*/0);
    return O;
}

torch::Tensor flash_fwd_v1_forward_py(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V, bool is_causal) {
    check_qkv_contract(Q, K, V, "flash_fwd_v1_forward");
    const auto d = dims_from(Q);
    TORCH_CHECK(d.D == 32 || d.D == 64,
                "flash_fwd_v1_forward: D must be 32 or 64 in v1 (got ", d.D, ")");
    auto O = torch::empty_like(Q);
    flash_fwd_v1_forward(Q.data_ptr<float>(), K.data_ptr<float>(),
                         V.data_ptr<float>(), O.data_ptr<float>(),
                         d.B, d.H, d.N, d.D, is_causal, /*stream=*/0);
    return O;
}

torch::Tensor flash_fwd_v2_forward_py(torch::Tensor Q, torch::Tensor K,
                                      torch::Tensor V, bool is_causal) {
    check_qkv_contract(Q, K, V, "flash_fwd_v2_forward");
    const auto d = dims_from(Q);
    TORCH_CHECK(d.D == 32 || d.D == 64,
                "flash_fwd_v2_forward: D must be 32 or 64 in v2 (got ", d.D, ")");
    auto O = torch::empty_like(Q);
    flash_fwd_v2_forward(Q.data_ptr<float>(), K.data_ptr<float>(),
                         V.data_ptr<float>(), O.data_ptr<float>(),
                         d.B, d.H, d.N, d.D, is_causal, /*stream=*/0);
    return O;
}

}  // namespace

}  // namespace flash_from_scratch

PYBIND11_MODULE(_C, m) {
    m.doc() = "flash_from_scratch C++/CUDA kernel bindings (M5).";

    m.def("attention_naive_forward",
          &flash_from_scratch::attention_naive_forward_py,
          "M2 three-kernel naive baseline (materializes S and P in HBM).",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("is_causal") = false);

    m.def("attention_online_ref_forward",
          &flash_from_scratch::attention_online_ref_forward_py,
          "M3 online-softmax reference (one CTA per query row; S/P never touch HBM).",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("is_causal") = false);

    m.def("flash_fwd_v1_forward",
          &flash_from_scratch::flash_fwd_v1_forward_py,
          "M4 FlashAttention v1 tiled forward (one CTA per Q-tile of Br rows).",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("is_causal") = false);

    m.def("flash_fwd_v2_forward",
          &flash_from_scratch::flash_fwd_v2_forward_py,
          "M6 Flash v2 tiled forward (same algebra as v1; padded sK, "
          "register-resident O-accumulator, 512-thread CTAs, float4 loads).",
          pybind11::arg("Q"), pybind11::arg("K"), pybind11::arg("V"),
          pybind11::arg("is_causal") = false);

    m.attr("kFlashV1Br") = flash_from_scratch::kFlashV1Br;
    m.attr("kFlashV1Bc") = flash_from_scratch::kFlashV1Bc;
    m.attr("kFlashV2Br") = flash_from_scratch::kFlashV2Br;
    m.attr("kFlashV2Bc") = flash_from_scratch::kFlashV2Bc;
    m.attr("kFlashV2RowsPerThread") = flash_from_scratch::kFlashV2RowsPerThread;
    m.attr("kOnlineRefBc") = flash_from_scratch::kOnlineRefBc;
}
