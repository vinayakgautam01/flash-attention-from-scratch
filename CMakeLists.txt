cmake_minimum_required(VERSION 3.24)
project(flash_from_scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA is optional so pure-C++ targets (attention_cpu_ref + its GoogleTest) build on
# Mac-local, while CUDA targets (`hello`, and M2+ kernels) only build where a CUDA
# toolkit is available (Colab T4, Modal A10G, GPU CI).
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING
            "CUDA architectures to compile for; override at configure time")
    endif()
    set(HAVE_CUDA TRUE)
else()
    set(HAVE_CUDA FALSE)
    message(STATUS "CUDA toolkit not found; skipping CUDA targets (pure-C++ builds only).")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

# M0 build-sanity target: prints GPU info + tiny launch. CUDA-only.
if(HAVE_CUDA)
    add_executable(hello csrc/hello.cu)
    target_compile_options(hello PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    )
endif()

# Torch linking is optional and off by default. Flip on once M10 (PyTorch extension) starts.
option(USE_TORCH "Link against libtorch for the PyTorch extension" OFF)
if(USE_TORCH)
    find_package(Torch REQUIRED)
    message(STATUS "Torch: ${TORCH_LIBRARIES}")
endif()

# M1: expose the CPU reference as an INTERFACE target so tests (and, later, CUDA
# kernels) can `#include "attention_cpu_ref.hpp"` without repeating include paths.
add_library(attention_cpu_ref INTERFACE)
target_include_directories(attention_cpu_ref INTERFACE ${CMAKE_SOURCE_DIR}/csrc)

# M2: naive multi-kernel attention baseline. Exposed as a static lib so both
# the GoogleTest binary and (later, in M4/M5) the pybind11 shim can link it.
# Declared after attention_cpu_ref so the target reference below resolves against
# an already-defined target (avoids relying on generate-time forward resolution).
if(HAVE_CUDA)
    add_library(attention_naive STATIC csrc/attention_naive.cu)
    target_include_directories(attention_naive PUBLIC ${CMAKE_SOURCE_DIR}/csrc)
    target_link_libraries(attention_naive PUBLIC attention_cpu_ref)
    set_target_properties(attention_naive PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    # Force C++17 on both host and device compile lines. GoogleTest 1.15+ emits
    # C++17-only attributes (e.g. [[maybe_unused]]) inside its parametrized-test
    # macros, and CMAKE_CUDA_STANDARD alone doesn't always propagate through
    # nvcc with CUDA_SEPARABLE_COMPILATION on some driver/toolkit combos.
    # Declaring the requirement via target_compile_features makes CMake fail at
    # configure time if the standard isn't available and guarantees -std=c++17.
    target_compile_features(attention_naive PUBLIC cxx_std_17 cuda_std_17)
    target_compile_options(attention_naive PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    )

    # M3: online-softmax reference kernel. Standalone artifact proving the
    # streaming (m, ℓ, Õ) recurrence works before M4 adds Q-tiling. Same
    # target shape as attention_naive (static lib, C++17, ptxas -v).
    add_library(attention_online_ref STATIC csrc/attention_online_ref.cu)
    target_include_directories(attention_online_ref PUBLIC ${CMAKE_SOURCE_DIR}/csrc)
    target_link_libraries(attention_online_ref PUBLIC attention_cpu_ref)
    set_target_properties(attention_online_ref PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(attention_online_ref PUBLIC cxx_std_17 cuda_std_17)
    target_compile_options(attention_online_ref PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    )

    # M4: FlashAttention v1 forward kernel. Tiles Q into Br-row blocks and
    # streams K/V in Bc-column tiles, applying M3's online-softmax recurrence
    # for Br rows in lock-step so S and P never touch HBM. Same target shape
    # as attention_online_ref (static lib, C++17, ptxas -v).
    add_library(flash_fwd_v1 STATIC csrc/flash_fwd_v1.cu)
    target_include_directories(flash_fwd_v1 PUBLIC ${CMAKE_SOURCE_DIR}/csrc)
    target_link_libraries(flash_fwd_v1 PUBLIC attention_cpu_ref)
    set_target_properties(flash_fwd_v1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(flash_fwd_v1 PUBLIC cxx_std_17 cuda_std_17)
    target_compile_options(flash_fwd_v1 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    )

    # M6: "Flash v2" — same algebra as v1, retuned for the memory hierarchy
    # (padded sK, register-resident Õ, 512-thread CTAs, float4 loads). Same
    # target shape as flash_fwd_v1. The `--ptxas-options=-v` flag below is not
    # incidental here: its output is a checked-in M6 deliverable
    # (docs/ptxas_v1_vs_v2.md), so keep it on.
    add_library(flash_fwd_v2 STATIC csrc/flash_fwd_v2_shared_kv.cu)
    target_include_directories(flash_fwd_v2 PUBLIC ${CMAKE_SOURCE_DIR}/csrc)
    target_link_libraries(flash_fwd_v2 PUBLIC attention_cpu_ref)
    set_target_properties(flash_fwd_v2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_features(flash_fwd_v2 PUBLIC cxx_std_17 cuda_std_17)
    target_compile_options(flash_fwd_v2 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    )
endif()

# M1: GoogleTest for pure-C++ host-side unit tests. Decision recorded in
# docs/AGENTS.md §8. On by default, controllable via BUILD_TESTING.
option(BUILD_TESTING "Build C++ unit tests" ON)
if(BUILD_TESTING)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.2
    )
    # Prevent GoogleTest from overriding our compiler/linker options on Windows.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)

    add_executable(test_attention_cpu_ref
        tests/cpp/test_attention_cpu_ref.cpp
    )
    target_link_libraries(test_attention_cpu_ref PRIVATE
        attention_cpu_ref
        GTest::gtest_main
    )
    gtest_discover_tests(test_attention_cpu_ref)

    # M2: naive CUDA kernel correctness — diffed against the M1 CPU reference.
    # Only built when CUDA is available (the kernel needs nvcc).
    if(HAVE_CUDA)
        add_executable(test_attention_naive
            tests/cpp/test_attention_naive.cu
        )
        target_link_libraries(test_attention_naive PRIVATE
            attention_naive
            attention_cpu_ref
            GTest::gtest_main
        )
        set_target_properties(test_attention_naive PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        # See attention_naive block above for the rationale. GoogleTest's
        # INSTANTIATE_TEST_SUITE_P macro requires C++17 in its expansion.
        target_compile_features(test_attention_naive PRIVATE cxx_std_17 cuda_std_17)
        gtest_discover_tests(test_attention_naive)

        # M3: online-softmax reference correctness — same pattern.
        add_executable(test_attention_online_ref
            tests/cpp/test_attention_online_ref.cu
        )
        target_link_libraries(test_attention_online_ref PRIVATE
            attention_online_ref
            attention_cpu_ref
            GTest::gtest_main
        )
        set_target_properties(test_attention_online_ref PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        target_compile_features(test_attention_online_ref PRIVATE cxx_std_17 cuda_std_17)
        gtest_discover_tests(test_attention_online_ref)

        # M4: FlashAttention v1 forward correctness — same pattern.
        add_executable(test_flash_fwd_v1
            tests/cpp/test_flash_fwd_v1.cu
        )
        target_link_libraries(test_flash_fwd_v1 PRIVATE
            flash_fwd_v1
            attention_cpu_ref
            GTest::gtest_main
        )
        set_target_properties(test_flash_fwd_v1 PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        target_compile_features(test_flash_fwd_v1 PRIVATE cxx_std_17 cuda_std_17)
        gtest_discover_tests(test_flash_fwd_v1)

        # M6: Flash v2 correctness. Links BOTH kernels because the suite
        # includes a direct v1-vs-v2 parity check (MILESTONES §M6: "correctness
        # parity with v1"), not just a diff against the CPU reference.
        add_executable(test_flash_fwd_v2
            tests/cpp/test_flash_fwd_v2.cu
        )
        target_link_libraries(test_flash_fwd_v2 PRIVATE
            flash_fwd_v2
            flash_fwd_v1
            attention_cpu_ref
            GTest::gtest_main
        )
        set_target_properties(test_flash_fwd_v2 PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )
        target_compile_features(test_flash_fwd_v2 PRIVATE cxx_std_17 cuda_std_17)
        gtest_discover_tests(test_flash_fwd_v2)
    endif()
endif()

# M5: pybind11 shim exposing every CUDA kernel to Python as
# `flash_from_scratch._C`. Gated on BUILD_PY_EXT so the default
# `scripts/build.sh` (which builds `hello` + GoogleTest) is unaffected.
# Enabled automatically by scikit-build-core via pyproject.toml.
# Requires CUDA (the shim links against the CUDA static libs) AND libtorch
# (the shim takes `torch::Tensor`). Auto-locates Torch via the running
# Python's `torch.utils.cmake_prefix_path`.
option(BUILD_PY_EXT "Build the flash_from_scratch._C pybind11 extension (M5)" OFF)
if(BUILD_PY_EXT)
    if(NOT HAVE_CUDA)
        message(FATAL_ERROR "BUILD_PY_EXT=ON requires CUDA; skip the Python "
                            "extension on hosts without a CUDA toolkit.")
    endif()

    # Locate Python (used both for Torch prefix resolution and pybind11 discovery).
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

    # Torch: resolve CMake prefix from the active Python. This is the standard
    # scikit-build-core / setuptools-cpp-extension recipe and avoids hard-coding
    # a libtorch path across Colab / Modal / local venvs.
    if(NOT Torch_DIR)
        execute_process(
            COMMAND ${Python_EXECUTABLE} -c
                "import torch, os; print(os.path.join(torch.utils.cmake_prefix_path, 'Torch'))"
            OUTPUT_VARIABLE Torch_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _torch_dir_rc
        )
        if(NOT _torch_dir_rc EQUAL 0)
            message(FATAL_ERROR "BUILD_PY_EXT=ON: failed to locate Torch's "
                                "CMake config via Python; is `torch` installed "
                                "in the environment used to configure CMake?")
        endif()
    endif()
    find_package(Torch REQUIRED CONFIG)

    # pybind11: prefer the copy bundled with `pip install pybind11` (found via
    # its own CMake helper). scikit-build-core supplies it in the isolated
    # build env; local devs get it from `pip install pybind11`.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pb11_rc
    )
    if(NOT _pb11_rc EQUAL 0)
        message(FATAL_ERROR "BUILD_PY_EXT=ON: pybind11 not found; "
                            "install with `pip install pybind11`.")
    endif()
    find_package(pybind11 REQUIRED CONFIG)

    # <torch/extension.h> needs the pybind11-Tensor caster which lives in
    # libtorch_python.so — that library is NOT included in ${TORCH_LIBRARIES}
    # and must be linked explicitly, otherwise `import _C` fails at load time
    # with an undefined-symbol error on `THPVariable_Wrap` (or similar).
    find_library(TORCH_PYTHON_LIBRARY
                 NAMES torch_python
                 PATHS "${TORCH_INSTALL_PREFIX}/lib"
                 REQUIRED)

    # Note: bindings source file is `.cu` (not `.cpp`) on purpose. The file
    # itself is pure C++ (pybind11 macros + torch::Tensor + TORCH_CHECK), but
    # naming it `.cu` makes CMake treat the `_C` target as a CUDA-language
    # target, which is what triggers the automatic device-link step against
    # our separably-compiled CUDA static libs (attention_naive et al.).
    # Alternatives (e.g. LINKER_LANGUAGE=CUDA on a .cpp target) fail to emit
    # the fatbin data into a pybind11 MODULE library, producing a `_C.so`
    # that loads with `undefined symbol: fatbinData` at import time.
    pybind11_add_module(_C MODULE csrc/bindings.cu)
    target_link_libraries(_C PRIVATE
        attention_naive
        attention_online_ref
        flash_fwd_v1
        flash_fwd_v2
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY}
    )
    # torch::Tensor / TORCH_CHECK need C++17.
    target_compile_features(_C PRIVATE cxx_std_17)
    # Torch's CMake config injects -D_GLIBCXX_USE_CXX11_ABI=... and other
    # required flags via ${TORCH_CXX_FLAGS}; propagate them to keep ABI in sync.
    if(TORCH_CXX_FLAGS)
        target_compile_options(_C PRIVATE ${TORCH_CXX_FLAGS})
    endif()
    # Our CUDA static libs are built with CUDA_SEPARABLE_COMPILATION ON, so
    # they contain `-dc`-compiled device object files. Linking them into a
    # CXX-language shared module requires nvcc's device-symbol-resolution
    # pass, which CMake runs when CUDA_RESOLVE_DEVICE_SYMBOLS is set on the
    # dependent target. Without this, `_C.so` builds but has unresolved
    # `__cudaRegisterFunction` / device relocations at load time.
    #
    # RPATH: torch ships libtorch.so / libtorch_python.so / libc10.so in
    # `${TORCH_INSTALL_PREFIX}/lib`. That directory is normally NOT on the
    # dynamic-linker's search path, so importing `_C.so` fails with
    # `ImportError: libtorch.so: cannot open shared object file`. We bake
    # the torch lib dir into both BUILD and INSTALL RPATHs so the loader
    # finds torch's libs at import time no matter how / where `_C.so` was
    # placed (source tree for editable installs, site-packages for wheels).
    # This is exactly what `torch.utils.cpp_extension.BuildExtension` does
    # for setuptools-driven builds; scikit-build-core doesn't inject it,
    # so we do it explicitly.
    # Because `bindings.cu` is CUDA-language, CMake infers the target's link
    # language as CUDA — so nvcc drives the final link, which properly emits
    # the fatbin. `CUDA_RESOLVE_DEVICE_SYMBOLS ON` then invokes the device-
    # link pass over the separably-compiled CUDA static libs so their device
    # code is resolved into `_C.so`.
    #
    # RPATH: torch ships libtorch.so / libtorch_python.so / libc10.so in
    # `${TORCH_INSTALL_PREFIX}/lib`, which is normally not on the dynamic
    # linker's search path. Baking it into BUILD_RPATH and INSTALL_RPATH
    # lets `_C.so` find torch's shared libs at import time whether it was
    # loaded from an editable source tree or a site-packages wheel. This is
    # the same RPATH that `torch.utils.cpp_extension.BuildExtension` sets
    # for setuptools-driven torch extensions.
    set_target_properties(_C PROPERTIES
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        BUILD_RPATH   "${TORCH_INSTALL_PREFIX}/lib"
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
    )

    # Install into the wheel under flash_from_scratch/ so
    # `from flash_from_scratch import _C` resolves.
    install(TARGETS _C LIBRARY DESTINATION flash_from_scratch)
endif()

if(HAVE_CUDA)
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "BUILD_PY_EXT      : ${BUILD_PY_EXT}")
