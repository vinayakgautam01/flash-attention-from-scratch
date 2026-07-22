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
endif()

if(HAVE_CUDA)
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "Build type        : ${CMAKE_BUILD_TYPE}")
