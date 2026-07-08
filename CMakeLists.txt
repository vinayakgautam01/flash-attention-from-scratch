cmake_minimum_required(VERSION 3.24)
project(flash_from_scratch LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING
        "CUDA architectures to compile for; override at configure time")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

# M0 build-sanity target: prints GPU info + tiny launch.
add_executable(hello csrc/hello.cu)
target_compile_options(hello PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
)

# Torch linking is optional and off by default. Flip on once M10 (PyTorch extension) starts.
option(USE_TORCH "Link against libtorch for the PyTorch extension" OFF)
if(USE_TORCH)
    find_package(Torch REQUIRED)
    message(STATUS "Torch: ${TORCH_LIBRARIES}")
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type        : ${CMAKE_BUILD_TYPE}")
