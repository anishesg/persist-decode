cmake_minimum_required(VERSION 3.20)
project(persist_decode CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Require sm_80 or higher (A100 / H100)
set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)

# Global compile options
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda")

# Source files for the static library (header-only device functions + .cu translation units)
add_library(persist_decode_lib STATIC
    src/persistent_layer.cu
    src/reference.cu
    src/persistent_multi_layer.cu
)

target_include_directories(persist_decode_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set_target_properties(persist_decode_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Test executable
add_executable(test_correctness tests/test_correctness.cu)
target_link_libraries(test_correctness PRIVATE persist_decode_lib)
set_target_properties(test_correctness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Benchmark executable
add_executable(bench_latency benchmarks/bench_latency.cu)
target_link_libraries(bench_latency PRIVATE persist_decode_lib)
set_target_properties(bench_latency PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Build type defaults
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
