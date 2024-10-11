cmake_minimum_required(VERSION 3.12)

project(AMXBench_benchmark)


# gemm benchmark
add_library(bench_utils SHARED benchmark_utils.cpp)
file(GLOB GEMM_SRC "gemm_*.cpp")
add_executable(gemm_benchmark ${GEMM_SRC})
target_link_libraries(gemm_benchmark PRIVATE common_utils bench_utils gemm)