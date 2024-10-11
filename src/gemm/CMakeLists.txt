cmake_minimum_required(VERSION 3.12)

project(AMXBench_gemm)


# gemm benchmark
file(GLOB SRC "*.cpp")
add_library(gemm SHARED ${SRC})
target_link_libraries(gemm PRIVATE common_utils amx)