cmake_minimum_required(VERSION 3.12)

project(AMXBench_amx)

# gemm benchmark
add_library(amx SHARED amx.cpp)
