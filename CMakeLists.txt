cmake_minimum_required(VERSION 3.12)
project(AMXBench)

include_directories(include)

set(CMAKE_CXX_STANDARD 23)

# set(CMAKE_C_COMPILER /opt/intel/oneapi/compiler/2024.2/bin/icx)
# set(CMAKE_CXX_COMPILER /opt/intel/oneapi/compiler/2024.2/bin/icpx)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_compile_options(-Ofast -march=native)
add_subdirectory(src/amx)
add_subdirectory(src/common)
add_subdirectory(src/benchmark)
add_subdirectory(src/gemm)