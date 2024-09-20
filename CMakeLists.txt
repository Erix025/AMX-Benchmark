cmake_minimum_required(VERSION 3.12)
project(AMX-Benchmark)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

# 添加源文件
file(GLOB SOURCES "src/*.cpp" "src/gemv/*.cpp" "src/gemv/test/*.cpp" "src/gemm/*.cpp")

# 添加可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 添加头文件搜索路径
target_include_directories(${PROJECT_NAME} PRIVATE include)

# 设置调试选项
option(DEBUG "Enable debug mode" OFF)
if(DEBUG)
    target_compile_options(${PROJECT_NAME} PRIVATE -g -O0 -march=native -fno-strict-aliasing)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Ofast -march=native -fno-strict-aliasing)
endif()