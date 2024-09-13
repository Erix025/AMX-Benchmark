cmake_minimum_required(VERSION 3.12)
project(AMX-Benchmark)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

# find_package(OpenMP REQUIRED)

# 添加源文件
file(GLOB SOURCES "src/*.cpp" "src/gemv/*.cpp")

# 添加可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 添加头文件搜索路径
target_include_directories(${PROJECT_NAME} PRIVATE include)
# target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(${PROJECT_NAME} PRIVATE -Ofast -march=native -fno-strict-aliasing)