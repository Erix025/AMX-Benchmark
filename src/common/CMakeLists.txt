cmake_minimum_required(VERSION 3.12)

project(AMXBench_utils)


# utils
file(GLOB SRC "*.cpp")
add_library(common_utils SHARED ${SRC})