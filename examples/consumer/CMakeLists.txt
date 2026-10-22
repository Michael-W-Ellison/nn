# Example of using DPAN from another CMake project.
#
# Installed package:
#   cmake -S examples/consumer -B build-consumer -DCMAKE_PREFIX_PATH=<dpan install prefix>
# Source tree (add_subdirectory):
#   cmake -S examples/consumer -B build-consumer -DDPAN_SOURCE_DIR=<path to dpan checkout>
cmake_minimum_required(VERSION 3.20)
project(dpan_consumer LANGUAGES C CXX)

set(DPAN_SOURCE_DIR "" CACHE PATH "Build DPAN from this source tree instead of find_package")

if(DPAN_SOURCE_DIR)
    add_subdirectory(${DPAN_SOURCE_DIR} dpan)
else()
    find_package(DPAN 0.1 REQUIRED)
endif()

add_executable(dpan_consumer main.cpp)
target_link_libraries(dpan_consumer PRIVATE dpan::dpan)

# The C API (a shared library), if DPAN was built with DPAN_BUILD_C_API
if(TARGET dpan::c_api)
    add_executable(dpan_c_consumer c_main.c)
    target_link_libraries(dpan_c_consumer PRIVATE dpan::c_api)
endif()
