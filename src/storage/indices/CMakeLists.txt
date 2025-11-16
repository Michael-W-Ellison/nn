# File: src/storage/indices/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Indices library
add_library(dpan_indices
    temporal_index.cpp
)

target_include_directories(dpan_indices PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Link to core library
target_link_libraries(dpan_indices PUBLIC
    dpan_core
)

# C++17 required
target_compile_features(dpan_indices PUBLIC cxx_std_17)
