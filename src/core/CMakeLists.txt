# File: src/core/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Core library
add_library(dpan_core
    types.cpp
    pattern_data.cpp
    pattern_node.cpp
    pattern_engine.cpp
)

target_include_directories(dpan_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Link to required libraries for PatternEngine
target_link_libraries(dpan_core PUBLIC
    dpan_storage
    dpan_similarity
    dpan_discovery
)

# C++17 required
target_compile_features(dpan_core PUBLIC cxx_std_17)
