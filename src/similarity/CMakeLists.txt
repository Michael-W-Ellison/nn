# File: src/similarity/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Similarity library
add_library(dpan_similarity
    similarity_metric.cpp
    geometric_similarity.cpp
    frequency_similarity.cpp
    statistical_similarity.cpp
    contextual_similarity.cpp
)

target_include_directories(dpan_similarity PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Link to core library
target_link_libraries(dpan_similarity PUBLIC
    dpan_core
)

# C++17 required
target_compile_features(dpan_similarity PUBLIC cxx_std_17)
