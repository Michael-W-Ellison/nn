# File: src/storage/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Storage library
add_library(dpan_storage
    pattern_database.cpp
)

target_include_directories(dpan_storage PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Link to core library
target_link_libraries(dpan_storage PUBLIC
    dpan_core
)

# C++17 required
target_compile_features(dpan_storage PUBLIC cxx_std_17)
