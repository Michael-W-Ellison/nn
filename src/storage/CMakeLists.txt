# File: src/storage/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Find SQLite3
find_package(PkgConfig REQUIRED)
pkg_check_modules(SQLITE3 REQUIRED sqlite3)

# Storage library
add_library(dpan_storage
    pattern_database.cpp
    memory_backend.cpp
    persistent_backend.cpp
)

target_include_directories(dpan_storage PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${SQLITE3_INCLUDE_DIRS}
)

# Link to core library and SQLite3
target_link_libraries(dpan_storage PUBLIC
    dpan_core
    ${SQLITE3_LIBRARIES}
)

# C++17 required
target_compile_features(dpan_storage PUBLIC cxx_std_17)

# Add indices subdirectory
add_subdirectory(indices)
