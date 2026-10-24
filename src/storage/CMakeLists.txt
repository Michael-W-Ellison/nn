# File: src/storage/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Storage library
add_library(dpan_storage STATIC
    pattern_database.cpp
    memory_backend.cpp
)

target_include_directories(dpan_storage PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dpan>
)

target_link_libraries(dpan_storage PUBLIC dpan_core)

# SQLite-backed PersistentBackend (optional)
if(DPAN_WITH_SQLITE)
    find_package(SQLite3 REQUIRED)
    target_sources(dpan_storage PRIVATE persistent_backend.cpp)
    target_link_libraries(dpan_storage PUBLIC SQLite::SQLite3)
    target_compile_definitions(dpan_storage PUBLIC DPAN_HAS_SQLITE=1)
endif()

# C++17 required
target_compile_features(dpan_storage PUBLIC cxx_std_17)

# Add indices subdirectory
add_subdirectory(indices)
