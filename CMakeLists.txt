cmake_minimum_required(VERSION 3.20)
project(DPAN VERSION 0.1.0 LANGUAGES CXX)

# Top-level build vs. consumed through add_subdirectory()/FetchContent
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(DPAN_IS_TOP_LEVEL ON)
else()
    set(DPAN_IS_TOP_LEVEL OFF)
endif()

option(DPAN_BUILD_TESTS "Build DPAN tests (downloads GoogleTest)" ${DPAN_IS_TOP_LEVEL})
option(DPAN_BUILD_EXAMPLES "Build DPAN example applications" ${DPAN_IS_TOP_LEVEL})
option(DPAN_BUILD_CLI "Build the dpan_cli interactive tool (requires libyaml)" ${DPAN_IS_TOP_LEVEL})
option(DPAN_INSTALL "Generate install rules and the DPAN CMake package" ON)
option(DPAN_WITH_SQLITE "Build the SQLite-backed PersistentBackend (requires SQLite3)" ON)
option(DPAN_WITH_ZLIB "Compress archive-tier files with zlib (requires ZLIB)" ON)
option(DPAN_WITH_BLAS "Use BLAS (cblas, e.g. OpenBLAS) for matrix products in the nn module" OFF)

# BLAS vendor: OpenBLAS unless the user chose one (FindBLAS BLA_VENDOR)
if(DPAN_WITH_BLAS AND NOT DEFINED BLA_VENDOR)
    set(BLA_VENDOR OpenBLAS)
endif()

if(DPAN_BUILD_CLI AND NOT DPAN_WITH_SQLITE)
    message(STATUS "DPAN_WITH_SQLITE is OFF: dpan_cli and its tests are not built")
    set(DPAN_BUILD_CLI OFF)
endif()

# C++17 standard required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Static libraries may be linked into consumers' shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(DPAN_IS_TOP_LEVEL)
    # Build type
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()

    # Compiler flags
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

include(GNUInstallDirs)

find_package(Threads REQUIRED)

# Library modules
add_subdirectory(src/core)
add_subdirectory(src/storage)
add_subdirectory(src/similarity)
add_subdirectory(src/discovery)
add_subdirectory(src/association)
add_subdirectory(src/memory)
add_subdirectory(src/learning)
add_subdirectory(src/nn)
add_subdirectory(src/agent)

target_link_libraries(dpan_core PUBLIC Threads::Threads)

set(DPAN_LIBRARIES
    dpan_core
    dpan_storage
    dpan_indices
    dpan_similarity
    dpan_discovery
    dpan_association
    dpan_memory
    dpan_learning
    dpan_nn
    dpan_agent
)

# Umbrella target: link dpan::dpan to use the whole library
add_library(dpan INTERFACE)
target_link_libraries(dpan INTERFACE ${DPAN_LIBRARIES})
add_library(dpan::dpan ALIAS dpan)
foreach(lib IN LISTS DPAN_LIBRARIES)
    string(REPLACE "dpan_" "" short_name ${lib})
    add_library(dpan::${short_name} ALIAS ${lib})
endforeach()

# CLI interface
if(DPAN_BUILD_CLI)
    add_subdirectory(src/cli)
endif()

# Tests
if(DPAN_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    add_subdirectory(tests)
endif()

# Example applications
if(DPAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Installation and CMake package (find_package(DPAN))
if(DPAN_INSTALL)
    include(CMakePackageConfigHelpers)

    install(TARGETS ${DPAN_LIBRARIES} dpan
        EXPORT DPANTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # Headers keep their module layout under include/dpan
    set(DPAN_HEADER_EXCLUDES PATTERN "cli" EXCLUDE)
    if(NOT DPAN_WITH_SQLITE)
        list(APPEND DPAN_HEADER_EXCLUDES PATTERN "persistent_backend.hpp" EXCLUDE)
    endif()
    install(DIRECTORY src/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dpan
        FILES_MATCHING PATTERN "*.hpp"
        ${DPAN_HEADER_EXCLUDES}
    )
    install(DIRECTORY include/learning
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dpan
        FILES_MATCHING PATTERN "*.hpp"
    )

    set(DPAN_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/DPAN)

    install(EXPORT DPANTargets
        NAMESPACE dpan::
        DESTINATION ${DPAN_CMAKE_DIR}
    )

    configure_package_config_file(
        cmake/DPANConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/DPANConfig.cmake
        INSTALL_DESTINATION ${DPAN_CMAKE_DIR}
    )
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/DPANConfigVersion.cmake
        COMPATIBILITY SameMinorVersion
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/DPANConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/DPANConfigVersion.cmake
        DESTINATION ${DPAN_CMAKE_DIR}
    )
endif()
