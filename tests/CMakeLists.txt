# Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Included here so every test subdirectory inherits the module's variables
# (gtest_discover_tests needs them in the calling directory's scope)
include(GoogleTest)

# Test subdirectories
add_subdirectory(core)
add_subdirectory(storage)
add_subdirectory(similarity)
add_subdirectory(discovery)
add_subdirectory(association)
add_subdirectory(memory)
add_subdirectory(learning)
add_subdirectory(integration)
add_subdirectory(benchmarks)
# The CLI uses PersistentBackend
if(DPAN_WITH_SQLITE)
    add_subdirectory(cli)
endif()
add_subdirectory(nn)
add_subdirectory(agent)
if(TARGET dpan_c)
    add_subdirectory(capi)
endif()

# Edge case tests
add_executable(edge_case_tests edge_case_tests.cpp)
target_link_libraries(edge_case_tests PRIVATE
    dpan_core
    dpan_storage
    dpan_association
    dpan_memory
    GTest::gtest_main
)
gtest_discover_tests(edge_case_tests)
