# Tests CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Test subdirectories
add_subdirectory(core)
add_subdirectory(storage)
add_subdirectory(similarity)
add_subdirectory(discovery)
add_subdirectory(association)
add_subdirectory(memory)
add_subdirectory(integration)
add_subdirectory(benchmarks)

# Edge case tests (added for TDD analysis)
add_executable(edge_case_tests edge_case_tests.cpp)
target_link_libraries(edge_case_tests PRIVATE
    dpan_core
    dpan_storage
    dpan_association
    dpan_memory
    gtest_main
)
add_test(NAME EdgeCaseTests COMMAND edge_case_tests)
