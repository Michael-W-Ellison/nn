# Core tests CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# Types test executable
add_executable(types_test
    types_test.cpp
)

target_link_libraries(types_test
    dpan_core
    GTest::gtest_main
)

# ContextVector test executable
add_executable(context_vector_test
    context_vector_test.cpp
)

target_link_libraries(context_vector_test
    dpan_core
    GTest::gtest_main
)

# PatternData test executable
add_executable(pattern_data_test
    pattern_data_test.cpp
)

target_link_libraries(pattern_data_test
    dpan_core
    GTest::gtest_main
)

# PatternNode test executable
add_executable(pattern_node_test
    pattern_node_test.cpp
)

target_link_libraries(pattern_node_test
    dpan_core
    GTest::gtest_main
)

# PatternEngine test executable
add_executable(pattern_engine_test
    pattern_engine_test.cpp
)

target_link_libraries(pattern_engine_test
    dpan_core
    dpan_storage
    dpan_similarity
    dpan_discovery
    GTest::gtest_main
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(types_test)
gtest_discover_tests(context_vector_test)
gtest_discover_tests(pattern_data_test)
gtest_discover_tests(pattern_node_test)
gtest_discover_tests(pattern_engine_test)

# Process clock and fake-clock reproducibility
add_executable(clock_test
    clock_test.cpp
)

target_link_libraries(clock_test
    dpan_core
    dpan_association
    dpan_memory
    GTest::gtest_main
)

gtest_discover_tests(clock_test)

# Portable serialization fixtures
add_executable(binary_io_test
    binary_io_test.cpp
)

target_link_libraries(binary_io_test
    dpan_core
    dpan_association
    dpan_nn
    GTest::gtest_main
)

gtest_discover_tests(binary_io_test)
