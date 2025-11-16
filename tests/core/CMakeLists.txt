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

# Register test with CTest
include(GoogleTest)
gtest_discover_tests(types_test)
