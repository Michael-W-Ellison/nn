# File: tests/memory/CMakeLists.txt
#
# Memory module tests

# UtilityCalculator test executable
add_executable(utility_calculator_test
    utility_calculator_test.cpp
)

target_link_libraries(utility_calculator_test
    dpan_memory
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(utility_calculator_test)
