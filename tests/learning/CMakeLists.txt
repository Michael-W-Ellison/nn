# Attention Mechanism Tests
cmake_minimum_required(VERSION 3.20)

# Attention utilities test
add_executable(attention_utils_test
    attention_utils_test.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/core/types.cpp
)

target_include_directories(attention_utils_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(attention_utils_test
    dpan_core
    GTest::gtest_main
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(attention_utils_test)
