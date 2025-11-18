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

# Basic attention mechanism test
add_executable(basic_attention_test
    basic_attention_test.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/basic_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/core/types.cpp
)

target_include_directories(basic_attention_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tests/learning
)

target_link_libraries(basic_attention_test
    dpan_core
    dpan_storage
    GTest::gtest_main
)

# Pattern importance test
add_executable(pattern_importance_test
    pattern_importance_test.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/pattern_importance.cpp
    ${PROJECT_SOURCE_DIR}/src/core/types.cpp
)

target_include_directories(pattern_importance_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tests/learning
)

target_link_libraries(pattern_importance_test
    dpan_core
    dpan_storage
    GTest::gtest_main
)

# Context-aware attention test
add_executable(context_aware_attention_test
    context_aware_attention_test.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/context_aware_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/basic_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/core/types.cpp
)

target_include_directories(context_aware_attention_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tests/learning
)

target_link_libraries(context_aware_attention_test
    dpan_core
    dpan_storage
    GTest::gtest_main
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(attention_utils_test)
gtest_discover_tests(basic_attention_test)
gtest_discover_tests(pattern_importance_test)
gtest_discover_tests(context_aware_attention_test)

