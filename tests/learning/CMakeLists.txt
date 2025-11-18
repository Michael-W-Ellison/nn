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

# Multi-head attention test
add_executable(multi_head_attention_test
    multi_head_attention_test.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/multi_head_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/context_aware_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/basic_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/semantic_attention_head.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/temporal_attention_head.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/structural_attention_head.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/association_attention_head.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/core/types.cpp
)

target_include_directories(multi_head_attention_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tests/learning
)

target_link_libraries(multi_head_attention_test
    dpan_core
    dpan_storage
    dpan_association
    dpan_similarity
    GTest::gtest_main
)

# Semantic attention head test
add_executable(semantic_attention_head_test
    semantic_attention_head_test.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/semantic_attention_head.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/core/types.cpp
)

target_include_directories(semantic_attention_head_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tests/learning
)

target_link_libraries(semantic_attention_head_test
    dpan_core
    dpan_storage
    dpan_similarity
    GTest::gtest_main
)

# Temporal attention head test
add_executable(temporal_attention_head_test
    temporal_attention_head_test.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/temporal_attention_head.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/core/types.cpp
)

target_include_directories(temporal_attention_head_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tests/learning
)

target_link_libraries(temporal_attention_head_test
    dpan_core
    dpan_storage
    GTest::gtest_main
)

# Structural attention head test
add_executable(structural_attention_head_test
    structural_attention_head_test.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/structural_attention_head.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/core/types.cpp
)

target_include_directories(structural_attention_head_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tests/learning
)

target_link_libraries(structural_attention_head_test
    dpan_core
    dpan_storage
    GTest::gtest_main
)

# Association attention head test
add_executable(association_attention_head_test
    association_attention_head_test.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/association_attention_head.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/core/types.cpp
)

target_include_directories(association_attention_head_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tests/learning
)

target_link_libraries(association_attention_head_test
    dpan_core
    dpan_storage
    dpan_association
    GTest::gtest_main
)

# Self-attention test
add_executable(self_attention_test
    self_attention_test.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/self_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/core/types.cpp
)

target_include_directories(self_attention_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tests/learning
)

target_link_libraries(self_attention_test
    dpan_core
    dpan_storage
    dpan_similarity
    GTest::gtest_main
)

# Relationship discovery test
add_executable(relationship_discovery_test
    relationship_discovery_test.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/self_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
    ${PROJECT_SOURCE_DIR}/src/core/types.cpp
)

target_include_directories(relationship_discovery_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/tests/learning
)

target_link_libraries(relationship_discovery_test
    dpan_core
    dpan_storage
    dpan_association
    dpan_similarity
    GTest::gtest_main
)

# Register tests with CTest
include(GoogleTest)
gtest_discover_tests(attention_utils_test)
gtest_discover_tests(basic_attention_test)
gtest_discover_tests(pattern_importance_test)
gtest_discover_tests(context_aware_attention_test)
gtest_discover_tests(multi_head_attention_test)
gtest_discover_tests(semantic_attention_head_test)
gtest_discover_tests(temporal_attention_head_test)
gtest_discover_tests(structural_attention_head_test)
gtest_discover_tests(association_attention_head_test)
gtest_discover_tests(self_attention_test)
gtest_discover_tests(relationship_discovery_test)

