# File: tests/similarity/CMakeLists.txt

# SimilarityMetric framework tests
add_executable(similarity_metric_test
    similarity_metric_test.cpp
)

target_link_libraries(similarity_metric_test
    dpan_core
    dpan_similarity
    gtest
    gtest_main
)

# Register test with CTest
include(GoogleTest)
gtest_discover_tests(similarity_metric_test)

# Geometric similarity tests
add_executable(geometric_similarity_test
    geometric_similarity_test.cpp
)

target_link_libraries(geometric_similarity_test
    dpan_core
    dpan_similarity
    gtest
    gtest_main
)

gtest_discover_tests(geometric_similarity_test)
