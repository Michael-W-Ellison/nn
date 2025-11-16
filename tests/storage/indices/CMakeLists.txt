# File: tests/storage/indices/CMakeLists.txt

# TemporalIndex tests
add_executable(temporal_index_test
    temporal_index_test.cpp
)

target_link_libraries(temporal_index_test
    dpan_core
    dpan_indices
    gtest
    gtest_main
)

# Register test with CTest
include(GoogleTest)
gtest_discover_tests(temporal_index_test)
