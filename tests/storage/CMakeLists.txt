# File: tests/storage/CMakeLists.txt

# PatternDatabase interface tests
add_executable(pattern_database_test
    pattern_database_test.cpp
)

target_link_libraries(pattern_database_test
    dpan_core
    dpan_storage
    gtest
    gtest_main
)

# Register test with CTest
include(GoogleTest)
gtest_discover_tests(pattern_database_test)
