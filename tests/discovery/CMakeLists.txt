# File: tests/discovery/CMakeLists.txt

# Pattern extractor tests
add_executable(pattern_extractor_test
    pattern_extractor_test.cpp
)

target_link_libraries(pattern_extractor_test
    dpan_core
    dpan_discovery
    gtest
    gtest_main
)

gtest_discover_tests(pattern_extractor_test)
