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

# Pattern matcher tests
add_executable(pattern_matcher_test
    pattern_matcher_test.cpp
)

target_link_libraries(pattern_matcher_test
    dpan_core
    dpan_storage
    dpan_similarity
    dpan_discovery
    gtest
    gtest_main
)

gtest_discover_tests(pattern_matcher_test)

# Pattern creator tests
add_executable(pattern_creator_test
    pattern_creator_test.cpp
)

target_link_libraries(pattern_creator_test
    dpan_core
    dpan_storage
    dpan_discovery
    gtest
    gtest_main
)

gtest_discover_tests(pattern_creator_test)
