# File: tests/integration/CMakeLists.txt

# Integration test executable
add_executable(integration_test
    integration_test.cpp
)

target_link_libraries(integration_test
    dpan_core
    dpan_storage
    dpan_similarity
    dpan_discovery
    gtest
    gtest_main
)

# Enable C++17 filesystem
target_compile_features(integration_test PRIVATE cxx_std_17)

gtest_discover_tests(integration_test)
