# File: tests/benchmarks/CMakeLists.txt
#
# Performance benchmarks, stress tests, and concurrency tests

# Association benchmarks
add_executable(association_benchmarks
    association_benchmarks.cpp
)

target_link_libraries(association_benchmarks
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(association_benchmarks)

# Storage benchmarks
add_executable(storage_benchmarks
    storage_benchmarks.cpp
)

target_link_libraries(storage_benchmarks
    dpan_storage
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(storage_benchmarks)

# Stress tests
add_executable(stress_tests
    stress_tests.cpp
)

target_link_libraries(stress_tests
    dpan_association
    dpan_storage
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(stress_tests)

# Concurrency tests
add_executable(concurrency_tests
    concurrency_tests.cpp
)

target_link_libraries(concurrency_tests
    dpan_association
    dpan_storage
    dpan_core
    GTest::gtest_main
)

# Link pthread for threading support
target_link_libraries(concurrency_tests
    pthread
)

gtest_discover_tests(concurrency_tests)

# CLI benchmarks
add_executable(cli_benchmarks
    cli_benchmarks.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/dpan_cli.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/basic_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
)

target_link_libraries(cli_benchmarks
    PRIVATE
        dpan_core
        dpan_storage
        dpan_association
        sqlite3
        pthread
        GTest::gtest_main
)

target_include_directories(cli_benchmarks
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_definitions(cli_benchmarks
    PRIVATE
        DPAN_CLI_TEST_BUILD
)

gtest_discover_tests(cli_benchmarks)
