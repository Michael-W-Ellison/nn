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
