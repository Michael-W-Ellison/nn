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

# MemoryBackend tests
add_executable(memory_backend_test
    memory_backend_test.cpp
)

target_link_libraries(memory_backend_test
    dpan_core
    dpan_storage
    gtest
    gtest_main
)

gtest_discover_tests(memory_backend_test)

# PersistentBackend tests
add_executable(persistent_backend_test
    persistent_backend_test.cpp
)

target_link_libraries(persistent_backend_test
    dpan_core
    dpan_storage
    gtest
    gtest_main
)

gtest_discover_tests(persistent_backend_test)

# LRUCache tests
add_executable(lru_cache_test
    lru_cache_test.cpp
)

target_link_libraries(lru_cache_test
    dpan_core
    gtest
    gtest_main
)

gtest_discover_tests(lru_cache_test)

# Add indices subdirectory
add_subdirectory(indices)
