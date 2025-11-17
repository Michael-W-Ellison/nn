# File: tests/memory/CMakeLists.txt
#
# Memory module tests

# UtilityCalculator test executable
add_executable(utility_calculator_test
    utility_calculator_test.cpp
)

target_link_libraries(utility_calculator_test
    dpan_memory
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(utility_calculator_test)

# AdaptiveThresholds test executable
add_executable(adaptive_thresholds_test
    adaptive_thresholds_test.cpp
)

target_link_libraries(adaptive_thresholds_test
    dpan_memory
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(adaptive_thresholds_test)

# UtilityTracker test executable
add_executable(utility_tracker_test
    utility_tracker_test.cpp
)

target_link_libraries(utility_tracker_test
    dpan_memory
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(utility_tracker_test)

# MemoryTier test executable
add_executable(memory_tier_test
    memory_tier_test.cpp
)

target_link_libraries(memory_tier_test
    dpan_memory
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(memory_tier_test)

# TierManager test executable
add_executable(tier_manager_test
    tier_manager_test.cpp
)

target_link_libraries(tier_manager_test
    dpan_memory
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(tier_manager_test)

# TieredStorage test executable
add_executable(tiered_storage_test
    tiered_storage_test.cpp
)

target_link_libraries(tiered_storage_test
    dpan_memory
    dpan_storage
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(tiered_storage_test)

# PatternPruner test executable
add_executable(pattern_pruner_test
    pattern_pruner_test.cpp
)

target_link_libraries(pattern_pruner_test
    dpan_memory
    dpan_storage
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(pattern_pruner_test)

# AssociationPruner test executable
add_executable(association_pruner_test
    association_pruner_test.cpp
)

target_link_libraries(association_pruner_test
    dpan_memory
    dpan_storage
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(association_pruner_test)

# Consolidator test executable
add_executable(consolidator_test
    consolidator_test.cpp
)

target_link_libraries(consolidator_test
    dpan_memory
    dpan_storage
    dpan_similarity
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(consolidator_test)
