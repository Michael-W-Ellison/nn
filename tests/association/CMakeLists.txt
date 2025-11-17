# File: tests/association/CMakeLists.txt
#
# Association module tests

# AssociationEdge test executable
add_executable(association_edge_test
    association_edge_test.cpp
)

target_link_libraries(association_edge_test
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(association_edge_test)

# AssociationMatrix test executable
add_executable(association_matrix_test
    association_matrix_test.cpp
)

target_link_libraries(association_matrix_test
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(association_matrix_test)

# CoOccurrenceTracker test executable
add_executable(co_occurrence_tracker_test
    co_occurrence_tracker_test.cpp
)

target_link_libraries(co_occurrence_tracker_test
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(co_occurrence_tracker_test)

# FormationRules test executable
add_executable(formation_rules_test
    formation_rules_test.cpp
)

target_link_libraries(formation_rules_test
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(formation_rules_test)

# TemporalLearner test executable
add_executable(temporal_learner_test
    temporal_learner_test.cpp
)

target_link_libraries(temporal_learner_test
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(temporal_learner_test)

# SpatialLearner test executable
add_executable(spatial_learner_test
    spatial_learner_test.cpp
)

target_link_libraries(spatial_learner_test
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(spatial_learner_test)

# CategoricalLearner test executable
add_executable(categorical_learner_test
    categorical_learner_test.cpp
)

target_link_libraries(categorical_learner_test
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(categorical_learner_test)

# ReinforcementManager test executable
add_executable(reinforcement_manager_test
    reinforcement_manager_test.cpp
)

target_link_libraries(reinforcement_manager_test
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(reinforcement_manager_test)

# StrengthNormalizer test executable
add_executable(strength_normalizer_test
    strength_normalizer_test.cpp
)

target_link_libraries(strength_normalizer_test
    dpan_association
    dpan_core
    GTest::gtest_main
)

gtest_discover_tests(strength_normalizer_test)
