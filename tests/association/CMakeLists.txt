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
