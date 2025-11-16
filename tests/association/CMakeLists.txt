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
