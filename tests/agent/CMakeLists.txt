# Agent module tests

add_executable(agent_environment_test environment_test.cpp)
target_link_libraries(agent_environment_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_environment_test)

add_executable(agent_encoders_test encoders_test.cpp)
target_link_libraries(agent_encoders_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_encoders_test)

add_executable(agent_reference_test reference_environments_test.cpp)
target_link_libraries(agent_reference_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_reference_test PROPERTIES LABELS "agent;reference" TIMEOUT 600)

add_executable(agent_test agent_test.cpp)
target_link_libraries(agent_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_test)

add_executable(agent_components_test components_test.cpp)
target_link_libraries(agent_components_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_components_test)
