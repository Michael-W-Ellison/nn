# Agent module tests

add_executable(agent_environment_test environment_test.cpp)
target_link_libraries(agent_environment_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_environment_test)

add_executable(agent_encoders_test encoders_test.cpp)
target_link_libraries(agent_encoders_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_encoders_test)

add_executable(agent_reference_test reference_environments_test.cpp)
target_link_libraries(agent_reference_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_reference_test PROPERTIES LABELS reference TIMEOUT 1800)

add_executable(agent_test agent_test.cpp)
target_link_libraries(agent_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_test)

add_executable(agent_components_test components_test.cpp)
target_link_libraries(agent_components_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_components_test)

add_executable(agent_robustness_test robustness_test.cpp)
target_link_libraries(agent_robustness_test dpan_agent GTest::gtest_main)
# The million-step memory test runs for about 30 minutes: it is registered
# separately with the label "long" (skip it with ctest -LE long).
# TEST_FILTER needs CMake 3.22; older versions register all tests together.
if(CMAKE_VERSION VERSION_LESS 3.22)
    gtest_discover_tests(agent_robustness_test PROPERTIES LABELS robustness TIMEOUT 3600)
else()
    gtest_discover_tests(agent_robustness_test
        TEST_FILTER "-*MillionSteps*"
        PROPERTIES LABELS robustness TIMEOUT 600)
    gtest_discover_tests(agent_robustness_test
        TEST_FILTER "*MillionSteps*"
        TEST_SUFFIX ".long"
        PROPERTIES LABELS long TIMEOUT 3600)
endif()

add_executable(agent_persistence_test persistence_test.cpp)
target_link_libraries(agent_persistence_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_persistence_test PROPERTIES TIMEOUT 600)

# Timing: run alone (RUN_SERIAL); skipped in builds without BLAS
add_executable(agent_performance_test performance_test.cpp)
target_link_libraries(agent_performance_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_performance_test PROPERTIES LABELS performance RUN_SERIAL TRUE)

add_executable(agent_json_test json_test.cpp)
target_link_libraries(agent_json_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_json_test)

add_executable(agent_actor_critic_test actor_critic_test.cpp)
target_link_libraries(agent_actor_critic_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_actor_critic_test)

add_executable(agent_continuous_test continuous_test.cpp)
target_link_libraries(agent_continuous_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_continuous_test PROPERTIES TIMEOUT 600)

# Uses beyond the reference environments (S1-S10, seeds 1-3); label
# "scenarios". Skipped unless DPAN_SCENARIOS is set:
#   DPAN_SCENARIOS=1 ctest --test-dir build -L scenarios
add_executable(agent_scenarios_test scenarios_test.cpp)
target_link_libraries(agent_scenarios_test dpan_agent GTest::gtest_main)
gtest_discover_tests(agent_scenarios_test PROPERTIES
    LABELS scenarios
    TIMEOUT 900
    SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")
