# Neural network module tests

add_executable(nn_core_test nn_core_test.cpp)
target_link_libraries(nn_core_test dpan_nn GTest::gtest_main)
gtest_discover_tests(nn_core_test)

add_executable(nn_integration_test nn_integration_test.cpp)
target_link_libraries(nn_integration_test dpan_nn dpan_storage GTest::gtest_main)
gtest_discover_tests(nn_integration_test)

add_executable(convolution_test convolution_test.cpp)
target_link_libraries(convolution_test dpan_nn GTest::gtest_main)
gtest_discover_tests(convolution_test)

add_executable(compute_test compute_test.cpp)
target_link_libraries(compute_test dpan_nn GTest::gtest_main)
gtest_discover_tests(compute_test)

add_executable(recurrent_test recurrent_test.cpp)
target_link_libraries(recurrent_test dpan_nn GTest::gtest_main)
gtest_discover_tests(recurrent_test)
