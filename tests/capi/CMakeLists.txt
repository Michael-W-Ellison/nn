# C API tests: link only the shared library dpan_c, as a C program would
add_executable(c_api_test c_api_test.cpp)
target_link_libraries(c_api_test dpan_c GTest::gtest_main)
gtest_discover_tests(c_api_test PROPERTIES TIMEOUT 600)
