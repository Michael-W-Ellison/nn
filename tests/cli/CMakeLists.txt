# CLI Tests CMakeLists.txt

add_executable(dpan_cli_test
    dpan_cli_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/dpan_cli.cpp
)

target_link_libraries(dpan_cli_test
    PRIVATE
        dpan_core
        dpan_storage
        dpan_association
        sqlite3
        pthread
        gtest_main
)

target_include_directories(dpan_cli_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

target_compile_definitions(dpan_cli_test
    PRIVATE
        DPAN_CLI_TEST_BUILD
)

add_test(NAME DPANCliTest COMMAND dpan_cli_test)

# Set test properties
set_tests_properties(DPANCliTest PROPERTIES
    TIMEOUT 60
    LABELS "cli;integration"
)
