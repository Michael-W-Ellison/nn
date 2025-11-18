# CLI Tests CMakeLists.txt

add_executable(dpan_cli_test
    dpan_cli_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/dpan_cli.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/cli_config.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/basic_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
)

find_library(YAML_LIBRARY NAMES yaml libyaml PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

target_link_libraries(dpan_cli_test
    PRIVATE
        dpan_core
        dpan_storage
        dpan_association
        sqlite3
        pthread
        gtest_main
        ${YAML_LIBRARY}
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

# Config tests
add_executable(cli_config_test
    cli_config_test.cpp
    ${PROJECT_SOURCE_DIR}/src/cli/cli_config.cpp
)

find_library(YAML_LIBRARY NAMES yaml libyaml PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

target_link_libraries(cli_config_test
    PRIVATE
        gtest_main
        ${YAML_LIBRARY}
)

target_include_directories(cli_config_test
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

add_test(NAME CliConfigTest COMMAND cli_config_test)

set_tests_properties(CliConfigTest PROPERTIES
    TIMEOUT 10
    LABELS "cli;config;unit"
)
