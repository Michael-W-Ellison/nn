# Find YAML library
find_library(YAML_LIBRARY NAMES yaml libyaml PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
if(NOT YAML_LIBRARY)
    message(FATAL_ERROR "libyaml not found. Please install libyaml-dev")
endif()

# CLI executable
add_executable(dpan_cli
    dpan_cli.cpp
    cli_config.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/basic_attention.cpp
    ${PROJECT_SOURCE_DIR}/src/learning/attention_utils.cpp
)

target_link_libraries(dpan_cli
    PRIVATE
        dpan_core
        dpan_storage
        dpan_association
        sqlite3
        pthread
        ${YAML_LIBRARY}
)

target_include_directories(dpan_cli
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)

# Install the CLI
install(TARGETS dpan_cli
    RUNTIME DESTINATION bin
)
