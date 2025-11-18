# CLI executable
add_executable(dpan_cli dpan_cli.cpp)

target_link_libraries(dpan_cli
    PRIVATE
        dpan_core
        dpan_storage
        dpan_association
        sqlite3
        pthread
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
