# File: examples/CMakeLists.txt
#
# Build configuration for DPAN example applications

cmake_minimum_required(VERSION 3.20)

# Association learning examples
add_subdirectory(association)

# Basic example - simple pattern recognition
add_executable(basic_example
    basic_example.cpp
)

target_link_libraries(basic_example
    dpan_core
    dpan_storage
    dpan_similarity
    dpan_discovery
)

# Anomaly detection example
add_executable(anomaly_detection
    anomaly_detection.cpp
)

target_link_libraries(anomaly_detection
    dpan_core
    dpan_storage
    dpan_similarity
    dpan_discovery
)

# Time series pattern recognition example
add_executable(time_series_patterns
    time_series_patterns.cpp
)

target_link_libraries(time_series_patterns
    dpan_core
    dpan_storage
    dpan_similarity
    dpan_discovery
)

# Neural network module and its DPAN integration
add_executable(neural_network_example nn/neural_network_example.cpp)
target_link_libraries(neural_network_example dpan_nn dpan_storage)

# Drop-in agent
add_executable(agent_example agent/agent_example.cpp)
target_link_libraries(agent_example dpan_agent)

# Drop-in agent through the C API (compiled as C)
if(TARGET dpan_c)
    add_executable(agent_c_example c_api/agent_c_example.c)
    target_link_libraries(agent_c_example dpan_c)
    set_target_properties(agent_c_example PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
    if(DPAN_BUILD_TESTS)
        add_test(NAME agent_c_example
                 COMMAND agent_c_example ${CMAKE_CURRENT_BINARY_DIR}/agent_c_example.dpag)
        set_tests_properties(agent_c_example PROPERTIES TIMEOUT 600)
    endif()
endif()

# Make examples available as build targets
add_custom_target(examples
    DEPENDS
        basic_example
        anomaly_detection
        time_series_patterns
        basic_learning
        custom_formation_rules
        activation_propagation_demo
        neural_network_example
        agent_example
        $<$<TARGET_EXISTS:dpan_c>:agent_c_example>
)

message(STATUS "Example applications configured:")
message(STATUS "  - basic_example: Basic pattern recognition")
message(STATUS "  - anomaly_detection: Anomaly detection system")
message(STATUS "  - time_series_patterns: Time series pattern recognition")
message(STATUS "  - basic_learning: Association learning basics")
message(STATUS "  - custom_formation_rules: Custom association rules")
message(STATUS "  - activation_propagation_demo: Activation propagation")
message(STATUS "  - neural_network_example: Neural networks and DPAN integration")
message(STATUS "  - agent_example: Drop-in agent (classification and anomalies)")
message(STATUS "Build all examples with: make examples")
