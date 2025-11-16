# File: examples/CMakeLists.txt
#
# Build configuration for DPAN example applications

cmake_minimum_required(VERSION 3.20)

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

# Make examples available as build targets
add_custom_target(examples
    DEPENDS
        basic_example
        anomaly_detection
        time_series_patterns
)

message(STATUS "Example applications configured:")
message(STATUS "  - basic_example: Basic pattern recognition")
message(STATUS "  - anomaly_detection: Anomaly detection system")
message(STATUS "  - time_series_patterns: Time series pattern recognition")
message(STATUS "Build all examples with: make examples")
