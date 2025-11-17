# File: src/memory/CMakeLists.txt
#
# Memory module: Utility-based memory management

add_library(dpan_memory
    utility_calculator.cpp
    adaptive_thresholds.cpp
    utility_tracker.cpp
)

target_include_directories(dpan_memory PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(dpan_memory PUBLIC
    dpan_core
    dpan_association
)
