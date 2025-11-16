# File: src/association/CMakeLists.txt
#
# Association Learning System module

add_library(dpan_association
    association_edge.cpp
    association_matrix.cpp
    co_occurrence_tracker.cpp
    formation_rules.cpp
)

target_include_directories(dpan_association PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(dpan_association PUBLIC
    dpan_core
)

# C++17 required
target_compile_features(dpan_association PUBLIC cxx_std_17)
