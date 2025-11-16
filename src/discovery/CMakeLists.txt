# File: src/discovery/CMakeLists.txt

# Discovery library
add_library(dpan_discovery
    pattern_extractor.cpp
    pattern_matcher.cpp
    pattern_creator.cpp
)

target_include_directories(dpan_discovery PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)

# Link to required libraries
target_link_libraries(dpan_discovery PUBLIC
    dpan_core
    dpan_storage
    dpan_similarity
)

# C++17 required
target_compile_features(dpan_discovery PUBLIC cxx_std_17)
