# File: src/discovery/CMakeLists.txt

# Discovery library
add_library(dpan_discovery STATIC
    pattern_extractor.cpp
    pattern_matcher.cpp
    pattern_creator.cpp
    pattern_refiner.cpp
)

target_include_directories(dpan_discovery PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dpan>
)

# Link to required libraries
target_link_libraries(dpan_discovery PUBLIC
    dpan_core
    dpan_storage
    dpan_similarity
)

# C++17 required
target_compile_features(dpan_discovery PUBLIC cxx_std_17)
