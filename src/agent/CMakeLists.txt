# File: src/agent/CMakeLists.txt
#
# Agent module: drop-in learner configured by an environment spec
# (docs/AGENT_PLAN.md). Phase 1: spec, validation and input encoders.

add_library(dpan_agent
    environment.cpp
    encoders.cpp
)

target_include_directories(dpan_agent PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dpan>
)

target_compile_features(dpan_agent PUBLIC cxx_std_17)
