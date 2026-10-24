# File: src/agent/CMakeLists.txt
#
# Agent module: drop-in learner configured by an environment spec
# (docs/AGENT_PLAN.md).

add_library(dpan_agent STATIC
    environment.cpp
    encoders.cpp
    network.cpp
    pattern_memory.cpp
    replay_buffer.cpp
    agent.cpp
    json.cpp
    actor_critic.cpp
    spec_json.cpp
)

target_include_directories(dpan_agent PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/dpan>
)

target_link_libraries(dpan_agent PUBLIC
    dpan_core
    dpan_association
    dpan_memory
    dpan_nn
)

target_compile_features(dpan_agent PUBLIC cxx_std_17)
