// File: src/agent/agent_testing.hpp
//
// Fault injection for tests of the agent's safety mechanisms. Not for use
// in applications.
#pragma once

#include "agent/agent.hpp"

namespace dpan {
namespace agent {

struct AgentTestAccess {
    /// Set one weight of the network (or the anomaly autoencoder if there
    /// is no network) to NaN. Returns false if no model exists yet.
    static bool InjectNaN(Agent& agent);
};

}  // namespace agent
}  // namespace dpan
