// File: src/agent/actor_critic.hpp
//
// TD3 (Fujimoto, van Hoof & Meger 2018) for CONTINUOUS_ACTION outputs: a
// deterministic actor and two critics on the agent's fused state vector.
//
//   actor:   state -> hidden -> hidden -> action in [-1, 1]^n (tanh)
//   critics: [state, action] -> hidden -> hidden -> Q
//
// Each training step updates both critics toward
//   y = reward + discount x min(Q1', Q2')(next state, actor'(next state) + clipped noise)
// (target policy smoothing; the primes are target networks), and every
// policy_delay steps moves the actor up the first critic's gradient and the
// target networks toward the live ones (Polyak averaging with rate tau).
// Actions are in [-1, 1]; the agent maps them to each output's bounds.
#pragma once

#include "nn/optimizer.hpp"
#include "nn/sequential.hpp"

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <random>
#include <vector>

namespace dpan {
namespace agent {

struct ActorCriticConfig {
    size_t hidden{64};              ///< Units in each hidden layer
    float actor_learning_rate{1e-3f};
    float critic_learning_rate{1e-3f};
    float tau{0.005f};              ///< Rate at which target networks follow the live ones
    float target_noise{0.2f};       ///< Std of the target policy smoothing noise
    float noise_clip{0.5f};         ///< Bound on that noise
    size_t policy_delay{2};         ///< Critic updates per actor update
    uint32_t seed{1};
};

class ActorCritic {
public:
    ActorCritic(size_t state_size, size_t action_size, const ActorCriticConfig& config);

    /// One row per transition. discounts: discount of the future value (0
    /// for a terminal step; next_states rows are then ignored)
    struct Batch {
        nn::Matrix states;
        nn::Matrix actions;  ///< In [-1, 1]
        std::vector<float> rewards;
        nn::Matrix next_states;
        std::vector<float> discounts;
    };

    struct Losses {
        float critic{0.0f};  ///< Mean squared error of both critics
        bool actor_updated{false};
    };

    /// Greedy actions in [-1, 1], one row per state
    nn::Matrix Act(const nn::Matrix& states) const;

    /// The first critic's value of the given actions
    std::vector<float> Value(const nn::Matrix& states, const nn::Matrix& actions) const;

    /// One TD3 update; `rng` draws the target smoothing noise
    Losses Train(const Batch& batch, std::mt19937& rng);

    size_t StateSize() const { return state_size_; }
    size_t ActionSize() const { return action_size_; }
    const ActorCriticConfig& Config() const { return config_; }
    uint64_t Updates() const { return updates_; }

    /// Trainable values of the live actor and critics
    size_t ParameterCount() const;

    /// False if any weight of any network (live or target) is NaN or infinite
    bool AllFinite();

    void Serialize(std::ostream& out) const;
    /// @throws std::runtime_error on malformed data
    static std::unique_ptr<ActorCritic> Deserialize(std::istream& in);

    /// For tests: the actor's parameters
    std::vector<nn::Parameter> ActorParameters() { return actor_.Parameters(); }

private:
    ActorCritic() = default;

    nn::Matrix Join(const nn::Matrix& states, const nn::Matrix& actions) const;
    std::vector<nn::Parameter> CriticParameters();
    static void SoftUpdate(nn::Sequential& target, nn::Sequential& live, float tau);

    size_t state_size_{0};
    size_t action_size_{0};
    ActorCriticConfig config_;
    nn::Sequential actor_, actor_target_;
    nn::Sequential critic1_, critic1_target_;
    nn::Sequential critic2_, critic2_target_;
    nn::Adam actor_optimizer_;
    nn::Adam critic_optimizer_;
    uint64_t updates_{0};
};

}  // namespace agent
}  // namespace dpan
