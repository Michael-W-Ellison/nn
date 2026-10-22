// File: tests/agent/actor_critic_test.cpp
#include "agent/actor_critic.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>

namespace dpan {
namespace agent {
namespace {

/// One-step task (discount 0): state s in [-1, 1]^2, reward -(a - target(s))^2
/// with target(s) = 0.6 s0 - 0.3 s1. The best action is target(s).
ActorCritic::Batch BanditBatch(std::mt19937& rng, size_t n) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    ActorCritic::Batch batch{nn::Matrix(n, 2), nn::Matrix(n, 1), std::vector<float>(n), nn::Matrix(n, 2),
                             std::vector<float>(n, 0.0f)};
    for (size_t r = 0; r < n; ++r) {
        float s0 = u(rng), s1 = u(rng), a = u(rng);
        batch.states(r, 0) = s0;
        batch.states(r, 1) = s1;
        batch.actions(r, 0) = a;
        float best = 0.6f * s0 - 0.3f * s1;
        batch.rewards[r] = -(a - best) * (a - best);
    }
    return batch;
}

TEST(ActorCriticTest, LearnsTheBestActionOfABandit) {
    ActorCriticConfig config;
    ActorCritic ac(2, 1, config);
    std::mt19937 data(1), noise(2);
    for (int i = 0; i < 3000; ++i) ac.Train(BanditBatch(data, 64), noise);
    EXPECT_EQ(3000u, ac.Updates());

    nn::Matrix states(5, 2);
    float probes[5][2] = {{0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0.5f, -0.5f}};
    for (size_t r = 0; r < 5; ++r) {
        states(r, 0) = probes[r][0];
        states(r, 1) = probes[r][1];
    }
    nn::Matrix actions = ac.Act(states);
    for (size_t r = 0; r < 5; ++r) {
        // The actor's action is in the top 5% of the critic's values over a
        // grid of actions (the actor follows a critic that keeps changing,
        // and the critic is nearly flat near its maximum, so values are
        // compared, not positions), and it is near the true best action;
        // the critic's approximation error near the optimum leaves an
        // offset of < 0.15
        nn::Matrix state(41, 2), grid(41, 1);
        for (size_t i = 0; i < 41; ++i) {
            state(i, 0) = probes[r][0];
            state(i, 1) = probes[r][1];
            grid(i, 0) = -1.0f + 0.05f * static_cast<float>(i);
        }
        auto q = ac.Value(state, grid);
        float grid_best = *std::max_element(q.begin(), q.end());
        float grid_worst = *std::min_element(q.begin(), q.end());
        nn::Matrix own(1, 2);
        own(0, 0) = probes[r][0];
        own(0, 1) = probes[r][1];
        float actor_value = ac.Value(own, nn::Matrix(1, 1, actions(r, 0)))[0];
        EXPECT_GE(actor_value, grid_best - 0.05f * (grid_best - grid_worst)) << "state " << r;
        float best = 0.6f * probes[r][0] - 0.3f * probes[r][1];
        EXPECT_NEAR(best, actions(r, 0), 0.15f) << "state " << r;
        EXPECT_LE(std::fabs(actions(r, 0)), 1.0f);
    }
    // The critic ranks the best action above a worse one
    nn::Matrix worse(5, 1, 0.9f);
    auto good = ac.Value(states, actions);
    auto bad = ac.Value(states, worse);
    for (size_t r = 1; r < 5; ++r) EXPECT_GT(good[r], bad[r]) << r;
}

TEST(ActorCriticTest, BootstrapsThroughTheTargetCritics) {
    // Two-step chain: from s = 1 every action leads to s = 0 with reward
    // 0; from s = 0 the reward is -(a - 0.5)^2 and the episode ends. With
    // discount 0.9, Q(1, a) = 0.9 x max Q(0, .) = 0 for every action, and
    // the actor at s = 0 must still learn a = 0.5.
    ActorCriticConfig config;
    ActorCritic ac(1, 1, config);
    std::mt19937 data(3), noise(4);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int i = 0; i < 3000; ++i) {
        const size_t n = 64;
        ActorCritic::Batch batch{nn::Matrix(n, 1), nn::Matrix(n, 1), std::vector<float>(n), nn::Matrix(n, 1),
                                 std::vector<float>(n)};
        for (size_t r = 0; r < n; ++r) {
            float a = u(data);
            bool first = r % 2 == 0;
            batch.states(r, 0) = first ? 1.0f : 0.0f;
            batch.actions(r, 0) = a;
            batch.rewards[r] = first ? 0.0f : -(a - 0.5f) * (a - 0.5f);
            batch.next_states(r, 0) = 0.0f;
            batch.discounts[r] = first ? 0.9f : 0.0f;
        }
        ac.Train(batch, noise);
    }
    nn::Matrix zero(1, 1, 0.0f), one(1, 1, 1.0f);
    EXPECT_NEAR(0.5f, ac.Act(zero)(0, 0), 0.1f);
    // Q(1, a) is about 0.9 x Q(0, 0.5) = 0 (target smoothing lowers it slightly)
    EXPECT_NEAR(0.0f, ac.Value(one, nn::Matrix(1, 1, 0.3f))[0], 0.05f);
}

TEST(ActorCriticTest, SavesLoadsAndCopiesExactly) {
    ActorCriticConfig config;
    config.hidden = 16;
    ActorCritic ac(2, 1, config);
    std::mt19937 data(5), noise(6);
    for (int i = 0; i < 25; ++i) ac.Train(BanditBatch(data, 32), noise);

    std::stringstream buffer;
    ac.Serialize(buffer);
    auto loaded = ActorCritic::Deserialize(buffer);
    ActorCritic copy = ac;
    EXPECT_EQ(ac.Updates(), loaded->Updates());
    EXPECT_EQ(ac.ParameterCount(), loaded->ParameterCount());

    // The three continue identically
    std::mt19937 data_b = data, data_c = data, noise_b = noise, noise_c = noise;
    for (int i = 0; i < 10; ++i) {
        auto a = ac.Train(BanditBatch(data, 32), noise);
        auto b = loaded->Train(BanditBatch(data_b, 32), noise_b);
        auto c = copy.Train(BanditBatch(data_c, 32), noise_c);
        EXPECT_EQ(a.critic, b.critic);
        EXPECT_EQ(a.critic, c.critic);
        EXPECT_EQ(a.actor_updated, b.actor_updated);
    }
    nn::Matrix states = BanditBatch(data, 8).states;
    EXPECT_EQ(ac.Act(states).Values(), loaded->Act(states).Values());
    EXPECT_EQ(ac.Act(states).Values(), copy.Act(states).Values());
    EXPECT_TRUE(ac.AllFinite());

    std::stringstream truncated(buffer.str().substr(0, 40));
    EXPECT_ANY_THROW(ActorCritic::Deserialize(truncated));
}

TEST(ActorCriticTest, DelaysActorUpdates) {
    ActorCriticConfig config;
    config.policy_delay = 3;
    ActorCritic ac(2, 1, config);
    std::mt19937 data(7), noise(8);
    std::vector<bool> updated;
    for (int i = 0; i < 6; ++i) updated.push_back(ac.Train(BanditBatch(data, 8), noise).actor_updated);
    EXPECT_EQ((std::vector<bool>{false, false, true, false, false, true}), updated);
    EXPECT_THROW(ActorCritic(0, 1, config), std::invalid_argument);
}

}  // namespace
}  // namespace agent
}  // namespace dpan
