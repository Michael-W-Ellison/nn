// File: tests/agent/memory_test.cpp
//
// Phase 9 (docs/AGENT_PLAN.md): memory. Tasks M1-M4 need information from
// earlier steps that the current observation does not show; their pass
// criteria were fixed before the first run. With DPAN_MEMORY_BASELINE set,
// M1-M4 run without memory and only print their results (to show that
// they need it).

#include "agent/agent.hpp"
#include "core/clock.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace dpan {
namespace agent {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr size_t kMemoryUnits = 32;

class MemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        clock_ = std::make_shared<ManualClock>(Timestamp::FromMicros(1'700'000'000'000'000));
        scope_ = std::make_unique<ScopedClock>(clock_);
    }
    void TearDown() override { scope_.reset(); }
    void Tick() { clock_->Advance(std::chrono::seconds(1)); }

    static bool Baseline() { return std::getenv("DPAN_MEMORY_BASELINE") != nullptr; }
    static AgentConfig Config() {
        AgentConfig config;
        config.memory_units = Baseline() ? 0 : kMemoryUnits;
        return config;
    }
    /// The criterion holds; in a baseline run, just report
    static void Check(bool pass, const std::string& result) {
        std::cout << (Baseline() ? "[without memory] " : "") << result << (pass ? " (pass)" : " (fail)") << "\n";
        if (!Baseline()) EXPECT_TRUE(pass) << result;
    }

    std::shared_ptr<ManualClock> clock_;
    std::unique_ptr<ScopedClock> scope_;

    // ------------------------------------------------------------------ M1
    /// Accuracy at the last step over the last 500 of 3,000 episodes
    double DelayedRecall(size_t delay) {
        const std::vector<std::string> cues = {"A", "B", "C", "D"};
        EnvironmentSpec spec;
        spec.inputs = {{"cue", InputType::CATEGORICAL, 0, cues}, {"noise", InputType::NUMERIC, 2}};
        spec.outputs = {{"recall", OutputType::CLASS, cues}};
        spec.feedback = {{"label", FeedbackType::LABEL, "recall"}};
        Agent agent(spec, Config());
        std::mt19937 rng(static_cast<uint32_t>(100 + delay));
        std::uniform_int_distribution<size_t> pick(0, cues.size() - 1);
        std::normal_distribution<float> noise(0.0f, 1.0f);
        const int episodes = 3000;
        int correct = 0;
        for (int e = 0; e < episodes; ++e) {
            const std::string& cue = cues[pick(rng)];
            StepId step = 0;
            for (size_t t = 0; t <= delay; ++t) {
                Observation o = {{"noise", std::vector<float>{noise(rng), noise(rng)}}};
                if (t == 0) o["cue"] = cue;
                step = agent.Observe(o);
                Tick();
            }
            Prediction p = agent.Predict(step, "recall");
            if (e >= episodes - 500 && p.available && p.label == cue) ++correct;
            agent.Feedback(step, "label", cue);
            agent.EndEpisode(false);
        }
        return correct / 500.0;
    }

    void CheckRecall(size_t delay) {
        double accuracy = DelayedRecall(delay);
        Check(accuracy >= 0.9, "M1 delay " + std::to_string(delay) + ": accuracy " + std::to_string(accuracy));
    }
};

TEST_F(MemoryTest, M1_DelayedRecall5) { CheckRecall(5); }
TEST_F(MemoryTest, M1_DelayedRecall20) { CheckRecall(20); }
TEST_F(MemoryTest, M1_DelayedRecall50) { CheckRecall(50); }

// -------------------------------------------------------------------- M2
TEST_F(MemoryTest, M2_TMaze) {
    EnvironmentSpec spec;
    spec.inputs = {{"position", InputType::NUMERIC, 1}, {"cue", InputType::CATEGORICAL, 0, {"left", "right"}}};
    spec.outputs = {{"move", OutputType::ACTION, {"forward", "left", "right"}}};
    spec.feedback = {{"reward", FeedbackType::REWARD, "move"}};
    Agent agent(spec, Config());
    std::mt19937 rng(201);
    std::bernoulli_distribution coin(0.5);
    const int length = 10;
    // Returns whether the episode ended with a turn toward the cue
    auto episode = [&](bool explore) {
        const std::string cue = coin(rng) ? "left" : "right";
        int position = 0;
        bool correct = false, ended = false;
        for (int t = 0; t < 30 && !ended; ++t) {
            Observation o = {{"position", static_cast<double>(position) / length}};
            if (t == 0) o["cue"] = cue;
            StepId step = agent.Observe(o);
            std::string move = agent.Act(step, "move", explore);
            double reward = 0.0;
            if (position < length) {
                if (move == "forward") ++position;
                else reward = -0.1;
            } else if (move == "forward") {
                reward = -0.1;
            } else {
                ended = true;
                correct = move == cue;
                reward = correct ? 4.0 : -1.0;
            }
            agent.Feedback(step, "reward", reward);
            Tick();
        }
        agent.EndEpisode(ended);  // cut off after 30 steps: not terminal
        return correct;
    };
    for (int e = 0; e < 2000; ++e) episode(true);
    int correct = 0;
    for (int e = 0; e < 100; ++e) correct += episode(false) ? 1 : 0;
    Check(correct >= 90, "M2: " + std::to_string(correct) + " of 100 greedy episodes turn toward the cue");
}

// -------------------------------------------------------------------- M3
// Environment F's cart-pole, observing only the cart position and the pole angle
class CartPole {
public:
    explicit CartPole(uint32_t seed) : rng_(seed) {}
    void Reset() {
        std::uniform_real_distribution<float> u(-0.05f, 0.05f);
        state_ = {u(rng_), u(rng_), u(rng_), u(rng_)};
    }
    bool Step(bool push_right) {
        const double gravity = 9.8, cart_mass = 1.0, pole_mass = 0.1, half_length = 0.5;
        const double total_mass = cart_mass + pole_mass, pole_moment = pole_mass * half_length;
        const double force = push_right ? 10.0 : -10.0, tau = 0.02;
        double x = state_[0], x_dot = state_[1], theta = state_[2], theta_dot = state_[3];
        double cos_t = std::cos(theta), sin_t = std::sin(theta);
        double temp = (force + pole_moment * theta_dot * theta_dot * sin_t) / total_mass;
        double theta_acc = (gravity * sin_t - cos_t * temp) /
                           (half_length * (4.0 / 3.0 - pole_mass * cos_t * cos_t / total_mass));
        double x_acc = temp - pole_moment * theta_acc * cos_t / total_mass;
        x += tau * x_dot;
        x_dot += tau * x_acc;
        theta += tau * theta_dot;
        theta_dot += tau * theta_acc;
        state_ = {float(x), float(x_dot), float(theta), float(theta_dot)};
        return std::fabs(x) > 2.4 || std::fabs(theta) > 12.0 * 2.0 * kPi / 360.0;
    }
    std::vector<float> Positions() const { return {state_[0], state_[2]}; }

private:
    std::mt19937 rng_;
    std::vector<float> state_;
};

TEST_F(MemoryTest, M3_CartPolePositionsOnly) {
    EnvironmentSpec spec;
    spec.inputs = {{"positions", InputType::NUMERIC, 2}};
    spec.outputs = {{"push", OutputType::ACTION, {"left", "right"}}};
    spec.feedback = {{"reward", FeedbackType::REWARD, "push"}};
    // Not at the defaults (32 windows of 8 steps: best average 186): 64
    // windows of 4 steps (docs/AGENT_PLAN.md, Phase 9)
    AgentConfig config = Config();
    config.action_memory_window = 4;
    config.action_memory_batch = 64;
    Agent agent(spec, config);
    CartPole pole(301);
    std::vector<int> rewards;
    int solved_at = -1;
    double best = 0.0;
    for (int e = 0; e < 2000 && solved_at < 0; ++e) {
        pole.Reset();
        int total = 0;
        bool failed = false;
        while (total < 200 && !failed) {
            StepId step = agent.Observe({{"positions", pole.Positions()}});
            failed = pole.Step(agent.Act(step, "push") == "right");
            agent.Feedback(step, "reward", 1.0);
            ++total;
            Tick();
        }
        agent.EndEpisode(failed);
        rewards.push_back(total);
        if (rewards.size() >= 100) {
            double average = std::accumulate(rewards.end() - 100, rewards.end(), 0.0) / 100.0;
            best = std::max(best, average);
            if (average >= 195.0) solved_at = e + 1;
        }
    }
    Check(solved_at > 0, "M3: " + (solved_at > 0 ? "average >= 195 after " + std::to_string(solved_at) + " episodes"
                                                 : std::string("not solved in 2000 episodes")) +
                             ", best 100-episode average " + std::to_string(best));
}

// -------------------------------------------------------------------- M4
TEST_F(MemoryTest, M4_RunningCount) {
    EnvironmentSpec spec;
    spec.inputs = {{"event", InputType::CATEGORICAL, 0, {"a", "b", "c"}}};
    spec.outputs = {{"count", OutputType::VALUE}};
    spec.feedback = {{"target", FeedbackType::TARGET, "count"}};
    Agent agent(spec, Config());
    std::mt19937 rng(401);
    std::uniform_int_distribution<int> pick(0, 2);
    const std::string events[] = {"a", "b", "c"};
    const int episodes = 1000;
    double error = 0.0;
    int counted = 0;
    for (int e = 0; e < episodes; ++e) {
        int count = 0;
        for (int t = 0; t < 50; ++t) {
            int event = pick(rng);
            if (event == 0) ++count;
            StepId step = agent.Observe({{"event", events[event]}});
            Prediction p = agent.Predict(step, "count");
            if (e >= episodes - 100) {
                error += p.available ? std::fabs(p.values[0] - count) : count;
                ++counted;
            }
            agent.Feedback(step, "target", static_cast<double>(count));
            Tick();
        }
        agent.EndEpisode(false);
    }
    double mae = error / counted;
    Check(mae <= 0.5, "M4: mean absolute error " + std::to_string(mae));
}

// ------------------------------------------------------------ unit tests
TEST_F(MemoryTest, RemembersThePreviousStepAndResetsAtEpisodeStarts) {
    // y = x of the step before in the same episode, 0 at an episode's first
    // step: the first part needs memory, the second a reset state
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 1}};
    spec.outputs = {{"y", OutputType::VALUE}};
    spec.feedback = {{"target", FeedbackType::TARGET, "y"}};
    AgentConfig config;
    config.memory_units = 8;
    config.memory_window = 6;
    config.memory_batch = 8;
    config.memory_history = 200;
    Agent agent(spec, config);
    std::mt19937 rng(9);
    struct Errors {
        double first = 0.0, later = 0.0;
        int firsts = 0, laters = 0;
    };
    auto run = [&](Agent& a, std::mt19937& r, int steps, std::vector<float>* trace) {
        Errors errors;
        float previous = 0.0f;
        for (int i = 0; i < steps; ++i) {
            float x = std::normal_distribution<float>(0.0f, 1.0f)(r);
            StepId step = a.Observe({{"x", static_cast<double>(x)}});
            Prediction p = a.Predict(step, "y");
            float y = p.available ? p.values[0] : 0.0f;
            if (trace) trace->push_back(y);
            if (i % 10 == 0) {
                errors.first += std::fabs(y - previous);
                ++errors.firsts;
            } else {
                errors.later += std::fabs(y - previous);
                ++errors.laters;
            }
            a.Feedback(step, "target", static_cast<double>(previous));
            previous = x;
            if (i % 10 == 9) {
                a.EndEpisode(false);
                previous = 0.0f;
            }
            Tick();
        }
        return errors;
    };
    run(agent, rng, 6000, nullptr);
    Errors e = run(agent, rng, 1000, nullptr);
    // Without memory the error is about E|x| = 0.8 at the later steps
    EXPECT_LT(e.later / e.laters, 0.3);
    EXPECT_LT(e.first / e.firsts, 0.3);

    // After save and load the agent continues exactly as the saved one
    std::stringstream buffer;
    agent.Save(buffer);
    Agent loaded = Agent::Load(buffer);
    std::mt19937 rng_a = rng, rng_b = rng;
    std::vector<float> a, b;
    run(agent, rng_a, 200, &a);
    run(loaded, rng_b, 200, &b);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));
    EXPECT_EQ(agent.Stats().training_steps, loaded.Stats().training_steps);

    AgentConfig bad;
    bad.memory_units = 4;
    bad.memory_history = 10;
    bad.memory_window = 20;
    EXPECT_THROW(Agent(spec, bad), std::invalid_argument);
}

}  // namespace
}  // namespace agent
}  // namespace dpan
