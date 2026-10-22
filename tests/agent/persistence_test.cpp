// File: tests/agent/persistence_test.cpp
//
// Phase 5 (docs/AGENT_PLAN.md): after save and load, an agent continues
// exactly as the saved one does.

#include "agent/agent.hpp"
#include "../test_temp_path.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>

namespace dpan {
namespace agent {
namespace {

/// Everything an agent returned, compared bit for bit
struct Trace {
    std::vector<float> numbers;
    std::vector<std::string> words;

    void Add(float x) { numbers.push_back(x); }
    void Add(const std::string& s) { words.push_back(s); }
    void Add(const Prediction& p) {
        Add(p.available ? 1.0f : 0.0f);
        Add(p.label);
        for (float v : p.probabilities) Add(v);
        for (float v : p.values) Add(v);
        Add(p.score);
        Add(p.confidence);
        Add(p.neural_weight);
    }
    void Add(const AgentStats& s) {
        for (uint64_t v : {s.steps, s.episodes, s.feedback, s.ignored_feedback, s.training_steps, s.drift_events,
                           s.rollbacks, s.widenings, s.patterns_pruned}) {
            Add(static_cast<float>(v));
        }
        for (size_t v : {s.pending_steps, s.replay_size, s.patterns, s.associations, s.trunk_width}) {
            Add(static_cast<float>(v));
        }
        Add(s.learning_rate);
        for (const auto& o : s.outputs) {
            Add(static_cast<float>(o.samples));
            Add(o.neural_quality);
            Add(o.pattern_quality);
            Add(o.neural_weight);
            Add(o.exploration);
        }
    }
};

void ExpectIdentical(const Trace& a, const Trace& b) {
    ASSERT_EQ(a.words, b.words);
    ASSERT_EQ(a.numbers.size(), b.numbers.size());
    // Bitwise, so NaN equals NaN and -0 differs from 0
    EXPECT_EQ(0, std::memcmp(a.numbers.data(), b.numbers.data(), a.numbers.size() * sizeof(float)))
        << "traces differ";
    for (size_t i = 0; i < a.numbers.size(); ++i) {
        if (std::memcmp(&a.numbers[i], &b.numbers[i], sizeof(float)) != 0) {
            ADD_FAILURE() << "first difference at value " << i << ": " << a.numbers[i] << " vs " << b.numbers[i];
            break;
        }
    }
}

/// Save `agent` to memory and load a copy
Agent Copy(const Agent& agent) {
    std::stringstream buffer;
    agent.Save(buffer);
    return Agent::Load(buffer);
}

// ============================================================================
// Every input and output type, delayed labels, outcomes
// ============================================================================

EnvironmentSpec MixedSpec() {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC},                  // size inferred
                   {"color", InputType::CATEGORICAL, 0, {"red", "green", "blue"}},
                   {"user", InputType::CATEGORICAL},           // open vocabulary
                   {"note", InputType::TEXT},
                   {"load", InputType::TIME_SERIES},
                   {"cam", InputType::IMAGE, 0, {}, 0, 0, 1}};
    spec.outputs = {{"kind", OutputType::CLASS},               // classes learned from labels
                    {"price", OutputType::VALUE, {}, 2},
                    {"next", OutputType::NEXT_OBSERVATION},
                    {"odd", OutputType::ANOMALY}};
    spec.feedback = {{"label", FeedbackType::LABEL, "kind"},
                     {"right", FeedbackType::OUTCOME, "kind"},
                     {"target", FeedbackType::TARGET, "price"}};
    return spec;
}

struct MixedStream {
    struct Step {
        Observation observation;
        std::string label;
        std::vector<float> target;
    };
    std::vector<Step> steps;

    explicit MixedStream(int count) {
        std::mt19937 rng(21);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::uniform_int_distribution<int> pick(0, 2), user(0, 40);
        const std::vector<std::string> colors = {"red", "green", "blue"};
        const std::vector<std::string> words = {"alpha", "beta", "gamma", "delta"};
        for (int i = 0; i < count; ++i) {
            int c = pick(rng);
            std::vector<float> x = {c + 0.3f * normal(rng), normal(rng)};
            Image image{4, 4, 1, std::vector<float>(16)};
            for (auto& p : image.pixels) p = 100.0f + 50.0f * c + 10.0f * normal(rng);
            Step step;
            step.observation = {{"x", x},
                                {"color", colors[c]},
                                {"user", "u" + std::to_string(user(rng))},
                                {"note", words[c] + " " + words[(c + 1) % 4]},
                                {"load", static_cast<double>(std::sin(i * 0.3) + 0.1 * normal(rng))},
                                {"cam", image}};
            if (i % 11 == 0) step.observation.erase("note");  // missing channels
            step.label = "k" + std::to_string(c);
            step.target = {x[0] * 2.0f + x[1], x[1] * x[1]};
            steps.push_back(std::move(step));
        }
    }
};

/// Runs steps [begin, end): observe, predict all outputs, delayed feedback
void RunMixed(Agent& agent, const MixedStream& stream, size_t begin, size_t end, Trace& trace) {
    for (size_t i = begin; i < end; ++i) {
        const auto& s = stream.steps[i];
        StepId step = agent.Observe(s.observation);
        for (const char* output : {"kind", "price", "next", "odd"}) trace.Add(agent.Predict(step, output));
        // Labels and targets for step i arrive 7 steps later
        if (step > 7) {
            StepId old = step - 7;
            const auto& late = stream.steps[old - 1];
            agent.Feedback(old, "label", late.label);
            agent.Feedback(old, "target", late.target);
        }
        if (i % 5 == 0) {
            agent.Feedback(step, "right", agent.Predict(step, "kind").label == s.label ? 1.0 : 0.0);
        }
    }
    trace.Add(agent.Stats());
}

AgentConfig FrequentPlateauChecks() {
    // Learning-rate drops on both sides of the save point, so the
    // schedule's state matters
    AgentConfig config;
    config.plateau_interval = 5;
    config.plateau_patience = 3;
    config.min_learning_rate = 1e-6f;
    return config;
}

TEST(PersistenceTest, MixedAgentContinuesExactly) {
    MixedStream stream(5000);
    Agent original(MixedSpec(), FrequentPlateauChecks());
    Trace before;
    // Save once the learning rate has dropped (at least 1500 steps in)
    size_t save_at = 1500;
    RunMixed(original, stream, 0, save_at, before);
    while (original.Stats().learning_rate >= original.Config().learning_rate && save_at < 3500) {
        RunMixed(original, stream, save_at, save_at + 1, before);
        ++save_at;
    }
    float rate_at_save = original.Stats().learning_rate;
    ASSERT_LT(rate_at_save, original.Config().learning_rate) << "no learning-rate drop yet";
    ASSERT_GT(original.Stats().training_steps, 100u);
    ASSERT_GT(original.Stats().pending_steps, 7u);

    Agent restored = Copy(original);
    Trace a, b;
    RunMixed(original, stream, save_at, save_at + 1500, a);
    RunMixed(restored, stream, save_at, save_at + 1500, b);
    ExpectIdentical(a, b);
    EXPECT_LT(original.Stats().learning_rate, rate_at_save) << "no learning-rate drop after the save";
}

TEST(PersistenceTest, SavedBeforeInputSizesAreKnown) {
    MixedStream stream(600);
    Agent original(MixedSpec());
    original.Observe({{"color", std::string("red")}});  // "x" and "load" sizes still unknown
    ASSERT_FALSE(original.Stats().network_ready);
    Agent restored = Copy(original);
    Trace a, b;
    RunMixed(original, stream, 0, 600, a);
    RunMixed(restored, stream, 0, 600, b);
    ExpectIdentical(a, b);
}

// ============================================================================
// Saved during drift recovery
// ============================================================================

TEST(PersistenceTest, SavedWhileRecoveringFromDrift) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 3}};
    spec.outputs = {{"kind", OutputType::CLASS, {"a", "b", "c"}}};
    spec.feedback = {{"label", FeedbackType::LABEL, "kind"}};

    std::mt19937 rng(5);
    std::normal_distribution<float> noise(0.0f, 0.4f);
    std::uniform_int_distribution<int> pick(0, 2);
    std::vector<std::pair<std::vector<float>, int>> data;
    for (int i = 0; i < 4500; ++i) {
        int c = pick(rng);
        std::vector<float> x(3, 0.0f);
        x[c] = 2.0f;
        for (auto& v : x) v += noise(rng);
        data.emplace_back(x, i < 2500 ? c : (c + 1) % 3);
    }
    auto run = [&](Agent& agent, size_t begin, size_t end, Trace& trace) {
        for (size_t i = begin; i < end; ++i) {
            StepId step = agent.Observe({{"x", data[i].first}});
            trace.Add(agent.Predict(step, "kind"));
            agent.Feedback(step, "label", static_cast<double>(data[i].second));
        }
        trace.Add(agent.Stats());
    };

    Agent original(spec);
    Trace before;
    size_t save_at = 2500;
    run(original, 0, save_at, before);
    while (!original.Stats().recovering && save_at < 3000) {
        run(original, save_at, save_at + 1, before);
        ++save_at;
    }
    ASSERT_TRUE(original.Stats().recovering) << "no drift detected";

    Agent restored = Copy(original);
    Trace a, b;
    run(original, save_at, 4500, a);
    run(restored, save_at, 4500, b);
    ExpectIdentical(a, b);
}

// ============================================================================
// Reinforcement learning, saved in the middle of an episode
// ============================================================================

struct Grid {
    std::mt19937 rng{31};
    int x = 0, y = 0, t = 0;
    void Reset() {
        std::uniform_int_distribution<int> cell(0, 3);
        x = cell(rng);
        y = cell(rng);
        t = 0;
    }
};

/// Runs `steps` agent steps; returns false once the goal or limit ends an episode
void RunGrid(Agent& agent, Grid& grid, int steps, Trace& trace) {
    for (int i = 0; i < steps; ++i) {
        StepId step = agent.Observe({{"position", std::vector<float>{float(grid.x), float(grid.y)}}});
        std::string move = agent.Act(step, "move");
        trace.Add(move);
        trace.Add(agent.Predict(step, "move"));
        if (move == "up") grid.y = std::min(4, grid.y + 1);
        else if (move == "down") grid.y = std::max(0, grid.y - 1);
        else if (move == "left") grid.x = std::max(0, grid.x - 1);
        else grid.x = std::min(4, grid.x + 1);
        bool goal = grid.x == 4 && grid.y == 4;
        agent.Feedback(step, "reward", goal ? 1.0 : -0.04);
        if (goal || ++grid.t >= 30) {
            agent.EndEpisode(goal);
            grid.Reset();
        }
    }
    trace.Add(agent.Stats());
}

TEST(PersistenceTest, ReinforcementLearnerContinuesExactlyMidEpisode) {
    EnvironmentSpec spec;
    spec.inputs = {{"position", InputType::NUMERIC, 2}};
    spec.outputs = {{"move", OutputType::ACTION, {"up", "down", "left", "right"}}};
    spec.feedback = {{"reward", FeedbackType::REWARD, "move"}};
    Agent original(spec);
    Grid grid;
    grid.Reset();
    Trace before;
    RunGrid(original, grid, 3007, before);
    ASSERT_GT(grid.t, 0) << "test should save in the middle of an episode";

    Agent restored = Copy(original);
    Grid grid_copy = grid;
    Trace a, b;
    RunGrid(original, grid, 3000, a);
    RunGrid(restored, grid_copy, 3000, b);
    ExpectIdentical(a, b);
}

// ============================================================================
// Files and errors
// ============================================================================

TEST(PersistenceTest, SavesToFileAndRejectsBadCheckpoints) {
    MixedStream stream(400);
    Agent original(MixedSpec());
    Trace before;
    RunMixed(original, stream, 0, 400, before);

    std::string path = test::UniqueTempPath("dpan_agent_checkpoint") + ".dpag";
    ASSERT_TRUE(original.Save(path));
    Agent loaded = Agent::Load(path);
    EXPECT_EQ(original.Stats().steps, loaded.Stats().steps);
    EXPECT_EQ(original.Spec().outputs.size(), loaded.Spec().outputs.size());
    EXPECT_EQ(original.Config().seed, loaded.Config().seed);

    std::string bytes;
    {
        std::ifstream file(path, std::ios::binary);
        bytes.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    }
    auto load_bytes = [](const std::string& data) {
        std::istringstream in(data);
        return Agent::Load(in);
    };
    std::string bad_magic = bytes;
    bad_magic[0] = 'X';
    EXPECT_THROW(load_bytes(bad_magic), std::runtime_error);
    std::string bad_version = bytes;
    bad_version[4] = 99;
    try {
        load_bytes(bad_version);
        ADD_FAILURE() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string::npos, std::string(e.what()).find("version 99"));
    }
    for (size_t cut : {size_t{10}, bytes.size() / 2, bytes.size() - 1}) {
        EXPECT_THROW(load_bytes(bytes.substr(0, cut)), std::runtime_error) << cut;
    }
    EXPECT_THROW(Agent::Load(path + ".missing"), std::runtime_error);
    EXPECT_FALSE(original.Save("/nonexistent-directory/agent.dpag"));
    std::remove(path.c_str());
}

}  // namespace
}  // namespace agent
}  // namespace dpan
