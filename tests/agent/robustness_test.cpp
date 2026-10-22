// File: tests/agent/robustness_test.cpp
//
// Phase 4 (docs/AGENT_PLAN.md): drift recovery, memory budget over a long
// stream, rollback of injected NaN values, capacity growth, hooks.

#include "agent/agent.hpp"
#include "agent/agent_testing.hpp"
#include "core/clock.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <unistd.h>

namespace dpan {
namespace agent {
namespace {

/// Resident memory of this process (Linux), 0 if unknown
size_t ResidentBytes() {
    std::ifstream statm("/proc/self/statm");
    size_t pages = 0, resident = 0;
    if (!(statm >> pages >> resident)) return 0;
    return resident * static_cast<size_t>(sysconf(_SC_PAGESIZE));
}

// Environment A of the reference suite: 3 classes, labels rotate at `drift`
struct DriftingClasses {
    std::mt19937 rng{11};
    std::normal_distribution<float> noise{0.0f, 0.6f};
    std::uniform_int_distribution<int> pick_class{0, 2}, pick_region{0, 4};
    std::uniform_real_distribution<float> u{0.0f, 1.0f};
    const std::vector<std::vector<float>> centers{{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}};
    const std::vector<std::string> regions{"north", "south", "east", "west", "center"};

    static EnvironmentSpec Spec() {
        EnvironmentSpec spec;
        spec.inputs = {{"x", InputType::NUMERIC, 4},
                       {"region", InputType::CATEGORICAL, 0, {"north", "south", "east", "west", "center"}}};
        spec.outputs = {{"kind", OutputType::CLASS}};
        spec.feedback = {{"label", FeedbackType::LABEL, "kind"}};
        return spec;
    }

    /// Observation and label of sample i (labels rotate from sample `drift`)
    std::pair<Observation, int> Next(int i, int drift) {
        int cluster = pick_class(rng);
        std::vector<float> x(4);
        for (int d = 0; d < 4; ++d) x[d] = centers[cluster][d] + noise(rng);
        std::string region = u(rng) < 0.6f ? regions[cluster] : regions[pick_region(rng)];
        return {{{"x", x}, {"region", region}}, i < drift ? cluster : (cluster + 1) % 3};
    }
};

// ============================================================================
// Drift recovery (done-criterion: recovers within 1,000 samples)
// ============================================================================

TEST(RobustnessTest, NetworkRecoversFromAbruptDriftWithin1000Samples) {
    Agent agent(DriftingClasses::Spec());
    std::vector<std::string> warnings;
    agent.SetLogger([&](LogLevel level, const std::string& message) {
        if (level == LogLevel::WARNING) warnings.push_back(message);
    });
    DriftingClasses env;
    const int drift = 6000;
    std::vector<int> correct;
    float neural_quality_after = 0.0f;
    for (int i = 0; i < 8000; ++i) {
        auto [observation, label] = env.Next(i, drift);
        StepId step = agent.Observe(observation);
        correct.push_back(agent.Predict(step, "kind").label == std::to_string(label));
        agent.Feedback(step, "label", static_cast<double>(label));
        if (i == drift - 1) {
            EXPECT_EQ(0u, agent.Stats().drift_events) << "false alarm before the drift";
        }
        if (i == drift + 999) neural_quality_after = agent.Stats().outputs[0].neural_quality;
    }
    AgentStats stats = agent.Stats();
    double recovered = std::accumulate(correct.begin() + drift + 800, correct.begin() + drift + 1000, 0) / 200.0;
    std::cout << "drift: " << stats.drift_events << " detection(s), network accuracy 1000 samples after "
              << neural_quality_after << ", combined accuracy 800-1000 after " << recovered << "\n";
    EXPECT_EQ(1u, stats.drift_events);
    EXPECT_FALSE(stats.recovering);
    EXPECT_GE(neural_quality_after, 0.9f);  // the network itself, not only the pattern side
    EXPECT_GE(recovered, 0.9);
    ASSERT_FALSE(warnings.empty());
    EXPECT_NE(std::string::npos, warnings[0].find("Drift detected on output 'kind'"));
}

TEST(RobustnessTest, RegressionRecoversFromAbruptDrift) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 2}};
    spec.outputs = {{"y", OutputType::VALUE}};
    spec.feedback = {{"target", FeedbackType::TARGET, "y"}};
    Agent agent(spec);
    std::mt19937 rng(14);
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    const int drift = 5000;
    float before = 0.0f, after = 0.0f;
    for (int i = 0; i < drift + 1000; ++i) {
        float a = u(rng), b = u(rng);
        float y = (i < drift ? 1.0f : -1.0f) * (a + 0.5f * b) + noise(rng);  // the relation flips
        StepId step = agent.Observe({{"x", std::vector<float>{a, b}}});
        agent.Feedback(step, "target", static_cast<double>(y));
        if (i == drift - 1) before = agent.Stats().outputs[0].neural_quality;
        if (i == drift + 999) after = agent.Stats().outputs[0].neural_quality;
    }
    std::cout << "regression drift: network error (standardized) " << before << " before, " << after
              << " 1000 samples after\n";
    EXPECT_GE(agent.Stats().drift_events, 1u);
    EXPECT_LE(after, std::max(0.05f, 2.0f * before));
}

// ============================================================================
// Rollback (done-criterion: injected NaN values are rolled back)
// ============================================================================

TEST(RobustnessTest, InjectedNaNIsRolledBack) {
    Agent agent(DriftingClasses::Spec());
    DriftingClasses env;
    auto run = [&](int count) {
        int right = 0;
        for (int i = 0; i < count; ++i) {
            auto [observation, label] = env.Next(i, 1 << 30);
            StepId step = agent.Observe(observation);
            Prediction p = agent.Predict(step, "kind");
            for (float probability : p.probabilities) EXPECT_FALSE(std::isnan(probability));
            right += p.label == std::to_string(label);
            agent.Feedback(step, "label", static_cast<double>(label));
        }
        return right;
    };
    run(2000);
    EXPECT_TRUE(AgentTestAccess::InjectNaN(agent));
    run(8);  // one training step finds the NaN
    AgentStats stats = agent.Stats();
    EXPECT_EQ(1u, stats.rollbacks);
    EXPECT_GE(run(500), 480);  // the restored network still works
    EXPECT_EQ(1u, agent.Stats().rollbacks);
}

TEST(RobustnessTest, InjectedNaNInAutoencoderIsRolledBack) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 3}};
    spec.outputs = {{"odd", OutputType::ANOMALY}};
    Agent agent(spec);
    std::mt19937 rng(15);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int i = 0; i < 500; ++i) agent.Observe({{"x", std::vector<float>{normal(rng), normal(rng), normal(rng)}}});
    EXPECT_TRUE(AgentTestAccess::InjectNaN(agent));
    StepId step = 0;
    for (int i = 0; i < 16; ++i) step = agent.Observe({{"x", std::vector<float>{normal(rng), normal(rng), normal(rng)}}});
    EXPECT_EQ(1u, agent.Stats().rollbacks);
    EXPECT_FALSE(std::isnan(agent.Predict(step, "odd").score));
}

// ============================================================================
// Capacity growth
// ============================================================================

TEST(RobustnessTest, WidensTheTrunkWhenTheNetworkTrailsThePatterns) {
    // Fine stripes: class = floor(10 x) mod 2. Nearest patterns resolve them
    // easily; a narrow, briefly trained network does not.
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 1}};
    spec.outputs = {{"stripe", OutputType::CLASS, {"even", "odd"}}};
    spec.feedback = {{"label", FeedbackType::LABEL, "stripe"}};
    AgentConfig config;
    config.capacity_min_samples = 1000;
    config.plateau_interval = 20;
    config.plateau_patience = 2;
    config.max_trunk_width = 128;
    Agent agent(spec, config);
    std::vector<std::string> info;
    agent.SetLogger([&](LogLevel level, const std::string& message) {
        if (level == LogLevel::INFO) info.push_back(message);
    });
    std::mt19937 rng(16);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (int i = 0; i < 6000; ++i) {
        float x = u(rng);
        StepId step = agent.Observe({{"x", static_cast<double>(x)}});
        agent.Feedback(step, "label", static_cast<double>(static_cast<int>(x * 10) % 2));
    }
    AgentStats stats = agent.Stats();
    std::cout << "capacity: " << stats.widenings << " widening(s), trunk width " << stats.trunk_width << "\n";
    EXPECT_GE(stats.widenings, 1u);
    EXPECT_GT(stats.trunk_width, 32u);
    EXPECT_LE(stats.trunk_width, 128u);
    bool logged = false;
    for (const auto& message : info) logged |= message.find("Trunk widened") != std::string::npos;
    EXPECT_TRUE(logged);
}

// ============================================================================
// Memory budget (done-criterion: within budget over 1 million steps)
// ============================================================================

TEST(RobustnessTest, MemoryStaysWithinBudgetOverAMillionSteps) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 4}, {"user", InputType::CATEGORICAL}};
    spec.outputs = {{"kind", OutputType::CLASS}, {"next", OutputType::NEXT_OBSERVATION}, {"odd", OutputType::ANOMALY}};
    spec.feedback = {{"label", FeedbackType::LABEL, "kind"}};
    AgentConfig config;
    config.memory_budget_bytes = 16u << 20;
    Agent agent(spec, config);

    std::mt19937 rng(17);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_int_distribution<int> user(0, 1000000);  // new categories all the time
    size_t peak = 0, resident_early = 0, resident_late = 0;
    const int total = 1000000;
    for (int i = 1; i <= total; ++i) {
        std::vector<float> x = {normal(rng), normal(rng), normal(rng), normal(rng)};
        StepId step = agent.Observe({{"x", x}, {"user", "u" + std::to_string(user(rng))}});
        agent.Feedback(step, "label", x[0] > 0 ? std::string("up") : std::string("down"));
        if (i % 10000 == 0) {
            AgentStats stats = agent.Stats();
            peak = std::max(peak, stats.memory_bytes);
            ASSERT_LE(stats.memory_bytes, config.memory_budget_bytes) << "at step " << i;
            if (i == 200000) resident_early = ResidentBytes();
        }
    }
    resident_late = ResidentBytes();
    AgentStats stats = agent.Stats();
    std::cout << "memory: peak estimate " << peak / 1024 << " KiB of " << config.memory_budget_bytes / 1024
              << " KiB; resident " << resident_early / 1024 << " KiB at step 200000, " << resident_late / 1024
              << " KiB at step 1000000; patterns " << stats.patterns << " (" << stats.patterns_pruned
              << " pruned), replay " << stats.replay_size << ", associations " << stats.associations << "\n";
    EXPECT_GT(stats.patterns_pruned, 0u);
    if (resident_early > 0) {
        // Nothing outside the estimate keeps growing
        EXPECT_LT(resident_late, resident_early + (8u << 20));
    }
}

}  // namespace
}  // namespace agent
}  // namespace dpan
