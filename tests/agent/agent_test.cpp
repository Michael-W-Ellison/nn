// File: tests/agent/agent_test.cpp
#include "agent/agent.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <thread>

namespace dpan {
namespace agent {
namespace {

EnvironmentSpec ClassSpec(std::vector<std::string> classes = {}) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 2}};
    spec.outputs = {{"kind", OutputType::CLASS, std::move(classes)}};
    spec.feedback = {{"label", FeedbackType::LABEL, "kind"}};
    return spec;
}

/// Two well-separated clusters: class 0 around (-2, -2), class 1 around (2, 2)
std::vector<float> Point(std::mt19937& rng, int cls) {
    std::normal_distribution<float> noise(0.0f, 0.5f);
    float center = cls == 0 ? -2.0f : 2.0f;
    return {center + noise(rng), center + noise(rng)};
}

TEST(AgentTest, RejectsUnsupportedOrInvalidSpecs) {
    EnvironmentSpec action;
    action.inputs = {{"x", InputType::NUMERIC}};
    action.outputs = {{"move", OutputType::ACTION, {"left", "right"}}};
    action.feedback = {{"reward", FeedbackType::REWARD, "move"}};
    EXPECT_THROW(Agent{action}, std::invalid_argument);

    EnvironmentSpec next;
    next.inputs = {{"note", InputType::TEXT}};
    next.outputs = {{"next", OutputType::NEXT_OBSERVATION}};
    EXPECT_THROW(Agent{next}, std::invalid_argument);

    EnvironmentSpec outcome_only = ClassSpec();
    outcome_only.feedback = {{"ok", FeedbackType::OUTCOME, "kind"}};
    EXPECT_THROW(Agent{outcome_only}, std::invalid_argument);  // classes must be listed

    EnvironmentSpec invalid = ClassSpec();
    invalid.feedback.clear();
    EXPECT_THROW(Agent{invalid}, std::invalid_argument);

    AgentConfig config;
    config.batch_size = 0;
    EXPECT_THROW(Agent(ClassSpec(), config), std::invalid_argument);
}

TEST(AgentTest, StepsExpireAndNamesAreChecked) {
    AgentConfig config;
    config.feedback_timeout = 3;
    Agent agent(ClassSpec(), config);
    StepId first = agent.Observe({{"x", std::vector<float>{0, 0}}});
    EXPECT_EQ(1u, first);
    for (int i = 0; i < 3; ++i) agent.Observe({{"x", std::vector<float>{0, 0}}});

    EXPECT_THROW(agent.Predict(first, "kind"), std::out_of_range);
    EXPECT_THROW(agent.Feedback(first, "label", std::string("a")), std::out_of_range);
    EXPECT_THROW(agent.Predict(99, "kind"), std::out_of_range);
    EXPECT_THROW(agent.Predict(4, "nothing"), std::invalid_argument);
    EXPECT_THROW(agent.Feedback(4, "nothing", 1.0), std::invalid_argument);
    EXPECT_THROW(agent.Observe({{"y", 1.0}}), std::invalid_argument);
    EXPECT_EQ(3u, agent.Stats().pending_steps);
}

TEST(AgentTest, NothingIsPredictedBeforeInputsAreKnown) {
    EnvironmentSpec spec = ClassSpec();
    spec.inputs[0].size = 0;  // inferred
    Agent agent(spec);
    StepId step = agent.Observe({});
    EXPECT_FALSE(agent.Predict(step, "kind").available);
    agent.Feedback(step, "label", std::string("a"));
    EXPECT_EQ(1u, agent.Stats().ignored_feedback);
    EXPECT_FALSE(agent.Stats().network_ready);

    step = agent.Observe({{"x", std::vector<float>{1, 2, 3}}});
    EXPECT_TRUE(agent.Stats().network_ready);
    EXPECT_EQ(4u, agent.Stats().fused_size);
    EXPECT_FALSE(agent.Predict(step, "kind").available);  // no labels yet
}

TEST(AgentTest, LearnsClassesFromLabels) {
    Agent agent(ClassSpec());
    std::vector<std::string> logged;
    agent.SetLogger([&](LogLevel, const std::string& message) { logged.push_back(message); });

    std::mt19937 rng(1);
    int correct = 0;
    for (int i = 0; i < 1500; ++i) {
        int cls = i % 2;
        StepId step = agent.Observe({{"x", Point(rng, cls)}});
        Prediction p = agent.Predict(step, "kind");
        if (i >= 1000) correct += p.available && p.label == (cls ? "cat" : "dog");
        agent.Feedback(step, "label", std::string(cls ? "cat" : "dog"));
    }
    EXPECT_GE(correct, 490);

    StepId step = agent.Observe({{"x", std::vector<float>{2, 2}}});
    Prediction p = agent.Predict(step, "kind");
    ASSERT_TRUE(p.available);
    EXPECT_EQ((std::vector<std::string>{"dog", "cat"}), p.classes);
    EXPECT_EQ("cat", p.label);
    EXPECT_NEAR(1.0f, p.probabilities[0] + p.probabilities[1], 1e-5f);
    EXPECT_GT(p.confidence, 0.5f);
    EXPECT_GT(p.neural_weight, 0.0f);  // past the cold start

    // Numbers are class names for learned classes
    agent.Feedback(step, "label", 3.0);
    EXPECT_EQ(3u, agent.Stats().outputs[0].classes);
    EXPECT_EQ("3", agent.Predict(agent.Observe({{"x", std::vector<float>{0, 0}}}), "kind").classes[2]);

    bool saw_ready = false, saw_class = false;
    for (const auto& message : logged) {
        saw_ready |= message.find("Inputs ready") != std::string::npos;
        saw_class |= message.find("new class 'cat'") != std::string::npos;
    }
    EXPECT_TRUE(saw_ready);
    EXPECT_TRUE(saw_class);
}

TEST(AgentTest, DeclaredClassesAcceptNamesAndIndicesOnly) {
    Agent agent(ClassSpec({"no", "yes"}));
    StepId step = agent.Observe({{"x", std::vector<float>{0, 0}}});
    EXPECT_NO_THROW(agent.Feedback(step, "label", 1.0));
    EXPECT_NO_THROW(agent.Feedback(step, "label", std::string("no")));
    EXPECT_THROW(agent.Feedback(step, "label", 2.0), std::invalid_argument);
    EXPECT_THROW(agent.Feedback(step, "label", std::string("maybe")), std::invalid_argument);
    EXPECT_THROW(agent.Feedback(step, "label", std::vector<float>{1}), std::invalid_argument);
}

TEST(AgentTest, LearnsFromDelayedFeedback) {
    Agent agent(ClassSpec());
    std::mt19937 rng(2);
    // Labels arrive 50 steps late, in batches
    std::vector<std::pair<StepId, int>> waiting;
    for (int i = 0; i < 1200; ++i) {
        int cls = (i / 3) % 2;
        waiting.emplace_back(agent.Observe({{"x", Point(rng, cls)}}), cls);
        if (waiting.size() == 50) {
            for (auto it = waiting.rbegin(); it != waiting.rend(); ++it) {
                agent.Feedback(it->first, "label", std::string(it->second ? "b" : "a"));
            }
            waiting.clear();
        }
    }
    int correct = 0;
    for (int i = 0; i < 100; ++i) {
        int cls = i % 2;
        StepId step = agent.Observe({{"x", Point(rng, cls)}});
        correct += agent.Predict(step, "kind").label == (cls ? "b" : "a");
    }
    EXPECT_GE(correct, 95);
    EXPECT_EQ(1200u, agent.Stats().feedback);
    EXPECT_GT(agent.Stats().training_steps, 100u);
}

TEST(AgentTest, LearnsFromOutcomesOnly) {
    EnvironmentSpec spec = ClassSpec({"left", "right"});
    spec.feedback = {{"worked", FeedbackType::OUTCOME, "kind"}};
    Agent agent(spec);
    std::mt19937 rng(3);
    int correct = 0;
    for (int i = 0; i < 3000; ++i) {
        int cls = i % 2;
        StepId step = agent.Observe({{"x", Point(rng, cls)}});
        Prediction p = agent.Predict(step, "kind");
        ASSERT_TRUE(p.available);  // predicts before anything is learned
        bool right = p.label == (cls ? "right" : "left");
        if (i >= 2500) correct += right;
        agent.Feedback(step, "worked", right ? 1.0 : 0.0);
    }
    EXPECT_GE(correct, 450);
}

TEST(AgentTest, OutcomeRefersToTheReturnedPrediction) {
    EnvironmentSpec spec = ClassSpec({"a", "b"});
    spec.feedback.push_back({"ok", FeedbackType::OUTCOME, "kind"});
    Agent agent(spec);
    StepId step = agent.Observe({{"x", std::vector<float>{0, 0}}});
    agent.Feedback(step, "ok", 1.0);  // nothing learned, nothing predicted: ignored
    EXPECT_EQ(1u, agent.Stats().ignored_feedback);
    EXPECT_THROW(agent.Feedback(step, "ok", std::string("yes")), std::invalid_argument);
}

TEST(AgentTest, PredictsValues) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 1}};
    spec.outputs = {{"y", OutputType::VALUE, {}, 2}};
    spec.feedback = {{"target", FeedbackType::TARGET, "y"}};
    Agent agent(spec);
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int i = 0; i < 3000; ++i) {
        float x = u(rng);
        StepId step = agent.Observe({{"x", static_cast<double>(x)}});
        agent.Feedback(step, "target", std::vector<float>{100 + 10 * x, -x});
    }
    StepId step = agent.Observe({{"x", 0.5}});
    Prediction p = agent.Predict(step, "y");
    ASSERT_TRUE(p.available);
    ASSERT_EQ(2u, p.values.size());
    EXPECT_NEAR(105.0f, p.values[0], 0.5f);
    EXPECT_NEAR(-0.5f, p.values[1], 0.1f);
    EXPECT_GT(p.confidence, 0.5f);

    EXPECT_THROW(agent.Feedback(step, "target", 1.0), std::invalid_argument);
    EXPECT_THROW(agent.Feedback(step, "target", std::string("x")), std::invalid_argument);
}

TEST(AgentTest, PredictsNextObservationInInputUnits) {
    EnvironmentSpec spec;
    spec.inputs = {{"level", InputType::NUMERIC, 1}, {"note", InputType::TEXT}};
    spec.outputs = {{"next", OutputType::NEXT_OBSERVATION}};
    Agent agent(spec);
    // Alternates 1000, 2000, 1000, ...; episodes of 20 steps
    StepId step = 0;
    for (int i = 0; i < 2000; ++i) {
        if (i % 20 == 0) agent.EndEpisode();
        step = agent.Observe({{"level", i % 2 ? 2000.0 : 1000.0}, {"note", std::string(i % 2 ? "high" : "low")}});
    }
    Prediction p = agent.Predict(step, "next");  // last value was 2000
    ASSERT_TRUE(p.available);
    ASSERT_EQ(1u, p.values.size());
    EXPECT_NEAR(1000.0f, p.values[0], 100.0f);
    EXPECT_GT(agent.Stats().associations, 0u);
}

TEST(AgentTest, AnomalyScoresArePercentiles) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 2}};
    spec.outputs = {{"odd", OutputType::ANOMALY}};
    Agent agent(spec);
    std::mt19937 rng(5);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    for (int i = 0; i < 2000; ++i) {
        StepId step = agent.Observe({{"x", std::vector<float>{noise(rng), noise(rng)}}});
        float score = agent.Predict(step, "odd").score;
        ASSERT_GE(score, 0.0f);
        ASSERT_LE(score, 1.0f);
    }
    StepId outlier = agent.Observe({{"x", std::vector<float>{8, -8}}});
    Prediction p = agent.Predict(outlier, "odd");
    EXPECT_GT(p.score, 0.99f);
    EXPECT_FLOAT_EQ(1.0f, p.confidence);
    EXPECT_FLOAT_EQ(0.5f, p.neural_weight);
}

TEST(AgentTest, StatsDescribeTheAgent) {
    Agent agent(ClassSpec());
    std::mt19937 rng(6);
    for (int i = 0; i < 400; ++i) {
        StepId step = agent.Observe({{"x", Point(rng, i % 2)}});
        agent.Feedback(step, "label", static_cast<double>(i % 2));
    }
    AgentStats stats = agent.Stats();
    EXPECT_EQ(400u, stats.steps);
    EXPECT_EQ(400u, stats.feedback);
    EXPECT_EQ(50u, stats.training_steps);
    EXPECT_EQ(400u, stats.replay_size);
    EXPECT_GT(stats.patterns, 0u);
    EXPECT_EQ(3u, stats.fused_size);
    EXPECT_EQ(32u, stats.trunk_width);
    ASSERT_EQ(1u, stats.outputs.size());
    EXPECT_EQ(400u, stats.outputs[0].samples);
    EXPECT_EQ(2u, stats.outputs[0].classes);
    EXPECT_GT(stats.outputs[0].neural_quality, 0.9f);
    EXPECT_GT(stats.outputs[0].pattern_quality, 0.9f);
}

TEST(AgentTest, SameSeedSameResults) {
    auto run = [] {
        Agent agent(ClassSpec());
        std::mt19937 rng(7);
        std::vector<float> probabilities;
        for (int i = 0; i < 300; ++i) {
            StepId step = agent.Observe({{"x", Point(rng, i % 2)}});
            Prediction p = agent.Predict(step, "kind");
            if (p.available) probabilities.push_back(p.probabilities[0]);
            agent.Feedback(step, "label", static_cast<double>(i % 2));
        }
        return probabilities;
    };
    EXPECT_EQ(run(), run());
}

TEST(AgentTest, SafeToUseFromSeveralThreads) {
    Agent agent(ClassSpec());
    auto worker = [&agent](int seed) {
        std::mt19937 rng(seed);
        for (int i = 0; i < 300; ++i) {
            StepId step = agent.Observe({{"x", Point(rng, i % 2)}});
            agent.Predict(step, "kind");
            agent.Feedback(step, "label", static_cast<double>(i % 2));
            agent.Stats();
        }
    };
    std::thread a(worker, 1), b(worker, 2);
    a.join();
    b.join();
    EXPECT_EQ(600u, agent.Stats().steps);
}

}  // namespace
}  // namespace agent
}  // namespace dpan
