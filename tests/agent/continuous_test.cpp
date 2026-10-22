// File: tests/agent/continuous_test.cpp
//
// CONTINUOUS_ACTION outputs (Phase 8, docs/AGENT_PLAN.md): spec, acting,
// learning, predictions, statistics, rollback.

#include "agent/agent.hpp"
#include "agent/agent_testing.hpp"
#include "agent/spec_json.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>

namespace dpan {
namespace agent {
namespace {

OutputSpec Continuous(const std::string& name, std::vector<float> low, std::vector<float> high) {
    OutputSpec output;
    output.name = name;
    output.type = OutputType::CONTINUOUS_ACTION;
    output.size = low.empty() ? 1 : low.size();
    output.low = std::move(low);
    output.high = std::move(high);
    return output;
}

/// One input x; one action in [0, 10]; reward -(a - (5 + 3x))^2
EnvironmentSpec BanditSpec() {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 1}};
    spec.outputs = {Continuous("dose", {0.0f}, {10.0f})};
    spec.feedback = {{"reward", FeedbackType::REWARD, "dose"}};
    return spec;
}

TEST(ContinuousTest, SpecValidation) {
    EnvironmentSpec spec = BanditSpec();
    EXPECT_TRUE(ValidationErrors(spec).empty());

    spec.outputs[0].low = {1.0f};
    spec.outputs[0].high = {1.0f};
    EXPECT_FALSE(ValidationErrors(spec).empty());  // low == high
    spec.outputs[0].low = {0.0f, 0.0f};
    spec.outputs[0].high = {1.0f, 1.0f};
    EXPECT_FALSE(ValidationErrors(spec).empty());  // size 1, two bounds
    spec.outputs[0].low = {std::nanf("")};
    spec.outputs[0].high = {1.0f};
    EXPECT_FALSE(ValidationErrors(spec).empty());
    spec.outputs[0].low.clear();
    spec.outputs[0].high.clear();
    EXPECT_TRUE(ValidationErrors(spec).empty());  // [-1, 1]
    spec.outputs[0].size = 0;
    EXPECT_FALSE(ValidationErrors(spec).empty());

    EnvironmentSpec no_feedback = BanditSpec();
    no_feedback.feedback.clear();
    auto errors = ValidationErrors(no_feedback);
    ASSERT_EQ(1u, errors.size());
    EXPECT_NE(std::string::npos, errors[0].find("REWARD or OUTCOME"));

    EnvironmentSpec label = BanditSpec();
    label.feedback = {{"label", FeedbackType::LABEL, "dose"}};
    EXPECT_FALSE(ValidationErrors(label).empty());

    EnvironmentSpec misplaced = BanditSpec();
    misplaced.outputs.push_back({"kind", OutputType::CLASS, {"a"}});
    misplaced.outputs.back().low = {0.0f};
    misplaced.feedback.push_back({"label", FeedbackType::LABEL, "kind"});
    EXPECT_FALSE(ValidationErrors(misplaced).empty());
}

TEST(ContinuousTest, SpecJson) {
    EnvironmentSpec spec = SpecFromJson(R"({
      "inputs": [{"name": "x", "type": "numeric"}],
      "feedback": [{"name": "reward", "type": "reward", "output": "arm"}],
      "outputs": [{"name": "arm", "type": "continuous_action", "size": 2, "low": [-1, 0], "high": [1, 5]}]})");
    ASSERT_EQ(1u, spec.outputs.size());
    EXPECT_EQ(OutputType::CONTINUOUS_ACTION, spec.outputs[0].type);
    EXPECT_EQ(2u, spec.outputs[0].size);
    EXPECT_EQ((std::vector<float>{-1.0f, 0.0f}), spec.outputs[0].low);
    EXPECT_EQ((std::vector<float>{1.0f, 5.0f}), spec.outputs[0].high);
    EXPECT_NO_THROW(Validate(spec));
    EXPECT_EQ(SpecToJson(spec), SpecToJson(SpecFromJson(SpecToJson(spec))));
    EXPECT_THROW(SpecFromJson(R"({"outputs": [{"name": "a", "type": "continuous_action", "low": ["x"]}]})"),
                 std::invalid_argument);
}

TEST(ContinuousTest, ActionsStayInBoundsAndAreRemembered) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 1}};
    spec.outputs = {Continuous("arm", {-3.0f, 100.0f}, {-1.0f, 200.0f})};
    spec.feedback = {{"reward", FeedbackType::REWARD, "arm"}};
    AgentConfig config;
    config.continuous_random_steps = 50;
    Agent agent(spec, config);
    EXPECT_EQ(1u, agent.Config().train_every);  // action settings

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int i = 0; i < 300; ++i) {
        StepId step = agent.Observe({{"x", static_cast<double>(u(rng))}});
        auto action = agent.ActContinuous(step, "arm", i % 3 != 0);
        ASSERT_EQ(2u, action.size());
        EXPECT_GE(action[0], -3.0f);
        EXPECT_LE(action[0], -1.0f);
        EXPECT_GE(action[1], 100.0f);
        EXPECT_LE(action[1], 200.0f);
        EXPECT_EQ(action, agent.ActContinuous(step, "arm"));  // same step: same action
        agent.Feedback(step, "reward", -std::fabs(action[0] + 2.0f));
        if (i % 10 == 9) agent.EndEpisode(true);
    }
    AgentStats stats = agent.Stats();
    EXPECT_FLOAT_EQ(0.1f, stats.outputs[0].exploration);  // past the random start
    EXPECT_EQ(OutputType::CONTINUOUS_ACTION, stats.outputs[0].type);
    EXPECT_GT(stats.outputs[0].samples, 250u);
    EXPECT_FLOAT_EQ(1.0f, stats.outputs[0].neural_weight);

    StepId step = agent.Observe({{"x", 0.5}});
    Prediction p = agent.Predict(step, "arm");
    EXPECT_TRUE(p.available);
    EXPECT_EQ(OutputType::CONTINUOUS_ACTION, p.type);
    ASSERT_EQ(2u, p.values.size());
    EXPECT_GE(p.values[0], -3.0f);
    EXPECT_LE(p.values[1], 200.0f);
    EXPECT_FLOAT_EQ(0.9f, p.confidence);
    EXPECT_TRUE(std::isfinite(p.score));
    // The greedy action is the prediction
    EXPECT_EQ(p.values, agent.ActContinuous(step, "arm", false));
}

TEST(ContinuousTest, WrongCallsAreRejected) {
    EnvironmentSpec spec = BanditSpec();
    spec.outputs.push_back({"move", OutputType::ACTION, {"left", "right"}});
    spec.feedback.push_back({"bonus", FeedbackType::REWARD, "move"});
    spec.feedback.push_back({"ok", FeedbackType::OUTCOME, "dose"});
    Agent agent(spec);
    StepId step = agent.Observe({{"x", 0.0}});
    EXPECT_THROW(agent.Act(step, "dose"), std::invalid_argument);
    EXPECT_THROW(agent.ActContinuous(step, "move"), std::invalid_argument);
    EXPECT_THROW(agent.ActContinuous(step, "nothing"), std::invalid_argument);
    EXPECT_THROW(agent.ActContinuous(step + 5, "dose"), std::out_of_range);
    EXPECT_THROW(agent.Feedback(step, "reward", 1.0), std::logic_error);  // before ActContinuous
    agent.ActContinuous(step, "dose");
    EXPECT_NO_THROW(agent.Feedback(step, "reward", 1.0));
    EXPECT_NO_THROW(agent.Feedback(step, "ok", 0.0));  // OUTCOME: reward -1
    EXPECT_THROW(agent.Feedback(step, "bonus", 1.0), std::logic_error);  // no discrete action yet
}

TEST(ContinuousTest, LearnsTheBestDoseOfABandit) {
    // One-step episodes: the best action is 5 + 3x for x in [-1, 1]
    AgentConfig config;
    config.continuous_random_steps = 500;
    Agent agent(BanditSpec(), config);
    std::mt19937 rng(2);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int i = 0; i < 4000; ++i) {
        float x = u(rng);
        StepId step = agent.Observe({{"x", static_cast<double>(x)}});
        float dose = agent.ActContinuous(step, "dose")[0];
        float best = 5.0f + 3.0f * x;
        agent.Feedback(step, "reward", -(dose - best) * (dose - best));
        agent.EndEpisode(true);
    }
    for (float x : {-0.8f, 0.0f, 0.7f}) {
        StepId step = agent.Observe({{"x", static_cast<double>(x)}});
        float dose = agent.ActContinuous(step, "dose", false)[0];
        EXPECT_NEAR(5.0f + 3.0f * x, dose, 0.6f) << "x = " << x;
        agent.EndEpisode(false);
    }
}

TEST(ContinuousTest, NonFiniteWeightsAreRolledBack) {
    Agent agent(BanditSpec());
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto run = [&](int count) {
        for (int i = 0; i < count; ++i) {
            StepId step = agent.Observe({{"x", static_cast<double>(u(rng))}});
            agent.ActContinuous(step, "dose");
            agent.Feedback(step, "reward", -1.0);
            agent.EndEpisode(true);
        }
    };
    run(1100);  // past a snapshot
    ASSERT_TRUE(AgentTestAccess::InjectNaN(agent));
    run(5);
    EXPECT_EQ(1u, agent.Stats().rollbacks);
    StepId step = agent.Observe({{"x", 0.0}});
    auto dose = agent.ActContinuous(step, "dose", false);
    EXPECT_TRUE(std::isfinite(dose[0]));
}

}  // namespace
}  // namespace agent
}  // namespace dpan
