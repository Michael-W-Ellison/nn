// File: tests/agent/scenarios_test.cpp
//
// Scenario tests: uses of the drop-in agent beyond the reference
// environments, each with the default AgentConfig, a pass criterion fixed
// before the first run, and seeds 1-3. They track how well the agent adapts
// to different uses (docs/AGENT_PLAN.md, "Scenario tests"). They run only
// with DPAN_SCENARIOS set (otherwise they are reported as skipped), so the
// regular suite does not depend on them:
//   DPAN_SCENARIOS=1 ctest --test-dir build -L scenarios
//
//   S1  text classification with misspellings         accuracy >= 0.90 (last 1000 of 3000)
//   S2  3 regression outputs, 10% missing inputs        R^2 >= 0.7 for every output
//   S3  text + numeric, labels 30% / outcomes 70%       accuracy >= 0.85, anomaly AUC >= 0.8
//       (plan environment H) with an ANOMALY output
//   S4  regression whose target changes at step 4000    MSE 1000-2000 after <= 2 x before
//   S5  250 open-vocabulary user IDs, imbalanced        mean per-class recall >= 0.8
//   S6  contextual bandit, outcomes only                greedy best action in 4/4 contexts
//   S7  continuous 2-D reaching (absolute positions)    mean final distance <= 0.1 after 300 episodes
//   S8  image regression (blob position)                R^2 >= 0.8
//   S9  two related time series                         MSE <= 0.5 x last value, both series
//   S10 inputs at scale 1e6 with offsets 1e7            accuracy >= 0.90

#include "agent/agent.hpp"
#include "core/clock.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace dpan {
namespace agent {
namespace {

std::shared_ptr<ManualClock> g_clock;
void Tick() { g_clock->Advance(std::chrono::seconds(1)); }

struct Result {
    bool pass;
    std::string metrics;
};

double AreaUnderRoc(const std::vector<float>& scores, const std::vector<int>& labels) {
    std::vector<size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return scores[a] < scores[b]; });
    std::vector<double> rank(scores.size());
    for (size_t i = 0; i < order.size();) {
        size_t j = i;
        while (j < order.size() && scores[order[j]] == scores[order[i]]) ++j;
        for (size_t k = i; k < j; ++k) rank[order[k]] = (i + j + 1) / 2.0;
        i = j;
    }
    double positives = 0, rank_sum = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (labels[i]) {
            positives++;
            rank_sum += rank[i];
        }
    }
    double negatives = labels.size() - positives;
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives);
}

std::string Fmt(double x) {
    std::ostringstream s;
    s.precision(4);
    s << x;
    return s.str();
}

// S1: text classification with keywords, filler and misspellings
Result S1(uint32_t seed) {
    EnvironmentSpec spec;
    spec.inputs = {{"message", InputType::TEXT}};
    spec.outputs = {{"topic", OutputType::CLASS}};
    spec.feedback = {{"label", FeedbackType::LABEL, "topic"}};
    Agent agent(spec);
    const std::vector<std::vector<std::string>> keywords = {
        {"invoice", "payment", "refund", "charge", "billing", "price"},
        {"crash", "error", "bug", "freeze", "install", "update"},
        {"delivery", "shipping", "package", "courier", "tracking", "arrived"}};
    const std::vector<std::string> filler = {"the", "my", "please", "help", "today", "with", "is", "not",
                                             "again", "thanks", "hello", "order", "account", "it", "was"};
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> cls(0, 2), kw(0, 5), fill(0, 14), count(3, 7);
    std::uniform_real_distribution<float> u(0, 1);
    auto misspell = [&](std::string w) {
        if (w.size() > 3 && u(rng) < 0.2f) std::swap(w[1], w[2]);
        return w;
    };
    int correct = 0;
    const int total = 3000;
    for (int i = 0; i < total; ++i) {
        int c = cls(rng);
        std::vector<std::string> words;
        int keys = 1 + (u(rng) < 0.5f);
        for (int k = 0; k < keys; ++k) words.push_back(misspell(keywords[c][kw(rng)]));
        // a keyword of another class sometimes appears too
        if (u(rng) < 0.2f) words.push_back(keywords[(c + 1 + (u(rng) < 0.5f)) % 3][kw(rng)]);
        int n = count(rng);
        for (int k = 0; k < n; ++k) words.push_back(filler[fill(rng)]);
        std::shuffle(words.begin(), words.end(), rng);
        std::string text;
        for (auto& w : words) text += (text.empty() ? "" : " ") + w;
        StepId step = agent.Observe({{"message", text}});
        Prediction p = agent.Predict(step, "topic");
        if (i >= total - 1000) correct += p.available && p.label == "t" + std::to_string(c);
        agent.Feedback(step, "label", "t" + std::to_string(c));
        Tick();
    }
    double accuracy = correct / 1000.0;
    return {accuracy >= 0.90, "accuracy " + Fmt(accuracy)};
}

// S2: three regression outputs, 10% missing input values
Result S2(uint32_t seed) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 5}};
    spec.outputs = {{"y", OutputType::VALUE, {}, 3}};
    spec.feedback = {{"target", FeedbackType::TARGET, "y"}};
    Agent agent(spec);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-2, 2), unit(0, 1);
    std::normal_distribution<float> noise(0, 0.1f);
    auto make = [&](std::vector<float>& x, std::vector<float>& y) {
        x.resize(5);
        for (auto& v : x) v = u(rng);
        y = {x[0] + x[1] + noise(rng), x[2] * x[3] + noise(rng), std::sin(2.0f * x[4]) + noise(rng)};
        for (auto& v : x) {
            if (unit(rng) < 0.1f) v = std::nanf("");
        }
    };
    std::vector<float> x, y;
    for (int i = 0; i < 6000; ++i) {
        make(x, y);
        StepId step = agent.Observe({{"x", x}});
        agent.Feedback(step, "target", y);
        Tick();
    }
    std::vector<std::vector<float>> preds, truths;
    for (int i = 0; i < 1000; ++i) {
        make(x, y);
        Prediction p = agent.Predict(agent.Observe({{"x", x}}), "y");
        if (!p.available) return {false, "no prediction"};
        preds.push_back(p.values);
        truths.push_back(y);
    }
    std::string metrics = "R2";
    bool pass = true;
    for (int k = 0; k < 3; ++k) {
        double mean = 0, sse = 0, sst = 0;
        for (auto& t : truths) mean += t[k] / truths.size();
        for (size_t i = 0; i < truths.size(); ++i) {
            sse += std::pow(preds[i][k] - truths[i][k], 2);
            sst += std::pow(truths[i][k] - mean, 2);
        }
        double r2 = 1 - sse / sst;
        metrics += " " + Fmt(r2);
        pass = pass && r2 >= 0.7;
    }
    return {pass, metrics};
}

// S3: plan environment H (text + numeric, label 30% / outcome 70%) with an
// anomaly output alongside
Result S3(uint32_t seed) {
    EnvironmentSpec spec;
    spec.inputs = {{"reading", InputType::NUMERIC, 3}, {"note", InputType::TEXT}};
    spec.outputs = {{"state", OutputType::CLASS, {"idle", "busy", "hot", "fault"}}, {"odd", OutputType::ANOMALY}};
    spec.feedback = {{"label", FeedbackType::LABEL, "state"}, {"right", FeedbackType::OUTCOME, "state"}};
    Agent agent(spec);
    const std::vector<std::string> names = {"idle", "busy", "hot", "fault"};
    const std::vector<std::vector<float>> centers = {{0, 0, 0}, {2, 0, 1}, {2, 2, 0}, {0, 2, 2}};
    const std::vector<std::vector<std::string>> words = {
        {"quiet", "standby", "waiting"}, {"load", "working", "queue"}, {"warm", "temperature", "fan"},
        {"alarm", "failure", "stopped"}};
    const std::vector<std::string> common = {"sensor", "report", "status", "unit", "check"};
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0, 0.7f);
    std::uniform_int_distribution<int> cls(0, 3), w3(0, 2), w5(0, 4);
    std::uniform_real_distribution<float> u(0, 1);
    const int total = 6000;
    int correct = 0, labelled_steps = 0;
    std::vector<float> scores;
    std::vector<int> anomalies;
    for (int i = 0; i < total; ++i) {
        int c = cls(rng);
        std::vector<float> r(3);
        for (int d = 0; d < 3; ++d) r[d] = centers[c][d] + noise(rng);
        std::string note = common[w5(rng)] + " " + (u(rng) < 0.7f ? words[c][w3(rng)] : words[cls(rng)][w3(rng)]);
        int anomaly = u(rng) < 0.01f;
        if (anomaly) r = {9.0f * (u(rng) < 0.5f ? 1 : -1), -7.0f, 8.0f};
        StepId step = agent.Observe({{"reading", r}, {"note", note}});
        Prediction p = agent.Predict(step, "state");
        Prediction a = agent.Predict(step, "odd");
        if (i >= 1000) {
            scores.push_back(a.score);
            anomalies.push_back(anomaly);
        }
        if (!anomaly) {
            if (i >= total - 2000) {
                correct += p.available && p.label == names[c];
                labelled_steps++;
            }
            if (u(rng) < 0.3f) agent.Feedback(step, "label", names[c]);
            else if (p.available) agent.Feedback(step, "right", p.label == names[c] ? 1.0 : 0.0);
        }
        Tick();
    }
    double accuracy = correct / static_cast<double>(labelled_steps);  // non-anomalous steps of the last 2000
    double auc = AreaUnderRoc(scores, anomalies);
    return {accuracy >= 0.85 && auc >= 0.8, "accuracy " + Fmt(accuracy) + ", anomaly AUC " + Fmt(auc)};
}

// S4: regression whose target function changes at step 4000
Result S4(uint32_t seed) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 2}};
    spec.outputs = {{"y", OutputType::VALUE}};
    spec.feedback = {{"target", FeedbackType::TARGET, "y"}};
    Agent agent(spec);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1, 1);
    std::normal_distribution<float> noise(0, 0.1f);
    double before = 0, after = 0, right_after = 0;
    for (int i = 0; i < 6000; ++i) {
        float x0 = u(rng), x1 = u(rng);
        float y = (i < 4000 ? 2 * x0 - x1 : -x0 + 3 * x1) + noise(rng);
        StepId step = agent.Observe({{"x", std::vector<float>{x0, x1}}});
        Prediction p = agent.Predict(step, "y");
        double e = p.available ? std::pow(p.values[0] - y, 2) : 10.0;
        if (i >= 3000 && i < 4000) before += e / 1000;
        if (i >= 4000 && i < 4200) right_after += e / 200;
        if (i >= 5000) after += e / 1000;
        agent.Feedback(step, "target", static_cast<double>(y));
        Tick();
    }
    return {after <= 2 * before, "MSE before " + Fmt(before) + ", first 200 after " + Fmt(right_after) +
                                     ", 1000-2000 after " + Fmt(after) + ", drifts " +
                                     std::to_string(agent.Stats().drift_events)};
}

// S5: open-vocabulary user IDs, imbalanced classes, new users at step 4000
Result S5(uint32_t seed) {
    EnvironmentSpec spec;
    spec.inputs = {{"user", InputType::CATEGORICAL}, {"activity", InputType::NUMERIC, 2}};
    spec.outputs = {{"segment", OutputType::CLASS}};
    spec.feedback = {{"label", FeedbackType::LABEL, "segment"}};
    Agent agent(spec);
    std::mt19937 rng(seed);
    std::discrete_distribution<int> share({60, 20, 10, 5, 5});
    std::vector<int> segment(250);
    for (auto& s : segment) s = share(rng);
    std::normal_distribution<float> noise(0, 1.0f);
    std::uniform_real_distribution<float> u(0, 1);
    const int total = 8000;
    std::vector<int> hits(5, 0), seen(5, 0);
    for (int i = 0; i < total; ++i) {
        int user = i < 4000 ? std::uniform_int_distribution<int>(0, 199)(rng)
                            : (u(rng) < 0.3f ? std::uniform_int_distribution<int>(200, 249)(rng)
                                             : std::uniform_int_distribution<int>(0, 199)(rng));
        int c = segment[user];
        // activity is weakly informative
        std::vector<float> activity = {0.5f * c + noise(rng), noise(rng)};
        StepId step = agent.Observe({{"user", "u" + std::to_string(user)}, {"activity", activity}});
        Prediction p = agent.Predict(step, "segment");
        if (i >= total - 2000) {
            seen[c]++;
            hits[c] += p.available && p.label == "s" + std::to_string(c);
        }
        agent.Feedback(step, "label", "s" + std::to_string(c));
        Tick();
    }
    double recall = 0;
    std::string per;
    int classes = 0;
    for (int c = 0; c < 5; ++c) {
        if (seen[c] == 0) continue;
        double r = static_cast<double>(hits[c]) / seen[c];
        recall += r;
        classes++;
        per += " " + Fmt(r);
    }
    recall /= classes;
    return {recall >= 0.8, "mean recall " + Fmt(recall) + " (per class" + per + ")"};
}

// S6: contextual bandit with outcomes only
Result S6(uint32_t seed) {
    EnvironmentSpec spec;
    spec.inputs = {{"context", InputType::CATEGORICAL, 0, {"a", "b", "c", "d"}}};
    spec.outputs = {{"offer", OutputType::ACTION, {"x", "y", "z"}}};
    spec.feedback = {{"accepted", FeedbackType::OUTCOME, "offer"}};
    Agent agent(spec);
    const std::vector<std::string> contexts = {"a", "b", "c", "d"};
    const std::vector<std::string> best = {"y", "x", "z", "y"};
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, 3);
    std::uniform_real_distribution<float> u(0, 1);
    int successes = 0;
    for (int i = 0; i < 3000; ++i) {
        int c = pick(rng);
        StepId step = agent.Observe({{"context", contexts[c]}});
        std::string offer = agent.Act(step, "offer");
        bool ok = u(rng) < (offer == best[c] ? 0.8f : 0.2f);
        if (i >= 2500) successes += ok;
        agent.Feedback(step, "accepted", ok ? 1.0 : 0.0);
        agent.EndEpisode(true);
        Tick();
    }
    int right = 0;
    std::string chosen;
    for (int c = 0; c < 4; ++c) {
        StepId step = agent.Observe({{"context", contexts[c]}});
        std::string offer = agent.Act(step, "offer", false);
        right += offer == best[c];
        chosen += offer;
        agent.EndEpisode(false);
    }
    return {right == 4, "greedy best in " + std::to_string(right) + "/4 contexts (" + chosen +
                            "), success rate last 500 " + Fmt(successes / 500.0)};
}

// S7: continuous 2-D reaching
Result S7(uint32_t seed) {
    EnvironmentSpec spec;
    spec.inputs = {{"state", InputType::NUMERIC, 4}};
    OutputSpec move;
    move.name = "velocity";
    move.type = OutputType::CONTINUOUS_ACTION;
    move.size = 2;
    move.low = {-0.1f, -0.1f};
    move.high = {0.1f, 0.1f};
    spec.outputs = {move};
    spec.feedback = {{"reward", FeedbackType::REWARD, "velocity"}};
    Agent agent(spec);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1, 1);
    auto episode = [&](bool explore) {
        float px = u(rng), py = u(rng), tx = u(rng), ty = u(rng);
        for (int t = 0; t < 50; ++t) {
            StepId step = agent.Observe({{"state", std::vector<float>{px, py, tx, ty}}});
            auto v = agent.ActContinuous(step, "velocity", explore);
            px = std::clamp(px + v[0], -1.0f, 1.0f);
            py = std::clamp(py + v[1], -1.0f, 1.0f);
            float d = std::hypot(px - tx, py - ty);
            agent.Feedback(step, "reward", -static_cast<double>(d));
            Tick();
        }
        agent.EndEpisode(false);
        return std::hypot(px - tx, py - ty);
    };
    for (int e = 0; e < 300; ++e) episode(true);
    double distance = 0;
    for (int e = 0; e < 20; ++e) distance += episode(false) / 20;
    return {distance <= 0.1, "mean final distance " + Fmt(distance)};
}

// S8: image regression (blob x position)
Result S8(uint32_t seed) {
    EnvironmentSpec spec;
    spec.inputs = {{"image", InputType::IMAGE, 0, {}, 16, 16, 1}};
    spec.outputs = {{"x", OutputType::VALUE}};
    spec.feedback = {{"target", FeedbackType::TARGET, "x"}};
    Agent agent(spec);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pos(2, 13);
    std::normal_distribution<float> noise(0, 15.0f);
    auto make = [&](Image& image) {
        image = Image{16, 16, 1, std::vector<float>(256)};
        int cx = pos(rng), cy = pos(rng);
        for (int y = 0; y < 16; ++y) {
            for (int x = 0; x < 16; ++x) {
                float d2 = static_cast<float>((x - cx) * (x - cx) + (y - cy) * (y - cy));
                image.pixels[y * 16 + x] = std::clamp(30.0f + 200.0f * std::exp(-d2 / 3.0f) + noise(rng), 0.0f, 255.0f);
            }
        }
        return static_cast<float>(cx);
    };
    Image image;
    for (int i = 0; i < 3000; ++i) {
        float x = make(image);
        StepId step = agent.Observe({{"image", image}});
        agent.Feedback(step, "target", static_cast<double>(x));
        Tick();
    }
    double sse = 0, sst = 0;
    std::vector<float> truth, pred;
    for (int i = 0; i < 500; ++i) {
        float x = make(image);
        Prediction p = agent.Predict(agent.Observe({{"image", image}}), "x");
        truth.push_back(x);
        pred.push_back(p.available ? p.values[0] : 7.5f);
    }
    double mean = std::accumulate(truth.begin(), truth.end(), 0.0) / truth.size();
    for (size_t i = 0; i < truth.size(); ++i) {
        sse += std::pow(pred[i] - truth[i], 2);
        sst += std::pow(truth[i] - mean, 2);
    }
    double r2 = 1 - sse / sst;
    return {r2 >= 0.8, "R2 " + Fmt(r2) + ", RMSE " + Fmt(std::sqrt(sse / truth.size())) + " pixels"};
}

// S9: two related time series
Result S9(uint32_t seed) {
    EnvironmentSpec spec;
    spec.inputs = {{"series", InputType::TIME_SERIES, 2}};
    spec.outputs = {{"next", OutputType::NEXT_OBSERVATION}};
    Agent agent(spec);
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0, 0.05f);
    const int total = 5000;
    std::vector<float> s1(total + 10), s2(total + 10);
    for (int t = 0; t < total + 10; ++t) s1[t] = std::sin(2 * 3.14159265f * t / 20) + noise(rng);
    for (int t = 0; t < total + 10; ++t) s2[t] = (t >= 3 ? s1[t - 3] : 0.0f) + noise(rng);
    double agent_err[2] = {0, 0}, last_err[2] = {0, 0};
    for (int t = 0; t < total; ++t) {
        StepId step = agent.Observe({{"series", std::vector<float>{s1[t], s2[t]}}});
        Prediction p = agent.Predict(step, "next");
        if (t >= total - 2000) {
            float next[2] = {s1[t + 1], s2[t + 1]}, now[2] = {s1[t], s2[t]};
            for (int k = 0; k < 2; ++k) {
                float guess = p.available && !std::isnan(p.values[k]) ? p.values[k] : now[k];
                agent_err[k] += std::pow(guess - next[k], 2);
                last_err[k] += std::pow(now[k] - next[k], 2);
            }
        }
        Tick();
    }
    double r0 = agent_err[0] / last_err[0], r1 = agent_err[1] / last_err[1];
    return {r0 <= 0.5 && r1 <= 0.5, "MSE ratio to last value: " + Fmt(r0) + ", " + Fmt(r1)};
}

// S10: raw inputs with large scale and offsets
Result S10(uint32_t seed) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 3}};
    spec.outputs = {{"kind", OutputType::CLASS}};
    spec.feedback = {{"label", FeedbackType::LABEL, "kind"}};
    Agent agent(spec);
    std::mt19937 rng(seed);
    std::normal_distribution<float> noise(0, 0.5f);
    std::uniform_int_distribution<int> cls(0, 2);
    const double scale[3] = {1e6, 3e5, 2e6}, offset[3] = {1e7, -5e6, 2.5e7};
    int correct = 0;
    for (int i = 0; i < 3000; ++i) {
        int c = cls(rng);
        std::vector<float> x(3);
        for (int d = 0; d < 3; ++d) x[d] = static_cast<float>(offset[d] + scale[d] * ((d == c ? 2.0 : 0.0) + noise(rng)));
        StepId step = agent.Observe({{"x", x}});
        Prediction p = agent.Predict(step, "kind");
        if (i >= 2000) correct += p.available && p.label == std::to_string(c);
        agent.Feedback(step, "label", static_cast<double>(c));
        Tick();
    }
    double accuracy = correct / 1000.0;
    return {accuracy >= 0.90, "accuracy " + Fmt(accuracy)};
}

using Scenario = Result (*)(uint32_t);
const Scenario kScenarios[] = {S1, S2, S3, S4, S5, S6, S7, S8, S9, S10};

class ScenarioTest : public ::testing::TestWithParam<std::tuple<int, uint32_t>> {
protected:
    void SetUp() override {
        if (!std::getenv("DPAN_SCENARIOS")) GTEST_SKIP() << "set DPAN_SCENARIOS=1 to run the scenario tests";
        g_clock = std::make_shared<ManualClock>(Timestamp::FromMicros(1'700'000'000'000'000));
        scope_ = std::make_unique<ScopedClock>(g_clock);
    }
    void TearDown() override { scope_.reset(); }

    std::unique_ptr<ScopedClock> scope_;
};

TEST_P(ScenarioTest, MeetsItsCriterion) {
    auto [scenario, seed] = GetParam();
    Result result = kScenarios[scenario - 1](seed);
    std::cout << "S" << scenario << " seed " << seed << ": " << result.metrics << "\n";
    RecordProperty("metrics", result.metrics);
    EXPECT_TRUE(result.pass) << "S" << scenario << " seed " << seed << ": " << result.metrics;
}

std::string Name(const ::testing::TestParamInfo<std::tuple<int, uint32_t>>& info) {
    return "S" + std::to_string(std::get<0>(info.param)) + "_seed" + std::to_string(std::get<1>(info.param));
}

INSTANTIATE_TEST_SUITE_P(Scenarios, ScenarioTest,
                         ::testing::Combine(::testing::Range(1, 11), ::testing::Values(1u, 2u, 3u)), Name);

}  // namespace
}  // namespace agent
}  // namespace dpan
