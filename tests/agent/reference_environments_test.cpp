// File: tests/agent/reference_environments_test.cpp
//
// Reference environments A-D (docs/AGENT_PLAN.md, section 6). Every agent
// uses the default AgentConfig; data comes from a fixed seed and time from a
// ManualClock.

#include "agent/agent.hpp"
#include "core/clock.hpp"
#include "nn/loss.hpp"
#include "nn/optimizer.hpp"
#include "nn/sequential.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace dpan {
namespace agent {
namespace {

constexpr double kPi = 3.14159265358979323846;

/// Fake clock advanced one second per observation
class ReferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        clock_ = std::make_shared<ManualClock>(Timestamp::FromMicros(1'700'000'000'000'000));
        scope_ = std::make_unique<ScopedClock>(clock_);
    }
    void Tick() { clock_->Advance(std::chrono::seconds(1)); }

    std::shared_ptr<ManualClock> clock_;
    std::unique_ptr<ScopedClock> scope_;
};

// ============================================================================
// A: stream classification with drift
// ============================================================================
//
// 3 classes, a 4-dimensional numeric input drawn around one center per class
// and a categorical region that matches the class 60% of the time. Labels
// arrive right after each prediction (prequential). At sample 6000 the
// labels rotate (class k becomes k+1). Pass: accuracy >= 90% on samples
// 2000-6000, on the 200 samples ending 1000 samples after the drift, and on
// all samples from 1000 after the drift to the end (9000).

TEST_F(ReferenceTest, A_StreamClassificationWithDrift) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 4},
                   {"region", InputType::CATEGORICAL, 0, {"north", "south", "east", "west", "center"}}};
    spec.outputs = {{"kind", OutputType::CLASS}};
    spec.feedback = {{"label", FeedbackType::LABEL, "kind"}};
    Agent agent(spec);

    const std::vector<std::vector<float>> centers = {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}};
    const std::vector<std::string> regions = {"north", "south", "east", "west", "center"};
    std::mt19937 rng(11);
    std::normal_distribution<float> noise(0.0f, 0.6f);
    std::uniform_int_distribution<int> pick_class(0, 2), pick_region(0, 4);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);

    const int total = 9000, drift = 6000;
    std::vector<int> correct(total, 0);
    for (int i = 0; i < total; ++i) {
        int cluster = pick_class(rng);
        std::vector<float> x(4);
        for (int d = 0; d < 4; ++d) x[d] = centers[cluster][d] + noise(rng);
        std::string region = u(rng) < 0.6f ? regions[cluster] : regions[pick_region(rng)];
        int label = i < drift ? cluster : (cluster + 1) % 3;

        StepId step = agent.Observe({{"x", x}, {"region", region}});
        Prediction p = agent.Predict(step, "kind");
        correct[i] = p.available && p.label == std::to_string(label);
        agent.Feedback(step, "label", static_cast<double>(label));
        Tick();
    }

    auto accuracy = [&](int begin, int end) {
        return static_cast<double>(std::accumulate(correct.begin() + begin, correct.begin() + end, 0)) /
               (end - begin);
    };
    double before = accuracy(2000, drift);
    double recovered = accuracy(drift + 800, drift + 1000);
    double after = accuracy(drift + 1000, total);
    RecordProperty("accuracy_before_drift", std::to_string(before));
    RecordProperty("accuracy_800_1000_after_drift", std::to_string(recovered));
    RecordProperty("accuracy_after_recovery", std::to_string(after));
    std::cout << "A: accuracy before drift " << before << ", 800-1000 after drift " << recovered
              << ", after " << after << "\n";
    EXPECT_GE(before, 0.90);
    EXPECT_GE(recovered, 0.90);
    EXPECT_GE(after, 0.90);
}

// ============================================================================
// B: stream regression
// ============================================================================
//
// y = sin(1.5 x0) + 0.5 x1^2 - x2 + 0.3 x0 x3 + noise(0.3), x uniform in
// [-2, 2]^4. 8000 samples streamed with immediate targets; then 1000 test
// samples predicted without feedback. The offline model has the agent's
// trunk shape (4 -> 32 -> 32 -> 1, ReLU), is trained with Adam for up to
// 300 epochs with early stopping on the same 8000 samples. Pass: agent test
// error <= 1.2 x offline test error.

TEST_F(ReferenceTest, B_StreamRegression) {
    EnvironmentSpec spec;
    spec.inputs = {{"x", InputType::NUMERIC, 4}};
    spec.outputs = {{"y", OutputType::VALUE}};
    spec.feedback = {{"target", FeedbackType::TARGET, "y"}};
    Agent agent(spec);

    std::mt19937 rng(21);
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);
    std::normal_distribution<float> noise(0.0f, 0.3f);
    auto make = [&](std::vector<float>& x) {
        for (auto& v : x) v = u(rng);
        return std::sin(1.5f * x[0]) + 0.5f * x[1] * x[1] - x[2] + 0.3f * x[0] * x[3] + noise(rng);
    };

    const int train = 8000, test = 1000;
    std::vector<std::vector<float>> train_x(train, std::vector<float>(4)), test_x(test, std::vector<float>(4));
    std::vector<float> train_y(train), test_y(test);
    for (int i = 0; i < train; ++i) train_y[i] = make(train_x[i]);
    for (int i = 0; i < test; ++i) test_y[i] = make(test_x[i]);

    for (int i = 0; i < train; ++i) {
        StepId step = agent.Observe({{"x", train_x[i]}});
        agent.Feedback(step, "target", static_cast<double>(train_y[i]));
        Tick();
    }
    double agent_error = 0.0;
    for (int i = 0; i < test; ++i) {
        StepId step = agent.Observe({{"x", test_x[i]}});
        Prediction p = agent.Predict(step, "y");
        ASSERT_TRUE(p.available);
        agent_error += std::pow(p.values[0] - test_y[i], 2);
    }
    agent_error /= test;

    nn::Sequential offline;
    offline.Emplace<nn::Dense>(4, 32, nn::Initialization::HE)
        .Emplace<nn::ReLU>()
        .Emplace<nn::Dense>(32, 32, nn::Initialization::HE)
        .Emplace<nn::ReLU>()
        .Emplace<nn::Dense>(32, 1);
    nn::Adam optimizer(1e-3f);
    nn::Sequential::FitOptions options;
    options.epochs = 300;
    options.batch_size = 32;
    options.validation_split = 0.1f;
    options.early_stopping_patience = 20;
    nn::Matrix x = nn::Matrix::FromRows(train_x);
    nn::Matrix y(train, 1, train_y);
    offline.Fit(x, y, nn::MeanSquaredError(), optimizer, options);
    double offline_error = 0.0;
    nn::Matrix predictions = offline.Predict(nn::Matrix::FromRows(test_x));
    for (int i = 0; i < test; ++i) offline_error += std::pow(predictions(i, 0) - test_y[i], 2);
    offline_error /= test;

    std::cout << "B: agent test MSE " << agent_error << ", offline " << offline_error << " (ratio "
              << agent_error / offline_error << ")\n";
    EXPECT_LE(agent_error, 1.2 * offline_error);
}

// ============================================================================
// C: time-series forecasting
// ============================================================================
//
// s(t) = sin(2 pi t / 25) + 0.5 sin(2 pi t / 7 + 1) + noise(0.1), one
// TIME_SERIES input, NEXT_OBSERVATION output, 6000 steps. Pass: mean squared
// error of the next-value prediction over steps 3000-6000 <= 0.8 x that of
// predicting the last value.

TEST_F(ReferenceTest, C_TimeSeriesForecasting) {
    EnvironmentSpec spec;
    spec.inputs = {{"load", InputType::TIME_SERIES, 1}};
    spec.outputs = {{"next", OutputType::NEXT_OBSERVATION}};
    Agent agent(spec);

    std::mt19937 rng(31);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    const int total = 6000;
    std::vector<float> s(total + 1);
    for (int t = 0; t <= total; ++t) {
        s[t] = static_cast<float>(std::sin(2 * kPi * t / 25) + 0.5 * std::sin(2 * kPi * t / 7 + 1)) + noise(rng);
    }

    double agent_error = 0.0, last_value_error = 0.0;
    int counted = 0;
    for (int t = 0; t < total; ++t) {
        StepId step = agent.Observe({{"load", static_cast<double>(s[t])}});
        Prediction p = agent.Predict(step, "next");
        if (t >= 3000) {
            ASSERT_TRUE(p.available);
            agent_error += std::pow(p.values[0] - s[t + 1], 2);
            last_value_error += std::pow(s[t] - s[t + 1], 2);
            ++counted;
        }
        Tick();
    }
    agent_error /= counted;
    last_value_error /= counted;
    std::cout << "C: agent MSE " << agent_error << ", last value " << last_value_error << " (ratio "
              << agent_error / last_value_error << ")\n";
    EXPECT_LE(agent_error, 0.8 * last_value_error);
}

// ============================================================================
// D: anomaly detection
// ============================================================================
//
// Inputs: 3 numeric metrics (m1 = z, m2 = 2z + noise, m3 independent) and a
// log message from 8 templates. 2% of the 10000 observations are anomalies,
// a third each: broken correlation (m2 = -2z, |z| in [1.5, 2.5]), an
// extreme value (m3 = +-6), or a message made of unusual words. No
// feedback. Pass: area under the ROC curve of the ANOMALY score >= 0.9 over
// observations 1000-10000.

double AreaUnderRoc(const std::vector<float>& scores, const std::vector<int>& labels) {
    std::vector<size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return scores[a] < scores[b]; });
    // Rank-sum (Mann-Whitney) with average ranks for ties
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

TEST_F(ReferenceTest, D_AnomalyDetection) {
    EnvironmentSpec spec;
    spec.inputs = {{"metrics", InputType::NUMERIC, 3}, {"message", InputType::TEXT}};
    spec.outputs = {{"unusual", OutputType::ANOMALY}};
    Agent agent(spec);

    const std::vector<std::string> templates = {
        "user {} logged in", "user {} logged out", "disk usage at {} percent",
        "request served in {} ms", "cache hit for key {}", "backup job {} finished",
        "connection from host {} accepted", "scheduled task {} started"};
    const std::vector<std::string> unusual = {"kernel", "panic", "segmentation", "fault",
                                              "overheat", "corrupted", "breach", "unauthorized"};
    std::mt19937 rng(41);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::uniform_int_distribution<int> pick_template(0, 7), pick_number(1, 999), pick_word(0, 7);

    auto fill = [](std::string text, int number) {
        return text.replace(text.find("{}"), 2, std::to_string(number));
    };

    const int total = 10000;
    std::vector<float> scores;
    std::vector<int> labels;
    for (int i = 0; i < total; ++i) {
        float z = normal(rng);
        std::vector<float> metrics = {z, 2 * z + 0.3f * normal(rng), normal(rng)};
        std::string message = fill(templates[pick_template(rng)], pick_number(rng));
        int anomaly = u(rng) < 0.02f;
        if (anomaly) {
            float kind = u(rng);
            if (kind < 1.0f / 3) {
                float magnitude = 1.5f + u(rng);
                float sign = u(rng) < 0.5f ? -1.0f : 1.0f;
                metrics[0] = sign * magnitude;
                metrics[1] = -2 * metrics[0];
            } else if (kind < 2.0f / 3) {
                metrics[2] = u(rng) < 0.5f ? -6.0f : 6.0f;
            } else {
                message = unusual[pick_word(rng)] + " " + unusual[pick_word(rng)] + " " + unusual[pick_word(rng)];
            }
        }
        StepId step = agent.Observe({{"metrics", metrics}, {"message", message}});
        Prediction p = agent.Predict(step, "unusual");
        if (i >= 1000) {
            scores.push_back(p.score);
            labels.push_back(anomaly);
        }
        Tick();
    }
    double auc = AreaUnderRoc(scores, labels);
    std::cout << "D: area under ROC " << auc << " ("
              << std::count(labels.begin(), labels.end(), 1) << " anomalies)\n";
    EXPECT_GE(auc, 0.9);
}

}  // namespace
}  // namespace agent
}  // namespace dpan
