// File: tests/agent/components_test.cpp
//
// Agent internals: network, pattern memory, replay buffer
#include "agent/drift_detector.hpp"
#include "agent/network.hpp"
#include "agent/pattern_memory.hpp"
#include "agent/replay_buffer.hpp"
#include "agent/word_sequence.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>

namespace dpan {
namespace agent {
namespace {

using nn::Matrix;

std::vector<ChannelLayout> Layout() {
    // A numeric channel (3 values) and a hashed channel (9 values, embedded to 4)
    return {{"x", InputType::NUMERIC, 0, 3, 0}, {"tag", InputType::CATEGORICAL, 3, 9, 4}};
}

TEST(AgentNetworkTest, ShapesFromLayout) {
    AgentNetwork network(Layout(), {{"c", HeadKind::CLASSIFIER, 3}, {"v", HeadKind::REGRESSOR, 2}}, 1e-3f, 1);
    EXPECT_EQ(12u, network.InputSize());
    EXPECT_EQ(3u + 4u + 1u, network.EmbeddedSize());  // projection + passed-through indicator
    EXPECT_EQ(32u, network.TrunkWidth());               // 2 x 8 = 16, raised to the minimum
    Matrix x(5, 12, 0.1f);
    EXPECT_EQ(3u, network.Infer(x, 0).Cols());
    EXPECT_EQ(2u, network.Infer(x, 1).Cols());
    EXPECT_THROW(network.Infer(Matrix(1, 11), 0), std::invalid_argument);

    AgentNetwork wide({{"x", InputType::NUMERIC, 0, 400, 0}}, {{"v", HeadKind::REGRESSOR, 1}}, 1e-3f, 1);
    EXPECT_EQ(512u, wide.TrunkWidth());
}

TEST(AgentNetworkTest, TrainsOnlyHeadsWithTargets) {
    AgentNetwork network(Layout(), {{"c", HeadKind::CLASSIFIER, 2}, {"v", HeadKind::REGRESSOR, 1}}, 1e-2f, 2);
    Matrix x(4, 12);
    for (size_t r = 0; r < 4; ++r) x(r, r % 3) = r < 2 ? 1.0f : -1.0f;
    HeadTarget a, b, value;
    a.label = 0;
    b.label = 1;
    value.values = {std::numeric_limits<float>::quiet_NaN()};  // no target for this element

    std::vector<std::vector<const HeadTarget*>> targets = {{&a, &a, &b, &b}, {nullptr, &value, nullptr, nullptr}};
    std::vector<float> first, last;
    for (int i = 0; i < 300; ++i) {
        auto losses = network.TrainBatch(x, targets, {1.0f, 1.0f}, 5.0f);
        if (i == 0) first = losses;
        last = losses;
    }
    EXPECT_LT(last[0], 0.1f * first[0]);
    EXPECT_TRUE(std::isnan(last[1]));  // regressor had no usable target
    auto probabilities = nn::Softmax::Compute(network.Infer(x, 0));
    EXPECT_GT(probabilities(0, 0), 0.9f);
    EXPECT_GT(probabilities(3, 1), 0.9f);
}

TEST(AgentNetworkTest, NegativeExamplesLowerTheClass) {
    AgentNetwork network(Layout(), {{"c", HeadKind::CLASSIFIER, 3}}, 1e-2f, 3);
    Matrix x(1, 12, 0.5f);
    float before = nn::Softmax::Compute(network.Infer(x, 0))(0, 1);
    HeadTarget negative;
    negative.negative = 1;
    for (int i = 0; i < 100; ++i) network.TrainBatch(x, {{&negative}}, {1.0f}, 5.0f);
    float after = nn::Softmax::Compute(network.Infer(x, 0))(0, 1);
    EXPECT_LT(after, 0.1f * before);
}

TEST(AgentNetworkTest, GrowingClassesKeepsLearnedOutputs) {
    AgentNetwork network(Layout(), {{"c", HeadKind::CLASSIFIER, 0}}, 1e-3f, 4);
    Matrix x(2, 12, 0.3f);
    EXPECT_EQ(0u, network.Infer(x, 0).Cols());
    network.GrowClasses(0, 2);
    Matrix before = network.Infer(x, 0);
    network.GrowClasses(0, 3);
    Matrix after = network.Infer(x, 0);
    ASSERT_EQ(3u, after.Cols());
    EXPECT_FLOAT_EQ(before(0, 0), after(0, 0));
    EXPECT_FLOAT_EQ(before(1, 1), after(1, 1));
    EXPECT_EQ(3u, network.Head(0).size);
}

TEST(AnomalyAutoencoderTest, UnusualInputsReconstructWorse) {
    AnomalyAutoencoder autoencoder(8, 1e-2f, 5);
    Matrix normal(32, 8);
    for (size_t r = 0; r < 32; ++r) {
        float z = static_cast<float>(r) / 16.0f - 1.0f;
        for (size_t c = 0; c < 8; ++c) normal(r, c) = z * (c % 2 ? 1.0f : -1.0f);
    }
    for (int i = 0; i < 500; ++i) autoencoder.TrainBatch(normal, 5.0f);
    Matrix odd(1, 8, 1.0f);
    float normal_error = autoencoder.Errors(normal.SelectRows({5}))[0];
    float odd_error = autoencoder.Errors(odd)[0];
    EXPECT_GT(odd_error, 10.0f * normal_error);
}

PatternMemory::Config MemoryConfig(size_t capacity = 100) {
    PatternMemory::Config config;
    config.dimension = 8;
    config.capacity = capacity;
    return config;
}

TEST(PatternMemoryTest, MergesCloseStatesAndCreatesDistantOnes) {
    PatternMemory memory(2, 0, {}, 1, MemoryConfig());
    auto a = memory.Project({0, 0});
    auto b = memory.Project({10, 0});
    auto first = memory.Insert(a, 1, {1.0f});
    EXPECT_TRUE(first.created);
    EXPECT_TRUE(std::isinf(first.nearest_distance));
    auto second = memory.Insert(b, 2, {2.0f});
    EXPECT_TRUE(second.created);
    // Identical to the first state: joins it
    auto third = memory.Insert(a, 3, {3.0f});
    EXPECT_FALSE(third.created);
    EXPECT_EQ(first.id, third.id);
    EXPECT_FLOAT_EQ(0.0f, third.nearest_distance);
    EXPECT_EQ(2u, memory.Size());
}

TEST(PatternMemoryTest, PrunesLowestUtilityPatterns) {
    PatternMemory memory(1, 1, {}, 0, MemoryConfig(3));
    // p2 (older) and p3 (newer): used once, one transition each
    auto p2 = memory.Insert(memory.Project({100}), 1, {});
    auto p3 = memory.Insert(memory.Project({-100}), 2, {});
    // p1: used often and labelled consistently
    auto p1 = memory.Insert(memory.Project({0}), 3, {});
    for (int i = 0; i < 50; ++i) {
        EXPECT_FALSE(memory.Insert(memory.Project({0}), 4 + i, {}).created);
        memory.AddLabel(p1.id, 0, 0);
    }
    memory.RecordTransition(p2.id, p1.id);
    memory.RecordTransition(p3.id, p1.id);
    EXPECT_GT(memory.Utility(p1.id, 62), memory.Utility(p3.id, 62));
    EXPECT_GT(memory.Utility(p3.id, 62), memory.Utility(p2.id, 62));

    auto p4 = memory.Insert(memory.Project({300}), 5000, {});
    EXPECT_TRUE(p4.created);
    EXPECT_EQ(3u, memory.Size());
    EXPECT_EQ(1u, memory.Pruned());
    EXPECT_TRUE(memory.Contains(p1.id));
    EXPECT_TRUE(memory.Contains(p3.id));
    EXPECT_FALSE(memory.Contains(p2.id));
    EXPECT_EQ(1u, memory.Associations().GetAssociationCount());  // p2's edge removed
}

TEST(PatternMemoryTest, StaysWithinByteBudget) {
    PatternMemory::Config config = MemoryConfig(100000);
    config.memory_budget_bytes = 64 * 1024;
    PatternMemory memory(4, 1, {}, 4, config);
    std::mt19937 rng(9);
    std::normal_distribution<float> normal(0.0f, 10.0f);
    PatternID previous;
    for (uint64_t step = 1; step <= 5000; ++step) {
        auto placed = memory.Insert(memory.Project({normal(rng), normal(rng), normal(rng), normal(rng)}), step,
                                    {1, 2, 3, 4});
        memory.AddLabel(placed.id, 0, step % 3);
        if (step > 1) memory.RecordTransition(previous, placed.id);
        previous = placed.id;
        ASSERT_LE(memory.MemoryBytes(), config.memory_budget_bytes) << step;
    }
    EXPECT_GT(memory.Pruned(), 0u);
    EXPECT_GT(memory.Size(), 20u);
}

TEST(PatternMemoryTest, VotesWithShortAndLongMemory) {
    PatternMemory memory(1, 1, {1}, 0, MemoryConfig());
    // Old labels near 0 say class 0; recent ones near 1 say class 1
    std::vector<PatternID> ids;
    for (int i = 0; i < 6; ++i) {
        auto placed = memory.Insert(memory.Project({static_cast<float>(i < 3 ? i * 10 : 100 + i * 10)}), i, {});
        memory.AddLabel(placed.id, 0, i < 3 ? 0 : 1);
        memory.AddValue(placed.id, 0, {static_cast<float>(i)});
        ids.push_back(placed.id);
    }
    auto query = memory.Project({10});
    auto all = memory.ClassVote(query, 0, 2, 3);
    ASSERT_TRUE(all.has_value());
    EXPECT_GT((*all)[0], 0.5f);
    auto recent = memory.ClassVote(query, 0, 2, 3, 3);  // only the last 3 labels
    ASSERT_TRUE(recent.has_value());
    EXPECT_FLOAT_EQ(1.0f, (*recent)[1]);

    auto value = memory.ValueVote(memory.Project({0}), 0, 1);
    ASSERT_TRUE(value.has_value());
    EXPECT_FLOAT_EQ(0.0f, (*value)[0]);
    EXPECT_FALSE(memory.ClassVote(query, 0, 0, 3).has_value());
}

TEST(PatternMemoryTest, LabelsFollowRecentFeedback) {
    PatternMemory memory(1, 1, {}, 0, MemoryConfig());
    auto placed = memory.Insert(memory.Project({0}), 1, {});
    for (int i = 0; i < 5; ++i) memory.AddLabel(placed.id, 0, 0);
    for (int i = 0; i < 15; ++i) memory.AddLabel(placed.id, 0, 1);
    auto vote = memory.ClassVote(memory.Project({0}), 0, 2, 1);
    EXPECT_GT((*vote)[1], 0.75f);
    memory.AddNegative(placed.id, 0, 1);
    EXPECT_LT((*memory.ClassVote(memory.Project({0}), 0, 2, 1))[1], (*vote)[1]);
}

TEST(PatternMemoryTest, PredictsNextObservedValuesFromTransitions) {
    PatternMemory memory(1, 0, {}, 1, MemoryConfig());
    auto low = memory.Insert(memory.Project({0}), 1, {10.0f}).id;
    auto high = memory.Insert(memory.Project({100}), 2, {20.0f}).id;
    for (int i = 0; i < 10; ++i) {
        memory.RecordTransition(low, high);
        memory.RecordTransition(high, low);
    }
    EXPECT_NEAR(20.0f, (*memory.PredictNextObserved(low))[0], 1e-4f);
    EXPECT_NEAR(10.0f, (*memory.PredictNextObserved(high))[0], 1e-4f);
    EXPECT_FALSE(memory.PredictNextObserved(PatternID(123456789)).has_value());
}

SamplePtr MakeSample(uint64_t step, int cls) {
    auto sample = std::make_shared<Sample>();
    sample->step = step;
    sample->balance_class = cls;
    sample->SetFused({0.0f, static_cast<float>(step), 0.0f});
    return sample;
}

TEST(ReplayBufferTest, StoresSparselyAndDrawsDistinctSamples) {
    Sample sample;
    sample.SetFused({0, 2, 0, 0, 3});
    EXPECT_EQ(2u, sample.indices.size());
    float row[5];
    sample.Fill(row, 5);
    EXPECT_FLOAT_EQ(3.0f, row[4]);
    EXPECT_FLOAT_EQ(0.0f, row[0]);

    ReplayBuffer buffer(10, 1);
    for (uint64_t i = 1; i <= 5; ++i) buffer.Add(MakeSample(i, 0));
    auto drawn = buffer.Draw(5);
    std::set<uint64_t> steps;
    for (const auto& s : drawn) steps.insert(s->step);
    EXPECT_EQ(5u, steps.size());
    EXPECT_EQ(5u, buffer.Draw(50).size());
}

TEST(ReplayBufferTest, EvictionKeepsClassesBalanced) {
    ReplayBuffer buffer(100, 2);
    // 90% of samples are class 0
    for (uint64_t i = 1; i <= 2000; ++i) buffer.Add(MakeSample(i, i % 10 == 0 ? 1 : 0));
    EXPECT_EQ(100u, buffer.Size());
    std::map<int, int> counts;
    for (const auto& s : buffer.Draw(100)) counts[s->balance_class]++;
    EXPECT_GE(counts[1], 35);
    // Old samples remain alongside recent ones
    uint64_t oldest = 2000;
    for (const auto& s : buffer.Draw(100)) oldest = std::min(oldest, s->step);
    EXPECT_LT(oldest, 1500u);
}

TEST(PageHinkleyTest, QuietOnStationaryNoisyLosses) {
    std::mt19937 rng(10);
    std::bernoulli_distribution error(0.3);
    std::normal_distribution<double> squared(1.0, 0.5);
    PageHinkley rate, regression;
    for (int i = 0; i < 100000; ++i) {
        ASSERT_FALSE(rate.Add(error(rng) ? 1.0 : 0.0)) << i;
        ASSERT_FALSE(regression.Add(std::fabs(squared(rng)))) << i;
    }
}

TEST(PageHinkleyTest, QuietOnRareErrorsOfAnAccurateModel) {
    std::mt19937 rng(11);
    std::bernoulli_distribution error(0.02);
    PageHinkley test;
    int alarms = 0;
    for (int i = 0; i < 100000; ++i) alarms += test.Add(error(rng) ? 1.0 : 0.0);
    EXPECT_LE(alarms, 2);
}

TEST(PageHinkleyTest, DetectsAnIncreaseQuickly) {
    std::mt19937 rng(12);
    std::bernoulli_distribution before(0.05), after(0.6);
    PageHinkley test;
    for (int i = 0; i < 2000; ++i) ASSERT_FALSE(test.Add(before(rng) ? 1.0 : 0.0));
    int delay = 0;
    while (!test.Add(after(rng) ? 1.0 : 0.0)) ++delay;
    EXPECT_LT(delay, 30);
    EXPECT_NEAR(0.05, test.MeanBeforeDrift(), 0.03);
    EXPECT_EQ(0u, test.Count());  // restarted
}

TEST(AgentNetworkTest, WideningKeepsOutputsAndStillTrains) {
    AgentNetwork network(Layout(), {{"c", HeadKind::CLASSIFIER, 3}, {"v", HeadKind::REGRESSOR, 2}}, 1e-2f, 6);
    std::mt19937 rng(13);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Matrix x(16, 12);
    for (size_t i = 0; i < x.Size(); ++i) x.Data()[i] = normal(rng);
    HeadTarget value;
    value.values = {1.0f, -1.0f};
    std::vector<const HeadTarget*> none(16, nullptr), values(16, &value);
    for (int i = 0; i < 20; ++i) network.TrainBatch(x, {none, values}, {1.0f, 1.0f}, 5.0f);

    Matrix before0 = network.Infer(x, 0), before1 = network.Infer(x, 1);
    size_t parameters = network.ParameterCount();
    network.Widen(48);
    EXPECT_EQ(48u, network.TrunkWidth());
    EXPECT_GT(network.ParameterCount(), parameters);
    Matrix after0 = network.Infer(x, 0), after1 = network.Infer(x, 1);
    for (size_t i = 0; i < before0.Size(); ++i) EXPECT_NEAR(before0.Data()[i], after0.Data()[i], 1e-4f);
    for (size_t i = 0; i < before1.Size(); ++i) EXPECT_NEAR(before1.Data()[i], after1.Data()[i], 1e-4f);

    float first = 0.0f, last = 0.0f;
    for (int i = 0; i < 200; ++i) {
        float loss = network.TrainBatch(x, {none, values}, {1.0f, 1.0f}, 5.0f)[1];
        if (i == 0) first = loss;
        last = loss;
    }
    EXPECT_LT(last, first);
    EXPECT_TRUE(network.AllFinite());
    network.AllParameters().front().value->Data()[0] = std::nanf("");
    EXPECT_FALSE(network.AllFinite());
}

TEST(AgentNetworkTest, MemoryLearnsADelayedCue) {
    // The sign of input 0 at the first step decides the class at the last;
    // the other steps show noise only
    std::vector<ChannelLayout> layout = {{"x", InputType::NUMERIC, 0, 2, 0}};
    AgentNetwork network(layout, {{"c", HeadKind::CLASSIFIER, 2}}, 1e-2f, 7, 32, 512, 16);
    EXPECT_EQ(16u, network.MemoryUnits());
    std::mt19937 rng(14);
    std::normal_distribution<float> noise(0.0f, 1.0f);
    const size_t steps = 8, rows = 16;
    auto episode = [&](std::vector<Matrix>& inputs, std::vector<int>& labels) {
        inputs.assign(steps, Matrix(rows, 2));
        labels.assign(rows, 0);
        for (size_t r = 0; r < rows; ++r) {
            labels[r] = noise(rng) > 0.0f ? 1 : 0;
            inputs[0](r, 0) = labels[r] ? 1.0f : -1.0f;
            for (size_t t = 1; t < steps; ++t) inputs[t](r, 1) = noise(rng);
        }
    };
    std::vector<HeadTarget> label_targets(2);
    label_targets[0].label = 0;
    label_targets[1].label = 1;
    for (int iteration = 0; iteration < 300; ++iteration) {
        SequenceBatch batch;
        std::vector<int> labels;
        episode(batch.inputs, labels);
        batch.initial = Matrix(rows, 16);
        std::vector<const HeadTarget*> targets(steps * rows, nullptr);
        for (size_t r = 0; r < rows; ++r) targets[(steps - 1) * rows + r] = &label_targets[labels[r]];
        network.TrainSequences(batch, {targets}, {1.0f}, 5.0f);
    }
    // Online: advance the state step by step, then read the head
    std::vector<Matrix> inputs;
    std::vector<int> labels;
    episode(inputs, labels);
    Matrix state(rows, 16);
    for (size_t t = 0; t < steps; ++t) state = network.Step(inputs[t], state);
    Matrix logits = network.Infer(inputs.back(), 0, &state);
    int correct = 0;
    for (size_t r = 0; r < rows; ++r) correct += (logits(r, 1) > logits(r, 0) ? 1 : 0) == labels[r];
    EXPECT_GE(correct, 15);
    EXPECT_THROW(network.TrainBatch(inputs[0], {std::vector<const HeadTarget*>(rows, nullptr)}, {1.0f}, 5.0f),
                 std::logic_error);
}

TEST(AgentNetworkTest, MemoryNetworkWidensCopiesAndSerializes) {
    AgentNetwork network(Layout(), {{"c", HeadKind::CLASSIFIER, 3}, {"v", HeadKind::REGRESSOR, 2}}, 1e-2f, 8, 32,
                         512, 6);
    std::mt19937 rng(15);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Matrix x(5, 12), state(5, 6);
    for (size_t i = 0; i < x.Size(); ++i) x.Data()[i] = normal(rng);
    for (size_t i = 0; i < state.Size(); ++i) state.Data()[i] = 0.5f * normal(rng);
    Matrix next = network.Step(x, state);
    Matrix before0 = network.Infer(x, 0, &next), before1 = network.Infer(x, 1, &next);

    std::stringstream buffer;
    network.Serialize(buffer);
    auto loaded = AgentNetwork::Deserialize(buffer);
    AgentNetwork copy(network);
    EXPECT_EQ(next.Values(), loaded->Step(x, state).Values());
    EXPECT_EQ(before0.Values(), loaded->Infer(x, 0, &next).Values());
    EXPECT_EQ(before1.Values(), copy.Infer(x, 1, &next).Values());
    EXPECT_EQ(network.ParameterCount(), loaded->ParameterCount());

    network.Widen(48);
    Matrix wide_next = network.Step(x, state);
    for (size_t i = 0; i < next.Size(); ++i) EXPECT_NEAR(next.Data()[i], wide_next.Data()[i], 1e-4f);
    Matrix after0 = network.Infer(x, 0, &next), after1 = network.Infer(x, 1, &next);
    for (size_t i = 0; i < before0.Size(); ++i) EXPECT_NEAR(before0.Data()[i], after0.Data()[i], 1e-4f);
    for (size_t i = 0; i < before1.Size(); ++i) EXPECT_NEAR(before1.Data()[i], after1.Data()[i], 1e-4f);

    AgentNetwork classes(Layout(), {{"c", HeadKind::CLASSIFIER, 2}}, 1e-2f, 9, 32, 512, 6);
    Matrix two = classes.Infer(x, 0, &next);
    classes.GrowClasses(0, 4);
    Matrix four = classes.Infer(x, 0, &next);
    for (size_t r = 0; r < 5; ++r) {
        for (size_t c = 0; c < 2; ++c) EXPECT_FLOAT_EQ(two(r, c), four(r, c));
    }
    EXPECT_THROW(network.Infer(x, 0, &x), std::invalid_argument);  // wrong state size
}

class WordSequenceTest : public ::testing::TestWithParam<bool> {};

TEST_P(WordSequenceTest, GradientsMatchFiniteDifferences) {
    const bool attention = GetParam();
    auto pretrained = std::make_shared<Matrix>(Matrix::FromRows({{0.5f, -0.2f, 0.1f}, {-0.3f, 0.4f, 0.2f}}));
    WordSequenceModel model(4, 3, 0, 5, true, attention, pretrained, false, 3);
    EXPECT_EQ((attention ? 2 * 5u : 2 * 2 * 5u) + 3u, model.OutputSize());
    // Rows: ids (2 pretrained, 3 learned + 1 shared): lengths 4, 2, 0
    Matrix ids = Matrix::FromRows({{1, 3, 4, 2}, {5, 6, 0, 0}, {0, 0, 0, 0}});
    std::mt19937 rng(4);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    Matrix weights(3, model.OutputSize());
    for (size_t i = 0; i < weights.Size(); ++i) weights.Data()[i] = normal(rng);
    auto loss = [&](WordSequenceModel& m) {
        Matrix out = m.Infer(ids);
        double sum = 0.0;
        for (size_t i = 0; i < out.Size(); ++i) sum += static_cast<double>(out.Data()[i]) * weights.Data()[i];
        return sum;
    };
    Matrix out = model.Forward(ids, true);
    EXPECT_EQ(out.Values(), model.Infer(ids).Values());
    for (size_t j = 0; j < model.OutputSize(); ++j) EXPECT_EQ(0.0f, out(2, j));  // no words
    model.ZeroGradients();
    model.Backward(weights);
    const float epsilon = 1e-3f;
    for (auto& parameter : model.Parameters()) {
        for (size_t i = 0; i < parameter.value->Size(); ++i) {
            float saved = parameter.value->Data()[i];
            parameter.value->Data()[i] = saved + epsilon;
            double plus = loss(model);
            parameter.value->Data()[i] = saved - epsilon;
            double minus = loss(model);
            parameter.value->Data()[i] = saved;
            double numeric = (plus - minus) / (2.0 * epsilon);
            ASSERT_NEAR(numeric, parameter.gradient->Data()[i], 2e-2 * std::max(1.0, std::fabs(numeric)));
        }
    }
    // The pretrained vectors are fixed and shared by copies; a deserialized
    // model needs them attached
    WordSequenceModel copy(model);
    EXPECT_EQ(out.Values(), copy.Infer(ids).Values());
    std::stringstream buffer;
    model.Serialize(buffer);
    WordSequenceModel loaded = WordSequenceModel::Deserialize(buffer);
    EXPECT_TRUE(loaded.NeedsPretrained());
    EXPECT_THROW(loaded.Infer(ids), std::logic_error);
    EXPECT_THROW(loaded.AttachPretrained(std::make_shared<Matrix>(3, 3)), std::invalid_argument);
    loaded.AttachPretrained(pretrained);
    EXPECT_EQ(out.Values(), loaded.Infer(ids).Values());
}

INSTANTIATE_TEST_SUITE_P(Pooling, WordSequenceTest, ::testing::Values(false, true),
                         [](const auto& info) { return info.param ? "Attention" : "FinalAndMean"; });

TEST(AgentNetworkTest, ReadsWordOrder) {
    // A TEXT channel of 4 features and 3 word ids: the class is which of
    // words 1 and 2 comes first (both always occur; the features are the same)
    ChannelLayout text{"t", InputType::TEXT, 0, 4 + 3 + 1, 2};
    text.sequence_length = 3;
    text.learned_words = 4;
    text.word_dimension = 4;
    text.sequence_units = 8;
    AgentNetwork network({text}, {{"c", HeadKind::CLASSIFIER, 2}}, 1e-2f, 10);
    Matrix x(32, 8);
    std::vector<HeadTarget> labels(2);
    labels[0].label = 0;
    labels[1].label = 1;
    std::vector<const HeadTarget*> targets(32);
    for (size_t r = 0; r < 32; ++r) {
        bool first = r % 2 == 0;
        x(r, 0) = 0.5f;
        x(r, 4) = first ? 1.0f : 2.0f;
        x(r, 5) = r % 4 < 2 ? 3.0f : (first ? 2.0f : 1.0f);
        x(r, 6) = r % 4 < 2 ? (first ? 2.0f : 1.0f) : 0.0f;
        targets[r] = &labels[first ? 0 : 1];
    }
    for (int i = 0; i < 300; ++i) network.TrainBatch(x, {targets}, {1.0f}, 5.0f);
    Matrix logits = network.Infer(x, 0);
    int correct = 0;
    for (size_t r = 0; r < 32; ++r) correct += (logits(r, 1) > logits(r, 0)) == (r % 2 == 1);
    EXPECT_EQ(32, correct);
}

TEST(ReplayBufferTest, StaysWithinByteBudget) {
    ReplayBuffer buffer(100000, 3, 20000);
    for (uint64_t i = 1; i <= 1000; ++i) {
        buffer.Add(MakeSample(i, 0));
        ASSERT_LE(buffer.MemoryBytes(), 20000u);
    }
    EXPECT_GT(buffer.Size(), 10u);
    EXPECT_LT(buffer.Size(), 1000u);
    // Growing a stored sample is measured when it is added again
    auto samples = buffer.Draw(1);
    samples[0]->targets.resize(50);
    buffer.Add(samples[0]);
    EXPECT_LE(buffer.MemoryBytes(), 20000u);
}

}  // namespace
}  // namespace agent
}  // namespace dpan
