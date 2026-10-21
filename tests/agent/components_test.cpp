// File: tests/agent/components_test.cpp
//
// Agent internals: network, pattern memory, replay buffer
#include "agent/network.hpp"
#include "agent/pattern_memory.hpp"
#include "agent/replay_buffer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <map>
#include <set>

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

TEST(PatternMemoryTest, EvictsLeastRecentlyUsed) {
    PatternMemory memory(1, 0, {}, 0, MemoryConfig(2));
    auto p1 = memory.Insert(memory.Project({0}), 1, {});
    auto p2 = memory.Insert(memory.Project({100}), 2, {});
    memory.Insert(memory.Project({0}), 3, {});  // touches p1
    memory.RecordTransition(p2.id, p1.id);
    EXPECT_EQ(1u, memory.Associations().GetAssociationCount());
    auto p3 = memory.Insert(memory.Project({-100}), 4, {});
    EXPECT_TRUE(p3.created);
    EXPECT_EQ(2u, memory.Size());
    EXPECT_TRUE(memory.Contains(p1.id));
    EXPECT_FALSE(memory.Contains(p2.id));
    EXPECT_EQ(0u, memory.Associations().GetAssociationCount());  // edges of p2 removed
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

}  // namespace
}  // namespace agent
}  // namespace dpan
