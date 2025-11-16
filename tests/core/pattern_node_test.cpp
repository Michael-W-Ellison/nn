// File: tests/core/pattern_node_test.cpp
#include "core/pattern_node.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace dpan {
namespace {

// Helper to create a simple pattern node
PatternNode CreateTestNode() {
    PatternID id = PatternID::Generate();
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    return PatternNode(id, data, PatternType::ATOMIC);
}

// ============================================================================
// Constructor and Basic Getters Tests
// ============================================================================

TEST(PatternNodeTest, ConstructorInitializesCorrectly) {
    PatternID id = PatternID::Generate();
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);

    PatternNode node(id, data, PatternType::COMPOSITE);

    EXPECT_EQ(id, node.GetID());
    EXPECT_EQ(PatternType::COMPOSITE, node.GetType());
    EXPECT_FLOAT_EQ(0.5f, node.GetActivationThreshold());
    EXPECT_FLOAT_EQ(0.0f, node.GetBaseActivation());
    EXPECT_FLOAT_EQ(0.5f, node.GetConfidenceScore());
    EXPECT_EQ(0u, node.GetAccessCount());
}

TEST(PatternNodeTest, DefaultConstructorCreatesValidNode) {
    PatternNode node;
    EXPECT_FALSE(node.GetID().IsValid());
    EXPECT_EQ(PatternType::ATOMIC, node.GetType());
}

TEST(PatternNodeTest, GetCreationTimeIsValid) {
    Timestamp before = Timestamp::Now();
    PatternNode node = CreateTestNode();
    Timestamp after = Timestamp::Now();

    Timestamp creation = node.GetCreationTime();
    EXPECT_GE(creation, before);
    EXPECT_LE(creation, after);
}

// ============================================================================
// Activation Threshold Tests
// ============================================================================

TEST(PatternNodeTest, SetAndGetActivationThreshold) {
    PatternNode node = CreateTestNode();

    node.SetActivationThreshold(0.75f);
    EXPECT_FLOAT_EQ(0.75f, node.GetActivationThreshold());

    node.SetActivationThreshold(0.25f);
    EXPECT_FLOAT_EQ(0.25f, node.GetActivationThreshold());
}

TEST(PatternNodeTest, SetAndGetBaseActivation) {
    PatternNode node = CreateTestNode();

    node.SetBaseActivation(0.3f);
    EXPECT_FLOAT_EQ(0.3f, node.GetBaseActivation());

    node.SetBaseActivation(-0.5f);
    EXPECT_FLOAT_EQ(-0.5f, node.GetBaseActivation());
}

// ============================================================================
// Confidence Score Tests
// ============================================================================

TEST(PatternNodeTest, SetConfidenceScoreClampsToRange) {
    PatternNode node = CreateTestNode();

    node.SetConfidenceScore(1.5f);  // Too high
    EXPECT_FLOAT_EQ(1.0f, node.GetConfidenceScore());

    node.SetConfidenceScore(-0.5f);  // Too low
    EXPECT_FLOAT_EQ(0.0f, node.GetConfidenceScore());

    node.SetConfidenceScore(0.7f);  // Valid
    EXPECT_FLOAT_EQ(0.7f, node.GetConfidenceScore());
}

TEST(PatternNodeTest, UpdateConfidenceDelta) {
    PatternNode node = CreateTestNode();
    node.SetConfidenceScore(0.5f);

    node.UpdateConfidence(0.2f);
    EXPECT_FLOAT_EQ(0.7f, node.GetConfidenceScore());

    node.UpdateConfidence(-0.3f);
    EXPECT_FLOAT_EQ(0.4f, node.GetConfidenceScore());

    // Test clamping
    node.UpdateConfidence(1.0f);
    EXPECT_FLOAT_EQ(1.0f, node.GetConfidenceScore());

    node.UpdateConfidence(-2.0f);
    EXPECT_FLOAT_EQ(0.0f, node.GetConfidenceScore());
}

// ============================================================================
// Access Tracking Tests
// ============================================================================

TEST(PatternNodeTest, RecordAccessIncrementsCount) {
    PatternNode node = CreateTestNode();

    EXPECT_EQ(0u, node.GetAccessCount());

    node.RecordAccess();
    EXPECT_EQ(1u, node.GetAccessCount());

    node.RecordAccess();
    EXPECT_EQ(2u, node.GetAccessCount());
}

TEST(PatternNodeTest, RecordAccessUpdatesTimestamp) {
    PatternNode node = CreateTestNode();

    Timestamp initial = node.GetLastAccessed();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    node.RecordAccess();
    Timestamp after = node.GetLastAccessed();

    EXPECT_GT(after, initial);
}

TEST(PatternNodeTest, IncrementAccessCountByValue) {
    PatternNode node = CreateTestNode();

    node.IncrementAccessCount(5);
    EXPECT_EQ(5u, node.GetAccessCount());

    node.IncrementAccessCount(3);
    EXPECT_EQ(8u, node.GetAccessCount());
}

// ============================================================================
// Sub-Patterns Tests
// ============================================================================

TEST(PatternNodeTest, AddSubPattern) {
    PatternNode node = CreateTestNode();

    EXPECT_FALSE(node.HasSubPatterns());

    PatternID sub_id1 = PatternID::Generate();
    node.AddSubPattern(sub_id1);

    EXPECT_TRUE(node.HasSubPatterns());

    auto subs = node.GetSubPatterns();
    EXPECT_EQ(1u, subs.size());
    EXPECT_EQ(sub_id1, subs[0]);
}

TEST(PatternNodeTest, AddMultipleSubPatterns) {
    PatternNode node = CreateTestNode();

    PatternID sub_id1 = PatternID::Generate();
    PatternID sub_id2 = PatternID::Generate();
    PatternID sub_id3 = PatternID::Generate();

    node.AddSubPattern(sub_id1);
    node.AddSubPattern(sub_id2);
    node.AddSubPattern(sub_id3);

    auto subs = node.GetSubPatterns();
    EXPECT_EQ(3u, subs.size());
}

TEST(PatternNodeTest, AddDuplicateSubPatternIgnored) {
    PatternNode node = CreateTestNode();

    PatternID sub_id = PatternID::Generate();

    node.AddSubPattern(sub_id);
    node.AddSubPattern(sub_id);  // Duplicate

    auto subs = node.GetSubPatterns();
    EXPECT_EQ(1u, subs.size());
}

TEST(PatternNodeTest, RemoveSubPattern) {
    PatternNode node = CreateTestNode();

    PatternID sub_id1 = PatternID::Generate();
    PatternID sub_id2 = PatternID::Generate();

    node.AddSubPattern(sub_id1);
    node.AddSubPattern(sub_id2);

    node.RemoveSubPattern(sub_id1);

    auto subs = node.GetSubPatterns();
    EXPECT_EQ(1u, subs.size());
    EXPECT_EQ(sub_id2, subs[0]);
}

TEST(PatternNodeTest, RemoveNonExistentSubPatternIsNoOp) {
    PatternNode node = CreateTestNode();

    PatternID sub_id1 = PatternID::Generate();
    PatternID sub_id2 = PatternID::Generate();

    node.AddSubPattern(sub_id1);
    node.RemoveSubPattern(sub_id2);  // Not in list

    auto subs = node.GetSubPatterns();
    EXPECT_EQ(1u, subs.size());
}

// ============================================================================
// Activation Computation Tests
// ============================================================================

TEST(PatternNodeTest, ComputeActivationWithMatchingFeatures) {
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 0.0f;
    fv[2] = 0.0f;

    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);

    // Test with identical features (cosine similarity = 1.0)
    FeatureVector input = fv;
    float activation = node.ComputeActivation(input);

    // Activation = (similarity + base) / 2 = (1.0 + 0.0) / 2 = 0.5
    EXPECT_FLOAT_EQ(0.5f, activation);
}

TEST(PatternNodeTest, ComputeActivationWithBaseActivation) {
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 0.0f;
    fv[2] = 0.0f;

    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);
    node.SetBaseActivation(0.2f);

    FeatureVector input = fv;
    float activation = node.ComputeActivation(input);

    // Activation = (1.0 + 0.2) / 2 = 0.6
    EXPECT_FLOAT_EQ(0.6f, activation);
}

TEST(PatternNodeTest, ComputeActivationWithDimensionMismatch) {
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;

    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);
    node.SetBaseActivation(0.3f);

    // Different dimension input
    FeatureVector input(5);
    float activation = node.ComputeActivation(input);

    // Should return base activation on dimension mismatch
    EXPECT_FLOAT_EQ(0.3f, activation);
}

TEST(PatternNodeTest, IsActivatedThresholdCheck) {
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 0.0f;
    fv[2] = 0.0f;

    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);

    node.SetActivationThreshold(0.4f);

    FeatureVector input = fv;  // Perfect match, activation = 0.5

    EXPECT_TRUE(node.IsActivated(input));

    node.SetActivationThreshold(0.6f);
    EXPECT_FALSE(node.IsActivated(input));
}

// ============================================================================
// Age Calculation Tests
// ============================================================================

TEST(PatternNodeTest, GetAgeIncreases) {
    PatternNode node = CreateTestNode();

    auto age1 = node.GetAge();

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto age2 = node.GetAge();

    EXPECT_GT(age2, age1);
}

// ============================================================================
// Serialization Tests
// ============================================================================

TEST(PatternNodeTest, SerializationRoundTrip) {
    PatternNode original = CreateTestNode();
    original.SetActivationThreshold(0.75f);
    original.SetBaseActivation(0.2f);
    original.SetConfidenceScore(0.8f);
    original.RecordAccess();
    original.RecordAccess();

    PatternID sub1 = PatternID::Generate();
    PatternID sub2 = PatternID::Generate();
    original.AddSubPattern(sub1);
    original.AddSubPattern(sub2);

    std::stringstream ss;
    original.Serialize(ss);

    PatternNode deserialized = PatternNode::Deserialize(ss);

    EXPECT_EQ(original.GetID(), deserialized.GetID());
    EXPECT_EQ(original.GetType(), deserialized.GetType());
    EXPECT_FLOAT_EQ(original.GetActivationThreshold(), deserialized.GetActivationThreshold());
    EXPECT_FLOAT_EQ(original.GetBaseActivation(), deserialized.GetBaseActivation());
    EXPECT_FLOAT_EQ(original.GetConfidenceScore(), deserialized.GetConfidenceScore());
    EXPECT_EQ(original.GetAccessCount(), deserialized.GetAccessCount());

    auto orig_subs = original.GetSubPatterns();
    auto deser_subs = deserialized.GetSubPatterns();
    EXPECT_EQ(orig_subs.size(), deser_subs.size());
}

// ============================================================================
// ToString Tests
// ============================================================================

TEST(PatternNodeTest, ToStringProducesReadableOutput) {
    PatternNode node = CreateTestNode();

    std::string str = node.ToString();

    EXPECT_NE(std::string::npos, str.find("PatternNode"));
    EXPECT_NE(std::string::npos, str.find("id="));
    EXPECT_NE(std::string::npos, str.find("type="));
}

// ============================================================================
// Memory Estimation Tests
// ============================================================================

TEST(PatternNodeTest, EstimateMemoryUsageIsReasonable) {
    PatternNode node = CreateTestNode();

    size_t memory = node.EstimateMemoryUsage();

    // Should be at least sizeof(PatternNode)
    EXPECT_GE(memory, sizeof(PatternNode));

    // Should be less than 10KB for a simple node
    EXPECT_LT(memory, 10000u);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST(PatternNodeTest, ConcurrentRecordAccessIsSafe) {
    PatternNode node = CreateTestNode();

    constexpr int kNumThreads = 10;
    constexpr int kAccessesPerThread = 100;

    std::vector<std::thread> threads;
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&node]() {
            for (int j = 0; j < kAccessesPerThread; ++j) {
                node.RecordAccess();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(kNumThreads * kAccessesPerThread, node.GetAccessCount());
}

TEST(PatternNodeTest, ConcurrentSubPatternModificationIsSafe) {
    PatternNode node = CreateTestNode();

    constexpr int kNumThreads = 10;
    std::vector<std::thread> threads;
    std::vector<PatternID> ids;

    // Pre-generate IDs
    for (int i = 0; i < kNumThreads; ++i) {
        ids.push_back(PatternID::Generate());
    }

    // Add sub-patterns concurrently
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&node, id = ids[i]]() {
            node.AddSubPattern(id);
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto subs = node.GetSubPatterns();
    EXPECT_EQ(kNumThreads, subs.size());
}

} // namespace
} // namespace dpan
