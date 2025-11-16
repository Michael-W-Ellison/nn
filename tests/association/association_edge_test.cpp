// File: tests/association/association_edge_test.cpp
#include "association/association_edge.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <sstream>

using namespace dpan;

// ============================================================================
// Constructor Tests
// ============================================================================

TEST(AssociationEdgeTest, DefaultConstructor) {
    AssociationEdge edge;
    EXPECT_EQ(0.5f, edge.GetStrength());
    EXPECT_EQ(0u, edge.GetCoOccurrenceCount());
}

TEST(AssociationEdgeTest, ParameterizedConstructor) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();

    AssociationEdge edge(src, tgt, AssociationType::CAUSAL, 0.8f);

    EXPECT_EQ(src, edge.GetSource());
    EXPECT_EQ(tgt, edge.GetTarget());
    EXPECT_EQ(AssociationType::CAUSAL, edge.GetType());
    EXPECT_FLOAT_EQ(0.8f, edge.GetStrength());
}

TEST(AssociationEdgeTest, StrengthBounding) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();

    // Test upper bound
    AssociationEdge edge1(src, tgt, AssociationType::SPATIAL, 1.5f);
    EXPECT_FLOAT_EQ(1.0f, edge1.GetStrength());

    // Test lower bound
    AssociationEdge edge2(src, tgt, AssociationType::SPATIAL, -0.5f);
    EXPECT_FLOAT_EQ(0.0f, edge2.GetStrength());
}

// ============================================================================
// Strength Management Tests
// ============================================================================

TEST(AssociationEdgeTest, SetStrength) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CATEGORICAL);

    edge.SetStrength(0.75f);
    EXPECT_FLOAT_EQ(0.75f, edge.GetStrength());

    // Test bounding
    edge.SetStrength(1.5f);
    EXPECT_FLOAT_EQ(1.0f, edge.GetStrength());

    edge.SetStrength(-0.5f);
    EXPECT_FLOAT_EQ(0.0f, edge.GetStrength());
}

TEST(AssociationEdgeTest, AdjustStrength) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CATEGORICAL, 0.5f);

    // Positive adjustment
    edge.AdjustStrength(0.3f);
    EXPECT_FLOAT_EQ(0.8f, edge.GetStrength());

    // Adjustment should respect upper bound
    edge.AdjustStrength(0.5f);
    EXPECT_FLOAT_EQ(1.0f, edge.GetStrength());

    // Negative adjustment
    edge.AdjustStrength(-0.3f);
    EXPECT_FLOAT_EQ(0.7f, edge.GetStrength());

    // Adjustment should respect lower bound
    edge.AdjustStrength(-1.0f);
    EXPECT_FLOAT_EQ(0.0f, edge.GetStrength());
}

// ============================================================================
// Co-occurrence Tracking Tests
// ============================================================================

TEST(AssociationEdgeTest, CoOccurrenceTracking) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL);

    EXPECT_EQ(0u, edge.GetCoOccurrenceCount());

    edge.IncrementCoOccurrence();
    EXPECT_EQ(1u, edge.GetCoOccurrenceCount());

    edge.IncrementCoOccurrence(5);
    EXPECT_EQ(6u, edge.GetCoOccurrenceCount());
}

// ============================================================================
// Temporal Correlation Tests
// ============================================================================

TEST(AssociationEdgeTest, TemporalCorrelation) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL);

    EXPECT_FLOAT_EQ(0.0f, edge.GetTemporalCorrelation());

    edge.SetTemporalCorrelation(0.7f);
    EXPECT_FLOAT_EQ(0.7f, edge.GetTemporalCorrelation());

    // Test bounding
    edge.SetTemporalCorrelation(1.5f);
    EXPECT_FLOAT_EQ(1.0f, edge.GetTemporalCorrelation());

    edge.SetTemporalCorrelation(-1.5f);
    EXPECT_FLOAT_EQ(-1.0f, edge.GetTemporalCorrelation());
}

TEST(AssociationEdgeTest, TemporalCorrelationUpdate) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL);

    edge.SetTemporalCorrelation(0.5f);

    // Update with new observation
    edge.UpdateTemporalCorrelation(0.8f, 0.5f);  // learning_rate = 0.5

    // Expected: 0.5 + 0.5 * (0.8 - 0.5) = 0.5 + 0.15 = 0.65
    EXPECT_NEAR(0.65f, edge.GetTemporalCorrelation(), 0.001f);
}

// ============================================================================
// Decay Tests
// ============================================================================

TEST(AssociationEdgeTest, DecayRate) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::SPATIAL);

    edge.SetDecayRate(0.05f);
    EXPECT_FLOAT_EQ(0.05f, edge.GetDecayRate());

    // Negative decay rate should be clamped to 0
    edge.SetDecayRate(-0.1f);
    EXPECT_FLOAT_EQ(0.0f, edge.GetDecayRate());
}

TEST(AssociationEdgeTest, ApplyDecay) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CATEGORICAL, 1.0f);

    edge.SetDecayRate(0.01f);

    // Apply decay for 100 seconds
    // s(t) = s(0) * exp(-0.01 * 100) = 1.0 * exp(-1) ≈ 0.368
    auto elapsed = std::chrono::seconds(100);
    edge.ApplyDecay(elapsed);

    EXPECT_NEAR(0.368f, edge.GetStrength(), 0.01f);
}

TEST(AssociationEdgeTest, ReinforcementTracking) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::FUNCTIONAL);

    Timestamp before = Timestamp::Now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    edge.RecordReinforcement();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Timestamp after = Timestamp::Now();

    Timestamp last_reinforcement = edge.GetLastReinforcement();
    EXPECT_GT(last_reinforcement, before);
    EXPECT_LT(last_reinforcement, after);
}

// ============================================================================
// Context Profile Tests
// ============================================================================

TEST(AssociationEdgeTest, ContextProfile) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::SPATIAL);

    ContextVector context;
    context.Set("temperature", 25.0f);
    context.Set("humidity", 60.0f);

    edge.SetContextProfile(context);

    const ContextVector& retrieved = edge.GetContextProfile();
    EXPECT_FLOAT_EQ(25.0f, retrieved.Get("temperature"));
    EXPECT_FLOAT_EQ(60.0f, retrieved.Get("humidity"));
}

TEST(AssociationEdgeTest, ContextProfileUpdate) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::SPATIAL);

    ContextVector initial;
    initial.Set("temperature", 20.0f);
    edge.SetContextProfile(initial);

    ContextVector observed;
    observed.Set("temperature", 30.0f);
    edge.UpdateContextProfile(observed, 0.5f);

    // Expected: 20 + 0.5 * (30 - 20) = 25
    const ContextVector& updated = edge.GetContextProfile();
    EXPECT_NEAR(25.0f, updated.Get("temperature"), 0.001f);
}

TEST(AssociationEdgeTest, ContextualStrength) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CATEGORICAL, 0.8f);

    // Set context profile
    ContextVector profile;
    profile.Set("time_of_day", 1.0f);
    profile.Set("location", 0.5f);
    edge.SetContextProfile(profile);

    // Test with matching context
    ContextVector matching_context;
    matching_context.Set("time_of_day", 1.0f);
    matching_context.Set("location", 0.5f);

    float contextual_strength = edge.GetContextualStrength(matching_context);
    // Should be close to base strength since contexts match
    EXPECT_NEAR(0.8f, contextual_strength, 0.1f);

    // Test with non-matching context
    ContextVector non_matching;
    non_matching.Set("time_of_day", 0.0f);
    non_matching.Set("location", 0.0f);

    float weak_strength = edge.GetContextualStrength(non_matching);
    // Should be weaker than base strength
    EXPECT_LT(weak_strength, 0.8f);
}

TEST(AssociationEdgeTest, ContextualStrengthWithEmptyProfile) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CATEGORICAL, 0.7f);

    ContextVector current_context;
    current_context.Set("test", 1.0f);

    // With empty profile, should return base strength
    float strength = edge.GetContextualStrength(current_context);
    EXPECT_FLOAT_EQ(0.7f, strength);
}

// ============================================================================
// Age and Activity Tests
// ============================================================================

TEST(AssociationEdgeTest, Age) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto age = edge.GetAge();
    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(age).count();

    EXPECT_GE(age_ms, 100);
    EXPECT_LT(age_ms, 200);  // Allow some overhead
}

TEST(AssociationEdgeTest, IsActive) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::FUNCTIONAL);

    // Should be active immediately after creation
    EXPECT_TRUE(edge.IsActive(std::chrono::seconds(1)));

    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    // Should not be active after 1.5 seconds with max_idle_time = 1s
    EXPECT_FALSE(edge.IsActive(std::chrono::seconds(1)));

    // Should be active with larger threshold
    EXPECT_TRUE(edge.IsActive(std::chrono::seconds(2)));
}

// ============================================================================
// Serialization Tests
// ============================================================================

TEST(AssociationEdgeTest, SerializationRoundTrip) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge original(src, tgt, AssociationType::COMPOSITIONAL, 0.75f);

    original.IncrementCoOccurrence(10);
    original.SetTemporalCorrelation(0.6f);
    original.SetDecayRate(0.02f);

    ContextVector context;
    context.Set("dim1", 1.0f);
    context.Set("dim2", 2.0f);
    original.SetContextProfile(context);

    // Serialize
    std::stringstream ss;
    original.Serialize(ss);

    // Deserialize
    auto deserialized = AssociationEdge::Deserialize(ss);
    ASSERT_NE(deserialized, nullptr);

    // Verify
    EXPECT_EQ(original.GetSource(), deserialized->GetSource());
    EXPECT_EQ(original.GetTarget(), deserialized->GetTarget());
    EXPECT_EQ(original.GetType(), deserialized->GetType());
    EXPECT_FLOAT_EQ(original.GetStrength(), deserialized->GetStrength());
    EXPECT_EQ(original.GetCoOccurrenceCount(), deserialized->GetCoOccurrenceCount());
    EXPECT_FLOAT_EQ(original.GetTemporalCorrelation(), deserialized->GetTemporalCorrelation());
    EXPECT_FLOAT_EQ(original.GetDecayRate(), deserialized->GetDecayRate());
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST(AssociationEdgeTest, ThreadSafeStrengthUpdates) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL, 0.5f);

    constexpr int kNumThreads = 10;
    constexpr int kUpdatesPerThread = 1000;

    std::vector<std::thread> threads;

    // Launch threads that adjust strength
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&edge]() {
            for (int j = 0; j < kUpdatesPerThread; ++j) {
                edge.AdjustStrength(0.0001f);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify final strength (should be bounded at 1.0)
    EXPECT_FLOAT_EQ(1.0f, edge.GetStrength());
}

TEST(AssociationEdgeTest, ThreadSafeCoOccurrenceUpdates) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL);

    constexpr int kNumThreads = 10;
    constexpr int kUpdatesPerThread = 100;

    std::vector<std::thread> threads;

    // Launch threads that increment co-occurrence
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&edge]() {
            for (int j = 0; j < kUpdatesPerThread; ++j) {
                edge.IncrementCoOccurrence();
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify total count
    EXPECT_EQ(kNumThreads * kUpdatesPerThread, edge.GetCoOccurrenceCount());
}

// ============================================================================
// Utility Tests
// ============================================================================

TEST(AssociationEdgeTest, ToString) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL, 0.8f);

    std::string str = edge.ToString();

    // Verify string contains key information
    EXPECT_NE(str.find("AssociationEdge"), std::string::npos);
    EXPECT_NE(str.find("CAUSAL"), std::string::npos);
    EXPECT_NE(str.find("strength=0.8"), std::string::npos);
}

TEST(AssociationEdgeTest, EstimateMemoryUsage) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::SPATIAL);

    size_t memory = edge.EstimateMemoryUsage();

    // Should be at least the size of the object
    EXPECT_GE(memory, sizeof(AssociationEdge));

    // Add some context and verify memory increases
    ContextVector context;
    context.Set("test1", 1.0f);
    context.Set("test2", 2.0f);
    edge.SetContextProfile(context);

    size_t memory_with_context = edge.EstimateMemoryUsage();
    EXPECT_GT(memory_with_context, memory);
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST(AssociationEdgeTest, Equality) {
    PatternID src1 = PatternID::Generate();
    PatternID tgt1 = PatternID::Generate();
    PatternID src2 = PatternID::Generate();

    AssociationEdge edge1(src1, tgt1, AssociationType::CAUSAL);
    AssociationEdge edge2(src1, tgt1, AssociationType::CAUSAL);
    AssociationEdge edge3(src2, tgt1, AssociationType::CAUSAL);

    EXPECT_TRUE(edge1 == edge2);
    EXPECT_FALSE(edge1 == edge3);
}

TEST(AssociationEdgeTest, Comparison) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();

    AssociationEdge strong(src, tgt, AssociationType::CAUSAL, 0.9f);
    AssociationEdge weak(src, tgt, AssociationType::CAUSAL, 0.3f);

    // operator< sorts by strength descending, so strong < weak
    EXPECT_TRUE(strong < weak);
    EXPECT_FALSE(weak < strong);
}

// ============================================================================
// Association Type Tests
// ============================================================================

TEST(AssociationEdgeTest, DifferentAssociationTypes) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();

    AssociationEdge causal(src, tgt, AssociationType::CAUSAL);
    AssociationEdge categorical(src, tgt, AssociationType::CATEGORICAL);
    AssociationEdge spatial(src, tgt, AssociationType::SPATIAL);
    AssociationEdge functional(src, tgt, AssociationType::FUNCTIONAL);
    AssociationEdge compositional(src, tgt, AssociationType::COMPOSITIONAL);

    EXPECT_EQ(AssociationType::CAUSAL, causal.GetType());
    EXPECT_EQ(AssociationType::CATEGORICAL, categorical.GetType());
    EXPECT_EQ(AssociationType::SPATIAL, spatial.GetType());
    EXPECT_EQ(AssociationType::FUNCTIONAL, functional.GetType());
    EXPECT_EQ(AssociationType::COMPOSITIONAL, compositional.GetType());
}
