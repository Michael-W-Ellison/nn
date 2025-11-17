// File: tests/memory/utility_calculator_test.cpp
#include "memory/utility_calculator.hpp"
#include "core/pattern_node.hpp"
#include "association/association_edge.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <sstream>

using namespace dpan;

// ============================================================================
// Config Validation Tests
// ============================================================================

TEST(UtilityCalculatorTest, ValidConfig) {
    UtilityCalculator::Config config;
    config.frequency_weight = 0.25f;
    config.recency_weight = 0.25f;
    config.association_weight = 0.25f;
    config.confidence_weight = 0.25f;
    config.frequency_decay = 0.01f;
    config.recency_decay = 0.05f;
    config.max_access_count = 1000.0f;

    EXPECT_TRUE(config.IsValid());

    // Should not throw
    EXPECT_NO_THROW(UtilityCalculator calculator(config));
}

TEST(UtilityCalculatorTest, InvalidWeightsSum) {
    UtilityCalculator::Config config;
    config.frequency_weight = 0.5f;
    config.recency_weight = 0.3f;
    config.association_weight = 0.3f;  // Sum > 1.0
    config.confidence_weight = 0.2f;

    EXPECT_FALSE(config.IsValid());

    // Constructor should throw
    EXPECT_THROW(UtilityCalculator calculator(config), std::invalid_argument);
}

TEST(UtilityCalculatorTest, NegativeWeights) {
    UtilityCalculator::Config config;
    config.frequency_weight = -0.1f;  // Negative weight
    config.recency_weight = 0.4f;
    config.association_weight = 0.4f;
    config.confidence_weight = 0.3f;

    EXPECT_FALSE(config.IsValid());

    // Constructor should throw
    EXPECT_THROW(UtilityCalculator calculator(config), std::invalid_argument);
}

TEST(UtilityCalculatorTest, InvalidDecayConstants) {
    UtilityCalculator::Config config;
    config.frequency_weight = 0.25f;
    config.recency_weight = 0.25f;
    config.association_weight = 0.25f;
    config.confidence_weight = 0.25f;
    config.frequency_decay = -0.01f;  // Negative decay
    config.recency_decay = 0.05f;

    EXPECT_FALSE(config.IsValid());

    // Constructor should throw
    EXPECT_THROW(UtilityCalculator calculator(config), std::invalid_argument);

    // Test zero decay
    config.frequency_decay = 0.0f;
    EXPECT_FALSE(config.IsValid());
}

TEST(UtilityCalculatorTest, InvalidMaxAccessCount) {
    UtilityCalculator::Config config;
    config.frequency_weight = 0.25f;
    config.recency_weight = 0.25f;
    config.association_weight = 0.25f;
    config.confidence_weight = 0.25f;
    config.frequency_decay = 0.01f;
    config.recency_decay = 0.05f;
    config.max_access_count = -100.0f;  // Negative max access count

    EXPECT_FALSE(config.IsValid());

    // Constructor should throw
    EXPECT_THROW(UtilityCalculator calculator(config), std::invalid_argument);

    // Test zero max access count
    config.max_access_count = 0.0f;
    EXPECT_FALSE(config.IsValid());
}

// ============================================================================
// Individual Component Tests
// ============================================================================

TEST(UtilityCalculatorTest, FrequencyScoreZeroAccess) {
    UtilityCalculator calculator;

    // Create test pattern and stats
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 0;

    std::vector<AssociationEdge> associations;

    auto breakdown = calculator.GetUtilityBreakdown(pattern, stats, associations);

    // Frequency score should be very close to 0 for zero accesses
    EXPECT_NEAR(0.0f, breakdown.frequency_score, 0.01f);
}

TEST(UtilityCalculatorTest, FrequencyScoreSaturation) {
    UtilityCalculator calculator;

    // Create test pattern and stats
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 10000;  // Very high access count

    std::vector<AssociationEdge> associations;

    auto breakdown = calculator.GetUtilityBreakdown(pattern, stats, associations);

    // Frequency score should saturate close to 1.0
    EXPECT_GT(breakdown.frequency_score, 0.99f);
    EXPECT_LE(breakdown.frequency_score, 1.0f);
}

TEST(UtilityCalculatorTest, RecencyScoreImmediate) {
    UtilityCalculator calculator;

    // Create test pattern and stats
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 1;
    stats.last_access = Timestamp::Now();  // Just accessed

    std::vector<AssociationEdge> associations;

    auto breakdown = calculator.GetUtilityBreakdown(pattern, stats, associations);

    // Recency score should be very close to 1.0 for immediate access
    EXPECT_GT(breakdown.recency_score, 0.99f);
    EXPECT_LE(breakdown.recency_score, 1.0f);
}

TEST(UtilityCalculatorTest, RecencyScoreDecay) {
    UtilityCalculator calculator;

    // Create test pattern and stats
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 1;
    // Last accessed 48 hours ago
    stats.last_access = Timestamp::Now() - std::chrono::hours(48);

    std::vector<AssociationEdge> associations;

    auto breakdown = calculator.GetUtilityBreakdown(pattern, stats, associations);

    // Recency score should have decayed significantly
    // R(p) = exp(-0.05 * 48) ≈ 0.09
    EXPECT_LT(breakdown.recency_score, 0.5f);
    EXPECT_GT(breakdown.recency_score, 0.0f);
}

TEST(UtilityCalculatorTest, AssociationScoreEmpty) {
    UtilityCalculator calculator;

    // Create test pattern and stats
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 5;
    stats.last_access = Timestamp::Now();

    std::vector<AssociationEdge> associations;  // Empty associations

    auto breakdown = calculator.GetUtilityBreakdown(pattern, stats, associations);

    // Association score should be 0 with no associations
    EXPECT_FLOAT_EQ(0.0f, breakdown.association_score);
}

TEST(UtilityCalculatorTest, AssociationScoreMultiple) {
    UtilityCalculator calculator;

    // Create test pattern and stats
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 5;
    stats.last_access = Timestamp::Now();

    // Create multiple associations with varying strengths
    std::vector<AssociationEdge> associations;
    PatternID target1 = PatternID::Generate();
    PatternID target2 = PatternID::Generate();
    PatternID target3 = PatternID::Generate();

    associations.emplace_back(pattern.GetID(), target1, AssociationType::CAUSAL, 0.8f);
    associations.emplace_back(pattern.GetID(), target2, AssociationType::SPATIAL, 0.6f);
    associations.emplace_back(pattern.GetID(), target3, AssociationType::CATEGORICAL, 0.4f);

    auto breakdown = calculator.GetUtilityBreakdown(pattern, stats, associations);

    // Association score should be average: (0.8 + 0.6 + 0.4) / 3 = 0.6
    EXPECT_NEAR(0.6f, breakdown.association_score, 0.01f);
}

TEST(UtilityCalculatorTest, ConfidenceScoreDefault) {
    UtilityCalculator calculator;

    // Create test pattern and stats
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 5;
    stats.last_access = Timestamp::Now();

    std::vector<AssociationEdge> associations;

    auto breakdown = calculator.GetUtilityBreakdown(pattern, stats, associations);

    // Default confidence should be 0.5
    EXPECT_FLOAT_EQ(0.5f, breakdown.confidence_score);
}

// Note: We would add ConfidenceScoreCustom test once PatternNode
// supports custom confidence values

// ============================================================================
// Pattern Utility Tests
// ============================================================================

TEST(UtilityCalculatorTest, PatternUtilityAllFactors) {
    UtilityCalculator calculator;

    // Create pattern with good stats and associations
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 100;
    stats.last_access = Timestamp::Now();

    std::vector<AssociationEdge> associations;
    PatternID target = PatternID::Generate();
    associations.emplace_back(pattern.GetID(), target, AssociationType::CAUSAL, 0.9f);

    float utility = calculator.CalculatePatternUtility(pattern, stats, associations);

    // Utility should be high with good stats
    EXPECT_GT(utility, 0.5f);
    EXPECT_LE(utility, 1.0f);
}

TEST(UtilityCalculatorTest, PatternUtilityNoAssociations) {
    UtilityCalculator calculator;

    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 50;
    stats.last_access = Timestamp::Now();

    std::vector<AssociationEdge> associations;  // Empty

    float utility = calculator.CalculatePatternUtility(pattern, stats, associations);

    // Should still have positive utility from frequency and recency
    EXPECT_GT(utility, 0.0f);
    EXPECT_LE(utility, 1.0f);
}

TEST(UtilityCalculatorTest, PatternUtilityBreakdown) {
    UtilityCalculator calculator;

    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 100;
    stats.last_access = Timestamp::Now();

    std::vector<AssociationEdge> associations;
    PatternID target = PatternID::Generate();
    associations.emplace_back(pattern.GetID(), target, AssociationType::CAUSAL, 0.8f);

    auto breakdown = calculator.GetUtilityBreakdown(pattern, stats, associations);

    // Verify total matches weighted sum
    float expected_total =
        0.3f * breakdown.frequency_score +
        0.3f * breakdown.recency_score +
        0.25f * breakdown.association_score +
        0.15f * breakdown.confidence_score;

    EXPECT_NEAR(expected_total, breakdown.total, 0.01f);
}

TEST(UtilityCalculatorTest, PatternUtilityBounds) {
    UtilityCalculator calculator;

    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    // Test minimum utility (no access, old)
    AccessStats stats_min;
    stats_min.access_count = 0;
    stats_min.last_access = Timestamp::Now() - std::chrono::hours(1000);

    std::vector<AssociationEdge> associations;

    float utility_min = calculator.CalculatePatternUtility(pattern, stats_min, associations);
    EXPECT_GE(utility_min, 0.0f);
    EXPECT_LE(utility_min, 1.0f);

    // Test maximum utility (high access, recent, strong associations)
    AccessStats stats_max;
    stats_max.access_count = 10000;
    stats_max.last_access = Timestamp::Now();

    PatternID target = PatternID::Generate();
    associations.emplace_back(pattern.GetID(), target, AssociationType::CAUSAL, 1.0f);

    float utility_max = calculator.CalculatePatternUtility(pattern, stats_max, associations);
    EXPECT_GE(utility_max, 0.0f);
    EXPECT_LE(utility_max, 1.0f);
}

TEST(UtilityCalculatorTest, PatternUtilityCustomWeights) {
    UtilityCalculator::Config config;
    config.frequency_weight = 0.5f;    // Emphasize frequency
    config.recency_weight = 0.1f;
    config.association_weight = 0.2f;
    config.confidence_weight = 0.2f;

    UtilityCalculator calculator(config);

    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 100;
    stats.last_access = Timestamp::Now() - std::chrono::hours(100);  // Old

    std::vector<AssociationEdge> associations;

    float utility = calculator.CalculatePatternUtility(pattern, stats, associations);

    // Utility should still be reasonable despite old access due to high frequency weight
    EXPECT_GT(utility, 0.3f);
}

// ============================================================================
// Association Utility Tests
// ============================================================================

TEST(UtilityCalculatorTest, AssociationUtilityStrength) {
    UtilityCalculator calculator;

    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();

    // Create strong association
    AssociationEdge strong_edge(source, target, AssociationType::CAUSAL, 0.9f);
    AssociationEdge weak_edge(source, target, AssociationType::CAUSAL, 0.2f);

    AccessStats stats;
    stats.access_count = 10;
    stats.last_access = Timestamp::Now();

    float strong_utility = calculator.CalculateAssociationUtility(strong_edge, stats, stats);
    float weak_utility = calculator.CalculateAssociationUtility(weak_edge, stats, stats);

    // Stronger edge should have higher utility
    EXPECT_GT(strong_utility, weak_utility);
}

TEST(UtilityCalculatorTest, AssociationUtilityEndpointFrequency) {
    UtilityCalculator calculator;

    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();

    AssociationEdge edge(source, target, AssociationType::CAUSAL, 0.7f);

    // High frequency endpoints
    AccessStats high_freq_stats;
    high_freq_stats.access_count = 1000;
    high_freq_stats.last_access = Timestamp::Now();

    // Low frequency endpoints
    AccessStats low_freq_stats;
    low_freq_stats.access_count = 5;
    low_freq_stats.last_access = Timestamp::Now();

    float high_freq_utility = calculator.CalculateAssociationUtility(
        edge, high_freq_stats, high_freq_stats);
    float low_freq_utility = calculator.CalculateAssociationUtility(
        edge, low_freq_stats, low_freq_stats);

    // Higher frequency endpoints should increase utility
    EXPECT_GT(high_freq_utility, low_freq_utility);
}

TEST(UtilityCalculatorTest, AssociationUtilityEndpointRecency) {
    UtilityCalculator calculator;

    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();

    AssociationEdge edge(source, target, AssociationType::CAUSAL, 0.7f);

    // Recent access
    AccessStats recent_stats;
    recent_stats.access_count = 10;
    recent_stats.last_access = Timestamp::Now();

    // Old access
    AccessStats old_stats;
    old_stats.access_count = 10;
    old_stats.last_access = Timestamp::Now() - std::chrono::hours(100);

    float recent_utility = calculator.CalculateAssociationUtility(
        edge, recent_stats, recent_stats);
    float old_utility = calculator.CalculateAssociationUtility(
        edge, old_stats, old_stats);

    // More recent endpoints should increase utility
    EXPECT_GT(recent_utility, old_utility);
}

TEST(UtilityCalculatorTest, AssociationUtilityBounds) {
    UtilityCalculator calculator;

    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();

    AssociationEdge edge(source, target, AssociationType::CAUSAL, 0.5f);

    AccessStats stats;
    stats.access_count = 50;
    stats.last_access = Timestamp::Now();

    float utility = calculator.CalculateAssociationUtility(edge, stats, stats);

    // Utility should be bounded [0, 1]
    EXPECT_GE(utility, 0.0f);
    EXPECT_LE(utility, 1.0f);
}

// ============================================================================
// AccessStats Tests
// ============================================================================

TEST(AccessStatsTest, RecordFirstAccess) {
    AccessStats stats;

    EXPECT_EQ(0u, stats.access_count);

    Timestamp before = Timestamp::Now();
    stats.RecordAccess();
    Timestamp after = Timestamp::Now();

    EXPECT_EQ(1u, stats.access_count);
    EXPECT_GE(stats.last_access, before);
    EXPECT_LE(stats.last_access, after);
    EXPECT_GE(stats.creation_time, before);
    EXPECT_LE(stats.creation_time, after);
    EXPECT_FLOAT_EQ(0.0f, stats.avg_access_interval);
}

TEST(AccessStatsTest, RecordMultipleAccesses) {
    AccessStats stats;

    Timestamp t1 = Timestamp::Now();
    stats.RecordAccess(t1);
    EXPECT_EQ(1u, stats.access_count);

    Timestamp t2 = t1 + std::chrono::seconds(10);
    stats.RecordAccess(t2);
    EXPECT_EQ(2u, stats.access_count);
    EXPECT_FLOAT_EQ(10.0f, stats.avg_access_interval);

    Timestamp t3 = t2 + std::chrono::seconds(20);
    stats.RecordAccess(t3);
    EXPECT_EQ(3u, stats.access_count);
    // EMA: alpha=0.3, avg = 0.3*20 + 0.7*10 = 13
    EXPECT_NEAR(13.0f, stats.avg_access_interval, 0.1f);
}

TEST(AccessStatsTest, TimeSinceLastAccess) {
    AccessStats stats;

    Timestamp past = Timestamp::Now() - std::chrono::seconds(100);
    stats.RecordAccess(past);

    auto time_since = stats.TimeSinceLastAccess();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(time_since).count();

    // Should be approximately 100 seconds
    EXPECT_GE(seconds, 99);
    EXPECT_LE(seconds, 101);
}

TEST(AccessStatsTest, AccessAge) {
    AccessStats stats;

    Timestamp past = Timestamp::Now() - std::chrono::seconds(50);
    stats.RecordAccess(past);

    auto age = stats.Age();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(age).count();

    // Age should be approximately 50 seconds
    EXPECT_GE(seconds, 49);
    EXPECT_LE(seconds, 51);
}

TEST(AccessStatsTest, AverageAccessInterval) {
    AccessStats stats;

    Timestamp t = Timestamp::Now();
    stats.RecordAccess(t);

    // Record accesses at regular 5-second intervals
    for (int i = 1; i <= 10; ++i) {
        t = t + std::chrono::seconds(5);
        stats.RecordAccess(t);
    }

    // Average should converge toward 5 seconds
    EXPECT_NEAR(5.0f, stats.avg_access_interval, 1.0f);
}

TEST(AccessStatsTest, SerializeDeserialize) {
    AccessStats original;
    original.access_count = 42;
    original.last_access = Timestamp::Now();
    original.creation_time = Timestamp::Now() - std::chrono::hours(24);
    original.avg_access_interval = 15.5f;

    // Serialize
    std::stringstream ss;
    original.Serialize(ss);

    // Deserialize
    AccessStats deserialized = AccessStats::Deserialize(ss);

    // Verify
    EXPECT_EQ(original.access_count, deserialized.access_count);
    EXPECT_EQ(original.last_access.ToMicros(),
              deserialized.last_access.ToMicros());
    EXPECT_EQ(original.creation_time.ToMicros(),
              deserialized.creation_time.ToMicros());
    EXPECT_FLOAT_EQ(original.avg_access_interval, deserialized.avg_access_interval);
}

// ============================================================================
// AccessTracker Tests
// ============================================================================

TEST(AccessTrackerTest, RecordPatternAccess) {
    AccessTracker tracker;

    PatternID pattern = PatternID::Generate();

    EXPECT_EQ(0u, tracker.GetTrackedPatternCount());
    EXPECT_EQ(nullptr, tracker.GetPatternStats(pattern));

    tracker.RecordPatternAccess(pattern);

    EXPECT_EQ(1u, tracker.GetTrackedPatternCount());

    const AccessStats* stats = tracker.GetPatternStats(pattern);
    ASSERT_NE(nullptr, stats);
    EXPECT_EQ(1u, stats->access_count);
}

TEST(AccessTrackerTest, RecordAssociationAccess) {
    AccessTracker tracker;

    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();

    EXPECT_EQ(0u, tracker.GetTrackedAssociationCount());
    EXPECT_EQ(nullptr, tracker.GetAssociationStats(source, target));

    tracker.RecordAssociationAccess(source, target);

    EXPECT_EQ(1u, tracker.GetTrackedAssociationCount());

    const AccessStats* stats = tracker.GetAssociationStats(source, target);
    ASSERT_NE(nullptr, stats);
    EXPECT_EQ(1u, stats->access_count);
}

TEST(AccessTrackerTest, GetPatternStats) {
    AccessTracker tracker;

    PatternID pattern1 = PatternID::Generate();
    PatternID pattern2 = PatternID::Generate();

    tracker.RecordPatternAccess(pattern1);
    tracker.RecordPatternAccess(pattern1);
    tracker.RecordPatternAccess(pattern2);

    const AccessStats* stats1 = tracker.GetPatternStats(pattern1);
    ASSERT_NE(nullptr, stats1);
    EXPECT_EQ(2u, stats1->access_count);

    const AccessStats* stats2 = tracker.GetPatternStats(pattern2);
    ASSERT_NE(nullptr, stats2);
    EXPECT_EQ(1u, stats2->access_count);

    // Non-existent pattern
    PatternID pattern3 = PatternID::Generate();
    EXPECT_EQ(nullptr, tracker.GetPatternStats(pattern3));
}

TEST(AccessTrackerTest, GetAssociationStats) {
    AccessTracker tracker;

    PatternID source1 = PatternID::Generate();
    PatternID target1 = PatternID::Generate();
    PatternID source2 = PatternID::Generate();
    PatternID target2 = PatternID::Generate();

    tracker.RecordAssociationAccess(source1, target1);
    tracker.RecordAssociationAccess(source1, target1);
    tracker.RecordAssociationAccess(source2, target2);

    const AccessStats* stats1 = tracker.GetAssociationStats(source1, target1);
    ASSERT_NE(nullptr, stats1);
    EXPECT_EQ(2u, stats1->access_count);

    const AccessStats* stats2 = tracker.GetAssociationStats(source2, target2);
    ASSERT_NE(nullptr, stats2);
    EXPECT_EQ(1u, stats2->access_count);

    // Non-existent association
    PatternID source3 = PatternID::Generate();
    PatternID target3 = PatternID::Generate();
    EXPECT_EQ(nullptr, tracker.GetAssociationStats(source3, target3));
}

TEST(AccessTrackerTest, PruneOldStats) {
    AccessTracker tracker;

    Timestamp old_time = Timestamp::Now() - std::chrono::hours(48);
    Timestamp recent_time = Timestamp::Now() - std::chrono::hours(1);

    PatternID old_pattern = PatternID::Generate();
    PatternID recent_pattern = PatternID::Generate();

    tracker.RecordPatternAccess(old_pattern, old_time);
    tracker.RecordPatternAccess(recent_pattern, recent_time);

    PatternID old_source = PatternID::Generate();
    PatternID old_target = PatternID::Generate();
    PatternID recent_source = PatternID::Generate();
    PatternID recent_target = PatternID::Generate();

    tracker.RecordAssociationAccess(old_source, old_target, old_time);
    tracker.RecordAssociationAccess(recent_source, recent_target, recent_time);

    EXPECT_EQ(2u, tracker.GetTrackedPatternCount());
    EXPECT_EQ(2u, tracker.GetTrackedAssociationCount());

    // Prune stats older than 24 hours
    Timestamp cutoff = Timestamp::Now() - std::chrono::hours(24);
    size_t removed = tracker.PruneOldStats(cutoff);

    EXPECT_EQ(2u, removed);  // 1 pattern + 1 association
    EXPECT_EQ(1u, tracker.GetTrackedPatternCount());
    EXPECT_EQ(1u, tracker.GetTrackedAssociationCount());

    // Old stats should be gone
    EXPECT_EQ(nullptr, tracker.GetPatternStats(old_pattern));
    EXPECT_EQ(nullptr, tracker.GetAssociationStats(old_source, old_target));

    // Recent stats should remain
    EXPECT_NE(nullptr, tracker.GetPatternStats(recent_pattern));
    EXPECT_NE(nullptr, tracker.GetAssociationStats(recent_source, recent_target));
}

TEST(AccessTrackerTest, Clear) {
    AccessTracker tracker;

    PatternID pattern = PatternID::Generate();
    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();

    tracker.RecordPatternAccess(pattern);
    tracker.RecordAssociationAccess(source, target);

    EXPECT_EQ(1u, tracker.GetTrackedPatternCount());
    EXPECT_EQ(1u, tracker.GetTrackedAssociationCount());

    tracker.Clear();

    EXPECT_EQ(0u, tracker.GetTrackedPatternCount());
    EXPECT_EQ(0u, tracker.GetTrackedAssociationCount());
    EXPECT_EQ(nullptr, tracker.GetPatternStats(pattern));
    EXPECT_EQ(nullptr, tracker.GetAssociationStats(source, target));
}

TEST(AccessTrackerTest, TrackedCounts) {
    AccessTracker tracker;

    EXPECT_EQ(0u, tracker.GetTrackedPatternCount());
    EXPECT_EQ(0u, tracker.GetTrackedAssociationCount());

    // Add patterns
    for (int i = 0; i < 5; ++i) {
        tracker.RecordPatternAccess(PatternID::Generate());
    }
    EXPECT_EQ(5u, tracker.GetTrackedPatternCount());

    // Add associations
    for (int i = 0; i < 3; ++i) {
        tracker.RecordAssociationAccess(PatternID::Generate(), PatternID::Generate());
    }
    EXPECT_EQ(3u, tracker.GetTrackedAssociationCount());
}

TEST(AccessTrackerTest, ConcurrentPatternAccess) {
    AccessTracker tracker;

    PatternID pattern = PatternID::Generate();

    constexpr int kNumThreads = 10;
    constexpr int kAccessesPerThread = 100;

    std::vector<std::thread> threads;

    // Launch threads that record accesses
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&tracker, pattern]() {
            for (int j = 0; j < kAccessesPerThread; ++j) {
                tracker.RecordPatternAccess(pattern);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify total access count
    const AccessStats* stats = tracker.GetPatternStats(pattern);
    ASSERT_NE(nullptr, stats);
    EXPECT_EQ(kNumThreads * kAccessesPerThread, stats->access_count);
}

TEST(AccessTrackerTest, ConcurrentAssociationAccess) {
    AccessTracker tracker;

    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();

    constexpr int kNumThreads = 10;
    constexpr int kAccessesPerThread = 100;

    std::vector<std::thread> threads;

    // Launch threads that record accesses
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&tracker, source, target]() {
            for (int j = 0; j < kAccessesPerThread; ++j) {
                tracker.RecordAssociationAccess(source, target);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify total access count
    const AccessStats* stats = tracker.GetAssociationStats(source, target);
    ASSERT_NE(nullptr, stats);
    EXPECT_EQ(kNumThreads * kAccessesPerThread, stats->access_count);
}

TEST(AccessTrackerTest, MixedConcurrentOps) {
    AccessTracker tracker;

    std::vector<PatternID> patterns;
    for (int i = 0; i < 5; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    constexpr int kNumThreads = 20;
    std::vector<std::thread> threads;

    // Half the threads record pattern accesses
    for (int i = 0; i < kNumThreads / 2; ++i) {
        threads.emplace_back([&tracker, &patterns]() {
            for (int j = 0; j < 100; ++j) {
                tracker.RecordPatternAccess(patterns[j % patterns.size()]);
            }
        });
    }

    // Other half record association accesses
    for (int i = 0; i < kNumThreads / 2; ++i) {
        threads.emplace_back([&tracker, &patterns]() {
            for (int j = 0; j < 100; ++j) {
                size_t src_idx = j % patterns.size();
                size_t tgt_idx = (j + 1) % patterns.size();
                tracker.RecordAssociationAccess(patterns[src_idx], patterns[tgt_idx]);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify stats were recorded (exact counts depend on modulo distribution)
    EXPECT_GT(tracker.GetTrackedPatternCount(), 0u);
    EXPECT_GT(tracker.GetTrackedAssociationCount(), 0u);

    // Verify at least some patterns have stats
    for (const auto& pattern : patterns) {
        const AccessStats* stats = tracker.GetPatternStats(pattern);
        EXPECT_NE(nullptr, stats);
        EXPECT_GT(stats->access_count, 0u);
    }
}
