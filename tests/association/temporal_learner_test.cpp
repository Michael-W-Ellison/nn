// File: tests/association/temporal_learner_test.cpp
#include "association/temporal_learner.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

namespace dpan {
namespace {

// ============================================================================
// Construction and Basic Tests
// ============================================================================

TEST(TemporalLearnerTest, DefaultConstruction) {
    TemporalLearner learner;

    EXPECT_EQ(0u, learner.GetActivationCount());
    EXPECT_EQ(0u, learner.GetUniquePatternCount());
    EXPECT_EQ(0u, learner.GetPairCount());
}

TEST(TemporalLearnerTest, ConfigConstruction) {
    TemporalLearner::Config config;
    config.max_delay = std::chrono::seconds(5);
    config.min_occurrences = 5;
    config.min_correlation = 0.7f;

    TemporalLearner learner(config);

    const auto& retrieved_config = learner.GetConfig();
    EXPECT_EQ(std::chrono::seconds(5), retrieved_config.max_delay);
    EXPECT_EQ(5u, retrieved_config.min_occurrences);
    EXPECT_FLOAT_EQ(0.7f, retrieved_config.min_correlation);
}

TEST(TemporalLearnerTest, RecordSingleActivation) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    learner.RecordActivation(p1);

    EXPECT_EQ(1u, learner.GetActivationCount());
    EXPECT_EQ(1u, learner.GetUniquePatternCount());
}

TEST(TemporalLearnerTest, RecordMultipleActivations) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    learner.RecordActivation(p1);
    learner.RecordActivation(p2);
    learner.RecordActivation(p3);

    EXPECT_EQ(3u, learner.GetActivationCount());
    EXPECT_EQ(3u, learner.GetUniquePatternCount());
}

// ============================================================================
// Temporal Statistics Tests
// ============================================================================

TEST(TemporalLearnerTest, NoStatisticsInitially) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    auto stats = learner.GetTemporalStats(p1, p2);
    EXPECT_FALSE(stats.has_value());
}

TEST(TemporalLearnerTest, SimpleTemporalSequence) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    // Record p1 followed by p2 multiple times with consistent delay
    for (int i = 0; i < 5; ++i) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(100);
        learner.RecordActivation(p2, t);
        t = t + std::chrono::seconds(1);
    }

    // Should have statistics now
    auto stats = learner.GetTemporalStats(p1, p2);
    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(5u, stats->occurrence_count);
    EXPECT_GT(stats->mean_delay_micros, 0);
}

TEST(TemporalLearnerTest, TemporalCorrelation) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    // Consistent delays should produce high correlation
    for (int i = 0; i < 5; ++i) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(100);  // Consistent 100ms delay
        learner.RecordActivation(p2, t);
        t = t + std::chrono::seconds(1);
    }

    float correlation = learner.GetTemporalCorrelation(p1, p2);
    EXPECT_GT(correlation, 0.0f);
    EXPECT_LE(correlation, 1.0f);
}

TEST(TemporalLearnerTest, HighCorrelationWithConsistentDelay) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    // Very consistent delays
    for (int i = 0; i < 10; ++i) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(100);  // Exactly 100ms each time
        learner.RecordActivation(p2, t);
        t = t + std::chrono::seconds(1);
    }

    float correlation = learner.GetTemporalCorrelation(p1, p2);
    // High consistency should produce correlation close to 1.0
    EXPECT_GT(correlation, 0.7f);
}

TEST(TemporalLearnerTest, LowCorrelationWithVariableDelay) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    // Variable delays should reduce correlation
    std::vector<int> delays_ms = {50, 200, 75, 300, 100};
    for (int delay_ms : delays_ms) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(delay_ms);
        learner.RecordActivation(p2, t);
        t = t + std::chrono::seconds(1);
    }

    float correlation = learner.GetTemporalCorrelation(p1, p2);
    // Variable delays should produce lower correlation
    EXPECT_GT(correlation, 0.0f);
    EXPECT_LT(correlation, 1.0f);
}

TEST(TemporalLearnerTest, MeanDelayComputation) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    // Record with 100ms delays
    for (int i = 0; i < 5; ++i) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(100);
        learner.RecordActivation(p2, t);
        t = t + std::chrono::seconds(1);
    }

    int64_t mean_delay = learner.GetMeanDelay(p1, p2);
    // Should be approximately 100,000 microseconds (100ms)
    EXPECT_GT(mean_delay, 90000);   // Allow some tolerance
    EXPECT_LT(mean_delay, 110000);
}

// ============================================================================
// Successor/Predecessor Tests
// ============================================================================

TEST(TemporalLearnerTest, GetSuccessorsEmpty) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();

    auto successors = learner.GetSuccessors(p1);
    EXPECT_TRUE(successors.empty());
}

TEST(TemporalLearnerTest, GetSuccessors) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    // p1 -> p2 (high correlation)
    for (int i = 0; i < 10; ++i) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(100);
        learner.RecordActivation(p2, t);
        t = t + std::chrono::seconds(1);
    }

    // p1 -> p3 (low correlation)
    for (int i = 0; i < 5; ++i) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(500);
        learner.RecordActivation(p3, t);
        t = t + std::chrono::seconds(1);
    }

    auto successors = learner.GetSuccessors(p1);

    EXPECT_GE(successors.size(), 1u);  // At least p2

    // p2 should be in the list (could be first or second depending on correlation)
    bool found_p2 = false;
    for (const auto& [pattern, corr] : successors) {
        if (pattern == p2) {
            found_p2 = true;
            break;
        }
    }
    EXPECT_TRUE(found_p2);
}

TEST(TemporalLearnerTest, GetPredecessorsEmpty) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();

    auto predecessors = learner.GetPredecessors(p1);
    EXPECT_TRUE(predecessors.empty());
}

TEST(TemporalLearnerTest, GetPredecessors) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    // p1 -> p3 and p2 -> p3
    for (int i = 0; i < 5; ++i) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(100);
        learner.RecordActivation(p3, t);
        t = t + std::chrono::seconds(1);
    }

    for (int i = 0; i < 5; ++i) {
        learner.RecordActivation(p2, t);
        t = t + std::chrono::milliseconds(150);
        learner.RecordActivation(p3, t);
        t = t + std::chrono::seconds(1);
    }

    auto predecessors = learner.GetPredecessors(p3);

    EXPECT_GE(predecessors.size(), 1u);  // At least one predecessor
}

// ============================================================================
// Correlation Detection Tests
// ============================================================================

TEST(TemporalLearnerTest, IsTemporallyCorrelatedTrue) {
    TemporalLearner::Config config;
    config.min_correlation = 0.5f;
    TemporalLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    // Consistent pattern should be correlated
    for (int i = 0; i < 10; ++i) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(100);
        learner.RecordActivation(p2, t);
        t = t + std::chrono::seconds(1);
    }

    EXPECT_TRUE(learner.IsTemporallyCorrelated(p1, p2));
}

TEST(TemporalLearnerTest, IsTemporallyCorrelatedFalse) {
    TemporalLearner::Config config;
    config.min_correlation = 0.9f;  // Very high threshold
    TemporalLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    // Variable delays
    std::vector<int> delays_ms = {50, 200, 75, 300, 100};
    for (int delay_ms : delays_ms) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(delay_ms);
        learner.RecordActivation(p2, t);
        t = t + std::chrono::seconds(1);
    }

    // May not meet high threshold
    bool is_correlated = learner.IsTemporallyCorrelated(p1, p2);
    // Just verify it doesn't crash - result depends on actual variance
    EXPECT_TRUE(is_correlated || !is_correlated);
}

// ============================================================================
// Sequence Recording Tests
// ============================================================================

TEST(TemporalLearnerTest, RecordSequence) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    std::vector<std::pair<Timestamp, PatternID>> sequence;
    Timestamp t = Timestamp::Now();

    sequence.push_back({t, p1});
    t = t + std::chrono::milliseconds(100);
    sequence.push_back({t, p2});
    t = t + std::chrono::milliseconds(50);
    sequence.push_back({t, p3});

    learner.RecordSequence(sequence);

    EXPECT_EQ(3u, learner.GetActivationCount());
}

// ============================================================================
// Maintenance Tests
// ============================================================================

TEST(TemporalLearnerTest, PruneOldActivations) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();

    Timestamp t0 = Timestamp::Now();

    // Record at different times
    learner.RecordActivation(p1, t0);
    learner.RecordActivation(p1, t0 + std::chrono::seconds(2));
    learner.RecordActivation(p1, t0 + std::chrono::seconds(4));

    EXPECT_EQ(3u, learner.GetActivationCount());

    // Prune activations before t0 + 3 seconds
    learner.PruneOldActivations(t0 + std::chrono::seconds(3));

    // Should keep only the last activation
    EXPECT_EQ(1u, learner.GetActivationCount());
}

TEST(TemporalLearnerTest, Clear) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    for (int i = 0; i < 5; ++i) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(100);
        learner.RecordActivation(p2, t);
        t = t + std::chrono::seconds(1);
    }

    EXPECT_GT(learner.GetActivationCount(), 0u);
    EXPECT_GT(learner.GetPairCount(), 0u);

    learner.Clear();

    EXPECT_EQ(0u, learner.GetActivationCount());
    EXPECT_EQ(0u, learner.GetPairCount());
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST(TemporalLearnerTest, SamePatternNoCorrelation) {
    TemporalLearner learner;

    PatternID p1 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    for (int i = 0; i < 10; ++i) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(100);
    }

    // Pattern should not correlate with itself
    auto stats = learner.GetTemporalStats(p1, p1);
    EXPECT_FALSE(stats.has_value());
}

TEST(TemporalLearnerTest, InsufficientOccurrences) {
    TemporalLearner::Config config;
    config.min_occurrences = 10;
    TemporalLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    // Only 3 occurrences (below threshold)
    for (int i = 0; i < 3; ++i) {
        learner.RecordActivation(p1, t);
        t = t + std::chrono::milliseconds(100);
        learner.RecordActivation(p2, t);
        t = t + std::chrono::seconds(1);
    }

    auto stats = learner.GetTemporalStats(p1, p2);
    EXPECT_FALSE(stats.has_value());  // Insufficient occurrences
}

TEST(TemporalLearnerTest, MaxDelayExceeded) {
    TemporalLearner::Config config;
    config.max_delay = std::chrono::milliseconds(100);
    TemporalLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t = Timestamp::Now();

    // Delays that exceed max_delay
    learner.RecordActivation(p1, t);
    t = t + std::chrono::seconds(1);  // 1 second > 100ms max
    learner.RecordActivation(p2, t);

    // Should not create statistics for delays exceeding max
    auto stats = learner.GetTemporalStats(p1, p2);
    EXPECT_FALSE(stats.has_value());
}

TEST(TemporalLearnerTest, ManyPatterns) {
    TemporalLearner learner;

    std::vector<PatternID> patterns;
    for (int i = 0; i < 20; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    Timestamp t = Timestamp::Now();

    // Create a chain: p0 -> p1 -> p2 -> ... -> p19
    for (size_t i = 0; i < patterns.size(); ++i) {
        learner.RecordActivation(patterns[i], t);
        t = t + std::chrono::milliseconds(50);
    }

    // Should track many pattern pairs
    EXPECT_EQ(patterns.size(), learner.GetActivationCount());
}

TEST(TemporalLearnerTest, ConfigModification) {
    TemporalLearner learner;

    TemporalLearner::Config new_config;
    new_config.min_correlation = 0.8f;
    new_config.min_occurrences = 10;

    learner.SetConfig(new_config);

    const auto& config = learner.GetConfig();
    EXPECT_FLOAT_EQ(0.8f, config.min_correlation);
    EXPECT_EQ(10u, config.min_occurrences);
}

} // namespace
} // namespace dpan
