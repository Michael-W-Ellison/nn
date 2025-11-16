// File: tests/association/co_occurrence_tracker_test.cpp
#include "association/co_occurrence_tracker.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

namespace dpan {
namespace {

// ============================================================================
// Basic Construction and Recording Tests
// ============================================================================

TEST(CoOccurrenceTrackerTest, DefaultConstruction) {
    CoOccurrenceTracker tracker;
    EXPECT_EQ(0u, tracker.GetActivationCount());
    EXPECT_EQ(0u, tracker.GetUniquePatternCount());
    EXPECT_EQ(0u, tracker.GetTotalWindows());
}

TEST(CoOccurrenceTrackerTest, RecordSingleActivation) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    tracker.RecordActivation(p1);

    EXPECT_EQ(1u, tracker.GetActivationCount());
    EXPECT_EQ(1u, tracker.GetUniquePatternCount());
}

TEST(CoOccurrenceTrackerTest, RecordMultipleActivations) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    tracker.RecordActivation(p1);
    tracker.RecordActivation(p2);
    tracker.RecordActivation(p3);

    EXPECT_EQ(3u, tracker.GetActivationCount());
    EXPECT_EQ(3u, tracker.GetUniquePatternCount());
}

TEST(CoOccurrenceTrackerTest, RecordBatchActivations) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    std::vector<PatternID> patterns = {p1, p2, p3};
    tracker.RecordActivations(patterns);

    EXPECT_EQ(3u, tracker.GetActivationCount());
    EXPECT_EQ(3u, tracker.GetUniquePatternCount());
    EXPECT_EQ(1u, tracker.GetTotalWindows());
}

// ============================================================================
// Co-occurrence Counting Tests
// ============================================================================

TEST(CoOccurrenceTrackerTest, NoCoOccurrence) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    EXPECT_EQ(0u, tracker.GetCoOccurrenceCount(p1, p2));
}

TEST(CoOccurrenceTrackerTest, SimpleCoOccurrence) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Record both in same batch (guaranteed co-occurrence)
    tracker.RecordActivations({p1, p2});

    EXPECT_EQ(1u, tracker.GetCoOccurrenceCount(p1, p2));
    EXPECT_EQ(1u, tracker.GetCoOccurrenceCount(p2, p1));  // Order independent
}

TEST(CoOccurrenceTrackerTest, MultipleCoOccurrences) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Record multiple times
    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p2});

    EXPECT_EQ(3u, tracker.GetCoOccurrenceCount(p1, p2));
}

TEST(CoOccurrenceTrackerTest, CoOccurrenceWithinWindow) {
    CoOccurrenceTracker::Config config;
    config.window_size = std::chrono::milliseconds(100);
    CoOccurrenceTracker tracker(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    auto t0 = Timestamp::Now();

    // Record p1 at t0
    tracker.RecordActivation(p1, t0);

    // Record p2 at t0 + 50ms (within window)
    tracker.RecordActivation(p2, t0 + std::chrono::milliseconds(50));

    // Should co-occur
    EXPECT_GT(tracker.GetCoOccurrenceCount(p1, p2), 0u);
}

TEST(CoOccurrenceTrackerTest, NoCoOccurrenceOutsideWindow) {
    CoOccurrenceTracker::Config config;
    config.window_size = std::chrono::milliseconds(100);
    CoOccurrenceTracker tracker(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    auto t0 = Timestamp::Now();

    // Record p1 at t0
    tracker.RecordActivation(p1, t0);

    // Record p2 at t0 + 200ms (outside window)
    tracker.RecordActivation(p2, t0 + std::chrono::milliseconds(200));

    // Should NOT co-occur
    EXPECT_EQ(0u, tracker.GetCoOccurrenceCount(p1, p2));
}

TEST(CoOccurrenceTrackerTest, ThreeWayCoOccurrence) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // All three patterns in same window
    tracker.RecordActivations({p1, p2, p3});

    // Each pair should have count of 1
    EXPECT_EQ(1u, tracker.GetCoOccurrenceCount(p1, p2));
    EXPECT_EQ(1u, tracker.GetCoOccurrenceCount(p1, p3));
    EXPECT_EQ(1u, tracker.GetCoOccurrenceCount(p2, p3));
    EXPECT_EQ(3u, tracker.GetCoOccurrencePairCount());
}

// ============================================================================
// Probability Tests
// ============================================================================

TEST(CoOccurrenceTrackerTest, ZeroProbability) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    EXPECT_FLOAT_EQ(0.0f, tracker.GetCoOccurrenceProbability(p1, p2));
}

TEST(CoOccurrenceTrackerTest, SimpleProbability) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Co-occur once out of 1 window
    tracker.RecordActivations({p1, p2});

    EXPECT_FLOAT_EQ(1.0f, tracker.GetCoOccurrenceProbability(p1, p2));
}

TEST(CoOccurrenceTrackerTest, PartialProbability) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // p1 and p2 co-occur in 2 out of 4 windows
    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p3});
    tracker.RecordActivations({p2, p3});

    float prob = tracker.GetCoOccurrenceProbability(p1, p2);
    EXPECT_FLOAT_EQ(0.5f, prob);  // 2/4
}

// ============================================================================
// Chi-Squared Significance Tests
// ============================================================================

TEST(CoOccurrenceTrackerTest, SignificanceWithMinCount) {
    CoOccurrenceTracker::Config config;
    config.min_co_occurrences = 3;
    CoOccurrenceTracker tracker(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Only 2 co-occurrences (below minimum)
    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p2});

    EXPECT_FALSE(tracker.IsSignificant(p1, p2));
}

TEST(CoOccurrenceTrackerTest, SignificantCoOccurrence) {
    CoOccurrenceTracker::Config config;
    config.min_co_occurrences = 3;
    CoOccurrenceTracker tracker(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // p1 and p2 co-occur frequently (10 times)
    for (int i = 0; i < 10; ++i) {
        tracker.RecordActivations({p1, p2});
    }

    // Add some noise with p3
    for (int i = 0; i < 5; ++i) {
        tracker.RecordActivations({p3});
    }

    // p1-p2 should be significant
    EXPECT_TRUE(tracker.IsSignificant(p1, p2));

    // p1-p3 should not be significant (never co-occur)
    EXPECT_FALSE(tracker.IsSignificant(p1, p3));
}

TEST(CoOccurrenceTrackerTest, ChiSquaredCalculation) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // Create a scenario with strong association between p1 and p2
    // p1 and p2 co-occur frequently (15 times)
    for (int i = 0; i < 15; ++i) {
        tracker.RecordActivations({p1, p2});
    }

    // p1 occurs alone sometimes (3 times)
    for (int i = 0; i < 3; ++i) {
        tracker.RecordActivations({p1});
    }

    // p2 occurs alone sometimes (2 times)
    for (int i = 0; i < 2; ++i) {
        tracker.RecordActivations({p2});
    }

    // Windows with neither p1 nor p2 (just p3)
    for (int i = 0; i < 5; ++i) {
        tracker.RecordActivations({p3});
    }

    float chi_squared = tracker.GetChiSquared(p1, p2);

    // Chi-squared should be large for strong association
    // With a=15, b=3, c=2, d=5, the chi-squared should be significant
    EXPECT_GT(chi_squared, 3.841f);  // Threshold for p < 0.05, df=1
}

// ============================================================================
// Query Methods Tests
// ============================================================================

TEST(CoOccurrenceTrackerTest, GetCoOccurringPatternsEmpty) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();

    auto co_occurring = tracker.GetCoOccurringPatterns(p1);
    EXPECT_TRUE(co_occurring.empty());
}

TEST(CoOccurrenceTrackerTest, GetCoOccurringPatterns) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    // p1 co-occurs with p2 (3 times), p3 (2 times), p4 (1 time)
    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p3});
    tracker.RecordActivations({p1, p3});
    tracker.RecordActivations({p1, p4});

    auto co_occurring = tracker.GetCoOccurringPatterns(p1);

    EXPECT_EQ(3u, co_occurring.size());

    // Should be sorted by count (descending)
    EXPECT_EQ(p2, co_occurring[0].first);
    EXPECT_EQ(3u, co_occurring[0].second);

    EXPECT_EQ(p3, co_occurring[1].first);
    EXPECT_EQ(2u, co_occurring[1].second);

    EXPECT_EQ(p4, co_occurring[2].first);
    EXPECT_EQ(1u, co_occurring[2].second);
}

TEST(CoOccurrenceTrackerTest, GetCoOccurringPatternsWithMinCount) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p3});

    // Only get patterns with count >= 2
    auto co_occurring = tracker.GetCoOccurringPatterns(p1, 2);

    EXPECT_EQ(1u, co_occurring.size());
    EXPECT_EQ(p2, co_occurring[0].first);
}

// ============================================================================
// Maintenance Tests
// ============================================================================

TEST(CoOccurrenceTrackerTest, PruneOldActivations) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    auto t0 = Timestamp::Now();

    // Record at different times
    tracker.RecordActivation(p1, t0);
    tracker.RecordActivation(p2, t0 + std::chrono::seconds(5));

    EXPECT_EQ(2u, tracker.GetActivationCount());

    // Prune activations before t0 + 3 seconds (should remove p1)
    tracker.PruneOldActivations(t0 + std::chrono::seconds(3));

    EXPECT_EQ(1u, tracker.GetActivationCount());
}

TEST(CoOccurrenceTrackerTest, Clear) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p1, p2});

    EXPECT_GT(tracker.GetActivationCount(), 0u);
    EXPECT_GT(tracker.GetCoOccurrenceCount(p1, p2), 0u);

    tracker.Clear();

    EXPECT_EQ(0u, tracker.GetActivationCount());
    EXPECT_EQ(0u, tracker.GetUniquePatternCount());
    EXPECT_EQ(0u, tracker.GetTotalWindows());
    EXPECT_EQ(0u, tracker.GetCoOccurrenceCount(p1, p2));
}

// ============================================================================
// Edge Cases and Stress Tests
// ============================================================================

TEST(CoOccurrenceTrackerTest, DuplicatePatternsInSameWindow) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Record p1 twice and p2 once in same window
    tracker.RecordActivations({p1, p1, p2});

    // Should count as single co-occurrence (unique patterns)
    EXPECT_EQ(1u, tracker.GetCoOccurrenceCount(p1, p2));
}

TEST(CoOccurrenceTrackerTest, SelfCoOccurrence) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();

    tracker.RecordActivations({p1, p1});

    // Pattern does not co-occur with itself
    EXPECT_EQ(0u, tracker.GetCoOccurrenceCount(p1, p1));
}

TEST(CoOccurrenceTrackerTest, ManyPatterns) {
    CoOccurrenceTracker tracker;

    // Create 100 patterns
    std::vector<PatternID> patterns;
    for (int i = 0; i < 100; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    // Record all patterns in same window
    tracker.RecordActivations(patterns);

    // Should create C(100, 2) = 4950 pairs
    EXPECT_EQ(4950u, tracker.GetCoOccurrencePairCount());
}

TEST(CoOccurrenceTrackerTest, LongRunningTracker) {
    CoOccurrenceTracker::Config config;
    config.window_size = std::chrono::milliseconds(10);
    CoOccurrenceTracker tracker(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    auto t0 = Timestamp::Now();

    // Record many activations over time
    for (int i = 0; i < 100; ++i) {
        auto t = t0 + std::chrono::milliseconds(i * 5);
        tracker.RecordActivation(p1, t);

        if (i % 2 == 0) {
            tracker.RecordActivation(p2, t);
        }
    }

    // Verify tracking works
    EXPECT_GT(tracker.GetCoOccurrenceCount(p1, p2), 0u);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(CoOccurrenceTrackerTest, Statistics) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    tracker.RecordActivations({p1, p2});
    tracker.RecordActivations({p2, p3});
    tracker.RecordActivations({p1, p3});

    EXPECT_EQ(6u, tracker.GetActivationCount());
    EXPECT_EQ(3u, tracker.GetUniquePatternCount());
    EXPECT_EQ(3u, tracker.GetTotalWindows());
    EXPECT_EQ(3u, tracker.GetCoOccurrencePairCount());
}

} // namespace
} // namespace dpan
