// File: tests/association/spatial_learner_test.cpp
#include "association/spatial_learner.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

using namespace dpan;

// ============================================================================
// Construction Tests
// ============================================================================

TEST(SpatialLearnerTest, DefaultConstruction) {
    SpatialLearner learner;

    EXPECT_EQ(0u, learner.GetPatternCount());
    EXPECT_EQ(0u, learner.GetTotalObservations());

    const auto& config = learner.GetConfig();
    EXPECT_FLOAT_EQ(0.7f, config.min_similarity_threshold);
    EXPECT_EQ(3u, config.min_observations);
    EXPECT_EQ(1000u, config.max_history);
    EXPECT_FLOAT_EQ(0.1f, config.learning_rate);
}

TEST(SpatialLearnerTest, ConfigConstruction) {
    SpatialLearner::Config config;
    config.min_similarity_threshold = 0.8f;
    config.min_observations = 5;
    config.max_history = 500;
    config.learning_rate = 0.2f;

    SpatialLearner learner(config);

    const auto& retrieved_config = learner.GetConfig();
    EXPECT_FLOAT_EQ(0.8f, retrieved_config.min_similarity_threshold);
    EXPECT_EQ(5u, retrieved_config.min_observations);
    EXPECT_EQ(500u, retrieved_config.max_history);
    EXPECT_FLOAT_EQ(0.2f, retrieved_config.learning_rate);
}

// ============================================================================
// Recording Tests
// ============================================================================

TEST(SpatialLearnerTest, RecordSingleContext) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    ContextVector ctx;
    ctx.Set("x", 1.0f);
    ctx.Set("y", 2.0f);

    learner.RecordSpatialContext(p1, ctx);

    EXPECT_EQ(1u, learner.GetPatternCount());
    EXPECT_EQ(1u, learner.GetObservationCount(p1));
}

TEST(SpatialLearnerTest, RecordMultipleContexts) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    for (int i = 0; i < 5; ++i) {
        ContextVector ctx;
        ctx.Set("x", static_cast<float>(i));
        learner.RecordSpatialContext(p1, ctx);
    }

    EXPECT_EQ(1u, learner.GetPatternCount());
    EXPECT_EQ(5u, learner.GetObservationCount(p1));
    EXPECT_EQ(5u, learner.GetTotalObservations());
}

TEST(SpatialLearnerTest, RecordWithCoOccurringPatterns) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    ContextVector ctx;
    ctx.Set("location", 1.0f);

    std::vector<PatternID> co_occurring = {p2, p3};
    learner.RecordSpatialContext(p1, ctx, co_occurring);

    auto history = learner.GetContextHistory(p1);
    ASSERT_EQ(1u, history.size());
    EXPECT_EQ(2u, history[0].co_occurring_patterns.size());
    EXPECT_EQ(p2, history[0].co_occurring_patterns[0]);
    EXPECT_EQ(p3, history[0].co_occurring_patterns[1]);
}

// ============================================================================
// Average Context Tests
// ============================================================================

TEST(SpatialLearnerTest, GetAverageContextNoObservations) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    ContextVector avg = learner.GetAverageContext(p1);
    EXPECT_TRUE(avg.IsEmpty());
}

TEST(SpatialLearnerTest, GetAverageContextSingleObservation) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    ContextVector ctx;
    ctx.Set("x", 10.0f);
    ctx.Set("y", 20.0f);

    learner.RecordSpatialContext(p1, ctx);

    ContextVector avg = learner.GetAverageContext(p1);
    EXPECT_FLOAT_EQ(10.0f, avg.Get("x"));
    EXPECT_FLOAT_EQ(20.0f, avg.Get("y"));
}

TEST(SpatialLearnerTest, GetAverageContextMultipleObservations) {
    SpatialLearner::Config config;
    config.learning_rate = 0.5f;  // 50% learning rate for easier math
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();

    // First observation: x=10
    ContextVector ctx1;
    ctx1.Set("x", 10.0f);
    learner.RecordSpatialContext(p1, ctx1);

    // Average should be 10
    ContextVector avg1 = learner.GetAverageContext(p1);
    EXPECT_FLOAT_EQ(10.0f, avg1.Get("x"));

    // Second observation: x=20
    // Expected: 10 + 0.5 * (20 - 10) = 15
    ContextVector ctx2;
    ctx2.Set("x", 20.0f);
    learner.RecordSpatialContext(p1, ctx2);

    ContextVector avg2 = learner.GetAverageContext(p1);
    EXPECT_FLOAT_EQ(15.0f, avg2.Get("x"));
}

// ============================================================================
// Spatial Similarity Tests
// ============================================================================

TEST(SpatialLearnerTest, SpatialSimilarityNoData) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    float similarity = learner.GetSpatialSimilarity(p1, p2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(SpatialLearnerTest, SpatialSimilarityInsufficientObservations) {
    SpatialLearner::Config config;
    config.min_observations = 3;
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    ContextVector ctx;
    ctx.Set("x", 1.0f);

    // Only 2 observations (less than min_observations)
    learner.RecordSpatialContext(p1, ctx);
    learner.RecordSpatialContext(p1, ctx);
    learner.RecordSpatialContext(p2, ctx);
    learner.RecordSpatialContext(p2, ctx);

    float similarity = learner.GetSpatialSimilarity(p1, p2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(SpatialLearnerTest, SpatialSimilarityIdenticalContexts) {
    SpatialLearner::Config config;
    config.min_observations = 3;
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    ContextVector ctx;
    ctx.Set("x", 1.0f);
    ctx.Set("y", 1.0f);

    // Record identical contexts for both patterns
    for (int i = 0; i < 3; ++i) {
        learner.RecordSpatialContext(p1, ctx);
        learner.RecordSpatialContext(p2, ctx);
    }

    float similarity = learner.GetSpatialSimilarity(p1, p2);
    EXPECT_NEAR(1.0f, similarity, 0.01f);  // Should be very close to 1.0
}

TEST(SpatialLearnerTest, SpatialSimilarityDifferentContexts) {
    SpatialLearner::Config config;
    config.min_observations = 3;
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Pattern 1: appears at x=1, y=0
    ContextVector ctx1;
    ctx1.Set("x", 1.0f);
    ctx1.Set("y", 0.0f);

    // Pattern 2: appears at x=1, y=1
    ContextVector ctx2;
    ctx2.Set("x", 1.0f);
    ctx2.Set("y", 1.0f);

    for (int i = 0; i < 3; ++i) {
        learner.RecordSpatialContext(p1, ctx1);
        learner.RecordSpatialContext(p2, ctx2);
    }

    float similarity = learner.GetSpatialSimilarity(p1, p2);
    EXPECT_GT(similarity, 0.0f);
    EXPECT_LT(similarity, 1.0f);
}

TEST(SpatialLearnerTest, SpatialSimilarityOrthogonalContexts) {
    SpatialLearner::Config config;
    config.min_observations = 3;
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Pattern 1: only x dimension
    ContextVector ctx1;
    ctx1.Set("x", 1.0f);
    ctx1.Set("y", 0.0f);

    // Pattern 2: only y dimension
    ContextVector ctx2;
    ctx2.Set("x", 0.0f);
    ctx2.Set("y", 1.0f);

    for (int i = 0; i < 3; ++i) {
        learner.RecordSpatialContext(p1, ctx1);
        learner.RecordSpatialContext(p2, ctx2);
    }

    float similarity = learner.GetSpatialSimilarity(p1, p2);
    EXPECT_NEAR(0.0f, similarity, 0.1f);  // Should be close to 0 (orthogonal)
}

// ============================================================================
// Spatial Relation Tests
// ============================================================================

TEST(SpatialLearnerTest, AreSpatiallyRelatedInsufficientData) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    EXPECT_FALSE(learner.AreSpatiallyRelated(p1, p2));
}

TEST(SpatialLearnerTest, AreSpatiallyRelatedSimilarContexts) {
    SpatialLearner::Config config;
    config.min_observations = 3;
    config.min_similarity_threshold = 0.7f;
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    ContextVector ctx;
    ctx.Set("location", 1.0f);
    ctx.Set("temperature", 25.0f);

    // Record identical contexts
    for (int i = 0; i < 3; ++i) {
        learner.RecordSpatialContext(p1, ctx);
        learner.RecordSpatialContext(p2, ctx);
    }

    EXPECT_TRUE(learner.AreSpatiallyRelated(p1, p2));
}

TEST(SpatialLearnerTest, AreSpatiallyRelatedDifferentContexts) {
    SpatialLearner::Config config;
    config.min_observations = 3;
    config.min_similarity_threshold = 0.7f;
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    ContextVector ctx1;
    ctx1.Set("location", 0.0f);

    ContextVector ctx2;
    ctx2.Set("location", 1.0f);

    for (int i = 0; i < 3; ++i) {
        learner.RecordSpatialContext(p1, ctx1);
        learner.RecordSpatialContext(p2, ctx2);
    }

    EXPECT_FALSE(learner.AreSpatiallyRelated(p1, p2));
}

TEST(SpatialLearnerTest, AreSpatiallyRelatedCustomThreshold) {
    SpatialLearner::Config config;
    config.min_observations = 3;
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // p1 context: mostly x dimension
    ContextVector ctx1;
    ctx1.Set("x", 1.0f);
    ctx1.Set("y", 0.1f);

    // p2 context: more balanced, but still similar to p1
    ContextVector ctx2;
    ctx2.Set("x", 0.9f);
    ctx2.Set("y", 0.5f);

    for (int i = 0; i < 3; ++i) {
        learner.RecordSpatialContext(p1, ctx1);
        learner.RecordSpatialContext(p2, ctx2);
    }

    // With very high threshold (0.99), they should not be related
    EXPECT_FALSE(learner.AreSpatiallyRelated(p1, p2, 0.99f));

    // With lower threshold, they should be related
    EXPECT_TRUE(learner.AreSpatiallyRelated(p1, p2, 0.5f));
}

// ============================================================================
// Spatially Similar Queries
// ============================================================================

TEST(SpatialLearnerTest, GetSpatiallySimilarNoData) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    auto similar = learner.GetSpatiallySimilar(p1);
    EXPECT_TRUE(similar.empty());
}

TEST(SpatialLearnerTest, GetSpatiallySimilarMultiplePatterns) {
    SpatialLearner::Config config;
    config.min_observations = 3;
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // p1 and p2 have similar contexts
    ContextVector ctx_similar;
    ctx_similar.Set("zone", 1.0f);

    // p3 has different context
    ContextVector ctx_different;
    ctx_different.Set("zone", 10.0f);

    for (int i = 0; i < 3; ++i) {
        learner.RecordSpatialContext(p1, ctx_similar);
        learner.RecordSpatialContext(p2, ctx_similar);
        learner.RecordSpatialContext(p3, ctx_different);
    }

    auto similar = learner.GetSpatiallySimilar(p1, 0.7f);

    // Should find p2 but not p3
    EXPECT_GE(similar.size(), 1u);

    bool found_p2 = false;
    for (const auto& [pattern, sim] : similar) {
        if (pattern == p2) {
            found_p2 = true;
            EXPECT_GE(sim, 0.7f);
        }
        EXPECT_NE(pattern, p1);  // Should not include self
    }
    EXPECT_TRUE(found_p2);
}

TEST(SpatialLearnerTest, GetSpatiallySimilarSortedByScore) {
    SpatialLearner::Config config;
    config.min_observations = 3;
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    ContextVector ctx1;
    ctx1.Set("x", 1.0f);
    ctx1.Set("y", 1.0f);

    // p2: very similar to p1
    ContextVector ctx2;
    ctx2.Set("x", 1.0f);
    ctx2.Set("y", 1.0f);

    // p3: less similar to p1
    ContextVector ctx3;
    ctx3.Set("x", 0.5f);
    ctx3.Set("y", 0.5f);

    for (int i = 0; i < 3; ++i) {
        learner.RecordSpatialContext(p1, ctx1);
        learner.RecordSpatialContext(p2, ctx2);
        learner.RecordSpatialContext(p3, ctx3);
    }

    auto similar = learner.GetSpatiallySimilar(p1);

    ASSERT_GE(similar.size(), 2u);

    // Should be sorted by similarity (descending)
    for (size_t i = 1; i < similar.size(); ++i) {
        EXPECT_GE(similar[i-1].second, similar[i].second);
    }
}

// ============================================================================
// Context History Tests
// ============================================================================

TEST(SpatialLearnerTest, GetContextHistoryEmpty) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    auto history = learner.GetContextHistory(p1);
    EXPECT_TRUE(history.empty());
}

TEST(SpatialLearnerTest, GetContextHistoryMultipleObservations) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    for (int i = 0; i < 5; ++i) {
        ContextVector ctx;
        ctx.Set("iteration", static_cast<float>(i));
        learner.RecordSpatialContext(p1, ctx);
    }

    auto history = learner.GetContextHistory(p1);
    ASSERT_EQ(5u, history.size());

    // Check order (should be in insertion order)
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(static_cast<float>(i), history[i].context.Get("iteration"));
    }
}

// ============================================================================
// Maintenance Tests
// ============================================================================

TEST(SpatialLearnerTest, PruneHistory) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    // Record 10 observations
    for (int i = 0; i < 10; ++i) {
        ContextVector ctx;
        ctx.Set("x", static_cast<float>(i));
        learner.RecordSpatialContext(p1, ctx);
    }

    EXPECT_EQ(10u, learner.GetObservationCount(p1));

    // Prune to keep only 5
    learner.PruneHistory(p1, 5);

    auto history = learner.GetContextHistory(p1);
    ASSERT_EQ(5u, history.size());

    // Should keep the most recent 5 (indices 5-9)
    EXPECT_FLOAT_EQ(5.0f, history[0].context.Get("x"));
    EXPECT_FLOAT_EQ(9.0f, history[4].context.Get("x"));
}

TEST(SpatialLearnerTest, MaxHistoryAutomatic) {
    SpatialLearner::Config config;
    config.max_history = 5;
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();

    // Record 10 observations (exceeds max_history)
    for (int i = 0; i < 10; ++i) {
        ContextVector ctx;
        ctx.Set("x", static_cast<float>(i));
        learner.RecordSpatialContext(p1, ctx);
    }

    auto history = learner.GetContextHistory(p1);
    EXPECT_EQ(5u, history.size());

    // Should have the most recent 5
    EXPECT_FLOAT_EQ(5.0f, history[0].context.Get("x"));
    EXPECT_FLOAT_EQ(9.0f, history[4].context.Get("x"));
}

TEST(SpatialLearnerTest, Clear) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    ContextVector ctx;
    ctx.Set("x", 1.0f);

    learner.RecordSpatialContext(p1, ctx);
    learner.RecordSpatialContext(p2, ctx);

    EXPECT_EQ(2u, learner.GetPatternCount());

    learner.Clear();

    EXPECT_EQ(0u, learner.GetPatternCount());
    EXPECT_EQ(0u, learner.GetTotalObservations());
}

TEST(SpatialLearnerTest, ClearPattern) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    ContextVector ctx;
    ctx.Set("x", 1.0f);

    learner.RecordSpatialContext(p1, ctx);
    learner.RecordSpatialContext(p2, ctx);

    EXPECT_EQ(2u, learner.GetPatternCount());

    learner.ClearPattern(p1);

    EXPECT_EQ(1u, learner.GetPatternCount());
    EXPECT_EQ(0u, learner.GetObservationCount(p1));
    EXPECT_EQ(1u, learner.GetObservationCount(p2));
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(SpatialLearnerTest, GetSpatialStatsNoData) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    auto stats = learner.GetSpatialStats(p1);
    EXPECT_FALSE(stats.has_value());
}

TEST(SpatialLearnerTest, GetSpatialStatsWithData) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    ContextVector ctx;
    ctx.Set("x", 10.0f);

    for (int i = 0; i < 5; ++i) {
        learner.RecordSpatialContext(p1, ctx);
    }

    auto stats = learner.GetSpatialStats(p1);
    ASSERT_TRUE(stats.has_value());
    EXPECT_EQ(5u, stats->observation_count);
    EXPECT_FLOAT_EQ(10.0f, stats->average_context.Get("x"));
}

TEST(SpatialLearnerTest, GetTotalObservationsMultiplePatterns) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    ContextVector ctx;
    ctx.Set("x", 1.0f);

    learner.RecordSpatialContext(p1, ctx);
    learner.RecordSpatialContext(p1, ctx);
    learner.RecordSpatialContext(p2, ctx);
    learner.RecordSpatialContext(p2, ctx);
    learner.RecordSpatialContext(p2, ctx);

    EXPECT_EQ(5u, learner.GetTotalObservations());
    EXPECT_EQ(2u, learner.GetObservationCount(p1));
    EXPECT_EQ(3u, learner.GetObservationCount(p2));
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(SpatialLearnerTest, EmptyContext) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    ContextVector empty_ctx;  // No dimensions set

    learner.RecordSpatialContext(p1, empty_ctx);

    EXPECT_EQ(1u, learner.GetObservationCount(p1));

    ContextVector avg = learner.GetAverageContext(p1);
    EXPECT_TRUE(avg.IsEmpty());
}

TEST(SpatialLearnerTest, SamePatternRecordedMultipleTimes) {
    SpatialLearner learner;
    PatternID p1 = PatternID::Generate();

    for (int i = 0; i < 100; ++i) {
        ContextVector ctx;
        ctx.Set("value", static_cast<float>(i));
        learner.RecordSpatialContext(p1, ctx);
    }

    EXPECT_EQ(100u, learner.GetObservationCount(p1));
    EXPECT_EQ(1u, learner.GetPatternCount());
}

TEST(SpatialLearnerTest, ContextDimensionDecay) {
    SpatialLearner::Config config;
    config.learning_rate = 0.5f;
    SpatialLearner learner(config);

    PatternID p1 = PatternID::Generate();

    // First observation has dimension "temp"
    ContextVector ctx1;
    ctx1.Set("temp", 10.0f);
    learner.RecordSpatialContext(p1, ctx1);

    ContextVector avg1 = learner.GetAverageContext(p1);
    EXPECT_FLOAT_EQ(10.0f, avg1.Get("temp"));

    // Second observation doesn't have "temp"
    // The "temp" dimension should decay
    ContextVector ctx2;
    ctx2.Set("humidity", 50.0f);
    learner.RecordSpatialContext(p1, ctx2);

    ContextVector avg2 = learner.GetAverageContext(p1);
    // temp should have decayed: 10 * (1 - 0.5) = 5
    EXPECT_FLOAT_EQ(5.0f, avg2.Get("temp"));
    // humidity should be added: 0 + 0.5 * (50 - 0) = 25
    EXPECT_FLOAT_EQ(25.0f, avg2.Get("humidity"));
}
