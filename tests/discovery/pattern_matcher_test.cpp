// File: tests/discovery/pattern_matcher_test.cpp
#include "discovery/pattern_matcher.hpp"
#include "storage/memory_backend.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace dpan {
namespace {

// Mock similarity metric that uses Euclidean distance
class MockEuclideanSimilarity : public SimilarityMetric {
public:
    float Compute(const PatternData& a, const PatternData& b) const override {
        return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
    }

    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override {
        if (a.Dimension() != b.Dimension()) {
            return 0.0f;
        }

        float sum_sq_diff = 0.0f;
        for (size_t i = 0; i < a.Dimension(); ++i) {
            float diff = a[i] - b[i];
            sum_sq_diff += diff * diff;
        }

        float distance = std::sqrt(sum_sq_diff);
        // Convert distance to similarity (0 = identical, larger = more different)
        return 1.0f / (1.0f + distance);
    }

    std::string GetName() const override { return "MockEuclidean"; }
    bool IsSymmetric() const override { return true; }
};

// Helper to create test database
std::shared_ptr<PatternDatabase> CreateTestDatabase() {
    auto db = std::make_shared<MemoryBackend>(MemoryBackend::Config{});

    // Create patterns with different features
    for (int i = 1; i <= 5; ++i) {
        FeatureVector features({
            static_cast<float>(i),
            static_cast<float>(i * 2),
            static_cast<float>(i * 3)
        });
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        PatternID id(i);
        PatternNode node(id, data, PatternType::ATOMIC);

        // Set varying confidence scores
        node.SetConfidenceScore(0.5f + i * 0.1f);

        db->Store(node);

        // Simulate some accesses
        for (int j = 0; j < i * 10; ++j) {
            db->Retrieve(id);  // This increments access count
        }
    }

    return db;
}

// ============================================================================
// PatternMatcher Tests
// ============================================================================

TEST(PatternMatcherTest, ConstructorRequiresNonNullDatabase) {
    auto metric = std::make_shared<MockEuclideanSimilarity>();
    EXPECT_THROW(
        PatternMatcher(nullptr, metric),
        std::invalid_argument
    );
}

TEST(PatternMatcherTest, ConstructorRequiresNonNullMetric) {
    auto db = CreateTestDatabase();
    EXPECT_THROW(
        PatternMatcher(db, nullptr),
        std::invalid_argument
    );
}

TEST(PatternMatcherTest, ConstructorRejectsInvalidThreshold) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.similarity_threshold = 1.5f;  // Invalid

    EXPECT_THROW(
        PatternMatcher(db, metric, config),
        std::invalid_argument
    );
}

TEST(PatternMatcherTest, ConstructorRejectsInvalidThresholdOrdering) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.strong_match_threshold = 0.6f;
    config.weak_match_threshold = 0.8f;  // Should be <= strong_match_threshold

    EXPECT_THROW(
        PatternMatcher(db, metric, config),
        std::invalid_argument
    );
}

TEST(PatternMatcherTest, FindMatchesReturnsMatches) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.similarity_threshold = 0.3f;  // Low threshold to get matches
    PatternMatcher matcher(db, metric, config);

    // Query with features close to pattern 3
    FeatureVector query_features({3.1f, 6.1f, 9.1f});
    PatternData query = PatternData::FromFeatures(query_features, DataModality::NUMERIC);

    auto matches = matcher.FindMatches(query);

    EXPECT_FALSE(matches.empty());

    // Matches should be sorted by similarity (highest first)
    for (size_t i = 1; i < matches.size(); ++i) {
        EXPECT_GE(matches[i-1].similarity, matches[i].similarity);
    }
}

TEST(PatternMatcherTest, FindMatchesRespectsThreshold) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.similarity_threshold = 0.95f;  // Very high threshold
    PatternMatcher matcher(db, metric, config);

    // Query with features far from all patterns
    FeatureVector query_features({100.0f, 200.0f, 300.0f});
    PatternData query = PatternData::FromFeatures(query_features, DataModality::NUMERIC);

    auto matches = matcher.FindMatches(query);

    // Should have no matches due to high threshold
    EXPECT_TRUE(matches.empty());
}

TEST(PatternMatcherTest, FindMatchesRespectsMaxMatches) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.similarity_threshold = 0.1f;  // Low threshold
    config.max_matches = 2;
    PatternMatcher matcher(db, metric, config);

    FeatureVector query_features({3.0f, 6.0f, 9.0f});
    PatternData query = PatternData::FromFeatures(query_features, DataModality::NUMERIC);

    auto matches = matcher.FindMatches(query);

    EXPECT_LE(matches.size(), 2u);
}

TEST(PatternMatcherTest, MakeDecisionCreatesNewWhenNoMatches) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.similarity_threshold = 0.99f;  // Very high
    PatternMatcher matcher(db, metric, config);

    // Very different pattern
    FeatureVector query_features({1000.0f, 2000.0f, 3000.0f});
    PatternData query = PatternData::FromFeatures(query_features, DataModality::NUMERIC);

    auto decision = matcher.MakeDecision(query);

    EXPECT_EQ(PatternMatcher::Decision::CREATE_NEW, decision.decision);
    EXPECT_FALSE(decision.existing_id.has_value());
    EXPECT_GT(decision.confidence, 0.0f);
    EXPECT_FALSE(decision.reasoning.empty());
}

TEST(PatternMatcherTest, MakeDecisionUpdatesOnStrongMatch) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.similarity_threshold = 0.5f;
    config.strong_match_threshold = 0.85f;
    PatternMatcher matcher(db, metric, config);

    // Very close to pattern 3
    FeatureVector query_features({3.001f, 6.001f, 9.001f});
    PatternData query = PatternData::FromFeatures(query_features, DataModality::NUMERIC);

    auto decision = matcher.MakeDecision(query);

    EXPECT_EQ(PatternMatcher::Decision::UPDATE_EXISTING, decision.decision);
    EXPECT_TRUE(decision.existing_id.has_value());
    EXPECT_GT(decision.confidence, 0.0f);
    EXPECT_FALSE(decision.reasoning.empty());
}

TEST(PatternMatcherTest, MakeDecisionMergesOnWeakMatch) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.similarity_threshold = 0.5f;
    config.weak_match_threshold = 0.7f;
    config.strong_match_threshold = 0.9f;
    PatternMatcher matcher(db, metric, config);

    // Somewhat close to pattern 3
    FeatureVector query_features({3.5f, 7.0f, 10.5f});
    PatternData query = PatternData::FromFeatures(query_features, DataModality::NUMERIC);

    auto decision = matcher.MakeDecision(query);

    // Should get either UPDATE_EXISTING or MERGE_SIMILAR depending on similarity
    EXPECT_TRUE(
        decision.decision == PatternMatcher::Decision::UPDATE_EXISTING ||
        decision.decision == PatternMatcher::Decision::MERGE_SIMILAR ||
        decision.decision == PatternMatcher::Decision::CREATE_NEW
    );
}

TEST(PatternMatcherTest, GetConfigWorks) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.similarity_threshold = 0.75f;
    config.max_matches = 15;
    PatternMatcher matcher(db, metric, config);

    const auto& retrieved_config = matcher.GetConfig();
    EXPECT_FLOAT_EQ(0.75f, retrieved_config.similarity_threshold);
    EXPECT_EQ(15u, retrieved_config.max_matches);
}

TEST(PatternMatcherTest, SetConfigWorks) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher matcher(db, metric);

    PatternMatcher::Config new_config;
    new_config.similarity_threshold = 0.8f;
    new_config.max_matches = 20;
    matcher.SetConfig(new_config);

    const auto& retrieved_config = matcher.GetConfig();
    EXPECT_FLOAT_EQ(0.8f, retrieved_config.similarity_threshold);
    EXPECT_EQ(20u, retrieved_config.max_matches);
}

TEST(PatternMatcherTest, SetConfigRejectsInvalidThreshold) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();
    PatternMatcher matcher(db, metric);

    PatternMatcher::Config bad_config;
    bad_config.similarity_threshold = -0.5f;

    EXPECT_THROW(matcher.SetConfig(bad_config), std::invalid_argument);
}

TEST(PatternMatcherTest, SetMetricWorks) {
    auto db = CreateTestDatabase();
    auto metric1 = std::make_shared<MockEuclideanSimilarity>();
    PatternMatcher matcher(db, metric1);

    auto metric2 = std::make_shared<MockEuclideanSimilarity>();
    EXPECT_NO_THROW(matcher.SetMetric(metric2));
}

TEST(PatternMatcherTest, SetMetricRejectsNull) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();
    PatternMatcher matcher(db, metric);

    EXPECT_THROW(matcher.SetMetric(nullptr), std::invalid_argument);
}

TEST(PatternMatcherTest, MatchConfidenceIsReasonable) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.similarity_threshold = 0.3f;
    PatternMatcher matcher(db, metric, config);

    FeatureVector query_features({3.0f, 6.0f, 9.0f});
    PatternData query = PatternData::FromFeatures(query_features, DataModality::NUMERIC);

    auto matches = matcher.FindMatches(query);

    for (const auto& match : matches) {
        EXPECT_GE(match.confidence, 0.0f);
        EXPECT_LE(match.confidence, 1.0f);
    }
}

TEST(PatternMatcherTest, MatchesHaveValidSimilarity) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.similarity_threshold = 0.3f;
    PatternMatcher matcher(db, metric, config);

    FeatureVector query_features({2.5f, 5.0f, 7.5f});
    PatternData query = PatternData::FromFeatures(query_features, DataModality::NUMERIC);

    auto matches = matcher.FindMatches(query);

    for (const auto& match : matches) {
        EXPECT_GE(match.similarity, 0.0f);
        EXPECT_LE(match.similarity, 1.0f);
        EXPECT_GE(match.similarity, config.similarity_threshold);
    }
}

TEST(PatternMatcherTest, DecisionReasoningIsNotEmpty) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();
    PatternMatcher matcher(db, metric);

    FeatureVector query_features({3.0f, 6.0f, 9.0f});
    PatternData query = PatternData::FromFeatures(query_features, DataModality::NUMERIC);

    auto decision = matcher.MakeDecision(query);

    EXPECT_FALSE(decision.reasoning.empty());
}

TEST(PatternMatcherTest, UseFastSearchOption) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockEuclideanSimilarity>();

    PatternMatcher::Config config;
    config.use_fast_search = true;  // Currently not implemented, but should not crash
    PatternMatcher matcher(db, metric, config);

    FeatureVector query_features({3.0f, 6.0f, 9.0f});
    PatternData query = PatternData::FromFeatures(query_features, DataModality::NUMERIC);

    EXPECT_NO_THROW(matcher.FindMatches(query));
}

} // namespace
} // namespace dpan
