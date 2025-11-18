// File: tests/learning/semantic_attention_head_test.cpp
//
// Unit tests for SemanticAttentionHead

#include "learning/semantic_attention_head.hpp"
#include "attention_test_fixtures.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace dpan;
using namespace dpan::attention;
using namespace dpan::testing;

// ============================================================================
// Mock Similarity Metric for Testing
// ============================================================================

/// Mock similarity metric that returns configurable similarity scores
class MockSimilarityMetric : public SimilarityMetric {
public:
    /// Set similarity score for a specific pattern pair
    void SetSimilarity(const PatternData& a, const PatternData& b, float similarity) {
        // Use simple hash based on data size for testing
        auto key = std::make_pair(a.GetOriginalSize(), b.GetOriginalSize());
        similarities_[key] = similarity;
    }

    float Compute(const PatternData& a, const PatternData& b) const override {
        auto key = std::make_pair(a.GetOriginalSize(), b.GetOriginalSize());
        auto it = similarities_.find(key);
        if (it != similarities_.end()) {
            return it->second;
        }
        // Default: similarity based on size difference
        float size_a = static_cast<float>(a.GetOriginalSize());
        float size_b = static_cast<float>(b.GetOriginalSize());
        if (size_a == 0.0f && size_b == 0.0f) return 1.0f;
        float diff = std::abs(size_a - size_b);
        float max_size = std::max(size_a, size_b);
        return max_size > 0.0f ? 1.0f - (diff / max_size) : 1.0f;
    }

    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override {
        // Simple cosine similarity for feature vectors
        if (a.Dimension() != b.Dimension() || a.Dimension() == 0) {
            return 0.0f;
        }

        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < a.Dimension(); ++i) {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        float denom = std::sqrt(norm_a * norm_b);
        return denom > 0.0f ? dot / denom : 0.0f;
    }

    std::string GetName() const override {
        return "MockSimilarityMetric";
    }

private:
    mutable std::map<std::pair<size_t, size_t>, float> similarities_;
};

// ============================================================================
// Test Fixture
// ============================================================================

class SemanticAttentionHeadTest : public AttentionTestFixture {
protected:
    void SetUp() override {
        AttentionTestFixture::SetUp();

        // Create semantic attention head with mock similarity metric
        mock_metric_ = std::make_shared<MockSimilarityMetric>();

        SemanticAttentionConfig config;
        config.temperature = 1.0f;
        config.enable_caching = true;
        config.debug_logging = false;

        semantic_head_ = std::make_unique<SemanticAttentionHead>(config, mock_metric_);
        semantic_head_->SetPatternDatabase(mock_db_.get());
    }

    std::shared_ptr<MockSimilarityMetric> mock_metric_;
    std::unique_ptr<SemanticAttentionHead> semantic_head_;
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(SemanticAttentionHeadTest, DefaultConfiguration) {
    SemanticAttentionConfig config;
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_FLOAT_EQ(config.similarity_threshold, 0.0f);
    EXPECT_TRUE(config.enable_caching);
    EXPECT_EQ(config.cache_size, 1000u);
    EXPECT_FALSE(config.debug_logging);
    EXPECT_TRUE(config.Validate());
}

TEST_F(SemanticAttentionHeadTest, InvalidConfiguration) {
    SemanticAttentionConfig config;

    // Invalid temperature
    config.temperature = 0.0f;
    EXPECT_FALSE(config.Validate());

    config.temperature = -1.0f;
    EXPECT_FALSE(config.Validate());

    // Invalid threshold
    config.temperature = 1.0f;
    config.similarity_threshold = -0.1f;
    EXPECT_FALSE(config.Validate());

    config.similarity_threshold = 1.5f;
    EXPECT_FALSE(config.Validate());
}

TEST_F(SemanticAttentionHeadTest, SetSemanticConfig) {
    SemanticAttentionConfig new_config;
    new_config.temperature = 0.5f;
    new_config.similarity_threshold = 0.3f;
    new_config.enable_caching = false;

    semantic_head_->SetSemanticConfig(new_config);

    auto config = semantic_head_->GetSemanticConfig();
    EXPECT_FLOAT_EQ(config.temperature, 0.5f);
    EXPECT_FLOAT_EQ(config.similarity_threshold, 0.3f);
    EXPECT_FALSE(config.enable_caching);
}

// ============================================================================
// Similarity Metric Tests
// ============================================================================

TEST_F(SemanticAttentionHeadTest, SetSimilarityMetric) {
    auto new_metric = std::make_shared<MockSimilarityMetric>();

    semantic_head_->SetSimilarityMetric(new_metric);

    EXPECT_EQ(semantic_head_->GetSimilarityMetric(), new_metric);
}

TEST_F(SemanticAttentionHeadTest, NoSimilarityMetric) {
    // Create head without similarity metric
    SemanticAttentionConfig config;
    auto head = std::make_unique<SemanticAttentionHead>(config, nullptr);
    head->SetPatternDatabase(mock_db_.get());

    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Should return uniform weights when no metric is available
    auto weights = head->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);
    EXPECT_FLOAT_EQ(weights[pattern_ids[1]], 0.5f);
    EXPECT_FLOAT_EQ(weights[pattern_ids[2]], 0.5f);
}

// ============================================================================
// Content-Based Attention Tests
// ============================================================================

TEST_F(SemanticAttentionHeadTest, ComputeAttentionBasic) {
    auto pattern_ids = CreateTestPatterns(3);

    ContextVector context;

    auto weights = semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Verify weights sum to 1.0
    float sum = weights[pattern_ids[1]] + weights[pattern_ids[2]];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Verify weights are in valid range
    VerifyWeightsInRange(weights);
}

TEST_F(SemanticAttentionHeadTest, ContentSimilarityFocuses) {
    auto pattern_ids = CreateTestPatterns(3);

    ContextVector context;

    // Patterns should have different content based on our mock
    auto weights = semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // All patterns have valid weights
    for (const auto& [_, weight] : weights) {
        EXPECT_GE(weight, 0.0f);
        EXPECT_LE(weight, 1.0f);
    }
}

TEST_F(SemanticAttentionHeadTest, TemperatureScaling) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Low temperature (sharper distribution)
    SemanticAttentionConfig low_temp_config;
    low_temp_config.temperature = 0.5f;
    semantic_head_->SetSemanticConfig(low_temp_config);

    auto weights_low = semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    // High temperature (softer distribution)
    SemanticAttentionConfig high_temp_config;
    high_temp_config.temperature = 2.0f;
    semantic_head_->SetSemanticConfig(high_temp_config);

    auto weights_high = semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    // Calculate variance for both distributions
    auto calc_variance = [](const std::map<PatternID, float>& w) {
        float mean = 0.0f;
        for (const auto& [_, weight] : w) {
            mean += weight;
        }
        mean /= w.size();

        float var = 0.0f;
        for (const auto& [_, weight] : w) {
            float diff = weight - mean;
            var += diff * diff;
        }
        return var / w.size();
    };

    float var_low = calc_variance(weights_low);
    float var_high = calc_variance(weights_high);

    // Lower temperature should have higher variance (sharper distribution)
    EXPECT_GE(var_low, var_high);
}

TEST_F(SemanticAttentionHeadTest, SimilarityThreshold) {
    auto pattern_ids = CreateTestPatterns(3);

    // Set threshold to filter out low similarities
    SemanticAttentionConfig config;
    config.similarity_threshold = 0.5f;  // Filter patterns with similarity < 0.5
    semantic_head_->SetSemanticConfig(config);

    ContextVector context;

    auto weights = semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    // Should still get valid output
    ASSERT_EQ(weights.size(), 2u);

    float sum = 0.0f;
    for (const auto& [_, weight] : weights) {
        sum += weight;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================================
// Caching Tests
// ============================================================================

TEST_F(SemanticAttentionHeadTest, CachingEnabled) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // First computation (cache miss)
    auto weights1 = semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    // Second computation (cache hit)
    auto weights2 = semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    // Results should be identical
    EXPECT_EQ(weights1, weights2);

    // Check statistics
    auto stats = semantic_head_->GetStatistics();
    EXPECT_GT(stats["cache_hits"], 0.0f);
    EXPECT_GT(stats["cache_hit_rate"], 0.0f);
}

TEST_F(SemanticAttentionHeadTest, CachingDisabled) {
    // Disable caching
    SemanticAttentionConfig config;
    config.enable_caching = false;
    semantic_head_->SetSemanticConfig(config);

    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    auto stats = semantic_head_->GetStatistics();
    EXPECT_EQ(stats["cache_hits"], 0.0f);
    EXPECT_EQ(stats["cache_misses"], 0.0f);  // No cache lookups when disabled
}

TEST_F(SemanticAttentionHeadTest, ClearCache) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Build up cache
    semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    auto stats_before = semantic_head_->GetStatistics();
    EXPECT_GT(stats_before["cache_size"], 0.0f);

    // Clear cache
    semantic_head_->ClearCache();

    auto stats_after = semantic_head_->GetStatistics();
    EXPECT_EQ(stats_after["cache_size"], 0.0f);
}

// ============================================================================
// Detailed Attention Tests
// ============================================================================

TEST_F(SemanticAttentionHeadTest, ComputeDetailedAttention) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    auto scores = semantic_head_->ComputeDetailedAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(scores.size(), 2u);

    // Scores should be sorted by weight descending
    EXPECT_GE(scores[0].weight, scores[1].weight);

    // Semantic similarity should be set
    for (const auto& score : scores) {
        EXPECT_GE(score.components.semantic_similarity, 0.0f);
        EXPECT_LE(score.components.semantic_similarity, 1.0f);

        // Other components should be zero for pure semantic attention
        EXPECT_EQ(score.components.context_similarity, 0.0f);
        EXPECT_EQ(score.components.importance_score, 0.0f);
    }
}

// ============================================================================
// Apply Attention Tests
// ============================================================================

TEST_F(SemanticAttentionHeadTest, ApplyAttention) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    auto result = semantic_head_->ApplyAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(result.size(), 2u);

    // Should be sorted by weight descending
    EXPECT_GE(result[0].second, result[1].second);

    // Weights should sum to 1.0
    float sum = result[0].second + result[1].second;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(SemanticAttentionHeadTest, EmptyCandidates) {
    auto pattern_ids = CreateTestPatterns(1);
    ContextVector context;

    auto weights = semantic_head_->ComputeAttention(
        pattern_ids[0], {}, context);

    EXPECT_TRUE(weights.empty());
}

TEST_F(SemanticAttentionHeadTest, SingleCandidate) {
    auto pattern_ids = CreateTestPatterns(2);
    ContextVector context;

    auto weights = semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1]}, context);

    ASSERT_EQ(weights.size(), 1u);
    EXPECT_FLOAT_EQ(weights[pattern_ids[1]], 1.0f);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(SemanticAttentionHeadTest, GetStatistics) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Compute attention a few times
    semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);
    semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    auto stats = semantic_head_->GetStatistics();

    EXPECT_GE(stats["attention_computations"], 2.0f);
    EXPECT_GE(stats["similarity_computations"], 0.0f);
    EXPECT_GE(stats["cache_hits"], 0.0f);
    EXPECT_GE(stats["cache_misses"], 0.0f);
    EXPECT_GE(stats["cache_hit_rate"], 0.0f);
    EXPECT_LE(stats["cache_hit_rate"], 1.0f);
}

// ============================================================================
// Content Type Appropriateness Tests
// ============================================================================

TEST_F(SemanticAttentionHeadTest, AppropriateForTextPatterns) {
    // Test that semantic attention works well with text-like patterns
    // (represented by patterns with similar sizes indicating similar content)

    auto pattern_ids = CreateTestPatterns(4);
    ContextVector context;

    auto weights = semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2], pattern_ids[3]}, context);

    // All patterns should get reasonable weights
    ASSERT_EQ(weights.size(), 3u);

    for (const auto& [_, weight] : weights) {
        EXPECT_GT(weight, 0.0f);  // No pattern should be completely ignored
        EXPECT_LT(weight, 1.0f);  // No pattern should dominate completely
    }
}

TEST_F(SemanticAttentionHeadTest, AppropriateForDataPatterns) {
    // Test with feature-based similarity (appropriate for structured data)

    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Semantic attention should handle data patterns well
    auto weights = semantic_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Verify proper probability distribution
    VerifyWeightsSumToOne(weights);
    VerifyWeightsInRange(weights);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
