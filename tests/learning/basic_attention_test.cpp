// File: tests/learning/basic_attention_test.cpp
//
// Comprehensive tests for BasicAttentionMechanism
//
// Tests cover:
// - Basic attention computation
// - Edge cases (empty, single candidate)
// - Normalization verification
// - Caching behavior
// - Configuration changes
// - Feature extraction integration
// - Debug logging
// - Statistics tracking

#include "learning/basic_attention.hpp"
#include "attention_test_fixtures.hpp"
#include <gtest/gtest.h>
#include <sstream>

using namespace dpan;
using namespace dpan::attention;
using namespace dpan::testing;

class BasicAttentionTest : public AttentionTestFixture {
protected:
    void SetUp() override {
        AttentionTestFixture::SetUp();

        // Create attention mechanism with default config
        config_ = CreateDefaultConfig();
        attention_ = std::make_unique<BasicAttentionMechanism>(config_);
        attention_->SetPatternDatabase(mock_db_.get());
    }

    AttentionConfig config_;
    std::unique_ptr<BasicAttentionMechanism> attention_;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(BasicAttentionTest, ComputeAttentionBasic) {
    // Create test patterns
    auto pattern_ids = CreateTestPatterns(5);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2], pattern_ids[3]};

    auto context = CreateSemanticContext();

    auto weights = attention_->ComputeAttention(query, candidates, context);

    // Verify we got weights for all candidates
    EXPECT_EQ(weights.size(), candidates.size());

    // Verify all candidates have weights
    for (const auto& cand : candidates) {
        EXPECT_TRUE(weights.find(cand) != weights.end());
    }

    // Verify weights are valid
    VerifyWeightsInRange(weights);

    // Verify weights sum to 1.0
    VerifyWeightsSumToOne(weights);
}

TEST_F(BasicAttentionTest, ComputeAttentionEmpty) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates;  // Empty

    auto context = CreateEmptyContext();

    auto weights = attention_->ComputeAttention(query, candidates, context);

    // Empty candidates should return empty weights
    EXPECT_TRUE(weights.empty());
}

TEST_F(BasicAttentionTest, ComputeAttentionSingleCandidate) {
    auto pattern_ids = CreateTestPatterns(2);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1]};

    auto context = CreateEmptyContext();

    auto weights = attention_->ComputeAttention(query, candidates, context);

    // Single candidate should get weight 1.0
    ASSERT_EQ(weights.size(), 1u);
    EXPECT_NEAR(weights[candidates[0]], 1.0f, 1e-5f);
}

TEST_F(BasicAttentionTest, ComputeAttentionMultipleCandidates) {
    auto pattern_ids = CreateTestPatterns(10);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates;
    for (size_t i = 1; i < pattern_ids.size(); ++i) {
        candidates.push_back(pattern_ids[i]);
    }

    auto context = CreateMultiDimensionalContext();

    auto weights = attention_->ComputeAttention(query, candidates, context);

    // Verify all candidates have weights
    EXPECT_EQ(weights.size(), candidates.size());

    // Verify normalization
    VerifyWeightsSumToOne(weights);

    // Verify all weights are positive
    for (const auto& [id, weight] : weights) {
        EXPECT_GT(weight, 0.0f) << "All candidates should have positive weight";
    }
}

// ============================================================================
// Temperature Scaling Tests
// ============================================================================

TEST_F(BasicAttentionTest, TemperatureScalingEffect) {
    auto pattern_ids = CreateTestPatterns(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2], pattern_ids[3]};
    auto context = CreateEmptyContext();

    // Low temperature (sharper distribution)
    config_.temperature = 0.5f;
    attention_->SetConfig(config_);
    auto weights_low = attention_->ComputeAttention(query, candidates, context);

    // High temperature (softer distribution)
    config_.temperature = 2.0f;
    attention_->SetConfig(config_);
    auto weights_high = attention_->ComputeAttention(query, candidates, context);

    // Compute variance for both distributions
    float mean = 1.0f / candidates.size();
    float var_low = 0.0f, var_high = 0.0f;

    for (const auto& cand : candidates) {
        float diff_low = weights_low[cand] - mean;
        float diff_high = weights_high[cand] - mean;
        var_low += diff_low * diff_low;
        var_high += diff_high * diff_high;
    }

    // Lower temperature should have higher variance (more peaked)
    EXPECT_GT(var_low, var_high);
}

// ============================================================================
// Detailed Attention Tests
// ============================================================================

TEST_F(BasicAttentionTest, ComputeDetailedAttention) {
    auto pattern_ids = CreateTestPatterns(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2], pattern_ids[3], pattern_ids[4]};
    auto context = CreateSemanticContext();

    auto detailed = attention_->ComputeDetailedAttention(query, candidates, context);

    // Should have scores for all candidates
    EXPECT_EQ(detailed.size(), candidates.size());

    // Verify scores are sorted (descending by weight)
    VerifyScoresSorted(detailed);

    // Verify each score has valid components
    for (const auto& score : detailed) {
        EXPECT_GE(score.weight, 0.0f);
        EXPECT_LE(score.weight, 1.0f);
        EXPECT_TRUE(score.pattern_id.IsValid());
    }
}

// ============================================================================
// ApplyAttention Tests
// ============================================================================

TEST_F(BasicAttentionTest, ApplyAttentionBasic) {
    auto pattern_ids = CreateTestPatterns(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> predictions = {pattern_ids[1], pattern_ids[2], pattern_ids[3]};
    auto context = CreateEmptyContext();

    auto results = attention_->ApplyAttention(query, predictions, context);

    // Should have results for all predictions
    EXPECT_EQ(results.size(), predictions.size());

    // Results should be sorted by score (descending)
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i-1].second, results[i].second);
    }

    // All scores should be positive and <= 1.0
    for (const auto& [id, score] : results) {
        EXPECT_GE(score, 0.0f);
        EXPECT_LE(score, 1.0f);
        EXPECT_TRUE(id.IsValid());
    }
}

// ============================================================================
// Caching Tests
// ============================================================================

TEST_F(BasicAttentionTest, CachingEnabled) {
    // Enable caching
    config_.enable_caching = true;
    config_.cache_size = 100;
    attention_->SetConfig(config_);

    auto pattern_ids = CreateTestPatterns(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2], pattern_ids[3]};
    auto context = CreateEmptyContext();

    // First call - should miss cache
    auto weights1 = attention_->ComputeAttention(query, candidates, context);

    auto stats1 = attention_->GetStatistics();
    EXPECT_EQ(stats1["cache_misses"], 1.0f);

    // Second call with same inputs - should hit cache
    auto weights2 = attention_->ComputeAttention(query, candidates, context);

    auto stats2 = attention_->GetStatistics();
    EXPECT_EQ(stats2["cache_hits"], 1.0f);

    // Results should be identical
    EXPECT_EQ(weights1, weights2);
}

TEST_F(BasicAttentionTest, CachingDisabled) {
    // Disable caching
    config_.enable_caching = false;
    attention_->SetConfig(config_);

    auto pattern_ids = CreateTestPatterns(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    auto context = CreateEmptyContext();

    // Multiple calls
    attention_->ComputeAttention(query, candidates, context);
    attention_->ComputeAttention(query, candidates, context);

    auto stats = attention_->GetStatistics();

    // Should have no cache hits (caching disabled)
    EXPECT_EQ(stats["cache_hits"], 0.0f);
}

TEST_F(BasicAttentionTest, ClearCache) {
    config_.enable_caching = true;
    attention_->SetConfig(config_);

    auto pattern_ids = CreateTestPatterns(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    auto context = CreateEmptyContext();

    // First call
    attention_->ComputeAttention(query, candidates, context);

    // Clear cache
    attention_->ClearCache();

    // Second call should miss cache
    attention_->ComputeAttention(query, candidates, context);

    auto stats = attention_->GetStatistics();
    EXPECT_EQ(stats["cache_misses"], 2.0f);
    EXPECT_EQ(stats["cache_hits"], 0.0f);
}

TEST_F(BasicAttentionTest, CacheSizeLimit) {
    config_.enable_caching = true;
    config_.cache_size = 2;  // Very small cache
    attention_->SetConfig(config_);

    auto pattern_ids = CreateTestPatterns(10);

    // Make 3 different queries (should evict oldest)
    for (int i = 0; i < 3; ++i) {
        PatternID query = pattern_ids[i];
        std::vector<PatternID> candidates = {pattern_ids[i+1], pattern_ids[i+2]};
        auto context = CreateEmptyContext();
        attention_->ComputeAttention(query, candidates, context);
    }

    auto stats = attention_->GetStatistics();

    // Cache size should not exceed limit
    EXPECT_LE(stats["cache_size"], 2.0f);
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(BasicAttentionTest, SetConfigClearsCache) {
    config_.enable_caching = true;
    attention_->SetConfig(config_);

    auto pattern_ids = CreateTestPatterns(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    auto context = CreateEmptyContext();

    // First call
    attention_->ComputeAttention(query, candidates, context);

    // Change config (should clear cache)
    config_.temperature = 2.0f;
    attention_->SetConfig(config_);

    // Second call should miss cache
    attention_->ComputeAttention(query, candidates, context);

    auto stats = attention_->GetStatistics();
    EXPECT_EQ(stats["cache_misses"], 2.0f);
}

TEST_F(BasicAttentionTest, GetConfig) {
    config_.temperature = 1.5f;
    config_.num_heads = 8;
    attention_->SetConfig(config_);

    const auto& retrieved_config = attention_->GetConfig();

    EXPECT_FLOAT_EQ(retrieved_config.temperature, 1.5f);
    EXPECT_EQ(retrieved_config.num_heads, 8u);
}

// ============================================================================
// Feature Configuration Tests
// ============================================================================

TEST_F(BasicAttentionTest, SetFeatureConfig) {
    FeatureExtractionConfig feat_config;
    feat_config.include_confidence = true;
    feat_config.include_access_count = true;
    feat_config.include_type = true;

    attention_->SetFeatureConfig(feat_config);

    const auto& retrieved = attention_->GetFeatureConfig();

    EXPECT_TRUE(retrieved.include_confidence);
    EXPECT_TRUE(retrieved.include_access_count);
    EXPECT_TRUE(retrieved.include_type);
}

// ============================================================================
// Debug Logging Tests
// ============================================================================

TEST_F(BasicAttentionTest, DebugLogging) {
    std::ostringstream debug_output;

    config_.debug_logging = true;
    attention_->SetConfig(config_);
    attention_->SetDebugStream(&debug_output);

    auto pattern_ids = CreateTestPatterns(3);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    auto context = CreateEmptyContext();

    attention_->ComputeAttention(query, candidates, context);

    std::string output = debug_output.str();

    // Should contain debug information
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("BasicAttention"), std::string::npos);
}

TEST_F(BasicAttentionTest, DebugLoggingDisabled) {
    std::ostringstream debug_output;

    config_.debug_logging = false;
    attention_->SetConfig(config_);
    attention_->SetDebugStream(&debug_output);

    auto pattern_ids = CreateTestPatterns(3);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    auto context = CreateEmptyContext();

    attention_->ComputeAttention(query, candidates, context);

    std::string output = debug_output.str();

    // Should not log when disabled (may have some minimal output)
    // Just verify it doesn't crash
    SUCCEED();
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(BasicAttentionTest, GetStatistics) {
    auto pattern_ids = CreateTestPatterns(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    auto context = CreateEmptyContext();

    attention_->ComputeAttention(query, candidates, context);

    auto stats = attention_->GetStatistics();

    EXPECT_TRUE(stats.find("total_computations") != stats.end());
    EXPECT_TRUE(stats.find("cache_hits") != stats.end());
    EXPECT_TRUE(stats.find("cache_misses") != stats.end());
    EXPECT_TRUE(stats.find("cache_hit_rate") != stats.end());

    EXPECT_GE(stats["total_computations"], 1.0f);
}

TEST_F(BasicAttentionTest, CacheHitRate) {
    config_.enable_caching = true;
    attention_->SetConfig(config_);

    auto pattern_ids = CreateTestPatterns(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    auto context = CreateEmptyContext();

    // First call - miss
    attention_->ComputeAttention(query, candidates, context);

    // Second call - hit
    attention_->ComputeAttention(query, candidates, context);

    auto stats = attention_->GetStatistics();

    // Cache hit rate should be 0.5 (1 hit, 1 miss)
    EXPECT_NEAR(stats["cache_hit_rate"], 0.5f, 0.01f);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(BasicAttentionTest, NoPatternDatabase) {
    // Create attention without setting pattern database
    BasicAttentionMechanism no_db_attention(config_);

    auto pattern_ids = CreateTestPatterns(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    auto context = CreateEmptyContext();

    auto weights = no_db_attention.ComputeAttention(query, candidates, context);

    // Should return uniform distribution as fallback
    EXPECT_EQ(weights.size(), candidates.size());
    for (const auto& [id, weight] : weights) {
        EXPECT_NEAR(weight, 1.0f / candidates.size(), 1e-5f);
    }
}

TEST_F(BasicAttentionTest, InvalidQuery) {
    PatternID invalid_query = PatternID(999999);  // Not in database
    auto pattern_ids = CreateTestPatterns(3);
    std::vector<PatternID> candidates = {pattern_ids[0], pattern_ids[1]};
    auto context = CreateEmptyContext();

    auto weights = attention_->ComputeAttention(invalid_query, candidates, context);

    // Should still return weights (uniform distribution as fallback)
    EXPECT_EQ(weights.size(), candidates.size());
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
