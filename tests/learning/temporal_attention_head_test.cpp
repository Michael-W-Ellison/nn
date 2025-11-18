// File: tests/learning/temporal_attention_head_test.cpp
//
// Unit tests for TemporalAttentionHead

#include "learning/temporal_attention_head.hpp"
#include "attention_test_fixtures.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>

using namespace dpan;
using namespace dpan::attention;
using namespace dpan::testing;

// ============================================================================
// Test Fixture
// ============================================================================

class TemporalAttentionHeadTest : public AttentionTestFixture {
protected:
    void SetUp() override {
        AttentionTestFixture::SetUp();

        // Create temporal attention head
        TemporalAttentionConfig config;
        config.decay_constant_ms = 1000.0f;  // 1 second decay
        config.temperature = 1.0f;
        config.enable_caching = true;
        config.debug_logging = false;

        temporal_head_ = std::make_unique<TemporalAttentionHead>(config);
        temporal_head_->SetPatternDatabase(mock_db_.get());
    }

    /// Create test pattern with specific access time offset
    PatternID CreatePatternWithAccessTime(int64_t offset_ms) {
        auto node = CreateTestPattern();
        PatternID pattern_id = node.GetID();

        // Record access to set last_accessed time
        node.RecordAccess();

        // Store in database
        mock_db_->Store(node);

        // Sleep to create time offset if needed
        if (offset_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(offset_ms));
        }

        return pattern_id;
    }

    std::unique_ptr<TemporalAttentionHead> temporal_head_;
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(TemporalAttentionHeadTest, DefaultConfiguration) {
    TemporalAttentionConfig config;
    EXPECT_FLOAT_EQ(config.decay_constant_ms, 1000.0f);
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_FLOAT_EQ(config.min_age_threshold_ms, 0.0f);
    EXPECT_FALSE(config.enable_caching);
    EXPECT_EQ(config.cache_size, 100u);
    EXPECT_FALSE(config.debug_logging);
    EXPECT_TRUE(config.Validate());
}

TEST_F(TemporalAttentionHeadTest, InvalidConfiguration) {
    TemporalAttentionConfig config;

    // Invalid decay constant
    config.decay_constant_ms = 0.0f;
    EXPECT_FALSE(config.Validate());

    config.decay_constant_ms = -1.0f;
    EXPECT_FALSE(config.Validate());

    // Invalid temperature
    config.decay_constant_ms = 1000.0f;
    config.temperature = 0.0f;
    EXPECT_FALSE(config.Validate());

    config.temperature = -1.0f;
    EXPECT_FALSE(config.Validate());

    // Invalid min age threshold
    config.temperature = 1.0f;
    config.min_age_threshold_ms = -1.0f;
    EXPECT_FALSE(config.Validate());
}

TEST_F(TemporalAttentionHeadTest, SetTemporalConfig) {
    TemporalAttentionConfig new_config;
    new_config.decay_constant_ms = 500.0f;
    new_config.temperature = 0.5f;
    new_config.enable_caching = false;

    temporal_head_->SetTemporalConfig(new_config);

    auto config = temporal_head_->GetTemporalConfig();
    EXPECT_FLOAT_EQ(config.decay_constant_ms, 500.0f);
    EXPECT_FLOAT_EQ(config.temperature, 0.5f);
    EXPECT_FALSE(config.enable_caching);
}

// ============================================================================
// Temporal Scoring Tests
// ============================================================================

TEST_F(TemporalAttentionHeadTest, ComputeAttentionBasic) {
    auto pattern_ids = CreateTestPatterns(3);

    ContextVector context;

    auto weights = temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Verify weights sum to 1.0
    float sum = weights[pattern_ids[1]] + weights[pattern_ids[2]];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Verify weights are in valid range
    VerifyWeightsInRange(weights);
}

TEST_F(TemporalAttentionHeadTest, RecentPatternsFavored) {
    // Create patterns with different access times
    // We'll create them in sequence, so newer ones are more recent

    auto old_pattern = CreatePatternWithAccessTime(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto mid_pattern = CreatePatternWithAccessTime(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    auto recent_pattern = CreatePatternWithAccessTime(0);

    ContextVector context;

    auto weights = temporal_head_->ComputeAttention(
        old_pattern, {old_pattern, mid_pattern, recent_pattern}, context);

    ASSERT_EQ(weights.size(), 3u);

    // Most recent pattern should have highest weight
    EXPECT_GT(weights[recent_pattern], weights[mid_pattern]);
    EXPECT_GT(weights[mid_pattern], weights[old_pattern]);
}

TEST_F(TemporalAttentionHeadTest, ExponentialDecay) {
    // Create pattern and access it
    auto node = CreateTestPattern();
    PatternID pattern_id = node.GetID();
    node.RecordAccess();
    mock_db_->Store(node);

    // Wait for some time to pass
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Create a very recent pattern for comparison
    auto recent_node = CreateTestPattern();
    PatternID recent_pattern = recent_node.GetID();
    recent_node.RecordAccess();
    mock_db_->Store(recent_node);

    ContextVector context;

    auto weights = temporal_head_->ComputeAttention(
        pattern_id, {pattern_id, recent_pattern}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Recent pattern should have higher weight due to exponential decay
    EXPECT_GT(weights[recent_pattern], weights[pattern_id]);
}

TEST_F(TemporalAttentionHeadTest, DecayConstantEffect) {
    auto pattern_ids = CreateTestPatterns(2);

    // Record access for both
    for (auto id : pattern_ids) {
        auto pattern_opt = mock_db_->Retrieve(id);
        if (pattern_opt) {
            auto& pattern = const_cast<PatternNode&>(*pattern_opt);
            pattern.RecordAccess();
        }
    }

    // Wait a bit
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    ContextVector context;

    // Test with fast decay (small constant)
    TemporalAttentionConfig fast_decay_config;
    fast_decay_config.decay_constant_ms = 50.0f;  // Fast decay
    temporal_head_->SetTemporalConfig(fast_decay_config);

    auto weights_fast = temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[0], pattern_ids[1]}, context);

    // Test with slow decay (large constant)
    TemporalAttentionConfig slow_decay_config;
    slow_decay_config.decay_constant_ms = 10000.0f;  // Slow decay
    temporal_head_->SetTemporalConfig(slow_decay_config);

    auto weights_slow = temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[0], pattern_ids[1]}, context);

    // With fast decay, even small time differences create large score differences
    // With slow decay, small time differences create small score differences
    float diff_fast = std::abs(weights_fast[pattern_ids[0]] - weights_fast[pattern_ids[1]]);
    float diff_slow = std::abs(weights_slow[pattern_ids[0]] - weights_slow[pattern_ids[1]]);

    // Slow decay should have smaller difference (more uniform over time)
    EXPECT_LE(diff_slow, diff_fast);
}

TEST_F(TemporalAttentionHeadTest, TemperatureScaling) {
    auto pattern_ids = CreateTestPatterns(3);

    // Access all patterns
    for (auto id : pattern_ids) {
        auto pattern_opt = mock_db_->Retrieve(id);
        if (pattern_opt) {
            auto& pattern = const_cast<PatternNode&>(*pattern_opt);
            pattern.RecordAccess();
        }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Create one recent pattern
    auto recent_node = CreateTestPattern();
    PatternID recent = recent_node.GetID();
    recent_node.RecordAccess();
    mock_db_->Store(recent_node);

    ContextVector context;

    // Low temperature (sharper distribution)
    TemporalAttentionConfig low_temp_config;
    low_temp_config.temperature = 0.5f;
    low_temp_config.decay_constant_ms = 1000.0f;
    temporal_head_->SetTemporalConfig(low_temp_config);

    auto weights_low = temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[0], pattern_ids[1], recent}, context);

    // High temperature (softer distribution)
    TemporalAttentionConfig high_temp_config;
    high_temp_config.temperature = 2.0f;
    high_temp_config.decay_constant_ms = 1000.0f;
    temporal_head_->SetTemporalConfig(high_temp_config);

    auto weights_high = temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[0], pattern_ids[1], recent}, context);

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

    // Lower temperature should have higher variance (sharper)
    EXPECT_GE(var_low, var_high);
}

// ============================================================================
// Caching Tests
// ============================================================================

TEST_F(TemporalAttentionHeadTest, CachingEnabled) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // First computation (cache miss)
    auto weights1 = temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    // Second computation immediately (cache hit - within 100ms threshold)
    auto weights2 = temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    // Results should be similar (cached values)
    EXPECT_EQ(weights1.size(), weights2.size());

    // Check statistics
    auto stats = temporal_head_->GetStatistics();
    EXPECT_GT(stats["cache_hits"], 0.0f);
}

TEST_F(TemporalAttentionHeadTest, CachingDisabled) {
    // Disable caching
    TemporalAttentionConfig config;
    config.enable_caching = false;
    temporal_head_->SetTemporalConfig(config);

    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    auto stats = temporal_head_->GetStatistics();
    EXPECT_EQ(stats["cache_hits"], 0.0f);
    EXPECT_EQ(stats["cache_misses"], 0.0f);  // No cache lookups when disabled
}

TEST_F(TemporalAttentionHeadTest, ClearCache) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Build up cache
    temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    auto stats_before = temporal_head_->GetStatistics();
    EXPECT_GT(stats_before["cache_size"], 0.0f);

    // Clear cache
    temporal_head_->ClearCache();

    auto stats_after = temporal_head_->GetStatistics();
    EXPECT_EQ(stats_after["cache_size"], 0.0f);
}

// ============================================================================
// Detailed Attention Tests
// ============================================================================

TEST_F(TemporalAttentionHeadTest, ComputeDetailedAttention) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    auto scores = temporal_head_->ComputeDetailedAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(scores.size(), 2u);

    // Scores should be sorted by weight descending
    EXPECT_GE(scores[0].weight, scores[1].weight);

    // Temporal score should be set
    for (const auto& score : scores) {
        EXPECT_GE(score.components.temporal_score, 0.0f);
        EXPECT_LE(score.components.temporal_score, 1.0f);

        // Other components should be zero for pure temporal attention
        EXPECT_EQ(score.components.semantic_similarity, 0.0f);
        EXPECT_EQ(score.components.context_similarity, 0.0f);
        EXPECT_EQ(score.components.importance_score, 0.0f);
    }
}

// ============================================================================
// Apply Attention Tests
// ============================================================================

TEST_F(TemporalAttentionHeadTest, ApplyAttention) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    auto result = temporal_head_->ApplyAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(result.size(), 2u);

    // Should be sorted by weight descending (most recent first)
    EXPECT_GE(result[0].second, result[1].second);

    // Weights should sum to 1.0
    float sum = result[0].second + result[1].second;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(TemporalAttentionHeadTest, EmptyCandidates) {
    auto pattern_ids = CreateTestPatterns(1);
    ContextVector context;

    auto weights = temporal_head_->ComputeAttention(
        pattern_ids[0], {}, context);

    EXPECT_TRUE(weights.empty());
}

TEST_F(TemporalAttentionHeadTest, SingleCandidate) {
    auto pattern_ids = CreateTestPatterns(2);
    ContextVector context;

    auto weights = temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1]}, context);

    ASSERT_EQ(weights.size(), 1u);
    EXPECT_FLOAT_EQ(weights[pattern_ids[1]], 1.0f);
}

TEST_F(TemporalAttentionHeadTest, NoPatternDatabase) {
    // Create head without pattern database
    TemporalAttentionConfig config;
    auto head = std::make_unique<TemporalAttentionHead>(config);

    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Should return uniform weights when no database is available
    auto weights = head->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Should be uniform
    EXPECT_NEAR(weights[pattern_ids[1]], 0.5f, 1e-5f);
    EXPECT_NEAR(weights[pattern_ids[2]], 0.5f, 1e-5f);
}

TEST_F(TemporalAttentionHeadTest, PatternsNeverAccessed) {
    // Create patterns but don't access them
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // All patterns have same timestamp (creation time)
    auto weights = temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Weights should be similar (same temporal score)
    float diff = std::abs(weights[pattern_ids[1]] - weights[pattern_ids[2]]);
    EXPECT_LT(diff, 0.1f);  // Should be very similar
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(TemporalAttentionHeadTest, GetStatistics) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Compute attention a few times
    temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);
    temporal_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    auto stats = temporal_head_->GetStatistics();

    EXPECT_GE(stats["attention_computations"], 2.0f);
    EXPECT_GE(stats["temporal_computations"], 0.0f);
    EXPECT_GE(stats["cache_hits"], 0.0f);
    EXPECT_GE(stats["cache_misses"], 0.0f);
    EXPECT_GE(stats["cache_hit_rate"], 0.0f);
    EXPECT_LE(stats["cache_hit_rate"], 1.0f);
}

// ============================================================================
// Time-Aware Behavior Tests
// ============================================================================

TEST_F(TemporalAttentionHeadTest, TimeProgression) {
    // Create first pattern (old)
    auto old_node = CreateTestPattern();
    PatternID old_pattern = old_node.GetID();
    old_node.RecordAccess();
    mock_db_->Store(old_node);

    // Wait for time to pass
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Create second pattern (recent)
    auto recent_node = CreateTestPattern();
    PatternID recent_pattern = recent_node.GetID();
    recent_node.RecordAccess();
    mock_db_->Store(recent_node);

    ContextVector context;

    // Measure attention
    auto weights = temporal_head_->ComputeAttention(
        old_pattern, {old_pattern, recent_pattern}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Recent pattern should have higher weight (more recent)
    EXPECT_GT(weights[recent_pattern], weights[old_pattern]);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
