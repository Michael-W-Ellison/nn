// File: tests/learning/association_attention_head_test.cpp
//
// Unit tests for AssociationAttentionHead

#include "learning/association_attention_head.hpp"
#include "attention_test_fixtures.hpp"
#include "association/association_matrix.hpp"
#include "association/association_edge.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace dpan;
using namespace dpan::attention;
using namespace dpan::testing;

// ============================================================================
// Test Fixture
// ============================================================================

class AssociationAttentionHeadTest : public AttentionTestFixture {
protected:
    void SetUp() override {
        AttentionTestFixture::SetUp();

        // Create association matrix
        association_matrix_ = std::make_unique<AssociationMatrix>();

        // Create association attention head
        AssociationAttentionConfig config;
        config.temperature = 1.0f;
        config.use_contextual_strength = false;
        config.default_strength = 0.1f;
        config.enable_caching = true;
        config.debug_logging = false;

        association_head_ = std::make_unique<AssociationAttentionHead>(config);
        association_head_->SetAssociationMatrix(association_matrix_.get());
    }

    /// Add association between patterns
    void AddAssociation(PatternID source, PatternID target, float strength,
                       AssociationType type = AssociationType::CATEGORICAL) {
        AssociationEdge edge(source, target, type, strength);
        association_matrix_->AddAssociation(edge);
    }

    std::unique_ptr<AssociationMatrix> association_matrix_;
    std::unique_ptr<AssociationAttentionHead> association_head_;
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(AssociationAttentionHeadTest, DefaultConfiguration) {
    AssociationAttentionConfig config;
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_FALSE(config.use_contextual_strength);
    EXPECT_FLOAT_EQ(config.strength_threshold, 0.0f);
    EXPECT_FLOAT_EQ(config.default_strength, 0.1f);
    EXPECT_FALSE(config.enable_caching);
    EXPECT_EQ(config.cache_size, 100u);
    EXPECT_FALSE(config.debug_logging);
    EXPECT_TRUE(config.Validate());
}

TEST_F(AssociationAttentionHeadTest, InvalidConfiguration) {
    AssociationAttentionConfig config;

    // Invalid temperature
    config.temperature = 0.0f;
    EXPECT_FALSE(config.Validate());

    config.temperature = -1.0f;
    EXPECT_FALSE(config.Validate());

    // Invalid strength threshold
    config.temperature = 1.0f;
    config.strength_threshold = -0.1f;
    EXPECT_FALSE(config.Validate());

    config.strength_threshold = 1.5f;
    EXPECT_FALSE(config.Validate());

    // Invalid default strength
    config.strength_threshold = 0.0f;
    config.default_strength = -0.1f;
    EXPECT_FALSE(config.Validate());

    config.default_strength = 1.5f;
    EXPECT_FALSE(config.Validate());
}

TEST_F(AssociationAttentionHeadTest, SetAssociationConfig) {
    AssociationAttentionConfig new_config;
    new_config.temperature = 0.5f;
    new_config.use_contextual_strength = true;
    new_config.default_strength = 0.2f;
    new_config.enable_caching = false;

    association_head_->SetAssociationConfig(new_config);

    auto config = association_head_->GetAssociationConfig();
    EXPECT_FLOAT_EQ(config.temperature, 0.5f);
    EXPECT_TRUE(config.use_contextual_strength);
    EXPECT_FLOAT_EQ(config.default_strength, 0.2f);
    EXPECT_FALSE(config.enable_caching);
}

// ============================================================================
// Association Strength Tests
// ============================================================================

TEST_F(AssociationAttentionHeadTest, StrongerAssociationHigherWeight) {
    auto pattern_ids = CreateTestPatterns(3);

    // Add associations with different strengths
    AddAssociation(pattern_ids[0], pattern_ids[1], 0.8f);  // Strong
    AddAssociation(pattern_ids[0], pattern_ids[2], 0.3f);  // Weak

    ContextVector context;

    auto weights = association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Stronger association should have higher weight
    EXPECT_GT(weights[pattern_ids[1]], weights[pattern_ids[2]]);
}

TEST_F(AssociationAttentionHeadTest, MissingAssociationUsesDefault) {
    auto pattern_ids = CreateTestPatterns(3);

    // Add association for only one candidate
    AddAssociation(pattern_ids[0], pattern_ids[1], 0.8f);
    // pattern_ids[2] has no association

    ContextVector context;

    auto weights = association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Pattern with association should have higher weight than missing one
    EXPECT_GT(weights[pattern_ids[1]], weights[pattern_ids[2]]);

    // Check statistics
    auto stats = association_head_->GetStatistics();
    EXPECT_GT(stats["missing_associations"], 0.0f);
}

TEST_F(AssociationAttentionHeadTest, EqualStrengthsEqualWeights) {
    auto pattern_ids = CreateTestPatterns(3);

    // Add associations with equal strengths
    AddAssociation(pattern_ids[0], pattern_ids[1], 0.5f);
    AddAssociation(pattern_ids[0], pattern_ids[2], 0.5f);

    ContextVector context;

    auto weights = association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Equal strengths should give approximately equal weights
    EXPECT_NEAR(weights[pattern_ids[1]], weights[pattern_ids[2]], 1e-5f);
}

TEST_F(AssociationAttentionHeadTest, MultipleAssociations) {
    auto pattern_ids = CreateTestPatterns(4);

    // Add associations with varying strengths
    AddAssociation(pattern_ids[0], pattern_ids[1], 0.9f);  // Strongest
    AddAssociation(pattern_ids[0], pattern_ids[2], 0.6f);  // Medium
    AddAssociation(pattern_ids[0], pattern_ids[3], 0.2f);  // Weakest

    ContextVector context;

    auto weights = association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2], pattern_ids[3]}, context);

    ASSERT_EQ(weights.size(), 3u);

    // Weights should be ordered by association strength
    EXPECT_GT(weights[pattern_ids[1]], weights[pattern_ids[2]]);
    EXPECT_GT(weights[pattern_ids[2]], weights[pattern_ids[3]]);

    // Weights should sum to 1.0
    float sum = weights[pattern_ids[1]] + weights[pattern_ids[2]] + weights[pattern_ids[3]];
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================================
// Strength Threshold Tests
// ============================================================================

TEST_F(AssociationAttentionHeadTest, StrengthThreshold) {
    auto pattern_ids = CreateTestPatterns(3);

    // Add associations
    AddAssociation(pattern_ids[0], pattern_ids[1], 0.7f);  // Above threshold
    AddAssociation(pattern_ids[0], pattern_ids[2], 0.3f);  // Below threshold

    // Set strength threshold
    AssociationAttentionConfig config;
    config.temperature = 1.0f;
    config.strength_threshold = 0.5f;  // Require at least 0.5
    association_head_->SetAssociationConfig(config);

    ContextVector context;

    auto weights = association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Pattern 1 (0.7 > threshold) should get most of the weight
    // Pattern 2 (0.3 < threshold) should be filtered to 0.0, then softmax normalized
    // After softmax: exp(0.7)/(exp(0.7)+exp(0.0)) ≈ 0.668
    EXPECT_GT(weights[pattern_ids[1]], 0.65f);  // Gets most weight (≈0.668)
    EXPECT_LT(weights[pattern_ids[2]], 0.35f);  // Gets less weight (≈0.332)
    EXPECT_GT(weights[pattern_ids[1]], weights[pattern_ids[2]]);  // pattern1 > pattern2
}

// ============================================================================
// Temperature Scaling Tests
// ============================================================================

TEST_F(AssociationAttentionHeadTest, TemperatureScaling) {
    auto pattern_ids = CreateTestPatterns(3);

    // Add associations with different strengths
    AddAssociation(pattern_ids[0], pattern_ids[1], 0.8f);
    AddAssociation(pattern_ids[0], pattern_ids[2], 0.4f);

    ContextVector context;

    // Low temperature (sharper distribution)
    AssociationAttentionConfig low_temp_config;
    low_temp_config.temperature = 0.5f;
    association_head_->SetAssociationConfig(low_temp_config);

    auto weights_low = association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    // Clear cache before changing config
    association_head_->ClearCache();

    // High temperature (softer distribution)
    AssociationAttentionConfig high_temp_config;
    high_temp_config.temperature = 2.0f;
    association_head_->SetAssociationConfig(high_temp_config);

    auto weights_high = association_head_->ComputeAttention(
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

    // Lower temperature should have higher variance (sharper)
    EXPECT_GE(var_low, var_high);
}

// ============================================================================
// Caching Tests
// ============================================================================

TEST_F(AssociationAttentionHeadTest, CachingEnabled) {
    auto pattern_ids = CreateTestPatterns(3);

    AddAssociation(pattern_ids[0], pattern_ids[1], 0.7f);
    AddAssociation(pattern_ids[0], pattern_ids[2], 0.5f);

    ContextVector context;

    // First computation (cache miss)
    auto weights1 = association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    // Second computation (cache hit)
    auto weights2 = association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    // Results should be identical
    EXPECT_EQ(weights1.size(), weights2.size());
    EXPECT_NEAR(weights1[pattern_ids[1]], weights2[pattern_ids[1]], 1e-6f);
    EXPECT_NEAR(weights1[pattern_ids[2]], weights2[pattern_ids[2]], 1e-6f);

    // Check statistics - should have cache hits from second call
    auto stats = association_head_->GetStatistics();
    EXPECT_GT(stats["cache_hits"], 0.0f);
}

TEST_F(AssociationAttentionHeadTest, CachingDisabled) {
    // Disable caching
    AssociationAttentionConfig config;
    config.enable_caching = false;
    association_head_->SetAssociationConfig(config);

    auto pattern_ids = CreateTestPatterns(3);

    AddAssociation(pattern_ids[0], pattern_ids[1], 0.7f);
    AddAssociation(pattern_ids[0], pattern_ids[2], 0.5f);

    ContextVector context;

    association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    auto stats = association_head_->GetStatistics();
    EXPECT_EQ(stats["cache_hits"], 0.0f);
    EXPECT_EQ(stats["cache_misses"], 0.0f);  // No cache lookups when disabled
}

TEST_F(AssociationAttentionHeadTest, ClearCache) {
    auto pattern_ids = CreateTestPatterns(3);

    AddAssociation(pattern_ids[0], pattern_ids[1], 0.7f);
    AddAssociation(pattern_ids[0], pattern_ids[2], 0.5f);

    ContextVector context;

    // Build up cache
    association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    auto stats_before = association_head_->GetStatistics();
    EXPECT_GT(stats_before["cache_size"], 0.0f);

    // Clear cache
    association_head_->ClearCache();

    auto stats_after = association_head_->GetStatistics();
    EXPECT_EQ(stats_after["cache_size"], 0.0f);
}

// ============================================================================
// Detailed Attention Tests
// ============================================================================

TEST_F(AssociationAttentionHeadTest, ComputeDetailedAttention) {
    auto pattern_ids = CreateTestPatterns(3);

    AddAssociation(pattern_ids[0], pattern_ids[1], 0.8f);
    AddAssociation(pattern_ids[0], pattern_ids[2], 0.4f);

    ContextVector context;

    auto scores = association_head_->ComputeDetailedAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(scores.size(), 2u);

    // Scores should be sorted by weight descending
    EXPECT_GE(scores[0].weight, scores[1].weight);

    // Importance score should be set (represents association strength)
    for (const auto& score : scores) {
        EXPECT_GE(score.components.importance_score, 0.0f);
        EXPECT_LE(score.components.importance_score, 1.0f);

        // Other components should be zero for pure association attention
        EXPECT_EQ(score.components.semantic_similarity, 0.0f);
        EXPECT_EQ(score.components.context_similarity, 0.0f);
        EXPECT_EQ(score.components.structural_score, 0.0f);
    }
}

// ============================================================================
// Apply Attention Tests
// ============================================================================

TEST_F(AssociationAttentionHeadTest, ApplyAttention) {
    auto pattern_ids = CreateTestPatterns(3);

    AddAssociation(pattern_ids[0], pattern_ids[1], 0.9f);
    AddAssociation(pattern_ids[0], pattern_ids[2], 0.3f);

    ContextVector context;

    auto result = association_head_->ApplyAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(result.size(), 2u);

    // Should be sorted by weight descending (strongest association first)
    EXPECT_GE(result[0].second, result[1].second);

    // Pattern 1 should be first (stronger association)
    EXPECT_EQ(result[0].first, pattern_ids[1]);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(AssociationAttentionHeadTest, EmptyCandidates) {
    auto pattern_ids = CreateTestPatterns(1);
    ContextVector context;

    auto weights = association_head_->ComputeAttention(pattern_ids[0], {}, context);

    EXPECT_TRUE(weights.empty());
}

TEST_F(AssociationAttentionHeadTest, SingleCandidate) {
    auto pattern_ids = CreateTestPatterns(2);

    AddAssociation(pattern_ids[0], pattern_ids[1], 0.7f);

    ContextVector context;

    auto weights = association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1]}, context);

    ASSERT_EQ(weights.size(), 1u);
    EXPECT_FLOAT_EQ(weights[pattern_ids[1]], 1.0f);
}

TEST_F(AssociationAttentionHeadTest, NoAssociationMatrix) {
    // Create head without association matrix
    AssociationAttentionConfig config;
    auto head = std::make_unique<AssociationAttentionHead>(config);

    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Should return uniform weights when no matrix is available
    auto weights = head->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Should be uniform
    EXPECT_NEAR(weights[pattern_ids[1]], 0.5f, 1e-5f);
    EXPECT_NEAR(weights[pattern_ids[2]], 0.5f, 1e-5f);
}

TEST_F(AssociationAttentionHeadTest, AllMissingAssociations) {
    auto pattern_ids = CreateTestPatterns(3);

    // Don't add any associations

    ContextVector context;

    auto weights = association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    ASSERT_EQ(weights.size(), 2u);

    // All should use default strength, so weights should be equal
    EXPECT_NEAR(weights[pattern_ids[1]], weights[pattern_ids[2]], 1e-5f);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(AssociationAttentionHeadTest, GetStatistics) {
    auto pattern_ids = CreateTestPatterns(3);

    AddAssociation(pattern_ids[0], pattern_ids[1], 0.8f);
    // pattern_ids[2] has no association

    ContextVector context;

    // Compute attention a few times
    association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);
    association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    auto stats = association_head_->GetStatistics();

    EXPECT_GE(stats["attention_computations"], 2.0f);
    EXPECT_GE(stats["association_lookups"], 0.0f);
    EXPECT_GE(stats["missing_associations"], 0.0f);
    EXPECT_GE(stats["cache_hits"], 0.0f);
    EXPECT_GE(stats["cache_misses"], 0.0f);
    EXPECT_GE(stats["cache_hit_rate"], 0.0f);
    EXPECT_LE(stats["cache_hit_rate"], 1.0f);
}

// ============================================================================
// Baseline Comparison Tests
// ============================================================================

TEST_F(AssociationAttentionHeadTest, BaselineForLearning) {
    auto pattern_ids = CreateTestPatterns(4);

    // Simulate learned associations (e.g., sequential pattern)
    AddAssociation(pattern_ids[0], pattern_ids[1], 0.9f);  // Strong learned link
    AddAssociation(pattern_ids[0], pattern_ids[2], 0.5f);  // Moderate learned link
    AddAssociation(pattern_ids[0], pattern_ids[3], 0.1f);  // Weak learned link

    ContextVector context;

    auto weights = association_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1], pattern_ids[2], pattern_ids[3]}, context);

    // Weights should directly reflect learned association strengths
    // (after softmax normalization)
    ASSERT_EQ(weights.size(), 3u);

    // Strongest association should have highest weight
    EXPECT_GT(weights[pattern_ids[1]], weights[pattern_ids[2]]);
    EXPECT_GT(weights[pattern_ids[2]], weights[pattern_ids[3]]);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
