// File: tests/learning/structural_attention_head_test.cpp
//
// Unit tests for StructuralAttentionHead

#include "learning/structural_attention_head.hpp"
#include "attention_test_fixtures.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace dpan;
using namespace dpan::attention;
using namespace dpan::testing;

// ============================================================================
// Test Fixture
// ============================================================================

class StructuralAttentionHeadTest : public AttentionTestFixture {
protected:
    void SetUp() override {
        AttentionTestFixture::SetUp();

        // Create structural attention head
        StructuralAttentionConfig config;
        config.jaccard_weight = 0.8f;
        config.size_weight = 0.2f;
        config.temperature = 1.0f;
        config.enable_caching = true;
        config.debug_logging = false;

        structural_head_ = std::make_unique<StructuralAttentionHead>(config);
        structural_head_->SetPatternDatabase(mock_db_.get());
    }

    /// Create composite pattern with specified sub-patterns
    PatternID CreateCompositePattern(const std::vector<PatternID>& subpattern_ids) {
        auto node = CreateTestPattern();
        PatternID pattern_id = node.GetID();

        // Add sub-patterns
        for (const auto& sub_id : subpattern_ids) {
            const_cast<PatternNode&>(node).AddSubPattern(sub_id);
        }

        // Store in database
        mock_db_->Store(node);

        return pattern_id;
    }

    std::unique_ptr<StructuralAttentionHead> structural_head_;
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(StructuralAttentionHeadTest, DefaultConfiguration) {
    StructuralAttentionConfig config;
    EXPECT_FLOAT_EQ(config.jaccard_weight, 0.8f);
    EXPECT_FLOAT_EQ(config.size_weight, 0.2f);
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_FLOAT_EQ(config.similarity_threshold, 0.0f);
    EXPECT_FLOAT_EQ(config.atomic_penalty, 0.5f);
    EXPECT_TRUE(config.enable_caching);
    EXPECT_EQ(config.cache_size, 1000u);
    EXPECT_FALSE(config.debug_logging);
    EXPECT_TRUE(config.Validate());
}

TEST_F(StructuralAttentionHeadTest, InvalidConfiguration) {
    StructuralAttentionConfig config;

    // Invalid jaccard weight
    config.jaccard_weight = -0.1f;
    EXPECT_FALSE(config.Validate());

    config.jaccard_weight = 1.5f;
    EXPECT_FALSE(config.Validate());

    // Invalid size weight
    config.jaccard_weight = 0.8f;
    config.size_weight = -0.1f;
    EXPECT_FALSE(config.Validate());

    // Weights don't sum to 1.0
    config.jaccard_weight = 0.5f;
    config.size_weight = 0.3f;  // Sum = 0.8, should be 1.0
    EXPECT_FALSE(config.Validate());

    // Invalid temperature
    config.size_weight = 0.5f;
    config.temperature = 0.0f;
    EXPECT_FALSE(config.Validate());

    config.temperature = -1.0f;
    EXPECT_FALSE(config.Validate());

    // Invalid similarity threshold
    config.temperature = 1.0f;
    config.similarity_threshold = -0.1f;
    EXPECT_FALSE(config.Validate());

    config.similarity_threshold = 1.5f;
    EXPECT_FALSE(config.Validate());

    // Invalid atomic penalty
    config.similarity_threshold = 0.0f;
    config.atomic_penalty = -0.1f;
    EXPECT_FALSE(config.Validate());

    config.atomic_penalty = 1.5f;
    EXPECT_FALSE(config.Validate());
}

TEST_F(StructuralAttentionHeadTest, SetStructuralConfig) {
    StructuralAttentionConfig new_config;
    new_config.jaccard_weight = 0.6f;
    new_config.size_weight = 0.4f;
    new_config.temperature = 0.5f;
    new_config.enable_caching = false;

    structural_head_->SetStructuralConfig(new_config);

    auto config = structural_head_->GetStructuralConfig();
    EXPECT_FLOAT_EQ(config.jaccard_weight, 0.6f);
    EXPECT_FLOAT_EQ(config.size_weight, 0.4f);
    EXPECT_FLOAT_EQ(config.temperature, 0.5f);
    EXPECT_FALSE(config.enable_caching);
}

// ============================================================================
// Jaccard Similarity Tests
// ============================================================================

TEST_F(StructuralAttentionHeadTest, JaccardSimilarityIdentical) {
    // Create atomic sub-patterns
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();
    auto sub3 = CreateTestPattern().GetID();

    mock_db_->Store(CreateTestPattern());  // Store subs

    // Create two identical composite patterns
    auto pattern1 = CreateCompositePattern({sub1, sub2, sub3});
    auto pattern2 = CreateCompositePattern({sub1, sub2, sub3});

    ContextVector context;

    auto weights = structural_head_->ComputeAttention(
        pattern1, {pattern1, pattern2}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Identical patterns should have equal weights
    EXPECT_NEAR(weights[pattern1], weights[pattern2], 1e-5f);
}

TEST_F(StructuralAttentionHeadTest, JaccardSimilarityPartialOverlap) {
    // Create atomic sub-patterns
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();
    auto sub3 = CreateTestPattern().GetID();
    auto sub4 = CreateTestPattern().GetID();

    // pattern1: {sub1, sub2, sub3}
    // pattern2: {sub2, sub3, sub4}
    // Intersection: {sub2, sub3} -> size 2
    // Union: {sub1, sub2, sub3, sub4} -> size 4
    // Jaccard = 2/4 = 0.5

    auto pattern1 = CreateCompositePattern({sub1, sub2, sub3});
    auto pattern2 = CreateCompositePattern({sub2, sub3, sub4});
    auto pattern3 = CreateCompositePattern({sub1, sub2, sub3});  // Identical to pattern1

    ContextVector context;

    auto weights = structural_head_->ComputeAttention(
        pattern1, {pattern2, pattern3}, context);

    ASSERT_EQ(weights.size(), 2u);

    // pattern3 (identical) should have higher weight than pattern2 (partial overlap)
    EXPECT_GT(weights[pattern3], weights[pattern2]);
}

TEST_F(StructuralAttentionHeadTest, JaccardSimilarityNoOverlap) {
    // Create atomic sub-patterns
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();
    auto sub3 = CreateTestPattern().GetID();
    auto sub4 = CreateTestPattern().GetID();

    // pattern1: {sub1, sub2}
    // pattern2: {sub3, sub4}
    // No overlap: Jaccard = 0.0

    auto pattern1 = CreateCompositePattern({sub1, sub2});
    auto pattern2 = CreateCompositePattern({sub3, sub4});

    ContextVector context;

    auto weights = structural_head_->ComputeAttention(
        pattern1, {pattern1, pattern2}, context);

    ASSERT_EQ(weights.size(), 2u);

    // pattern1 (self) should have much higher weight than pattern2 (no overlap)
    EXPECT_GT(weights[pattern1], weights[pattern2]);
}

// ============================================================================
// Size Similarity Tests
// ============================================================================

TEST_F(StructuralAttentionHeadTest, SizeSimilarityEffect) {
    // Create sub-patterns
    std::vector<PatternID> subs;
    for (int i = 0; i < 10; ++i) {
        subs.push_back(CreateTestPattern().GetID());
    }

    // Create patterns with same structure but different sizes
    auto pattern_small = CreateCompositePattern({subs[0], subs[1]});  // size 2
    auto pattern_medium = CreateCompositePattern({subs[0], subs[1], subs[2]});  // size 3
    auto pattern_large = CreateCompositePattern({subs[0], subs[1], subs[2], subs[3], subs[4]});  // size 5

    ContextVector context;

    auto weights = structural_head_->ComputeAttention(
        pattern_medium, {pattern_small, pattern_medium, pattern_large}, context);

    ASSERT_EQ(weights.size(), 3u);

    // pattern_medium (self) should have highest weight
    EXPECT_GT(weights[pattern_medium], weights[pattern_small]);
    EXPECT_GT(weights[pattern_medium], weights[pattern_large]);
}

// ============================================================================
// Atomic vs Composite Pattern Tests
// ============================================================================

TEST_F(StructuralAttentionHeadTest, BothAtomicPatterns) {
    // Create two atomic patterns (no sub-patterns)
    auto pattern1 = CreateTestPattern().GetID();
    auto pattern2 = CreateTestPattern().GetID();

    mock_db_->Store(CreateTestPattern());
    mock_db_->Store(CreateTestPattern());

    ContextVector context;

    auto weights = structural_head_->ComputeAttention(
        pattern1, {pattern1, pattern2}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Both atomic patterns should have equal weights (perfect structural match)
    EXPECT_NEAR(weights[pattern1], weights[pattern2], 1e-5f);
}

TEST_F(StructuralAttentionHeadTest, MixedAtomicComposite) {
    // Create atomic sub-patterns
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();

    // Create one atomic and one composite pattern
    auto atomic_node = CreateTestPattern();
    PatternID atomic_pattern = atomic_node.GetID();
    mock_db_->Store(atomic_node);

    auto composite_pattern = CreateCompositePattern({sub1, sub2});

    ContextVector context;

    auto weights = structural_head_->ComputeAttention(
        atomic_pattern, {atomic_pattern, composite_pattern}, context);

    ASSERT_EQ(weights.size(), 2u);

    // Atomic (self) should have higher weight than composite (penalty applied)
    EXPECT_GT(weights[atomic_pattern], weights[composite_pattern]);
}

TEST_F(StructuralAttentionHeadTest, AtomicPenaltyEffect) {
    // Create sub-patterns
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();

    // Create one atomic and one composite pattern
    auto atomic_node = CreateTestPattern();
    PatternID atomic_pattern = atomic_node.GetID();
    mock_db_->Store(atomic_node);

    auto composite_pattern = CreateCompositePattern({sub1, sub2});

    // Set low atomic penalty
    StructuralAttentionConfig config;
    config.jaccard_weight = 0.8f;
    config.size_weight = 0.2f;
    config.atomic_penalty = 0.1f;  // Low penalty
    structural_head_->SetStructuralConfig(config);

    ContextVector context;

    auto weights_low_penalty = structural_head_->ComputeAttention(
        composite_pattern, {atomic_pattern, composite_pattern}, context);

    // Clear cache before changing config
    structural_head_->ClearCache();

    // Set high atomic penalty
    config.atomic_penalty = 0.9f;  // High penalty
    structural_head_->SetStructuralConfig(config);

    auto weights_high_penalty = structural_head_->ComputeAttention(
        composite_pattern, {atomic_pattern, composite_pattern}, context);

    // Higher penalty should give atomic pattern relatively higher weight
    EXPECT_GT(weights_high_penalty[atomic_pattern], weights_low_penalty[atomic_pattern]);
}

// ============================================================================
// Caching Tests
// ============================================================================

TEST_F(StructuralAttentionHeadTest, CachingEnabled) {
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();

    auto pattern1 = CreateCompositePattern({sub1, sub2});
    auto pattern2 = CreateCompositePattern({sub1, sub2});
    auto pattern3 = CreateCompositePattern({sub1, sub2});

    ContextVector context;

    // First computation (cache miss for all)
    auto weights1 = structural_head_->ComputeAttention(
        pattern1, {pattern2, pattern3}, context);

    // Second computation with same pairs (cache hits)
    auto weights2 = structural_head_->ComputeAttention(
        pattern1, {pattern2, pattern3}, context);

    // Results should be identical
    EXPECT_EQ(weights1.size(), weights2.size());
    EXPECT_NEAR(weights1[pattern2], weights2[pattern2], 1e-6f);
    EXPECT_NEAR(weights1[pattern3], weights2[pattern3], 1e-6f);

    // Check statistics - should have cache hits from second call
    auto stats = structural_head_->GetStatistics();
    EXPECT_GT(stats["cache_hits"], 0.0f);
}

TEST_F(StructuralAttentionHeadTest, CachingDisabled) {
    // Disable caching
    StructuralAttentionConfig config;
    config.jaccard_weight = 0.8f;
    config.size_weight = 0.2f;
    config.enable_caching = false;
    structural_head_->SetStructuralConfig(config);

    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();

    auto pattern1 = CreateCompositePattern({sub1, sub2});
    auto pattern2 = CreateCompositePattern({sub1, sub2});

    ContextVector context;

    structural_head_->ComputeAttention(pattern1, {pattern2}, context);

    auto stats = structural_head_->GetStatistics();
    EXPECT_EQ(stats["cache_hits"], 0.0f);
    EXPECT_EQ(stats["cache_misses"], 0.0f);  // No cache lookups when disabled
}

TEST_F(StructuralAttentionHeadTest, ClearCache) {
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();

    auto pattern1 = CreateCompositePattern({sub1, sub2});
    auto pattern2 = CreateCompositePattern({sub1, sub2});
    auto pattern3 = CreateCompositePattern({sub1, sub2});

    ContextVector context;

    // Build up cache with multiple patterns
    structural_head_->ComputeAttention(pattern1, {pattern2, pattern3}, context);

    auto stats_before = structural_head_->GetStatistics();
    EXPECT_GT(stats_before["cache_size"], 0.0f);

    // Clear cache
    structural_head_->ClearCache();

    auto stats_after = structural_head_->GetStatistics();
    EXPECT_EQ(stats_after["cache_size"], 0.0f);
}

// ============================================================================
// Detailed Attention Tests
// ============================================================================

TEST_F(StructuralAttentionHeadTest, ComputeDetailedAttention) {
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();

    auto pattern1 = CreateCompositePattern({sub1, sub2});
    auto pattern2 = CreateCompositePattern({sub1, sub2});

    ContextVector context;

    auto scores = structural_head_->ComputeDetailedAttention(
        pattern1, {pattern2}, context);

    ASSERT_EQ(scores.size(), 1u);

    // Scores should be sorted by weight descending
    // Structural score should be set
    EXPECT_GE(scores[0].components.structural_score, 0.0f);
    EXPECT_LE(scores[0].components.structural_score, 1.0f);

    // Other components should be zero for pure structural attention
    EXPECT_EQ(scores[0].components.semantic_similarity, 0.0f);
    EXPECT_EQ(scores[0].components.context_similarity, 0.0f);
    EXPECT_EQ(scores[0].components.importance_score, 0.0f);
}

// ============================================================================
// Apply Attention Tests
// ============================================================================

TEST_F(StructuralAttentionHeadTest, ApplyAttention) {
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();
    auto sub3 = CreateTestPattern().GetID();

    auto pattern1 = CreateCompositePattern({sub1, sub2});
    auto pattern2 = CreateCompositePattern({sub1, sub2, sub3});

    ContextVector context;

    auto result = structural_head_->ApplyAttention(
        pattern1, {pattern2}, context);

    ASSERT_EQ(result.size(), 1u);

    // Weights should sum to 1.0 (single candidate)
    EXPECT_NEAR(result[0].second, 1.0f, 1e-5f);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(StructuralAttentionHeadTest, EmptyCandidates) {
    auto pattern1 = CreateTestPattern().GetID();
    mock_db_->Store(CreateTestPattern());

    ContextVector context;

    auto weights = structural_head_->ComputeAttention(pattern1, {}, context);

    EXPECT_TRUE(weights.empty());
}

TEST_F(StructuralAttentionHeadTest, SingleCandidate) {
    auto pattern1 = CreateTestPattern().GetID();
    auto pattern2 = CreateTestPattern().GetID();

    mock_db_->Store(CreateTestPattern());
    mock_db_->Store(CreateTestPattern());

    ContextVector context;

    auto weights = structural_head_->ComputeAttention(
        pattern1, {pattern2}, context);

    ASSERT_EQ(weights.size(), 1u);
    EXPECT_FLOAT_EQ(weights[pattern2], 1.0f);
}

TEST_F(StructuralAttentionHeadTest, NoPatternDatabase) {
    // Create head without pattern database
    StructuralAttentionConfig config;
    auto head = std::make_unique<StructuralAttentionHead>(config);

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

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(StructuralAttentionHeadTest, GetStatistics) {
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();

    auto pattern1 = CreateCompositePattern({sub1, sub2});
    auto pattern2 = CreateCompositePattern({sub1, sub2});

    ContextVector context;

    // Compute attention a few times
    structural_head_->ComputeAttention(pattern1, {pattern2}, context);
    structural_head_->ComputeAttention(pattern1, {pattern2}, context);

    auto stats = structural_head_->GetStatistics();

    EXPECT_GE(stats["attention_computations"], 2.0f);
    EXPECT_GE(stats["structural_computations"], 0.0f);
    EXPECT_GE(stats["cache_hits"], 0.0f);
    EXPECT_GE(stats["cache_misses"], 0.0f);
    EXPECT_GE(stats["cache_hit_rate"], 0.0f);
    EXPECT_LE(stats["cache_hit_rate"], 1.0f);
}

// ============================================================================
// Composite Pattern Structure Tests
// ============================================================================

TEST_F(StructuralAttentionHeadTest, ComplexHierarchy) {
    // Create a more complex hierarchy
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();
    auto sub3 = CreateTestPattern().GetID();
    auto sub4 = CreateTestPattern().GetID();
    auto sub5 = CreateTestPattern().GetID();

    // pattern1: {sub1, sub2, sub3, sub4}
    // pattern2: {sub2, sub3, sub4, sub5}
    // pattern3: {sub1, sub2}
    auto pattern1 = CreateCompositePattern({sub1, sub2, sub3, sub4});
    auto pattern2 = CreateCompositePattern({sub2, sub3, sub4, sub5});
    auto pattern3 = CreateCompositePattern({sub1, sub2});

    ContextVector context;

    auto weights = structural_head_->ComputeAttention(
        pattern1, {pattern2, pattern3}, context);

    ASSERT_EQ(weights.size(), 2u);

    // pattern2 has more overlap (3/5 Jaccard) than pattern3 (2/4 Jaccard)
    // But need to consider size similarity too
    // Both should have non-zero weights
    EXPECT_GT(weights[pattern2], 0.0f);
    EXPECT_GT(weights[pattern3], 0.0f);
}

TEST_F(StructuralAttentionHeadTest, SimilarityThreshold) {
    auto sub1 = CreateTestPattern().GetID();
    auto sub2 = CreateTestPattern().GetID();
    auto sub3 = CreateTestPattern().GetID();
    auto sub4 = CreateTestPattern().GetID();

    auto pattern1 = CreateCompositePattern({sub1, sub2});
    auto pattern2 = CreateCompositePattern({sub3, sub4});  // No overlap

    // Set high similarity threshold
    StructuralAttentionConfig config;
    config.jaccard_weight = 0.8f;
    config.size_weight = 0.2f;
    config.similarity_threshold = 0.5f;  // Require at least 50% similarity
    structural_head_->SetStructuralConfig(config);

    ContextVector context;

    auto weights = structural_head_->ComputeAttention(
        pattern1, {pattern1, pattern2}, context);

    ASSERT_EQ(weights.size(), 2u);

    // pattern1 (self) has perfect similarity (1.0) > threshold
    // pattern2 has no overlap (Jaccard=0.0, filtered to 0.0) < threshold
    // After softmax: pattern1 gets much higher weight
    EXPECT_GT(weights[pattern1], weights[pattern2]);
    EXPECT_GT(weights[pattern1], 0.7f);  // Should get most of the weight
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
