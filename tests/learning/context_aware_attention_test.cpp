// File: tests/learning/context_aware_attention_test.cpp
//
// Comprehensive tests for ContextAwareAttention
//
// Tests cover:
// - Context history storage and retrieval
// - Context similarity computation
// - Combined semantic + context attention
// - Context sensitivity (different contexts yield different results)
// - Circular buffer behavior
// - Thread safety
// - Configuration validation

#include "learning/context_aware_attention.hpp"
#include "attention_test_fixtures.hpp"
#include <gtest/gtest.h>

using namespace dpan;
using namespace dpan::attention;
using namespace dpan::testing;

class ContextAwareAttentionTest : public AttentionTestFixture {
protected:
    void SetUp() override {
        AttentionTestFixture::SetUp();

        // Create context-aware attention with default config
        AttentionConfig attn_config = CreateDefaultConfig();
        ContextAwareConfig ctx_config;
        attention_ = std::make_unique<ContextAwareAttention>(attn_config, ctx_config);
        attention_->SetPatternDatabase(mock_db_.get());
    }

    std::unique_ptr<ContextAwareAttention> attention_;
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(ContextAwareAttentionTest, ContextConfigValidate) {
    ContextAwareConfig config;
    config.max_context_history = 10;
    config.semantic_weight = 0.5f;
    config.context_weight = 0.5f;

    EXPECT_TRUE(config.Validate());
}

TEST_F(ContextAwareAttentionTest, ContextConfigValidateInvalidSum) {
    ContextAwareConfig config;
    config.semantic_weight = 0.7f;
    config.context_weight = 0.7f;

    EXPECT_FALSE(config.Validate());
}

TEST_F(ContextAwareAttentionTest, ContextConfigNormalize) {
    ContextAwareConfig config;
    config.semantic_weight = 1.0f;
    config.context_weight = 1.0f;

    config.Normalize();

    EXPECT_NEAR(config.semantic_weight, 0.5f, 1e-5f);
    EXPECT_NEAR(config.context_weight, 0.5f, 1e-5f);
}

// ============================================================================
// Context History Tests
// ============================================================================

TEST_F(ContextAwareAttentionTest, RecordActivation) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    ContextVector context;
    context.Set("dim1", 0.5f);
    context.Set("dim2", 0.8f);

    attention_->RecordActivation(pattern_id, context);

    auto history = attention_->GetContextHistory(pattern_id);

    ASSERT_EQ(history.size(), 1u);
    EXPECT_NEAR(history[0].Get("dim1"), 0.5f, 1e-5f);
    EXPECT_NEAR(history[0].Get("dim2"), 0.8f, 1e-5f);
}

TEST_F(ContextAwareAttentionTest, MultipleActivations) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    // Record 3 activations
    for (int i = 0; i < 3; ++i) {
        ContextVector context;
        context.Set("value", static_cast<float>(i));
        attention_->RecordActivation(pattern_id, context);
    }

    auto history = attention_->GetContextHistory(pattern_id);

    ASSERT_EQ(history.size(), 3u);

    // Most recent should be first
    EXPECT_NEAR(history[0].Get("value"), 2.0f, 1e-5f);
    EXPECT_NEAR(history[1].Get("value"), 1.0f, 1e-5f);
    EXPECT_NEAR(history[2].Get("value"), 0.0f, 1e-5f);
}

TEST_F(ContextAwareAttentionTest, CircularBufferBehavior) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    // Set small history size
    ContextAwareConfig config;
    config.max_context_history = 3;  // Only keep 3 contexts
    config.semantic_weight = 0.5f;
    config.context_weight = 0.5f;
    attention_->SetContextConfig(config);

    // Record 5 activations (should only keep last 3)
    for (int i = 0; i < 5; ++i) {
        ContextVector context;
        context.Set("value", static_cast<float>(i));
        attention_->RecordActivation(pattern_id, context);
    }

    auto history = attention_->GetContextHistory(pattern_id);

    // Should only have 3 (most recent)
    ASSERT_EQ(history.size(), 3u);

    // Should be [4, 3, 2] (oldest two dropped)
    EXPECT_NEAR(history[0].Get("value"), 4.0f, 1e-5f);
    EXPECT_NEAR(history[1].Get("value"), 3.0f, 1e-5f);
    EXPECT_NEAR(history[2].Get("value"), 2.0f, 1e-5f);
}

TEST_F(ContextAwareAttentionTest, GetHistoryNoActivations) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    auto history = attention_->GetContextHistory(pattern_id);

    EXPECT_TRUE(history.empty());
}

TEST_F(ContextAwareAttentionTest, ClearContextHistory) {
    auto pattern_ids = CreateTestPatterns(2);

    // Record activations for both patterns
    for (const auto& pattern_id : pattern_ids) {
        ContextVector context;
        context.Set("value", 1.0f);
        attention_->RecordActivation(pattern_id, context);
    }

    // Clear all history
    attention_->ClearContextHistory();

    // Both should have empty history
    EXPECT_TRUE(attention_->GetContextHistory(pattern_ids[0]).empty());
    EXPECT_TRUE(attention_->GetContextHistory(pattern_ids[1]).empty());
}

TEST_F(ContextAwareAttentionTest, ClearContextHistorySpecificPattern) {
    auto pattern_ids = CreateTestPatterns(2);

    // Record activations for both patterns
    for (const auto& pattern_id : pattern_ids) {
        ContextVector context;
        context.Set("value", 1.0f);
        attention_->RecordActivation(pattern_id, context);
    }

    // Clear only first pattern
    attention_->ClearContextHistory(pattern_ids[0]);

    // First should be empty, second should still have history
    EXPECT_TRUE(attention_->GetContextHistory(pattern_ids[0]).empty());
    EXPECT_FALSE(attention_->GetContextHistory(pattern_ids[1]).empty());
}

// ============================================================================
// Context Similarity Tests
// ============================================================================

TEST_F(ContextAwareAttentionTest, ContextSimilarityNoHistory) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    ContextVector query_context;
    query_context.Set("dim1", 0.5f);

    float similarity = attention_->ComputeContextSimilarity(query_context, pattern_id);

    // No history should return neutral score (0.5)
    EXPECT_NEAR(similarity, 0.5f, 1e-5f);
}

TEST_F(ContextAwareAttentionTest, ContextSimilarityIdentical) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    ContextVector context;
    context.Set("dim1", 0.8f);
    context.Set("dim2", 0.6f);

    // Record activation with specific context
    attention_->RecordActivation(pattern_id, context);

    // Query with identical context
    float similarity = attention_->ComputeContextSimilarity(context, pattern_id);

    // Identical contexts should have similarity close to 1.0
    EXPECT_GT(similarity, 0.95f);
}

TEST_F(ContextAwareAttentionTest, ContextSimilarityDifferent) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    // Record activation with one context
    ContextVector context1;
    context1.Set("dim1", 1.0f);
    context1.Set("dim2", 0.0f);
    attention_->RecordActivation(pattern_id, context1);

    // Query with orthogonal context
    ContextVector context2;
    context2.Set("dim1", 0.0f);
    context2.Set("dim2", 1.0f);

    float similarity = attention_->ComputeContextSimilarity(context2, pattern_id);

    // Orthogonal contexts should have similarity around 0.5
    // (cosine similarity 0.0 normalized to [0,1] is 0.5)
    EXPECT_NEAR(similarity, 0.5f, 0.1f);
}

TEST_F(ContextAwareAttentionTest, ContextSimilarityMaximum) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    // Record multiple activations with different contexts
    ContextVector context1;
    context1.Set("dim1", 1.0f);
    attention_->RecordActivation(pattern_id, context1);

    ContextVector context2;
    context2.Set("dim2", 1.0f);
    attention_->RecordActivation(pattern_id, context2);

    ContextVector context3;
    context3.Set("dim3", 1.0f);
    attention_->RecordActivation(pattern_id, context3);

    // Query with context matching context3
    ContextVector query_context;
    query_context.Set("dim3", 1.0f);

    float similarity = attention_->ComputeContextSimilarity(query_context, pattern_id);

    // Should return maximum similarity (matching context3)
    EXPECT_GT(similarity, 0.95f);
}

// ============================================================================
// Context-Aware Attention Tests
// ============================================================================

TEST_F(ContextAwareAttentionTest, ComputeAttentionBasic) {
    auto pattern_ids = CreateTestPatterns(3);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};

    ContextVector context = CreateSemanticContext();

    auto weights = attention_->ComputeAttention(query, candidates, context);

    // Verify weights are valid
    EXPECT_EQ(weights.size(), candidates.size());
    VerifyWeightsInRange(weights);
    VerifyWeightsSumToOne(weights);
}

TEST_F(ContextAwareAttentionTest, ContextSensitivity) {
    auto pattern_ids = CreateTestPatterns(3);

    PatternID query = pattern_ids[0];
    PatternID candidate1 = pattern_ids[1];
    PatternID candidate2 = pattern_ids[2];

    // Use 100% context weight to completely isolate context effects
    ContextAwareConfig config;
    config.semantic_weight = 0.0f;
    config.context_weight = 1.0f;
    attention_->SetContextConfig(config);

    // Record activations with different contexts (use multiple dimensions)
    ContextVector context_a;
    context_a.Set("environment", 1.0f);
    context_a.Set("other", 0.0f);
    attention_->RecordActivation(candidate1, context_a);

    ContextVector context_b;
    context_b.Set("environment", 0.0f);
    context_b.Set("other", 1.0f);
    attention_->RecordActivation(candidate2, context_b);

    // Query with context similar to candidate1
    ContextVector query_context_a;
    query_context_a.Set("environment", 0.9f);
    query_context_a.Set("other", 0.1f);

    auto weights_a = attention_->ComputeAttention(
        query,
        {candidate1, candidate2},
        query_context_a
    );

    // Candidate1 should have higher weight (context match)
    EXPECT_GT(weights_a[candidate1], weights_a[candidate2]);

    // Query with context similar to candidate2
    ContextVector query_context_b;
    query_context_b.Set("environment", 0.1f);
    query_context_b.Set("other", 0.9f);

    auto weights_b = attention_->ComputeAttention(
        query,
        {candidate1, candidate2},
        query_context_b
    );

    // Candidate2 should have higher weight now
    EXPECT_GT(weights_b[candidate2], weights_b[candidate1]);
}

TEST_F(ContextAwareAttentionTest, SemanticVsContextWeights) {
    auto pattern_ids = CreateTestPatterns(2);

    PatternID query = pattern_ids[0];
    PatternID candidate = pattern_ids[1];

    // Record activation with specific context
    ContextVector historical_context;
    historical_context.Set("factor", 1.0f);
    attention_->RecordActivation(candidate, historical_context);

    // Test with semantic-only weight
    ContextAwareConfig config_semantic;
    config_semantic.semantic_weight = 1.0f;
    config_semantic.context_weight = 0.0f;
    attention_->SetContextConfig(config_semantic);

    ContextVector query_context;
    query_context.Set("factor", 1.0f);

    float weight_semantic_only = attention_->ComputeAttention(
        query, {candidate}, query_context
    )[candidate];

    // Test with context-only weight
    ContextAwareConfig config_context;
    config_context.semantic_weight = 0.0f;
    config_context.context_weight = 1.0f;
    attention_->SetContextConfig(config_context);

    float weight_context_only = attention_->ComputeAttention(
        query, {candidate}, query_context
    )[candidate];

    // Both should be valid (single candidate gets weight 1.0)
    EXPECT_NEAR(weight_semantic_only, 1.0f, 1e-5f);
    EXPECT_NEAR(weight_context_only, 1.0f, 1e-5f);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(ContextAwareAttentionTest, GetStatistics) {
    auto pattern_ids = CreateTestPatterns(2);

    // Record some activations
    attention_->RecordActivation(pattern_ids[0], CreateSemanticContext());
    attention_->RecordActivation(pattern_ids[1], CreateTemporalContext());

    // Compute some attention
    attention_->ComputeAttention(
        pattern_ids[0],
        {pattern_ids[1]},
        CreateSemanticContext()
    );

    auto stats = attention_->GetStatistics();

    EXPECT_TRUE(stats.find("context_similarity_computations") != stats.end());
    EXPECT_TRUE(stats.find("context_activations_recorded") != stats.end());
    EXPECT_TRUE(stats.find("patterns_with_history") != stats.end());
    EXPECT_TRUE(stats.find("avg_history_size") != stats.end());

    EXPECT_GE(stats["context_activations_recorded"], 2.0f);
    EXPECT_GE(stats["patterns_with_history"], 2.0f);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(ContextAwareAttentionTest, EmptyCandidates) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates;

    auto weights = attention_->ComputeAttention(query, candidates, CreateEmptyContext());

    EXPECT_TRUE(weights.empty());
}

TEST_F(ContextAwareAttentionTest, SingleCandidate) {
    auto pattern_ids = CreateTestPatterns(2);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1]};

    auto weights = attention_->ComputeAttention(query, candidates, CreateEmptyContext());

    ASSERT_EQ(weights.size(), 1u);
    EXPECT_NEAR(weights[candidates[0]], 1.0f, 1e-5f);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
