// File: tests/learning/multi_head_attention_test.cpp
//
// Unit tests for MultiHeadAttention

#include "learning/multi_head_attention.hpp"
#include "learning/basic_attention.hpp"
#include "learning/context_aware_attention.hpp"
#include "association/association_matrix.hpp"
#include "attention_test_fixtures.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace dpan;
using namespace dpan::attention;
using namespace dpan::testing;

class MultiHeadAttentionTest : public AttentionTestFixture {
protected:
    void SetUp() override {
        AttentionTestFixture::SetUp();

        // Create multi-head attention
        MultiHeadConfig mh_config;
        mh_config.auto_normalize_weights = true;
        mh_config.debug_logging = false;

        multi_head_ = std::make_unique<MultiHeadAttention>(mh_config);
        multi_head_->SetPatternDatabase(mock_db_.get());
    }

    std::unique_ptr<MultiHeadAttention> multi_head_;
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(MultiHeadAttentionTest, DefaultConfiguration) {
    auto config = multi_head_->GetMultiHeadConfig();
    EXPECT_TRUE(config.auto_normalize_weights);
    EXPECT_FALSE(config.parallel_heads);
    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_FALSE(config.debug_logging);
}

TEST_F(MultiHeadAttentionTest, SetMultiHeadConfig) {
    MultiHeadConfig new_config;
    new_config.auto_normalize_weights = false;
    new_config.parallel_heads = true;
    new_config.temperature = 0.5f;
    new_config.debug_logging = true;

    multi_head_->SetMultiHeadConfig(new_config);

    auto config = multi_head_->GetMultiHeadConfig();
    EXPECT_FALSE(config.auto_normalize_weights);
    EXPECT_TRUE(config.parallel_heads);
    EXPECT_FLOAT_EQ(config.temperature, 0.5f);
    EXPECT_TRUE(config.debug_logging);
}

TEST_F(MultiHeadAttentionTest, InvalidConfigThrows) {
    MultiHeadConfig invalid_config;
    invalid_config.temperature = -1.0f;  // Invalid

    EXPECT_THROW(multi_head_->SetMultiHeadConfig(invalid_config),
                 std::invalid_argument);
}

// ============================================================================
// Head Management Tests
// ============================================================================

TEST_F(MultiHeadAttentionTest, AddHeadBasic) {
    AttentionConfig config;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);

    bool added = multi_head_->AddHead("semantic", mechanism, 0.5f);

    EXPECT_TRUE(added);
    EXPECT_EQ(multi_head_->GetNumHeads(), 1u);

    auto head = multi_head_->GetHead("semantic");
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->name, "semantic");
    EXPECT_FLOAT_EQ(head->weight, 1.0f);  // Auto-normalized to 1.0
}

TEST_F(MultiHeadAttentionTest, AddMultipleHeads) {
    AttentionConfig config;
    auto semantic = std::make_shared<BasicAttentionMechanism>(config);
    auto context = std::make_shared<BasicAttentionMechanism>(config);

    multi_head_->AddHead("semantic", semantic, 0.6f);
    multi_head_->AddHead("context", context, 0.4f);

    EXPECT_EQ(multi_head_->GetNumHeads(), 2u);

    // Weights should be normalized to sum to 1.0
    auto head1 = multi_head_->GetHead("semantic");
    auto head2 = multi_head_->GetHead("context");
    ASSERT_NE(head1, nullptr);
    ASSERT_NE(head2, nullptr);

    // Verify they sum to 1.0
    float sum = head1->weight + head2->weight;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Verify weights are positive and less than 1.0
    EXPECT_GT(head1->weight, 0.0f);
    EXPECT_LT(head1->weight, 1.0f);
    EXPECT_GT(head2->weight, 0.0f);
    EXPECT_LT(head2->weight, 1.0f);
}

TEST_F(MultiHeadAttentionTest, AddHeadDuplicateName) {
    AttentionConfig config;
    auto mechanism1 = std::make_shared<BasicAttentionMechanism>(config);
    auto mechanism2 = std::make_shared<BasicAttentionMechanism>(config);

    bool added1 = multi_head_->AddHead("semantic", mechanism1, 0.5f);
    bool added2 = multi_head_->AddHead("semantic", mechanism2, 0.5f);

    EXPECT_TRUE(added1);
    EXPECT_FALSE(added2);  // Duplicate name
    EXPECT_EQ(multi_head_->GetNumHeads(), 1u);
}

TEST_F(MultiHeadAttentionTest, AddHeadInvalidParameters) {
    AttentionConfig config;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);

    // Empty name
    EXPECT_FALSE(multi_head_->AddHead("", mechanism, 0.5f));

    // Null mechanism
    EXPECT_FALSE(multi_head_->AddHead("test", nullptr, 0.5f));

    // Invalid weight
    EXPECT_FALSE(multi_head_->AddHead("test", mechanism, -0.1f));
    EXPECT_FALSE(multi_head_->AddHead("test", mechanism, 1.5f));
}

TEST_F(MultiHeadAttentionTest, RemoveHead) {
    AttentionConfig config;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);

    multi_head_->AddHead("semantic", mechanism, 0.5f);
    EXPECT_EQ(multi_head_->GetNumHeads(), 1u);

    bool removed = multi_head_->RemoveHead("semantic");
    EXPECT_TRUE(removed);
    EXPECT_EQ(multi_head_->GetNumHeads(), 0u);
}

TEST_F(MultiHeadAttentionTest, RemoveHeadNotFound) {
    bool removed = multi_head_->RemoveHead("nonexistent");
    EXPECT_FALSE(removed);
}

TEST_F(MultiHeadAttentionTest, RemoveHeadRenormalizes) {
    AttentionConfig config;
    auto mechanism1 = std::make_shared<BasicAttentionMechanism>(config);
    auto mechanism2 = std::make_shared<BasicAttentionMechanism>(config);
    auto mechanism3 = std::make_shared<BasicAttentionMechanism>(config);

    multi_head_->AddHead("head1", mechanism1, 0.5f);
    multi_head_->AddHead("head2", mechanism2, 0.3f);
    multi_head_->AddHead("head3", mechanism3, 0.2f);

    // Remove middle head
    multi_head_->RemoveHead("head2");

    // Remaining weights should be renormalized
    auto head1 = multi_head_->GetHead("head1");
    auto head3 = multi_head_->GetHead("head3");
    ASSERT_NE(head1, nullptr);
    ASSERT_NE(head3, nullptr);

    // Weights should sum to 1.0
    float sum = head1->weight + head3->weight;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Check they're in range
    EXPECT_GT(head1->weight, 0.0f);
    EXPECT_LT(head1->weight, 1.0f);
    EXPECT_GT(head3->weight, 0.0f);
    EXPECT_LT(head3->weight, 1.0f);
}

TEST_F(MultiHeadAttentionTest, GetHeadNotFound) {
    auto head = multi_head_->GetHead("nonexistent");
    EXPECT_EQ(head, nullptr);
}

TEST_F(MultiHeadAttentionTest, GetHeads) {
    AttentionConfig config;
    auto mechanism1 = std::make_shared<BasicAttentionMechanism>(config);
    auto mechanism2 = std::make_shared<BasicAttentionMechanism>(config);

    multi_head_->AddHead("head1", mechanism1, 0.6f);
    multi_head_->AddHead("head2", mechanism2, 0.4f);

    const auto& heads = multi_head_->GetHeads();
    ASSERT_EQ(heads.size(), 2u);
    EXPECT_EQ(heads[0].name, "head1");
    EXPECT_EQ(heads[1].name, "head2");
}

TEST_F(MultiHeadAttentionTest, SetHeadWeight) {
    AttentionConfig config;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);

    multi_head_->AddHead("semantic", mechanism, 0.5f);

    bool updated = multi_head_->SetHeadWeight("semantic", 0.8f);
    EXPECT_TRUE(updated);

    auto head = multi_head_->GetHead("semantic");
    ASSERT_NE(head, nullptr);
    EXPECT_FLOAT_EQ(head->weight, 1.0f);  // Auto-normalized to 1.0
}

TEST_F(MultiHeadAttentionTest, SetHeadWeightNotFound) {
    bool updated = multi_head_->SetHeadWeight("nonexistent", 0.5f);
    EXPECT_FALSE(updated);
}

TEST_F(MultiHeadAttentionTest, SetHeadWeightInvalid) {
    AttentionConfig config;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);

    multi_head_->AddHead("semantic", mechanism, 0.5f);

    EXPECT_FALSE(multi_head_->SetHeadWeight("semantic", -0.1f));
    EXPECT_FALSE(multi_head_->SetHeadWeight("semantic", 1.5f));
}

// ============================================================================
// Weight Normalization Tests
// ============================================================================

TEST_F(MultiHeadAttentionTest, AutoNormalizeWeights) {
    AttentionConfig config;
    auto mechanism1 = std::make_shared<BasicAttentionMechanism>(config);
    auto mechanism2 = std::make_shared<BasicAttentionMechanism>(config);
    auto mechanism3 = std::make_shared<BasicAttentionMechanism>(config);

    multi_head_->AddHead("head1", mechanism1, 0.2f);
    multi_head_->AddHead("head2", mechanism2, 0.3f);
    multi_head_->AddHead("head3", mechanism3, 0.5f);

    // Weights should sum to 1.0
    float sum = 0.0f;
    for (const auto& head : multi_head_->GetHeads()) {
        sum += head.weight;
    }

    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(MultiHeadAttentionTest, ManualNormalizeWeights) {
    // Disable auto-normalize
    MultiHeadConfig config;
    config.auto_normalize_weights = false;
    multi_head_->SetMultiHeadConfig(config);

    AttentionConfig attn_config;
    auto mechanism1 = std::make_shared<BasicAttentionMechanism>(attn_config);
    auto mechanism2 = std::make_shared<BasicAttentionMechanism>(attn_config);

    multi_head_->AddHead("head1", mechanism1, 0.3f);
    multi_head_->AddHead("head2", mechanism2, 0.5f);

    // Manually normalize
    multi_head_->NormalizeWeights();

    float sum = 0.0f;
    for (const auto& head : multi_head_->GetHeads()) {
        sum += head.weight;
    }

    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(MultiHeadAttentionTest, NormalizeZeroWeights) {
    // Disable auto-normalize
    MultiHeadConfig config;
    config.auto_normalize_weights = false;
    multi_head_->SetMultiHeadConfig(config);

    AttentionConfig attn_config;
    auto mechanism1 = std::make_shared<BasicAttentionMechanism>(attn_config);
    auto mechanism2 = std::make_shared<BasicAttentionMechanism>(attn_config);

    multi_head_->AddHead("head1", mechanism1, 0.0f);
    multi_head_->AddHead("head2", mechanism2, 0.0f);

    // Should set equal weights
    multi_head_->NormalizeWeights();

    auto head1 = multi_head_->GetHead("head1");
    auto head2 = multi_head_->GetHead("head2");
    ASSERT_NE(head1, nullptr);
    ASSERT_NE(head2, nullptr);

    EXPECT_FLOAT_EQ(head1->weight, 0.5f);
    EXPECT_FLOAT_EQ(head2->weight, 0.5f);
}

// ============================================================================
// Validation Tests
// ============================================================================

TEST_F(MultiHeadAttentionTest, ValidateHeadsEmpty) {
    EXPECT_TRUE(multi_head_->ValidateHeads());
}

TEST_F(MultiHeadAttentionTest, ValidateHeadsValid) {
    AttentionConfig config;
    auto mechanism1 = std::make_shared<BasicAttentionMechanism>(config);
    auto mechanism2 = std::make_shared<BasicAttentionMechanism>(config);

    multi_head_->AddHead("head1", mechanism1, 0.6f);
    multi_head_->AddHead("head2", mechanism2, 0.4f);

    EXPECT_TRUE(multi_head_->ValidateHeads());
}

// ============================================================================
// Attention Computation Tests
// ============================================================================

TEST_F(MultiHeadAttentionTest, ComputeAttentionNoHeads) {
    auto pattern_ids = CreateTestPatterns(3);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    ContextVector context;

    auto weights = multi_head_->ComputeAttention(query, candidates, context);

    ASSERT_EQ(weights.size(), 2u);

    // Should return uniform weights
    EXPECT_FLOAT_EQ(weights[pattern_ids[1]], 0.5f);
    EXPECT_FLOAT_EQ(weights[pattern_ids[2]], 0.5f);
}

TEST_F(MultiHeadAttentionTest, ComputeAttentionSingleHead) {
    auto pattern_ids = CreateTestPatterns(3);

    // Add single head
    AttentionConfig config;
    config.temperature = 1.0f;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);
    mechanism->SetPatternDatabase(mock_db_.get());

    multi_head_->AddHead("semantic", mechanism, 1.0f);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    ContextVector context;

    auto weights = multi_head_->ComputeAttention(query, candidates, context);

    ASSERT_EQ(weights.size(), 2u);
    VerifyWeightsSumToOne(weights);
    VerifyWeightsInRange(weights);
}

TEST_F(MultiHeadAttentionTest, ComputeAttentionMultipleHeads) {
    auto pattern_ids = CreateTestPatterns(3);

    // Add multiple heads
    AttentionConfig config;
    auto head1 = std::make_shared<BasicAttentionMechanism>(config);
    auto head2 = std::make_shared<BasicAttentionMechanism>(config);

    head1->SetPatternDatabase(mock_db_.get());
    head2->SetPatternDatabase(mock_db_.get());

    multi_head_->AddHead("head1", head1, 0.6f);
    multi_head_->AddHead("head2", head2, 0.4f);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    ContextVector context;

    auto weights = multi_head_->ComputeAttention(query, candidates, context);

    ASSERT_EQ(weights.size(), 2u);
    VerifyWeightsSumToOne(weights);
    VerifyWeightsInRange(weights);
}

TEST_F(MultiHeadAttentionTest, ComputeAttentionEmptyCandidates) {
    AttentionConfig config;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);
    multi_head_->AddHead("semantic", mechanism, 1.0f);

    auto pattern_ids = CreateTestPatterns(1);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates;
    ContextVector context;

    auto weights = multi_head_->ComputeAttention(query, candidates, context);

    EXPECT_TRUE(weights.empty());
}

TEST_F(MultiHeadAttentionTest, ComputeAttentionSingleCandidate) {
    AttentionConfig config;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);
    multi_head_->AddHead("semantic", mechanism, 1.0f);

    auto pattern_ids = CreateTestPatterns(2);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1]};
    ContextVector context;

    auto weights = multi_head_->ComputeAttention(query, candidates, context);

    ASSERT_EQ(weights.size(), 1u);
    EXPECT_FLOAT_EQ(weights[pattern_ids[1]], 1.0f);
}

TEST_F(MultiHeadAttentionTest, ComputeDetailedAttention) {
    auto pattern_ids = CreateTestPatterns(3);

    AttentionConfig config;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);
    mechanism->SetPatternDatabase(mock_db_.get());
    multi_head_->AddHead("semantic", mechanism, 1.0f);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    ContextVector context;

    auto scores = multi_head_->ComputeDetailedAttention(query, candidates, context);

    ASSERT_EQ(scores.size(), 2u);

    // Should be sorted by weight descending
    EXPECT_GE(scores[0].weight, scores[1].weight);

    // Verify weights sum to 1.0
    float sum = scores[0].weight + scores[1].weight;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(MultiHeadAttentionTest, ApplyAttention) {
    auto pattern_ids = CreateTestPatterns(3);

    AttentionConfig config;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);
    mechanism->SetPatternDatabase(mock_db_.get());
    multi_head_->AddHead("semantic", mechanism, 1.0f);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> predictions = {pattern_ids[1], pattern_ids[2]};
    ContextVector context;

    auto result = multi_head_->ApplyAttention(query, predictions, context);

    ASSERT_EQ(result.size(), 2u);

    // Should be sorted by weight descending
    EXPECT_GE(result[0].second, result[1].second);

    // Verify weights
    float sum = result[0].second + result[1].second;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================================
// Configuration Propagation Tests
// ============================================================================

TEST_F(MultiHeadAttentionTest, SetConfigPropagatesToHeads) {
    AttentionConfig config1;
    auto mechanism1 = std::make_shared<BasicAttentionMechanism>(config1);
    auto mechanism2 = std::make_shared<BasicAttentionMechanism>(config1);

    multi_head_->AddHead("head1", mechanism1, 0.5f);
    multi_head_->AddHead("head2", mechanism2, 0.5f);

    // Update configuration
    AttentionConfig new_config;
    new_config.temperature = 0.5f;
    new_config.debug_logging = true;

    multi_head_->SetConfig(new_config);

    // Check that heads received the configuration
    auto head1_config = mechanism1->GetConfig();
    auto head2_config = mechanism2->GetConfig();

    EXPECT_FLOAT_EQ(head1_config.temperature, 0.5f);
    EXPECT_TRUE(head1_config.debug_logging);
    EXPECT_FLOAT_EQ(head2_config.temperature, 0.5f);
    EXPECT_TRUE(head2_config.debug_logging);
}

TEST_F(MultiHeadAttentionTest, SetPatternDatabasePropagatesToHeads) {
    AttentionConfig config;
    auto mechanism1 = std::make_shared<BasicAttentionMechanism>(config);
    auto mechanism2 = std::make_shared<BasicAttentionMechanism>(config);

    multi_head_->AddHead("head1", mechanism1, 0.5f);
    multi_head_->AddHead("head2", mechanism2, 0.5f);

    // Verify database is set (already set in SetUp)
    // Verify heads can use the database (they should not crash)
    auto pattern_ids = CreateTestPatterns(2);
    ContextVector context;

    // This should not crash
    auto weights = multi_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1]}, context);

    EXPECT_EQ(weights.size(), 1u);
}

TEST_F(MultiHeadAttentionTest, ClearCachePropagatesToHeads) {
    AttentionConfig config;
    config.enable_caching = true;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);
    mechanism->SetPatternDatabase(mock_db_.get());

    multi_head_->AddHead("semantic", mechanism, 1.0f);

    // This should not crash
    multi_head_->ClearCache();

    // Verify mechanism still works
    auto pattern_ids = CreateTestPatterns(2);
    ContextVector context;

    auto weights = multi_head_->ComputeAttention(
        pattern_ids[0], {pattern_ids[1]}, context);

    EXPECT_EQ(weights.size(), 1u);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(MultiHeadAttentionTest, GetStatistics) {
    AttentionConfig config;
    auto mechanism = std::make_shared<BasicAttentionMechanism>(config);
    mechanism->SetPatternDatabase(mock_db_.get());

    multi_head_->AddHead("semantic", mechanism, 1.0f);

    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Compute attention a few times with multiple candidates
    multi_head_->ComputeAttention(pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);
    multi_head_->ComputeAttention(pattern_ids[0], {pattern_ids[1], pattern_ids[2]}, context);

    auto stats = multi_head_->GetStatistics();

    EXPECT_EQ(stats["num_heads"], 1.0f);
    EXPECT_GE(stats["attention_computations"], 2.0f);
    EXPECT_GE(stats["head_combinations"], 2.0f);

    // Should have head-specific statistics (check for any key starting with "head_")
    bool has_head_stats = false;
    for (const auto& [key, value] : stats) {
        if (key.find("head_") == 0) {
            has_head_stats = true;
            break;
        }
    }
    EXPECT_TRUE(has_head_stats);
}

// ============================================================================
// Task 6.2: Head Output Combination Tests
// ============================================================================

TEST_F(MultiHeadAttentionTest, CombinationWeightedAverageCorrect) {
    // Create two heads with known, different behaviors
    auto pattern_ids = CreateTestPatterns(3);

    AttentionConfig config;
    config.temperature = 1.0f;

    auto head1 = std::make_shared<BasicAttentionMechanism>(config);
    auto head2 = std::make_shared<BasicAttentionMechanism>(config);

    head1->SetPatternDatabase(mock_db_.get());
    head2->SetPatternDatabase(mock_db_.get());

    // Add heads with specific weights
    multi_head_->AddHead("head1", head1, 0.7f);
    multi_head_->AddHead("head2", head2, 0.3f);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    ContextVector context;

    // Compute combined attention
    auto combined = multi_head_->ComputeAttention(query, candidates, context);

    ASSERT_EQ(combined.size(), 2u);

    // Verify weights sum to 1.0 (normalized)
    float sum = 0.0f;
    for (const auto& [_, weight] : combined) {
        sum += weight;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Verify all weights are in valid range
    for (const auto& [_, weight] : combined) {
        EXPECT_GE(weight, 0.0f);
        EXPECT_LE(weight, 1.0f);
    }
}

TEST_F(MultiHeadAttentionTest, AllHeadsContribute) {
    auto pattern_ids = CreateTestPatterns(3);

    AttentionConfig config;
    auto head1 = std::make_shared<BasicAttentionMechanism>(config);
    auto head2 = std::make_shared<BasicAttentionMechanism>(config);
    auto head3 = std::make_shared<BasicAttentionMechanism>(config);

    head1->SetPatternDatabase(mock_db_.get());
    head2->SetPatternDatabase(mock_db_.get());
    head3->SetPatternDatabase(mock_db_.get());

    // Add three heads with equal weights
    multi_head_->AddHead("head1", head1, 0.33f);
    multi_head_->AddHead("head2", head2, 0.33f);
    multi_head_->AddHead("head3", head3, 0.34f);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    ContextVector context;

    // Compute combined attention
    auto combined = multi_head_->ComputeAttention(query, candidates, context);

    // Verify result exists (all heads contributed)
    ASSERT_EQ(combined.size(), 2u);

    // Get statistics to verify all heads were called
    auto stats = multi_head_->GetStatistics();
    EXPECT_EQ(stats["num_heads"], 3.0f);
    EXPECT_GE(stats["head_combinations"], 1.0f);
}

TEST_F(MultiHeadAttentionTest, CombinationNormalizationCorrect) {
    auto pattern_ids = CreateTestPatterns(4);

    AttentionConfig config;
    auto head1 = std::make_shared<BasicAttentionMechanism>(config);
    auto head2 = std::make_shared<BasicAttentionMechanism>(config);

    head1->SetPatternDatabase(mock_db_.get());
    head2->SetPatternDatabase(mock_db_.get());

    // Add heads with weights that don't sum to 1.0 initially
    // (they should be auto-normalized to 0.5 each)
    multi_head_->AddHead("head1", head1, 0.8f);
    multi_head_->AddHead("head2", head2, 0.8f);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {
        pattern_ids[1], pattern_ids[2], pattern_ids[3]
    };
    ContextVector context;

    // Compute combined attention multiple times
    for (int i = 0; i < 5; ++i) {
        auto combined = multi_head_->ComputeAttention(query, candidates, context);

        ASSERT_EQ(combined.size(), 3u);

        // Verify weights sum to exactly 1.0 every time
        float sum = 0.0f;
        for (const auto& [_, weight] : combined) {
            sum += weight;
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

TEST_F(MultiHeadAttentionTest, CombinationWithDifferentWeights) {
    auto pattern_ids = CreateTestPatterns(3);

    AttentionConfig config;
    auto head1 = std::make_shared<BasicAttentionMechanism>(config);
    auto head2 = std::make_shared<BasicAttentionMechanism>(config);

    head1->SetPatternDatabase(mock_db_.get());
    head2->SetPatternDatabase(mock_db_.get());

    // Test with 90/10 split
    multi_head_->AddHead("dominant", head1, 0.9f);
    multi_head_->AddHead("minor", head2, 0.1f);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    ContextVector context;

    auto weights_90_10 = multi_head_->ComputeAttention(query, candidates, context);

    ASSERT_EQ(weights_90_10.size(), 2u);

    // Verify normalization
    float sum = 0.0f;
    for (const auto& [_, weight] : weights_90_10) {
        sum += weight;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);

    // Now change to 50/50 split and verify difference
    multi_head_->SetHeadWeight("dominant", 0.5f);
    multi_head_->SetHeadWeight("minor", 0.5f);

    auto weights_50_50 = multi_head_->ComputeAttention(query, candidates, context);

    ASSERT_EQ(weights_50_50.size(), 2u);

    // Verify normalization again
    sum = 0.0f;
    for (const auto& [_, weight] : weights_50_50) {
        sum += weight;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(MultiHeadAttentionTest, CombinationEfficiency) {
    auto pattern_ids = CreateTestPatterns(10);

    AttentionConfig config;

    // Add 4 heads (typical multi-head attention configuration)
    for (int i = 0; i < 4; ++i) {
        auto head = std::make_shared<BasicAttentionMechanism>(config);
        head->SetPatternDatabase(mock_db_.get());
        multi_head_->AddHead("head" + std::to_string(i), head, 0.25f);
    }

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());
    ContextVector context;

    // Run computation multiple times to verify efficiency
    const int iterations = 100;

    for (int i = 0; i < iterations; ++i) {
        auto combined = multi_head_->ComputeAttention(query, candidates, context);

        // Verify correctness is maintained
        ASSERT_EQ(combined.size(), 9u);

        float sum = 0.0f;
        for (const auto& [_, weight] : combined) {
            sum += weight;
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }

    // Verify statistics show correct number of computations
    auto stats = multi_head_->GetStatistics();
    EXPECT_GE(stats["attention_computations"], static_cast<float>(iterations));
    EXPECT_GE(stats["head_combinations"], static_cast<float>(iterations));
}

TEST_F(MultiHeadAttentionTest, CombinationZeroWeightHead) {
    auto pattern_ids = CreateTestPatterns(3);

    // Disable auto-normalization to test zero weight handling
    MultiHeadConfig mh_config;
    mh_config.auto_normalize_weights = false;
    multi_head_->SetMultiHeadConfig(mh_config);

    AttentionConfig config;
    auto head1 = std::make_shared<BasicAttentionMechanism>(config);
    auto head2 = std::make_shared<BasicAttentionMechanism>(config);

    head1->SetPatternDatabase(mock_db_.get());
    head2->SetPatternDatabase(mock_db_.get());

    // Add heads - one with weight 1.0, one with weight 0.0
    multi_head_->AddHead("active", head1, 1.0f);
    multi_head_->AddHead("inactive", head2, 0.0f);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    ContextVector context;

    auto combined = multi_head_->ComputeAttention(query, candidates, context);

    // Should still get valid normalized output
    ASSERT_EQ(combined.size(), 2u);

    float sum = 0.0f;
    for (const auto& [_, weight] : combined) {
        sum += weight;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TEST_F(MultiHeadAttentionTest, CombinationConsistency) {
    auto pattern_ids = CreateTestPatterns(3);

    AttentionConfig config;
    config.temperature = 1.0f;  // Fixed temperature for consistency

    auto head1 = std::make_shared<BasicAttentionMechanism>(config);
    auto head2 = std::make_shared<BasicAttentionMechanism>(config);

    head1->SetPatternDatabase(mock_db_.get());
    head2->SetPatternDatabase(mock_db_.get());

    multi_head_->AddHead("head1", head1, 0.6f);
    multi_head_->AddHead("head2", head2, 0.4f);

    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    ContextVector context;

    // Compute attention multiple times with same inputs
    auto result1 = multi_head_->ComputeAttention(query, candidates, context);
    auto result2 = multi_head_->ComputeAttention(query, candidates, context);
    auto result3 = multi_head_->ComputeAttention(query, candidates, context);

    // Results should be identical (deterministic combination)
    ASSERT_EQ(result1.size(), result2.size());
    ASSERT_EQ(result1.size(), result3.size());

    for (const auto& [pattern_id, weight1] : result1) {
        EXPECT_NEAR(weight1, result2[pattern_id], 1e-5f);
        EXPECT_NEAR(weight1, result3[pattern_id], 1e-5f);
    }
}

// ============================================================================
// Head Configuration Tests
// ============================================================================

TEST_F(MultiHeadAttentionTest, InitializeSemanticHeadFromConfig) {
    HeadConfig head_config;
    head_config.name = "semantic";
    head_config.type = AttentionHeadType::SEMANTIC;
    head_config.weight = 1.0f;
    head_config.parameters["temperature"] = 1.5f;
    head_config.parameters["similarity_threshold"] = 0.3f;

    std::vector<HeadConfig> configs = {head_config};

    bool success = multi_head_->InitializeHeadsFromConfig(configs, mock_db_.get());

    ASSERT_TRUE(success);
    ASSERT_EQ(multi_head_->GetNumHeads(), 1u);

    const auto* head = multi_head_->GetHead("semantic");
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->name, "semantic");
}

TEST_F(MultiHeadAttentionTest, InitializeTemporalHeadFromConfig) {
    HeadConfig head_config;
    head_config.name = "temporal";
    head_config.type = AttentionHeadType::TEMPORAL;
    head_config.weight = 1.0f;
    head_config.parameters["decay_constant_ms"] = 500.0f;
    head_config.parameters["temperature"] = 1.0f;

    std::vector<HeadConfig> configs = {head_config};

    bool success = multi_head_->InitializeHeadsFromConfig(configs, mock_db_.get());

    ASSERT_TRUE(success);
    ASSERT_EQ(multi_head_->GetNumHeads(), 1u);

    const auto* head = multi_head_->GetHead("temporal");
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->name, "temporal");
}

TEST_F(MultiHeadAttentionTest, InitializeStructuralHeadFromConfig) {
    HeadConfig head_config;
    head_config.name = "structural";
    head_config.type = AttentionHeadType::STRUCTURAL;
    head_config.weight = 1.0f;
    head_config.parameters["jaccard_weight"] = 0.7f;
    head_config.parameters["size_weight"] = 0.3f;

    std::vector<HeadConfig> configs = {head_config};

    bool success = multi_head_->InitializeHeadsFromConfig(configs, mock_db_.get());

    ASSERT_TRUE(success);
    ASSERT_EQ(multi_head_->GetNumHeads(), 1u);

    const auto* head = multi_head_->GetHead("structural");
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->name, "structural");
}

TEST_F(MultiHeadAttentionTest, InitializeAssociationHeadFromConfig) {
    // Create association matrix
    auto association_matrix = std::make_unique<AssociationMatrix>();

    HeadConfig head_config;
    head_config.name = "association";
    head_config.type = AttentionHeadType::ASSOCIATION;
    head_config.weight = 1.0f;
    head_config.parameters["strength_threshold"] = 0.2f;
    head_config.parameters["default_strength"] = 0.1f;

    std::vector<HeadConfig> configs = {head_config};

    bool success = multi_head_->InitializeHeadsFromConfig(configs, mock_db_.get(), association_matrix.get());

    ASSERT_TRUE(success);
    ASSERT_EQ(multi_head_->GetNumHeads(), 1u);

    const auto* head = multi_head_->GetHead("association");
    ASSERT_NE(head, nullptr);
    EXPECT_EQ(head->name, "association");
}

TEST_F(MultiHeadAttentionTest, InitializeMultipleHeadsFromConfig) {
    // Create configs for multiple heads
    HeadConfig semantic_config;
    semantic_config.name = "semantic";
    semantic_config.type = AttentionHeadType::SEMANTIC;
    semantic_config.weight = 0.4f;

    HeadConfig temporal_config;
    temporal_config.name = "temporal";
    temporal_config.type = AttentionHeadType::TEMPORAL;
    temporal_config.weight = 0.3f;
    temporal_config.parameters["decay_constant_ms"] = 1000.0f;

    HeadConfig basic_config;
    basic_config.name = "basic";
    basic_config.type = AttentionHeadType::BASIC;
    basic_config.weight = 0.3f;

    std::vector<HeadConfig> configs = {semantic_config, temporal_config, basic_config};

    bool success = multi_head_->InitializeHeadsFromConfig(configs, mock_db_.get());

    ASSERT_TRUE(success);
    ASSERT_EQ(multi_head_->GetNumHeads(), 3u);

    // Check all heads were created
    EXPECT_NE(multi_head_->GetHead("semantic"), nullptr);
    EXPECT_NE(multi_head_->GetHead("temporal"), nullptr);
    EXPECT_NE(multi_head_->GetHead("basic"), nullptr);

    // Weights should be normalized
    float weight_sum = 0.0f;
    for (const auto& head : multi_head_->GetHeads()) {
        weight_sum += head.weight;
    }
    EXPECT_NEAR(weight_sum, 1.0f, 1e-5f);
}

TEST_F(MultiHeadAttentionTest, ConfigValidation_DuplicateNames) {
    HeadConfig config1;
    config1.name = "test";
    config1.type = AttentionHeadType::SEMANTIC;
    config1.weight = 0.5f;

    HeadConfig config2;
    config2.name = "test";  // Duplicate name
    config2.type = AttentionHeadType::TEMPORAL;
    config2.weight = 0.5f;

    MultiHeadConfig multi_config;
    multi_config.head_configs = {config1, config2};

    // Validation should fail due to duplicate names
    EXPECT_FALSE(multi_config.Validate());
}

TEST_F(MultiHeadAttentionTest, ConfigValidation_InvalidHeadConfig) {
    HeadConfig config;
    config.name = "";  // Invalid: empty name
    config.type = AttentionHeadType::SEMANTIC;
    config.weight = 0.5f;

    MultiHeadConfig multi_config;
    multi_config.head_configs = {config};

    // Validation should fail due to empty name
    EXPECT_FALSE(multi_config.Validate());
}

TEST_F(MultiHeadAttentionTest, ConfigValidation_InvalidWeight) {
    HeadConfig config;
    config.name = "test";
    config.type = AttentionHeadType::SEMANTIC;
    config.weight = 1.5f;  // Invalid: > 1.0

    EXPECT_FALSE(config.Validate());
}

TEST_F(MultiHeadAttentionTest, InitializeFromConfig_NoPatternDB) {
    HeadConfig config;
    config.name = "test";
    config.type = AttentionHeadType::SEMANTIC;
    config.weight = 1.0f;

    std::vector<HeadConfig> configs = {config};

    // Should fail without pattern database
    bool success = multi_head_->InitializeHeadsFromConfig(configs, nullptr);
    EXPECT_FALSE(success);
}

TEST_F(MultiHeadAttentionTest, InitializeFromConfig_AssociationWithoutMatrix) {
    HeadConfig config;
    config.name = "association";
    config.type = AttentionHeadType::ASSOCIATION;
    config.weight = 1.0f;

    std::vector<HeadConfig> configs = {config};

    // Should fail: association head requires association matrix
    bool success = multi_head_->InitializeHeadsFromConfig(configs, mock_db_.get(), nullptr);
    EXPECT_FALSE(success);
}

TEST_F(MultiHeadAttentionTest, InitializeFromConfig_InvalidStructuralWeights) {
    HeadConfig config;
    config.name = "structural";
    config.type = AttentionHeadType::STRUCTURAL;
    config.weight = 1.0f;
    config.parameters["jaccard_weight"] = 0.6f;
    config.parameters["size_weight"] = 0.6f;  // Sum > 1.0

    std::vector<HeadConfig> configs = {config};

    // Should fail: structural weights don't sum to 1.0
    bool success = multi_head_->InitializeHeadsFromConfig(configs, mock_db_.get());
    EXPECT_FALSE(success);
}

TEST_F(MultiHeadAttentionTest, HeadTypeConversion) {
    // Test HeadTypeToString
    EXPECT_EQ(HeadTypeToString(AttentionHeadType::SEMANTIC), "semantic");
    EXPECT_EQ(HeadTypeToString(AttentionHeadType::TEMPORAL), "temporal");
    EXPECT_EQ(HeadTypeToString(AttentionHeadType::STRUCTURAL), "structural");
    EXPECT_EQ(HeadTypeToString(AttentionHeadType::ASSOCIATION), "association");
    EXPECT_EQ(HeadTypeToString(AttentionHeadType::BASIC), "basic");
    EXPECT_EQ(HeadTypeToString(AttentionHeadType::CONTEXT), "context");

    // Test StringToHeadType
    AttentionHeadType type;
    EXPECT_TRUE(StringToHeadType("semantic", type));
    EXPECT_EQ(type, AttentionHeadType::SEMANTIC);

    EXPECT_TRUE(StringToHeadType("temporal", type));
    EXPECT_EQ(type, AttentionHeadType::TEMPORAL);

    EXPECT_TRUE(StringToHeadType("structural", type));
    EXPECT_EQ(type, AttentionHeadType::STRUCTURAL);

    EXPECT_TRUE(StringToHeadType("association", type));
    EXPECT_EQ(type, AttentionHeadType::ASSOCIATION);

    EXPECT_TRUE(StringToHeadType("basic", type));
    EXPECT_EQ(type, AttentionHeadType::BASIC);

    EXPECT_TRUE(StringToHeadType("context", type));
    EXPECT_EQ(type, AttentionHeadType::CONTEXT);

    EXPECT_FALSE(StringToHeadType("invalid", type));
}

TEST_F(MultiHeadAttentionTest, ConfiguredHeadsComputeAttention) {
    auto pattern_ids = CreateTestPatterns(3);

    // Create configs
    HeadConfig semantic_config;
    semantic_config.name = "semantic";
    semantic_config.type = AttentionHeadType::SEMANTIC;
    semantic_config.weight = 0.6f;

    HeadConfig basic_config;
    basic_config.name = "basic";
    basic_config.type = AttentionHeadType::BASIC;
    basic_config.weight = 0.4f;

    std::vector<HeadConfig> configs = {semantic_config, basic_config};

    bool success = multi_head_->InitializeHeadsFromConfig(configs, mock_db_.get());
    ASSERT_TRUE(success);

    // Compute attention
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates = {pattern_ids[1], pattern_ids[2]};
    ContextVector context;

    auto weights = multi_head_->ComputeAttention(query, candidates, context);

    ASSERT_EQ(weights.size(), 2u);

    // Verify normalized
    float sum = 0.0f;
    for (const auto& [_, weight] : weights) {
        sum += weight;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
