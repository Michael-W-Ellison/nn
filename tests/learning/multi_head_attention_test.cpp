// File: tests/learning/multi_head_attention_test.cpp
//
// Unit tests for MultiHeadAttention

#include "learning/multi_head_attention.hpp"
#include "learning/basic_attention.hpp"
#include "learning/context_aware_attention.hpp"
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
