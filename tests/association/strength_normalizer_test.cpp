// File: tests/association/strength_normalizer_test.cpp
#include "association/strength_normalizer.hpp"
#include <gtest/gtest.h>

using namespace dpan;
using namespace dpan::StrengthNormalizer;

// ============================================================================
// Basic Normalization Tests
// ============================================================================

TEST(StrengthNormalizerTest, NormalizeOutgoingBasic) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // Create outgoing associations with total strength != 1.0
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.3f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.6f));

    // Sum = 0.9, should normalize to 1.0
    bool normalized = NormalizeOutgoing(matrix, p1);
    EXPECT_TRUE(normalized);

    // Check that sum is now 1.0
    float sum = GetOutgoingStrengthSum(matrix, p1);
    EXPECT_NEAR(1.0f, sum, 0.001f);

    // Check relative strengths preserved
    auto edge1 = matrix.GetAssociation(p1, p2);
    auto edge2 = matrix.GetAssociation(p1, p3);

    ASSERT_NE(nullptr, edge1);
    ASSERT_NE(nullptr, edge2);

    // Original ratio was 0.3:0.6 = 1:2
    // Should still be approximately 1:2
    EXPECT_NEAR(edge1->GetStrength() * 2.0f, edge2->GetStrength(), 0.01f);
}

TEST(StrengthNormalizerTest, NormalizeOutgoingPreservesRatios) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    // Create associations with specific ratios
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.1f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.2f));
    matrix.AddAssociation(AssociationEdge(p1, p4, AssociationType::CAUSAL, 0.3f));

    NormalizeOutgoing(matrix, p1);

    auto edge1 = matrix.GetAssociation(p1, p2);
    auto edge2 = matrix.GetAssociation(p1, p3);
    auto edge3 = matrix.GetAssociation(p1, p4);

    // Ratios should be 1:2:3
    EXPECT_NEAR(edge1->GetStrength() * 2.0f, edge2->GetStrength(), 0.01f);
    EXPECT_NEAR(edge1->GetStrength() * 3.0f, edge3->GetStrength(), 0.01f);
}

TEST(StrengthNormalizerTest, NormalizeIncomingBasic) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // Create incoming associations to p3
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.4f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.4f));

    // Sum = 0.8, should normalize to 1.0
    bool normalized = NormalizeIncoming(matrix, p3);
    EXPECT_TRUE(normalized);

    // Check that sum is now 1.0
    float sum = GetIncomingStrengthSum(matrix, p3);
    EXPECT_NEAR(1.0f, sum, 0.001f);
}

TEST(StrengthNormalizerTest, NormalizeBidirectional) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // p2 has both incoming and outgoing
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));  // Incoming to p2
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.6f));  // Outgoing from p2

    auto [outgoing_norm, incoming_norm] = NormalizeBidirectional(matrix, p2);

    EXPECT_TRUE(outgoing_norm);
    EXPECT_TRUE(incoming_norm);

    EXPECT_NEAR(1.0f, GetOutgoingStrengthSum(matrix, p2), 0.001f);
    EXPECT_NEAR(1.0f, GetIncomingStrengthSum(matrix, p2), 0.001f);
}

// ============================================================================
// Batch Normalization Tests
// ============================================================================

TEST(StrengthNormalizerTest, NormalizeOutgoingBatch) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    // p1 -> {p2, p3}
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.3f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.3f));

    // p2 -> {p4}
    matrix.AddAssociation(AssociationEdge(p2, p4, AssociationType::CAUSAL, 0.5f));

    std::vector<PatternID> patterns = {p1, p2};
    size_t normalized = NormalizeOutgoingBatch(matrix, patterns);

    EXPECT_EQ(2u, normalized);

    EXPECT_NEAR(1.0f, GetOutgoingStrengthSum(matrix, p1), 0.001f);
    EXPECT_NEAR(1.0f, GetOutgoingStrengthSum(matrix, p2), 0.001f);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST(StrengthNormalizerTest, GetOutgoingStrengthSum) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.25f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.35f));

    float sum = GetOutgoingStrengthSum(matrix, p1);
    EXPECT_NEAR(0.6f, sum, 0.001f);
}

TEST(StrengthNormalizerTest, GetIncomingStrengthSum) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.2f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.3f));

    float sum = GetIncomingStrengthSum(matrix, p3);
    EXPECT_NEAR(0.5f, sum, 0.001f);
}

TEST(StrengthNormalizerTest, IsNormalized) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.5f));

    EXPECT_TRUE(IsNormalized(matrix, p1, 0.01f));

    // Add another edge, should no longer be normalized
    PatternID p4 = PatternID::Generate();
    matrix.AddAssociation(AssociationEdge(p1, p4, AssociationType::CAUSAL, 0.3f));

    EXPECT_FALSE(IsNormalized(matrix, p1, 0.01f));
}

TEST(StrengthNormalizerTest, GetNormalizationFactor) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.4f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.6f));

    // Sum = 1.0, factor should be 1.0
    float factor = GetNormalizationFactor(matrix, p1);
    EXPECT_NEAR(1.0f, factor, 0.001f);

    // Add another edge
    PatternID p4 = PatternID::Generate();
    matrix.AddAssociation(AssociationEdge(p1, p4, AssociationType::CAUSAL, 0.5f));

    // Sum = 1.5, factor should be 1/1.5 = 0.6667
    factor = GetNormalizationFactor(matrix, p1);
    EXPECT_NEAR(0.6667f, factor, 0.001f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(StrengthNormalizerTest, NormalizeEmptyPattern) {
    AssociationMatrix matrix;
    PatternID p1 = PatternID::Generate();

    // Pattern with no associations
    bool normalized = NormalizeOutgoing(matrix, p1);
    EXPECT_FALSE(normalized);  // Nothing to normalize
}

TEST(StrengthNormalizerTest, NormalizeSingleEdge) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.7f));

    bool normalized = NormalizeOutgoing(matrix, p1);
    EXPECT_TRUE(normalized);

    // Single edge should now have strength 1.0
    auto edge = matrix.GetAssociation(p1, p2);
    ASSERT_NE(nullptr, edge);
    EXPECT_NEAR(1.0f, edge->GetStrength(), 0.001f);
}

TEST(StrengthNormalizerTest, NormalizeAlreadyNormalized) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // Already sums to 1.0
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.4f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.6f));

    bool normalized = NormalizeOutgoing(matrix, p1);
    EXPECT_FALSE(normalized);  // Already normalized, no change needed

    // Strengths should remain unchanged
    auto edge1 = matrix.GetAssociation(p1, p2);
    auto edge2 = matrix.GetAssociation(p1, p3);

    EXPECT_NEAR(0.4f, edge1->GetStrength(), 0.001f);
    EXPECT_NEAR(0.6f, edge2->GetStrength(), 0.001f);
}

TEST(StrengthNormalizerTest, NormalizeWithZeroStrengths) {
    AssociationMatrix matrix;
    Config config;
    config.preserve_zeros = true;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.3f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.0f));  // Zero
    matrix.AddAssociation(AssociationEdge(p1, p4, AssociationType::CAUSAL, 0.3f));

    bool normalized = NormalizeOutgoing(matrix, p1, config);
    EXPECT_TRUE(normalized);

    // Zero should stay zero, others normalized
    auto edge_zero = matrix.GetAssociation(p1, p3);
    EXPECT_NEAR(0.0f, edge_zero->GetStrength(), 0.001f);

    // Non-zero edges should sum to 1.0
    float sum = GetOutgoingStrengthSum(matrix, p1);
    EXPECT_NEAR(1.0f, sum, 0.001f);
}

TEST(StrengthNormalizerTest, NormalizeWithMinThreshold) {
    AssociationMatrix matrix;
    Config config;
    config.min_strength_threshold = 0.1f;  // Ignore edges below 0.1

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.3f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.05f));  // Below threshold
    matrix.AddAssociation(AssociationEdge(p1, p4, AssociationType::CAUSAL, 0.3f));

    bool normalized = NormalizeOutgoing(matrix, p1, config);
    EXPECT_TRUE(normalized);

    // Only edges above threshold should be normalized (0.3 + 0.3 = 0.6 -> normalized)
    // Edge below threshold should remain unchanged
    auto edge_below = matrix.GetAssociation(p1, p3);
    EXPECT_NEAR(0.05f, edge_below->GetStrength(), 0.001f);
}

TEST(StrengthNormalizerTest, NormalizeVerySmallStrengths) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // Very small strengths
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.001f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.002f));

    // Use config with very low threshold to include these small strengths
    Config config;
    config.min_strength_threshold = 0.0001f;

    bool normalized = NormalizeOutgoing(matrix, p1, config);
    EXPECT_TRUE(normalized);

    // Should still sum to 1.0
    float sum = GetOutgoingStrengthSum(matrix, p1);
    EXPECT_NEAR(1.0f, sum, 0.001f);

    // Ratios preserved: 1:2
    auto edge1 = matrix.GetAssociation(p1, p2);
    auto edge2 = matrix.GetAssociation(p1, p3);

    EXPECT_NEAR(edge1->GetStrength() * 2.0f, edge2->GetStrength(), 0.01f);
}

TEST(StrengthNormalizerTest, NormalizeManyEdges) {
    AssociationMatrix matrix;

    PatternID source = PatternID::Generate();
    std::vector<PatternID> targets;

    // Create 100 outgoing edges with equal strength
    for (int i = 0; i < 100; ++i) {
        PatternID target = PatternID::Generate();
        targets.push_back(target);
        matrix.AddAssociation(AssociationEdge(source, target, AssociationType::CAUSAL, 0.02f));
    }

    // Sum = 2.0, should normalize to 1.0
    bool normalized = NormalizeOutgoing(matrix, source);
    EXPECT_TRUE(normalized);

    float sum = GetOutgoingStrengthSum(matrix, source);
    EXPECT_NEAR(1.0f, sum, 0.01f);  // Slightly larger tolerance for many edges

    // Each edge should be 0.01 (1.0 / 100)
    for (const auto& target : targets) {
        auto edge = matrix.GetAssociation(source, target);
        EXPECT_NEAR(0.01f, edge->GetStrength(), 0.001f);
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST(StrengthNormalizerTest, ConfigDefaultValues) {
    Config config;

    EXPECT_FLOAT_EQ(0.01f, config.min_strength_threshold);
    EXPECT_FALSE(config.preserve_zeros);
    EXPECT_EQ(NormalizationMode::OUTGOING, config.mode);
}

TEST(StrengthNormalizerTest, NormalizationModeOutgoing) {
    Config config;
    config.mode = NormalizationMode::OUTGOING;

    EXPECT_EQ(NormalizationMode::OUTGOING, config.mode);
}

TEST(StrengthNormalizerTest, NormalizationModeIncoming) {
    Config config;
    config.mode = NormalizationMode::INCOMING;

    EXPECT_EQ(NormalizationMode::INCOMING, config.mode);
}

TEST(StrengthNormalizerTest, NormalizationModeBidirectional) {
    Config config;
    config.mode = NormalizationMode::BIDIRECTIONAL;

    EXPECT_EQ(NormalizationMode::BIDIRECTIONAL, config.mode);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(StrengthNormalizerTest, AnalyzeNormalizationEmptyMatrix) {
    AssociationMatrix matrix;

    auto stats = AnalyzeNormalization(matrix);

    EXPECT_EQ(0u, stats.patterns_processed);
    EXPECT_EQ(0u, stats.patterns_normalized);
    EXPECT_EQ(0u, stats.edges_updated);
}
