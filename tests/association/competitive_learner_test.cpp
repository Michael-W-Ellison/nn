// File: tests/association/competitive_learner_test.cpp
#include "association/competitive_learner.hpp"
#include "association/association_matrix.hpp"
#include "association/association_edge.hpp"
#include "core/pattern_node.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace dpan {
namespace {

// Helper function to create a test pattern
PatternID CreateTestPattern() {
    return PatternID::Generate();
}

// Helper function to create test association edge
AssociationEdge CreateTestEdge(PatternID source, PatternID target, float strength) {
    AssociationEdge edge(source, target, AssociationType::CATEGORICAL, strength);
    return edge;
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(CompetitiveLearnerTest, FindStrongestReturnsMaxStrength) {
    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();
    PatternID p4 = CreateTestPattern();

    auto e1 = std::make_unique<AssociationEdge>(CreateTestEdge(p1, p2, 0.3f));
    auto e2 = std::make_unique<AssociationEdge>(CreateTestEdge(p1, p3, 0.7f));  // Strongest
    auto e3 = std::make_unique<AssociationEdge>(CreateTestEdge(p1, p4, 0.5f));

    std::vector<const AssociationEdge*> edges = {e1.get(), e2.get(), e3.get()};

    const AssociationEdge* strongest = CompetitiveLearner::FindStrongest(edges);

    ASSERT_NE(nullptr, strongest);
    EXPECT_FLOAT_EQ(0.7f, strongest->GetStrength());
    EXPECT_EQ(p3, strongest->GetTarget());
}

TEST(CompetitiveLearnerTest, FindStrongestEmptyVector) {
    std::vector<const AssociationEdge*> edges;

    const AssociationEdge* strongest = CompetitiveLearner::FindStrongest(edges);

    EXPECT_EQ(nullptr, strongest);
}

TEST(CompetitiveLearnerTest, CalculateWinnerStrengthBoostsCorrectly) {
    float current = 0.6f;
    float beta = 0.3f;

    float new_strength = CompetitiveLearner::CalculateWinnerStrength(current, beta);

    // Formula: s_new = s_old + β × (1 - s_old)
    // Expected: 0.6 + 0.3 × (1 - 0.6) = 0.6 + 0.3 × 0.4 = 0.6 + 0.12 = 0.72
    EXPECT_FLOAT_EQ(0.72f, new_strength);
}

TEST(CompetitiveLearnerTest, CalculateWinnerStrengthBounded) {
    float current = 0.95f;
    float beta = 0.5f;

    float new_strength = CompetitiveLearner::CalculateWinnerStrength(current, beta);

    // Should not exceed 1.0
    EXPECT_LE(new_strength, 1.0f);
    EXPECT_GE(new_strength, 0.0f);
}

TEST(CompetitiveLearnerTest, CalculateLoserStrengthSuppressesCorrectly) {
    float current = 0.6f;
    float beta = 0.3f;

    float new_strength = CompetitiveLearner::CalculateLoserStrength(current, beta);

    // Formula: s_new = s_old × (1 - β)
    // Expected: 0.6 × (1 - 0.3) = 0.6 × 0.7 = 0.42
    EXPECT_FLOAT_EQ(0.42f, new_strength);
}

TEST(CompetitiveLearnerTest, CalculateLoserStrengthBounded) {
    float current = 0.05f;
    float beta = 0.9f;

    float new_strength = CompetitiveLearner::CalculateLoserStrength(current, beta);

    // Should not go below 0.0
    EXPECT_GE(new_strength, 0.0f);
    EXPECT_LE(new_strength, 1.0f);
}

// ============================================================================
// Filter Function Tests
// ============================================================================

TEST(CompetitiveLearnerTest, FilterByTypeCorrect) {
    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();
    PatternID p4 = CreateTestPattern();

    auto e1 = std::make_unique<AssociationEdge>(p1, p2, AssociationType::CAUSAL, 0.5f);
    auto e2 = std::make_unique<AssociationEdge>(p1, p3, AssociationType::SPATIAL, 0.6f);
    auto e3 = std::make_unique<AssociationEdge>(p1, p4, AssociationType::CAUSAL, 0.7f);

    std::vector<const AssociationEdge*> edges = {e1.get(), e2.get(), e3.get()};

    auto causal = CompetitiveLearner::FilterByType(edges, AssociationType::CAUSAL);

    EXPECT_EQ(2u, causal.size());

    for (const auto* edge : causal) {
        EXPECT_EQ(AssociationType::CAUSAL, edge->GetType());
    }
}

TEST(CompetitiveLearnerTest, FilterByStrengthCorrect) {
    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();
    PatternID p4 = CreateTestPattern();

    auto e1 = std::make_unique<AssociationEdge>(CreateTestEdge(p1, p2, 0.05f));
    auto e2 = std::make_unique<AssociationEdge>(CreateTestEdge(p1, p3, 0.15f));
    auto e3 = std::make_unique<AssociationEdge>(CreateTestEdge(p1, p4, 0.25f));

    std::vector<const AssociationEdge*> edges = {e1.get(), e2.get(), e3.get()};

    auto filtered = CompetitiveLearner::FilterByStrength(edges, 0.1f);

    EXPECT_EQ(2u, filtered.size());

    for (const auto* edge : filtered) {
        EXPECT_GE(edge->GetStrength(), 0.1f);
    }
}

// ============================================================================
// Competition Application Tests
// ============================================================================

TEST(CompetitiveLearnerTest, ApplyCompetitionBoostsWinnerSuppressesLosers) {
    AssociationMatrix matrix;

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();
    PatternID p4 = CreateTestPattern();

    // Create associations with different strengths
    matrix.AddAssociation(CreateTestEdge(p1, p2, 0.3f));
    matrix.AddAssociation(CreateTestEdge(p1, p3, 0.7f));  // Winner
    matrix.AddAssociation(CreateTestEdge(p1, p4, 0.5f));

    CompetitiveLearner::Config config;
    config.competition_factor = 0.3f;

    bool applied = CompetitiveLearner::ApplyCompetition(matrix, p1, config);

    EXPECT_TRUE(applied);

    // Check winner (p3) was boosted
    const auto* edge_to_p3 = matrix.GetAssociation(p1, p3);
    ASSERT_NE(nullptr, edge_to_p3);
    float expected_winner = 0.7f + 0.3f * (1.0f - 0.7f);  // 0.7 + 0.09 = 0.79
    EXPECT_NEAR(expected_winner, edge_to_p3->GetStrength(), 0.01f);

    // Check losers (p2 and p4) were suppressed
    const auto* edge_to_p2 = matrix.GetAssociation(p1, p2);
    ASSERT_NE(nullptr, edge_to_p2);
    float expected_loser1 = 0.3f * (1.0f - 0.3f);  // 0.3 * 0.7 = 0.21
    EXPECT_NEAR(expected_loser1, edge_to_p2->GetStrength(), 0.01f);

    const auto* edge_to_p4 = matrix.GetAssociation(p1, p4);
    ASSERT_NE(nullptr, edge_to_p4);
    float expected_loser2 = 0.5f * (1.0f - 0.3f);  // 0.5 * 0.7 = 0.35
    EXPECT_NEAR(expected_loser2, edge_to_p4->GetStrength(), 0.01f);
}

TEST(CompetitiveLearnerTest, ApplyCompetitionRequiresMinimumAssociations) {
    AssociationMatrix matrix;

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();

    // Only one association - not enough to compete
    matrix.AddAssociation(CreateTestEdge(p1, p2, 0.5f));

    CompetitiveLearner::Config config;
    config.min_competing_associations = 2;

    bool applied = CompetitiveLearner::ApplyCompetition(matrix, p1, config);

    EXPECT_FALSE(applied);  // Not enough associations
}

TEST(CompetitiveLearnerTest, ApplyCompetitionRespectsMinStrengthThreshold) {
    AssociationMatrix matrix;

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();

    // Create associations, one below threshold
    matrix.AddAssociation(CreateTestEdge(p1, p2, 0.005f));  // Below threshold
    matrix.AddAssociation(CreateTestEdge(p1, p3, 0.7f));

    CompetitiveLearner::Config config;
    config.min_strength_threshold = 0.01f;
    config.min_competing_associations = 2;

    bool applied = CompetitiveLearner::ApplyCompetition(matrix, p1, config);

    // Should not apply because only 1 association meets the threshold
    EXPECT_FALSE(applied);
}

TEST(CompetitiveLearnerTest, ApplyTypedCompetitionOnlyAffectsSameType) {
    AssociationMatrix matrix;

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();
    PatternID p4 = CreateTestPattern();

    // Create associations of different types
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.3f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.7f));  // Winner for CAUSAL
    matrix.AddAssociation(AssociationEdge(p1, p4, AssociationType::SPATIAL, 0.9f)); // Different type

    CompetitiveLearner::Config config;
    config.competition_factor = 0.3f;

    bool applied = CompetitiveLearner::ApplyTypedCompetition(
        matrix, p1, AssociationType::CAUSAL, config
    );

    EXPECT_TRUE(applied);

    // CAUSAL associations should be affected
    const auto* causal_winner = matrix.GetAssociation(p1, p3);
    ASSERT_NE(nullptr, causal_winner);
    EXPECT_GT(causal_winner->GetStrength(), 0.7f);  // Boosted

    const auto* causal_loser = matrix.GetAssociation(p1, p2);
    ASSERT_NE(nullptr, causal_loser);
    EXPECT_LT(causal_loser->GetStrength(), 0.3f);  // Suppressed

    // SPATIAL association should be unchanged
    const auto* spatial = matrix.GetAssociation(p1, p4);
    ASSERT_NE(nullptr, spatial);
    EXPECT_FLOAT_EQ(0.9f, spatial->GetStrength());  // Unchanged
}

TEST(CompetitiveLearnerTest, ApplyCompetitionBatchProcessesMultiplePatterns) {
    AssociationMatrix matrix;

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();
    PatternID p4 = CreateTestPattern();
    PatternID p5 = CreateTestPattern();

    // Pattern p1 has competing associations
    matrix.AddAssociation(CreateTestEdge(p1, p3, 0.3f));
    matrix.AddAssociation(CreateTestEdge(p1, p4, 0.7f));

    // Pattern p2 has competing associations
    matrix.AddAssociation(CreateTestEdge(p2, p3, 0.6f));
    matrix.AddAssociation(CreateTestEdge(p2, p5, 0.4f));

    std::vector<PatternID> patterns = {p1, p2};

    CompetitiveLearner::Config config;
    config.competition_factor = 0.3f;

    size_t applied_count = CompetitiveLearner::ApplyCompetitionBatch(matrix, patterns, config);

    EXPECT_EQ(2u, applied_count);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(CompetitiveLearnerTest, AnalyzeCompetitionProvidesCorrectStats) {
    AssociationMatrix matrix;

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();
    PatternID p4 = CreateTestPattern();

    // Create associations
    matrix.AddAssociation(CreateTestEdge(p1, p2, 0.3f));
    matrix.AddAssociation(CreateTestEdge(p1, p3, 0.7f));  // Winner
    matrix.AddAssociation(CreateTestEdge(p1, p4, 0.5f));

    CompetitiveLearner::Config config;
    config.competition_factor = 0.3f;

    auto stats = CompetitiveLearner::AnalyzeCompetition(matrix, p1, config);

    EXPECT_EQ(1u, stats.patterns_processed);
    EXPECT_EQ(1u, stats.competitions_applied);
    EXPECT_EQ(1u, stats.winners_boosted);
    EXPECT_EQ(2u, stats.losers_suppressed);

    // Total strength before: 0.3 + 0.7 + 0.5 = 1.5
    EXPECT_NEAR(1.5f, stats.total_strength_before, 0.01f);

    // Winner boost and loser suppression should be positive
    EXPECT_GT(stats.average_winner_boost, 0.0f);
    EXPECT_GT(stats.average_loser_suppression, 0.0f);
}

TEST(CompetitiveLearnerTest, ApplyCompetitionWithStatsReturnsCorrectStats) {
    AssociationMatrix matrix;

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();

    matrix.AddAssociation(CreateTestEdge(p1, p2, 0.4f));
    matrix.AddAssociation(CreateTestEdge(p1, p3, 0.6f));  // Winner

    CompetitiveLearner::Config config;
    config.competition_factor = 0.25f;

    auto stats = CompetitiveLearner::ApplyCompetitionWithStats(matrix, p1, config);

    EXPECT_EQ(1u, stats.competitions_applied);
    EXPECT_EQ(1u, stats.winners_boosted);
    EXPECT_EQ(1u, stats.losers_suppressed);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(CompetitiveLearnerTest, NoCompetitionWithNoAssociations) {
    AssociationMatrix matrix;
    PatternID p1 = CreateTestPattern();

    CompetitiveLearner::Config config;

    bool applied = CompetitiveLearner::ApplyCompetition(matrix, p1, config);

    EXPECT_FALSE(applied);
}

TEST(CompetitiveLearnerTest, CompetitionFactorZeroMeansNoChange) {
    AssociationMatrix matrix;

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();

    matrix.AddAssociation(CreateTestEdge(p1, p2, 0.3f));
    matrix.AddAssociation(CreateTestEdge(p1, p3, 0.7f));

    CompetitiveLearner::Config config;
    config.competition_factor = 0.0f;  // No competition

    CompetitiveLearner::ApplyCompetition(matrix, p1, config);

    // Strengths should remain unchanged
    const auto* edge1 = matrix.GetAssociation(p1, p2);
    const auto* edge2 = matrix.GetAssociation(p1, p3);

    ASSERT_NE(nullptr, edge1);
    ASSERT_NE(nullptr, edge2);

    EXPECT_FLOAT_EQ(0.3f, edge1->GetStrength());
    EXPECT_FLOAT_EQ(0.7f, edge2->GetStrength());
}

TEST(CompetitiveLearnerTest, CompetitionFactorOneMeansWinnerTakesAll) {
    AssociationMatrix matrix;

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();

    matrix.AddAssociation(CreateTestEdge(p1, p2, 0.3f));
    matrix.AddAssociation(CreateTestEdge(p1, p3, 0.7f));  // Winner

    CompetitiveLearner::Config config;
    config.competition_factor = 1.0f;  // Complete competition

    CompetitiveLearner::ApplyCompetition(matrix, p1, config);

    const auto* winner = matrix.GetAssociation(p1, p3);
    const auto* loser = matrix.GetAssociation(p1, p2);

    ASSERT_NE(nullptr, winner);
    ASSERT_NE(nullptr, loser);

    // Winner should be boosted to 1.0
    EXPECT_FLOAT_EQ(1.0f, winner->GetStrength());

    // Loser should be suppressed to 0.0
    EXPECT_FLOAT_EQ(0.0f, loser->GetStrength());
}

TEST(CompetitiveLearnerTest, ApplyCompetitionIncomingWorks) {
    AssociationMatrix matrix;

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();
    PatternID target = CreateTestPattern();

    // Multiple sources pointing to same target
    matrix.AddAssociation(CreateTestEdge(p1, target, 0.3f));
    matrix.AddAssociation(CreateTestEdge(p2, target, 0.7f));  // Winner
    matrix.AddAssociation(CreateTestEdge(p3, target, 0.5f));

    CompetitiveLearner::Config config;
    config.competition_factor = 0.3f;

    bool applied = CompetitiveLearner::ApplyCompetitionIncoming(matrix, target, config);

    EXPECT_TRUE(applied);

    // Check winner was boosted
    const auto* winner = matrix.GetAssociation(p2, target);
    ASSERT_NE(nullptr, winner);
    EXPECT_GT(winner->GetStrength(), 0.7f);

    // Check losers were suppressed
    const auto* loser1 = matrix.GetAssociation(p1, target);
    const auto* loser2 = matrix.GetAssociation(p3, target);

    ASSERT_NE(nullptr, loser1);
    ASSERT_NE(nullptr, loser2);

    EXPECT_LT(loser1->GetStrength(), 0.3f);
    EXPECT_LT(loser2->GetStrength(), 0.5f);
}

} // namespace
} // namespace dpan
