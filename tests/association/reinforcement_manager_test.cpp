// File: tests/association/reinforcement_manager_test.cpp
#include "association/reinforcement_manager.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

using namespace dpan;

// ============================================================================
// Construction Tests
// ============================================================================

TEST(ReinforcementManagerTest, DefaultConstruction) {
    ReinforcementManager manager;

    const auto& config = manager.GetConfig();
    EXPECT_FLOAT_EQ(0.1f, config.learning_rate);
    EXPECT_FLOAT_EQ(0.01f, config.decay_rate);
    EXPECT_FLOAT_EQ(0.1f, config.min_strength);
    EXPECT_FLOAT_EQ(1.0f, config.max_strength);
    EXPECT_FLOAT_EQ(0.05f, config.prune_threshold);

    const auto& stats = manager.GetStats();
    EXPECT_EQ(0u, stats.reinforcements);
    EXPECT_EQ(0u, stats.weakenings);
    EXPECT_EQ(0u, stats.decays);
    EXPECT_EQ(0u, stats.pruned);
}

TEST(ReinforcementManagerTest, ConfigConstruction) {
    ReinforcementManager::Config config;
    config.learning_rate = 0.2f;
    config.decay_rate = 0.02f;
    config.min_strength = 0.2f;
    config.max_strength = 0.9f;
    config.prune_threshold = 0.1f;

    ReinforcementManager manager(config);

    const auto& retrieved_config = manager.GetConfig();
    EXPECT_FLOAT_EQ(0.2f, retrieved_config.learning_rate);
    EXPECT_FLOAT_EQ(0.02f, retrieved_config.decay_rate);
    EXPECT_FLOAT_EQ(0.2f, retrieved_config.min_strength);
    EXPECT_FLOAT_EQ(0.9f, retrieved_config.max_strength);
    EXPECT_FLOAT_EQ(0.1f, retrieved_config.prune_threshold);
}

// ============================================================================
// Single Edge Reinforcement Tests
// ============================================================================

TEST(ReinforcementManagerTest, ReinforceIncreasesStrength) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    float initial_strength = edge.GetStrength();
    manager.Reinforce(edge, 1.0f);
    float new_strength = edge.GetStrength();

    EXPECT_GT(new_strength, initial_strength);
}

TEST(ReinforcementManagerTest, ReinforceHebbianFormula) {
    ReinforcementManager::Config config;
    config.learning_rate = 0.1f;
    ReinforcementManager manager(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    manager.Reinforce(edge, 1.0f);

    // Expected: s_new = s + η × (1 - s) × reward
    // s_new = 0.5 + 0.1 × (1 - 0.5) × 1.0 = 0.5 + 0.05 = 0.55
    EXPECT_NEAR(0.55f, edge.GetStrength(), 0.001f);
}

TEST(ReinforcementManagerTest, WeakenDecreasesStrength) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    float initial_strength = edge.GetStrength();
    manager.Weaken(edge, 1.0f);
    float new_strength = edge.GetStrength();

    EXPECT_LT(new_strength, initial_strength);
}

TEST(ReinforcementManagerTest, WeakenFormula) {
    ReinforcementManager::Config config;
    config.learning_rate = 0.1f;
    ReinforcementManager manager(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    manager.Weaken(edge, 1.0f);

    // Expected: s_new = s - η × s × penalty
    // s_new = 0.5 - 0.1 × 0.5 × 1.0 = 0.5 - 0.05 = 0.45
    // But min_strength is 0.1, so should clamp
    float new_strength = edge.GetStrength();
    EXPECT_NEAR(0.45f, new_strength, 0.001f);
    EXPECT_GE(new_strength, manager.GetConfig().min_strength);
}

TEST(ReinforcementManagerTest, StrengthBounds) {
    ReinforcementManager::Config config;
    config.min_strength = 0.2f;
    config.max_strength = 0.8f;
    ReinforcementManager manager(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Test upper bound
    AssociationEdge edge1(p1, p2, AssociationType::CAUSAL, 0.75f);
    manager.Reinforce(edge1, 1.0f);
    EXPECT_LE(edge1.GetStrength(), 0.8f);

    // Test lower bound
    AssociationEdge edge2(p1, p2, AssociationType::CAUSAL, 0.25f);
    for (int i = 0; i < 20; ++i) {
        manager.Weaken(edge2, 1.0f);
    }
    EXPECT_GE(edge2.GetStrength(), 0.2f);
}

TEST(ReinforcementManagerTest, RewardClamping) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge1(p1, p2, AssociationType::CAUSAL, 0.5f);
    AssociationEdge edge2(p1, p2, AssociationType::CAUSAL, 0.5f);

    // Reward > 1.0 should be clamped to 1.0
    manager.Reinforce(edge1, 2.0f);
    manager.Reinforce(edge2, 1.0f);

    EXPECT_NEAR(edge1.GetStrength(), edge2.GetStrength(), 0.001f);
}

// ============================================================================
// Decay Tests
// ============================================================================

TEST(ReinforcementManagerTest, ApplyDecayReducesStrength) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.8f);

    float initial_strength = edge.GetStrength();
    manager.ApplyDecay(edge, std::chrono::seconds(10));
    float new_strength = edge.GetStrength();

    EXPECT_LT(new_strength, initial_strength);
}

TEST(ReinforcementManagerTest, DecayExponentialFormula) {
    ReinforcementManager::Config config;
    config.decay_rate = 0.1f;  // Higher decay for easier testing
    ReinforcementManager manager(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 1.0f);

    manager.ApplyDecay(edge, std::chrono::seconds(1));

    // Expected: s_new = s × exp(-d × t)
    // s_new = 1.0 × exp(-0.1 × 1) ≈ 0.9048
    EXPECT_NEAR(0.9048f, edge.GetStrength(), 0.01f);
}

TEST(ReinforcementManagerTest, LongerDecayProducesSmallerStrength) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge1(p1, p2, AssociationType::CAUSAL, 0.8f);
    AssociationEdge edge2(p1, p2, AssociationType::CAUSAL, 0.8f);

    manager.ApplyDecay(edge1, std::chrono::seconds(1));
    manager.ApplyDecay(edge2, std::chrono::seconds(10));

    EXPECT_GT(edge1.GetStrength(), edge2.GetStrength());
}

// ============================================================================
// Batch Reinforcement Tests
// ============================================================================

TEST(ReinforcementManagerTest, ReinforceBatch) {
    ReinforcementManager manager;
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.5f));

    std::vector<std::pair<PatternID, PatternID>> pairs = {{p1, p2}, {p2, p3}};
    manager.ReinforceBatch(matrix, pairs, 1.0f);

    auto edge1 = matrix.GetAssociation(p1, p2);
    auto edge2 = matrix.GetAssociation(p2, p3);

    ASSERT_NE(nullptr, edge1);
    ASSERT_NE(nullptr, edge2);
    EXPECT_GT(edge1->GetStrength(), 0.5f);
    EXPECT_GT(edge2->GetStrength(), 0.5f);
}

TEST(ReinforcementManagerTest, WeakenBatch) {
    ReinforcementManager manager;
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.5f));

    std::vector<std::pair<PatternID, PatternID>> pairs = {{p1, p2}, {p2, p3}};
    manager.WeakenBatch(matrix, pairs, 1.0f);

    auto edge1 = matrix.GetAssociation(p1, p2);
    auto edge2 = matrix.GetAssociation(p2, p3);

    ASSERT_NE(nullptr, edge1);
    ASSERT_NE(nullptr, edge2);
    EXPECT_LT(edge1->GetStrength(), 0.5f);
    EXPECT_LT(edge2->GetStrength(), 0.5f);
}

TEST(ReinforcementManagerTest, ApplyDecayAll) {
    ReinforcementManager manager;
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.8f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.8f));

    manager.ApplyDecayAll(matrix, std::chrono::seconds(10), false);

    auto edge1 = matrix.GetAssociation(p1, p2);
    auto edge2 = matrix.GetAssociation(p2, p3);

    ASSERT_NE(nullptr, edge1);
    ASSERT_NE(nullptr, edge2);
    EXPECT_LT(edge1->GetStrength(), 0.8f);
    EXPECT_LT(edge2->GetStrength(), 0.8f);
}

TEST(ReinforcementManagerTest, DISABLED_ApplyDecayAllWithAutoPrune) {
    ReinforcementManager::Config config;
    config.prune_threshold = 0.5f;
    config.decay_rate = 0.5f;  // High decay rate
    ReinforcementManager manager(config);

    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.9f));

    EXPECT_EQ(2u, matrix.GetAssociationCount());

    // Apply heavy decay
    manager.ApplyDecayAll(matrix, std::chrono::seconds(10), true);

    // p1->p2 should be pruned, p2->p3 should survive
    EXPECT_LT(matrix.GetAssociationCount(), 2u);
}

// ============================================================================
// Prediction-Based Reinforcement Tests
// ============================================================================

TEST(ReinforcementManagerTest, ReinforcePredictionTruePositive) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    float initial_strength = edge.GetStrength();
    manager.ReinforcePrediction(edge, true, true);  // Predicted and occurred

    EXPECT_GT(edge.GetStrength(), initial_strength);
}

TEST(ReinforcementManagerTest, ReinforcePredictionFalsePositive) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    float initial_strength = edge.GetStrength();
    manager.ReinforcePrediction(edge, true, false);  // Predicted but didn't occur

    EXPECT_LT(edge.GetStrength(), initial_strength);
}

TEST(ReinforcementManagerTest, ReinforcePredictionFalseNegative) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    float initial_strength = edge.GetStrength();
    manager.ReinforcePrediction(edge, false, true);  // Didn't predict but occurred

    // Should slightly reinforce (missed opportunity)
    EXPECT_GT(edge.GetStrength(), initial_strength);
}

TEST(ReinforcementManagerTest, ReinforcePredictionTrueNegative) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    float initial_strength = edge.GetStrength();
    manager.ReinforcePrediction(edge, false, false);  // Didn't predict and didn't occur

    // No change expected
    EXPECT_FLOAT_EQ(initial_strength, edge.GetStrength());
}

TEST(ReinforcementManagerTest, ReinforcePredictionsBatch) {
    ReinforcementManager manager;
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.5f));

    std::vector<std::tuple<PatternID, PatternID, bool, bool>> predictions = {
        {p1, p2, true, true},   // True positive
        {p2, p3, true, false}   // False positive
    };

    manager.ReinforcePredictionsBatch(matrix, predictions);

    auto edge1 = matrix.GetAssociation(p1, p2);
    auto edge2 = matrix.GetAssociation(p2, p3);

    ASSERT_NE(nullptr, edge1);
    ASSERT_NE(nullptr, edge2);

    EXPECT_GT(edge1->GetStrength(), 0.5f);  // Strengthened
    EXPECT_LT(edge2->GetStrength(), 0.5f);  // Weakened
}

// ============================================================================
// Pruning Tests
// ============================================================================

TEST(ReinforcementManagerTest, ShouldPrune) {
    ReinforcementManager::Config config;
    config.prune_threshold = 0.3f;
    ReinforcementManager manager(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge weak_edge(p1, p2, AssociationType::CAUSAL, 0.2f);
    AssociationEdge strong_edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    EXPECT_TRUE(manager.ShouldPrune(weak_edge));
    EXPECT_FALSE(manager.ShouldPrune(strong_edge));
}

TEST(ReinforcementManagerTest, DISABLED_CountPrunableEdges) {
    ReinforcementManager::Config config;
    config.prune_threshold = 0.3f;
    ReinforcementManager manager(config);

    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.2f));  // Prunable
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.5f));  // Not prunable
    matrix.AddAssociation(AssociationEdge(p3, p4, AssociationType::CAUSAL, 0.1f));  // Prunable

    size_t count = manager.CountPrunableEdges(matrix);
    EXPECT_EQ(2u, count);
}

TEST(ReinforcementManagerTest, DISABLED_PruneWeakAssociations) {
    ReinforcementManager::Config config;
    config.prune_threshold = 0.3f;
    ReinforcementManager manager(config);

    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.2f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p3, p4, AssociationType::CAUSAL, 0.1f));

    EXPECT_EQ(3u, matrix.GetAssociationCount());

    size_t pruned = manager.PruneWeakAssociations(matrix);
    EXPECT_EQ(2u, pruned);
    EXPECT_EQ(1u, matrix.GetAssociationCount());
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(ReinforcementManagerTest, StatisticsTracking) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    manager.Reinforce(edge, 1.0f);
    manager.Reinforce(edge, 1.0f);
    manager.Weaken(edge, 1.0f);
    manager.ApplyDecay(edge, std::chrono::seconds(1));

    const auto& stats = manager.GetStats();
    EXPECT_EQ(2u, stats.reinforcements);
    EXPECT_EQ(1u, stats.weakenings);
    EXPECT_EQ(1u, stats.decays);
}

TEST(ReinforcementManagerTest, ResetStats) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    manager.Reinforce(edge, 1.0f);
    manager.Weaken(edge, 1.0f);

    EXPECT_GT(manager.GetStats().reinforcements, 0u);

    manager.ResetStats();

    const auto& stats = manager.GetStats();
    EXPECT_EQ(0u, stats.reinforcements);
    EXPECT_EQ(0u, stats.weakenings);
    EXPECT_EQ(0u, stats.decays);
    EXPECT_EQ(0u, stats.pruned);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ReinforcementManagerTest, ReinforceAlreadyMaxStrength) {
    ReinforcementManager::Config config;
    config.max_strength = 0.9f;
    ReinforcementManager manager(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.9f);

    manager.Reinforce(edge, 1.0f);

    EXPECT_FLOAT_EQ(0.9f, edge.GetStrength());  // Should stay at max
}

TEST(ReinforcementManagerTest, WeakenAlreadyMinStrength) {
    ReinforcementManager::Config config;
    config.min_strength = 0.2f;
    ReinforcementManager manager(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.2f);

    manager.Weaken(edge, 1.0f);

    EXPECT_FLOAT_EQ(0.2f, edge.GetStrength());  // Should stay at min
}

TEST(ReinforcementManagerTest, ZeroReward) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    manager.Reinforce(edge, 0.0f);

    EXPECT_FLOAT_EQ(0.5f, edge.GetStrength());  // No change
}

TEST(ReinforcementManagerTest, ZeroPenalty) {
    ReinforcementManager manager;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);

    manager.Weaken(edge, 0.0f);

    EXPECT_FLOAT_EQ(0.5f, edge.GetStrength());  // No change
}

TEST(ReinforcementManagerTest, BatchWithNonExistentEdges) {
    ReinforcementManager manager;
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));

    std::vector<std::pair<PatternID, PatternID>> pairs = {
        {p1, p2},  // Exists
        {p2, p3}   // Doesn't exist
    };

    // Should not crash
    manager.ReinforceBatch(matrix, pairs, 1.0f);

    auto edge1 = matrix.GetAssociation(p1, p2);
    ASSERT_NE(nullptr, edge1);
    EXPECT_GT(edge1->GetStrength(), 0.5f);
}
