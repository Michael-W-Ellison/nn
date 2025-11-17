// File: tests/memory/association_pruner_test.cpp
#include "memory/association_pruner.hpp"
#include "storage/memory_backend.hpp"
#include <gtest/gtest.h>
#include <thread>

namespace dpan {
namespace {

// ============================================================================
// Test Fixture
// ============================================================================

class AssociationPrunerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create patterns for testing
        for (int i = 0; i < 10; ++i) {
            patterns_.push_back(PatternID::Generate());
        }
    }

    // Helper to create an association with specific strength
    AssociationEdge CreateEdge(
        PatternID source,
        PatternID target,
        float strength,
        AssociationType type = AssociationType::CAUSAL
    ) {
        AssociationEdge edge(source, target, type, strength);
        return edge;
    }

    // Helper to add edge to matrix
    bool AddEdge(
        AssociationMatrix& matrix,
        PatternID source,
        PatternID target,
        float strength
    ) {
        auto edge = CreateEdge(source, target, strength);
        return matrix.AddAssociation(edge);
    }

    // Helper to create stale edge (old reinforcement time)
    AssociationEdge CreateStaleEdge(
        PatternID source,
        PatternID target,
        float strength
    ) {
        AssociationEdge edge(source, target, AssociationType::CAUSAL, strength);
        // Edge is already created with current timestamp, we can't easily make it stale
        // In real usage, edges would naturally become stale over time
        return edge;
    }

    std::vector<PatternID> patterns_;
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(AssociationPrunerTest, Config_DefaultValid) {
    AssociationPruner::Config config;
    EXPECT_NO_THROW(AssociationPruner pruner(config));
}

TEST_F(AssociationPrunerTest, Config_InvalidWeakStrengthTooLow) {
    AssociationPruner::Config config;
    config.weak_strength_threshold = -0.1f;
    EXPECT_THROW(AssociationPruner pruner(config), std::invalid_argument);
}

TEST_F(AssociationPrunerTest, Config_InvalidWeakStrengthTooHigh) {
    AssociationPruner::Config config;
    config.weak_strength_threshold = 1.5f;
    EXPECT_THROW(AssociationPruner pruner(config), std::invalid_argument);
}

TEST_F(AssociationPrunerTest, Config_InvalidMinStrengthHigherThanWeak) {
    AssociationPruner::Config config;
    config.weak_strength_threshold = 0.1f;
    config.min_association_strength = 0.2f;
    EXPECT_THROW(AssociationPruner pruner(config), std::invalid_argument);
}

TEST_F(AssociationPrunerTest, Config_InvalidMaxPathLengthZero) {
    AssociationPruner::Config config;
    config.max_path_length = 0;
    EXPECT_THROW(AssociationPruner pruner(config), std::invalid_argument);
}

TEST_F(AssociationPrunerTest, Config_InvalidMaxPruneBatchZero) {
    AssociationPruner::Config config;
    config.max_prune_batch = 0;
    EXPECT_THROW(AssociationPruner pruner(config), std::invalid_argument);
}

TEST_F(AssociationPrunerTest, Config_InvalidHubThresholdZero) {
    AssociationPruner::Config config;
    config.hub_threshold = 0;
    EXPECT_THROW(AssociationPruner pruner(config), std::invalid_argument);
}

TEST_F(AssociationPrunerTest, Config_SetConfigValid) {
    AssociationPruner pruner;

    AssociationPruner::Config new_config;
    new_config.weak_strength_threshold = 0.05f;
    new_config.min_association_strength = 0.01f;

    EXPECT_NO_THROW(pruner.SetConfig(new_config));
    EXPECT_FLOAT_EQ(0.05f, pruner.GetConfig().weak_strength_threshold);
}

TEST_F(AssociationPrunerTest, Config_SetConfigInvalid) {
    AssociationPruner pruner;

    AssociationPruner::Config bad_config;
    bad_config.weak_strength_threshold = -0.1f;

    EXPECT_THROW(pruner.SetConfig(bad_config), std::invalid_argument);
}

// ============================================================================
// Weak Association Detection Tests
// ============================================================================

TEST_F(AssociationPrunerTest, IsWeak_BelowThreshold) {
    AssociationPruner::Config config;
    config.weak_strength_threshold = 0.1f;
    AssociationPruner pruner(config);

    auto edge = CreateEdge(patterns_[0], patterns_[1], 0.05f);
    EXPECT_TRUE(pruner.IsWeak(edge));
}

TEST_F(AssociationPrunerTest, IsWeak_AboveThreshold) {
    AssociationPruner::Config config;
    config.weak_strength_threshold = 0.1f;
    AssociationPruner pruner(config);

    auto edge = CreateEdge(patterns_[0], patterns_[1], 0.2f);
    EXPECT_FALSE(pruner.IsWeak(edge));
}

TEST_F(AssociationPrunerTest, IsWeak_ExactlyAtThreshold) {
    AssociationPruner::Config config;
    config.weak_strength_threshold = 0.1f;
    AssociationPruner pruner(config);

    auto edge = CreateEdge(patterns_[0], patterns_[1], 0.1f);
    EXPECT_FALSE(pruner.IsWeak(edge));  // Should be false (>= threshold)
}

TEST_F(AssociationPrunerTest, IsWeak_ZeroStrength) {
    AssociationPruner::Config config;
    config.weak_strength_threshold = 0.1f;
    AssociationPruner pruner(config);

    auto edge = CreateEdge(patterns_[0], patterns_[1], 0.0f);
    EXPECT_TRUE(pruner.IsWeak(edge));
}

// ============================================================================
// Stale Association Detection Tests
// ============================================================================

TEST_F(AssociationPrunerTest, IsStale_RecentEdge) {
    AssociationPruner::Config config;
    config.staleness_threshold = std::chrono::hours(24);
    AssociationPruner pruner(config);

    auto edge = CreateEdge(patterns_[0], patterns_[1], 0.5f);
    // Edge is just created, so it's not stale
    EXPECT_FALSE(pruner.IsStale(edge));
}

TEST_F(AssociationPrunerTest, IsStale_JustCreated) {
    AssociationPruner::Config config;
    config.staleness_threshold = std::chrono::hours(24 * 30);  // 30 days
    AssociationPruner pruner(config);

    auto edge = CreateEdge(patterns_[0], patterns_[1], 0.5f);
    EXPECT_FALSE(pruner.IsStale(edge));
}

TEST_F(AssociationPrunerTest, IsStale_VeryShortThreshold) {
    AssociationPruner::Config config;
    config.staleness_threshold = std::chrono::milliseconds(1);
    AssociationPruner pruner(config);

    auto edge = CreateEdge(patterns_[0], patterns_[1], 0.5f);

    // Sleep to make edge stale
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    EXPECT_TRUE(pruner.IsStale(edge));
}

// ============================================================================
// Redundancy Detection Tests
// ============================================================================

TEST_F(AssociationPrunerTest, IsRedundant_NoAlternativePath) {
    AssociationPruner::Config config;
    config.enable_redundancy_detection = true;
    AssociationPruner pruner(config);

    AssociationMatrix matrix;

    // Add single edge
    AddEdge(matrix, patterns_[0], patterns_[1], 0.5f);

    const AssociationEdge* edge = matrix.GetAssociation(patterns_[0], patterns_[1]);
    ASSERT_NE(nullptr, edge);

    EXPECT_FALSE(pruner.IsRedundant(*edge, matrix));
}

TEST_F(AssociationPrunerTest, IsRedundant_StrongerAlternativePath) {
    AssociationPruner::Config config;
    config.enable_redundancy_detection = true;
    config.redundancy_path_strength_threshold = 0.5f;
    AssociationPruner pruner(config);

    AssociationMatrix matrix;

    // Direct path: A -> C with strength 0.3
    AddEdge(matrix, patterns_[0], patterns_[2], 0.3f);

    // Alternative path: A -> B -> C with stronger combined strength
    AddEdge(matrix, patterns_[0], patterns_[1], 0.8f);
    AddEdge(matrix, patterns_[1], patterns_[2], 0.8f);
    // Combined: 0.8 * 0.8 = 0.64 > 0.3

    const AssociationEdge* direct_edge = matrix.GetAssociation(patterns_[0], patterns_[2]);
    ASSERT_NE(nullptr, direct_edge);

    EXPECT_TRUE(pruner.IsRedundant(*direct_edge, matrix));
}

TEST_F(AssociationPrunerTest, IsRedundant_WeakerAlternativePath) {
    AssociationPruner::Config config;
    config.enable_redundancy_detection = true;
    config.redundancy_path_strength_threshold = 0.5f;
    AssociationPruner pruner(config);

    AssociationMatrix matrix;

    // Direct path: A -> C with strength 0.8
    AddEdge(matrix, patterns_[0], patterns_[2], 0.8f);

    // Alternative path: A -> B -> C with weaker combined strength
    AddEdge(matrix, patterns_[0], patterns_[1], 0.3f);
    AddEdge(matrix, patterns_[1], patterns_[2], 0.3f);
    // Combined: 0.3 * 0.3 = 0.09 < 0.8

    const AssociationEdge* direct_edge = matrix.GetAssociation(patterns_[0], patterns_[2]);
    ASSERT_NE(nullptr, direct_edge);

    EXPECT_FALSE(pruner.IsRedundant(*direct_edge, matrix));
}

TEST_F(AssociationPrunerTest, IsRedundant_DisabledDetection) {
    AssociationPruner::Config config;
    config.enable_redundancy_detection = false;
    AssociationPruner pruner(config);

    AssociationMatrix matrix;

    // Even with alternative path, should return false when disabled
    AddEdge(matrix, patterns_[0], patterns_[2], 0.3f);
    AddEdge(matrix, patterns_[0], patterns_[1], 0.8f);
    AddEdge(matrix, patterns_[1], patterns_[2], 0.8f);

    const AssociationEdge* direct_edge = matrix.GetAssociation(patterns_[0], patterns_[2]);
    ASSERT_NE(nullptr, direct_edge);

    EXPECT_FALSE(pruner.IsRedundant(*direct_edge, matrix));
}

// ============================================================================
// Alternative Path Finding Tests
// ============================================================================

TEST_F(AssociationPrunerTest, FindAlternativePath_DirectPathIgnored) {
    AssociationPruner::Config config;
    config.max_path_length = 2;
    AssociationPruner pruner(config);

    AssociationMatrix matrix;

    // Only direct path exists
    AddEdge(matrix, patterns_[0], patterns_[1], 0.8f);

    // Should return 0 because direct path is ignored
    float path_strength = pruner.FindAlternativePath(
        patterns_[0], patterns_[1], matrix, 0.8f
    );

    EXPECT_FLOAT_EQ(0.0f, path_strength);
}

TEST_F(AssociationPrunerTest, FindAlternativePath_TwoHopPath) {
    AssociationPruner::Config config;
    config.max_path_length = 3;
    AssociationPruner pruner(config);

    AssociationMatrix matrix;

    // Two-hop path: A -> B -> C
    AddEdge(matrix, patterns_[0], patterns_[1], 0.6f);
    AddEdge(matrix, patterns_[1], patterns_[2], 0.7f);

    float path_strength = pruner.FindAlternativePath(
        patterns_[0], patterns_[2], matrix, 0.0f
    );

    // Expected: 0.6 * 0.7 = 0.42
    EXPECT_NEAR(0.42f, path_strength, 0.01f);
}

TEST_F(AssociationPrunerTest, FindAlternativePath_ThreeHopPath) {
    AssociationPruner::Config config;
    config.max_path_length = 4;
    AssociationPruner pruner(config);

    AssociationMatrix matrix;

    // Three-hop path: A -> B -> C -> D
    AddEdge(matrix, patterns_[0], patterns_[1], 0.8f);
    AddEdge(matrix, patterns_[1], patterns_[2], 0.8f);
    AddEdge(matrix, patterns_[2], patterns_[3], 0.8f);

    float path_strength = pruner.FindAlternativePath(
        patterns_[0], patterns_[3], matrix, 0.0f
    );

    // Expected: 0.8 * 0.8 * 0.8 = 0.512
    EXPECT_NEAR(0.512f, path_strength, 0.01f);
}

TEST_F(AssociationPrunerTest, FindAlternativePath_MaxDepthLimit) {
    AssociationPruner::Config config;
    config.max_path_length = 2;  // Only 2 hops allowed
    AssociationPruner pruner(config);

    AssociationMatrix matrix;

    // Three-hop path: A -> B -> C -> D
    AddEdge(matrix, patterns_[0], patterns_[1], 0.8f);
    AddEdge(matrix, patterns_[1], patterns_[2], 0.8f);
    AddEdge(matrix, patterns_[2], patterns_[3], 0.8f);

    float path_strength = pruner.FindAlternativePath(
        patterns_[0], patterns_[3], matrix, 0.0f
    );

    // Should return 0 because path requires 3 hops but limit is 2
    EXPECT_FLOAT_EQ(0.0f, path_strength);
}

// ============================================================================
// Safety Check Tests
// ============================================================================

TEST_F(AssociationPrunerTest, IsSafeToPrune_BelowMinStrength) {
    AssociationPruner::Config config;
    config.min_association_strength = 0.05f;
    AssociationPruner pruner(config);

    AssociationMatrix matrix;
    auto edge = CreateEdge(patterns_[0], patterns_[1], 0.03f);

    EXPECT_TRUE(pruner.IsSafeToPrune(edge, matrix));
}

TEST_F(AssociationPrunerTest, IsSafeToPrune_AboveMinStrength) {
    AssociationPruner::Config config;
    config.min_association_strength = 0.05f;
    AssociationPruner pruner(config);

    AssociationMatrix matrix;
    auto edge = CreateEdge(patterns_[0], patterns_[1], 0.1f);

    EXPECT_FALSE(pruner.IsSafeToPrune(edge, matrix));
}

TEST_F(AssociationPrunerTest, IsHub_ManyAssociations) {
    AssociationPruner::Config config;
    config.hub_threshold = 5;  // Patterns_ has 10 items, so we create 9 edges
    AssociationPruner pruner(config);

    AssociationMatrix matrix;

    // Create pattern with many outgoing edges (9 edges total)
    for (size_t i = 1; i < patterns_.size(); ++i) {
        AddEdge(matrix, patterns_[0], patterns_[i], 0.5f);
    }

    EXPECT_TRUE(pruner.IsHub(patterns_[0], matrix));
}

TEST_F(AssociationPrunerTest, IsHub_FewAssociations) {
    AssociationPruner::Config config;
    config.hub_threshold = 10;
    AssociationPruner pruner(config);

    AssociationMatrix matrix;

    // Create pattern with few edges
    AddEdge(matrix, patterns_[0], patterns_[1], 0.5f);
    AddEdge(matrix, patterns_[0], patterns_[2], 0.5f);

    EXPECT_FALSE(pruner.IsHub(patterns_[0], matrix));
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(AssociationPrunerTest, Statistics_InitiallyZero) {
    AssociationPruner pruner;

    const auto& stats = pruner.GetStatistics();
    EXPECT_EQ(0u, stats.total_prune_operations);
    EXPECT_EQ(0u, stats.total_associations_removed);
    EXPECT_EQ(0u, stats.weak_removed);
    EXPECT_EQ(0u, stats.stale_removed);
    EXPECT_EQ(0u, stats.redundant_removed);
}

TEST_F(AssociationPrunerTest, Statistics_ResetWorks) {
    AssociationPruner pruner;

    // Manually modify stats (in real usage, would be done by pruning operations)
    pruner.ResetStatistics();

    const auto& stats = pruner.GetStatistics();
    EXPECT_EQ(0u, stats.total_prune_operations);
    EXPECT_EQ(0u, stats.total_associations_removed);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(AssociationPrunerTest, PruneAssociations_EmptyMatrix) {
    AssociationPruner pruner;
    AssociationMatrix matrix;

    auto result = pruner.PruneAssociations(matrix, nullptr);

    EXPECT_EQ(0u, result.total_pruned);
    EXPECT_EQ(0u, result.associations_before);
    EXPECT_EQ(0u, result.associations_after);
}

TEST_F(AssociationPrunerTest, PruneAssociations_NoCandidates) {
    AssociationPruner::Config config;
    config.weak_strength_threshold = 0.1f;
    AssociationPruner pruner(config);

    AssociationMatrix matrix;

    // Add strong associations
    AddEdge(matrix, patterns_[0], patterns_[1], 0.8f);
    AddEdge(matrix, patterns_[1], patterns_[2], 0.7f);

    auto result = pruner.PruneAssociations(matrix, nullptr);

    // Should not prune anything (all associations strong)
    EXPECT_EQ(0u, result.total_pruned);
}

TEST_F(AssociationPrunerTest, PruneAssociations_ResultStatistics) {
    AssociationPruner pruner;
    AssociationMatrix matrix;

    // Add some associations
    AddEdge(matrix, patterns_[0], patterns_[1], 0.5f);
    AddEdge(matrix, patterns_[1], patterns_[2], 0.6f);

    auto result = pruner.PruneAssociations(matrix, nullptr);

    // Verify result structure
    EXPECT_EQ(2u, result.associations_before);
    EXPECT_EQ(result.associations_before - result.total_pruned, result.associations_after);
}

} // namespace
} // namespace dpan
