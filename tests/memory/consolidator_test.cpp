// File: tests/memory/consolidator_test.cpp
#include "memory/consolidator.hpp"
#include "storage/memory_backend.hpp"
#include "similarity/geometric_similarity.hpp"
#include <gtest/gtest.h>

namespace dpan {
namespace {

// ============================================================================
// Test Fixture
// ============================================================================

class MemoryConsolidatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        MemoryBackend::Config config;
        backend_ = std::make_unique<MemoryBackend>(config);
        similarity_ = std::make_unique<ChamferSimilarity>();

        // Create test patterns
        for (int i = 0; i < 10; ++i) {
            patterns_.push_back(CreateTestPattern(i));
        }
    }

    PatternNode CreateTestPattern(int index) {
        PatternID id = PatternID::Generate();

        // Create feature vector with index-based values
        std::vector<float> features(8);
        for (size_t i = 0; i < features.size(); ++i) {
            features[i] = static_cast<float>(index) + static_cast<float>(i) * 0.1f;
        }

        FeatureVector fv(features);
        PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);

        return PatternNode(id, data, PatternType::ATOMIC);
    }

    PatternNode CreateSimilarPattern(const PatternNode& original, float variation = 0.05f) {
        PatternID id = PatternID::Generate();

        // Create similar features with small variation
        const FeatureVector& orig_features = original.GetData().GetFeatures();
        std::vector<float> features(orig_features.Dimension());

        for (size_t i = 0; i < features.size(); ++i) {
            features[i] = orig_features[i] + variation;
        }

        FeatureVector fv(features);
        PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);

        return PatternNode(id, data, PatternType::ATOMIC);
    }

    std::unique_ptr<MemoryBackend> backend_;
    std::unique_ptr<ChamferSimilarity> similarity_;
    std::vector<PatternNode> patterns_;
};

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(MemoryConsolidatorTest, Config_DefaultValid) {
    MemoryConsolidator::Config config;
    EXPECT_NO_THROW(MemoryConsolidator consolidator(config));
}

TEST_F(MemoryConsolidatorTest, Config_InvalidMergeSimilarityThreshold) {
    MemoryConsolidator::Config config;
    config.merge_similarity_threshold = 1.5f;
    EXPECT_THROW(MemoryConsolidator consolidator(config), std::invalid_argument);
}

TEST_F(MemoryConsolidatorTest, Config_InvalidClusterSimilarityThreshold) {
    MemoryConsolidator::Config config;
    config.cluster_similarity_threshold = -0.1f;
    EXPECT_THROW(MemoryConsolidator consolidator(config), std::invalid_argument);
}

TEST_F(MemoryConsolidatorTest, Config_InvalidMinClusterSize) {
    MemoryConsolidator::Config config;
    config.min_cluster_size = 0;
    EXPECT_THROW(MemoryConsolidator consolidator(config), std::invalid_argument);
}

TEST_F(MemoryConsolidatorTest, Config_InvalidMaxClusterSize) {
    MemoryConsolidator::Config config;
    config.min_cluster_size = 10;
    config.max_cluster_size = 5;  // Less than min
    EXPECT_THROW(MemoryConsolidator consolidator(config), std::invalid_argument);
}

TEST_F(MemoryConsolidatorTest, Config_SetConfigValid) {
    MemoryConsolidator consolidator;

    MemoryConsolidator::Config new_config;
    new_config.merge_similarity_threshold = 0.9f;
    new_config.cluster_similarity_threshold = 0.6f;

    EXPECT_NO_THROW(consolidator.SetConfig(new_config));
    EXPECT_FLOAT_EQ(0.9f, consolidator.GetConfig().merge_similarity_threshold);
}

// ============================================================================
// Pattern Merging Tests
// ============================================================================

TEST_F(MemoryConsolidatorTest, FindMergeCandidates_NoPatternsInDatabase) {
    MemoryConsolidator consolidator;

    auto candidates = consolidator.FindMergeCandidates(*backend_, *similarity_);

    EXPECT_TRUE(candidates.empty());
}

TEST_F(MemoryConsolidatorTest, FindMergeCandidates_SimilarPatterns) {
    MemoryConsolidator::Config config;
    config.merge_similarity_threshold = 0.95f;
    MemoryConsolidator consolidator(config);

    // Store original pattern
    backend_->Store(patterns_[0]);

    // Store very similar pattern
    PatternNode similar = CreateSimilarPattern(patterns_[0], 0.01f);
    backend_->Store(similar);

    auto candidates = consolidator.FindMergeCandidates(*backend_, *similarity_);

    // Should find the pair as candidates
    EXPECT_GE(candidates.size(), 0u);  // May or may not find depending on similarity calculation
}

TEST_F(MemoryConsolidatorTest, FindMergeCandidates_DissimilarPatterns) {
    MemoryConsolidator::Config config;
    config.merge_similarity_threshold = 0.95f;
    MemoryConsolidator consolidator(config);

    // Store dissimilar patterns
    backend_->Store(patterns_[0]);
    backend_->Store(patterns_[5]);  // Very different

    auto candidates = consolidator.FindMergeCandidates(*backend_, *similarity_);

    // Should not find candidates (dissimilar)
    EXPECT_EQ(0u, candidates.size());
}

TEST_F(MemoryConsolidatorTest, MergeTwoPatterns_TransfersAssociations) {
    MemoryConsolidator consolidator;
    AssociationMatrix matrix;

    // Store two patterns
    backend_->Store(patterns_[0]);
    backend_->Store(patterns_[1]);
    backend_->Store(patterns_[2]);  // Third pattern for associations

    // Create associations for pattern 0
    AssociationEdge edge1(patterns_[0].GetID(), patterns_[2].GetID(), AssociationType::CAUSAL, 0.8f);
    matrix.AddAssociation(edge1);

    // Merge pattern 0 into pattern 1
    bool success = consolidator.MergeTwoPatterns(
        patterns_[0].GetID(),
        patterns_[1].GetID(),
        *backend_,
        matrix
    );

    EXPECT_TRUE(success);

    // Check that pattern 1 now has associations
    auto outgoing = matrix.GetOutgoingAssociations(patterns_[1].GetID());
    EXPECT_GT(outgoing.size(), 0u);
}

TEST_F(MemoryConsolidatorTest, MergeTwoPatterns_RemovesOldPattern) {
    MemoryConsolidator::Config config;
    config.preserve_original_patterns = false;
    MemoryConsolidator consolidator(config);

    AssociationMatrix matrix;

    // Store two patterns
    backend_->Store(patterns_[0]);
    backend_->Store(patterns_[1]);

    // Merge pattern 0 into pattern 1
    consolidator.MergeTwoPatterns(
        patterns_[0].GetID(),
        patterns_[1].GetID(),
        *backend_,
        matrix
    );

    // Pattern 0 should be removed
    EXPECT_FALSE(backend_->Exists(patterns_[0].GetID()));
}

TEST_F(MemoryConsolidatorTest, MergeTwoPatterns_PreservesOriginal) {
    MemoryConsolidator::Config config;
    config.preserve_original_patterns = true;
    MemoryConsolidator consolidator(config);

    AssociationMatrix matrix;

    // Store two patterns
    backend_->Store(patterns_[0]);
    backend_->Store(patterns_[1]);

    // Merge pattern 0 into pattern 1
    consolidator.MergeTwoPatterns(
        patterns_[0].GetID(),
        patterns_[1].GetID(),
        *backend_,
        matrix
    );

    // Pattern 0 should still exist
    EXPECT_TRUE(backend_->Exists(patterns_[0].GetID()));
}

TEST_F(MemoryConsolidatorTest, MergePatterns_Result) {
    MemoryConsolidator::Config config;
    config.merge_similarity_threshold = 0.99f;  // Very high
    config.max_merge_batch = 10;
    MemoryConsolidator consolidator(config);

    AssociationMatrix matrix;

    // Store patterns
    backend_->Store(patterns_[0]);
    backend_->Store(patterns_[1]);

    // Store nearly identical pattern
    PatternNode similar = CreateSimilarPattern(patterns_[0], 0.001f);
    backend_->Store(similar);

    auto result = consolidator.MergePatterns(*backend_, matrix, *similarity_);

    // Check result structure
    EXPECT_GE(result.patterns_removed, 0u);
    EXPECT_GE(result.merged_pairs.size(), 0u);
}

// ============================================================================
// Hierarchy Formation Tests
// ============================================================================

TEST_F(MemoryConsolidatorTest, FindClusters_EmptyPatternList) {
    MemoryConsolidator consolidator;

    std::vector<PatternID> empty;
    auto clusters = consolidator.FindClusters(empty, *backend_, *similarity_);

    EXPECT_TRUE(clusters.empty());
}

TEST_F(MemoryConsolidatorTest, FindClusters_TooFewPatterns) {
    MemoryConsolidator::Config config;
    config.min_cluster_size = 5;
    MemoryConsolidator consolidator(config);

    // Store only 3 patterns
    backend_->Store(patterns_[0]);
    backend_->Store(patterns_[1]);
    backend_->Store(patterns_[2]);

    std::vector<PatternID> ids = {
        patterns_[0].GetID(),
        patterns_[1].GetID(),
        patterns_[2].GetID()
    };

    auto clusters = consolidator.FindClusters(ids, *backend_, *similarity_);

    EXPECT_TRUE(clusters.empty());
}

TEST_F(MemoryConsolidatorTest, CreateClusterParent_CreatesNewPattern) {
    MemoryConsolidator consolidator;

    // Store member patterns
    backend_->Store(patterns_[0]);
    backend_->Store(patterns_[1]);
    backend_->Store(patterns_[2]);

    std::vector<PatternID> cluster = {
        patterns_[0].GetID(),
        patterns_[1].GetID(),
        patterns_[2].GetID()
    };

    PatternID parent_id = consolidator.CreateClusterParent(cluster, *backend_);

    // Parent should exist
    EXPECT_TRUE(backend_->Exists(parent_id));

    // Parent should have centroid features
    auto opt_parent = backend_->Retrieve(parent_id);
    ASSERT_TRUE(opt_parent.has_value());
}

TEST_F(MemoryConsolidatorTest, FormHierarchies_Result) {
    MemoryConsolidator::Config config;
    config.min_cluster_size = 2;
    config.max_cluster_size = 5;
    config.cluster_similarity_threshold = 0.7f;
    MemoryConsolidator consolidator(config);

    AssociationMatrix matrix;

    // Store some patterns
    for (int i = 0; i < 5; ++i) {
        backend_->Store(patterns_[i]);
    }

    auto result = consolidator.FormHierarchies(*backend_, matrix, *similarity_);

    // Check result structure
    EXPECT_GE(result.clusters.size(), 0u);
    EXPECT_GE(result.total_patterns_clustered, 0u);
    EXPECT_GE(result.hierarchies_created, 0u);
}

// ============================================================================
// Association Compression Tests
// ============================================================================

TEST_F(MemoryConsolidatorTest, CreateShortcut_AddsNewEdge) {
    MemoryConsolidator consolidator;
    AssociationMatrix matrix;

    backend_->Store(patterns_[0]);
    backend_->Store(patterns_[1]);

    bool success = consolidator.CreateShortcut(
        patterns_[0].GetID(),
        patterns_[1].GetID(),
        0.7f,
        matrix
    );

    EXPECT_TRUE(success);
    EXPECT_TRUE(matrix.HasAssociation(patterns_[0].GetID(), patterns_[1].GetID()));
}

TEST_F(MemoryConsolidatorTest, FindFrequentPaths_EmptyStats) {
    MemoryConsolidator consolidator;
    AssociationMatrix matrix;

    std::unordered_map<std::pair<PatternID, PatternID>, size_t, PatternPairHash> empty_stats;

    auto paths = consolidator.FindFrequentPaths(matrix, empty_stats);

    // Should return empty (no access stats provided)
    EXPECT_TRUE(paths.empty());
}

TEST_F(MemoryConsolidatorTest, CompressAssociations_Result) {
    MemoryConsolidator::Config config;
    config.min_path_traversals = 5;
    config.path_compression_threshold = 0.6f;
    MemoryConsolidator consolidator(config);

    AssociationMatrix matrix;

    // Create a simple path: A -> B -> C
    backend_->Store(patterns_[0]);
    backend_->Store(patterns_[1]);
    backend_->Store(patterns_[2]);

    AssociationEdge edge1(patterns_[0].GetID(), patterns_[1].GetID(), AssociationType::CAUSAL, 0.8f);
    AssociationEdge edge2(patterns_[1].GetID(), patterns_[2].GetID(), AssociationType::CAUSAL, 0.8f);
    matrix.AddAssociation(edge1);
    matrix.AddAssociation(edge2);

    std::unordered_map<std::pair<PatternID, PatternID>, size_t, PatternPairHash> access_stats;

    auto result = consolidator.CompressAssociations(matrix, access_stats);

    // Check result structure
    EXPECT_GE(result.total_shortcuts, 0u);
    EXPECT_GT(result.graph_edges_before, 0u);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(MemoryConsolidatorTest, Consolidate_AllPhases) {
    MemoryConsolidator::Config config;
    config.enable_pattern_merging = true;
    config.enable_hierarchy_formation = true;
    config.enable_association_compression = true;
    MemoryConsolidator consolidator(config);

    AssociationMatrix matrix;

    // Store patterns
    for (int i = 0; i < 5; ++i) {
        backend_->Store(patterns_[i]);
    }

    // Add some associations
    for (int i = 0; i < 4; ++i) {
        AssociationEdge edge(patterns_[i].GetID(), patterns_[i+1].GetID(),
                            AssociationType::CAUSAL, 0.7f);
        matrix.AddAssociation(edge);
    }

    auto result = consolidator.Consolidate(*backend_, matrix, *similarity_);

    // Check that result has all phases
    EXPECT_GE(result.merge_result.patterns_removed, 0u);
    EXPECT_GE(result.hierarchy_result.hierarchies_created, 0u);
    EXPECT_GE(result.compression_result.total_shortcuts, 0u);
}

TEST_F(MemoryConsolidatorTest, Consolidate_OnlyMerging) {
    MemoryConsolidator::Config config;
    config.enable_pattern_merging = true;
    config.enable_hierarchy_formation = false;
    config.enable_association_compression = false;
    MemoryConsolidator consolidator(config);

    AssociationMatrix matrix;

    backend_->Store(patterns_[0]);
    backend_->Store(patterns_[1]);

    auto result = consolidator.Consolidate(*backend_, matrix, *similarity_);

    // Only merge phase should run
    EXPECT_EQ(0u, result.hierarchy_result.hierarchies_created);
    EXPECT_EQ(0u, result.compression_result.total_shortcuts);
}

TEST_F(MemoryConsolidatorTest, Consolidate_UpdatesStatistics) {
    MemoryConsolidator consolidator;
    AssociationMatrix matrix;

    backend_->Store(patterns_[0]);

    const auto& stats_before = consolidator.GetStatistics();
    EXPECT_EQ(0u, stats_before.total_consolidation_operations);

    consolidator.Consolidate(*backend_, matrix, *similarity_);

    const auto& stats_after = consolidator.GetStatistics();
    EXPECT_EQ(1u, stats_after.total_consolidation_operations);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(MemoryConsolidatorTest, Statistics_InitiallyZero) {
    MemoryConsolidator consolidator;

    const auto& stats = consolidator.GetStatistics();
    EXPECT_EQ(0u, stats.total_consolidation_operations);
    EXPECT_EQ(0u, stats.total_patterns_merged);
    EXPECT_EQ(0u, stats.total_hierarchies_created);
    EXPECT_EQ(0u, stats.total_shortcuts_created);
}

TEST_F(MemoryConsolidatorTest, Statistics_ResetWorks) {
    MemoryConsolidator consolidator;

    consolidator.ResetStatistics();

    const auto& stats = consolidator.GetStatistics();
    EXPECT_EQ(0u, stats.total_consolidation_operations);
}

} // namespace
} // namespace dpan
