// File: tests/learning/relationship_discovery_test.cpp
//
// Unit tests for DiscoverRelatedPatterns

#include "learning/self_attention.hpp"
#include "association/association_matrix.hpp"
#include "association/association_edge.hpp"
#include "attention_test_fixtures.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace dpan;
using namespace dpan::attention;
using namespace dpan::testing;

class RelationshipDiscoveryTest : public AttentionTestFixture {
protected:
    void SetUp() override {
        AttentionTestFixture::SetUp();

        // Create self-attention with default config
        SelfAttentionConfig config;
        self_attn_ = std::make_unique<SelfAttention>(config);
        self_attn_->SetPatternDatabase(mock_db_.get());

        // Create association matrix
        association_matrix_ = std::make_unique<AssociationMatrix>();
    }

    /// Create patterns with varying features for testing similarity
    std::vector<PatternID> CreatePatternsWithFeatures(size_t count) {
        std::vector<PatternID> ids;

        for (size_t i = 0; i < count; ++i) {
            PatternID id = PatternID::Generate();

            // Create feature vector with varying values
            FeatureVector features(10);
            for (size_t j = 0; j < 10; ++j) {
                // Create different patterns: pattern i has higher values at index i
                features[j] = (i == j) ? 1.0f : 0.1f;
            }

            // Create pattern data from features
            PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
            PatternNode node(id, data, PatternType::ATOMIC);

            mock_db_->Store(node);
            ids.push_back(id);
        }

        return ids;
    }

    /// Add explicit association between two patterns
    void AddExplicitAssociation(
        PatternID source,
        PatternID target,
        AssociationType type,
        float strength) {

        AssociationEdge edge(source, target, type);
        edge.SetStrength(strength);
        association_matrix_->AddAssociation(edge);
    }

    std::unique_ptr<SelfAttention> self_attn_;
    std::unique_ptr<AssociationMatrix> association_matrix_;
};

// ============================================================================
// Basic Discovery Tests
// ============================================================================

TEST_F(RelationshipDiscoveryTest, BasicDiscovery) {
    auto pattern_ids = CreatePatternsWithFeatures(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 3, context);

    EXPECT_EQ(result.query, query);
    EXPECT_EQ(result.relationships.size(), 3u);

    // Should be sorted by attention weight
    for (size_t i = 1; i < result.relationships.size(); ++i) {
        EXPECT_GE(result.relationships[i-1].attention_weight,
                  result.relationships[i].attention_weight);
    }
}

TEST_F(RelationshipDiscoveryTest, DiscoveryWithAllCandidates) {
    auto pattern_ids = CreatePatternsWithFeatures(6);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());

    ContextVector context;
    // Request more than available
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 10, context);

    EXPECT_EQ(result.query, query);
    // Should return all 5 candidates
    EXPECT_EQ(result.relationships.size(), 5u);
}

TEST_F(RelationshipDiscoveryTest, EmptyCandidates) {
    auto pattern_ids = CreatePatternsWithFeatures(1);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates;  // Empty

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 5, context);

    EXPECT_EQ(result.query, query);
    EXPECT_EQ(result.relationships.size(), 0u);
}

TEST_F(RelationshipDiscoveryTest, TopKLimiting) {
    auto pattern_ids = CreatePatternsWithFeatures(10);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());

    ContextVector context;

    // Test different k values
    auto result_3 = self_attn_->DiscoverRelatedPatterns(query, candidates, 3, context);
    EXPECT_EQ(result_3.relationships.size(), 3u);

    auto result_5 = self_attn_->DiscoverRelatedPatterns(query, candidates, 5, context);
    EXPECT_EQ(result_5.relationships.size(), 5u);

    // Top-3 should be subset of top-5
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(result_3.relationships[i].pattern, result_5.relationships[i].pattern);
    }
}

// ============================================================================
// Novel Relationship Tests
// ============================================================================

TEST_F(RelationshipDiscoveryTest, NovelRelationships) {
    auto pattern_ids = CreatePatternsWithFeatures(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());

    // Don't set association matrix - all relationships should be novel
    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 4, context);

    EXPECT_EQ(result.relationships.size(), 4u);

    // All should be novel (no explicit associations)
    for (const auto& rel : result.relationships) {
        EXPECT_FALSE(rel.has_explicit_association);
        EXPECT_TRUE(rel.is_novel());
        EXPECT_FALSE(rel.is_confirmed());
    }

    EXPECT_EQ(result.novel_count(), 4u);
    EXPECT_EQ(result.confirmed_count(), 0u);
}

TEST_F(RelationshipDiscoveryTest, IdentifyNovelRelationships) {
    auto pattern_ids = CreatePatternsWithFeatures(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());

    // Add explicit associations for some patterns
    AddExplicitAssociation(query, pattern_ids[1], AssociationType::CAUSAL, 0.8f);
    AddExplicitAssociation(query, pattern_ids[2], AssociationType::CATEGORICAL, 0.9f);
    // pattern_ids[3] and [4] have no explicit associations

    self_attn_->SetAssociationMatrix(association_matrix_.get());

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 4, context);

    EXPECT_EQ(result.relationships.size(), 4u);

    // Count novel vs confirmed
    size_t novel_count = 0;
    size_t confirmed_count = 0;

    for (const auto& rel : result.relationships) {
        if (rel.is_novel()) {
            ++novel_count;
            EXPECT_FALSE(rel.has_explicit_association);
        } else {
            ++confirmed_count;
            EXPECT_TRUE(rel.has_explicit_association);
        }
    }

    // We added 2 explicit associations, so 2 should be confirmed, 2 novel
    EXPECT_EQ(confirmed_count, 2u);
    EXPECT_EQ(novel_count, 2u);

    EXPECT_EQ(result.novel_count(), 2u);
    EXPECT_EQ(result.confirmed_count(), 2u);
}

TEST_F(RelationshipDiscoveryTest, GetNovelRelationships) {
    auto pattern_ids = CreatePatternsWithFeatures(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());

    // Add explicit association for first candidate only
    AddExplicitAssociation(query, pattern_ids[1], AssociationType::CAUSAL, 0.8f);

    self_attn_->SetAssociationMatrix(association_matrix_.get());

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 4, context);

    auto novel = result.get_novel_relationships();

    // Should have 3 novel relationships
    EXPECT_EQ(novel.size(), 3u);

    // All should be novel
    for (const auto& rel : novel) {
        EXPECT_TRUE(rel.is_novel());
        EXPECT_FALSE(rel.has_explicit_association);
        // Should not include pattern_ids[1]
        EXPECT_NE(rel.pattern, pattern_ids[1]);
    }
}

// ============================================================================
// Confirmed Relationship Tests
// ============================================================================

TEST_F(RelationshipDiscoveryTest, ConfirmedRelationships) {
    auto pattern_ids = CreatePatternsWithFeatures(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());

    // Add explicit associations for all candidates
    AddExplicitAssociation(query, pattern_ids[1], AssociationType::CAUSAL, 0.8f);
    AddExplicitAssociation(query, pattern_ids[2], AssociationType::CATEGORICAL, 0.9f);
    AddExplicitAssociation(query, pattern_ids[3], AssociationType::SPATIAL, 0.7f);
    AddExplicitAssociation(query, pattern_ids[4], AssociationType::FUNCTIONAL, 0.6f);

    self_attn_->SetAssociationMatrix(association_matrix_.get());

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 4, context);

    EXPECT_EQ(result.relationships.size(), 4u);

    // All should be confirmed (have explicit associations)
    for (const auto& rel : result.relationships) {
        EXPECT_TRUE(rel.has_explicit_association);
        EXPECT_FALSE(rel.is_novel());
        EXPECT_TRUE(rel.is_confirmed());
        EXPECT_GT(rel.explicit_strength, 0.0f);
    }

    EXPECT_EQ(result.novel_count(), 0u);
    EXPECT_EQ(result.confirmed_count(), 4u);
}

TEST_F(RelationshipDiscoveryTest, GetConfirmedRelationships) {
    auto pattern_ids = CreatePatternsWithFeatures(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());

    // Add explicit associations for two candidates
    AddExplicitAssociation(query, pattern_ids[1], AssociationType::CAUSAL, 0.8f);
    AddExplicitAssociation(query, pattern_ids[3], AssociationType::CATEGORICAL, 0.9f);

    self_attn_->SetAssociationMatrix(association_matrix_.get());

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 4, context);

    auto confirmed = result.get_confirmed_relationships();

    // Should have 2 confirmed relationships
    EXPECT_EQ(confirmed.size(), 2u);

    // All should be confirmed
    for (const auto& rel : confirmed) {
        EXPECT_TRUE(rel.is_confirmed());
        EXPECT_TRUE(rel.has_explicit_association);
        EXPECT_GT(rel.explicit_strength, 0.0f);
    }
}

TEST_F(RelationshipDiscoveryTest, ExplicitAssociationDetails) {
    auto pattern_ids = CreatePatternsWithFeatures(3);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates{pattern_ids[1], pattern_ids[2]};

    // Add associations with different types and strengths
    AddExplicitAssociation(query, pattern_ids[1], AssociationType::CAUSAL, 0.85f);
    AddExplicitAssociation(query, pattern_ids[2], AssociationType::SPATIAL, 0.65f);

    self_attn_->SetAssociationMatrix(association_matrix_.get());

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 2, context);

    EXPECT_EQ(result.relationships.size(), 2u);

    // Check details of confirmed relationships
    for (const auto& rel : result.relationships) {
        EXPECT_TRUE(rel.has_explicit_association);

        if (rel.pattern == pattern_ids[1]) {
            EXPECT_EQ(rel.explicit_type, AssociationType::CAUSAL);
            EXPECT_FLOAT_EQ(rel.explicit_strength, 0.85f);
        } else if (rel.pattern == pattern_ids[2]) {
            EXPECT_EQ(rel.explicit_type, AssociationType::SPATIAL);
            EXPECT_FLOAT_EQ(rel.explicit_strength, 0.65f);
        }
    }
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(RelationshipDiscoveryTest, UtilityFunctions) {
    auto pattern_ids = CreatePatternsWithFeatures(6);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());

    // Add 2 explicit associations
    AddExplicitAssociation(query, pattern_ids[1], AssociationType::CAUSAL, 0.8f);
    AddExplicitAssociation(query, pattern_ids[3], AssociationType::CATEGORICAL, 0.9f);

    self_attn_->SetAssociationMatrix(association_matrix_.get());

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 5, context);

    // Test counts
    EXPECT_EQ(result.novel_count(), 3u);
    EXPECT_EQ(result.confirmed_count(), 2u);

    // Test get_novel_relationships
    auto novel = result.get_novel_relationships();
    EXPECT_EQ(novel.size(), 3u);
    for (const auto& rel : novel) {
        EXPECT_TRUE(rel.is_novel());
    }

    // Test get_confirmed_relationships
    auto confirmed = result.get_confirmed_relationships();
    EXPECT_EQ(confirmed.size(), 2u);
    for (const auto& rel : confirmed) {
        EXPECT_TRUE(rel.is_confirmed());
    }
}

TEST_F(RelationshipDiscoveryTest, AttentionWeightsSorted) {
    auto pattern_ids = CreatePatternsWithFeatures(8);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 5, context);

    EXPECT_EQ(result.relationships.size(), 5u);

    // Verify sorting by attention weight (descending)
    for (size_t i = 1; i < result.relationships.size(); ++i) {
        EXPECT_GE(result.relationships[i-1].attention_weight,
                  result.relationships[i].attention_weight);
    }

    // All weights should be in [0, 1]
    for (const auto& rel : result.relationships) {
        EXPECT_GE(rel.attention_weight, 0.0f);
        EXPECT_LE(rel.attention_weight, 1.0f);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(RelationshipDiscoveryTest, QueryInCandidates) {
    auto pattern_ids = CreatePatternsWithFeatures(5);
    PatternID query = pattern_ids[0];

    // Include query in candidates (should be filtered out)
    std::vector<PatternID> candidates = pattern_ids;

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 3, context);

    EXPECT_EQ(result.query, query);
    // Should return 3 patterns (excluding query itself)
    EXPECT_EQ(result.relationships.size(), 3u);

    // Query should not appear in results
    for (const auto& rel : result.relationships) {
        EXPECT_NE(rel.pattern, query);
    }
}

TEST_F(RelationshipDiscoveryTest, SingleCandidate) {
    auto pattern_ids = CreatePatternsWithFeatures(2);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates{pattern_ids[1]};

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 5, context);

    EXPECT_EQ(result.relationships.size(), 1u);
    EXPECT_EQ(result.relationships[0].pattern, pattern_ids[1]);
}

TEST_F(RelationshipDiscoveryTest, ContextSensitiveDiscovery) {
    auto pattern_ids = CreatePatternsWithFeatures(5);
    PatternID query = pattern_ids[0];
    std::vector<PatternID> candidates(pattern_ids.begin() + 1, pattern_ids.end());

    ContextVector empty_context;
    ContextVector semantic_context = CreateSemanticContext();

    // Discovery with different contexts
    auto result_empty = self_attn_->DiscoverRelatedPatterns(query, candidates, 3, empty_context);
    auto result_semantic = self_attn_->DiscoverRelatedPatterns(query, candidates, 3, semantic_context);

    // Both should return 3 results
    EXPECT_EQ(result_empty.relationships.size(), 3u);
    EXPECT_EQ(result_semantic.relationships.size(), 3u);

    // Results should be sorted
    for (size_t i = 1; i < result_empty.relationships.size(); ++i) {
        EXPECT_GE(result_empty.relationships[i-1].attention_weight,
                  result_empty.relationships[i].attention_weight);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
