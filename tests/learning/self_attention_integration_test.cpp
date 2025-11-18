// File: tests/learning/self_attention_integration_test.cpp
//
// Integration tests for self-attention mechanism
//
// These tests demonstrate complete workflows combining:
// - Self-attention matrix computation
// - Relationship discovery
// - Comparison with explicit associations
// - Novel relationship identification
//
// Tests show realistic usage scenarios for the DPAN system.

#include "learning/self_attention.hpp"
#include "association/association_matrix.hpp"
#include "association/association_edge.hpp"
#include "attention_test_fixtures.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <iostream>

using namespace dpan;
using namespace dpan::attention;
using namespace dpan::testing;

class SelfAttentionIntegrationTest : public AttentionTestFixture {
protected:
    void SetUp() override {
        AttentionTestFixture::SetUp();

        // Create self-attention with realistic config
        SelfAttentionConfig config;
        config.temperature = 1.0f;
        config.mask_diagonal = false;  // Allow self-attention for similarity
        config.normalization = NormalizationMode::ROW_WISE;
        config.enable_caching = true;
        config.cache_size = 100;

        self_attn_ = std::make_unique<SelfAttention>(config);
        self_attn_->SetPatternDatabase(mock_db_.get());

        // Create association matrix
        association_matrix_ = std::make_unique<AssociationMatrix>();
        self_attn_->SetAssociationMatrix(association_matrix_.get());
    }

    /// Create patterns with controlled similarity
    /// Patterns in same group have high similarity, different groups have low similarity
    std::vector<std::vector<PatternID>> CreatePatternClusters(
        size_t num_clusters,
        size_t patterns_per_cluster) {

        std::vector<std::vector<PatternID>> clusters;

        for (size_t cluster = 0; cluster < num_clusters; ++cluster) {
            std::vector<PatternID> cluster_patterns;

            for (size_t i = 0; i < patterns_per_cluster; ++i) {
                PatternID id = PatternID::Generate();

                // Create feature vector
                // Patterns in same cluster have similar features
                FeatureVector features(20);
                for (size_t j = 0; j < 20; ++j) {
                    if (j == cluster) {
                        features[j] = 0.9f;  // High in cluster dimension
                    } else if (j == cluster + num_clusters) {
                        features[j] = 0.3f + (i * 0.1f);  // Varies within cluster
                    } else {
                        features[j] = 0.1f;  // Low in other dimensions
                    }
                }

                PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
                PatternNode node(id, data, PatternType::ATOMIC);
                node.SetConfidenceScore(0.8f);

                mock_db_->Store(node);
                cluster_patterns.push_back(id);
            }

            clusters.push_back(cluster_patterns);
        }

        return clusters;
    }

    /// Add explicit associations to create a known association graph
    void CreateAssociationGraph(const std::vector<std::vector<PatternID>>& clusters) {
        // Add intra-cluster associations (CATEGORICAL)
        for (const auto& cluster : clusters) {
            for (size_t i = 0; i < cluster.size(); ++i) {
                for (size_t j = i + 1; j < cluster.size(); ++j) {
                    AssociationEdge edge(cluster[i], cluster[j], AssociationType::CATEGORICAL);
                    edge.SetStrength(0.85f);
                    association_matrix_->AddAssociation(edge);

                    // Bidirectional
                    AssociationEdge reverse(cluster[j], cluster[i], AssociationType::CATEGORICAL);
                    reverse.SetStrength(0.85f);
                    association_matrix_->AddAssociation(reverse);
                }
            }
        }

        // Add some inter-cluster associations (CAUSAL, less common)
        if (clusters.size() >= 2) {
            // Cluster 0 -> Cluster 1 (causal relationship)
            AssociationEdge edge(clusters[0][0], clusters[1][0], AssociationType::CAUSAL);
            edge.SetStrength(0.70f);
            association_matrix_->AddAssociation(edge);
        }
    }

    std::unique_ptr<SelfAttention> self_attn_;
    std::unique_ptr<AssociationMatrix> association_matrix_;
};

// ============================================================================
// Complete Workflow Tests
// ============================================================================

TEST_F(SelfAttentionIntegrationTest, CompleteWorkflow_MatrixToDiscovery) {
    // Create 3 clusters of 4 patterns each
    auto clusters = CreatePatternClusters(3, 4);
    CreateAssociationGraph(clusters);

    // Flatten all patterns
    std::vector<PatternID> all_patterns;
    for (const auto& cluster : clusters) {
        all_patterns.insert(all_patterns.end(), cluster.begin(), cluster.end());
    }

    ContextVector context;

    // STEP 1: Compute self-attention matrix
    auto attention_matrix = self_attn_->ComputeAttentionMatrixDense(all_patterns, context);

    // Verify matrix structure
    ASSERT_EQ(attention_matrix.size(), 12u);  // 3 clusters * 4 patterns
    for (const auto& row : attention_matrix) {
        ASSERT_EQ(row.size(), 12u);

        // Each row should sum to 1.0 (row-wise normalization)
        float row_sum = 0.0f;
        for (float val : row) {
            row_sum += val;
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
    }

    // STEP 2: Discover relationships for a query pattern
    PatternID query = clusters[0][0];  // First pattern from first cluster
    std::vector<PatternID> candidates(all_patterns.begin() + 1, all_patterns.end());

    auto discovery_result = self_attn_->DiscoverRelatedPatterns(query, candidates, 5, context);

    // Verify discovery results
    EXPECT_EQ(discovery_result.query, query);
    EXPECT_EQ(discovery_result.relationships.size(), 5u);

    // Results should be sorted by attention weight
    for (size_t i = 1; i < discovery_result.relationships.size(); ++i) {
        EXPECT_GE(discovery_result.relationships[i-1].attention_weight,
                  discovery_result.relationships[i].attention_weight);
    }

    // STEP 3: Analyze novel vs confirmed relationships
    size_t novel_count = discovery_result.novel_count();
    size_t confirmed_count = discovery_result.confirmed_count();

    EXPECT_GT(novel_count, 0u);  // Should find some novel relationships
    EXPECT_GT(confirmed_count, 0u);  // Should confirm some explicit associations

    // STEP 4: Examine specific relationships
    auto novel = discovery_result.get_novel_relationships();
    auto confirmed = discovery_result.get_confirmed_relationships();

    // Novel relationships should have no explicit association
    for (const auto& rel : novel) {
        EXPECT_FALSE(rel.has_explicit_association);
        EXPECT_TRUE(rel.is_novel());
        EXPECT_GT(rel.attention_weight, 0.0f);
    }

    // Confirmed relationships should have explicit associations
    for (const auto& rel : confirmed) {
        EXPECT_TRUE(rel.has_explicit_association);
        EXPECT_TRUE(rel.is_confirmed());
        EXPECT_GT(rel.explicit_strength, 0.0f);
    }
}

TEST_F(SelfAttentionIntegrationTest, DiscoverClusterStructure) {
    // Create 3 clusters of 5 patterns each
    auto clusters = CreatePatternClusters(3, 5);
    // Don't add explicit associations - test pure discovery

    PatternID query = clusters[0][2];  // Middle pattern from first cluster

    // Get all other patterns
    std::vector<PatternID> all_patterns;
    for (const auto& cluster : clusters) {
        for (const auto& p : cluster) {
            if (p != query) {
                all_patterns.push_back(p);
            }
        }
    }

    ContextVector context;

    // Discover top 8 relationships
    auto result = self_attn_->DiscoverRelatedPatterns(query, all_patterns, 8, context);

    EXPECT_EQ(result.relationships.size(), 8u);

    // The top relationships should primarily be from the same cluster
    // (patterns with high similarity)
    size_t same_cluster_count = 0;
    for (size_t i = 0; i < std::min(size_t(4), result.relationships.size()); ++i) {
        const auto& rel = result.relationships[i];

        // Check if this pattern is in the same cluster as query
        bool in_same_cluster = false;
        for (const auto& p : clusters[0]) {
            if (p == rel.pattern) {
                in_same_cluster = true;
                break;
            }
        }

        if (in_same_cluster) {
            ++same_cluster_count;
        }
    }

    // Expect at least 3 of top 4 to be from same cluster
    EXPECT_GE(same_cluster_count, 3u);

    // All should be novel (no explicit associations)
    EXPECT_EQ(result.novel_count(), 8u);
    EXPECT_EQ(result.confirmed_count(), 0u);
}

TEST_F(SelfAttentionIntegrationTest, CompareImplicitVsExplicitAssociations) {
    // Create clusters
    auto clusters = CreatePatternClusters(2, 6);

    // Add partial explicit associations
    // Only associate first 3 patterns in each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = i + 1; j < 3; ++j) {
                AssociationEdge edge(clusters[c][i], clusters[c][j], AssociationType::CATEGORICAL);
                edge.SetStrength(0.80f);
                association_matrix_->AddAssociation(edge);
            }
        }
    }

    PatternID query = clusters[0][0];
    std::vector<PatternID> candidates;

    // Gather all candidates from first cluster
    for (size_t i = 1; i < clusters[0].size(); ++i) {
        candidates.push_back(clusters[0][i]);
    }

    ContextVector context;
    auto result = self_attn_->DiscoverRelatedPatterns(query, candidates, 5, context);

    // Should find both confirmed and novel relationships
    size_t confirmed = result.confirmed_count();
    size_t novel = result.novel_count();

    // We added explicit associations for patterns 1 and 2 with pattern 0
    EXPECT_EQ(confirmed, 2u);

    // Patterns 3, 4, 5 should be novel (no explicit associations)
    EXPECT_EQ(novel, 3u);

    // Verify specific patterns
    auto confirmed_rels = result.get_confirmed_relationships();
    EXPECT_EQ(confirmed_rels.size(), 2u);

    for (const auto& rel : confirmed_rels) {
        // Should be patterns 1 or 2
        EXPECT_TRUE(rel.pattern == clusters[0][1] || rel.pattern == clusters[0][2]);
        EXPECT_FLOAT_EQ(rel.explicit_strength, 0.80f);
        EXPECT_EQ(rel.explicit_type, AssociationType::CATEGORICAL);
    }

    auto novel_rels = result.get_novel_relationships();
    EXPECT_EQ(novel_rels.size(), 3u);

    for (const auto& rel : novel_rels) {
        // Should be patterns 3, 4, or 5
        EXPECT_TRUE(rel.pattern == clusters[0][3] ||
                    rel.pattern == clusters[0][4] ||
                    rel.pattern == clusters[0][5]);
        EXPECT_FALSE(rel.has_explicit_association);
    }
}

// ============================================================================
// Attention Analysis Tests
// ============================================================================

TEST_F(SelfAttentionIntegrationTest, FindMostAttendedPatterns_InClusters) {
    auto clusters = CreatePatternClusters(3, 5);

    std::vector<PatternID> all_patterns;
    for (const auto& cluster : clusters) {
        all_patterns.insert(all_patterns.end(), cluster.begin(), cluster.end());
    }

    ContextVector context;

    // Find most attended patterns
    auto top_attended = self_attn_->FindMostAttendedPatterns(all_patterns, 5, context);

    EXPECT_EQ(top_attended.size(), 5u);

    // Should be sorted by attention
    for (size_t i = 1; i < top_attended.size(); ++i) {
        EXPECT_GE(top_attended[i-1].second, top_attended[i].second);
    }

    // All attention scores should be valid
    for (const auto& [pattern, attention] : top_attended) {
        EXPECT_GT(attention, 0.0f);
        EXPECT_LE(attention, 1.0f);
    }
}

TEST_F(SelfAttentionIntegrationTest, ComputeAttentionEntropy_ShowsDistribution) {
    auto clusters = CreatePatternClusters(2, 4);

    std::vector<PatternID> all_patterns;
    for (const auto& cluster : clusters) {
        all_patterns.insert(all_patterns.end(), cluster.begin(), cluster.end());
    }

    ContextVector context;

    // Compute entropy for each pattern
    auto entropy_map = self_attn_->ComputeAttentionEntropy(all_patterns, context);

    EXPECT_EQ(entropy_map.size(), 8u);  // 2 clusters * 4 patterns

    // All entropy values should be non-negative
    for (const auto& [pattern, entropy] : entropy_map) {
        EXPECT_GE(entropy, 0.0f);
        // Entropy should be reasonable (not too high for focused attention)
        EXPECT_LT(entropy, 5.0f);  // log2(8) = 3, so max would be around 3
    }
}

// ============================================================================
// Configuration Impact Tests
// ============================================================================

TEST_F(SelfAttentionIntegrationTest, TemperatureAffectsDiscovery) {
    auto clusters = CreatePatternClusters(2, 5);

    PatternID query = clusters[0][0];
    std::vector<PatternID> candidates;
    for (const auto& cluster : clusters) {
        for (const auto& p : cluster) {
            if (p != query) {
                candidates.push_back(p);
            }
        }
    }

    ContextVector context;

    // Low temperature (sharper distribution)
    SelfAttentionConfig low_temp_config;
    low_temp_config.temperature = 0.1f;
    self_attn_->SetConfig(low_temp_config);
    auto low_temp_result = self_attn_->DiscoverRelatedPatterns(query, candidates, 5, context);

    // High temperature (more uniform distribution)
    SelfAttentionConfig high_temp_config;
    high_temp_config.temperature = 5.0f;
    self_attn_->SetConfig(high_temp_config);
    auto high_temp_result = self_attn_->DiscoverRelatedPatterns(query, candidates, 5, context);

    // Both should return 5 results
    EXPECT_EQ(low_temp_result.relationships.size(), 5u);
    EXPECT_EQ(high_temp_result.relationships.size(), 5u);

    // Low temperature should have more peaked distribution
    // (higher max attention weight)
    float low_temp_max = low_temp_result.relationships[0].attention_weight;
    float high_temp_max = high_temp_result.relationships[0].attention_weight;

    EXPECT_GT(low_temp_max, high_temp_max);

    // Low temperature should have more variance in weights
    float low_temp_range = low_temp_result.relationships[0].attention_weight -
                           low_temp_result.relationships[4].attention_weight;
    float high_temp_range = high_temp_result.relationships[0].attention_weight -
                            high_temp_result.relationships[4].attention_weight;

    EXPECT_GT(low_temp_range, high_temp_range);
}

TEST_F(SelfAttentionIntegrationTest, DiagonalMaskingAffectsResults) {
    auto clusters = CreatePatternClusters(2, 4);

    std::vector<PatternID> all_patterns;
    for (const auto& cluster : clusters) {
        all_patterns.insert(all_patterns.end(), cluster.begin(), cluster.end());
    }

    ContextVector context;

    // Without diagonal masking
    SelfAttentionConfig no_mask_config;
    no_mask_config.mask_diagonal = false;
    self_attn_->SetConfig(no_mask_config);
    auto matrix_no_mask = self_attn_->ComputeAttentionMatrixDense(all_patterns, context);

    // With diagonal masking
    SelfAttentionConfig mask_config;
    mask_config.mask_diagonal = true;
    self_attn_->SetConfig(mask_config);
    auto matrix_with_mask = self_attn_->ComputeAttentionMatrixDense(all_patterns, context);

    // Check diagonal values
    for (size_t i = 0; i < 8; ++i) {
        // Without masking, diagonal can have non-zero values
        // With masking, diagonal should be near zero
        EXPECT_LT(matrix_with_mask[i][i], 0.01f);
    }
}

// ============================================================================
// Realistic Usage Scenarios
// ============================================================================

TEST_F(SelfAttentionIntegrationTest, ScenarioBuildingRecommendationSystem) {
    // Simulate a pattern-based recommendation system
    // Patterns represent user behaviors or items
    auto user_clusters = CreatePatternClusters(3, 6);  // 3 user groups, 6 patterns each

    // User's current pattern
    PatternID current_user_pattern = user_clusters[0][0];

    // Add some known associations (e.g., from previous recommendations)
    // Add associations from current user pattern to a few others
    for (size_t i = 1; i < 3; ++i) {
        AssociationEdge edge(
            current_user_pattern,
            user_clusters[0][i],
            AssociationType::FUNCTIONAL
        );
        edge.SetStrength(0.75f);
        association_matrix_->AddAssociation(edge);
    }

    // All other patterns are candidates
    std::vector<PatternID> all_candidates;
    for (const auto& cluster : user_clusters) {
        for (const auto& p : cluster) {
            if (p != current_user_pattern) {
                all_candidates.push_back(p);
            }
        }
    }

    ContextVector context;

    // Discover top 10 recommendations
    auto recommendations = self_attn_->DiscoverRelatedPatterns(
        current_user_pattern,
        all_candidates,
        10,
        context
    );

    EXPECT_EQ(recommendations.relationships.size(), 10u);

    // Analyze recommendations
    size_t novel_recommendations = recommendations.novel_count();
    size_t confirmed_recommendations = recommendations.confirmed_count();

    // Should have both types
    EXPECT_GT(novel_recommendations, 0u);
    EXPECT_GT(confirmed_recommendations, 0u);

    std::cout << "\n=== Recommendation System Results ===" << std::endl;
    std::cout << "Current user pattern: " << current_user_pattern.value() << std::endl;
    std::cout << "Total recommendations: " << recommendations.relationships.size() << std::endl;
    std::cout << "Novel recommendations: " << novel_recommendations << std::endl;
    std::cout << "Confirmed recommendations: " << confirmed_recommendations << std::endl;

    std::cout << "\nTop 5 recommendations:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), recommendations.relationships.size()); ++i) {
        const auto& rec = recommendations.relationships[i];
        std::cout << (i+1) << ". Pattern " << rec.pattern.value()
                  << " (attention: " << rec.attention_weight
                  << ", " << (rec.is_novel() ? "NOVEL" : "CONFIRMED") << ")" << std::endl;
    }
}

TEST_F(SelfAttentionIntegrationTest, ScenarioAnomalyDetection) {
    // Create normal pattern clusters
    auto normal_clusters = CreatePatternClusters(2, 8);

    // Create anomalous patterns (different features)
    std::vector<PatternID> anomalies;
    for (size_t i = 0; i < 3; ++i) {
        PatternID id = PatternID::Generate();

        // Anomalous features (different from normal patterns)
        FeatureVector features(20);
        for (size_t j = 0; j < 20; ++j) {
            features[j] = (j % 2 == 0) ? 0.9f : 0.1f;  // Alternating pattern
        }

        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        PatternNode node(id, data, PatternType::ATOMIC);
        mock_db_->Store(node);
        anomalies.push_back(id);
    }

    // Mix all patterns
    std::vector<PatternID> all_patterns;
    for (const auto& cluster : normal_clusters) {
        all_patterns.insert(all_patterns.end(), cluster.begin(), cluster.end());
    }
    all_patterns.insert(all_patterns.end(), anomalies.begin(), anomalies.end());

    ContextVector context;

    // For an anomalous pattern, relationships should be weaker
    PatternID anomalous_query = anomalies[0];
    std::vector<PatternID> candidates(all_patterns.begin() + 1, all_patterns.end());

    auto anomaly_result = self_attn_->DiscoverRelatedPatterns(
        anomalous_query,
        candidates,
        5,
        context
    );

    // For a normal pattern, relationships should be stronger
    PatternID normal_query = normal_clusters[0][0];
    std::vector<PatternID> normal_candidates;
    for (const auto& p : all_patterns) {
        if (p != normal_query) {
            normal_candidates.push_back(p);
        }
    }

    auto normal_result = self_attn_->DiscoverRelatedPatterns(
        normal_query,
        normal_candidates,
        5,
        context
    );

    // Compare attention weights
    float anomaly_avg_attention = 0.0f;
    for (const auto& rel : anomaly_result.relationships) {
        anomaly_avg_attention += rel.attention_weight;
    }
    anomaly_avg_attention /= anomaly_result.relationships.size();

    float normal_avg_attention = 0.0f;
    for (const auto& rel : normal_result.relationships) {
        normal_avg_attention += rel.attention_weight;
    }
    normal_avg_attention /= normal_result.relationships.size();

    std::cout << "\n=== Anomaly Detection Results ===" << std::endl;
    std::cout << "Anomalous pattern avg attention: " << anomaly_avg_attention << std::endl;
    std::cout << "Normal pattern avg attention: " << normal_avg_attention << std::endl;

    // Normal patterns should have higher average attention
    // (stronger connections to other patterns)
    EXPECT_GT(normal_avg_attention, anomaly_avg_attention);
}

// ============================================================================
// Performance and Caching Tests
// ============================================================================

TEST_F(SelfAttentionIntegrationTest, CachingImprovesPerformance) {
    auto clusters = CreatePatternClusters(2, 6);

    std::vector<PatternID> all_patterns;
    for (const auto& cluster : clusters) {
        all_patterns.insert(all_patterns.end(), cluster.begin(), cluster.end());
    }

    ContextVector context;

    // Enable caching
    SelfAttentionConfig cache_config;
    cache_config.enable_caching = true;
    cache_config.cache_size = 100;
    self_attn_->SetConfig(cache_config);

    // Clear cache to start fresh
    self_attn_->ClearCache();

    // First computation (cache miss) - use sparse version which implements caching
    auto stats_before = self_attn_->GetStatistics();
    auto matrix1 = self_attn_->ComputeAttentionMatrix(all_patterns, context);
    auto stats_after_first = self_attn_->GetStatistics();

    // Verify cache miss
    EXPECT_GT(stats_after_first["cache_misses"], stats_before["cache_misses"]);

    // Second computation (should hit cache)
    auto matrix2 = self_attn_->ComputeAttentionMatrix(all_patterns, context);
    auto stats_after_second = self_attn_->GetStatistics();

    // Verify cache hit
    EXPECT_GT(stats_after_second["cache_hits"], stats_after_first["cache_hits"]);

    // Matrices should be identical
    EXPECT_EQ(matrix1.size(), matrix2.size());
    for (const auto& entry : matrix1) {
        ASSERT_TRUE(matrix2.count(entry.first) > 0);
        EXPECT_FLOAT_EQ(matrix1[entry.first], matrix2[entry.first]);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
