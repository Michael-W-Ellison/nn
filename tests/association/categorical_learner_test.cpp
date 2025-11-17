// File: tests/association/categorical_learner_test.cpp
#include "association/categorical_learner.hpp"
#include <gtest/gtest.h>
#include <random>

using namespace dpan;

// Helper function to create a feature vector
FeatureVector CreateFeatureVector(const std::vector<float>& values) {
    FeatureVector fv(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        fv[i] = values[i];
    }
    return fv;
}

// ============================================================================
// Construction Tests
// ============================================================================

TEST(CategoricalLearnerTest, DefaultConstruction) {
    CategoricalLearner learner;

    EXPECT_EQ(0u, learner.GetPatternCount());
    EXPECT_EQ(0u, learner.GetNumClusters());

    const auto& config = learner.GetConfig();
    EXPECT_EQ(5u, config.num_clusters);
    EXPECT_EQ(100u, config.max_iterations);
    EXPECT_FLOAT_EQ(0.001f, config.convergence_threshold);
    EXPECT_FALSE(config.auto_recompute);
}

TEST(CategoricalLearnerTest, ConfigConstruction) {
    CategoricalLearner::Config config;
    config.num_clusters = 3;
    config.max_iterations = 50;
    config.convergence_threshold = 0.01f;
    config.auto_recompute = true;

    CategoricalLearner learner(config);

    const auto& retrieved_config = learner.GetConfig();
    EXPECT_EQ(3u, retrieved_config.num_clusters);
    EXPECT_EQ(50u, retrieved_config.max_iterations);
    EXPECT_FLOAT_EQ(0.01f, retrieved_config.convergence_threshold);
    EXPECT_TRUE(retrieved_config.auto_recompute);
}

// ============================================================================
// Pattern Management Tests
// ============================================================================

TEST(CategoricalLearnerTest, AddPattern) {
    CategoricalLearner learner;
    PatternID p1 = PatternID::Generate();

    auto features = CreateFeatureVector({1.0f, 2.0f, 3.0f});
    learner.AddPattern(p1, features);

    EXPECT_EQ(1u, learner.GetPatternCount());
    EXPECT_TRUE(learner.HasPattern(p1));
}

TEST(CategoricalLearnerTest, AddMultiplePatterns) {
    CategoricalLearner learner;

    for (int i = 0; i < 10; ++i) {
        PatternID p = PatternID::Generate();
        auto features = CreateFeatureVector({static_cast<float>(i), 0.0f});
        learner.AddPattern(p, features);
    }

    EXPECT_EQ(10u, learner.GetPatternCount());
}

TEST(CategoricalLearnerTest, RemovePattern) {
    CategoricalLearner learner;
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    auto features = CreateFeatureVector({1.0f, 2.0f});
    learner.AddPattern(p1, features);
    learner.AddPattern(p2, features);

    EXPECT_EQ(2u, learner.GetPatternCount());

    learner.RemovePattern(p1);

    EXPECT_EQ(1u, learner.GetPatternCount());
    EXPECT_FALSE(learner.HasPattern(p1));
    EXPECT_TRUE(learner.HasPattern(p2));
}

TEST(CategoricalLearnerTest, GetFeatures) {
    CategoricalLearner learner;
    PatternID p1 = PatternID::Generate();

    auto features = CreateFeatureVector({1.0f, 2.0f, 3.0f});
    learner.AddPattern(p1, features);

    auto retrieved = learner.GetFeatures(p1);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(3u, retrieved->Dimension());
    EXPECT_FLOAT_EQ(1.0f, (*retrieved)[0]);
    EXPECT_FLOAT_EQ(2.0f, (*retrieved)[1]);
    EXPECT_FLOAT_EQ(3.0f, (*retrieved)[2]);
}

TEST(CategoricalLearnerTest, GetFeaturesNonExistent) {
    CategoricalLearner learner;
    PatternID p1 = PatternID::Generate();

    auto features = learner.GetFeatures(p1);
    EXPECT_FALSE(features.has_value());
}

// ============================================================================
// Clustering Tests
// ============================================================================

TEST(CategoricalLearnerTest, ComputeClustersInsufficientPatterns) {
    CategoricalLearner::Config config;
    config.num_clusters = 3;
    CategoricalLearner learner(config);

    // Add only 2 patterns (less than num_clusters)
    for (int i = 0; i < 2; ++i) {
        PatternID p = PatternID::Generate();
        auto features = CreateFeatureVector({static_cast<float>(i), 0.0f});
        learner.AddPattern(p, features);
    }

    bool success = learner.ComputeClusters();
    EXPECT_FALSE(success);
    EXPECT_EQ(0u, learner.GetNumClusters());
}

TEST(CategoricalLearnerTest, ComputeClustersBasic) {
    CategoricalLearner::Config config;
    config.num_clusters = 2;
    CategoricalLearner learner(config);

    // Add patterns in two distinct groups
    std::vector<PatternID> group1, group2;

    // Group 1: around (0, 0)
    for (int i = 0; i < 5; ++i) {
        PatternID p = PatternID::Generate();
        auto features = CreateFeatureVector({0.1f * i, 0.1f * i});
        learner.AddPattern(p, features);
        group1.push_back(p);
    }

    // Group 2: around (10, 10)
    for (int i = 0; i < 5; ++i) {
        PatternID p = PatternID::Generate();
        auto features = CreateFeatureVector({10.0f + 0.1f * i, 10.0f + 0.1f * i});
        learner.AddPattern(p, features);
        group2.push_back(p);
    }

    bool success = learner.ComputeClusters();
    EXPECT_TRUE(success);
    EXPECT_EQ(2u, learner.GetNumClusters());

    // All patterns should be assigned to clusters
    for (const auto& p : group1) {
        EXPECT_TRUE(learner.GetClusterID(p).has_value());
    }
    for (const auto& p : group2) {
        EXPECT_TRUE(learner.GetClusterID(p).has_value());
    }
}

TEST(CategoricalLearnerTest, ComputeClustersThreeGroups) {
    CategoricalLearner::Config config;
    config.num_clusters = 3;
    CategoricalLearner learner(config);

    // Create 3 well-separated clusters
    for (int cluster = 0; cluster < 3; ++cluster) {
        for (int i = 0; i < 5; ++i) {
            PatternID p = PatternID::Generate();
            float base = cluster * 10.0f;
            auto features = CreateFeatureVector({base + 0.1f * i, base + 0.1f * i});
            learner.AddPattern(p, features);
        }
    }

    bool success = learner.ComputeClusters();
    EXPECT_TRUE(success);
    EXPECT_EQ(3u, learner.GetNumClusters());
}

TEST(CategoricalLearnerTest, GetClusterInfo) {
    CategoricalLearner::Config config;
    config.num_clusters = 2;
    CategoricalLearner learner(config);

    // Add patterns
    for (int i = 0; i < 10; ++i) {
        PatternID p = PatternID::Generate();
        auto features = CreateFeatureVector({static_cast<float>(i), 0.0f});
        learner.AddPattern(p, features);
    }

    learner.ComputeClusters();

    auto info0 = learner.GetClusterInfo(0);
    ASSERT_TRUE(info0.has_value());
    EXPECT_EQ(0u, info0->cluster_id);
    EXPECT_GT(info0->members.size(), 0u);
    EXPECT_EQ(2u, info0->centroid.Dimension());  // 2D features

    auto info_invalid = learner.GetClusterInfo(10);
    EXPECT_FALSE(info_invalid.has_value());
}

TEST(CategoricalLearnerTest, GetAllClusters) {
    CategoricalLearner::Config config;
    config.num_clusters = 3;
    CategoricalLearner learner(config);

    // Add patterns
    for (int i = 0; i < 15; ++i) {
        PatternID p = PatternID::Generate();
        auto features = CreateFeatureVector({static_cast<float>(i), 0.0f});
        learner.AddPattern(p, features);
    }

    learner.ComputeClusters();

    auto clusters = learner.GetAllClusters();
    EXPECT_EQ(3u, clusters.size());

    size_t total_members = 0;
    for (const auto& cluster : clusters) {
        total_members += cluster.members.size();
    }
    EXPECT_EQ(15u, total_members);
}

TEST(CategoricalLearnerTest, ClearClusters) {
    CategoricalLearner::Config config;
    config.num_clusters = 2;
    CategoricalLearner learner(config);

    // Add patterns and compute clusters
    for (int i = 0; i < 10; ++i) {
        PatternID p = PatternID::Generate();
        auto features = CreateFeatureVector({static_cast<float>(i), 0.0f});
        learner.AddPattern(p, features);
    }

    learner.ComputeClusters();
    EXPECT_EQ(2u, learner.GetNumClusters());

    learner.ClearClusters();
    EXPECT_EQ(0u, learner.GetNumClusters());
    EXPECT_EQ(10u, learner.GetPatternCount());  // Patterns still there
}

// ============================================================================
// Categorical Queries Tests
// ============================================================================

TEST(CategoricalLearnerTest, AreCategoricallyRelatedSameCluster) {
    CategoricalLearner::Config config;
    config.num_clusters = 2;
    CategoricalLearner learner(config);

    // Add two patterns with very similar features (should cluster together)
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    learner.AddPattern(p1, CreateFeatureVector({1.0f, 1.0f}));
    learner.AddPattern(p2, CreateFeatureVector({1.1f, 1.1f}));

    // Add patterns far away to create a second cluster
    for (int i = 0; i < 5; ++i) {
        PatternID p = PatternID::Generate();
        learner.AddPattern(p, CreateFeatureVector({100.0f, 100.0f}));
    }

    learner.ComputeClusters();

    EXPECT_TRUE(learner.AreCategoricallyRelated(p1, p2));
}

TEST(CategoricalLearnerTest, AreCategoricallyRelatedDifferentClusters) {
    CategoricalLearner::Config config;
    config.num_clusters = 2;
    CategoricalLearner learner(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Add patterns in different locations
    learner.AddPattern(p1, CreateFeatureVector({0.0f, 0.0f}));
    learner.AddPattern(p2, CreateFeatureVector({100.0f, 100.0f}));

    // Add more patterns to ensure clustering works
    for (int i = 0; i < 5; ++i) {
        learner.AddPattern(PatternID::Generate(), CreateFeatureVector({0.1f * i, 0.1f * i}));
        learner.AddPattern(PatternID::Generate(), CreateFeatureVector({100.0f + 0.1f * i, 100.0f + 0.1f * i}));
    }

    learner.ComputeClusters();

    EXPECT_FALSE(learner.AreCategoricallyRelated(p1, p2));
}

TEST(CategoricalLearnerTest, GetClusterID) {
    CategoricalLearner::Config config;
    config.num_clusters = 2;
    CategoricalLearner learner(config);

    PatternID p1 = PatternID::Generate();
    learner.AddPattern(p1, CreateFeatureVector({1.0f, 2.0f}));

    // Before clustering
    EXPECT_FALSE(learner.GetClusterID(p1).has_value());

    // Add more patterns
    for (int i = 0; i < 5; ++i) {
        learner.AddPattern(PatternID::Generate(), CreateFeatureVector({static_cast<float>(i), 0.0f}));
    }

    learner.ComputeClusters();

    // After clustering
    auto cluster_id = learner.GetClusterID(p1);
    ASSERT_TRUE(cluster_id.has_value());
    EXPECT_LT(*cluster_id, 2u);
}

TEST(CategoricalLearnerTest, GetPatternCluster) {
    CategoricalLearner::Config config;
    config.num_clusters = 2;
    CategoricalLearner learner(config);

    PatternID p1 = PatternID::Generate();
    learner.AddPattern(p1, CreateFeatureVector({1.0f, 2.0f}));

    for (int i = 0; i < 5; ++i) {
        learner.AddPattern(PatternID::Generate(), CreateFeatureVector({static_cast<float>(i), 0.0f}));
    }

    learner.ComputeClusters();

    auto cluster_info = learner.GetPatternCluster(p1);
    ASSERT_TRUE(cluster_info.has_value());
    EXPECT_LT(cluster_info->cluster_id, 2u);
    EXPECT_GE(cluster_info->distance_to_centroid, 0.0f);
    EXPECT_GE(cluster_info->similarity_to_centroid, 0.0f);
    EXPECT_LE(cluster_info->similarity_to_centroid, 1.0f);
}

TEST(CategoricalLearnerTest, GetClusterMembers) {
    CategoricalLearner::Config config;
    config.num_clusters = 2;
    CategoricalLearner learner(config);

    std::vector<PatternID> group1;

    // Add a group of similar patterns
    for (int i = 0; i < 5; ++i) {
        PatternID p = PatternID::Generate();
        learner.AddPattern(p, CreateFeatureVector({0.1f * i, 0.1f * i}));
        group1.push_back(p);
    }

    // Add a far away group
    for (int i = 0; i < 5; ++i) {
        learner.AddPattern(PatternID::Generate(), CreateFeatureVector({100.0f + i, 100.0f + i}));
    }

    learner.ComputeClusters();

    // Check that group1 patterns are in the same cluster
    auto members = learner.GetClusterMembers(group1[0]);
    EXPECT_GE(members.size(), 3u);  // At least some of the group should be together

    // Members should not include the query pattern itself
    for (const auto& member : members) {
        EXPECT_NE(member, group1[0]);
    }
}

TEST(CategoricalLearnerTest, GetCategoriallySimilar) {
    CategoricalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // p1 and p2 are similar, p3 is different
    learner.AddPattern(p1, CreateFeatureVector({1.0f, 0.0f, 0.0f}));
    learner.AddPattern(p2, CreateFeatureVector({0.9f, 0.1f, 0.0f}));
    learner.AddPattern(p3, CreateFeatureVector({0.0f, 0.0f, 1.0f}));

    auto similar = learner.GetCategoriallyimilar(p1, 0.5f);

    // Should find p2 but not p3
    bool found_p2 = false;
    bool found_p3 = false;

    for (const auto& [pattern, sim] : similar) {
        if (pattern == p2) found_p2 = true;
        if (pattern == p3) found_p3 = true;
    }

    EXPECT_TRUE(found_p2);
    EXPECT_FALSE(found_p3);
}

TEST(CategoricalLearnerTest, GetCategoriallySimilarSorted) {
    CategoricalLearner learner;

    PatternID p1 = PatternID::Generate();

    // Add patterns with varying similarity to p1
    learner.AddPattern(p1, CreateFeatureVector({1.0f, 0.0f}));
    learner.AddPattern(PatternID::Generate(), CreateFeatureVector({0.9f, 0.1f}));  // High similarity
    learner.AddPattern(PatternID::Generate(), CreateFeatureVector({0.5f, 0.5f}));  // Medium
    learner.AddPattern(PatternID::Generate(), CreateFeatureVector({0.0f, 1.0f}));  // Low

    auto similar = learner.GetCategoriallyimilar(p1);

    // Should be sorted by similarity (descending)
    for (size_t i = 1; i < similar.size(); ++i) {
        EXPECT_GE(similar[i-1].second, similar[i].second);
    }
}

// ============================================================================
// Feature Similarity Tests
// ============================================================================

TEST(CategoricalLearnerTest, ComputeFeatureSimilarity) {
    CategoricalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    learner.AddPattern(p1, CreateFeatureVector({1.0f, 0.0f}));
    learner.AddPattern(p2, CreateFeatureVector({1.0f, 0.0f}));

    float similarity = learner.ComputeFeatureSimilarity(p1, p2);
    EXPECT_NEAR(1.0f, similarity, 0.01f);
}

TEST(CategoricalLearnerTest, ComputeFeatureSimilarityDifferent) {
    CategoricalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    learner.AddPattern(p1, CreateFeatureVector({1.0f, 0.0f}));
    learner.AddPattern(p2, CreateFeatureVector({0.0f, 1.0f}));

    float similarity = learner.ComputeFeatureSimilarity(p1, p2);
    EXPECT_NEAR(0.0f, similarity, 0.01f);
}

TEST(CategoricalLearnerTest, ComputeFeatureSimilarityNonExistent) {
    CategoricalLearner learner;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    float similarity = learner.ComputeFeatureSimilarity(p1, p2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(CategoricalLearnerTest, GetClusteringStats) {
    CategoricalLearner::Config config;
    config.num_clusters = 2;
    CategoricalLearner learner(config);

    // Add patterns
    for (int i = 0; i < 10; ++i) {
        PatternID p = PatternID::Generate();
        learner.AddPattern(p, CreateFeatureVector({static_cast<float>(i), 0.0f}));
    }

    learner.ComputeClusters();

    auto stats = learner.GetClusteringStats();
    EXPECT_EQ(10u, stats.num_patterns);
    EXPECT_EQ(2u, stats.num_clusters);
    EXPECT_EQ(0u, stats.num_unassigned);
    EXPECT_GT(stats.average_cluster_size, 0.0f);
    EXPECT_GE(stats.average_intra_cluster_similarity, 0.0f);
}

// ============================================================================
// Maintenance Tests
// ============================================================================

TEST(CategoricalLearnerTest, Clear) {
    CategoricalLearner::Config config;
    config.num_clusters = 2;
    CategoricalLearner learner(config);

    // Add patterns and compute clusters
    for (int i = 0; i < 10; ++i) {
        PatternID p = PatternID::Generate();
        learner.AddPattern(p, CreateFeatureVector({static_cast<float>(i), 0.0f}));
    }

    learner.ComputeClusters();

    EXPECT_EQ(10u, learner.GetPatternCount());
    EXPECT_EQ(2u, learner.GetNumClusters());

    learner.Clear();

    EXPECT_EQ(0u, learner.GetPatternCount());
    EXPECT_EQ(0u, learner.GetNumClusters());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(CategoricalLearnerTest, EmptyLearner) {
    CategoricalLearner learner;

    EXPECT_EQ(0u, learner.GetPatternCount());
    EXPECT_EQ(0u, learner.GetNumClusters());

    bool success = learner.ComputeClusters();
    EXPECT_FALSE(success);
}

TEST(CategoricalLearnerTest, SinglePattern) {
    CategoricalLearner::Config config;
    config.num_clusters = 1;
    CategoricalLearner learner(config);

    PatternID p1 = PatternID::Generate();
    learner.AddPattern(p1, CreateFeatureVector({1.0f, 2.0f}));

    bool success = learner.ComputeClusters();
    EXPECT_TRUE(success);

    auto cluster_id = learner.GetClusterID(p1);
    ASSERT_TRUE(cluster_id.has_value());
    EXPECT_EQ(0u, *cluster_id);
}

TEST(CategoricalLearnerTest, AllIdenticalFeatures) {
    CategoricalLearner::Config config;
    config.num_clusters = 2;
    CategoricalLearner learner(config);

    // Add multiple patterns with identical features
    auto features = CreateFeatureVector({1.0f, 2.0f, 3.0f});
    for (int i = 0; i < 5; ++i) {
        PatternID p = PatternID::Generate();
        learner.AddPattern(p, features);
    }

    bool success = learner.ComputeClusters();
    EXPECT_TRUE(success);

    // All patterns should be in the same cluster (likely cluster 0)
    auto first_cluster = learner.GetClusterID(learner.GetAllClusters()[0].members[0]);
    for (const auto& cluster : learner.GetAllClusters()) {
        for (const auto& member : cluster.members) {
            auto cluster_id = learner.GetClusterID(member);
            ASSERT_TRUE(cluster_id.has_value());
            // All should be assigned (some clusters may be empty)
        }
    }
}
