// File: tests/similarity/similarity_metric_test.cpp
#include "similarity/similarity_metric.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace dpan {
namespace {

// ============================================================================
// Mock Similarity Metrics for Testing
// ============================================================================

/// Simple cosine similarity metric for testing
class CosineSimilarityMetric : public SimilarityMetric {
public:
    float Compute(const PatternData& a, const PatternData& b) const override {
        // Extract features and compute
        FeatureVector fa = a.GetFeatures();
        FeatureVector fb = b.GetFeatures();
        return ComputeFromFeatures(fa, fb);
    }

    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override {
        return a.CosineSimilarity(b);
    }

    std::string GetName() const override { return "Cosine"; }
    bool IsSymmetric() const override { return true; }
};

/// Euclidean distance-based similarity metric for testing
class EuclideanSimilarityMetric : public SimilarityMetric {
public:
    float Compute(const PatternData& a, const PatternData& b) const override {
        FeatureVector fa = a.GetFeatures();
        FeatureVector fb = b.GetFeatures();
        return ComputeFromFeatures(fa, fb);
    }

    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override {
        // Convert distance to similarity: similarity = 1 / (1 + distance)
        float distance = a.EuclideanDistance(b);
        return 1.0f / (1.0f + distance);
    }

    std::string GetName() const override { return "Euclidean"; }
    bool IsSymmetric() const override { return true; }
};

/// Constant similarity metric for testing
class ConstantMetric : public SimilarityMetric {
public:
    explicit ConstantMetric(float value) : value_(value) {}

    float Compute(const PatternData&, const PatternData&) const override {
        return value_;
    }

    float ComputeFromFeatures(const FeatureVector&, const FeatureVector&) const override {
        return value_;
    }

    std::string GetName() const override { return "Constant"; }

private:
    float value_;
};

// ============================================================================
// Helper Functions
// ============================================================================

PatternData CreateTestPattern(const std::vector<float>& values) {
    FeatureVector features(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        features[i] = values[i];
    }
    return PatternData::FromFeatures(features, DataModality::NUMERIC);
}

// ============================================================================
// Basic Metric Tests
// ============================================================================

TEST(SimilarityMetricTest, CosineSimilarityWorks) {
    CosineSimilarityMetric metric;

    PatternData a = CreateTestPattern({1.0f, 0.0f, 0.0f});
    PatternData b = CreateTestPattern({1.0f, 0.0f, 0.0f});
    PatternData c = CreateTestPattern({0.0f, 1.0f, 0.0f});

    // Identical patterns should have similarity 1.0
    float sim_aa = metric.Compute(a, a);
    EXPECT_FLOAT_EQ(1.0f, sim_aa);

    // Same direction should have similarity 1.0
    float sim_ab = metric.Compute(a, b);
    EXPECT_FLOAT_EQ(1.0f, sim_ab);

    // Perpendicular vectors should have similarity 0.0
    float sim_ac = metric.Compute(a, c);
    EXPECT_FLOAT_EQ(0.0f, sim_ac);
}

TEST(SimilarityMetricTest, EuclideanSimilarityWorks) {
    EuclideanSimilarityMetric metric;

    PatternData a = CreateTestPattern({1.0f, 0.0f});
    PatternData b = CreateTestPattern({1.0f, 0.0f});
    PatternData c = CreateTestPattern({5.0f, 0.0f});

    // Identical patterns should have similarity 1.0
    float sim_aa = metric.Compute(a, a);
    EXPECT_FLOAT_EQ(1.0f, sim_aa);

    // Same pattern should have similarity 1.0
    float sim_ab = metric.Compute(a, b);
    EXPECT_FLOAT_EQ(1.0f, sim_ab);

    // Distant patterns should have lower similarity
    float sim_ac = metric.Compute(a, c);
    EXPECT_LT(sim_ac, 1.0f);
    EXPECT_GT(sim_ac, 0.0f);
}

TEST(SimilarityMetricTest, MetricIsSymmetric) {
    CosineSimilarityMetric metric;

    PatternData a = CreateTestPattern({1.0f, 2.0f, 3.0f});
    PatternData b = CreateTestPattern({4.0f, 5.0f, 6.0f});

    EXPECT_TRUE(metric.IsSymmetric());

    float sim_ab = metric.Compute(a, b);
    float sim_ba = metric.Compute(b, a);

    EXPECT_FLOAT_EQ(sim_ab, sim_ba);
}

TEST(SimilarityMetricTest, BatchComputationWorks) {
    CosineSimilarityMetric metric;

    PatternData query = CreateTestPattern({1.0f, 0.0f, 0.0f});

    std::vector<PatternData> candidates = {
        CreateTestPattern({1.0f, 0.0f, 0.0f}),  // sim = 1.0
        CreateTestPattern({0.0f, 1.0f, 0.0f}),  // sim = 0.0
        CreateTestPattern({0.707f, 0.707f, 0.0f})  // sim ≈ 0.707
    };

    auto results = metric.ComputeBatch(query, candidates);

    ASSERT_EQ(3u, results.size());
    EXPECT_FLOAT_EQ(1.0f, results[0]);
    EXPECT_FLOAT_EQ(0.0f, results[1]);
    EXPECT_NEAR(0.707f, results[2], 0.01f);
}

TEST(SimilarityMetricTest, FeatureVectorBatchWorks) {
    CosineSimilarityMetric metric;

    FeatureVector query(3);
    query[0] = 1.0f;
    query[1] = 0.0f;
    query[2] = 0.0f;

    std::vector<FeatureVector> candidates;
    FeatureVector v1(3);
    v1[0] = 1.0f;
    candidates.push_back(v1);

    FeatureVector v2(3);
    v2[1] = 1.0f;
    candidates.push_back(v2);

    auto results = metric.ComputeBatchFromFeatures(query, candidates);

    ASSERT_EQ(2u, results.size());
    EXPECT_FLOAT_EQ(1.0f, results[0]);
    EXPECT_FLOAT_EQ(0.0f, results[1]);
}

// ============================================================================
// CompositeMetric Tests
// ============================================================================

TEST(CompositeMetricTest, EmptyCompositeReturnsZero) {
    CompositeMetric composite;

    PatternData a = CreateTestPattern({1.0f, 0.0f});
    PatternData b = CreateTestPattern({0.0f, 1.0f});

    float similarity = composite.Compute(a, b);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(CompositeMetricTest, SingleMetricWorks) {
    CompositeMetric composite;
    auto cosine = std::make_shared<CosineSimilarityMetric>();

    composite.AddMetric(cosine, 1.0f);

    PatternData a = CreateTestPattern({1.0f, 0.0f, 0.0f});
    PatternData b = CreateTestPattern({1.0f, 0.0f, 0.0f});

    float similarity = composite.Compute(a, b);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(CompositeMetricTest, WeightedAverageWorks) {
    CompositeMetric composite;

    auto constant1 = std::make_shared<ConstantMetric>(1.0f);
    auto constant0 = std::make_shared<ConstantMetric>(0.0f);

    // Add with equal weights
    composite.AddMetric(constant1, 1.0f);
    composite.AddMetric(constant0, 1.0f);

    PatternData a = CreateTestPattern({1.0f});
    PatternData b = CreateTestPattern({2.0f});

    // Should be average of 1.0 and 0.0 = 0.5
    float similarity = composite.Compute(a, b);
    EXPECT_FLOAT_EQ(0.5f, similarity);
}

TEST(CompositeMetricTest, UnequalWeightsWork) {
    CompositeMetric composite;

    auto constant1 = std::make_shared<ConstantMetric>(1.0f);
    auto constant0 = std::make_shared<ConstantMetric>(0.0f);

    // 75% weight to constant1, 25% to constant0
    composite.AddMetric(constant1, 3.0f);
    composite.AddMetric(constant0, 1.0f);

    PatternData a = CreateTestPattern({1.0f});
    PatternData b = CreateTestPattern({2.0f});

    // Should be 0.75 * 1.0 + 0.25 * 0.0 = 0.75
    float similarity = composite.Compute(a, b);
    EXPECT_FLOAT_EQ(0.75f, similarity);
}

TEST(CompositeMetricTest, GetMetricCountWorks) {
    CompositeMetric composite;

    EXPECT_EQ(0u, composite.GetMetricCount());

    composite.AddMetric(std::make_shared<CosineSimilarityMetric>(), 1.0f);
    EXPECT_EQ(1u, composite.GetMetricCount());

    composite.AddMetric(std::make_shared<EuclideanSimilarityMetric>(), 1.0f);
    EXPECT_EQ(2u, composite.GetMetricCount());
}

TEST(CompositeMetricTest, ClearRemovesAllMetrics) {
    CompositeMetric composite;

    composite.AddMetric(std::make_shared<CosineSimilarityMetric>(), 1.0f);
    composite.AddMetric(std::make_shared<EuclideanSimilarityMetric>(), 1.0f);

    EXPECT_EQ(2u, composite.GetMetricCount());

    composite.Clear();

    EXPECT_EQ(0u, composite.GetMetricCount());
}

TEST(CompositeMetricTest, NullMetricIsIgnored) {
    CompositeMetric composite;

    composite.AddMetric(nullptr, 1.0f);

    EXPECT_EQ(0u, composite.GetMetricCount());
}

TEST(CompositeMetricTest, NegativeWeightClampedToZero) {
    CompositeMetric composite;

    auto constant1 = std::make_shared<ConstantMetric>(1.0f);
    auto constant0 = std::make_shared<ConstantMetric>(0.0f);

    composite.AddMetric(constant1, 1.0f);
    composite.AddMetric(constant0, -1.0f);  // Negative weight

    PatternData a = CreateTestPattern({1.0f});
    PatternData b = CreateTestPattern({2.0f});

    // constant0 with negative weight should be treated as 0 weight
    // Result should be 1.0 (only constant1 contributes)
    float similarity = composite.Compute(a, b);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(CompositeMetricTest, AllZeroWeightsUsesUniform) {
    CompositeMetric composite;

    auto constant1 = std::make_shared<ConstantMetric>(1.0f);
    auto constant0 = std::make_shared<ConstantMetric>(0.0f);

    composite.AddMetric(constant1, 0.0f);
    composite.AddMetric(constant0, 0.0f);

    PatternData a = CreateTestPattern({1.0f});
    PatternData b = CreateTestPattern({2.0f});

    // Should use uniform weights: 0.5 each
    float similarity = composite.Compute(a, b);
    EXPECT_FLOAT_EQ(0.5f, similarity);
}

TEST(CompositeMetricTest, IsSymmetricWhenAllMetricsSymmetric) {
    CompositeMetric composite;

    auto cosine = std::make_shared<CosineSimilarityMetric>();
    auto euclidean = std::make_shared<EuclideanSimilarityMetric>();

    composite.AddMetric(cosine, 1.0f);
    composite.AddMetric(euclidean, 1.0f);

    EXPECT_TRUE(composite.IsSymmetric());
}

TEST(CompositeMetricTest, ComputeFromFeaturesWorks) {
    CompositeMetric composite;

    auto cosine = std::make_shared<CosineSimilarityMetric>();
    composite.AddMetric(cosine, 1.0f);

    FeatureVector a(3);
    a[0] = 1.0f;

    FeatureVector b(3);
    b[0] = 1.0f;

    float similarity = composite.ComputeFromFeatures(a, b);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(CompositeMetricTest, BatchComputationWorks) {
    CompositeMetric composite;

    auto constant1 = std::make_shared<ConstantMetric>(1.0f);
    auto constant0 = std::make_shared<ConstantMetric>(0.0f);

    composite.AddMetric(constant1, 1.0f);
    composite.AddMetric(constant0, 1.0f);

    PatternData query = CreateTestPattern({1.0f});
    std::vector<PatternData> candidates = {
        CreateTestPattern({2.0f}),
        CreateTestPattern({3.0f}),
        CreateTestPattern({4.0f})
    };

    auto results = composite.ComputeBatch(query, candidates);

    ASSERT_EQ(3u, results.size());
    // All should be 0.5 (average of 1.0 and 0.0)
    for (float result : results) {
        EXPECT_FLOAT_EQ(0.5f, result);
    }
}

TEST(CompositeMetricTest, GetNameReturnsComposite) {
    CompositeMetric composite;
    EXPECT_EQ("Composite", composite.GetName());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(SimilarityMetricTest, RealWorldScenario) {
    // Create a composite metric combining cosine and euclidean
    CompositeMetric composite;

    auto cosine = std::make_shared<CosineSimilarityMetric>();
    auto euclidean = std::make_shared<EuclideanSimilarityMetric>();

    // 60% cosine, 40% euclidean
    composite.AddMetric(cosine, 0.6f);
    composite.AddMetric(euclidean, 0.4f);

    // Create test patterns
    PatternData p1 = CreateTestPattern({1.0f, 2.0f, 3.0f});
    PatternData p2 = CreateTestPattern({1.1f, 2.1f, 3.1f});  // Very similar
    PatternData p3 = CreateTestPattern({10.0f, 20.0f, 30.0f});  // Different magnitude, same direction

    // p1 vs p2 should have high similarity
    float sim_12 = composite.Compute(p1, p2);
    EXPECT_GT(sim_12, 0.9f);

    // p1 vs p3 should have high cosine (same direction) but lower euclidean
    float sim_13 = composite.Compute(p1, p3);
    EXPECT_GT(sim_13, 0.5f);
    EXPECT_LT(sim_13, sim_12);
}

} // namespace
} // namespace dpan
