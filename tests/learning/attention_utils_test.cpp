// File: tests/learning/attention_utils_test.cpp
//
// Comprehensive tests for attention utility functions
//
// Tests cover:
// - Normal operation
// - Edge cases (empty, zero, NaN, inf)
// - Numerical stability
// - Mathematical correctness

#include "learning/attention_utils.hpp"
#include "core/types.hpp"
#include "core/pattern_node.hpp"
#include "core/pattern_data.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

using namespace dpan;
using namespace dpan::attention;

// Test fixture for attention utilities
class AttentionUtilsTest : public ::testing::Test {
protected:
    const float kTolerance = 1e-5f;
};

// ============================================================================
// Softmax Tests
// ============================================================================

TEST_F(AttentionUtilsTest, SoftmaxBasic) {
    std::vector<float> scores = {1.0f, 2.0f, 3.0f};
    auto weights = Softmax(scores);

    // Verify weights sum to 1.0
    float sum = 0.0f;
    for (float w : weights) {
        sum += w;
    }
    EXPECT_NEAR(sum, 1.0f, kTolerance);

    // Verify weights are sorted (higher score = higher weight)
    EXPECT_GT(weights[2], weights[1]);
    EXPECT_GT(weights[1], weights[0]);

    // Verify all weights are in valid range
    for (float w : weights) {
        EXPECT_GE(w, 0.0f);
        EXPECT_LE(w, 1.0f);
    }
}

TEST_F(AttentionUtilsTest, SoftmaxUniformScores) {
    std::vector<float> scores = {1.0f, 1.0f, 1.0f, 1.0f};
    auto weights = Softmax(scores);

    // All weights should be equal (uniform distribution)
    float expected = 0.25f;
    for (float w : weights) {
        EXPECT_NEAR(w, expected, kTolerance);
    }
}

TEST_F(AttentionUtilsTest, SoftmaxEmptyInput) {
    std::vector<float> scores;
    auto weights = Softmax(scores);

    EXPECT_TRUE(weights.empty());
}

TEST_F(AttentionUtilsTest, SoftmaxSingleElement) {
    std::vector<float> scores = {5.0f};
    auto weights = Softmax(scores);

    ASSERT_EQ(weights.size(), 1u);
    EXPECT_NEAR(weights[0], 1.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, SoftmaxHighTemperature) {
    std::vector<float> scores = {1.0f, 2.0f, 3.0f};

    auto weights_low = Softmax(scores, 0.5f);   // Sharper
    auto weights_high = Softmax(scores, 2.0f);  // Softer

    // High temperature should create more uniform distribution
    float variance_low = 0.0f, variance_high = 0.0f;
    float mean = 1.0f / 3.0f;

    for (size_t i = 0; i < 3; ++i) {
        variance_low += std::pow(weights_low[i] - mean, 2);
        variance_high += std::pow(weights_high[i] - mean, 2);
    }

    EXPECT_GT(variance_low, variance_high)
        << "Lower temperature should have higher variance (more peaked)";
}

TEST_F(AttentionUtilsTest, SoftmaxNumericalStability) {
    // Very large scores that would overflow without max-subtraction
    std::vector<float> scores = {1000.0f, 1001.0f, 1002.0f};
    auto weights = Softmax(scores);

    // Should not produce NaN or inf
    for (float w : weights) {
        EXPECT_TRUE(std::isfinite(w));
    }

    // Should still sum to 1.0
    float sum = 0.0f;
    for (float w : weights) {
        sum += w;
    }
    EXPECT_NEAR(sum, 1.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, SoftmaxWithNaN) {
    std::vector<float> scores = {1.0f, NAN, 3.0f};
    auto weights = Softmax(scores);

    // Should fallback to uniform distribution or handle gracefully
    EXPECT_EQ(weights.size(), 3u);

    for (float w : weights) {
        EXPECT_TRUE(std::isfinite(w));
    }
}

TEST_F(AttentionUtilsTest, SoftmaxWithInfinity) {
    std::vector<float> scores = {1.0f, INFINITY, 3.0f};
    auto weights = Softmax(scores);

    EXPECT_EQ(weights.size(), 3u);

    for (float w : weights) {
        EXPECT_TRUE(std::isfinite(w));
    }
}

TEST_F(AttentionUtilsTest, SoftmaxMapVersion) {
    std::map<PatternID, float> scores;
    scores[PatternID(1)] = 1.0f;
    scores[PatternID(2)] = 2.0f;
    scores[PatternID(3)] = 3.0f;

    auto weights = Softmax(scores);

    // Verify size preserved
    EXPECT_EQ(weights.size(), 3u);

    // Verify sum to 1.0
    float sum = 0.0f;
    for (const auto& [id, weight] : weights) {
        sum += weight;
    }
    EXPECT_NEAR(sum, 1.0f, kTolerance);

    // Verify ordering
    EXPECT_GT(weights[PatternID(3)], weights[PatternID(2)]);
    EXPECT_GT(weights[PatternID(2)], weights[PatternID(1)]);
}

// ============================================================================
// Dot Product Tests
// ============================================================================

TEST_F(AttentionUtilsTest, DotProductBasic) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    float dot = DotProduct(a, b);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_NEAR(dot, 32.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, DotProductOrthogonal) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};

    float dot = DotProduct(a, b);

    EXPECT_NEAR(dot, 0.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, DotProductIdentical) {
    std::vector<float> a = {3.0f, 4.0f};

    float dot = DotProduct(a, a);

    // 3*3 + 4*4 = 9 + 16 = 25
    EXPECT_NEAR(dot, 25.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, DotProductEmpty) {
    std::vector<float> a;
    std::vector<float> b = {1.0f, 2.0f};

    float dot = DotProduct(a, b);

    EXPECT_NEAR(dot, 0.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, DotProductDifferentSizes) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f};

    float dot = DotProduct(a, b);

    // Should use minimum length: 1*4 + 2*5 = 14
    EXPECT_NEAR(dot, 14.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, ScaledDotProductBasic) {
    std::vector<float> query = {1.0f, 2.0f, 3.0f};
    std::vector<float> key = {4.0f, 5.0f, 6.0f};

    float scaled = ScaledDotProduct(query, key, true);
    float unscaled = ScaledDotProduct(query, key, false);

    // Unscaled should equal regular dot product
    EXPECT_NEAR(unscaled, 32.0f, kTolerance);

    // Scaled should be divided by sqrt(3)
    EXPECT_NEAR(scaled, 32.0f / std::sqrt(3.0f), kTolerance);
}

TEST_F(AttentionUtilsTest, ScaledDotProductPreventsSaturation) {
    // Large vectors would cause saturation without scaling
    std::vector<float> query(100, 1.0f);
    std::vector<float> key(100, 1.0f);

    float scaled = ScaledDotProduct(query, key, true);
    float unscaled = ScaledDotProduct(query, key, false);

    EXPECT_NEAR(unscaled, 100.0f, kTolerance);
    EXPECT_NEAR(scaled, 100.0f / 10.0f, kTolerance);  // sqrt(100) = 10
}

// ============================================================================
// Cosine Similarity Tests
// ============================================================================

TEST_F(AttentionUtilsTest, CosineSimilarityIdentical) {
    std::vector<float> a = {3.0f, 4.0f};

    float sim = CosineSimilarity(a, a);

    EXPECT_NEAR(sim, 1.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, CosineSimilarityOrthogonal) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};

    float sim = CosineSimilarity(a, b);

    EXPECT_NEAR(sim, 0.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, CosineSimilarityOpposite) {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {-1.0f, 0.0f};

    float sim = CosineSimilarity(a, b);

    EXPECT_NEAR(sim, -1.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, CosineSimilarityScaleInvariant) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {2.0f, 4.0f, 6.0f};  // 2x scaling

    float sim = CosineSimilarity(a, b);

    // Should be 1.0 (same direction, different magnitude)
    EXPECT_NEAR(sim, 1.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, CosineSimilarityZeroVector) {
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {0.0f, 0.0f};

    float sim = CosineSimilarity(a, b);

    EXPECT_NEAR(sim, 0.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, CosineSimilarityEmpty) {
    std::vector<float> a;
    std::vector<float> b = {1.0f, 2.0f};

    float sim = CosineSimilarity(a, b);

    EXPECT_NEAR(sim, 0.0f, kTolerance);
}

// ============================================================================
// L2 Norm Tests
// ============================================================================

TEST_F(AttentionUtilsTest, L2NormBasic) {
    std::vector<float> vec = {3.0f, 4.0f};

    float norm = L2Norm(vec);

    // sqrt(9 + 16) = sqrt(25) = 5
    EXPECT_NEAR(norm, 5.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, L2NormUnitVector) {
    std::vector<float> vec = {1.0f, 0.0f, 0.0f};

    float norm = L2Norm(vec);

    EXPECT_NEAR(norm, 1.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, L2NormZeroVector) {
    std::vector<float> vec = {0.0f, 0.0f, 0.0f};

    float norm = L2Norm(vec);

    EXPECT_NEAR(norm, 0.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, L2NormEmpty) {
    std::vector<float> vec;

    float norm = L2Norm(vec);

    EXPECT_NEAR(norm, 0.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, NormalizeL2Basic) {
    std::vector<float> vec = {3.0f, 4.0f};

    auto normalized = NormalizeL2(vec);

    ASSERT_EQ(normalized.size(), 2u);
    EXPECT_NEAR(normalized[0], 0.6f, kTolerance);
    EXPECT_NEAR(normalized[1], 0.8f, kTolerance);

    // Verify unit length
    float norm = L2Norm(normalized);
    EXPECT_NEAR(norm, 1.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, NormalizeL2ZeroVector) {
    std::vector<float> vec = {0.0f, 0.0f};

    auto normalized = NormalizeL2(vec);

    // Should return zero vector unchanged
    ASSERT_EQ(normalized.size(), 2u);
    EXPECT_NEAR(normalized[0], 0.0f, kTolerance);
    EXPECT_NEAR(normalized[1], 0.0f, kTolerance);
}

// ============================================================================
// Score Combination Tests
// ============================================================================

TEST_F(AttentionUtilsTest, CombineScoresBasic) {
    float score_a = 0.8f;
    float score_b = 0.6f;
    float weight_a = 0.4f;
    float weight_b = 0.6f;

    float combined = CombineScores(score_a, score_b, weight_a, weight_b);

    // 0.4 * 0.8 + 0.6 * 0.6 = 0.32 + 0.36 = 0.68
    EXPECT_NEAR(combined, 0.68f, kTolerance);
}

TEST_F(AttentionUtilsTest, CombineScoresEqualWeights) {
    float score_a = 0.8f;
    float score_b = 0.6f;

    float combined = CombineScores(score_a, score_b, 0.5f, 0.5f);

    // Should be average: (0.8 + 0.6) / 2 = 0.7
    EXPECT_NEAR(combined, 0.7f, kTolerance);
}

TEST_F(AttentionUtilsTest, CombineScoresPureA) {
    float score_a = 0.8f;
    float score_b = 0.6f;

    float combined = CombineScores(score_a, score_b, 1.0f, 0.0f);

    EXPECT_NEAR(combined, score_a, kTolerance);
}

// ============================================================================
// Utility Function Tests
// ============================================================================

TEST_F(AttentionUtilsTest, ClampInRange) {
    EXPECT_NEAR(Clamp(0.5f, 0.0f, 1.0f), 0.5f, kTolerance);
}

TEST_F(AttentionUtilsTest, ClampBelowMin) {
    EXPECT_NEAR(Clamp(-0.5f, 0.0f, 1.0f), 0.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, ClampAboveMax) {
    EXPECT_NEAR(Clamp(1.5f, 0.0f, 1.0f), 1.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, ApplyTemperatureBasic) {
    std::vector<float> scores = {2.0f, 4.0f, 6.0f};

    auto scaled = ApplyTemperature(scores, 2.0f);

    ASSERT_EQ(scaled.size(), 3u);
    EXPECT_NEAR(scaled[0], 1.0f, kTolerance);
    EXPECT_NEAR(scaled[1], 2.0f, kTolerance);
    EXPECT_NEAR(scaled[2], 3.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, ApplyTemperatureInvalid) {
    std::vector<float> scores = {1.0f, 2.0f};

    auto scaled = ApplyTemperature(scores, 0.0f);  // Invalid temperature

    // Should return unchanged
    EXPECT_EQ(scaled, scores);
}

TEST_F(AttentionUtilsTest, IsValidFinite) {
    EXPECT_TRUE(IsValid(0.0f));
    EXPECT_TRUE(IsValid(1.0f));
    EXPECT_TRUE(IsValid(-1.0f));
    EXPECT_TRUE(IsValid(1e10f));
}

TEST_F(AttentionUtilsTest, IsValidNaN) {
    EXPECT_FALSE(IsValid(NAN));
}

TEST_F(AttentionUtilsTest, IsValidInfinity) {
    EXPECT_FALSE(IsValid(INFINITY));
    EXPECT_FALSE(IsValid(-INFINITY));
}

TEST_F(AttentionUtilsTest, SafeDivideNormal) {
    float result = SafeDivide(10.0f, 2.0f, 0.0f);

    EXPECT_NEAR(result, 5.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, SafeDivideByZero) {
    float result = SafeDivide(10.0f, 0.0f, 99.0f);

    EXPECT_NEAR(result, 99.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, SafeDivideOverflow) {
    float huge = std::numeric_limits<float>::max();
    float result = SafeDivide(huge, 0.0001f, 100.0f);

    // Result would overflow, should return fallback
    EXPECT_NEAR(result, 100.0f, kTolerance);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(AttentionUtilsTest, SoftmaxDotProductPipeline) {
    // Simulate attention computation pipeline
    std::vector<float> query = {1.0f, 2.0f, 3.0f};
    std::vector<float> key1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> key2 = {3.0f, 2.0f, 1.0f};
    std::vector<float> key3 = {0.0f, 1.0f, 0.0f};

    // Compute scores
    std::vector<float> scores;
    scores.push_back(ScaledDotProduct(query, key1));
    scores.push_back(ScaledDotProduct(query, key2));
    scores.push_back(ScaledDotProduct(query, key3));

    // Normalize with softmax
    auto weights = Softmax(scores);

    // Verify valid probability distribution
    ASSERT_EQ(weights.size(), 3u);

    float sum = 0.0f;
    for (float w : weights) {
        EXPECT_GE(w, 0.0f);
        EXPECT_LE(w, 1.0f);
        sum += w;
    }
    EXPECT_NEAR(sum, 1.0f, kTolerance);

    // First key should have highest weight (most similar to query)
    EXPECT_GT(weights[0], weights[1]);
    EXPECT_GT(weights[0], weights[2]);
}

// ============================================================================
// Feature Extraction Tests
// ============================================================================

TEST_F(AttentionUtilsTest, GetFeatureDimensionBasic) {
    size_t base_dim = 128;

    FeatureExtractionConfig config;
    config.include_confidence = true;
    config.include_access_count = true;
    config.include_age = false;
    config.include_type = false;

    size_t total = GetFeatureDimension(base_dim, config);

    // 128 + confidence(1) + access_count(1) = 130
    EXPECT_EQ(total, 130u);
}

TEST_F(AttentionUtilsTest, GetFeatureDimensionAllFeatures) {
    size_t base_dim = 64;

    FeatureExtractionConfig config;
    config.include_confidence = true;
    config.include_access_count = true;
    config.include_age = true;
    config.include_type = true;

    size_t total = GetFeatureDimension(base_dim, config);

    // 64 + confidence(1) + access(1) + age(1) + type(3) = 70
    EXPECT_EQ(total, 70u);
}

TEST_F(AttentionUtilsTest, GetFeatureDimensionNoMetadata) {
    size_t base_dim = 100;

    FeatureExtractionConfig config;
    config.include_confidence = false;
    config.include_access_count = false;
    config.include_age = false;
    config.include_type = false;

    size_t total = GetFeatureDimension(base_dim, config);

    // Only base features
    EXPECT_EQ(total, 100u);
}

TEST_F(AttentionUtilsTest, ExtractFeaturesBasic) {
    // Create a simple pattern node with known properties
    PatternID id = PatternID::Generate();
    PatternData data = PatternData::FromFeatures(
        FeatureVector(std::vector<float>{1.0f, 2.0f, 3.0f}),
        DataModality::NUMERIC
    );

    PatternNode node(id, data, PatternType::ATOMIC);
    node.SetConfidenceScore(0.75f);
    node.IncrementAccessCount(100);

    FeatureExtractionConfig config;
    config.include_confidence = true;
    config.include_access_count = true;
    config.include_age = false;
    config.include_type = false;
    config.max_access_count = 1000;

    auto features = ExtractFeatures(node, config);

    // Should have: base(3) + confidence(1) + access(1) = 5
    ASSERT_EQ(features.size(), 5u);

    // Check base features
    EXPECT_NEAR(features[0], 1.0f, kTolerance);
    EXPECT_NEAR(features[1], 2.0f, kTolerance);
    EXPECT_NEAR(features[2], 3.0f, kTolerance);

    // Check confidence
    EXPECT_NEAR(features[3], 0.75f, kTolerance);

    // Check normalized access count: 100/1000 = 0.1
    EXPECT_NEAR(features[4], 0.1f, kTolerance);
}

TEST_F(AttentionUtilsTest, ExtractFeaturesWithType) {
    PatternID id = PatternID::Generate();
    PatternData data = PatternData::FromFeatures(
        FeatureVector(std::vector<float>{1.0f, 2.0f}),
        DataModality::NUMERIC
    );

    // Test ATOMIC type
    PatternNode atomic_node(id, data, PatternType::ATOMIC);
    atomic_node.SetConfidenceScore(0.5f);

    FeatureExtractionConfig config;
    config.include_confidence = true;
    config.include_access_count = false;
    config.include_age = false;
    config.include_type = true;

    auto features = ExtractFeatures(atomic_node, config);

    // base(2) + confidence(1) + type(3) = 6
    ASSERT_EQ(features.size(), 6u);

    // Check one-hot encoding for ATOMIC
    EXPECT_NEAR(features[3], 1.0f, kTolerance);  // ATOMIC = 1
    EXPECT_NEAR(features[4], 0.0f, kTolerance);  // COMPOSITE = 0
    EXPECT_NEAR(features[5], 0.0f, kTolerance);  // META = 0
}

TEST_F(AttentionUtilsTest, ExtractFeaturesCompositeType) {
    PatternID id = PatternID::Generate();
    PatternData data = PatternData::FromFeatures(
        FeatureVector(std::vector<float>{1.0f}),
        DataModality::NUMERIC
    );

    PatternNode composite_node(id, data, PatternType::COMPOSITE);

    FeatureExtractionConfig config;
    config.include_confidence = false;
    config.include_access_count = false;
    config.include_age = false;
    config.include_type = true;

    auto features = ExtractFeatures(composite_node, config);

    // base(1) + type(3) = 4
    ASSERT_EQ(features.size(), 4u);

    // Check one-hot encoding for COMPOSITE
    EXPECT_NEAR(features[1], 0.0f, kTolerance);  // ATOMIC = 0
    EXPECT_NEAR(features[2], 1.0f, kTolerance);  // COMPOSITE = 1
    EXPECT_NEAR(features[3], 0.0f, kTolerance);  // META = 0
}

TEST_F(AttentionUtilsTest, ExtractFeaturesAccessCountClamping) {
    PatternID id = PatternID::Generate();
    PatternData data = PatternData::FromFeatures(
        FeatureVector(std::vector<float>{1.0f}),
        DataModality::NUMERIC
    );

    PatternNode node(id, data, PatternType::ATOMIC);
    node.IncrementAccessCount(20000);  // Exceeds max_access_count

    FeatureExtractionConfig config;
    config.include_confidence = false;
    config.include_access_count = true;
    config.max_access_count = 10000;

    auto features = ExtractFeatures(node, config);

    // base(1) + access(1) = 2
    ASSERT_EQ(features.size(), 2u);

    // Should be clamped to 1.0
    EXPECT_NEAR(features[1], 1.0f, kTolerance);
}

TEST_F(AttentionUtilsTest, ExtractFeaturesConfidenceClamping) {
    PatternID id = PatternID::Generate();
    PatternData data = PatternData::FromFeatures(
        FeatureVector(std::vector<float>{1.0f}),
        DataModality::NUMERIC
    );

    PatternNode node(id, data, PatternType::ATOMIC);
    // Manually set confidence beyond normal range (shouldn't happen, but test robustness)
    node.SetConfidenceScore(1.5f);

    FeatureExtractionConfig config;
    config.include_confidence = true;
    config.include_access_count = false;

    auto features = ExtractFeatures(node, config);

    // Should be clamped to 1.0
    EXPECT_LE(features[1], 1.0f);
}

TEST_F(AttentionUtilsTest, ExtractFeaturesNoMetadata) {
    PatternID id = PatternID::Generate();
    std::vector<float> base_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    PatternData data = PatternData::FromFeatures(
        FeatureVector(base_data),
        DataModality::NUMERIC
    );

    PatternNode node(id, data, PatternType::ATOMIC);
    node.SetConfidenceScore(0.9f);
    node.IncrementAccessCount(500);

    FeatureExtractionConfig config;
    config.include_confidence = false;
    config.include_access_count = false;
    config.include_age = false;
    config.include_type = false;

    auto features = ExtractFeatures(node, config);

    // Only base features
    ASSERT_EQ(features.size(), 5u);

    // All metadata should be excluded
    for (size_t i = 0; i < features.size(); ++i) {
        EXPECT_NEAR(features[i], base_data[i], kTolerance);
    }
}

TEST_F(AttentionUtilsTest, ExtractFeaturesConsistentDimensions) {
    // Create two different patterns
    PatternID id1 = PatternID::Generate();
    PatternID id2 = PatternID::Generate();

    PatternData data1 = PatternData::FromFeatures(
        FeatureVector(std::vector<float>{1.0f, 2.0f, 3.0f}),
        DataModality::NUMERIC
    );

    PatternData data2 = PatternData::FromFeatures(
        FeatureVector(std::vector<float>{4.0f, 5.0f, 6.0f}),
        DataModality::NUMERIC
    );

    PatternNode node1(id1, data1, PatternType::ATOMIC);
    PatternNode node2(id2, data2, PatternType::COMPOSITE);

    node1.SetConfidenceScore(0.3f);
    node2.SetConfidenceScore(0.8f);

    FeatureExtractionConfig config;
    config.include_confidence = true;
    config.include_access_count = true;
    config.include_type = true;

    auto features1 = ExtractFeatures(node1, config);
    auto features2 = ExtractFeatures(node2, config);

    // Both should have same dimensionality
    EXPECT_EQ(features1.size(), features2.size());

    // base(3) + confidence(1) + access(1) + type(3) = 8
    EXPECT_EQ(features1.size(), 8u);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
