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

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
