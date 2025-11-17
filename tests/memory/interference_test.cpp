// File: tests/memory/interference_test.cpp
#include "memory/interference.hpp"
#include "similarity/similarity_metric.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace dpan;

// ============================================================================
// Mock Similarity Metric for Testing
// ============================================================================

class MockSimilarityMetric : public SimilarityMetric {
public:
    void SetSimilarity(float similarity) {
        predefined_similarity_ = similarity;
    }

    float Compute(const PatternData& a, const PatternData& b) const override {
        return predefined_similarity_;
    }

    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override {
        return predefined_similarity_;
    }

    std::string GetName() const override {
        return "MockSimilarityMetric";
    }

private:
    float predefined_similarity_{0.5f};
};

// ============================================================================
// Test Fixtures
// ============================================================================

class InterferenceCalculatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        similarity_metric_ = std::make_shared<MockSimilarityMetric>();

        InterferenceCalculator::Config config;
        config.interference_factor = 0.1f;
        config.similarity_threshold = 0.5f;

        calculator_ = std::make_unique<InterferenceCalculator>(config, similarity_metric_);
    }

    FeatureVector CreateTestFeatures(size_t dim = 10, float value = 0.5f) {
        FeatureVector::StorageType data(dim, value);
        return FeatureVector(data);
    }

    std::shared_ptr<MockSimilarityMetric> similarity_metric_;
    std::unique_ptr<InterferenceCalculator> calculator_;
};

// ============================================================================
// Configuration Tests (3 tests)
// ============================================================================

TEST_F(InterferenceCalculatorTest, ValidConfiguration) {
    InterferenceCalculator::Config config;
    config.interference_factor = 0.15f;
    config.similarity_threshold = 0.6f;

    EXPECT_TRUE(config.IsValid());
    EXPECT_NO_THROW(InterferenceCalculator calc(config));
}

TEST_F(InterferenceCalculatorTest, InvalidInterferenceFactor) {
    InterferenceCalculator::Config config;

    // Negative factor
    config.interference_factor = -0.1f;
    EXPECT_FALSE(config.IsValid());

    // Factor > 1.0
    config.interference_factor = 1.5f;
    EXPECT_FALSE(config.IsValid());
}

TEST_F(InterferenceCalculatorTest, InvalidSimilarityThreshold) {
    InterferenceCalculator::Config config;

    // Negative threshold
    config.similarity_threshold = -0.1f;
    EXPECT_FALSE(config.IsValid());

    // Threshold > 1.0
    config.similarity_threshold = 1.5f;
    EXPECT_FALSE(config.IsValid());
}

// ============================================================================
// Basic Interference Calculation Tests (6 tests)
// ============================================================================

TEST_F(InterferenceCalculatorTest, CalculateBasicInterference) {
    FeatureVector target_features = CreateTestFeatures();
    FeatureVector source_features = CreateTestFeatures();

    // Set similarity to 0.8
    similarity_metric_->SetSimilarity(0.8f);

    float source_strength = 0.5f;

    // I = similarity × strength = 0.8 × 0.5 = 0.4
    float interference = calculator_->CalculateInterference(
        target_features, source_features, source_strength
    );

    EXPECT_NEAR(0.4f, interference, 0.01f);
}

TEST_F(InterferenceCalculatorTest, LowSimilarityNoInterference) {
    FeatureVector target_features = CreateTestFeatures();
    FeatureVector source_features = CreateTestFeatures();

    // Similarity below threshold (0.5)
    similarity_metric_->SetSimilarity(0.3f);

    float interference = calculator_->CalculateInterference(
        target_features, source_features, 0.8f
    );

    EXPECT_FLOAT_EQ(0.0f, interference);
}

TEST_F(InterferenceCalculatorTest, HighSimilarityHighInterference) {
    FeatureVector target_features = CreateTestFeatures();
    FeatureVector source_features = CreateTestFeatures();

    // Very high similarity
    similarity_metric_->SetSimilarity(0.95f);

    float source_strength = 0.9f;

    // I = 0.95 × 0.9 = 0.855
    float interference = calculator_->CalculateInterference(
        target_features, source_features, source_strength
    );

    EXPECT_NEAR(0.855f, interference, 0.01f);
}

TEST_F(InterferenceCalculatorTest, ZeroStrengthNoInterference) {
    FeatureVector target_features = CreateTestFeatures();
    FeatureVector source_features = CreateTestFeatures();

    similarity_metric_->SetSimilarity(0.9f);

    // Zero strength means no interference
    float interference = calculator_->CalculateInterference(
        target_features, source_features, 0.0f
    );

    EXPECT_FLOAT_EQ(0.0f, interference);
}

TEST_F(InterferenceCalculatorTest, InterferenceNeverExceedsOne) {
    FeatureVector target_features = CreateTestFeatures();
    FeatureVector source_features = CreateTestFeatures();

    similarity_metric_->SetSimilarity(1.0f);

    float interference = calculator_->CalculateInterference(
        target_features, source_features, 1.0f
    );

    EXPECT_LE(interference, 1.0f);
    EXPECT_GE(interference, 0.0f);
}

TEST_F(InterferenceCalculatorTest, NoSimilarityMetricNoInterference) {
    // Create calculator without similarity metric
    InterferenceCalculator calc;

    FeatureVector target_features = CreateTestFeatures();
    FeatureVector source_features = CreateTestFeatures();

    float interference = calc.CalculateInterference(
        target_features, source_features, 0.9f
    );

    EXPECT_FLOAT_EQ(0.0f, interference);
}

// ============================================================================
// Apply Interference Tests (5 tests)
// ============================================================================

TEST_F(InterferenceCalculatorTest, ApplyInterferenceReducesStrength) {
    float original_strength = 1.0f;
    float total_interference = 0.5f;

    // s' = s × (1 - α × I) = 1.0 × (1 - 0.1 × 0.5) = 0.95
    float new_strength = calculator_->ApplyInterference(
        original_strength, total_interference
    );

    EXPECT_NEAR(0.95f, new_strength, 0.01f);
    EXPECT_LT(new_strength, original_strength);
}

TEST_F(InterferenceCalculatorTest, ApplyInterferenceWithHighInterference) {
    float original_strength = 0.8f;
    float total_interference = 1.0f;  // Maximum interference

    // s' = 0.8 × (1 - 0.1 × 1.0) = 0.72
    float new_strength = calculator_->ApplyInterference(
        original_strength, total_interference
    );

    EXPECT_NEAR(0.72f, new_strength, 0.01f);
    EXPECT_GT(new_strength, 0.0f);  // Should not go to zero with α=0.1
}

TEST_F(InterferenceCalculatorTest, NoInterferenceNoReduction) {
    float original_strength = 0.7f;
    float zero_interference = 0.0f;

    float new_strength = calculator_->ApplyInterference(
        original_strength, zero_interference
    );

    EXPECT_FLOAT_EQ(original_strength, new_strength);
}

TEST_F(InterferenceCalculatorTest, ApplyInterferenceNeverExceedsOriginal) {
    float original_strength = 0.6f;

    for (float interference = 0.0f; interference <= 1.0f; interference += 0.1f) {
        float new_strength = calculator_->ApplyInterference(
            original_strength, interference
        );

        EXPECT_LE(new_strength, original_strength);
        EXPECT_GE(new_strength, 0.0f);
    }
}

TEST_F(InterferenceCalculatorTest, InterferenceFactorAffectsReduction) {
    InterferenceCalculator::Config low_config;
    low_config.interference_factor = 0.05f;  // Low factor
    InterferenceCalculator low_calc(low_config);

    InterferenceCalculator::Config high_config;
    high_config.interference_factor = 0.2f;  // High factor
    InterferenceCalculator high_calc(high_config);

    float original = 1.0f;
    float interference = 0.5f;

    float low_result = low_calc.ApplyInterference(original, interference);
    float high_result = high_calc.ApplyInterference(original, interference);

    // Higher factor = more reduction
    EXPECT_GT(low_result, high_result);
}

// ============================================================================
// Edge Cases and Boundary Tests (4 tests)
// ============================================================================

TEST_F(InterferenceCalculatorTest, InvalidStrengthValuesHandled) {
    FeatureVector target_features = CreateTestFeatures();
    FeatureVector source_features = CreateTestFeatures();

    similarity_metric_->SetSimilarity(0.8f);

    // Negative strength
    float result1 = calculator_->CalculateInterference(
        target_features, source_features, -0.5f
    );
    EXPECT_FLOAT_EQ(0.0f, result1);

    // Strength > 1.0
    float result2 = calculator_->CalculateInterference(
        target_features, source_features, 1.5f
    );
    EXPECT_FLOAT_EQ(0.0f, result2);
}

TEST_F(InterferenceCalculatorTest, TotalInterferenceClampedCorrectly) {
    // Test that total interference values are clamped

    float original = 0.8f;

    // Negative total interference
    float result1 = calculator_->ApplyInterference(original, -0.5f);
    EXPECT_FLOAT_EQ(original, result1);  // Should treat as 0

    // Total interference > 1.0 should be clamped
    float result2 = calculator_->ApplyInterference(original, 2.0f);
    EXPECT_LT(result2, original);
    EXPECT_GE(result2, 0.0f);
}

TEST_F(InterferenceCalculatorTest, SimilarityThresholdEnforced) {
    FeatureVector target_features = CreateTestFeatures();
    FeatureVector source_features = CreateTestFeatures();

    // Set similarity just below threshold
    similarity_metric_->SetSimilarity(0.49f);

    float interference = calculator_->CalculateInterference(
        target_features, source_features, 1.0f
    );

    EXPECT_FLOAT_EQ(0.0f, interference);  // Below threshold = no interference

    // Set similarity at threshold
    similarity_metric_->SetSimilarity(0.5f);

    interference = calculator_->CalculateInterference(
        target_features, source_features, 1.0f
    );

    EXPECT_GT(interference, 0.0f);  // At threshold = interference occurs
}

TEST_F(InterferenceCalculatorTest, ConfigurationCanBeUpdated) {
    InterferenceCalculator::Config new_config;
    new_config.interference_factor = 0.25f;
    new_config.similarity_threshold = 0.7f;

    calculator_->SetConfig(new_config);

    EXPECT_FLOAT_EQ(0.25f, calculator_->GetConfig().interference_factor);
    EXPECT_FLOAT_EQ(0.7f, calculator_->GetConfig().similarity_threshold);
}

// ============================================================================
// Integration Test (1 test)
// ============================================================================

TEST_F(InterferenceCalculatorTest, FullInterferenceWorkflow) {
    // Simulate realistic interference scenario
    FeatureVector pattern1 = CreateTestFeatures(10, 0.5f);
    FeatureVector pattern2 = CreateTestFeatures(10, 0.6f);
    FeatureVector pattern3 = CreateTestFeatures(10, 0.7f);

    // Pattern 1 is target
    // Patterns 2 and 3 interfere with it

    similarity_metric_->SetSimilarity(0.8f);

    float pattern2_strength = 0.7f;
    float pattern3_strength = 0.6f;

    // Calculate interference from each source
    float interference_from_2 = calculator_->CalculateInterference(
        pattern1, pattern2, pattern2_strength
    );

    float interference_from_3 = calculator_->CalculateInterference(
        pattern1, pattern3, pattern3_strength
    );

    // Total interference
    float total_interference = interference_from_2 + interference_from_3;
    total_interference = std::min(1.0f, total_interference);

    // Apply to pattern1's strength
    float pattern1_original_strength = 0.9f;
    float pattern1_new_strength = calculator_->ApplyInterference(
        pattern1_original_strength, total_interference
    );

    // Verify results
    EXPECT_GT(interference_from_2, 0.0f);
    EXPECT_GT(interference_from_3, 0.0f);
    EXPECT_LT(pattern1_new_strength, pattern1_original_strength);
    EXPECT_GT(pattern1_new_strength, 0.0f);
}
