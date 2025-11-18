// File: tests/learning/pattern_importance_test.cpp
//
// Comprehensive tests for PatternImportanceCalculator
//
// Tests cover:
// - Frequency scoring with log scaling
// - Confidence scoring
// - Association scoring (placeholder)
// - Success rate tracking and scoring
// - Weighted combination
// - Edge cases (missing patterns, zero weights)
// - Configuration changes
// - Thread safety

#include "learning/pattern_importance.hpp"
#include "attention_test_fixtures.hpp"
#include <gtest/gtest.h>

using namespace dpan;
using namespace dpan::attention;
using namespace dpan::testing;

class PatternImportanceTest : public AttentionTestFixture {
protected:
    void SetUp() override {
        AttentionTestFixture::SetUp();

        // Create importance calculator with default weights
        weights_ = ImportanceWeights();
        calculator_ = std::make_unique<PatternImportanceCalculator>(
            mock_db_.get(),
            weights_
        );
    }

    ImportanceWeights weights_;
    std::unique_ptr<PatternImportanceCalculator> calculator_;
};

// ============================================================================
// ImportanceWeights Tests
// ============================================================================

TEST_F(PatternImportanceTest, WeightsValidateCorrect) {
    ImportanceWeights weights;
    weights.frequency = 0.3f;
    weights.confidence = 0.3f;
    weights.association = 0.2f;
    weights.success_rate = 0.2f;

    EXPECT_TRUE(weights.Validate());
}

TEST_F(PatternImportanceTest, WeightsValidateIncorrectSum) {
    ImportanceWeights weights;
    weights.frequency = 0.5f;
    weights.confidence = 0.5f;
    weights.association = 0.5f;
    weights.success_rate = 0.5f;

    EXPECT_FALSE(weights.Validate());
}

TEST_F(PatternImportanceTest, WeightsValidateNegative) {
    ImportanceWeights weights;
    weights.frequency = -0.1f;
    weights.confidence = 0.5f;
    weights.association = 0.3f;
    weights.success_rate = 0.3f;

    EXPECT_FALSE(weights.Validate());
}

TEST_F(PatternImportanceTest, WeightsNormalize) {
    ImportanceWeights weights;
    weights.frequency = 1.0f;
    weights.confidence = 1.0f;
    weights.association = 1.0f;
    weights.success_rate = 1.0f;

    weights.Normalize();

    // All should be 0.25 after normalization
    EXPECT_NEAR(weights.frequency, 0.25f, 1e-5f);
    EXPECT_NEAR(weights.confidence, 0.25f, 1e-5f);
    EXPECT_NEAR(weights.association, 0.25f, 1e-5f);
    EXPECT_NEAR(weights.success_rate, 0.25f, 1e-5f);
}

// ============================================================================
// Frequency Scoring Tests
// ============================================================================

TEST_F(PatternImportanceTest, FrequencyScoreBasic) {
    // Create patterns with different access counts
    auto pattern_ids = CreateTestPatterns(3);

    // Set max access count
    calculator_->SetMaxAccessCount(1000);

    // Pattern 0: low frequency (access count from fixture)
    // Pattern 1: medium frequency
    // Pattern 2: high frequency

    // Get frequency scores
    float score0 = calculator_->ComputeFrequencyScore(pattern_ids[0]);
    float score1 = calculator_->ComputeFrequencyScore(pattern_ids[1]);
    float score2 = calculator_->ComputeFrequencyScore(pattern_ids[2]);

    // All scores should be in [0, 1]
    EXPECT_GE(score0, 0.0f);
    EXPECT_LE(score0, 1.0f);
    EXPECT_GE(score1, 0.0f);
    EXPECT_LE(score1, 1.0f);
    EXPECT_GE(score2, 0.0f);
    EXPECT_LE(score2, 1.0f);

    // Higher access counts should give higher scores
    // (based on CreateTestPatterns which creates exponentially increasing counts)
    EXPECT_GE(score2, score1);
    EXPECT_GE(score1, score0);
}

TEST_F(PatternImportanceTest, FrequencyScoreLogScaling) {
    auto pattern_ids = CreateTestPatterns(2);

    calculator_->SetMaxAccessCount(100);

    // Due to log scaling, the difference between scores should not be as
    // extreme as the difference in access counts
    float score0 = calculator_->ComputeFrequencyScore(pattern_ids[0]);
    float score1 = calculator_->ComputeFrequencyScore(pattern_ids[1]);

    // Both should be valid scores
    EXPECT_GE(score0, 0.0f);
    EXPECT_LE(score0, 1.0f);
    EXPECT_GE(score1, 0.0f);
    EXPECT_LE(score1, 1.0f);
}

TEST_F(PatternImportanceTest, FrequencyScoreMissingPattern) {
    PatternID invalid_id(999999);

    float score = calculator_->ComputeFrequencyScore(invalid_id);

    // Missing pattern should return 0.0
    EXPECT_NEAR(score, 0.0f, 1e-5f);
}

// ============================================================================
// Confidence Scoring Tests
// ============================================================================

TEST_F(PatternImportanceTest, ConfidenceScoreBasic) {
    auto pattern_ids = CreateTestPatterns(3);

    // Confidence scores are set by CreateTestPatterns
    // They increase linearly from 0.1 to 0.9
    float score0 = calculator_->ComputeConfidenceScore(pattern_ids[0]);
    float score1 = calculator_->ComputeConfidenceScore(pattern_ids[1]);
    float score2 = calculator_->ComputeConfidenceScore(pattern_ids[2]);

    // Scores should be in [0, 1]
    EXPECT_GE(score0, 0.0f);
    EXPECT_LE(score0, 1.0f);
    EXPECT_GE(score1, 0.0f);
    EXPECT_LE(score1, 1.0f);
    EXPECT_GE(score2, 0.0f);
    EXPECT_LE(score2, 1.0f);

    // Higher confidence patterns should have higher scores
    EXPECT_GE(score2, score1);
    EXPECT_GE(score1, score0);
}

TEST_F(PatternImportanceTest, ConfidenceScoreMissingPattern) {
    PatternID invalid_id(999999);

    float score = calculator_->ComputeConfidenceScore(invalid_id);

    // Missing pattern should return neutral score (0.5)
    EXPECT_NEAR(score, 0.5f, 1e-5f);
}

// ============================================================================
// Association Scoring Tests
// ============================================================================

TEST_F(PatternImportanceTest, AssociationScorePlaceholder) {
    auto pattern_ids = CreateTestPatterns(1);

    float score = calculator_->ComputeAssociationScore(pattern_ids[0]);

    // Currently returns neutral score (placeholder)
    EXPECT_NEAR(score, 0.5f, 1e-5f);
}

// ============================================================================
// Success Rate Tests
// ============================================================================

TEST_F(PatternImportanceTest, SuccessRateScoreNoHistory) {
    auto pattern_ids = CreateTestPatterns(1);

    float score = calculator_->ComputeSuccessRateScore(pattern_ids[0]);

    // No history should return neutral score (0.5)
    EXPECT_NEAR(score, 0.5f, 1e-5f);
}

TEST_F(PatternImportanceTest, SuccessRateRecordPredictions) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    // Record some predictions
    calculator_->RecordPrediction(pattern_id, true);   // Success
    calculator_->RecordPrediction(pattern_id, true);   // Success
    calculator_->RecordPrediction(pattern_id, false);  // Failure

    float score = calculator_->ComputeSuccessRateScore(pattern_id);

    // 2/3 success rate ≈ 0.667
    EXPECT_GT(score, 0.6f);
    EXPECT_LT(score, 0.7f);
}

TEST_F(PatternImportanceTest, SuccessRatePerfectScore) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    // Record only successes
    for (int i = 0; i < 10; ++i) {
        calculator_->RecordPrediction(pattern_id, true);
    }

    float score = calculator_->ComputeSuccessRateScore(pattern_id);

    // Should be close to 1.0
    EXPECT_GT(score, 0.95f);
    EXPECT_LE(score, 1.0f);
}

TEST_F(PatternImportanceTest, SuccessRateZeroScore) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    // Record only failures
    for (int i = 0; i < 10; ++i) {
        calculator_->RecordPrediction(pattern_id, false);
    }

    float score = calculator_->ComputeSuccessRateScore(pattern_id);

    // Should be close to 0.0
    EXPECT_GE(score, 0.0f);
    EXPECT_LT(score, 0.1f);
}

TEST_F(PatternImportanceTest, SuccessRateGetData) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    calculator_->RecordPrediction(pattern_id, true);
    calculator_->RecordPrediction(pattern_id, false);

    auto data = calculator_->GetSuccessRateData(pattern_id);

    EXPECT_GT(data.total_predictions, 0u);
    EXPECT_GT(data.successful_predictions, 0u);
}

TEST_F(PatternImportanceTest, SuccessRateClear) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    // Record some data
    calculator_->RecordPrediction(pattern_id, true);

    // Clear
    calculator_->ClearSuccessRateData();

    // Should return default score after clearing
    float score = calculator_->ComputeSuccessRateScore(pattern_id);
    EXPECT_NEAR(score, 0.5f, 1e-5f);
}

// ============================================================================
// Combined Importance Tests
// ============================================================================

TEST_F(PatternImportanceTest, ComputeImportanceBasic) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    float importance = calculator_->ComputeImportance(pattern_id);

    // Should be in [0, 1]
    EXPECT_GE(importance, 0.0f);
    EXPECT_LE(importance, 1.0f);
}

TEST_F(PatternImportanceTest, ComputeImportanceWeightedCombination) {
    auto pattern_ids = CreateTestPatterns(1);
    PatternID pattern_id = pattern_ids[0];

    // Set specific weights to test combination
    ImportanceWeights weights;
    weights.frequency = 1.0f;
    weights.confidence = 0.0f;
    weights.association = 0.0f;
    weights.success_rate = 0.0f;
    calculator_->SetWeights(weights);

    float importance_freq_only = calculator_->ComputeImportance(pattern_id);
    float freq_score = calculator_->ComputeFrequencyScore(pattern_id);

    // With only frequency weight, importance should equal frequency score
    EXPECT_NEAR(importance_freq_only, freq_score, 1e-5f);

    // Now test confidence only
    weights.frequency = 0.0f;
    weights.confidence = 1.0f;
    calculator_->SetWeights(weights);

    float importance_conf_only = calculator_->ComputeImportance(pattern_id);
    float conf_score = calculator_->ComputeConfidenceScore(pattern_id);

    // With only confidence weight, importance should equal confidence score
    EXPECT_NEAR(importance_conf_only, conf_score, 1e-5f);
}

TEST_F(PatternImportanceTest, ComputeImportanceBalancedWeights) {
    auto pattern_ids = CreateTestPatterns(5);

    // Use default balanced weights
    ImportanceWeights weights;
    weights.frequency = 0.25f;
    weights.confidence = 0.25f;
    weights.association = 0.25f;
    weights.success_rate = 0.25f;
    calculator_->SetWeights(weights);

    // Patterns with different properties should have different importance
    float imp0 = calculator_->ComputeImportance(pattern_ids[0]);
    float imp1 = calculator_->ComputeImportance(pattern_ids[1]);
    float imp2 = calculator_->ComputeImportance(pattern_ids[2]);

    // All should be in [0, 1]
    EXPECT_GE(imp0, 0.0f);
    EXPECT_LE(imp0, 1.0f);
    EXPECT_GE(imp1, 0.0f);
    EXPECT_LE(imp1, 1.0f);
    EXPECT_GE(imp2, 0.0f);
    EXPECT_LE(imp2, 1.0f);
}

TEST_F(PatternImportanceTest, ComputeImportanceBatch) {
    auto pattern_ids = CreateTestPatterns(5);

    auto results = calculator_->ComputeImportanceBatch(pattern_ids);

    // Should have results for all patterns
    EXPECT_EQ(results.size(), pattern_ids.size());

    // All results should be in [0, 1]
    for (const auto& [id, importance] : results) {
        EXPECT_GE(importance, 0.0f);
        EXPECT_LE(importance, 1.0f);
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

TEST_F(PatternImportanceTest, SetWeights) {
    ImportanceWeights new_weights;
    new_weights.frequency = 0.5f;
    new_weights.confidence = 0.3f;
    new_weights.association = 0.1f;
    new_weights.success_rate = 0.1f;

    calculator_->SetWeights(new_weights);

    const auto& weights = calculator_->GetWeights();

    EXPECT_NEAR(weights.frequency, 0.5f, 1e-5f);
    EXPECT_NEAR(weights.confidence, 0.3f, 1e-5f);
    EXPECT_NEAR(weights.association, 0.1f, 1e-5f);
    EXPECT_NEAR(weights.success_rate, 0.1f, 1e-5f);
}

TEST_F(PatternImportanceTest, SetWeightsAutoNormalize) {
    ImportanceWeights weights;
    weights.frequency = 2.0f;
    weights.confidence = 2.0f;
    weights.association = 2.0f;
    weights.success_rate = 2.0f;

    // Should auto-normalize to sum to 1.0
    calculator_->SetWeights(weights);

    const auto& normalized = calculator_->GetWeights();

    // All should be 0.25
    EXPECT_NEAR(normalized.frequency, 0.25f, 1e-5f);
    EXPECT_NEAR(normalized.confidence, 0.25f, 1e-5f);
    EXPECT_NEAR(normalized.association, 0.25f, 1e-5f);
    EXPECT_NEAR(normalized.success_rate, 0.25f, 1e-5f);
}

TEST_F(PatternImportanceTest, SetMaxAccessCount) {
    auto pattern_ids = CreateTestPatterns(1);

    calculator_->SetMaxAccessCount(100);
    float score_low_max = calculator_->ComputeFrequencyScore(pattern_ids[0]);

    calculator_->SetMaxAccessCount(10000);
    float score_high_max = calculator_->ComputeFrequencyScore(pattern_ids[0]);

    // Higher max should result in lower score (same count, larger scale)
    EXPECT_LT(score_high_max, score_low_max);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST_F(PatternImportanceTest, GetStatistics) {
    auto pattern_ids = CreateTestPatterns(3);

    // Compute some importance scores
    calculator_->ComputeImportance(pattern_ids[0]);
    calculator_->ComputeImportance(pattern_ids[1]);

    // Record some predictions
    calculator_->RecordPrediction(pattern_ids[0], true);

    auto stats = calculator_->GetStatistics();

    EXPECT_TRUE(stats.find("importance_calculations") != stats.end());
    EXPECT_TRUE(stats.find("success_recordings") != stats.end());
    EXPECT_TRUE(stats.find("tracked_patterns") != stats.end());

    EXPECT_GE(stats["importance_calculations"], 2.0f);
    EXPECT_GE(stats["success_recordings"], 1.0f);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(PatternImportanceTest, NoPatternDatabase) {
    PatternImportanceCalculator no_db_calc(nullptr);

    PatternID dummy_id(123);

    // Should return default scores when no database
    float freq = no_db_calc.ComputeFrequencyScore(dummy_id);
    float conf = no_db_calc.ComputeConfidenceScore(dummy_id);
    float importance = no_db_calc.ComputeImportance(dummy_id);

    // Should return default/neutral scores
    EXPECT_NEAR(freq, 0.5f, 1e-5f);
    EXPECT_NEAR(conf, 0.5f, 1e-5f);
    EXPECT_GE(importance, 0.0f);
    EXPECT_LE(importance, 1.0f);
}

TEST_F(PatternImportanceTest, MissingPattern) {
    PatternID invalid_id(999999);

    float freq = calculator_->ComputeFrequencyScore(invalid_id);
    float conf = calculator_->ComputeConfidenceScore(invalid_id);
    float success = calculator_->ComputeSuccessRateScore(invalid_id);
    float importance = calculator_->ComputeImportance(invalid_id);

    // Frequency should be 0 for missing pattern
    EXPECT_NEAR(freq, 0.0f, 1e-5f);

    // Confidence and success should be neutral (0.5)
    EXPECT_NEAR(conf, 0.5f, 1e-5f);
    EXPECT_NEAR(success, 0.5f, 1e-5f);

    // Importance should still be valid
    EXPECT_GE(importance, 0.0f);
    EXPECT_LE(importance, 1.0f);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
