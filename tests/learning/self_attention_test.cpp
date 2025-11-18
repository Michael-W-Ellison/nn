// File: tests/learning/self_attention_test.cpp
//
// Unit tests for SelfAttention

#include "learning/self_attention.hpp"
#include "attention_test_fixtures.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <cmath>

using namespace dpan;
using namespace dpan::attention;
using namespace dpan::testing;

class SelfAttentionTest : public AttentionTestFixture {
protected:
    void SetUp() override {
        AttentionTestFixture::SetUp();

        // Create self-attention with default config
        SelfAttentionConfig config;
        self_attn_ = std::make_unique<SelfAttention>(config);
        self_attn_->SetPatternDatabase(mock_db_.get());
    }

    /// Create patterns with varying feature vectors for testing similarity
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

    std::unique_ptr<SelfAttention> self_attn_;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(SelfAttentionTest, DefaultConfiguration) {
    const auto& config = self_attn_->GetConfig();

    EXPECT_FLOAT_EQ(config.temperature, 1.0f);
    EXPECT_FALSE(config.mask_diagonal);
    EXPECT_EQ(config.normalization, NormalizationMode::ROW_WISE);
    EXPECT_FLOAT_EQ(config.attention_threshold, 0.0f);
    EXPECT_FALSE(config.enable_caching);
}

TEST_F(SelfAttentionTest, SetConfiguration) {
    SelfAttentionConfig new_config;
    new_config.temperature = 0.5f;
    new_config.mask_diagonal = true;
    new_config.normalization = NormalizationMode::COLUMN_WISE;

    self_attn_->SetConfig(new_config);

    const auto& config = self_attn_->GetConfig();
    EXPECT_FLOAT_EQ(config.temperature, 0.5f);
    EXPECT_TRUE(config.mask_diagonal);
    EXPECT_EQ(config.normalization, NormalizationMode::COLUMN_WISE);
}

TEST_F(SelfAttentionTest, InvalidConfigThrows) {
    SelfAttentionConfig invalid_config;
    invalid_config.temperature = 0.0f;  // Invalid

    EXPECT_THROW(self_attn_->SetConfig(invalid_config), std::invalid_argument);
}

// ============================================================================
// Attention Matrix Computation Tests
// ============================================================================

TEST_F(SelfAttentionTest, ComputeAttentionMatrixBasic) {
    auto pattern_ids = CreateTestPatterns(3);

    ContextVector context;
    auto matrix = self_attn_->ComputeAttentionMatrix(pattern_ids, context);

    // Should have N×N entries (but some may be zero/filtered)
    EXPECT_GT(matrix.size(), 0u);

    // Verify all entries are valid probabilities
    for (const auto& [pair, weight] : matrix) {
        EXPECT_GE(weight, 0.0f);
        EXPECT_LE(weight, 1.0f);
    }
}

TEST_F(SelfAttentionTest, ComputeAttentionMatrixDense) {
    auto pattern_ids = CreateTestPatterns(4);

    ContextVector context;
    auto matrix = self_attn_->ComputeAttentionMatrixDense(pattern_ids, context);

    ASSERT_EQ(matrix.size(), 4u);
    for (const auto& row : matrix) {
        ASSERT_EQ(row.size(), 4u);
    }

    // Verify each row sums to 1.0 (row-wise normalization)
    for (size_t i = 0; i < 4; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < 4; ++j) {
            row_sum += matrix[i][j];
            EXPECT_GE(matrix[i][j], 0.0f);
            EXPECT_LE(matrix[i][j], 1.0f);
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f);
    }
}

TEST_F(SelfAttentionTest, ComputeAttentionMatrixEmpty) {
    std::vector<PatternID> empty_patterns;

    ContextVector context;
    auto matrix = self_attn_->ComputeAttentionMatrix(empty_patterns, context);

    EXPECT_TRUE(matrix.empty());
}

TEST_F(SelfAttentionTest, ComputeAttentionMatrixSinglePattern) {
    auto pattern_ids = CreateTestPatterns(1);

    ContextVector context;
    auto matrix = self_attn_->ComputeAttentionMatrixDense(pattern_ids, context);

    ASSERT_EQ(matrix.size(), 1u);
    ASSERT_EQ(matrix[0].size(), 1u);

    // Single pattern should attend to itself with weight 1.0
    EXPECT_NEAR(matrix[0][0], 1.0f, 1e-5f);
}

TEST_F(SelfAttentionTest, GetQueryAttention) {
    auto pattern_ids = CreateTestPatterns(4);
    PatternID query = pattern_ids[1];

    ContextVector context;
    auto attention = self_attn_->GetQueryAttention(query, pattern_ids, context);

    // Should have attention to all patterns (or some if sparse)
    EXPECT_GT(attention.size(), 0u);

    // Weights should sum to 1.0
    float sum = 0.0f;
    for (const auto& [_, weight] : attention) {
        sum += weight;
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================================
// Diagonal Masking Tests
// ============================================================================

TEST_F(SelfAttentionTest, DiagonalMasking) {
    SelfAttentionConfig config;
    config.mask_diagonal = true;
    self_attn_->SetConfig(config);

    auto pattern_ids = CreateTestPatterns(3);

    ContextVector context;
    auto matrix = self_attn_->ComputeAttentionMatrixDense(pattern_ids, context);

    ASSERT_EQ(matrix.size(), 3u);

    // Diagonal should be zero (or very small)
    for (size_t i = 0; i < 3; ++i) {
        EXPECT_LT(matrix[i][i], 0.01f);  // Should be near zero
    }

    // Rows should still sum to 1.0
    for (size_t i = 0; i < 3; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < 3; ++j) {
            row_sum += matrix[i][j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f);
    }
}

TEST_F(SelfAttentionTest, NoDiagonalMasking) {
    SelfAttentionConfig config;
    config.mask_diagonal = false;
    self_attn_->SetConfig(config);

    auto pattern_ids = CreateTestPatterns(3);

    ContextVector context;
    auto matrix = self_attn_->ComputeAttentionMatrixDense(pattern_ids, context);

    // Diagonal entries should be non-zero
    bool has_nonzero_diagonal = false;
    for (size_t i = 0; i < 3; ++i) {
        if (matrix[i][i] > 0.01f) {
            has_nonzero_diagonal = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_diagonal);
}

// ============================================================================
// Normalization Mode Tests
// ============================================================================

TEST_F(SelfAttentionTest, RowWiseNormalization) {
    SelfAttentionConfig config;
    config.normalization = NormalizationMode::ROW_WISE;
    self_attn_->SetConfig(config);

    auto pattern_ids = CreateTestPatterns(4);

    ContextVector context;
    auto matrix = self_attn_->ComputeAttentionMatrixDense(pattern_ids, context);

    // Each row should sum to 1.0
    for (size_t i = 0; i < 4; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < 4; ++j) {
            row_sum += matrix[i][j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f);
    }
}

TEST_F(SelfAttentionTest, ColumnWiseNormalization) {
    SelfAttentionConfig config;
    config.normalization = NormalizationMode::COLUMN_WISE;
    self_attn_->SetConfig(config);

    auto pattern_ids = CreateTestPatterns(4);

    ContextVector context;
    auto matrix = self_attn_->ComputeAttentionMatrixDense(pattern_ids, context);

    // Each column should sum to 1.0
    for (size_t j = 0; j < 4; ++j) {
        float col_sum = 0.0f;
        for (size_t i = 0; i < 4; ++i) {
            col_sum += matrix[i][j];
        }
        EXPECT_NEAR(col_sum, 1.0f, 1e-5f);
    }
}

TEST_F(SelfAttentionTest, BidirectionalNormalization) {
    SelfAttentionConfig config;
    config.normalization = NormalizationMode::BIDIRECTIONAL;
    self_attn_->SetConfig(config);

    auto pattern_ids = CreateTestPatterns(4);

    ContextVector context;
    auto matrix = self_attn_->ComputeAttentionMatrixDense(pattern_ids, context);

    // Both rows and columns should be normalized
    // Note: Exact sum depends on bidirectional normalization order
    // Just verify all weights are valid probabilities
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            EXPECT_GE(matrix[i][j], 0.0f);
            EXPECT_LE(matrix[i][j], 1.0f);
        }
    }
}

// ============================================================================
// Temperature Tests
// ============================================================================

TEST_F(SelfAttentionTest, TemperatureScaling) {
    // Use patterns with varying features so temperature has an effect
    auto pattern_ids = CreatePatternsWithFeatures(4);
    ContextVector context;

    // Low temperature (sharper distribution)
    SelfAttentionConfig low_temp_config;
    low_temp_config.temperature = 0.1f;
    self_attn_->SetConfig(low_temp_config);
    auto low_temp_matrix = self_attn_->ComputeAttentionMatrixDense(pattern_ids, context);

    // High temperature (more uniform distribution)
    SelfAttentionConfig high_temp_config;
    high_temp_config.temperature = 10.0f;
    self_attn_->SetConfig(high_temp_config);
    auto high_temp_matrix = self_attn_->ComputeAttentionMatrixDense(pattern_ids, context);

    // Low temperature should have more peaked distribution (higher max values)
    float low_temp_max = 0.0f;
    float high_temp_max = 0.0f;

    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            low_temp_max = std::max(low_temp_max, low_temp_matrix[i][j]);
            high_temp_max = std::max(high_temp_max, high_temp_matrix[i][j]);
        }
    }

    EXPECT_GT(low_temp_max, high_temp_max);
}

// ============================================================================
// Threshold Tests
// ============================================================================

TEST_F(SelfAttentionTest, AttentionThresholding) {
    SelfAttentionConfig config;
    config.attention_threshold = 0.2f;  // Only keep attention > 0.2
    self_attn_->SetConfig(config);

    auto pattern_ids = CreateTestPatterns(5);

    ContextVector context;
    auto matrix = self_attn_->ComputeAttentionMatrixDense(pattern_ids, context);

    // All non-zero entries should be >= threshold or very close to 0
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            if (matrix[i][j] > 0.01f) {
                EXPECT_GE(matrix[i][j], config.attention_threshold);
            }
        }
    }

    // Rows should still be normalized
    for (size_t i = 0; i < 5; ++i) {
        float row_sum = 0.0f;
        for (size_t j = 0; j < 5; ++j) {
            row_sum += matrix[i][j];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f);
    }
}

// ============================================================================
// Analysis Utility Tests
// ============================================================================

TEST_F(SelfAttentionTest, FindMostAttendedPatterns) {
    auto pattern_ids = CreateTestPatterns(5);

    ContextVector context;
    auto top_patterns = self_attn_->FindMostAttendedPatterns(pattern_ids, 3, context);

    // Should return top 3
    ASSERT_EQ(top_patterns.size(), 3u);

    // Should be sorted by attention (descending)
    EXPECT_GE(top_patterns[0].second, top_patterns[1].second);
    EXPECT_GE(top_patterns[1].second, top_patterns[2].second);

    // All attention values should be in valid range
    for (const auto& [_, attention] : top_patterns) {
        EXPECT_GE(attention, 0.0f);
        EXPECT_LE(attention, 1.0f);
    }
}

TEST_F(SelfAttentionTest, FindMostAttentivePatterns) {
    auto pattern_ids = CreateTestPatterns(5);

    ContextVector context;
    auto top_patterns = self_attn_->FindMostAttentivePatterns(pattern_ids, 3, context);

    // Should return top 3
    ASSERT_EQ(top_patterns.size(), 3u);

    // Should be sorted by attention (descending)
    EXPECT_GE(top_patterns[0].second, top_patterns[1].second);
    EXPECT_GE(top_patterns[1].second, top_patterns[2].second);
}

TEST_F(SelfAttentionTest, ComputeAttentionEntropy) {
    auto pattern_ids = CreateTestPatterns(4);

    ContextVector context;
    auto entropy_map = self_attn_->ComputeAttentionEntropy(pattern_ids, context);

    ASSERT_EQ(entropy_map.size(), 4u);

    // All entropy values should be >= 0
    for (const auto& [_, entropy] : entropy_map) {
        EXPECT_GE(entropy, 0.0f);
    }
}

TEST_F(SelfAttentionTest, EntropyHighTemperature) {
    // Use patterns with varying features so temperature has an effect
    auto pattern_ids = CreatePatternsWithFeatures(4);
    ContextVector context;

    // High temperature should produce higher entropy (more uniform)
    SelfAttentionConfig high_temp_config;
    high_temp_config.temperature = 10.0f;
    self_attn_->SetConfig(high_temp_config);
    auto high_temp_entropy = self_attn_->ComputeAttentionEntropy(pattern_ids, context);

    // Low temperature should produce lower entropy (more peaked)
    SelfAttentionConfig low_temp_config;
    low_temp_config.temperature = 0.1f;
    self_attn_->SetConfig(low_temp_config);
    auto low_temp_entropy = self_attn_->ComputeAttentionEntropy(pattern_ids, context);

    // Calculate average entropy
    float high_temp_avg = 0.0f;
    for (const auto& [_, entropy] : high_temp_entropy) {
        high_temp_avg += entropy;
    }
    high_temp_avg /= high_temp_entropy.size();

    float low_temp_avg = 0.0f;
    for (const auto& [_, entropy] : low_temp_entropy) {
        low_temp_avg += entropy;
    }
    low_temp_avg /= low_temp_entropy.size();

    EXPECT_GT(high_temp_avg, low_temp_avg);
}

// ============================================================================
// Caching Tests
// ============================================================================

TEST_F(SelfAttentionTest, CachingEnabled) {
    SelfAttentionConfig config;
    config.enable_caching = true;
    config.cache_size = 5;
    self_attn_->SetConfig(config);

    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // First computation (cache miss)
    auto matrix1 = self_attn_->ComputeAttentionMatrix(pattern_ids, context);

    // Second computation (cache hit)
    auto matrix2 = self_attn_->ComputeAttentionMatrix(pattern_ids, context);

    // Results should be identical
    EXPECT_EQ(matrix1.size(), matrix2.size());
    for (const auto& [pair, weight1] : matrix1) {
        EXPECT_NEAR(weight1, matrix2.at(pair), 1e-6f);
    }

    // Check statistics
    auto stats = self_attn_->GetStatistics();
    EXPECT_GT(stats["cache_hits"], 0.0f);
}

TEST_F(SelfAttentionTest, CachingDisabled) {
    SelfAttentionConfig config;
    config.enable_caching = false;
    self_attn_->SetConfig(config);

    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Compute twice
    self_attn_->ComputeAttentionMatrix(pattern_ids, context);
    self_attn_->ComputeAttentionMatrix(pattern_ids, context);

    // Should have no cache hits
    auto stats = self_attn_->GetStatistics();
    EXPECT_EQ(stats["cache_hits"], 0.0f);
    EXPECT_GT(stats["cache_misses"], 0.0f);
}

TEST_F(SelfAttentionTest, ClearCache) {
    SelfAttentionConfig config;
    config.enable_caching = true;
    self_attn_->SetConfig(config);

    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Compute and cache
    self_attn_->ComputeAttentionMatrix(pattern_ids, context);

    // Clear cache
    self_attn_->ClearCache();

    // Check cache size is 0
    auto stats = self_attn_->GetStatistics();
    EXPECT_EQ(stats["cache_size"], 0.0f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(SelfAttentionTest, SymmetricPatterns) {
    // Create patterns with identical features (should have uniform attention)
    auto pattern1 = CreateTestPattern(1.0f, 5);
    auto pattern2 = CreateTestPattern(1.0f, 5);
    auto pattern3 = CreateTestPattern(1.0f, 5);

    mock_db_->Store(pattern1);
    mock_db_->Store(pattern2);
    mock_db_->Store(pattern3);

    std::vector<PatternID> pattern_ids = {
        pattern1.GetID(), pattern2.GetID(), pattern3.GetID()
    };

    ContextVector context;
    auto matrix = self_attn_->ComputeAttentionMatrixDense(pattern_ids, context);

    // All patterns are identical, so attention should be relatively uniform
    // Each entry should be close to 1/3
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            EXPECT_NEAR(matrix[i][j], 1.0f / 3.0f, 0.2f);
        }
    }
}

TEST_F(SelfAttentionTest, GetStatistics) {
    auto pattern_ids = CreateTestPatterns(3);
    ContextVector context;

    // Compute a few times
    self_attn_->ComputeAttentionMatrix(pattern_ids, context);
    self_attn_->ComputeAttentionMatrix(pattern_ids, context);

    auto stats = self_attn_->GetStatistics();

    EXPECT_GE(stats["matrix_computations"], 1.0f);
    EXPECT_GE(stats["cache_misses"], 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
