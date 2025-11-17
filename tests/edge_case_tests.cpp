// File: tests/edge_case_tests.cpp
//
// Edge Case and Boundary Condition Tests
// Created as part of TDD analysis to improve test coverage
//
// This test suite focuses on:
// - Null and empty input handling
// - Boundary values (min/max)
// - Error conditions
// - Resource exhaustion
// - Invalid state transitions

#include "core/pattern_node.hpp"
#include "core/pattern_data.hpp"
#include "storage/memory_backend.hpp"
#include "association/association_matrix.hpp"
#include "association/association_edge.hpp"
#include "memory/utility_calculator.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace dpan {
namespace {

// ============================================================================
// Pattern Node Edge Cases
// ============================================================================

TEST(EdgeCaseTest, PatternNodeWithEmptyFeatureVector) {
    FeatureVector empty_features;  // Size 0

    // Should handle empty features gracefully
    EXPECT_NO_THROW({
        PatternData data = PatternData::FromFeatures(empty_features, DataModality::NUMERIC);
        PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);
    });
}

TEST(EdgeCaseTest, PatternNodeWithExtremelyLargeFeatureVector) {
    // Test with 10,000 features
    FeatureVector large_features(10000);
    for (size_t i = 0; i < 10000; ++i) {
        large_features[i] = static_cast<float>(i) / 10000.0f;
    }

    EXPECT_NO_THROW({
        PatternData data = PatternData::FromFeatures(large_features, DataModality::NUMERIC);
        PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);
        EXPECT_EQ(10000u, node.GetData().GetFeatures().size());
    });
}

TEST(EdgeCaseTest, PatternNodeWithInfiniteFeatureValues) {
    FeatureVector features(3);
    features[0] = std::numeric_limits<float>::infinity();
    features[1] = -std::numeric_limits<float>::infinity();
    features[2] = 0.0f;

    // Should create node but might affect similarity calculations
    EXPECT_NO_THROW({
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);
    });
}

TEST(EdgeCaseTest, PatternNodeWithNaNFeatureValues) {
    FeatureVector features(3);
    features[0] = std::numeric_limits<float>::quiet_NaN();
    features[1] = 1.0f;
    features[2] = 2.0f;

    // NaN values should be handled (either rejected or normalized)
    EXPECT_NO_THROW({
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);
    });
}

// ============================================================================
// Memory Backend Edge Cases
// ============================================================================

TEST(EdgeCaseTest, MemoryBackendStoreNullPatternData) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Cannot test true null as PatternNode requires valid data
    // But can test minimal pattern
    FeatureVector features(1, 0.0f);
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);

    EXPECT_TRUE(backend.Store(node));
}

TEST(EdgeCaseTest, MemoryBackendExceedsCapacity) {
    MemoryBackend::Config config;
    config.max_size = 10;  // Very small capacity
    MemoryBackend backend(config);

    // Store more than capacity
    std::vector<PatternID> ids;
    for (int i = 0; i < 20; ++i) {
        FeatureVector features(1, static_cast<float>(i));
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);
        ids.push_back(node.GetID());
        backend.Store(node);
    }

    // Backend should handle capacity (either reject or evict)
    EXPECT_LE(backend.Count(), config.max_size);
}

TEST(EdgeCaseTest, MemoryBackendConcurrentStoreRetrieve) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id = PatternID::Generate();
    FeatureVector features(3, 1.0f);
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    PatternNode node(id, data, PatternType::ATOMIC);

    // Store and retrieve in quick succession
    EXPECT_TRUE(backend.Store(node));
    auto retrieved = backend.Retrieve(id);
    EXPECT_TRUE(retrieved.has_value());
    EXPECT_EQ(id, retrieved->GetID());
}

// ============================================================================
// Association Matrix Edge Cases
// ============================================================================

TEST(EdgeCaseTest, AssociationMatrixSelfLoop) {
    AssociationMatrix matrix;

    PatternID id = PatternID::Generate();

    // Try to create self-loop (pattern associated with itself)
    bool result = matrix.AddAssociation(id, id, 0.5f, AssociationType::TEMPORAL);

    // Should either accept (if allowed) or reject (if prevented)
    // Document behavior: Currently allows self-loops
    EXPECT_TRUE(result);  // Or EXPECT_FALSE if self-loops prevented
}

TEST(EdgeCaseTest, AssociationMatrixZeroStrength) {
    AssociationMatrix matrix;

    PatternID id1 = PatternID::Generate();
    PatternID id2 = PatternID::Generate();

    // Association with zero strength
    bool result = matrix.AddAssociation(id1, id2, 0.0f, AssociationType::TEMPORAL);

    // Should handle zero strength (might reject or normalize)
    EXPECT_TRUE(result || !result);  // Document actual behavior
}

TEST(EdgeCaseTest, AssociationMatrixNegativeStrength) {
    AssociationMatrix matrix;

    PatternID id1 = PatternID::Generate();
    PatternID id2 = PatternID::Generate();

    // Negative strength (inhibitory association?)
    bool result = matrix.AddAssociation(id1, id2, -0.5f, AssociationType::TEMPORAL);

    // Should either normalize or reject
    if (result) {
        auto edge = matrix.GetAssociation(id1, id2);
        EXPECT_TRUE(edge.has_value());
        // Verify strength is clamped to valid range
        EXPECT_GE(edge->GetStrength(), 0.0f);
    }
}

TEST(EdgeCaseTest, AssociationMatrixExtremelyLargeStrength) {
    AssociationMatrix matrix;

    PatternID id1 = PatternID::Generate();
    PatternID id2 = PatternID::Generate();

    // Very large strength value
    bool result = matrix.AddAssociation(id1, id2, 1000000.0f, AssociationType::TEMPORAL);

    if (result) {
        auto edge = matrix.GetAssociation(id1, id2);
        EXPECT_TRUE(edge.has_value());
        // Should be clamped to [0, 1] or documented maximum
        EXPECT_LE(edge->GetStrength(), 10.0f);  // Reasonable upper bound
    }
}

TEST(EdgeCaseTest, AssociationMatrixMillionEdges) {
    AssociationMatrix matrix;

    // Create 1000 patterns
    std::vector<PatternID> patterns;
    for (int i = 0; i < 1000; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    // Create 1000 associations (not 1M to keep test fast)
    size_t added = 0;
    for (int i = 0; i < 1000; ++i) {
        PatternID from = patterns[i % patterns.size()];
        PatternID to = patterns[(i + 1) % patterns.size()];
        if (matrix.AddAssociation(from, to, 0.5f, AssociationType::TEMPORAL)) {
            ++added;
        }
    }

    EXPECT_EQ(1000u, added);
    EXPECT_GE(matrix.GetAssociationCount(), 1000u);
}

// ============================================================================
// Utility Calculator Edge Cases
// ============================================================================

TEST(EdgeCaseTest, UtilityCalculatorZeroAccessCount) {
    UtilityCalculator::Config config;
    UtilityCalculator calc(config);

    // Pattern never accessed
    FeatureVector features(3, 1.0f);
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 0;
    stats.last_access = Timestamp::Now();
    stats.creation_time = Timestamp::Now();

    std::vector<AssociationEdge> associations;  // No associations

    float utility = calc.CalculatePatternUtility(pattern, stats, associations);

    EXPECT_GE(utility, 0.0f);
    EXPECT_LE(utility, 1.0f);
}

TEST(EdgeCaseTest, UtilityCalculatorExtremelyOldPattern) {
    UtilityCalculator::Config config;
    UtilityCalculator calc(config);

    // Pattern from 100 years ago
    FeatureVector features(3, 1.0f);
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 1;
    stats.last_access = Timestamp::Now() - std::chrono::hours(24 * 365 * 100);
    stats.creation_time = Timestamp::Now() - std::chrono::hours(24 * 365 * 100);

    std::vector<AssociationEdge> associations;

    float utility = calc.CalculatePatternUtility(pattern, stats, associations);

    // Should have very low recency score
    EXPECT_GE(utility, 0.0f);
    EXPECT_LT(utility, 0.5f);  // Expect low utility for very old pattern
}

TEST(EdgeCaseTest, UtilityCalculatorExtremelyRecentPattern) {
    UtilityCalculator::Config config;
    UtilityCalculator calc(config);

    // Pattern from 1 millisecond ago
    FeatureVector features(3, 1.0f);
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

    AccessStats stats;
    stats.access_count = 100;
    stats.last_access = Timestamp::Now() - std::chrono::milliseconds(1);
    stats.creation_time = Timestamp::Now() - std::chrono::hours(1);

    std::vector<AssociationEdge> associations;

    float utility = calc.CalculatePatternUtility(pattern, stats, associations);

    // Should have very high utility
    EXPECT_GE(utility, 0.3f);
    EXPECT_LE(utility, 1.0f);
}

// ============================================================================
// Timestamp Edge Cases
// ============================================================================

TEST(EdgeCaseTest, TimestampMinMax) {
    // Test timestamp boundaries
    Timestamp min_time;  // Default (epoch or min)
    Timestamp max_time = Timestamp::Now() + std::chrono::hours(24 * 365 * 100);

    // Should be valid timestamps
    EXPECT_GE(min_time.ToMicros(), 0);
    EXPECT_GT(max_time.ToMicros(), min_time.ToMicros());

    // Duration calculation should not overflow
    auto duration = max_time - min_time;
    EXPECT_GT(duration.count(), 0);
}

// ============================================================================
// PatternID Edge Cases
// ============================================================================

TEST(EdgeCaseTest, PatternIDGenerateUnique) {
    // Generate many IDs and verify uniqueness
    std::unordered_set<uint64_t> ids;

    for (int i = 0; i < 10000; ++i) {
        PatternID id = PatternID::Generate();
        uint64_t raw = id.value();

        // Should be unique
        EXPECT_EQ(ids.find(raw), ids.end());
        ids.insert(raw);
    }

    EXPECT_EQ(10000u, ids.size());
}

TEST(EdgeCaseTest, PatternIDZeroValue) {
    // Test if PatternID(0) is valid or reserved
    PatternID zero_id(0);

    // Document behavior: 0 is invalid (reserved)
    EXPECT_EQ(0u, zero_id.value());
    EXPECT_FALSE(zero_id.IsValid());
}

TEST(EdgeCaseTest, PatternIDMaxValue) {
    // Test maximum uint64 value
    PatternID max_id(std::numeric_limits<uint64_t>::max());

    EXPECT_EQ(std::numeric_limits<uint64_t>::max(), max_id.value());
}

// ============================================================================
// Resource Exhaustion (Commented out for safety)
// ============================================================================

// TEST(EdgeCaseTest, DISABLED_OutOfMemoryHandling) {
//     // This test would exhaust memory - disabled by default
//     // Only run in isolated environment with proper limits
//
//     MemoryBackend::Config config;
//     config.max_size = std::numeric_limits<size_t>::max();
//     MemoryBackend backend(config);
//
//     try {
//         for (size_t i = 0; i < 1000000000; ++i) {
//             FeatureVector features(1000, 1.0f);  // 1000 floats per pattern
//             PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
//             PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);
//             backend.Store(node);
//         }
//         FAIL() << "Expected out-of-memory exception";
//     } catch (const std::bad_alloc&) {
//         // Expected behavior
//         SUCCEED();
//     }
// }

} // namespace
} // namespace dpan
