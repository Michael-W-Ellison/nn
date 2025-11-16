// File: tests/discovery/pattern_refiner_test.cpp
#include "discovery/pattern_refiner.hpp"
#include "storage/memory_backend.hpp"
#include <gtest/gtest.h>

namespace dpan {
namespace {

// Helper to create test database
std::shared_ptr<PatternDatabase> CreateTestDatabase() {
    return std::make_shared<MemoryBackend>(MemoryBackend::Config{});
}

// Helper to create a test pattern in the database
PatternID CreateTestPattern(
    std::shared_ptr<PatternDatabase> db,
    const std::vector<float>& feature_values,
    float confidence = 0.5f) {

    FeatureVector features(feature_values);
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    PatternID id(db->FindAll().size() + 1);
    PatternNode node(id, data, PatternType::ATOMIC);
    node.SetConfidenceScore(confidence);
    node.SetActivationThreshold(0.5f);
    node.SetBaseActivation(0.0f);

    db->Store(node);
    return id;
}

// ============================================================================
// PatternRefiner Tests
// ============================================================================

TEST(PatternRefinerTest, ConstructorRequiresNonNullDatabase) {
    EXPECT_THROW(PatternRefiner(nullptr), std::invalid_argument);
}

TEST(PatternRefinerTest, UpdatePatternWorks) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    // Create initial pattern
    PatternID id = CreateTestPattern(db, {1.0f, 2.0f, 3.0f}, 0.7f);

    // Update with new data
    FeatureVector new_features({4.0f, 5.0f, 6.0f});
    PatternData new_data = PatternData::FromFeatures(new_features, DataModality::NUMERIC);

    bool success = refiner.UpdatePattern(id, new_data);
    EXPECT_TRUE(success);

    // Verify update
    auto node_opt = db->Retrieve(id);
    ASSERT_TRUE(node_opt.has_value());

    const auto& updated_features = node_opt->GetData().GetFeatures();
    EXPECT_FLOAT_EQ(4.0f, updated_features[0]);
    EXPECT_FLOAT_EQ(5.0f, updated_features[1]);
    EXPECT_FLOAT_EQ(6.0f, updated_features[2]);

    // Verify confidence is preserved
    EXPECT_FLOAT_EQ(0.7f, node_opt->GetConfidenceScore());
}

TEST(PatternRefinerTest, UpdatePatternReturnsFalseForNonExistentPattern) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    FeatureVector features({1.0f, 2.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    bool success = refiner.UpdatePattern(PatternID(9999), data);
    EXPECT_FALSE(success);
}

TEST(PatternRefinerTest, UpdatePatternPreservesStatistics) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    // Create pattern with specific statistics
    PatternID id = CreateTestPattern(db, {1.0f, 2.0f}, 0.8f);

    auto original_node = db->Retrieve(id);
    ASSERT_TRUE(original_node.has_value());
    float original_threshold = original_node->GetActivationThreshold();

    // Update pattern
    FeatureVector new_features({3.0f, 4.0f});
    PatternData new_data = PatternData::FromFeatures(new_features, DataModality::NUMERIC);
    refiner.UpdatePattern(id, new_data);

    // Verify statistics preserved
    auto updated_node = db->Retrieve(id);
    ASSERT_TRUE(updated_node.has_value());
    EXPECT_FLOAT_EQ(original_threshold, updated_node->GetActivationThreshold());
    EXPECT_FLOAT_EQ(0.8f, updated_node->GetConfidenceScore());
}

TEST(PatternRefinerTest, AdjustConfidenceIncreasesOnCorrectMatch) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    PatternID id = CreateTestPattern(db, {1.0f, 2.0f}, 0.5f);

    // Adjust confidence for correct match
    refiner.AdjustConfidence(id, true);

    auto node_opt = db->Retrieve(id);
    ASSERT_TRUE(node_opt.has_value());

    // Confidence should increase (default rate is 0.1)
    EXPECT_FLOAT_EQ(0.6f, node_opt->GetConfidenceScore());
}

TEST(PatternRefinerTest, AdjustConfidenceDecreasesOnIncorrectMatch) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    PatternID id = CreateTestPattern(db, {1.0f, 2.0f}, 0.5f);

    // Adjust confidence for incorrect match
    refiner.AdjustConfidence(id, false);

    auto node_opt = db->Retrieve(id);
    ASSERT_TRUE(node_opt.has_value());

    // Confidence should decrease
    EXPECT_FLOAT_EQ(0.4f, node_opt->GetConfidenceScore());
}

TEST(PatternRefinerTest, AdjustConfidenceClampsToBounds) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    // Test upper bound
    PatternID id1 = CreateTestPattern(db, {1.0f}, 0.95f);
    refiner.AdjustConfidence(id1, true);  // Would go to 1.05
    auto node1 = db->Retrieve(id1);
    ASSERT_TRUE(node1.has_value());
    EXPECT_FLOAT_EQ(1.0f, node1->GetConfidenceScore());

    // Test lower bound
    PatternID id2 = CreateTestPattern(db, {2.0f}, 0.05f);
    refiner.AdjustConfidence(id2, false);  // Would go to -0.05
    auto node2 = db->Retrieve(id2);
    ASSERT_TRUE(node2.has_value());
    EXPECT_FLOAT_EQ(0.0f, node2->GetConfidenceScore());
}

TEST(PatternRefinerTest, SetConfidenceAdjustmentRateWorks) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    refiner.SetConfidenceAdjustmentRate(0.2f);
    EXPECT_FLOAT_EQ(0.2f, refiner.GetConfidenceAdjustmentRate());

    PatternID id = CreateTestPattern(db, {1.0f}, 0.5f);
    refiner.AdjustConfidence(id, true);

    auto node = db->Retrieve(id);
    ASSERT_TRUE(node.has_value());
    EXPECT_FLOAT_EQ(0.7f, node->GetConfidenceScore());  // 0.5 + 0.2
}

TEST(PatternRefinerTest, SetConfidenceAdjustmentRateRejectsInvalidValues) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    EXPECT_THROW(refiner.SetConfidenceAdjustmentRate(0.0f), std::invalid_argument);
    EXPECT_THROW(refiner.SetConfidenceAdjustmentRate(-0.1f), std::invalid_argument);
    EXPECT_THROW(refiner.SetConfidenceAdjustmentRate(1.5f), std::invalid_argument);
}

TEST(PatternRefinerTest, SplitPatternWorks) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    PatternID id = CreateTestPattern(db, {1.0f, 2.0f, 3.0f});

    auto result = refiner.SplitPattern(id, 2);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(2u, result.new_pattern_ids.size());

    // Verify new patterns exist
    for (const auto& new_id : result.new_pattern_ids) {
        EXPECT_TRUE(db->Exists(new_id));
    }
}

TEST(PatternRefinerTest, SplitPatternRequiresAtLeastTwoClusters) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    PatternID id = CreateTestPattern(db, {1.0f, 2.0f});

    auto result = refiner.SplitPattern(id, 1);
    EXPECT_FALSE(result.success);

    result = refiner.SplitPattern(id, 0);
    EXPECT_FALSE(result.success);
}

TEST(PatternRefinerTest, SplitPatternReturnsFalseForNonExistentPattern) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    auto result = refiner.SplitPattern(PatternID(9999), 2);
    EXPECT_FALSE(result.success);
}

TEST(PatternRefinerTest, MergePatternsWorks) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    // Create two similar patterns
    PatternID id1 = CreateTestPattern(db, {1.0f, 2.0f}, 0.6f);
    PatternID id2 = CreateTestPattern(db, {1.1f, 2.1f}, 0.7f);

    auto result = refiner.MergePatterns({id1, id2});

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(db->Exists(result.merged_id));

    // Verify merged pattern has averaged confidence
    auto merged_node = db->Retrieve(result.merged_id);
    ASSERT_TRUE(merged_node.has_value());
    EXPECT_FLOAT_EQ(0.65f, merged_node->GetConfidenceScore());  // (0.6 + 0.7) / 2
}

TEST(PatternRefinerTest, MergePatternsRequiresAtLeastTwoPatterns) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    PatternID id = CreateTestPattern(db, {1.0f, 2.0f});

    auto result = refiner.MergePatterns({id});
    EXPECT_FALSE(result.success);

    result = refiner.MergePatterns({});
    EXPECT_FALSE(result.success);
}

TEST(PatternRefinerTest, MergePatternsReturnsFalseIfAnyPatternMissing) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    PatternID id1 = CreateTestPattern(db, {1.0f, 2.0f});
    PatternID id2(9999);  // Non-existent

    auto result = refiner.MergePatterns({id1, id2});
    EXPECT_FALSE(result.success);
}

TEST(PatternRefinerTest, MergePatternsReturnsFalseForDifferentTypes) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    // Create two patterns of different types
    FeatureVector f1({1.0f, 2.0f});
    FeatureVector f2({3.0f, 4.0f});
    PatternData d1 = PatternData::FromFeatures(f1, DataModality::NUMERIC);
    PatternData d2 = PatternData::FromFeatures(f2, DataModality::NUMERIC);

    PatternID id1(1);
    PatternNode node1(id1, d1, PatternType::ATOMIC);
    db->Store(node1);

    PatternID id2(2);
    PatternNode node2(id2, d2, PatternType::COMPOSITE);
    db->Store(node2);

    auto result = refiner.MergePatterns({id1, id2});
    EXPECT_FALSE(result.success);
}

TEST(PatternRefinerTest, MergePatternsAveragesParameters) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    // Create patterns with different parameters
    PatternID id1 = CreateTestPattern(db, {1.0f}, 0.6f);
    PatternID id2 = CreateTestPattern(db, {2.0f}, 0.8f);

    // Set different activation thresholds
    auto node1 = db->Retrieve(id1).value();
    node1.SetActivationThreshold(0.4f);
    db->Store(node1);

    auto node2 = db->Retrieve(id2).value();
    node2.SetActivationThreshold(0.6f);
    db->Store(node2);

    auto result = refiner.MergePatterns({id1, id2});
    ASSERT_TRUE(result.success);

    auto merged = db->Retrieve(result.merged_id);
    ASSERT_TRUE(merged.has_value());

    // Check averaged values
    EXPECT_FLOAT_EQ(0.7f, merged->GetConfidenceScore());      // (0.6 + 0.8) / 2
    EXPECT_FLOAT_EQ(0.5f, merged->GetActivationThreshold());  // (0.4 + 0.6) / 2
}

TEST(PatternRefinerTest, MergePatternsHandlesCompositePatterns) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    // Create atomic patterns
    PatternID atomic1 = CreateTestPattern(db, {1.0f});
    PatternID atomic2 = CreateTestPattern(db, {2.0f});
    PatternID atomic3 = CreateTestPattern(db, {3.0f});

    // Create composite patterns
    FeatureVector cf1(std::vector<float>{4.0f});
    FeatureVector cf2(std::vector<float>{5.0f});
    PatternData cd1 = PatternData::FromFeatures(cf1, DataModality::NUMERIC);
    PatternData cd2 = PatternData::FromFeatures(cf2, DataModality::NUMERIC);

    PatternID comp1(100);
    PatternNode comp_node1(comp1, cd1, PatternType::COMPOSITE);
    comp_node1.AddSubPattern(atomic1);
    comp_node1.AddSubPattern(atomic2);
    comp_node1.SetConfidenceScore(0.5f);
    db->Store(comp_node1);

    PatternID comp2(101);
    PatternNode comp_node2(comp2, cd2, PatternType::COMPOSITE);
    comp_node2.AddSubPattern(atomic2);  // Overlapping sub-pattern
    comp_node2.AddSubPattern(atomic3);
    comp_node2.SetConfidenceScore(0.5f);
    db->Store(comp_node2);

    // Merge composite patterns
    auto result = refiner.MergePatterns({comp1, comp2});
    ASSERT_TRUE(result.success);

    auto merged = db->Retrieve(result.merged_id);
    ASSERT_TRUE(merged.has_value());
    EXPECT_EQ(PatternType::COMPOSITE, merged->GetType());

    // Should have 3 unique sub-patterns (atomic1, atomic2, atomic3)
    auto sub_patterns = merged->GetSubPatterns();
    EXPECT_EQ(3u, sub_patterns.size());
}

TEST(PatternRefinerTest, NeedsSplittingDetectsLowConfidence) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    // Low confidence pattern should need splitting
    PatternID id1 = CreateTestPattern(db, {1.0f}, 0.2f);
    EXPECT_TRUE(refiner.NeedsSplitting(id1));

    // High confidence pattern should not need splitting
    PatternID id2 = CreateTestPattern(db, {2.0f}, 0.8f);
    EXPECT_FALSE(refiner.NeedsSplitting(id2));
}

TEST(PatternRefinerTest, NeedsSplittingReturnsFalseForNonExistentPattern) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    EXPECT_FALSE(refiner.NeedsSplitting(PatternID(9999)));
}

TEST(PatternRefinerTest, ShouldMergeDetectsSimilarPatterns) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    // Very similar patterns
    PatternID id1 = CreateTestPattern(db, {1.0f, 2.0f, 3.0f});
    PatternID id2 = CreateTestPattern(db, {1.01f, 2.01f, 3.01f});

    // With high similarity threshold, these should merge
    refiner.SetMergeSimilarityThreshold(0.9f);
    EXPECT_TRUE(refiner.ShouldMerge(id1, id2));
}

TEST(PatternRefinerTest, ShouldMergeRejectsDissimilarPatterns) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    // Very different patterns
    PatternID id1 = CreateTestPattern(db, {1.0f, 2.0f});
    PatternID id2 = CreateTestPattern(db, {100.0f, 200.0f});

    refiner.SetMergeSimilarityThreshold(0.9f);
    EXPECT_FALSE(refiner.ShouldMerge(id1, id2));
}

TEST(PatternRefinerTest, ShouldMergeReturnsFalseForDifferentTypes) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    FeatureVector f1(std::vector<float>{1.0f});
    FeatureVector f2(std::vector<float>{1.0f});
    PatternData d1 = PatternData::FromFeatures(f1, DataModality::NUMERIC);
    PatternData d2 = PatternData::FromFeatures(f2, DataModality::NUMERIC);

    PatternID id1(1);
    PatternNode node1(id1, d1, PatternType::ATOMIC);
    db->Store(node1);

    PatternID id2(2);
    PatternNode node2(id2, d2, PatternType::META);
    db->Store(node2);

    EXPECT_FALSE(refiner.ShouldMerge(id1, id2));
}

TEST(PatternRefinerTest, ShouldMergeReturnsFalseIfAnyPatternMissing) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    PatternID id1 = CreateTestPattern(db, {1.0f});
    PatternID id2(9999);

    EXPECT_FALSE(refiner.ShouldMerge(id1, id2));
}

TEST(PatternRefinerTest, SetVarianceThresholdWorks) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    refiner.SetVarianceThreshold(0.7f);
    EXPECT_FLOAT_EQ(0.7f, refiner.GetVarianceThreshold());
}

TEST(PatternRefinerTest, SetVarianceThresholdRejectsInvalidValues) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    EXPECT_THROW(refiner.SetVarianceThreshold(-0.1f), std::invalid_argument);
    EXPECT_THROW(refiner.SetVarianceThreshold(1.5f), std::invalid_argument);
}

TEST(PatternRefinerTest, SetMinInstancesForSplitWorks) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    refiner.SetMinInstancesForSplit(20);
    EXPECT_EQ(20u, refiner.GetMinInstancesForSplit());
}

TEST(PatternRefinerTest, SetMergeSimilarityThresholdWorks) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    refiner.SetMergeSimilarityThreshold(0.98f);
    EXPECT_FLOAT_EQ(0.98f, refiner.GetMergeSimilarityThreshold());
}

TEST(PatternRefinerTest, SetMergeSimilarityThresholdRejectsInvalidValues) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    EXPECT_THROW(refiner.SetMergeSimilarityThreshold(-0.1f), std::invalid_argument);
    EXPECT_THROW(refiner.SetMergeSimilarityThreshold(1.5f), std::invalid_argument);
}

TEST(PatternRefinerTest, GetDefaultValuesCorrect) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    EXPECT_FLOAT_EQ(0.5f, refiner.GetVarianceThreshold());
    EXPECT_EQ(10u, refiner.GetMinInstancesForSplit());
    EXPECT_FLOAT_EQ(0.95f, refiner.GetMergeSimilarityThreshold());
    EXPECT_FLOAT_EQ(0.1f, refiner.GetConfidenceAdjustmentRate());
}

TEST(PatternRefinerTest, MultipleConfidenceAdjustments) {
    auto db = CreateTestDatabase();
    PatternRefiner refiner(db);

    PatternID id = CreateTestPattern(db, {1.0f}, 0.5f);

    // Multiple correct matches
    refiner.AdjustConfidence(id, true);   // 0.6
    refiner.AdjustConfidence(id, true);   // 0.7
    refiner.AdjustConfidence(id, false);  // 0.6
    refiner.AdjustConfidence(id, true);   // 0.7

    auto node = db->Retrieve(id);
    ASSERT_TRUE(node.has_value());
    EXPECT_FLOAT_EQ(0.7f, node->GetConfidenceScore());
}

} // namespace
} // namespace dpan
