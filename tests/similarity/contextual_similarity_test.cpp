// File: tests/similarity/contextual_similarity_test.cpp
#include "similarity/contextual_similarity.hpp"
#include "core/pattern_node.hpp"
#include <gtest/gtest.h>

namespace dpan {
namespace {

// ============================================================================
// ContextVectorSimilarity Tests
// ============================================================================

TEST(ContextVectorSimilarityTest, IdenticalVectorsReturnOne) {
    ContextVectorSimilarity metric;

    ContextVector cv1;
    cv1.Set("dim1", 1.0f);
    cv1.Set("dim2", 2.0f);
    cv1.Set("dim3", 3.0f);

    ContextVector cv2;
    cv2.Set("dim1", 1.0f);
    cv2.Set("dim2", 2.0f);
    cv2.Set("dim3", 3.0f);

    float similarity = metric.ComputeFromContext(cv1, cv2);
    EXPECT_NEAR(1.0f, similarity, 1e-5f);
}

TEST(ContextVectorSimilarityTest, OrthogonalVectorsReturnZero) {
    ContextVectorSimilarity metric;

    ContextVector cv1;
    cv1.Set("dim1", 1.0f);
    cv1.Set("dim2", 0.0f);

    ContextVector cv2;
    cv2.Set("dim1", 0.0f);
    cv2.Set("dim2", 1.0f);

    float similarity = metric.ComputeFromContext(cv1, cv2);
    EXPECT_NEAR(0.0f, similarity, 1e-5f);
}

TEST(ContextVectorSimilarityTest, EmptyVectorsReturnZero) {
    ContextVectorSimilarity metric;

    ContextVector cv1;
    ContextVector cv2;

    float similarity = metric.ComputeFromContext(cv1, cv2);
    // Empty vectors have no information, cosine similarity is undefined/zero
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(ContextVectorSimilarityTest, OneEmptyVectorReturnsZero) {
    ContextVectorSimilarity metric;

    ContextVector cv1;
    cv1.Set("dim1", 1.0f);

    ContextVector cv2;

    float similarity = metric.ComputeFromContext(cv1, cv2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(ContextVectorSimilarityTest, GetNameReturnsCorrectName) {
    ContextVectorSimilarity metric;
    EXPECT_EQ("ContextVector", metric.GetName());
}

TEST(ContextVectorSimilarityTest, IsSymmetric) {
    ContextVectorSimilarity metric;

    ContextVector cv1;
    cv1.Set("dim1", 1.0f);
    cv1.Set("dim2", 2.0f);

    ContextVector cv2;
    cv2.Set("dim1", 3.0f);
    cv2.Set("dim2", 4.0f);

    float sim1 = metric.ComputeFromContext(cv1, cv2);
    float sim2 = metric.ComputeFromContext(cv2, cv1);

    EXPECT_FLOAT_EQ(sim1, sim2);
}

// ============================================================================
// TemporalSimilarity Tests
// ============================================================================

TEST(TemporalSimilarityTest, IdenticalTimestampsReturnOne) {
    TemporalSimilarity metric(1000);  // 1 second window

    Timestamp t1 = Timestamp::Now();
    Timestamp t2 = t1;

    float similarity = metric.ComputeFromTimestamps(t1, t2);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(TemporalSimilarityTest, TimestampsWithinWindowReturnHigh) {
    TemporalSimilarity metric(1000);  // 1 second window

    Timestamp t1 = Timestamp::Now();
    Timestamp t2 = Timestamp::FromMicros(t1.ToMicros() + 500000);  // +500ms

    float similarity = metric.ComputeFromTimestamps(t1, t2);
    EXPECT_GT(similarity, 0.5f);
}

TEST(TemporalSimilarityTest, TimestampsOutsideWindowReturnLow) {
    TemporalSimilarity metric(1000);  // 1 second window

    Timestamp t1 = Timestamp::Now();
    Timestamp t2 = Timestamp::FromMicros(t1.ToMicros() + 5000000);  // +5000ms

    float similarity = metric.ComputeFromTimestamps(t1, t2);
    EXPECT_LT(similarity, 0.1f);
}

TEST(TemporalSimilarityTest, GetNameReturnsCorrectName) {
    TemporalSimilarity metric;
    EXPECT_EQ("Temporal", metric.GetName());
}

TEST(TemporalSimilarityTest, IsSymmetric) {
    TemporalSimilarity metric(1000);

    Timestamp t1 = Timestamp::Now();
    Timestamp t2 = Timestamp::FromMicros(t1.ToMicros() + 500000);  // +500ms

    float sim1 = metric.ComputeFromTimestamps(t1, t2);
    float sim2 = metric.ComputeFromTimestamps(t2, t1);

    EXPECT_FLOAT_EQ(sim1, sim2);
}

// ============================================================================
// HierarchicalSimilarity Tests
// ============================================================================

TEST(HierarchicalSimilarityTest, IdenticalSubPatternsReturnOne) {
    HierarchicalSimilarity metric;

    std::vector<PatternID> sp1 = {PatternID(1), PatternID(2), PatternID(3)};
    std::vector<PatternID> sp2 = {PatternID(1), PatternID(2), PatternID(3)};

    float similarity = metric.ComputeFromSubPatterns(sp1, sp2);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(HierarchicalSimilarityTest, DisjointSubPatternsReturnZero) {
    HierarchicalSimilarity metric;

    std::vector<PatternID> sp1 = {PatternID(1), PatternID(2), PatternID(3)};
    std::vector<PatternID> sp2 = {PatternID(4), PatternID(5), PatternID(6)};

    float similarity = metric.ComputeFromSubPatterns(sp1, sp2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(HierarchicalSimilarityTest, PartialOverlapReturnsPartialSimilarity) {
    HierarchicalSimilarity metric;

    std::vector<PatternID> sp1 = {PatternID(1), PatternID(2), PatternID(3)};
    std::vector<PatternID> sp2 = {PatternID(2), PatternID(3), PatternID(4)};

    float similarity = metric.ComputeFromSubPatterns(sp1, sp2);
    // Intersection: {2, 3} = 2, Union: {1, 2, 3, 4} = 4
    EXPECT_FLOAT_EQ(0.5f, similarity);
}

TEST(HierarchicalSimilarityTest, EmptySubPatternsReturnOne) {
    HierarchicalSimilarity metric;

    std::vector<PatternID> sp1;
    std::vector<PatternID> sp2;

    float similarity = metric.ComputeFromSubPatterns(sp1, sp2);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(HierarchicalSimilarityTest, OneEmptySubPatternsReturnZero) {
    HierarchicalSimilarity metric;

    std::vector<PatternID> sp1 = {PatternID(1), PatternID(2)};
    std::vector<PatternID> sp2;

    float similarity = metric.ComputeFromSubPatterns(sp1, sp2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(HierarchicalSimilarityTest, GetNameReturnsCorrectName) {
    HierarchicalSimilarity metric;
    EXPECT_EQ("Hierarchical", metric.GetName());
}

// ============================================================================
// StatisticalProfileSimilarity Tests
// ============================================================================

TEST(StatisticalProfileSimilarityTest, IdenticalProfilesReturnOne) {
    StatisticalProfileSimilarity metric;

    auto profile1 = StatisticalProfileSimilarity::Profile::Create(100, 0.8f, 0.5f, 1000);
    auto profile2 = StatisticalProfileSimilarity::Profile::Create(100, 0.8f, 0.5f, 1000);

    float similarity = metric.ComputeFromProfiles(profile1, profile2);
    EXPECT_NEAR(1.0f, similarity, 1e-5f);
}

TEST(StatisticalProfileSimilarityTest, DifferentProfilesReturnLessThanOne) {
    StatisticalProfileSimilarity metric;

    auto profile1 = StatisticalProfileSimilarity::Profile::Create(100, 0.8f, 0.5f, 1000);
    auto profile2 = StatisticalProfileSimilarity::Profile::Create(10, 0.2f, 0.1f, 100000);

    float similarity = metric.ComputeFromProfiles(profile1, profile2);
    EXPECT_LT(similarity, 1.0f);
    EXPECT_GT(similarity, 0.0f);
}

TEST(StatisticalProfileSimilarityTest, ZeroAccessCountsReturnHigh) {
    StatisticalProfileSimilarity metric;

    auto profile1 = StatisticalProfileSimilarity::Profile::Create(0, 0.5f, 0.0f, 0);
    auto profile2 = StatisticalProfileSimilarity::Profile::Create(0, 0.5f, 0.0f, 0);

    float similarity = metric.ComputeFromProfiles(profile1, profile2);
    EXPECT_NEAR(1.0f, similarity, 1e-5f);
}

TEST(StatisticalProfileSimilarityTest, GetNameReturnsCorrectName) {
    StatisticalProfileSimilarity metric;
    EXPECT_EQ("StatisticalProfile", metric.GetName());
}

TEST(StatisticalProfileSimilarityTest, CustomWeights) {
    StatisticalProfileSimilarity metric({2.0f, 1.0f, 0.0f, 0.0f});  // Weight access more

    auto profile1 = StatisticalProfileSimilarity::Profile::Create(100, 0.8f, 0.5f, 1000);
    auto profile2 = StatisticalProfileSimilarity::Profile::Create(100, 0.2f, 0.1f, 5000);

    float similarity = metric.ComputeFromProfiles(profile1, profile2);
    // Should be high due to matching access counts
    EXPECT_GT(similarity, 0.6f);
}

// ============================================================================
// TypeSimilarity Tests
// ============================================================================

TEST(TypeSimilarityTest, IdenticalTypesReturnOne) {
    TypeSimilarity metric(true);

    float similarity = metric.ComputeFromTypes(PatternType::ATOMIC, PatternType::ATOMIC);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(TypeSimilarityTest, DifferentTypesStrictReturnZero) {
    TypeSimilarity metric(true);

    float similarity = metric.ComputeFromTypes(PatternType::ATOMIC, PatternType::COMPOSITE);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(TypeSimilarityTest, RelatedTypesNonStrictReturnPartial) {
    TypeSimilarity metric(false);

    float similarity = metric.ComputeFromTypes(PatternType::COMPOSITE, PatternType::META);
    EXPECT_FLOAT_EQ(0.5f, similarity);
}

TEST(TypeSimilarityTest, UnrelatedTypesNonStrictReturnZero) {
    TypeSimilarity metric(false);

    float similarity = metric.ComputeFromTypes(PatternType::ATOMIC, PatternType::COMPOSITE);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(TypeSimilarityTest, GetNameReturnsCorrectName) {
    TypeSimilarity metric;
    EXPECT_EQ("Type", metric.GetName());
}

// ============================================================================
// MetadataSimilarity Tests
// ============================================================================

TEST(MetadataSimilarityTest, DefaultConstructorCreatesAllMetrics) {
    MetadataSimilarity metric;

    // Should have 5 metrics by default
    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(MetadataSimilarityTest, CustomConstructorCreatesSelectedMetrics) {
    MetadataSimilarity metric(false, false, false, false, false);

    FeatureVector fv1({1.0f, 2.0f});
    FeatureVector fv2({3.0f, 4.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(0.0f, similarity);  // No metrics
}

TEST(MetadataSimilarityTest, AddMetricWorks) {
    MetadataSimilarity metric(false, false, false, false, false);

    metric.AddMetric(std::make_shared<ContextVectorSimilarity>(), 1.0f);

    FeatureVector fv1({1.0f, 2.0f});
    FeatureVector fv2({1.0f, 2.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-5f);
}

TEST(MetadataSimilarityTest, ClearRemovesAllMetrics) {
    MetadataSimilarity metric;
    metric.Clear();

    FeatureVector fv1({1.0f, 2.0f});
    FeatureVector fv2({3.0f, 4.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(MetadataSimilarityTest, GetNameReturnsCorrectName) {
    MetadataSimilarity metric;
    EXPECT_EQ("Metadata", metric.GetName());
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(ContextualSimilarityTest, WorksWithPatternData) {
    ContextVectorSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f});

    PatternData p1 = PatternData::FromFeatures(fv1, DataModality::NUMERIC);
    PatternData p2 = PatternData::FromFeatures(fv2, DataModality::NUMERIC);

    // Should use ComputeFromFeatures internally
    float similarity = metric.Compute(p1, p2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(ContextualSimilarityTest, AllMetricsReturnValidRange) {
    ContextVectorSimilarity cv_metric;
    TemporalSimilarity temporal_metric;
    HierarchicalSimilarity hier_metric;
    StatisticalProfileSimilarity stat_metric;
    TypeSimilarity type_metric;

    FeatureVector fv1({1.0f, 2.0f});
    FeatureVector fv2({3.0f, 4.0f});

    float cv_sim = cv_metric.ComputeFromFeatures(fv1, fv2);
    float temporal_sim = temporal_metric.ComputeFromFeatures(fv1, fv2);
    float hier_sim = hier_metric.ComputeFromFeatures(fv1, fv2);
    float stat_sim = stat_metric.ComputeFromFeatures(fv1, fv2);
    float type_sim = type_metric.ComputeFromFeatures(fv1, fv2);

    // All should be in valid range
    EXPECT_GE(cv_sim, 0.0f);
    EXPECT_LE(cv_sim, 1.0f);
    EXPECT_GE(temporal_sim, 0.0f);
    EXPECT_LE(temporal_sim, 1.0f);
    EXPECT_GE(hier_sim, 0.0f);
    EXPECT_LE(hier_sim, 1.0f);
    EXPECT_GE(stat_sim, 0.0f);
    EXPECT_LE(stat_sim, 1.0f);
    EXPECT_GE(type_sim, 0.0f);
    EXPECT_LE(type_sim, 1.0f);
}

} // namespace
} // namespace dpan
