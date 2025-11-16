// File: tests/similarity/geometric_similarity_test.cpp
#include "similarity/geometric_similarity.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace dpan {
namespace {

// ============================================================================
// Point and PointSet Tests
// ============================================================================

TEST(PointTest, DefaultConstructorZeros) {
    Point<3> p;
    EXPECT_FLOAT_EQ(0.0f, p[0]);
    EXPECT_FLOAT_EQ(0.0f, p[1]);
    EXPECT_FLOAT_EQ(0.0f, p[2]);
}

TEST(PointTest, ConstructorFromArray) {
    float data[] = {1.0f, 2.0f, 3.0f};
    Point<3> p(data);
    EXPECT_FLOAT_EQ(1.0f, p[0]);
    EXPECT_FLOAT_EQ(2.0f, p[1]);
    EXPECT_FLOAT_EQ(3.0f, p[2]);
}

TEST(PointTest, DistanceTo) {
    float data1[] = {0.0f, 0.0f};
    float data2[] = {3.0f, 4.0f};
    Point<2> p1(data1);
    Point<2> p2(data2);

    float dist = p1.DistanceTo(p2);
    EXPECT_FLOAT_EQ(5.0f, dist);  // 3-4-5 triangle
}

TEST(PointTest, SquaredDistanceTo) {
    float data1[] = {0.0f, 0.0f};
    float data2[] = {3.0f, 4.0f};
    Point<2> p1(data1);
    Point<2> p2(data2);

    float sq_dist = p1.SquaredDistanceTo(p2);
    EXPECT_FLOAT_EQ(25.0f, sq_dist);
}

TEST(PointSetTest, FromFeatureVector2D) {
    FeatureVector fv({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    auto pointset = PointSet<2>::FromFeatureVector(fv);

    EXPECT_EQ(3u, pointset.Size());
    EXPECT_FLOAT_EQ(1.0f, pointset.points[0][0]);
    EXPECT_FLOAT_EQ(2.0f, pointset.points[0][1]);
    EXPECT_FLOAT_EQ(3.0f, pointset.points[1][0]);
    EXPECT_FLOAT_EQ(4.0f, pointset.points[1][1]);
    EXPECT_FLOAT_EQ(5.0f, pointset.points[2][0]);
    EXPECT_FLOAT_EQ(6.0f, pointset.points[2][1]);
}

TEST(PointSetTest, FromFeatureVector1D) {
    FeatureVector fv({1.0f, 2.0f, 3.0f});
    auto pointset = PointSet<1>::FromFeatureVector(fv);

    EXPECT_EQ(3u, pointset.Size());
    EXPECT_FLOAT_EQ(1.0f, pointset.points[0][0]);
    EXPECT_FLOAT_EQ(2.0f, pointset.points[1][0]);
    EXPECT_FLOAT_EQ(3.0f, pointset.points[2][0]);
}

// ============================================================================
// Hausdorff Similarity Tests
// ============================================================================

TEST(HausdorffSimilarityTest, IdenticalPointsReturnOne) {
    HausdorffSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 1.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(HausdorffSimilarityTest, DifferentPointsReturnLessThanOne) {
    HausdorffSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 2.0f, 2.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 1.0f);
    EXPECT_GT(similarity, 0.0f);
}

TEST(HausdorffSimilarityTest, EmptyFeatureVectorReturnsZero) {
    HausdorffSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f});
    FeatureVector fv2;

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(HausdorffSimilarityTest, IsSymmetric) {
    HausdorffSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 2.0f, 2.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_FLOAT_EQ(sim1, sim2);
}

TEST(HausdorffSimilarityTest, GetNameReturnsCorrectName) {
    HausdorffSimilarity metric;
    EXPECT_EQ("Hausdorff", metric.GetName());
}

TEST(HausdorffSimilarityTest, IsMetricReturnsTrue) {
    HausdorffSimilarity metric;
    EXPECT_TRUE(metric.IsMetric());
}

TEST(HausdorffSimilarityTest, IsSymmetricReturnsTrue) {
    HausdorffSimilarity metric;
    EXPECT_TRUE(metric.IsSymmetric());
}

// ============================================================================
// Chamfer Similarity Tests
// ============================================================================

TEST(ChamferSimilarityTest, IdenticalPointsReturnOne) {
    ChamferSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 1.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(ChamferSimilarityTest, DifferentPointsReturnLessThanOne) {
    ChamferSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 2.0f, 2.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 1.0f);
    EXPECT_GT(similarity, 0.0f);
}

TEST(ChamferSimilarityTest, EmptyFeatureVectorReturnsZero) {
    ChamferSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f});
    FeatureVector fv2;

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(ChamferSimilarityTest, IsSymmetric) {
    ChamferSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 2.0f, 2.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_FLOAT_EQ(sim1, sim2);
}

TEST(ChamferSimilarityTest, GetNameReturnsCorrectName) {
    ChamferSimilarity metric;
    EXPECT_EQ("Chamfer", metric.GetName());
}

TEST(ChamferSimilarityTest, IsMetricReturnsFalse) {
    ChamferSimilarity metric;
    EXPECT_FALSE(metric.IsMetric());
}

// ============================================================================
// Modified Hausdorff Similarity Tests
// ============================================================================

TEST(ModifiedHausdorffSimilarityTest, IdenticalPointsReturnOne) {
    ModifiedHausdorffSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 1.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(ModifiedHausdorffSimilarityTest, DifferentPointsReturnLessThanOne) {
    ModifiedHausdorffSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 2.0f, 2.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 1.0f);
    EXPECT_GT(similarity, 0.0f);
}

TEST(ModifiedHausdorffSimilarityTest, MoreRobustToOutliersThanHausdorff) {
    ModifiedHausdorffSimilarity modified_metric;
    HausdorffSimilarity hausdorff_metric;

    // Create point sets where one has an outlier
    FeatureVector fv1({0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f});
    FeatureVector fv2({0.0f, 0.0f, 1.0f, 0.0f, 100.0f, 0.0f});  // Last point is outlier

    float modified_sim = modified_metric.ComputeFromFeatures(fv1, fv2);
    float hausdorff_sim = hausdorff_metric.ComputeFromFeatures(fv1, fv2);

    // Modified Hausdorff should be less affected by the outlier
    EXPECT_GT(modified_sim, hausdorff_sim);
}

TEST(ModifiedHausdorffSimilarityTest, GetNameReturnsCorrectName) {
    ModifiedHausdorffSimilarity metric;
    EXPECT_EQ("ModifiedHausdorff", metric.GetName());
}

TEST(ModifiedHausdorffSimilarityTest, IsSymmetric) {
    ModifiedHausdorffSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 2.0f, 2.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_FLOAT_EQ(sim1, sim2);
}

// ============================================================================
// Procrustes Similarity Tests
// ============================================================================

TEST(ProcrusteSimilarityTest, IdenticalPointsReturnOne) {
    ProcrusteSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 1.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(ProcrusteSimilarityTest, TranslatedShapesShouldBeSimilar) {
    ProcrusteSimilarity metric;

    // Triangle at origin
    FeatureVector fv1({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f});
    // Same triangle translated by (10, 10)
    FeatureVector fv2({10.0f, 10.0f, 11.0f, 10.0f, 10.0f, 11.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);

    // After centering and normalization, should be very similar
    EXPECT_GT(similarity, 0.95f);
}

TEST(ProcrusteSimilarityTest, ScaledShapesShouldBeSimilar) {
    ProcrusteSimilarity metric;

    // Triangle
    FeatureVector fv1({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f});
    // Same triangle scaled by 2
    FeatureVector fv2({0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 2.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);

    // After normalization, should be very similar
    EXPECT_GT(similarity, 0.95f);
}

TEST(ProcrusteSimilarityTest, DifferentShapesReturnLowerSimilarity) {
    ProcrusteSimilarity metric;

    // Right triangle
    FeatureVector fv1({0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f});
    // Different shape
    FeatureVector fv2({0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 1.0f);
}

TEST(ProcrusteSimilarityTest, DifferentSizedPointSetsReturnZero) {
    ProcrusteSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(ProcrusteSimilarityTest, GetNameReturnsCorrectName) {
    ProcrusteSimilarity metric;
    EXPECT_EQ("Procrustes", metric.GetName());
}

TEST(ProcrusteSimilarityTest, IsSymmetric) {
    ProcrusteSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 2.0f, 2.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_FLOAT_EQ(sim1, sim2);
}

// ============================================================================
// PatternData Integration Tests
// ============================================================================

TEST(GeometricSimilarityTest, WorksWithPatternData) {
    HausdorffSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f});
    FeatureVector fv2({0.0f, 0.0f, 2.0f, 2.0f});

    PatternData p1 = PatternData::FromFeatures(fv1, DataModality::NUMERIC);
    PatternData p2 = PatternData::FromFeatures(fv2, DataModality::NUMERIC);

    float similarity = metric.Compute(p1, p2);
    EXPECT_GT(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

// ============================================================================
// 1D Point Tests
// ============================================================================

TEST(GeometricSimilarityTest, WorksWith1DPoints) {
    HausdorffSimilarity metric;

    // Odd-length feature vector should use 1D points
    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(GeometricSimilarityTest, Chamfer1DPoints) {
    ChamferSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({1.0f, 2.0f, 4.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 1.0f);
    EXPECT_GT(similarity, 0.0f);
}

// ============================================================================
// Similarity Range Tests
// ============================================================================

TEST(GeometricSimilarityTest, HausdorffSimilarityInRange) {
    HausdorffSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f});
    FeatureVector fv2({0.5f, 0.5f, 1.5f, 1.5f, 2.5f, 2.5f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(GeometricSimilarityTest, ChamferSimilarityInRange) {
    ChamferSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f});
    FeatureVector fv2({0.5f, 0.5f, 1.5f, 1.5f, 2.5f, 2.5f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(GeometricSimilarityTest, ProcrusteSimilarityInRange) {
    ProcrusteSimilarity metric;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f});
    FeatureVector fv2({0.5f, 0.5f, 1.5f, 1.5f, 2.5f, 2.5f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

// ============================================================================
// Comparative Tests
// ============================================================================

TEST(GeometricSimilarityTest, DifferentMetricsProduceDifferentResults) {
    HausdorffSimilarity hausdorff;
    ChamferSimilarity chamfer;
    ModifiedHausdorffSimilarity modified;

    FeatureVector fv1({0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f});
    FeatureVector fv2({0.0f, 0.0f, 1.0f, 0.0f, 10.0f, 0.0f});

    float hausdorff_sim = hausdorff.ComputeFromFeatures(fv1, fv2);
    float chamfer_sim = chamfer.ComputeFromFeatures(fv1, fv2);
    float modified_sim = modified.ComputeFromFeatures(fv1, fv2);

    // They should produce different similarity values
    // Modified should be more robust to the outlier
    EXPECT_NE(hausdorff_sim, chamfer_sim);
    EXPECT_GT(modified_sim, hausdorff_sim);
}

} // namespace
} // namespace dpan
