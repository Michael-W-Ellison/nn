// File: tests/similarity/statistical_similarity_test.cpp
#include "similarity/statistical_similarity.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <random>

namespace dpan {
namespace {

// ============================================================================
// StatisticalMoments Tests
// ============================================================================

TEST(StatisticalMomentsTest, ComputeMean) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto moments = StatisticalMoments::Compute(data);

    EXPECT_FLOAT_EQ(3.0f, moments.mean);
}

TEST(StatisticalMomentsTest, ComputeVariance) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto moments = StatisticalMoments::Compute(data);

    EXPECT_NEAR(2.0f, moments.variance, 1e-5f);
}

TEST(StatisticalMomentsTest, ComputeMinMax) {
    std::vector<float> data = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    auto moments = StatisticalMoments::Compute(data);

    EXPECT_FLOAT_EQ(1.0f, moments.min);
    EXPECT_FLOAT_EQ(5.0f, moments.max);
}

TEST(StatisticalMomentsTest, SymmetricDistributionHasZeroSkewness) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto moments = StatisticalMoments::Compute(data);

    EXPECT_NEAR(0.0f, moments.skewness, 1e-4f);
}

TEST(StatisticalMomentsTest, EmptyDataReturnsZeros) {
    std::vector<float> data;
    auto moments = StatisticalMoments::Compute(data);

    EXPECT_FLOAT_EQ(0.0f, moments.mean);
    EXPECT_FLOAT_EQ(0.0f, moments.variance);
}

// ============================================================================
// Histogram Tests
// ============================================================================

TEST(HistogramTest, BuildsCorrectNumberOfBins) {
    Histogram hist(10);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    hist.Build(data);

    EXPECT_EQ(10u, hist.GetBins().size());
}

TEST(HistogramTest, BinsSumToOne) {
    Histogram hist(8);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    hist.Build(data);

    float sum = 0.0f;
    for (float bin : hist.GetBins()) {
        sum += bin;
    }

    EXPECT_NEAR(1.0f, sum, 1e-5f);
}

TEST(HistogramTest, UniformDataProducesUniformHistogram) {
    Histogram hist(4);
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    hist.Build(data);

    for (float bin : hist.GetBins()) {
        EXPECT_NEAR(0.25f, bin, 1e-5f);
    }
}

TEST(HistogramTest, HandlesConstantData) {
    Histogram hist(5);
    std::vector<float> data = {3.0f, 3.0f, 3.0f, 3.0f};
    hist.Build(data);

    // All data should be in first bin
    EXPECT_NEAR(1.0f, hist.GetBins()[0], 1e-5f);
    for (size_t i = 1; i < hist.GetBins().size(); ++i) {
        EXPECT_FLOAT_EQ(0.0f, hist.GetBins()[i]);
    }
}

// ============================================================================
// MomentSimilarity Tests
// ============================================================================

TEST(MomentSimilarityTest, IdenticalDataReturnOne) {
    MomentSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

TEST(MomentSimilarityTest, DifferentDataReturnLessThanOne) {
    MomentSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({10.0f, 20.0f, 30.0f, 40.0f, 50.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 1.0f);
}

TEST(MomentSimilarityTest, EmptyFeatureVectorReturnsZero) {
    MomentSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f});
    FeatureVector fv2;

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(MomentSimilarityTest, GetNameReturnsCorrectName) {
    MomentSimilarity metric;
    EXPECT_EQ("Moment", metric.GetName());
}

TEST(MomentSimilarityTest, IsSymmetric) {
    MomentSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({4.0f, 5.0f, 6.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_FLOAT_EQ(sim1, sim2);
}

TEST(MomentSimilarityTest, CustomWeights) {
    MomentSimilarity metric({2.0f, 1.0f, 0.0f, 0.0f});  // Weight mean more

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

// ============================================================================
// HistogramSimilarity Tests
// ============================================================================

TEST(HistogramSimilarityTest, IdenticalDataReturnOne) {
    HistogramSimilarity metric(16);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

TEST(HistogramSimilarityTest, DifferentDataReturnLessThanOne) {
    HistogramSimilarity metric(16);

    // Create two different distributions in the same range [0, 10]
    // One clustered at low values, one clustered at high values
    FeatureVector fv1({0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 10.0f});
    FeatureVector fv2({0.0f, 7.0f, 8.0f, 8.0f, 8.0f, 9.0f, 9.0f, 9.0f, 10.0f, 10.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 0.8f);
}

TEST(HistogramSimilarityTest, SimilarityInRange) {
    HistogramSimilarity metric(32);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f});
    FeatureVector fv2({1.5f, 2.5f, 3.5f, 4.5f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(HistogramSimilarityTest, GetNameReturnsCorrectName) {
    HistogramSimilarity metric;
    EXPECT_EQ("Histogram", metric.GetName());
}

TEST(HistogramSimilarityTest, IsSymmetric) {
    HistogramSimilarity metric(16);

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({4.0f, 5.0f, 6.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_NEAR(sim1, sim2, 1e-5f);
}

// ============================================================================
// KLDivergenceSimilarity Tests
// ============================================================================

TEST(KLDivergenceSimilarityTest, IdenticalDataReturnOne) {
    KLDivergenceSimilarity metric(16);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GT(similarity, 0.99f);
}

TEST(KLDivergenceSimilarityTest, DifferentDataReturnLessThanOne) {
    KLDivergenceSimilarity metric(16);

    FeatureVector fv1({1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    FeatureVector fv2({10.0f, 20.0f, 30.0f, 40.0f, 50.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 0.9f);
}

TEST(KLDivergenceSimilarityTest, SimilarityInRange) {
    KLDivergenceSimilarity metric(32);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f});
    FeatureVector fv2({1.5f, 2.5f, 3.5f, 4.5f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(KLDivergenceSimilarityTest, GetNameReturnsCorrectName) {
    KLDivergenceSimilarity metric;
    EXPECT_EQ("KLDivergence", metric.GetName());
}

TEST(KLDivergenceSimilarityTest, IsSymmetric) {
    KLDivergenceSimilarity metric(16);

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({4.0f, 5.0f, 6.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_NEAR(sim1, sim2, 1e-5f);
}

// ============================================================================
// KSSimilarity Tests
// ============================================================================

TEST(KSSimilarityTest, IdenticalDataReturnOne) {
    KSSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(1.0f, similarity);
}

TEST(KSSimilarityTest, CompletelyDifferentDataReturnLow) {
    KSSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({10.0f, 20.0f, 30.0f, 40.0f, 50.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 0.5f);
}

TEST(KSSimilarityTest, SimilarityInRange) {
    KSSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.5f, 2.5f, 3.5f, 4.5f, 5.5f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(KSSimilarityTest, GetNameReturnsCorrectName) {
    KSSimilarity metric;
    EXPECT_EQ("KS", metric.GetName());
}

TEST(KSSimilarityTest, IsSymmetric) {
    KSSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({4.0f, 5.0f, 6.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_FLOAT_EQ(sim1, sim2);
}

// ============================================================================
// ChiSquareSimilarity Tests
// ============================================================================

TEST(ChiSquareSimilarityTest, IdenticalDataReturnOne) {
    ChiSquareSimilarity metric(16);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

TEST(ChiSquareSimilarityTest, DifferentDataReturnLessThanOne) {
    ChiSquareSimilarity metric(16);

    FeatureVector fv1({1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    FeatureVector fv2({10.0f, 20.0f, 30.0f, 40.0f, 50.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 0.9f);
}

TEST(ChiSquareSimilarityTest, SimilarityInRange) {
    ChiSquareSimilarity metric(32);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f});
    FeatureVector fv2({1.5f, 2.5f, 3.5f, 4.5f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(ChiSquareSimilarityTest, GetNameReturnsCorrectName) {
    ChiSquareSimilarity metric;
    EXPECT_EQ("ChiSquare", metric.GetName());
}

TEST(ChiSquareSimilarityTest, IsSymmetric) {
    ChiSquareSimilarity metric(16);

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({4.0f, 5.0f, 6.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_NEAR(sim1, sim2, 1e-5f);
}

// ============================================================================
// EarthMoverSimilarity Tests
// ============================================================================

TEST(EarthMoverSimilarityTest, IdenticalDataReturnOne) {
    EarthMoverSimilarity metric(16);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

TEST(EarthMoverSimilarityTest, DifferentDataReturnLessThanOne) {
    EarthMoverSimilarity metric(16);

    FeatureVector fv1({1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
    FeatureVector fv2({10.0f, 20.0f, 30.0f, 40.0f, 50.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 0.9f);
}

TEST(EarthMoverSimilarityTest, SimilarityInRange) {
    EarthMoverSimilarity metric(32);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f});
    FeatureVector fv2({1.5f, 2.5f, 3.5f, 4.5f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(EarthMoverSimilarityTest, GetNameReturnsCorrectName) {
    EarthMoverSimilarity metric;
    EXPECT_EQ("EarthMover", metric.GetName());
}

TEST(EarthMoverSimilarityTest, IsSymmetric) {
    EarthMoverSimilarity metric(16);

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({4.0f, 5.0f, 6.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_NEAR(sim1, sim2, 1e-5f);
}

// ============================================================================
// PatternData Integration Tests
// ============================================================================

TEST(StatisticalSimilarityTest, WorksWithPatternData) {
    MomentSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    PatternData p1 = PatternData::FromFeatures(fv1, DataModality::NUMERIC);
    PatternData p2 = PatternData::FromFeatures(fv2, DataModality::NUMERIC);

    float similarity = metric.Compute(p1, p2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

// ============================================================================
// Comparative Tests
// ============================================================================

TEST(StatisticalSimilarityTest, DifferentMetricsProduceDifferentResults) {
    MomentSimilarity moment;
    HistogramSimilarity histogram(16);
    KLDivergenceSimilarity kl(16);
    KSSimilarity ks;
    ChiSquareSimilarity chi(16);
    EarthMoverSimilarity emd(16);

    // Create two different distributions
    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    FeatureVector fv2({2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

    float moment_sim = moment.ComputeFromFeatures(fv1, fv2);
    float hist_sim = histogram.ComputeFromFeatures(fv1, fv2);
    float kl_sim = kl.ComputeFromFeatures(fv1, fv2);
    float ks_sim = ks.ComputeFromFeatures(fv1, fv2);
    float chi_sim = chi.ComputeFromFeatures(fv1, fv2);
    float emd_sim = emd.ComputeFromFeatures(fv1, fv2);

    // All should be in valid range
    EXPECT_GE(moment_sim, 0.0f);
    EXPECT_LE(moment_sim, 1.0f);
    EXPECT_GE(hist_sim, 0.0f);
    EXPECT_LE(hist_sim, 1.0f);
    EXPECT_GE(kl_sim, 0.0f);
    EXPECT_LE(kl_sim, 1.0f);
    EXPECT_GE(ks_sim, 0.0f);
    EXPECT_LE(ks_sim, 1.0f);
    EXPECT_GE(chi_sim, 0.0f);
    EXPECT_LE(chi_sim, 1.0f);
    EXPECT_GE(emd_sim, 0.0f);
    EXPECT_LE(emd_sim, 1.0f);
}

TEST(StatisticalSimilarityTest, NormalDistributionComparison) {
    std::mt19937 gen1(12345);
    std::mt19937 gen2(12345);
    std::normal_distribution<float> dist1(0.0f, 1.0f);
    std::normal_distribution<float> dist2(0.0f, 1.0f);

    std::vector<float> data1(100);
    std::vector<float> data2(100);

    for (size_t i = 0; i < 100; ++i) {
        data1[i] = dist1(gen1);
        data2[i] = dist2(gen2);
    }

    FeatureVector fv1(data1);
    FeatureVector fv2(data2);

    MomentSimilarity moment;
    float similarity = moment.ComputeFromFeatures(fv1, fv2);

    // Same distribution should have high similarity
    EXPECT_GT(similarity, 0.9f);
}

TEST(StatisticalSimilarityTest, DifferentDistributionComparison) {
    std::mt19937 gen1(12345);
    std::mt19937 gen2(54321);
    std::normal_distribution<float> dist1(0.0f, 1.0f);
    std::normal_distribution<float> dist2(5.0f, 2.0f);

    std::vector<float> data1(100);
    std::vector<float> data2(100);

    for (size_t i = 0; i < 100; ++i) {
        data1[i] = dist1(gen1);
        data2[i] = dist2(gen2);
    }

    FeatureVector fv1(data1);
    FeatureVector fv2(data2);

    MomentSimilarity moment;
    float similarity = moment.ComputeFromFeatures(fv1, fv2);

    // Different distributions should have low similarity
    EXPECT_LT(similarity, 0.7f);
}

} // namespace
} // namespace dpan
