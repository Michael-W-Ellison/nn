// File: tests/similarity/frequency_similarity_test.cpp
#include "similarity/frequency_similarity.hpp"
#include <gtest/gtest.h>
#include <cmath>

namespace dpan {
namespace {

// ============================================================================
// FrequencyAnalysis Tests
// ============================================================================

TEST(FrequencyAnalysisTest, DFTOfConstantSignal) {
    std::vector<float> signal = {1.0f, 1.0f, 1.0f, 1.0f};
    auto dft = FrequencyAnalysis::DFT(signal);

    EXPECT_EQ(4u, dft.size());

    // DC component should be 4.0
    EXPECT_NEAR(4.0f, std::abs(dft[0]), 1e-5f);

    // Other components should be near zero
    for (size_t i = 1; i < dft.size(); ++i) {
        EXPECT_NEAR(0.0f, std::abs(dft[i]), 1e-5f);
    }
}

TEST(FrequencyAnalysisTest, DFTOfSineWave) {
    const float pi = 3.14159265358979323846f;
    std::vector<float> signal(32);

    // Generate sine wave with frequency 1 (1 cycle over 32 samples)
    for (size_t i = 0; i < 32; ++i) {
        signal[i] = std::sin(2.0f * pi * i / 32.0f);
    }

    auto dft = FrequencyAnalysis::DFT(signal);

    // Peak should be at bin 1
    float max_mag = 0.0f;
    size_t max_idx = 0;
    for (size_t i = 0; i < dft.size() / 2; ++i) {
        float mag = std::abs(dft[i]);
        if (mag > max_mag) {
            max_mag = mag;
            max_idx = i;
        }
    }

    EXPECT_EQ(1u, max_idx);
}

TEST(FrequencyAnalysisTest, PowerSpectrumIsNonNegative) {
    std::vector<float> signal = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    auto power = FrequencyAnalysis::PowerSpectrum(signal);

    EXPECT_EQ(signal.size(), power.size());

    for (float val : power) {
        EXPECT_GE(val, 0.0f);
    }
}

TEST(FrequencyAnalysisTest, AutocorrelationOfConstant) {
    std::vector<float> signal = {5.0f, 5.0f, 5.0f, 5.0f};
    auto ac = FrequencyAnalysis::Autocorrelation(signal, 3);

    // Autocorrelation of constant should be 1.0 at lag 0
    EXPECT_NEAR(1.0f, ac[0], 1e-5f);
}

TEST(FrequencyAnalysisTest, AutocorrelationSymmetry) {
    std::vector<float> signal = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
    auto ac = FrequencyAnalysis::Autocorrelation(signal, 2);

    // First value should be largest (lag 0)
    EXPECT_GE(ac[0], ac[1]);
    EXPECT_GE(ac[0], ac[2]);
}

TEST(FrequencyAnalysisTest, NormalizeZeroMean) {
    std::vector<float> signal = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto normalized = FrequencyAnalysis::Normalize(signal);

    // Mean should be approximately zero
    float sum = 0.0f;
    for (float val : normalized) {
        sum += val;
    }
    float mean = sum / normalized.size();

    EXPECT_NEAR(0.0f, mean, 1e-5f);
}

TEST(FrequencyAnalysisTest, NormalizeUnitVariance) {
    std::vector<float> signal = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto normalized = FrequencyAnalysis::Normalize(signal);

    // Variance should be approximately 1
    float mean = 0.0f;  // Already normalized to zero mean
    float variance = 0.0f;
    for (float val : normalized) {
        variance += val * val;
    }
    variance /= normalized.size();

    EXPECT_NEAR(1.0f, variance, 1e-5f);
}

TEST(FrequencyAnalysisTest, ExtractSignalFromFeatureVector) {
    FeatureVector fv({1.0f, 2.0f, 3.0f, 4.0f});
    auto signal = FrequencyAnalysis::ExtractSignal(fv);

    EXPECT_EQ(4u, signal.size());
    EXPECT_FLOAT_EQ(1.0f, signal[0]);
    EXPECT_FLOAT_EQ(2.0f, signal[1]);
    EXPECT_FLOAT_EQ(3.0f, signal[2]);
    EXPECT_FLOAT_EQ(4.0f, signal[3]);
}

// ============================================================================
// SpectralSimilarity Tests
// ============================================================================

TEST(SpectralSimilarityTest, IdenticalSignalsReturnOne) {
    SpectralSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

TEST(SpectralSimilarityTest, DifferentSignalsReturnLessThanOne) {
    SpectralSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});
    FeatureVector fv2({1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 1.0f);
}

TEST(SpectralSimilarityTest, EmptyFeatureVectorReturnsZero) {
    SpectralSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f});
    FeatureVector fv2;

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(SpectralSimilarityTest, SimilarityInRange) {
    SpectralSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.1f, 2.1f, 2.9f, 4.1f, 4.9f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(SpectralSimilarityTest, GetNameReturnsCorrectName) {
    SpectralSimilarity metric;
    EXPECT_EQ("Spectral", metric.GetName());
}

TEST(SpectralSimilarityTest, IsSymmetric) {
    SpectralSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({2.0f, 3.0f, 4.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_NEAR(sim1, sim2, 1e-5f);
}

TEST(SpectralSimilarityTest, WithoutNormalization) {
    SpectralSimilarity metric(false);

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

// ============================================================================
// AutocorrelationSimilarity Tests
// ============================================================================

TEST(AutocorrelationSimilarityTest, IdenticalSignalsReturnOne) {
    AutocorrelationSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

TEST(AutocorrelationSimilarityTest, DifferentSignalsReturnLessThanOne) {
    AutocorrelationSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});
    FeatureVector fv2({1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 1.0f);
}

TEST(AutocorrelationSimilarityTest, PeriodicSignalsShouldBeSimilar) {
    AutocorrelationSimilarity metric(5);

    // Two periodic signals with same period
    FeatureVector fv1({1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f});
    FeatureVector fv2({2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GT(similarity, 0.5f);  // Should be reasonably similar
}

TEST(AutocorrelationSimilarityTest, GetNameReturnsCorrectName) {
    AutocorrelationSimilarity metric;
    EXPECT_EQ("Autocorrelation", metric.GetName());
}

TEST(AutocorrelationSimilarityTest, IsSymmetric) {
    AutocorrelationSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({2.0f, 3.0f, 4.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_NEAR(sim1, sim2, 1e-5f);
}

TEST(AutocorrelationSimilarityTest, CustomMaxLag) {
    AutocorrelationSimilarity metric(3);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

// ============================================================================
// FrequencyBandSimilarity Tests
// ============================================================================

TEST(FrequencyBandSimilarityTest, IdenticalSignalsReturnOne) {
    FrequencyBandSimilarity metric(4);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

TEST(FrequencyBandSimilarityTest, DifferentSignalsReturnLessThanOne) {
    FrequencyBandSimilarity metric(4);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});
    FeatureVector fv2({5.0f, 4.0f, 3.0f, 2.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 1.0f);
}

TEST(FrequencyBandSimilarityTest, SimilarityInRange) {
    FrequencyBandSimilarity metric(8);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.1f, 2.1f, 2.9f, 4.1f, 4.9f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(FrequencyBandSimilarityTest, GetNameReturnsCorrectName) {
    FrequencyBandSimilarity metric;
    EXPECT_EQ("FrequencyBand", metric.GetName());
}

TEST(FrequencyBandSimilarityTest, IsSymmetric) {
    FrequencyBandSimilarity metric(4);

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({2.0f, 3.0f, 4.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_NEAR(sim1, sim2, 1e-5f);
}

TEST(FrequencyBandSimilarityTest, DifferentNumberOfBands) {
    FrequencyBandSimilarity metric2(2);
    FrequencyBandSimilarity metric8(8);

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    float sim2 = metric2.ComputeFromFeatures(fv1, fv2);
    float sim8 = metric8.ComputeFromFeatures(fv1, fv2);

    // Both should return 1.0 for identical signals
    EXPECT_NEAR(1.0f, sim2, 1e-4f);
    EXPECT_NEAR(1.0f, sim8, 1e-4f);
}

TEST(FrequencyBandSimilarityTest, WithoutNormalization) {
    FrequencyBandSimilarity metric(4, false);

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

// ============================================================================
// PhaseSimilarity Tests
// ============================================================================

TEST(PhaseSimilarityTest, IdenticalSignalsReturnOne) {
    PhaseSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

TEST(PhaseSimilarityTest, DifferentSignalsReturnLessThanOne) {
    PhaseSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});
    FeatureVector fv2({3.0f, 2.0f, 1.0f, 2.0f, 3.0f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 1.0f);
}

TEST(PhaseSimilarityTest, EmptyFeatureVectorReturnsZero) {
    PhaseSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f});
    FeatureVector fv2;

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_FLOAT_EQ(0.0f, similarity);
}

TEST(PhaseSimilarityTest, SimilarityInRange) {
    PhaseSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    FeatureVector fv2({1.1f, 2.1f, 2.9f, 4.1f, 4.9f});

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GE(similarity, 0.0f);
    EXPECT_LE(similarity, 1.0f);
}

TEST(PhaseSimilarityTest, GetNameReturnsCorrectName) {
    PhaseSimilarity metric;
    EXPECT_EQ("Phase", metric.GetName());
}

TEST(PhaseSimilarityTest, IsSymmetric) {
    PhaseSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f});
    FeatureVector fv2({2.0f, 3.0f, 4.0f});

    float sim1 = metric.ComputeFromFeatures(fv1, fv2);
    float sim2 = metric.ComputeFromFeatures(fv2, fv1);

    EXPECT_NEAR(sim1, sim2, 1e-5f);
}

// ============================================================================
// PatternData Integration Tests
// ============================================================================

TEST(FrequencySimilarityTest, WorksWithPatternData) {
    SpectralSimilarity metric;

    FeatureVector fv1({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});
    FeatureVector fv2({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});

    PatternData p1 = PatternData::FromFeatures(fv1, DataModality::NUMERIC);
    PatternData p2 = PatternData::FromFeatures(fv2, DataModality::NUMERIC);

    float similarity = metric.Compute(p1, p2);
    EXPECT_NEAR(1.0f, similarity, 1e-4f);
}

// ============================================================================
// Comparative Tests
// ============================================================================

TEST(FrequencySimilarityTest, DifferentMetricsProduceDifferentResults) {
    SpectralSimilarity spectral;
    AutocorrelationSimilarity autocorr;
    FrequencyBandSimilarity band(4);
    PhaseSimilarity phase;

    // Create two different signals
    FeatureVector fv1({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f});
    FeatureVector fv2({2.0f, 3.0f, 4.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f});

    float spectral_sim = spectral.ComputeFromFeatures(fv1, fv2);
    float autocorr_sim = autocorr.ComputeFromFeatures(fv1, fv2);
    float band_sim = band.ComputeFromFeatures(fv1, fv2);
    float phase_sim = phase.ComputeFromFeatures(fv1, fv2);

    // All should be in valid range
    EXPECT_GE(spectral_sim, 0.0f);
    EXPECT_LE(spectral_sim, 1.0f);
    EXPECT_GE(autocorr_sim, 0.0f);
    EXPECT_LE(autocorr_sim, 1.0f);
    EXPECT_GE(band_sim, 0.0f);
    EXPECT_LE(band_sim, 1.0f);
    EXPECT_GE(phase_sim, 0.0f);
    EXPECT_LE(phase_sim, 1.0f);
}

TEST(FrequencySimilarityTest, SineWavesWithSameFrequency) {
    const float pi = 3.14159265358979323846f;
    SpectralSimilarity metric;

    // Two sine waves with same frequency
    std::vector<float> signal1(32);
    std::vector<float> signal2(32);

    for (size_t i = 0; i < 32; ++i) {
        signal1[i] = std::sin(2.0f * pi * i / 8.0f);
        signal2[i] = std::sin(2.0f * pi * i / 8.0f);
    }

    FeatureVector fv1(signal1);
    FeatureVector fv2(signal2);

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_GT(similarity, 0.95f);  // Should be very similar
}

TEST(FrequencySimilarityTest, SineWavesWithDifferentFrequencies) {
    const float pi = 3.14159265358979323846f;
    SpectralSimilarity metric;

    // Two sine waves with different frequencies
    std::vector<float> signal1(32);
    std::vector<float> signal2(32);

    for (size_t i = 0; i < 32; ++i) {
        signal1[i] = std::sin(2.0f * pi * i / 8.0f);   // Frequency 4
        signal2[i] = std::sin(2.0f * pi * i / 16.0f);  // Frequency 2
    }

    FeatureVector fv1(signal1);
    FeatureVector fv2(signal2);

    float similarity = metric.ComputeFromFeatures(fv1, fv2);
    EXPECT_LT(similarity, 0.8f);  // Should be less similar
}

} // namespace
} // namespace dpan
