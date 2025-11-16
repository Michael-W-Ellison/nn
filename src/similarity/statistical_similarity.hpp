// File: src/similarity/statistical_similarity.hpp
#pragma once

#include "similarity_metric.hpp"
#include <vector>
#include <cmath>

namespace dpan {

/// Statistical properties of a signal/distribution
struct StatisticalMoments {
    float mean{0.0f};
    float variance{0.0f};
    float skewness{0.0f};
    float kurtosis{0.0f};
    float min{0.0f};
    float max{0.0f};

    /// Compute statistical moments from data
    static StatisticalMoments Compute(const std::vector<float>& data);
};

/// Histogram representation for distribution comparison
class Histogram {
public:
    /// Constructor
    /// @param num_bins Number of histogram bins
    explicit Histogram(size_t num_bins = 32) : num_bins_(num_bins) {
        if (num_bins_ == 0) {
            num_bins_ = 1;
        }
    }

    /// Build histogram from data
    /// @param data Input data
    void Build(const std::vector<float>& data);

    /// Get bin counts (normalized to sum to 1)
    const std::vector<float>& GetBins() const { return bins_; }

    /// Get number of bins
    size_t GetNumBins() const { return num_bins_; }

private:
    size_t num_bins_;
    std::vector<float> bins_;
    float min_val_{0.0f};
    float max_val_{0.0f};
};

/// Moment Similarity
///
/// Compares statistical moments (mean, variance, skewness, kurtosis).
/// Useful for comparing overall distribution properties.
///
/// Use cases:
/// - Quality control (comparing product measurements)
/// - Sensor data validation
/// - Distribution shape comparison
class MomentSimilarity : public SimilarityMetric {
public:
    /// Constructor
    /// @param weights Weights for [mean, variance, skewness, kurtosis]
    explicit MomentSimilarity(const std::vector<float>& weights = {1.0f, 1.0f, 0.5f, 0.5f})
        : weights_(weights) {
        if (weights_.size() != 4) {
            weights_ = {1.0f, 1.0f, 0.5f, 0.5f};
        }
        NormalizeWeights();
    }

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "Moment"; }
    bool IsSymmetric() const override { return true; }

private:
    std::vector<float> weights_;

    void NormalizeWeights();

    /// Compute similarity between two sets of moments
    static float CompareMoments(const StatisticalMoments& a,
                               const StatisticalMoments& b,
                               const std::vector<float>& weights);
};

/// Histogram Similarity (Bhattacharyya Coefficient)
///
/// Compares probability distributions using histogram overlap.
/// Measures similarity between two probability distributions.
///
/// Use cases:
/// - Image histogram comparison
/// - Distribution matching
/// - Anomaly detection
class HistogramSimilarity : public SimilarityMetric {
public:
    /// Constructor
    /// @param num_bins Number of histogram bins
    explicit HistogramSimilarity(size_t num_bins = 32) : num_bins_(num_bins) {}

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "Histogram"; }
    bool IsSymmetric() const override { return true; }

private:
    size_t num_bins_;

    /// Compute Bhattacharyya coefficient
    static float BhattacharyyaCoefficient(const std::vector<float>& hist_a,
                                         const std::vector<float>& hist_b);
};

/// Kullback-Leibler Divergence Similarity
///
/// Measures how one probability distribution differs from another.
/// KL divergence is not symmetric, so we use symmetric KL divergence.
///
/// Use cases:
/// - Information theory applications
/// - Machine learning (comparing distributions)
/// - Statistical hypothesis testing
class KLDivergenceSimilarity : public SimilarityMetric {
public:
    /// Constructor
    /// @param num_bins Number of histogram bins
    /// @param epsilon Small value to avoid log(0)
    explicit KLDivergenceSimilarity(size_t num_bins = 32, float epsilon = 1e-10f)
        : num_bins_(num_bins), epsilon_(epsilon) {}

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "KLDivergence"; }
    bool IsSymmetric() const override { return true; }  // Using symmetric KL

private:
    size_t num_bins_;
    float epsilon_;

    /// Compute symmetric KL divergence
    static float SymmetricKLDivergence(const std::vector<float>& hist_a,
                                      const std::vector<float>& hist_b,
                                      float epsilon);
};

/// Kolmogorov-Smirnov Test Similarity
///
/// Measures the maximum difference between cumulative distributions.
/// Non-parametric test for comparing distributions.
///
/// Use cases:
/// - Statistical testing
/// - Comparing empirical distributions
/// - Goodness-of-fit testing
class KSSimilarity : public SimilarityMetric {
public:
    KSSimilarity() = default;

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "KS"; }
    bool IsSymmetric() const override { return true; }

private:
    /// Compute KS statistic
    static float KSStatistic(const std::vector<float>& data_a,
                            const std::vector<float>& data_b);
};

/// Chi-Square Test Similarity
///
/// Compares observed vs expected frequencies in histograms.
/// Commonly used for categorical data comparison.
///
/// Use cases:
/// - Categorical data comparison
/// - Goodness-of-fit testing
/// - Feature distribution comparison
class ChiSquareSimilarity : public SimilarityMetric {
public:
    /// Constructor
    /// @param num_bins Number of histogram bins
    explicit ChiSquareSimilarity(size_t num_bins = 32) : num_bins_(num_bins) {}

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "ChiSquare"; }
    bool IsSymmetric() const override { return true; }

private:
    size_t num_bins_;

    /// Compute Chi-square statistic
    static float ChiSquareStatistic(const std::vector<float>& hist_a,
                                   const std::vector<float>& hist_b);
};

/// Earth Mover's Distance (Wasserstein Distance) Similarity
///
/// Measures the minimum cost to transform one distribution to another.
/// Takes into account the "distance" between bins.
///
/// Use cases:
/// - Image retrieval
/// - Document similarity
/// - Distribution comparison with spatial/ordering information
class EarthMoverSimilarity : public SimilarityMetric {
public:
    /// Constructor
    /// @param num_bins Number of histogram bins
    explicit EarthMoverSimilarity(size_t num_bins = 32) : num_bins_(num_bins) {}

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "EarthMover"; }
    bool IsSymmetric() const override { return true; }

private:
    size_t num_bins_;

    /// Compute Earth Mover's Distance (1D case)
    static float EMD1D(const std::vector<float>& hist_a,
                      const std::vector<float>& hist_b);
};

} // namespace dpan
