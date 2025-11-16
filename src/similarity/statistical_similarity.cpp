// File: src/similarity/statistical_similarity.cpp
#include "statistical_similarity.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace dpan {

// ============================================================================
// StatisticalMoments Implementation
// ============================================================================

StatisticalMoments StatisticalMoments::Compute(const std::vector<float>& data) {
    StatisticalMoments moments;

    if (data.empty()) {
        return moments;
    }

    size_t n = data.size();

    // Min and max
    moments.min = *std::min_element(data.begin(), data.end());
    moments.max = *std::max_element(data.begin(), data.end());

    // Mean
    moments.mean = std::accumulate(data.begin(), data.end(), 0.0f) / n;

    // Variance, skewness, kurtosis
    float m2 = 0.0f;  // Second central moment
    float m3 = 0.0f;  // Third central moment
    float m4 = 0.0f;  // Fourth central moment

    for (float val : data) {
        float diff = val - moments.mean;
        float diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
        m4 += diff2 * diff2;
    }

    m2 /= n;
    m3 /= n;
    m4 /= n;

    moments.variance = m2;

    if (m2 > 1e-10f) {
        float std = std::sqrt(m2);
        moments.skewness = m3 / (std * std * std);
        moments.kurtosis = m4 / (m2 * m2) - 3.0f;  // Excess kurtosis
    } else {
        moments.skewness = 0.0f;
        moments.kurtosis = 0.0f;
    }

    return moments;
}

// ============================================================================
// Histogram Implementation
// ============================================================================

void Histogram::Build(const std::vector<float>& data) {
    if (data.empty()) {
        bins_.assign(num_bins_, 0.0f);
        return;
    }

    // Find min and max
    min_val_ = *std::min_element(data.begin(), data.end());
    max_val_ = *std::max_element(data.begin(), data.end());

    // Handle case where all values are the same
    if (std::abs(max_val_ - min_val_) < 1e-10f) {
        bins_.assign(num_bins_, 0.0f);
        bins_[0] = 1.0f;
        return;
    }

    // Build histogram
    bins_.assign(num_bins_, 0.0f);
    float range = max_val_ - min_val_;

    for (float val : data) {
        size_t bin = static_cast<size_t>((val - min_val_) / range * num_bins_);
        if (bin >= num_bins_) {
            bin = num_bins_ - 1;
        }
        bins_[bin] += 1.0f;
    }

    // Normalize to sum to 1
    float total = static_cast<float>(data.size());
    for (auto& bin : bins_) {
        bin /= total;
    }
}

// ============================================================================
// MomentSimilarity Implementation
// ============================================================================

void MomentSimilarity::NormalizeWeights() {
    float sum = std::accumulate(weights_.begin(), weights_.end(), 0.0f);
    if (sum > 1e-10f) {
        for (auto& w : weights_) {
            w /= sum;
        }
    } else {
        // Default to uniform weights
        std::fill(weights_.begin(), weights_.end(), 0.25f);
    }
}

float MomentSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float MomentSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    std::vector<float> data_a(a.Data().begin(), a.Data().end());
    std::vector<float> data_b(b.Data().begin(), b.Data().end());

    auto moments_a = StatisticalMoments::Compute(data_a);
    auto moments_b = StatisticalMoments::Compute(data_b);

    return CompareMoments(moments_a, moments_b, weights_);
}

float MomentSimilarity::CompareMoments(const StatisticalMoments& a,
                                      const StatisticalMoments& b,
                                      const std::vector<float>& weights) {
    // Compute normalized differences for each moment
    std::vector<float> diffs(4);

    // Mean difference (normalized by range)
    float range = std::max(std::abs(a.max - a.min), std::abs(b.max - b.min));
    if (range > 1e-10f) {
        diffs[0] = std::abs(a.mean - b.mean) / range;
    } else {
        diffs[0] = 0.0f;
    }

    // Variance difference (normalized)
    float max_var = std::max(a.variance, b.variance);
    if (max_var > 1e-10f) {
        diffs[1] = std::abs(a.variance - b.variance) / max_var;
    } else {
        diffs[1] = 0.0f;
    }

    // Skewness difference
    diffs[2] = std::abs(a.skewness - b.skewness);
    if (diffs[2] > 2.0f) diffs[2] = 2.0f;  // Clamp to reasonable range
    diffs[2] /= 2.0f;  // Normalize to [0, 1]

    // Kurtosis difference
    diffs[3] = std::abs(a.kurtosis - b.kurtosis);
    if (diffs[3] > 4.0f) diffs[3] = 4.0f;  // Clamp to reasonable range
    diffs[3] /= 4.0f;  // Normalize to [0, 1]

    // Weighted average of differences
    float weighted_diff = 0.0f;
    for (size_t i = 0; i < 4; ++i) {
        weighted_diff += weights[i] * diffs[i];
    }

    // Convert difference to similarity
    return 1.0f - weighted_diff;
}

// ============================================================================
// HistogramSimilarity Implementation
// ============================================================================

float HistogramSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float HistogramSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    std::vector<float> data_a(a.Data().begin(), a.Data().end());
    std::vector<float> data_b(b.Data().begin(), b.Data().end());

    Histogram hist_a(num_bins_);
    Histogram hist_b(num_bins_);

    hist_a.Build(data_a);
    hist_b.Build(data_b);

    return BhattacharyyaCoefficient(hist_a.GetBins(), hist_b.GetBins());
}

float HistogramSimilarity::BhattacharyyaCoefficient(const std::vector<float>& hist_a,
                                                   const std::vector<float>& hist_b) {
    if (hist_a.size() != hist_b.size() || hist_a.empty()) {
        return 0.0f;
    }

    float bc = 0.0f;
    for (size_t i = 0; i < hist_a.size(); ++i) {
        bc += std::sqrt(hist_a[i] * hist_b[i]);
    }

    return bc;
}

// ============================================================================
// KLDivergenceSimilarity Implementation
// ============================================================================

float KLDivergenceSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float KLDivergenceSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    std::vector<float> data_a(a.Data().begin(), a.Data().end());
    std::vector<float> data_b(b.Data().begin(), b.Data().end());

    Histogram hist_a(num_bins_);
    Histogram hist_b(num_bins_);

    hist_a.Build(data_a);
    hist_b.Build(data_b);

    float kl_div = SymmetricKLDivergence(hist_a.GetBins(), hist_b.GetBins(), epsilon_);

    // Convert divergence to similarity: similarity = 1 / (1 + divergence)
    return 1.0f / (1.0f + kl_div);
}

float KLDivergenceSimilarity::SymmetricKLDivergence(const std::vector<float>& hist_a,
                                                   const std::vector<float>& hist_b,
                                                   float epsilon) {
    if (hist_a.size() != hist_b.size() || hist_a.empty()) {
        return std::numeric_limits<float>::infinity();
    }

    // Symmetric KL: (KL(a||b) + KL(b||a)) / 2
    float kl_ab = 0.0f;
    float kl_ba = 0.0f;

    for (size_t i = 0; i < hist_a.size(); ++i) {
        float p = std::max(hist_a[i], epsilon);
        float q = std::max(hist_b[i], epsilon);

        kl_ab += p * std::log(p / q);
        kl_ba += q * std::log(q / p);
    }

    return (kl_ab + kl_ba) / 2.0f;
}

// ============================================================================
// KSSimilarity Implementation
// ============================================================================

float KSSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float KSSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    std::vector<float> data_a(a.Data().begin(), a.Data().end());
    std::vector<float> data_b(b.Data().begin(), b.Data().end());

    float ks_stat = KSStatistic(data_a, data_b);

    // Convert KS statistic [0, 1] to similarity
    return 1.0f - ks_stat;
}

float KSSimilarity::KSStatistic(const std::vector<float>& data_a,
                               const std::vector<float>& data_b) {
    if (data_a.empty() || data_b.empty()) {
        return 1.0f;
    }

    // Sort both datasets
    std::vector<float> sorted_a = data_a;
    std::vector<float> sorted_b = data_b;
    std::sort(sorted_a.begin(), sorted_a.end());
    std::sort(sorted_b.begin(), sorted_b.end());

    // Compute empirical CDFs and find maximum difference
    size_t i = 0, j = 0;
    float max_diff = 0.0f;
    float n_a = static_cast<float>(sorted_a.size());
    float n_b = static_cast<float>(sorted_b.size());

    while (i < sorted_a.size() && j < sorted_b.size()) {
        float val_a = sorted_a[i];
        float val_b = sorted_b[j];

        if (val_a < val_b) {
            // Advance in a
            ++i;
            float cdf_a = static_cast<float>(i) / n_a;
            float cdf_b = static_cast<float>(j) / n_b;
            max_diff = std::max(max_diff, std::abs(cdf_a - cdf_b));
        } else if (val_b < val_a) {
            // Advance in b
            ++j;
            float cdf_a = static_cast<float>(i) / n_a;
            float cdf_b = static_cast<float>(j) / n_b;
            max_diff = std::max(max_diff, std::abs(cdf_a - cdf_b));
        } else {
            // Values are equal, advance both
            ++i;
            ++j;
            float cdf_a = static_cast<float>(i) / n_a;
            float cdf_b = static_cast<float>(j) / n_b;
            max_diff = std::max(max_diff, std::abs(cdf_a - cdf_b));
        }
    }

    // Process remaining elements
    while (i < sorted_a.size()) {
        ++i;
        float cdf_a = static_cast<float>(i) / n_a;
        float cdf_b = 1.0f;
        max_diff = std::max(max_diff, std::abs(cdf_a - cdf_b));
    }

    while (j < sorted_b.size()) {
        ++j;
        float cdf_a = 1.0f;
        float cdf_b = static_cast<float>(j) / n_b;
        max_diff = std::max(max_diff, std::abs(cdf_a - cdf_b));
    }

    return max_diff;
}

// ============================================================================
// ChiSquareSimilarity Implementation
// ============================================================================

float ChiSquareSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float ChiSquareSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    std::vector<float> data_a(a.Data().begin(), a.Data().end());
    std::vector<float> data_b(b.Data().begin(), b.Data().end());

    Histogram hist_a(num_bins_);
    Histogram hist_b(num_bins_);

    hist_a.Build(data_a);
    hist_b.Build(data_b);

    float chi_sq = ChiSquareStatistic(hist_a.GetBins(), hist_b.GetBins());

    // Convert chi-square to similarity: similarity = 1 / (1 + chi_square)
    return 1.0f / (1.0f + chi_sq);
}

float ChiSquareSimilarity::ChiSquareStatistic(const std::vector<float>& hist_a,
                                             const std::vector<float>& hist_b) {
    if (hist_a.size() != hist_b.size() || hist_a.empty()) {
        return std::numeric_limits<float>::infinity();
    }

    float chi_sq = 0.0f;

    for (size_t i = 0; i < hist_a.size(); ++i) {
        float observed = hist_a[i];
        float expected = hist_b[i];

        // Average the two to make symmetric
        float avg = (observed + expected) / 2.0f;

        if (avg > 1e-10f) {
            float diff = observed - expected;
            chi_sq += (diff * diff) / avg;
        }
    }

    return chi_sq;
}

// ============================================================================
// EarthMoverSimilarity Implementation
// ============================================================================

float EarthMoverSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float EarthMoverSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    std::vector<float> data_a(a.Data().begin(), a.Data().end());
    std::vector<float> data_b(b.Data().begin(), b.Data().end());

    Histogram hist_a(num_bins_);
    Histogram hist_b(num_bins_);

    hist_a.Build(data_a);
    hist_b.Build(data_b);

    float emd = EMD1D(hist_a.GetBins(), hist_b.GetBins());

    // Convert EMD to similarity: similarity = 1 / (1 + emd)
    return 1.0f / (1.0f + emd);
}

float EarthMoverSimilarity::EMD1D(const std::vector<float>& hist_a,
                                 const std::vector<float>& hist_b) {
    if (hist_a.size() != hist_b.size() || hist_a.empty()) {
        return std::numeric_limits<float>::infinity();
    }

    // For 1D histograms, EMD can be computed efficiently as the L1 distance
    // between cumulative distributions
    float emd = 0.0f;
    float cum_a = 0.0f;
    float cum_b = 0.0f;

    for (size_t i = 0; i < hist_a.size(); ++i) {
        cum_a += hist_a[i];
        cum_b += hist_b[i];
        emd += std::abs(cum_a - cum_b);
    }

    return emd;
}

} // namespace dpan
