// File: src/similarity/similarity_metric.hpp
#pragma once

#include "core/pattern_data.hpp"
#include <vector>
#include <string>
#include <memory>

namespace dpan {

/// Abstract base class for similarity metrics
///
/// Defines the interface for computing similarity between patterns.
/// Similarity values are normalized to [0.0, 1.0] where:
/// - 0.0 = completely dissimilar
/// - 1.0 = identical
///
/// Implementations can work with PatternData (full pattern) or
/// FeatureVector (extracted features) for performance.
class SimilarityMetric {
public:
    virtual ~SimilarityMetric() = default;

    /// Compute similarity between two patterns using full pattern data
    /// @param a First pattern
    /// @param b Second pattern
    /// @return Similarity score [0.0, 1.0]
    virtual float Compute(const PatternData& a, const PatternData& b) const = 0;

    /// Compute similarity using feature vectors (typically faster)
    /// @param a First feature vector
    /// @param b Second feature vector
    /// @return Similarity score [0.0, 1.0]
    virtual float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const = 0;

    /// Compute similarity between query and multiple candidates (batch)
    /// Default implementation calls Compute for each candidate
    /// @param query Query pattern
    /// @param candidates Candidate patterns
    /// @return Vector of similarity scores
    virtual std::vector<float> ComputeBatch(
        const PatternData& query,
        const std::vector<PatternData>& candidates) const;

    /// Compute similarity using feature vectors (batch)
    /// @param query Query feature vector
    /// @param candidates Candidate feature vectors
    /// @return Vector of similarity scores
    virtual std::vector<float> ComputeBatchFromFeatures(
        const FeatureVector& query,
        const std::vector<FeatureVector>& candidates) const;

    /// Get the name of this metric
    /// @return Metric name
    virtual std::string GetName() const = 0;

    /// Check if metric is symmetric: similarity(a,b) == similarity(b,a)
    /// @return true if symmetric
    virtual bool IsSymmetric() const { return true; }

    /// Check if metric satisfies triangle inequality
    /// (required for true distance metrics)
    /// @return true if satisfies triangle inequality
    virtual bool IsMetric() const { return false; }
};

/// Composite metric: weighted combination of multiple metrics
///
/// Combines multiple similarity metrics using weighted averaging.
/// Useful for multi-modal similarity or combining different aspects.
///
/// Example:
///   CompositeMetric composite;
///   composite.AddMetric(geometric_metric, 0.5);
///   composite.AddMetric(statistical_metric, 0.5);
///   float similarity = composite.Compute(pattern1, pattern2);
class CompositeMetric : public SimilarityMetric {
public:
    /// Default constructor
    CompositeMetric() = default;

    /// Add a metric with a weight
    /// @param metric Similarity metric to add
    /// @param weight Weight for this metric (will be normalized)
    void AddMetric(std::shared_ptr<SimilarityMetric> metric, float weight);

    /// Remove all metrics
    void Clear();

    /// Get number of constituent metrics
    /// @return Number of metrics
    size_t GetMetricCount() const;

    /// Compute weighted average of all constituent metrics
    /// @param a First pattern
    /// @param b Second pattern
    /// @return Weighted average similarity [0.0, 1.0]
    float Compute(const PatternData& a, const PatternData& b) const override;

    /// Compute weighted average using feature vectors
    /// @param a First feature vector
    /// @param b Second feature vector
    /// @return Weighted average similarity [0.0, 1.0]
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;

    /// Batch computation using weighted average
    /// @param query Query pattern
    /// @param candidates Candidate patterns
    /// @return Vector of similarity scores
    std::vector<float> ComputeBatch(
        const PatternData& query,
        const std::vector<PatternData>& candidates) const override;

    /// Get metric name
    /// @return "Composite"
    std::string GetName() const override { return "Composite"; }

    /// Composite is symmetric if all constituent metrics are symmetric
    /// @return true if all metrics are symmetric
    bool IsSymmetric() const override;

private:
    /// List of (metric, weight) pairs
    std::vector<std::pair<std::shared_ptr<SimilarityMetric>, float>> metrics_;

    /// Normalized weights (sum to 1.0)
    std::vector<float> normalized_weights_;

    /// Recompute normalized weights after adding metrics
    void NormalizeWeights();
};

} // namespace dpan
