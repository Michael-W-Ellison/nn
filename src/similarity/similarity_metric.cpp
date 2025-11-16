// File: src/similarity/similarity_metric.cpp
#include "similarity/similarity_metric.hpp"
#include <algorithm>
#include <numeric>

namespace dpan {

// ============================================================================
// SimilarityMetric Base Class
// ============================================================================

std::vector<float> SimilarityMetric::ComputeBatch(
        const PatternData& query,
        const std::vector<PatternData>& candidates) const {
    std::vector<float> results;
    results.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        results.push_back(Compute(query, candidate));
    }

    return results;
}

std::vector<float> SimilarityMetric::ComputeBatchFromFeatures(
        const FeatureVector& query,
        const std::vector<FeatureVector>& candidates) const {
    std::vector<float> results;
    results.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        results.push_back(ComputeFromFeatures(query, candidate));
    }

    return results;
}

// ============================================================================
// CompositeMetric Implementation
// ============================================================================

void CompositeMetric::AddMetric(std::shared_ptr<SimilarityMetric> metric, float weight) {
    if (!metric) {
        return;  // Ignore null metrics
    }

    if (weight < 0.0f) {
        weight = 0.0f;  // Clamp negative weights to zero
    }

    metrics_.emplace_back(metric, weight);
    NormalizeWeights();
}

void CompositeMetric::Clear() {
    metrics_.clear();
    normalized_weights_.clear();
}

size_t CompositeMetric::GetMetricCount() const {
    return metrics_.size();
}

float CompositeMetric::Compute(const PatternData& a, const PatternData& b) const {
    if (metrics_.empty()) {
        return 0.0f;  // No metrics, return minimum similarity
    }

    float total_similarity = 0.0f;

    for (size_t i = 0; i < metrics_.size(); ++i) {
        float similarity = metrics_[i].first->Compute(a, b);
        total_similarity += similarity * normalized_weights_[i];
    }

    return total_similarity;
}

float CompositeMetric::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (metrics_.empty()) {
        return 0.0f;
    }

    float total_similarity = 0.0f;

    for (size_t i = 0; i < metrics_.size(); ++i) {
        float similarity = metrics_[i].first->ComputeFromFeatures(a, b);
        total_similarity += similarity * normalized_weights_[i];
    }

    return total_similarity;
}

std::vector<float> CompositeMetric::ComputeBatch(
        const PatternData& query,
        const std::vector<PatternData>& candidates) const {
    if (metrics_.empty()) {
        return std::vector<float>(candidates.size(), 0.0f);
    }

    // Initialize results
    std::vector<float> results(candidates.size(), 0.0f);

    // Accumulate weighted similarities from each metric
    for (size_t i = 0; i < metrics_.size(); ++i) {
        auto metric_results = metrics_[i].first->ComputeBatch(query, candidates);

        for (size_t j = 0; j < candidates.size(); ++j) {
            results[j] += metric_results[j] * normalized_weights_[i];
        }
    }

    return results;
}

bool CompositeMetric::IsSymmetric() const {
    // Composite is symmetric if all constituent metrics are symmetric
    return std::all_of(metrics_.begin(), metrics_.end(),
        [](const auto& pair) { return pair.first->IsSymmetric(); });
}

void CompositeMetric::NormalizeWeights() {
    normalized_weights_.clear();

    if (metrics_.empty()) {
        return;
    }

    // Calculate sum of weights
    float total_weight = 0.0f;
    for (const auto& [metric, weight] : metrics_) {
        total_weight += weight;
    }

    // Normalize to sum to 1.0
    if (total_weight > 0.0f) {
        for (const auto& [metric, weight] : metrics_) {
            normalized_weights_.push_back(weight / total_weight);
        }
    } else {
        // All weights are zero, use uniform distribution
        float uniform_weight = 1.0f / metrics_.size();
        for (size_t i = 0; i < metrics_.size(); ++i) {
            normalized_weights_.push_back(uniform_weight);
        }
    }
}

} // namespace dpan
