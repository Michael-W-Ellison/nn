// File: src/memory/interference.cpp
#include "memory/interference.hpp"
#include "similarity/similarity_metric.hpp"
#include <algorithm>
#include <stdexcept>

namespace dpan {

InterferenceCalculator::InterferenceCalculator()
    : config_(), similarity_metric_(nullptr) {
}

InterferenceCalculator::InterferenceCalculator(const Config& config)
    : config_(config), similarity_metric_(nullptr) {
    if (!config_.IsValid()) {
        throw std::invalid_argument("Invalid InterferenceCalculator configuration");
    }
}

InterferenceCalculator::InterferenceCalculator(std::shared_ptr<SimilarityMetric> similarity_metric)
    : config_(), similarity_metric_(similarity_metric) {
}

InterferenceCalculator::InterferenceCalculator(
    const Config& config,
    std::shared_ptr<SimilarityMetric> similarity_metric
) : config_(config), similarity_metric_(similarity_metric) {
    if (!config_.IsValid()) {
        throw std::invalid_argument("Invalid InterferenceCalculator configuration");
    }
}

void InterferenceCalculator::SetConfig(const Config& config) {
    if (!config.IsValid()) {
        throw std::invalid_argument("Invalid InterferenceCalculator configuration");
    }
    config_ = config;
}

void InterferenceCalculator::SetSimilarityMetric(std::shared_ptr<SimilarityMetric> metric) {
    similarity_metric_ = metric;
}

float InterferenceCalculator::CalculateInterference(
    const FeatureVector& target_features,
    const FeatureVector& source_features,
    float source_strength
) const {
    // Validate inputs
    if (source_strength < 0.0f || source_strength > 1.0f) {
        return 0.0f;
    }

    // Need similarity metric
    if (!similarity_metric_) {
        return 0.0f;
    }

    // Check if similar enough to interfere
    if (!AreSimilarEnough(target_features, source_features)) {
        return 0.0f;
    }

    // Compute similarity
    float similarity = similarity_metric_->ComputeFromFeatures(target_features, source_features);

    // I(target, source) = similarity × strength(source)
    float interference = similarity * source_strength;

    // Clamp to valid range
    return std::max(0.0f, std::min(1.0f, interference));
}

float InterferenceCalculator::ApplyInterference(
    float original_strength,
    float total_interference
) const {
    // Validate inputs
    if (original_strength < 0.0f || original_strength > 1.0f) {
        return original_strength;
    }

    total_interference = std::max(0.0f, std::min(1.0f, total_interference));

    // s' = s × (1 - α × I_total)
    float reduction_factor = 1.0f - (config_.interference_factor * total_interference);
    reduction_factor = std::max(0.0f, reduction_factor);

    float new_strength = original_strength * reduction_factor;

    // Ensure result is valid
    return std::max(0.0f, std::min(new_strength, original_strength));
}

bool InterferenceCalculator::AreSimilarEnough(
    const FeatureVector& f1,
    const FeatureVector& f2
) const {
    if (!similarity_metric_) {
        return false;
    }

    float similarity = similarity_metric_->ComputeFromFeatures(f1, f2);
    return similarity >= config_.similarity_threshold;
}

} // namespace dpan
