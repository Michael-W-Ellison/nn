// File: src/memory/interference.hpp
#pragma once

#include "core/types.hpp"
#include "core/pattern_data.hpp"
#include <memory>

namespace dpan {

// Forward declarations
class PatternNode;
class SimilarityMetric;

/**
 * @brief Models memory interference between similar patterns
 *
 * Interference occurs when similar patterns compete for memory resources.
 * Based on the formula:
 *   I(p1, p2) = similarity(p1, p2) × strength(p2)
 *   s'(p1) = s(p1) × (1 - α × I_total(p1))
 *
 * Where α is the interference factor (default 0.1)
 */
class InterferenceCalculator {
public:
    struct Config {
        float interference_factor{0.1f};        // α parameter [0.0, 1.0]
        float similarity_threshold{0.5f};       // Min similarity for interference [0.0, 1.0]

        bool IsValid() const {
            return interference_factor >= 0.0f && interference_factor <= 1.0f &&
                   similarity_threshold >= 0.0f && similarity_threshold <= 1.0f;
        }
    };

    InterferenceCalculator();
    explicit InterferenceCalculator(const Config& config);
    explicit InterferenceCalculator(std::shared_ptr<SimilarityMetric> similarity_metric);
    InterferenceCalculator(const Config& config, std::shared_ptr<SimilarityMetric> similarity_metric);

    /**
     * @brief Calculate interference from source to target pattern
     *
     * I(target, source) = similarity(target, source) × strength(source)
     *
     * @param target_features Target pattern features
     * @param source_features Source pattern features
     * @param source_strength Strength of source pattern [0.0, 1.0]
     * @return Interference amount [0.0, 1.0]
     */
    float CalculateInterference(
        const FeatureVector& target_features,
        const FeatureVector& source_features,
        float source_strength
    ) const;

    /**
     * @brief Apply interference effect to pattern strength
     *
     * s'(p) = s(p) × (1 - α × total_interference)
     *
     * @param original_strength Original pattern strength [0.0, 1.0]
     * @param total_interference Total interference affecting pattern [0.0, 1.0]
     * @return Reduced strength after interference [0.0, 1.0]
     */
    float ApplyInterference(
        float original_strength,
        float total_interference
    ) const;

    void SetConfig(const Config& config);
    const Config& GetConfig() const { return config_; }

    void SetSimilarityMetric(std::shared_ptr<SimilarityMetric> metric);
    std::shared_ptr<SimilarityMetric> GetSimilarityMetric() const { return similarity_metric_; }

private:
    Config config_;
    std::shared_ptr<SimilarityMetric> similarity_metric_;

    bool AreSimilarEnough(const FeatureVector& f1, const FeatureVector& f2) const;
};

} // namespace dpan
