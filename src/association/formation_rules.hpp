// File: src/association/formation_rules.hpp
#pragma once

#include "association/association_edge.hpp"
#include "association/co_occurrence_tracker.hpp"
#include "core/pattern_node.hpp"
#include <optional>
#include <vector>

namespace dpan {

/// AssociationFormationRules: Evaluates co-occurrence data to form associations
///
/// Analyzes pattern co-occurrences and determines:
/// 1. Whether an association should be formed (statistical significance)
/// 2. Type of association (causal, spatial, categorical, etc.)
/// 3. Initial strength of the association
///
/// Uses statistical thresholds and pattern analysis to prevent spurious associations.
///
/// Thread-safety: Not thread-safe. External synchronization required.
class AssociationFormationRules {
public:
    /// Configuration for formation criteria
    struct Config {
        Config() = default;
        /// Minimum co-occurrence count to form association
        uint32_t min_co_occurrences{5};
        /// Minimum chi-squared value (3.841 = p < 0.05, df=1)
        float min_chi_squared{3.841f};
        /// Minimum temporal correlation for causal classification
        float min_temporal_correlation{0.7f};
        /// Minimum spatial context similarity
        float min_spatial_similarity{0.7f};
        /// Minimum categorical similarity
        float min_categorical_similarity{0.6f};
        /// Initial strength for newly formed associations
        float initial_strength{0.5f};
    };

    // ========================================================================
    // Construction
    // ========================================================================

    AssociationFormationRules();
    explicit AssociationFormationRules(const Config& config);
    ~AssociationFormationRules() = default;

    // ========================================================================
    // Formation Evaluation
    // ========================================================================

    /// Evaluate if association should be formed based on statistical criteria
    /// @param tracker Co-occurrence tracker with statistical data
    /// @param p1 First pattern ID
    /// @param p2 Second pattern ID
    /// @return True if association meets formation criteria
    bool ShouldFormAssociation(
        const CoOccurrenceTracker& tracker,
        PatternID p1,
        PatternID p2
    ) const;

    /// Determine type of association based on pattern analysis
    /// @param pattern1 First pattern node
    /// @param pattern2 Second pattern node
    /// @param activation_sequence Temporal sequence of pattern activations
    /// @return Classified association type
    AssociationType ClassifyAssociationType(
        const PatternNode& pattern1,
        const PatternNode& pattern2,
        const std::vector<std::pair<Timestamp, PatternID>>& activation_sequence
    ) const;

    /// Calculate initial strength for new association
    /// @param tracker Co-occurrence tracker
    /// @param p1 First pattern ID
    /// @param p2 Second pattern ID
    /// @param type Association type
    /// @return Initial strength in [0,1]
    float CalculateInitialStrength(
        const CoOccurrenceTracker& tracker,
        PatternID p1,
        PatternID p2,
        AssociationType type
    ) const;

    /// Create association edge from co-occurrence data
    /// @param tracker Co-occurrence tracker
    /// @param pattern1 First pattern node
    /// @param pattern2 Second pattern node
    /// @param activation_sequence Temporal activation sequence
    /// @return Optional association edge (nullopt if criteria not met)
    std::optional<AssociationEdge> CreateAssociation(
        const CoOccurrenceTracker& tracker,
        const PatternNode& pattern1,
        const PatternNode& pattern2,
        const std::vector<std::pair<Timestamp, PatternID>>& activation_sequence
    ) const;

    // ========================================================================
    // Configuration Access
    // ========================================================================

    const Config& GetConfig() const { return config_; }
    void SetConfig(const Config& config) { config_ = config; }

private:
    Config config_;

    // ========================================================================
    // Type Classification Helpers
    // ========================================================================

    /// Check if association is causal (p1 consistently precedes p2)
    /// @param p1 First pattern ID
    /// @param p2 Second pattern ID
    /// @param sequence Temporal activation sequence
    /// @return True if causal relationship detected
    bool IsCausal(
        PatternID p1,
        PatternID p2,
        const std::vector<std::pair<Timestamp, PatternID>>& sequence
    ) const;

    /// Check if association is spatial (similar context patterns)
    /// @param p1 First pattern node
    /// @param p2 Second pattern node
    /// @return True if spatial relationship detected
    bool IsSpatial(const PatternNode& p1, const PatternNode& p2) const;

    /// Check if association is categorical (belong to same cluster)
    /// @param p1 First pattern node
    /// @param p2 Second pattern node
    /// @return True if categorical relationship detected
    bool IsCategorical(const PatternNode& p1, const PatternNode& p2) const;

    /// Check if association is functional (serve similar purposes)
    /// @param p1 First pattern node
    /// @param p2 Second pattern node
    /// @return True if functional relationship detected
    bool IsFunctional(const PatternNode& p1, const PatternNode& p2) const;

    /// Check if association is compositional (one contains the other)
    /// @param p1 First pattern node
    /// @param p2 Second pattern node
    /// @return True if compositional relationship detected
    bool IsCompositional(const PatternNode& p1, const PatternNode& p2) const;
};

} // namespace dpan
