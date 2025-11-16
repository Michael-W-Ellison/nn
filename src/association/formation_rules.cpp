// File: src/association/formation_rules.cpp
#include "association/formation_rules.hpp"
#include <algorithm>
#include <cmath>

namespace dpan {

// ============================================================================
// Construction
// ============================================================================

AssociationFormationRules::AssociationFormationRules()
    : AssociationFormationRules(Config())
{
}

AssociationFormationRules::AssociationFormationRules(const Config& config)
    : config_(config)
{
}

// ============================================================================
// Formation Evaluation
// ============================================================================

bool AssociationFormationRules::ShouldFormAssociation(
    const CoOccurrenceTracker& tracker,
    PatternID p1,
    PatternID p2
) const {
    // Check minimum co-occurrence count
    uint32_t co_count = tracker.GetCoOccurrenceCount(p1, p2);
    if (co_count < config_.min_co_occurrences) {
        return false;
    }

    // Check statistical significance using chi-squared test
    if (!tracker.IsSignificant(p1, p2)) {
        return false;
    }

    // Additional chi-squared threshold check
    float chi_squared = tracker.GetChiSquared(p1, p2);
    if (chi_squared < config_.min_chi_squared) {
        return false;
    }

    return true;
}

AssociationType AssociationFormationRules::ClassifyAssociationType(
    const PatternNode& pattern1,
    const PatternNode& pattern2,
    const std::vector<std::pair<Timestamp, PatternID>>& activation_sequence
) const {
    // Try to classify in order of specificity (most specific first)

    // 1. Compositional (most specific - one pattern contains the other)
    if (IsCompositional(pattern1, pattern2)) {
        return AssociationType::COMPOSITIONAL;
    }

    // 2. Causal (requires temporal data - p1 consistently precedes p2)
    if (IsCausal(pattern1.GetID(), pattern2.GetID(), activation_sequence)) {
        return AssociationType::CAUSAL;
    }

    // 3. Functional (patterns serve similar role in different contexts)
    if (IsFunctional(pattern1, pattern2)) {
        return AssociationType::FUNCTIONAL;
    }

    // 4. Spatial (appear in similar spatial/contextual configurations)
    if (IsSpatial(pattern1, pattern2)) {
        return AssociationType::SPATIAL;
    }

    // 5. Categorical (default fallback - patterns cluster together)
    return AssociationType::CATEGORICAL;
}

float AssociationFormationRules::CalculateInitialStrength(
    const CoOccurrenceTracker& tracker,
    PatternID p1,
    PatternID p2,
    AssociationType type
) const {
    // Base strength from co-occurrence probability
    float prob = tracker.GetCoOccurrenceProbability(p1, p2);

    // Normalize to [0, 0.7] range with logarithmic scaling to leave room for bonuses
    // log(1 + x) provides good compression for high values
    float base_strength = 0.7f * std::log(1.0f + prob) / std::log(2.0f);

    // Boost based on statistical significance (chi-squared value)
    float chi_squared = tracker.GetChiSquared(p1, p2);
    // Use logarithmic scaling for chi-squared too
    float significance_boost = std::min(0.15f, std::log(1.0f + chi_squared) / 25.0f);

    // Type-specific strength adjustments (additive)
    float type_bonus = 0.0f;
    switch (type) {
        case AssociationType::CAUSAL:
        case AssociationType::COMPOSITIONAL:
            // Stronger types get a bonus
            type_bonus = 0.15f;
            break;
        case AssociationType::FUNCTIONAL:
            type_bonus = 0.08f;
            break;
        case AssociationType::SPATIAL:
        case AssociationType::CATEGORICAL:
            type_bonus = 0.0f;
            break;
    }

    // Combine factors and clamp to [0, 1]
    float strength = base_strength + significance_boost + type_bonus;
    return std::clamp(strength, 0.0f, 1.0f);
}

std::optional<AssociationEdge> AssociationFormationRules::CreateAssociation(
    const CoOccurrenceTracker& tracker,
    const PatternNode& pattern1,
    const PatternNode& pattern2,
    const std::vector<std::pair<Timestamp, PatternID>>& activation_sequence
) const {
    PatternID p1 = pattern1.GetID();
    PatternID p2 = pattern2.GetID();

    // Check formation criteria
    if (!ShouldFormAssociation(tracker, p1, p2)) {
        return std::nullopt;
    }

    // Classify association type
    AssociationType type = ClassifyAssociationType(pattern1, pattern2, activation_sequence);

    // Calculate initial strength
    float strength = CalculateInitialStrength(tracker, p1, p2, type);

    // Create edge with computed parameters
    AssociationEdge edge(p1, p2, type, strength);

    // Set co-occurrence count for tracking
    uint32_t co_count = tracker.GetCoOccurrenceCount(p1, p2);
    for (uint32_t i = 0; i < co_count; ++i) {
        edge.IncrementCoOccurrence();
    }

    return edge;
}

// ============================================================================
// Type Classification Helpers
// ============================================================================

bool AssociationFormationRules::IsCausal(
    PatternID p1,
    PatternID p2,
    const std::vector<std::pair<Timestamp, PatternID>>& sequence
) const {
    if (sequence.empty()) {
        return false;
    }

    // Count consecutive pairs where p1 precedes p2 within a short time window
    int p1_before_p2 = 0;
    int p2_before_p1 = 0;

    // Maximum time gap to consider patterns as causally related (500ms)
    const auto max_gap = std::chrono::milliseconds(500);

    for (size_t i = 0; i + 1 < sequence.size(); ++i) {
        PatternID current = sequence[i].second;
        Timestamp current_time = sequence[i].first;
        PatternID next = sequence[i + 1].second;
        Timestamp next_time = sequence[i + 1].first;

        // Only count if patterns appear within the temporal window
        auto time_gap = next_time - current_time;
        if (time_gap > max_gap) {
            continue;  // Too far apart to be causally related
        }

        // Check if this is a consecutive p1->p2 or p2->p1 pair
        if (current == p1 && next == p2) {
            p1_before_p2++;
        } else if (current == p2 && next == p1) {
            p2_before_p1++;
        }
    }

    // Causal if one direction is significantly more common
    int total = p1_before_p2 + p2_before_p1;
    if (total == 0) {
        return false;
    }

    // Require at least 70% in one direction (configurable threshold)
    float ratio = static_cast<float>(std::max(p1_before_p2, p2_before_p1)) / total;
    return ratio >= config_.min_temporal_correlation;
}

bool AssociationFormationRules::IsSpatial(
    const PatternNode& p1,
    const PatternNode& p2
) const {
    // Check if patterns have similar context profiles
    // This would compare spatial/contextual features from pattern activations

    // For now, we use a simple heuristic based on pattern type
    // In Phase 1, patterns don't have rich spatial context yet
    // This will be enhanced in later phases

    // Placeholder: Check if both patterns are atomic (may appear in similar contexts)
    if (p1.GetType() == PatternType::ATOMIC && p2.GetType() == PatternType::ATOMIC) {
        // Could check context similarity here if available
        // For now, return false to avoid over-classification as spatial
        return false;
    }

    return false;
}

bool AssociationFormationRules::IsCategorical(
    const PatternNode& p1,
    const PatternNode& p2
) const {
    // Check if patterns belong to same category/cluster
    // This would use similarity metrics from Phase 1

    // Patterns are categorical if they:
    // 1. Have the same type
    // 2. Could potentially be clustered together

    // Check type similarity
    if (p1.GetType() == p2.GetType()) {
        // Same type suggests potential categorical relationship
        // This is a weak signal, so only used as fallback
        return true;
    }

    // Default: patterns can be categorized together
    return true;
}

bool AssociationFormationRules::IsFunctional(
    const PatternNode& p1,
    const PatternNode& p2
) const {
    // Patterns are functional if they serve similar purposes in different contexts
    // This would check:
    // 1. Similar association profiles (same neighbors)
    // 2. Similar activation patterns
    // 3. Substitutability in different contexts

    // For now, we use a heuristic based on pattern structure
    // Composite patterns of same size might be functional equivalents
    if (p1.GetType() == PatternType::COMPOSITE && p2.GetType() == PatternType::COMPOSITE) {
        // Check if they have similar number of sub-patterns
        const auto& p1_subs = p1.GetSubPatterns();
        const auto& p2_subs = p2.GetSubPatterns();

        if (p1_subs.size() == p2_subs.size() && p1_subs.size() > 0) {
            // Similar complexity might indicate functional similarity
            // But this is weak, so we don't return true yet
            // Will be enhanced with actual association profile analysis
        }
    }

    return false;
}

bool AssociationFormationRules::IsCompositional(
    const PatternNode& p1,
    const PatternNode& p2
) const {
    // Check if one pattern contains the other as a sub-pattern
    // This is the strongest form of relationship

    const auto& p1_subs = p1.GetSubPatterns();
    const auto& p2_subs = p2.GetSubPatterns();

    // Does p1 contain p2?
    if (std::find(p1_subs.begin(), p1_subs.end(), p2.GetID()) != p1_subs.end()) {
        return true;
    }

    // Does p2 contain p1?
    if (std::find(p2_subs.begin(), p2_subs.end(), p1.GetID()) != p2_subs.end()) {
        return true;
    }

    return false;
}

} // namespace dpan
