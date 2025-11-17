// File: src/association/competitive_learner.cpp
#include "association/competitive_learner.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace dpan {
namespace CompetitiveLearner {

// ============================================================================
// Single Pattern Competition
// ============================================================================

bool ApplyCompetition(
    AssociationMatrix& matrix,
    PatternID pattern,
    const Config& config
) {
    // Get all outgoing associations
    auto outgoing = matrix.GetOutgoingAssociations(pattern);

    if (outgoing.empty() || outgoing.size() < config.min_competing_associations) {
        return false;  // Not enough associations to compete
    }

    // Filter by strength threshold
    auto eligible = FilterByStrength(outgoing, config.min_strength_threshold);

    if (eligible.empty() || eligible.size() < config.min_competing_associations) {
        return false;  // Not enough eligible associations
    }

    // Filter by type if configured
    if (config.filter_by_type && !config.allowed_types.empty()) {
        std::vector<const AssociationEdge*> type_filtered;
        for (AssociationType type : config.allowed_types) {
            auto typed = FilterByType(eligible, type);
            type_filtered.insert(type_filtered.end(), typed.begin(), typed.end());
        }
        eligible = type_filtered;

        if (eligible.empty() || eligible.size() < config.min_competing_associations) {
            return false;
        }
    }

    // Find the strongest association
    const AssociationEdge* winner = FindStrongest(eligible);

    if (!winner) {
        return false;
    }

    // Apply competition: boost winner, suppress losers
    for (const auto* edge : eligible) {
        float old_strength = edge->GetStrength();
        float new_strength;

        if (edge == winner) {
            // Boost winner
            new_strength = CalculateWinnerStrength(old_strength, config.competition_factor);
        } else {
            // Suppress loser
            new_strength = CalculateLoserStrength(old_strength, config.competition_factor);
        }

        // Clone edge and update strength
        auto updated_edge = edge->Clone();
        updated_edge->SetStrength(new_strength);
        matrix.UpdateAssociation(edge->GetSource(), edge->GetTarget(), *updated_edge);
    }

    return true;
}

bool ApplyCompetitionIncoming(
    AssociationMatrix& matrix,
    PatternID pattern,
    const Config& config
) {
    // Get all incoming associations
    auto incoming = matrix.GetIncomingAssociations(pattern);

    if (incoming.empty() || incoming.size() < config.min_competing_associations) {
        return false;  // Not enough associations to compete
    }

    // Filter by strength threshold
    auto eligible = FilterByStrength(incoming, config.min_strength_threshold);

    if (eligible.empty() || eligible.size() < config.min_competing_associations) {
        return false;  // Not enough eligible associations
    }

    // Filter by type if configured
    if (config.filter_by_type && !config.allowed_types.empty()) {
        std::vector<const AssociationEdge*> type_filtered;
        for (AssociationType type : config.allowed_types) {
            auto typed = FilterByType(eligible, type);
            type_filtered.insert(type_filtered.end(), typed.begin(), typed.end());
        }
        eligible = type_filtered;

        if (eligible.empty() || eligible.size() < config.min_competing_associations) {
            return false;
        }
    }

    // Find the strongest association
    const AssociationEdge* winner = FindStrongest(eligible);

    if (!winner) {
        return false;
    }

    // Apply competition: boost winner, suppress losers
    for (const auto* edge : eligible) {
        float old_strength = edge->GetStrength();
        float new_strength;

        if (edge == winner) {
            // Boost winner
            new_strength = CalculateWinnerStrength(old_strength, config.competition_factor);
        } else {
            // Suppress loser
            new_strength = CalculateLoserStrength(old_strength, config.competition_factor);
        }

        // Clone edge and update strength
        auto updated_edge = edge->Clone();
        updated_edge->SetStrength(new_strength);
        matrix.UpdateAssociation(edge->GetSource(), edge->GetTarget(), *updated_edge);
    }

    return true;
}

// ============================================================================
// Typed Competition
// ============================================================================

bool ApplyTypedCompetition(
    AssociationMatrix& matrix,
    PatternID pattern,
    AssociationType type,
    const Config& config
) {
    // Get all outgoing associations
    auto outgoing = matrix.GetOutgoingAssociations(pattern);

    if (outgoing.empty()) {
        return false;
    }

    // Filter by type
    auto typed = FilterByType(outgoing, type);

    if (typed.empty() || typed.size() < config.min_competing_associations) {
        return false;  // Not enough associations of this type
    }

    // Filter by strength threshold
    auto eligible = FilterByStrength(typed, config.min_strength_threshold);

    if (eligible.empty() || eligible.size() < config.min_competing_associations) {
        return false;
    }

    // Find the strongest association
    const AssociationEdge* winner = FindStrongest(eligible);

    if (!winner) {
        return false;
    }

    // Apply competition: boost winner, suppress losers
    for (const auto* edge : eligible) {
        float old_strength = edge->GetStrength();
        float new_strength;

        if (edge == winner) {
            // Boost winner
            new_strength = CalculateWinnerStrength(old_strength, config.competition_factor);
        } else {
            // Suppress loser
            new_strength = CalculateLoserStrength(old_strength, config.competition_factor);
        }

        // Clone edge and update strength
        auto updated_edge = edge->Clone();
        updated_edge->SetStrength(new_strength);
        matrix.UpdateAssociation(edge->GetSource(), edge->GetTarget(), *updated_edge);
    }

    return true;
}

// ============================================================================
// Batch Competition
// ============================================================================

size_t ApplyCompetitionBatch(
    AssociationMatrix& matrix,
    const std::vector<PatternID>& patterns,
    const Config& config
) {
    size_t applied_count = 0;

    for (const auto& pattern : patterns) {
        if (ApplyCompetition(matrix, pattern, config)) {
            applied_count++;
        }
    }

    return applied_count;
}

size_t ApplyCompetitionAll(
    AssociationMatrix& matrix,
    const Config& config
) {
    // Note: This is limited by the AssociationMatrix API
    // Without a method to get all unique patterns, we cannot efficiently
    // apply competition to all patterns in the matrix
    // This would require AssociationMatrix to expose a GetAllPatterns() method
    // For now, return 0
    // TODO: Update when AssociationMatrix provides pattern iteration
    return 0;
}

// ============================================================================
// Utility Functions
// ============================================================================

const AssociationEdge* FindStrongest(
    const std::vector<const AssociationEdge*>& associations
) {
    if (associations.empty()) {
        return nullptr;
    }

    const AssociationEdge* strongest = associations[0];
    float max_strength = strongest->GetStrength();

    for (const auto* edge : associations) {
        float strength = edge->GetStrength();
        if (strength > max_strength) {
            max_strength = strength;
            strongest = edge;
        }
    }

    return strongest;
}

float CalculateWinnerStrength(
    float current_strength,
    float competition_factor
) {
    // Formula: s_new = s_old + β × (1 - s_old)
    // This boosts the strength towards 1.0
    float boost = competition_factor * (1.0f - current_strength);
    float new_strength = current_strength + boost;

    // Ensure strength stays in valid range [0, 1]
    return std::clamp(new_strength, 0.0f, 1.0f);
}

float CalculateLoserStrength(
    float current_strength,
    float competition_factor
) {
    // Formula: s_new = s_old × (1 - β)
    // This suppresses the strength towards 0.0
    float new_strength = current_strength * (1.0f - competition_factor);

    // Ensure strength stays in valid range [0, 1]
    return std::clamp(new_strength, 0.0f, 1.0f);
}

std::vector<const AssociationEdge*> FilterByType(
    const std::vector<const AssociationEdge*>& associations,
    AssociationType type
) {
    std::vector<const AssociationEdge*> filtered;

    for (const auto* edge : associations) {
        if (edge->GetType() == type) {
            filtered.push_back(edge);
        }
    }

    return filtered;
}

std::vector<const AssociationEdge*> FilterByStrength(
    const std::vector<const AssociationEdge*>& associations,
    float min_threshold
) {
    std::vector<const AssociationEdge*> filtered;

    for (const auto* edge : associations) {
        if (edge->GetStrength() >= min_threshold) {
            filtered.push_back(edge);
        }
    }

    return filtered;
}

// ============================================================================
// Statistics
// ============================================================================

CompetitionStats AnalyzeCompetition(
    const AssociationMatrix& matrix,
    PatternID pattern,
    const Config& config
) {
    CompetitionStats stats;
    stats.patterns_processed = 1;

    // Get all outgoing associations
    auto outgoing = matrix.GetOutgoingAssociations(pattern);

    if (outgoing.empty() || outgoing.size() < config.min_competing_associations) {
        return stats;  // No competition to analyze
    }

    // Filter by strength threshold
    auto eligible = FilterByStrength(outgoing, config.min_strength_threshold);

    if (eligible.empty() || eligible.size() < config.min_competing_associations) {
        return stats;
    }

    // Find the strongest association
    const AssociationEdge* winner = FindStrongest(eligible);

    if (!winner) {
        return stats;
    }

    // Calculate statistics
    stats.competitions_applied = 1;

    float total_before = 0.0f;
    float total_after = 0.0f;
    float total_winner_boost = 0.0f;
    float total_loser_suppression = 0.0f;
    size_t winner_count = 0;
    size_t loser_count = 0;

    for (const auto* edge : eligible) {
        float old_strength = edge->GetStrength();
        float new_strength;

        total_before += old_strength;

        if (edge == winner) {
            // Boost winner
            new_strength = CalculateWinnerStrength(old_strength, config.competition_factor);
            total_winner_boost += (new_strength - old_strength);
            winner_count++;
        } else {
            // Suppress loser
            new_strength = CalculateLoserStrength(old_strength, config.competition_factor);
            total_loser_suppression += (old_strength - new_strength);
            loser_count++;
        }

        total_after += new_strength;
    }

    stats.winners_boosted = winner_count;
    stats.losers_suppressed = loser_count;
    stats.total_strength_before = total_before;
    stats.total_strength_after = total_after;

    if (winner_count > 0) {
        stats.average_winner_boost = total_winner_boost / winner_count;
    }

    if (loser_count > 0) {
        stats.average_loser_suppression = total_loser_suppression / loser_count;
    }

    return stats;
}

CompetitionStats ApplyCompetitionWithStats(
    AssociationMatrix& matrix,
    PatternID pattern,
    const Config& config
) {
    // First analyze to get statistics
    CompetitionStats stats = AnalyzeCompetition(matrix, pattern, config);

    // Then apply the actual competition
    bool applied = ApplyCompetition(matrix, pattern, config);

    if (!applied) {
        // Reset stats if competition wasn't actually applied
        stats.competitions_applied = 0;
    }

    return stats;
}

} // namespace CompetitiveLearner
} // namespace dpan
