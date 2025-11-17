// File: src/association/strength_normalizer.cpp
#include "association/strength_normalizer.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace dpan {
namespace StrengthNormalizer {

// ============================================================================
// Single Pattern Normalization
// ============================================================================

bool NormalizeOutgoing(
    AssociationMatrix& matrix,
    PatternID pattern,
    const Config& config
) {
    // Get all outgoing associations
    auto outgoing = matrix.GetOutgoingAssociations(pattern);

    if (outgoing.empty()) {
        return false;  // No associations to normalize
    }

    // Compute total strength
    float total_strength = 0.0f;
    std::vector<const AssociationEdge*> edges_to_normalize;

    for (const auto* edge : outgoing) {
        float strength = edge->GetStrength();
        if (strength >= config.min_strength_threshold || config.preserve_zeros) {
            total_strength += strength;
            edges_to_normalize.push_back(edge);
        }
    }

    // If total is already approximately 1.0 or zero, no normalization needed
    if (total_strength < 1e-6f || std::abs(total_strength - 1.0f) < 0.001f) {
        return false;
    }

    // Normalize each edge
    float normalization_factor = 1.0f / total_strength;

    for (const auto* edge : edges_to_normalize) {
        float old_strength = edge->GetStrength();
        float new_strength = old_strength * normalization_factor;

        // Clone edge and update strength
        auto updated_edge = edge->Clone();
        updated_edge->SetStrength(new_strength);
        matrix.UpdateAssociation(edge->GetSource(), edge->GetTarget(), *updated_edge);
    }

    return true;
}

bool NormalizeIncoming(
    AssociationMatrix& matrix,
    PatternID pattern,
    const Config& config
) {
    // Get all incoming associations
    auto incoming = matrix.GetIncomingAssociations(pattern);

    if (incoming.empty()) {
        return false;  // No associations to normalize
    }

    // Compute total strength
    float total_strength = 0.0f;
    std::vector<const AssociationEdge*> edges_to_normalize;

    for (const auto* edge : incoming) {
        float strength = edge->GetStrength();
        if (strength >= config.min_strength_threshold || config.preserve_zeros) {
            total_strength += strength;
            edges_to_normalize.push_back(edge);
        }
    }

    // If total is already approximately 1.0 or zero, no normalization needed
    if (total_strength < 1e-6f || std::abs(total_strength - 1.0f) < 0.001f) {
        return false;
    }

    // Normalize each edge
    float normalization_factor = 1.0f / total_strength;

    for (const auto* edge : edges_to_normalize) {
        float old_strength = edge->GetStrength();
        float new_strength = old_strength * normalization_factor;

        // Clone edge and update strength
        auto updated_edge = edge->Clone();
        updated_edge->SetStrength(new_strength);
        matrix.UpdateAssociation(edge->GetSource(), edge->GetTarget(), *updated_edge);
    }

    return true;
}

std::pair<bool, bool> NormalizeBidirectional(
    AssociationMatrix& matrix,
    PatternID pattern,
    const Config& config
) {
    bool outgoing_normalized = NormalizeOutgoing(matrix, pattern, config);
    bool incoming_normalized = NormalizeIncoming(matrix, pattern, config);

    return {outgoing_normalized, incoming_normalized};
}

// ============================================================================
// Batch Normalization
// ============================================================================

size_t NormalizeOutgoingBatch(
    AssociationMatrix& matrix,
    const std::vector<PatternID>& patterns,
    const Config& config
) {
    size_t normalized_count = 0;

    for (const auto& pattern : patterns) {
        if (NormalizeOutgoing(matrix, pattern, config)) {
            normalized_count++;
        }
    }

    return normalized_count;
}

size_t NormalizeAll(
    AssociationMatrix& matrix,
    const Config& config
) {
    // Note: This is limited by the AssociationMatrix API
    // Without a method to get all unique patterns, we cannot efficiently
    // normalize all patterns in the matrix
    // This would require AssociationMatrix to expose a GetAllPatterns() method
    // For now, return 0
    // TODO: Update when AssociationMatrix provides pattern iteration
    return 0;
}

// ============================================================================
// Utility Functions
// ============================================================================

float GetOutgoingStrengthSum(
    const AssociationMatrix& matrix,
    PatternID pattern
) {
    auto outgoing = matrix.GetOutgoingAssociations(pattern);

    float total = 0.0f;
    for (const auto* edge : outgoing) {
        total += edge->GetStrength();
    }

    return total;
}

float GetIncomingStrengthSum(
    const AssociationMatrix& matrix,
    PatternID pattern
) {
    auto incoming = matrix.GetIncomingAssociations(pattern);

    float total = 0.0f;
    for (const auto* edge : incoming) {
        total += edge->GetStrength();
    }

    return total;
}

bool IsNormalized(
    const AssociationMatrix& matrix,
    PatternID pattern,
    float tolerance
) {
    float sum = GetOutgoingStrengthSum(matrix, pattern);

    // If no outgoing edges, consider it normalized
    if (sum < 1e-6f) {
        return true;
    }

    return std::abs(sum - 1.0f) <= tolerance;
}

float GetNormalizationFactor(
    const AssociationMatrix& matrix,
    PatternID pattern
) {
    float sum = GetOutgoingStrengthSum(matrix, pattern);

    if (sum < 1e-6f) {
        return 1.0f;  // Avoid division by zero
    }

    return 1.0f / sum;
}

// ============================================================================
// Statistics
// ============================================================================

NormalizationStats AnalyzeNormalization(
    const AssociationMatrix& matrix,
    const Config& config
) {
    NormalizationStats stats;

    // Note: This is limited by the AssociationMatrix API
    // Without a method to iterate over all unique patterns, we cannot
    // efficiently analyze the entire matrix
    // This would require AssociationMatrix to expose a GetAllPatterns() method
    // For now, return empty stats
    // TODO: Update when AssociationMatrix provides pattern iteration

    stats.min_strength_sum = std::numeric_limits<float>::max();
    stats.max_strength_sum = 0.0f;

    return stats;
}

} // namespace StrengthNormalizer
} // namespace dpan
