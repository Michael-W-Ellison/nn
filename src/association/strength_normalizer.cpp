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
    size_t normalized_count = 0;

    for (const auto& pattern : matrix.GetAllPatterns()) {
        bool normalized = false;
        switch (config.mode) {
            case NormalizationMode::OUTGOING:
                normalized = NormalizeOutgoing(matrix, pattern, config);
                break;
            case NormalizationMode::INCOMING:
                normalized = NormalizeIncoming(matrix, pattern, config);
                break;
            case NormalizationMode::BIDIRECTIONAL: {
                auto [out, in] = NormalizeBidirectional(matrix, pattern, config);
                normalized = out || in;
                break;
            }
        }
        if (normalized) {
            normalized_count++;
        }
    }

    return normalized_count;
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

    // Read-only analysis of outgoing strength sums: patterns_normalized counts
    // patterns whose sum is already within tolerance of 1.0, edges_updated
    // counts edges a NormalizeAll() would rescale
    constexpr float kTolerance = 0.01f;
    double total_sum = 0.0;

    for (const auto& pattern : matrix.GetAllPatterns()) {
        auto outgoing = matrix.GetOutgoingAssociations(pattern);

        float sum = 0.0f;
        size_t eligible = 0;
        for (const auto* edge : outgoing) {
            if (edge->GetStrength() >= config.min_strength_threshold) {
                sum += edge->GetStrength();
                eligible++;
            }
        }
        if (eligible == 0) {
            continue;
        }

        if (stats.patterns_processed == 0) {
            stats.min_strength_sum = sum;
            stats.max_strength_sum = sum;
        } else {
            stats.min_strength_sum = std::min(stats.min_strength_sum, sum);
            stats.max_strength_sum = std::max(stats.max_strength_sum, sum);
        }
        stats.patterns_processed++;
        total_sum += sum;

        if (std::abs(sum - 1.0f) <= kTolerance) {
            stats.patterns_normalized++;
        } else {
            stats.edges_updated += eligible;
        }
    }

    if (stats.patterns_processed > 0) {
        stats.average_strength_sum =
            static_cast<float>(total_sum / stats.patterns_processed);
    }

    return stats;
}

} // namespace StrengthNormalizer
} // namespace dpan
