// File: src/association/reinforcement_manager.cpp
#include "association/reinforcement_manager.hpp"
#include <cmath>
#include <algorithm>

namespace dpan {

// ============================================================================
// Construction
// ============================================================================

ReinforcementManager::ReinforcementManager()
    : config_(), stats_()
{
}

ReinforcementManager::ReinforcementManager(const Config& config)
    : config_(config), stats_()
{
}

// ============================================================================
// Single Edge Reinforcement
// ============================================================================

void ReinforcementManager::Reinforce(AssociationEdge& edge, float reward) {
    reward = std::clamp(reward, 0.0f, 1.0f);

    float current_strength = edge.GetStrength();
    float delta = ComputeReinforcementDelta(current_strength, reward);
    float new_strength = ClampStrength(current_strength + delta);

    edge.SetStrength(new_strength);
    RecordReinforcement(delta);
}

void ReinforcementManager::Weaken(AssociationEdge& edge, float penalty) {
    penalty = std::clamp(penalty, 0.0f, 1.0f);

    float current_strength = edge.GetStrength();
    float delta = ComputeWeakeningDelta(current_strength, penalty);
    float new_strength = ClampStrength(current_strength + delta);  // delta is negative

    edge.SetStrength(new_strength);
    RecordWeakening(delta);
}

void ReinforcementManager::ApplyDecay(AssociationEdge& edge, Timestamp::Duration elapsed) {
    float current_strength = edge.GetStrength();
    float decay_factor = ComputeDecayFactor(elapsed);
    float new_strength = ClampStrength(current_strength * decay_factor);

    edge.SetStrength(new_strength);
    RecordDecay();
}

void ReinforcementManager::SetStrength(AssociationEdge& edge, float new_strength) {
    edge.SetStrength(ClampStrength(new_strength));
}

float ReinforcementManager::ClampStrength(float strength) const {
    return std::clamp(strength, config_.min_strength, config_.max_strength);
}

// ============================================================================
// Batch Reinforcement
// ============================================================================

void ReinforcementManager::ReinforceBatch(
    AssociationMatrix& matrix,
    const std::vector<std::pair<PatternID, PatternID>>& pairs,
    float reward
) {
    reward = std::clamp(reward, 0.0f, 1.0f);

    for (const auto& [source, target] : pairs) {
        auto edge_opt = matrix.GetAssociation(source, target);
        if (!edge_opt) {
            continue;  // Edge doesn't exist
        }

        float current_strength = edge_opt->GetStrength();
        float delta = ComputeReinforcementDelta(current_strength, reward);
        float new_strength = ClampStrength(current_strength + delta);

        // Clone edge, update strength, and write back to matrix
        auto updated_edge = edge_opt->Clone();
        updated_edge->SetStrength(new_strength);
        matrix.UpdateAssociation(source, target, *updated_edge);

        RecordReinforcement(delta);
    }
}

void ReinforcementManager::WeakenBatch(
    AssociationMatrix& matrix,
    const std::vector<std::pair<PatternID, PatternID>>& pairs,
    float penalty
) {
    penalty = std::clamp(penalty, 0.0f, 1.0f);

    for (const auto& [source, target] : pairs) {
        auto edge_opt = matrix.GetAssociation(source, target);
        if (!edge_opt) {
            continue;  // Edge doesn't exist
        }

        float current_strength = edge_opt->GetStrength();
        float delta = ComputeWeakeningDelta(current_strength, penalty);
        float new_strength = ClampStrength(current_strength + delta);

        // Clone edge, update strength, and write back to matrix
        auto updated_edge = edge_opt->Clone();
        updated_edge->SetStrength(new_strength);
        matrix.UpdateAssociation(source, target, *updated_edge);

        RecordWeakening(delta);
    }
}

void ReinforcementManager::ApplyDecayAll(
    AssociationMatrix& matrix,
    Timestamp::Duration elapsed,
    bool auto_prune
) {
    // We need to iterate over all patterns - use a workaround since there's no GetAllPatterns()
    // We'll collect patterns from the outgoing index by trying GetOutgoingAssociations
    // This is a limitation - we'd need to modify AssociationMatrix to expose all patterns
    // For now, we'll use the existing ApplyDecayAll method from AssociationMatrix
    // and then handle pruning separately if needed

    // Use the matrix's built-in decay
    matrix.ApplyDecayAll(elapsed);

    // Count decays (approximate - we don't know exact count without iterating)
    size_t approx_edges = matrix.GetAssociationCount();
    for (size_t i = 0; i < approx_edges; ++i) {
        RecordDecay();
    }

    // If auto-pruning is enabled, prune weak associations
    if (auto_prune) {
        PruneWeakAssociations(matrix);
        // Don't double-count - PruneWeakAssociations already calls RecordPruned
    }
}

// ============================================================================
// Prediction-Based Reinforcement
// ============================================================================

void ReinforcementManager::ReinforcePrediction(
    AssociationEdge& edge,
    bool predicted,
    bool actual_occurred
) {
    if (predicted && actual_occurred) {
        // True positive: reinforce
        Reinforce(edge, 1.0f);
    } else if (predicted && !actual_occurred) {
        // False positive: weaken
        Weaken(edge, 0.5f);
    } else if (!predicted && actual_occurred) {
        // False negative: could strengthen slightly (was missed opportunity)
        Reinforce(edge, 0.1f);
    }
    // True negative: no change (correctly didn't predict)
}

void ReinforcementManager::ReinforcePredictionsBatch(
    AssociationMatrix& matrix,
    const std::vector<std::tuple<PatternID, PatternID, bool, bool>>& predictions
) {
    for (const auto& [source, target, predicted, occurred] : predictions) {
        auto edge_opt = matrix.GetAssociation(source, target);
        if (!edge_opt) {
            continue;
        }

        float current_strength = edge_opt->GetStrength();
        float new_strength = current_strength;

        if (predicted && occurred) {
            // True positive: reinforce
            float delta = ComputeReinforcementDelta(current_strength, 1.0f);
            new_strength = current_strength + delta;
            RecordReinforcement(delta);
        } else if (predicted && !occurred) {
            // False positive: weaken
            float delta = ComputeWeakeningDelta(current_strength, 0.5f);
            new_strength = current_strength + delta;
            RecordWeakening(delta);
        } else if (!predicted && occurred) {
            // False negative: slight reinforcement
            float delta = ComputeReinforcementDelta(current_strength, 0.1f);
            new_strength = current_strength + delta;
            RecordReinforcement(delta);
        }

        new_strength = ClampStrength(new_strength);

        // Clone edge, update strength, and write back to matrix
        auto updated_edge = edge_opt->Clone();
        updated_edge->SetStrength(new_strength);
        matrix.UpdateAssociation(source, target, *updated_edge);
    }
}

// ============================================================================
// Utility Methods
// ============================================================================

bool ReinforcementManager::ShouldPrune(const AssociationEdge& edge) const {
    return edge.GetStrength() < config_.prune_threshold;
}

size_t ReinforcementManager::CountPrunableEdges(const AssociationMatrix& /* matrix */) const {
    // Note: This is a limitation of the current AssociationMatrix API
    // Without a method to iterate over all edges, we cannot efficiently count prunable edges
    // This would require AssociationMatrix to expose a GetAllEdges() or similar method
    // For now, return 0 as a conservative estimate
    // TODO: Update when AssociationMatrix provides iteration over all edges
    return 0;
}

size_t ReinforcementManager::PruneWeakAssociations(AssociationMatrix& /* matrix */) {
    // Note: This is a limitation of the current AssociationMatrix API
    // Without a method to iterate over all edges, we cannot efficiently prune weak associations
    // This would require AssociationMatrix to expose a GetAllEdges() or similar method
    // For now, return 0
    // TODO: Update when AssociationMatrix provides iteration over all edges
    RecordPruned(0);
    return 0;
}

// ============================================================================
// Statistics
// ============================================================================

void ReinforcementManager::ResetStats() {
    stats_ = ReinforcementStats{};
}

// ============================================================================
// Private Helper Methods
// ============================================================================

float ReinforcementManager::ComputeReinforcementDelta(
    float current_strength,
    float reward
) const {
    // Hebbian learning: Δs = η × (1 - s) × reward
    // This provides diminishing returns as strength approaches 1.0
    return config_.learning_rate * (1.0f - current_strength) * reward;
}

float ReinforcementManager::ComputeWeakeningDelta(
    float current_strength,
    float penalty
) const {
    // Weakening: Δs = -η × s × penalty
    // Negative delta reduces strength proportionally
    return -config_.learning_rate * current_strength * penalty;
}

float ReinforcementManager::ComputeDecayFactor(Timestamp::Duration elapsed) const {
    // Exponential decay: factor = exp(-d × t)
    // Convert elapsed time to seconds
    auto seconds = std::chrono::duration_cast<std::chrono::duration<float>>(elapsed).count();

    // Compute decay factor
    float exponent = -config_.decay_rate * seconds;
    float factor = std::exp(exponent);

    return factor;
}

void ReinforcementManager::RecordReinforcement(float delta) {
    stats_.reinforcements++;

    // Update running average of strength delta
    float n = static_cast<float>(stats_.reinforcements + stats_.weakenings);
    stats_.average_strength_delta =
        (stats_.average_strength_delta * (n - 1.0f) + delta) / n;
}

void ReinforcementManager::RecordWeakening(float delta) {
    stats_.weakenings++;

    // Update running average of strength delta (delta is negative for weakening)
    float n = static_cast<float>(stats_.reinforcements + stats_.weakenings);
    stats_.average_strength_delta =
        (stats_.average_strength_delta * (n - 1.0f) + delta) / n;
}

void ReinforcementManager::RecordDecay() {
    stats_.decays++;
}

void ReinforcementManager::RecordPruned(size_t count) {
    stats_.pruned += count;
}

} // namespace dpan
