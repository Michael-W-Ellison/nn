// File: src/memory/pattern_pruner.cpp
//
// Implementation of Pattern Pruner

#include "memory/pattern_pruner.hpp"
#include "similarity/similarity_metric.hpp"
#include <algorithm>
#include <stdexcept>

namespace dpan {

// ============================================================================
// Config
// ============================================================================

bool PatternPruner::Config::IsValid() const {
    if (utility_threshold < 0.0f || utility_threshold > 1.0f) {
        return false;
    }

    if (min_associations_for_hub == 0 || min_associations_for_hub > 1000) {
        return false;
    }

    if (min_pattern_age.count() <= 0) {
        return false;
    }

    if (strong_association_threshold < 0.0f || strong_association_threshold > 1.0f) {
        return false;
    }

    if (merge_similarity_threshold < 0.0f || merge_similarity_threshold > 1.0f) {
        return false;
    }

    if (max_prune_batch == 0 || max_prune_batch > 100000) {
        return false;
    }

    return true;
}

// ============================================================================
// Constructor
// ============================================================================

PatternPruner::PatternPruner(const Config& config)
    : config_(config) {

    if (!config_.IsValid()) {
        throw std::invalid_argument("Invalid PatternPruner configuration");
    }
}

// ============================================================================
// Main Pruning Operations
// ============================================================================

PatternPruner::PruneResult PatternPruner::PrunePatterns(
    PatternDatabase& pattern_db,
    AssociationMatrix& assoc_matrix,
    const std::unordered_map<PatternID, float>& utilities) {

    PruneResult result;

    // Select candidates for pruning (low utility patterns)
    auto candidates = SelectPruneCandidates(utilities);

    // Limit to batch size
    if (candidates.size() > config_.max_prune_batch) {
        candidates.resize(config_.max_prune_batch);
    }

    // Process each candidate
    for (const auto& pattern_id : candidates) {
        // Get utility
        auto util_it = utilities.find(pattern_id);
        float utility = (util_it != utilities.end()) ? util_it->second : 0.0f;

        // Retrieve pattern
        auto pattern_opt = pattern_db.Retrieve(pattern_id);
        if (!pattern_opt) {
            continue;  // Pattern not found, skip
        }

        const auto& pattern = *pattern_opt;

        // Check if safe to prune
        if (!IsSafeToPrune(pattern_id, pattern, assoc_matrix, utility)) {
            result.patterns_kept_safe++;
            continue;
        }

        // Try to merge if enabled
        bool merged = false;
        if (config_.enable_merging) {
            auto merge_candidate = FindMergeCandidate(pattern, pattern_db);
            if (merge_candidate) {
                if (MergePatterns(pattern_id, *merge_candidate, pattern_db, assoc_matrix)) {
                    result.merged_patterns.emplace_back(pattern_id, *merge_candidate);
                    result.bytes_freed += EstimatePatternSize(pattern);
                    merged = true;
                }
            }
        }

        // If not merged, just prune
        if (!merged) {
            if (PrunePattern(pattern_id, pattern, pattern_db, assoc_matrix, utility)) {
                result.pruned_patterns.push_back(pattern_id);
                result.bytes_freed += EstimatePatternSize(pattern);
            }
        }
    }

    return result;
}

bool PatternPruner::PrunePattern(
    PatternID id,
    const PatternNode& pattern,
    PatternDatabase& pattern_db,
    AssociationMatrix& assoc_matrix,
    float utility) {

    (void)utility;  // May be used for logging/stats

    // Remove all associations involving this pattern
    auto outgoing = assoc_matrix.GetOutgoingAssociations(id);
    auto incoming = assoc_matrix.GetIncomingAssociations(id);

    size_t associations_removed = 0;

    // Remove outgoing associations
    for (const auto* edge : outgoing) {
        if (edge) {
            if (assoc_matrix.RemoveAssociation(edge->GetSource(), edge->GetTarget())) {
                associations_removed++;
            }
        }
    }

    // Remove incoming associations
    for (const auto* edge : incoming) {
        if (edge) {
            if (assoc_matrix.RemoveAssociation(edge->GetSource(), edge->GetTarget())) {
                associations_removed++;
            }
        }
    }

    // Remove pattern from database
    return pattern_db.Delete(id);
}

// ============================================================================
// Safety Checks
// ============================================================================

bool PatternPruner::IsSafeToPrune(
    PatternID id,
    const PatternNode& pattern,
    const AssociationMatrix& assoc_matrix,
    float utility) const {

    // Check utility threshold
    if (utility >= config_.utility_threshold) {
        return false;  // Utility too high
    }

    // Check if recently created
    if (IsRecentlyCreated(pattern)) {
        return false;  // Too young to prune
    }

    // Check if hub
    if (IsHub(id, assoc_matrix)) {
        return false;  // Hub pattern, keep it
    }

    // Check if has strong associations
    if (HasStrongAssociations(id, assoc_matrix)) {
        return false;  // Important connections, keep it
    }

    return true;  // Safe to prune
}

bool PatternPruner::IsHub(PatternID id, const AssociationMatrix& assoc_matrix) const {
    auto outgoing = assoc_matrix.GetOutgoingAssociations(id);
    auto incoming = assoc_matrix.GetIncomingAssociations(id);

    size_t total_associations = outgoing.size() + incoming.size();

    return total_associations >= config_.min_associations_for_hub;
}

bool PatternPruner::IsRecentlyCreated(const PatternNode& pattern) const {
    auto creation_time = pattern.GetCreationTime();
    auto age = Timestamp::Now() - creation_time;

    return age < config_.min_pattern_age;
}

bool PatternPruner::HasStrongAssociations(
    PatternID id,
    const AssociationMatrix& assoc_matrix) const {

    // Check outgoing associations
    auto outgoing = assoc_matrix.GetOutgoingAssociations(id);
    for (const auto* edge : outgoing) {
        if (edge && edge->GetStrength() >= config_.strong_association_threshold) {
            return true;
        }
    }

    // Check incoming associations
    auto incoming = assoc_matrix.GetIncomingAssociations(id);
    for (const auto* edge : incoming) {
        if (edge && edge->GetStrength() >= config_.strong_association_threshold) {
            return true;
        }
    }

    return false;
}

// ============================================================================
// Pattern Merging
// ============================================================================

std::optional<PatternID> PatternPruner::FindMergeCandidate(
    const PatternNode& pattern,
    const PatternDatabase& pattern_db) const {

    // Note: QuerySimilar is not implemented in PatternDatabase yet
    // For now, return nullopt (no merge candidate found)
    // TODO: Implement similarity search when QuerySimilar is available
    (void)pattern;
    (void)pattern_db;
    return std::nullopt;
}

bool PatternPruner::MergePatterns(
    PatternID old_pattern,
    PatternID new_pattern,
    PatternDatabase& pattern_db,
    AssociationMatrix& assoc_matrix) {

    // Transfer outgoing associations
    auto outgoing = assoc_matrix.GetOutgoingAssociations(old_pattern);
    for (const auto* edge : outgoing) {
        if (!edge) continue;

        PatternID target = edge->GetTarget();

        // Skip if target is the new pattern (would create self-loop)
        if (target == new_pattern) {
            continue;
        }

        // Create new association from new_pattern to target
        // Use existing strength and type
        AssociationEdge new_edge(
            new_pattern,
            target,
            edge->GetType(),
            edge->GetStrength()
        );
        assoc_matrix.AddAssociation(new_edge);

        // Remove old association
        assoc_matrix.RemoveAssociation(old_pattern, target);
    }

    // Transfer incoming associations
    auto incoming = assoc_matrix.GetIncomingAssociations(old_pattern);
    for (const auto* edge : incoming) {
        if (!edge) continue;

        PatternID source = edge->GetSource();

        // Skip if source is the new pattern (would create self-loop)
        if (source == new_pattern) {
            continue;
        }

        // Create new association from source to new_pattern
        AssociationEdge new_edge(
            source,
            new_pattern,
            edge->GetType(),
            edge->GetStrength()
        );
        assoc_matrix.AddAssociation(new_edge);

        // Remove old association
        assoc_matrix.RemoveAssociation(source, old_pattern);
    }

    // Delete old pattern from database
    return pattern_db.Delete(old_pattern);
}

// ============================================================================
// Statistics and Configuration
// ============================================================================

void PatternPruner::SetConfig(const Config& config) {
    if (!config.IsValid()) {
        throw std::invalid_argument("Invalid PatternPruner configuration");
    }

    config_ = config;
}

// ============================================================================
// Helper Methods
// ============================================================================

size_t PatternPruner::EstimatePatternSize(const PatternNode& pattern) const {
    // Rough estimate: PatternNode object size + feature vector
    size_t base_size = sizeof(PatternNode);
    size_t features_size = pattern.GetData().GetFeatures().Dimension() * sizeof(float);
    return base_size + features_size;
}

std::vector<PatternID> PatternPruner::SelectPruneCandidates(
    const std::unordered_map<PatternID, float>& utilities) const {

    std::vector<std::pair<PatternID, float>> candidates;

    // Collect patterns below utility threshold
    for (const auto& [pattern_id, utility] : utilities) {
        if (utility < config_.utility_threshold) {
            candidates.emplace_back(pattern_id, utility);
        }
    }

    // Sort by utility (lowest first)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) {
                  return a.second < b.second;
              });

    // Extract pattern IDs
    std::vector<PatternID> result;
    result.reserve(candidates.size());
    for (const auto& [pattern_id, _] : candidates) {
        result.push_back(pattern_id);
    }

    return result;
}

} // namespace dpan
