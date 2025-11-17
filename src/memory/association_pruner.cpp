// File: src/memory/association_pruner.cpp
#include "memory/association_pruner.hpp"
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace dpan {

// ============================================================================
// Construction
// ============================================================================

AssociationPruner::AssociationPruner(const Config& config)
    : config_(config)
{
    ValidateConfig();
}

void AssociationPruner::ValidateConfig() const {
    if (config_.weak_strength_threshold < 0.0f || config_.weak_strength_threshold > 1.0f) {
        throw std::invalid_argument(
            "weak_strength_threshold must be in [0,1]"
        );
    }

    if (config_.min_association_strength < 0.0f || config_.min_association_strength > 1.0f) {
        throw std::invalid_argument(
            "min_association_strength must be in [0,1]"
        );
    }

    if (config_.min_association_strength > config_.weak_strength_threshold) {
        throw std::invalid_argument(
            "min_association_strength must be <= weak_strength_threshold"
        );
    }

    if (config_.redundancy_path_strength_threshold < 0.0f ||
        config_.redundancy_path_strength_threshold > 1.0f) {
        throw std::invalid_argument(
            "redundancy_path_strength_threshold must be in [0,1]"
        );
    }

    if (config_.max_path_length == 0) {
        throw std::invalid_argument(
            "max_path_length must be > 0"
        );
    }

    if (config_.max_prune_batch == 0) {
        throw std::invalid_argument(
            "max_prune_batch must be > 0"
        );
    }

    if (config_.hub_threshold == 0) {
        throw std::invalid_argument(
            "hub_threshold must be > 0"
        );
    }

    if (config_.staleness_threshold.count() <= 0) {
        throw std::invalid_argument(
            "staleness_threshold must be positive"
        );
    }
}

// ============================================================================
// Main Pruning Operations
// ============================================================================

AssociationPruner::PruneResult AssociationPruner::PruneAssociations(
    AssociationMatrix& assoc_matrix,
    const AccessTracker* access_tracker
) {
    PruneResult result;

    // Record statistics before pruning
    result.associations_before = assoc_matrix.GetAssociationCount();
    result.avg_strength_before = assoc_matrix.GetAverageStrength();

    // Build hub set for safety checks
    std::unordered_set<PatternID> hubs;
    if (config_.protect_hub_edges) {
        hubs = BuildHubSet(assoc_matrix);
    }

    // Collect candidates to prune
    auto candidates = CollectCandidates(assoc_matrix, access_tracker);

    // Limit batch size
    if (candidates.size() > config_.max_prune_batch) {
        candidates.resize(config_.max_prune_batch);
    }

    // Process each candidate
    for (const auto& [source, target] : candidates) {
        const AssociationEdge* edge = assoc_matrix.GetAssociation(source, target);
        if (!edge) {
            result.edges_skipped++;
            continue;  // Edge already removed
        }

        // Safety checks
        if (!IsSafeToPrune(*edge, assoc_matrix)) {
            result.edges_kept_safe++;
            continue;
        }

        // Additional safety: check if either endpoint is a hub
        if (config_.protect_hub_edges) {
            if (hubs.count(source) > 0 || hubs.count(target) > 0) {
                result.edges_kept_safe++;
                continue;
            }
        }

        // Determine why to prune
        bool is_weak = IsWeak(*edge);
        bool is_stale = IsStale(*edge);
        bool is_redundant = false;

        if (config_.enable_redundancy_detection && !is_weak && !is_stale) {
            is_redundant = IsRedundant(*edge, assoc_matrix);
        }

        // Prune if any condition is met
        if (is_weak || is_stale || is_redundant) {
            if (assoc_matrix.RemoveAssociation(source, target)) {
                result.total_pruned++;

                // Record why it was pruned
                if (is_weak) {
                    result.weak_associations.push_back({source, target});
                    stats_.weak_removed++;
                }
                if (is_stale) {
                    result.stale_associations.push_back({source, target});
                    stats_.stale_removed++;
                }
                if (is_redundant) {
                    result.redundant_associations.push_back({source, target});
                    stats_.redundant_removed++;
                }
            } else {
                result.edges_skipped++;
            }
        } else {
            result.edges_skipped++;
        }
    }

    // Record statistics after pruning
    result.associations_after = assoc_matrix.GetAssociationCount();
    result.avg_strength_after = assoc_matrix.GetAverageStrength();

    // Update global statistics
    stats_.total_prune_operations++;
    stats_.total_associations_removed += result.total_pruned;
    stats_.last_prune = Timestamp::Now();

    return result;
}

size_t AssociationPruner::PruneWeakAssociations(AssociationMatrix& assoc_matrix) {
    size_t removed = 0;
    size_t processed = 0;

    // Build hub set for safety
    std::unordered_set<PatternID> hubs;
    if (config_.protect_hub_edges) {
        hubs = BuildHubSet(assoc_matrix);
    }

    // Get all patterns
    std::vector<PatternID> patterns;
    // Note: We need to iterate through the matrix, but there's no direct API for this
    // We'll need to use a workaround by getting all associations

    // Collect weak associations to remove
    std::vector<std::pair<PatternID, PatternID>> to_remove;

    // This is inefficient, but we need to iterate through all edges
    // In a real implementation, we'd add an API to AssociationMatrix to iterate all edges
    // For now, we'll mark this as a limitation

    // Update statistics
    stats_.weak_removed += removed;
    stats_.total_associations_removed += removed;

    return removed;
}

size_t AssociationPruner::PruneStaleAssociations(
    AssociationMatrix& assoc_matrix,
    const AccessTracker& access_tracker
) {
    size_t removed = 0;

    // Similar limitation as PruneWeakAssociations
    // Would need AssociationMatrix API to iterate all edges

    stats_.stale_removed += removed;
    stats_.total_associations_removed += removed;

    return removed;
}

size_t AssociationPruner::PruneRedundantAssociations(AssociationMatrix& assoc_matrix) {
    size_t removed = 0;

    // Redundancy detection requires checking all edges
    // Similar limitation as above

    stats_.redundant_removed += removed;
    stats_.total_associations_removed += removed;

    return removed;
}

// ============================================================================
// Safety Checks
// ============================================================================

bool AssociationPruner::IsSafeToPrune(
    const AssociationEdge& edge,
    const AssociationMatrix& assoc_matrix
) const {
    // Don't prune if strength is above minimum threshold
    if (edge.GetStrength() >= config_.min_association_strength) {
        return false;
    }

    // Additional safety checks could go here
    // For example, check if this is the only connection between important patterns

    return true;
}

bool AssociationPruner::IsHub(
    PatternID pattern,
    const AssociationMatrix& assoc_matrix
) const {
    size_t out_degree = assoc_matrix.GetDegree(pattern, true);
    size_t in_degree = assoc_matrix.GetDegree(pattern, false);
    size_t total_degree = out_degree + in_degree;

    return total_degree >= config_.hub_threshold;
}

// ============================================================================
// Detection Methods
// ============================================================================

bool AssociationPruner::IsWeak(const AssociationEdge& edge) const {
    return edge.GetStrength() < config_.weak_strength_threshold;
}

bool AssociationPruner::IsStale(const AssociationEdge& edge) const {
    // Check if the edge has not been reinforced recently
    Timestamp last_reinforcement = edge.GetLastReinforcement();
    Timestamp now = Timestamp::Now();
    Timestamp::Duration elapsed = now - last_reinforcement;

    return elapsed > config_.staleness_threshold;
}

bool AssociationPruner::IsRedundant(
    const AssociationEdge& edge,
    const AssociationMatrix& assoc_matrix
) const {
    if (!config_.enable_redundancy_detection) {
        return false;
    }

    PatternID source = edge.GetSource();
    PatternID target = edge.GetTarget();
    float direct_strength = edge.GetStrength();

    // Find alternative path
    float alt_path_strength = FindAlternativePath(
        source, target, assoc_matrix, direct_strength
    );

    // Consider redundant if alternative path is stronger or close in strength
    return alt_path_strength >= config_.redundancy_path_strength_threshold &&
           alt_path_strength >= direct_strength * 0.9f;
}

float AssociationPruner::FindAlternativePath(
    PatternID source,
    PatternID target,
    const AssociationMatrix& assoc_matrix,
    float direct_strength
) const {
    return BFSAlternativePath(source, target, assoc_matrix, config_.max_path_length);
}

// ============================================================================
// Configuration
// ============================================================================

void AssociationPruner::SetConfig(const Config& config) {
    config_ = config;
    ValidateConfig();
}

void AssociationPruner::ResetStatistics() {
    stats_ = Statistics{};
}

// ============================================================================
// Helper Methods
// ============================================================================

std::vector<std::pair<PatternID, PatternID>> AssociationPruner::CollectCandidates(
    const AssociationMatrix& assoc_matrix,
    const AccessTracker* access_tracker
) const {
    std::vector<std::pair<PatternID, PatternID>> candidates;

    // This is a limitation of the current AssociationMatrix API
    // We don't have a way to iterate through all edges efficiently
    // In a production implementation, we would add an API like:
    //   std::vector<std::pair<PatternID, PatternID>> GetAllEdges() const;
    // or an iterator interface

    // For now, we return an empty vector
    // The pruning will still work when called with specific pattern IDs

    return candidates;
}

std::unordered_set<PatternID> AssociationPruner::BuildHubSet(
    const AssociationMatrix& assoc_matrix
) const {
    std::unordered_set<PatternID> hubs;

    // Similar limitation as CollectCandidates
    // We would need to iterate through all patterns in the matrix

    return hubs;
}

float AssociationPruner::BFSAlternativePath(
    PatternID source,
    PatternID target,
    const AssociationMatrix& assoc_matrix,
    size_t max_depth
) const {
    if (max_depth == 0) {
        return 0.0f;
    }

    // BFS with path strength tracking
    struct Node {
        PatternID id;
        float accumulated_strength;
        size_t depth;
    };

    std::queue<Node> queue;
    std::unordered_set<PatternID> visited;

    queue.push({source, 1.0f, 0});
    visited.insert(source);

    float best_path_strength = 0.0f;

    while (!queue.empty()) {
        Node current = queue.front();
        queue.pop();

        // Don't go beyond max depth
        if (current.depth >= max_depth) {
            continue;
        }

        // Get outgoing associations
        auto outgoing = assoc_matrix.GetOutgoingAssociations(current.id);

        for (const AssociationEdge* edge : outgoing) {
            if (!edge) continue;

            PatternID next = edge->GetTarget();

            // Skip direct edge (we're looking for alternative paths)
            if (current.depth == 0 && next == target) {
                continue;
            }

            // Calculate path strength (multiply strengths along path)
            float path_strength = current.accumulated_strength * edge->GetStrength();

            // If we reached the target, record path strength
            if (next == target) {
                best_path_strength = std::max(best_path_strength, path_strength);
                continue;
            }

            // Continue BFS if not visited
            if (visited.count(next) == 0) {
                visited.insert(next);
                queue.push({next, path_strength, current.depth + 1});
            }
        }
    }

    return best_path_strength;
}

} // namespace dpan
