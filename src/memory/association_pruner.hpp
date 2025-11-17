// File: src/memory/association_pruner.hpp
#pragma once

#include "core/types.hpp"
#include "association/association_edge.hpp"
#include "association/association_matrix.hpp"
#include "memory/utility_calculator.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

namespace dpan {

/// AssociationPruner: Remove weak, redundant, or stale associations
///
/// Implements intelligent association removal to maintain graph quality:
/// - Prunes weak associations (low strength)
/// - Detects and removes redundant edges (transitively implied)
/// - Removes stale associations (no recent reinforcement)
/// - Handles contradictory associations (conflicting evidence)
///
/// Thread-safe through AssociationMatrix operations
class AssociationPruner {
public:
    /// Configuration for association pruning behavior
    struct Config {
        // Strength threshold for weak associations
        float weak_strength_threshold{0.1f};

        // Staleness threshold (associations not reinforced in this time)
        Timestamp::Duration staleness_threshold{std::chrono::hours(24 * 30)};  // 30 days

        // Redundancy detection settings
        bool enable_redundancy_detection{true};
        float redundancy_path_strength_threshold{0.5f};  // Minimum path strength to consider
        size_t max_path_length{3};                       // Maximum path length to check

        // Batch processing
        size_t max_prune_batch{1000};

        // Safety settings
        float min_association_strength{0.01f};  // Never remove stronger than this
        bool protect_hub_edges{true};           // Don't remove edges to/from hubs
        size_t hub_threshold{50};               // Patterns with >=50 edges are hubs

        // Contradictory association handling
        bool enable_contradiction_detection{false};  // Experimental feature
        float contradiction_threshold{0.8f};          // Opposite types with high strength
    };

    /// Result of pruning operation
    struct PruneResult {
        std::vector<std::pair<PatternID, PatternID>> weak_associations;
        std::vector<std::pair<PatternID, PatternID>> stale_associations;
        std::vector<std::pair<PatternID, PatternID>> redundant_associations;
        std::vector<std::pair<PatternID, PatternID>> contradictory_associations;

        size_t total_pruned{0};
        size_t edges_kept_safe{0};
        size_t edges_skipped{0};

        // Statistics before/after
        size_t associations_before{0};
        size_t associations_after{0};
        float avg_strength_before{0.0f};
        float avg_strength_after{0.0f};
    };

    // ========================================================================
    // Construction
    // ========================================================================

    /// Construct with default configuration
    AssociationPruner() = default;

    /// Construct with custom configuration
    /// @throws std::invalid_argument if config is invalid
    explicit AssociationPruner(const Config& config);

    // ========================================================================
    // Main Pruning Operations
    // ========================================================================

    /// Prune associations based on all criteria
    /// @param assoc_matrix Association matrix to prune
    /// @param access_tracker Optional access tracker for staleness detection
    /// @return Result with details of pruned associations
    PruneResult PruneAssociations(
        AssociationMatrix& assoc_matrix,
        const AccessTracker* access_tracker = nullptr
    );

    /// Prune only weak associations
    /// @param assoc_matrix Association matrix to prune
    /// @return Number of associations removed
    size_t PruneWeakAssociations(AssociationMatrix& assoc_matrix);

    /// Prune only stale associations
    /// @param assoc_matrix Association matrix to prune
    /// @param access_tracker Access tracker for staleness detection
    /// @return Number of associations removed
    size_t PruneStaleAssociations(
        AssociationMatrix& assoc_matrix,
        const AccessTracker& access_tracker
    );

    /// Prune only redundant associations
    /// @param assoc_matrix Association matrix to prune
    /// @return Number of associations removed
    size_t PruneRedundantAssociations(AssociationMatrix& assoc_matrix);

    // ========================================================================
    // Safety Checks
    // ========================================================================

    /// Check if association is safe to prune
    /// @param edge Association edge to check
    /// @param assoc_matrix Association matrix for context
    /// @return True if safe to prune
    bool IsSafeToPrune(
        const AssociationEdge& edge,
        const AssociationMatrix& assoc_matrix
    ) const;

    /// Check if pattern is a hub (many associations)
    /// @param pattern Pattern ID to check
    /// @param assoc_matrix Association matrix for degree calculation
    /// @return True if pattern is a hub
    bool IsHub(PatternID pattern, const AssociationMatrix& assoc_matrix) const;

    // ========================================================================
    // Detection Methods
    // ========================================================================

    /// Check if association is weak
    /// @param edge Association edge to check
    /// @return True if strength below threshold
    bool IsWeak(const AssociationEdge& edge) const;

    /// Check if association is stale
    /// @param edge Association edge to check
    /// @return True if no recent reinforcement
    bool IsStale(const AssociationEdge& edge) const;

    /// Check if association is redundant (implied by stronger path)
    /// @param edge Association edge to check
    /// @param assoc_matrix Association matrix for path search
    /// @return True if redundant path exists
    bool IsRedundant(
        const AssociationEdge& edge,
        const AssociationMatrix& assoc_matrix
    ) const;

    /// Find alternative path between source and target
    /// @param source Source pattern
    /// @param target Target pattern
    /// @param assoc_matrix Association matrix
    /// @param direct_strength Strength of direct edge (to compare against)
    /// @return Path strength if found, 0.0 if no path
    float FindAlternativePath(
        PatternID source,
        PatternID target,
        const AssociationMatrix& assoc_matrix,
        float direct_strength
    ) const;

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Set configuration (validates before applying)
    /// @throws std::invalid_argument if config is invalid
    void SetConfig(const Config& config);

    /// Get current configuration
    const Config& GetConfig() const { return config_; }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get statistics about pruning operations
    struct Statistics {
        size_t total_prune_operations{0};
        size_t total_associations_removed{0};
        size_t weak_removed{0};
        size_t stale_removed{0};
        size_t redundant_removed{0};
        size_t contradictory_removed{0};
        Timestamp last_prune;
    };

    /// Get pruning statistics
    const Statistics& GetStatistics() const { return stats_; }

    /// Reset statistics
    void ResetStatistics();

private:
    Config config_;
    Statistics stats_;

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Validate configuration
    /// @throws std::invalid_argument if invalid
    void ValidateConfig() const;

    /// Collect candidate associations for pruning
    /// @param assoc_matrix Association matrix
    /// @param access_tracker Optional access tracker
    /// @return Vector of (source, target) pairs to consider
    std::vector<std::pair<PatternID, PatternID>> CollectCandidates(
        const AssociationMatrix& assoc_matrix,
        const AccessTracker* access_tracker
    ) const;

    /// Build hub pattern set for efficient lookups
    /// @param assoc_matrix Association matrix
    /// @return Set of hub pattern IDs
    std::unordered_set<PatternID> BuildHubSet(
        const AssociationMatrix& assoc_matrix
    ) const;

    /// Perform BFS to find alternative path
    /// @param source Source pattern
    /// @param target Target pattern
    /// @param assoc_matrix Association matrix
    /// @param max_depth Maximum search depth
    /// @return Path strength if found, 0.0 otherwise
    float BFSAlternativePath(
        PatternID source,
        PatternID target,
        const AssociationMatrix& assoc_matrix,
        size_t max_depth
    ) const;
};

} // namespace dpan
