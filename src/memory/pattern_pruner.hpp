// File: src/memory/pattern_pruner.hpp
//
// Pattern Pruner - Intelligent Pattern Removal
//
// Implements intelligent pruning to remove low-utility patterns while
// maintaining system knowledge quality. Includes safety checks to prevent
// critical data loss and optional pattern merging to reduce redundancy.
//
// Pruning Strategy:
//   For each pattern p:
//     1. Calculate utility U(p)
//     2. IF U(p) < threshold THEN
//         a. Check if p is safe to delete (safety checks)
//         b. If safe to delete:
//            - Remove p from pattern database
//            - Update all associations involving p
//            - Optionally merge p with similar patterns
//         c. Else:
//            - Keep p but demote to lower tier
//     3. Record pruning decision for analysis
//
// Safety Checks:
//   - Don't prune if pattern is hub (>50 strong associations)
//   - Don't prune if recently created (<24 hours)
//   - Don't prune if has strong associations (>0.7 strength)

#pragma once

#include "core/pattern_node.hpp"
#include "storage/pattern_database.hpp"
#include "association/association_matrix.hpp"
#include "memory/utility_calculator.hpp"
#include <vector>
#include <unordered_map>
#include <optional>
#include <chrono>

namespace dpan {

/// Pattern Pruner for intelligent pattern removal
class PatternPruner {
public:
    /// Configuration for pattern pruning
    struct Config {
        /// Utility threshold below which patterns are candidates for pruning
        float utility_threshold{0.2f};

        /// Minimum number of associations for a pattern to be considered a hub
        size_t min_associations_for_hub{50};

        /// Minimum age before a pattern can be pruned
        Timestamp::Duration min_pattern_age{std::chrono::hours(24)};

        /// Minimum association strength to be considered "strong"
        float strong_association_threshold{0.7f};

        /// Enable pattern merging
        bool enable_merging{true};

        /// Similarity threshold for merging patterns (0.95 = very similar)
        float merge_similarity_threshold{0.95f};

        /// Maximum number of patterns to process in one batch
        size_t max_prune_batch{1000};

        /// Validate configuration
        bool IsValid() const;
    };

    /// Result of a pruning operation
    struct PruneResult {
        /// Patterns that were pruned (removed)
        std::vector<PatternID> pruned_patterns;

        /// Patterns that were merged (old ID, new ID)
        std::vector<std::pair<PatternID, PatternID>> merged_patterns;

        /// Number of associations updated during pruning
        size_t associations_updated{0};

        /// Approximate bytes freed by pruning
        size_t bytes_freed{0};

        /// Number of patterns that were candidates but kept (safety checks)
        size_t patterns_kept_safe{0};

        /// Number of patterns skipped (too young, too important, etc.)
        size_t patterns_skipped{0};
    };

    /// Construct with configuration
    explicit PatternPruner(const Config& config);

    // ========================================================================
    // Main Pruning Operations
    // ========================================================================

    /// Prune low-utility patterns from the database
    ///
    /// @param pattern_db Pattern database to prune from
    /// @param assoc_matrix Association matrix for safety checks and updates
    /// @param utilities Map of pattern ID to utility score
    /// @return Result containing pruned patterns, merged patterns, and statistics
    PruneResult PrunePatterns(
        PatternDatabase& pattern_db,
        AssociationMatrix& assoc_matrix,
        const std::unordered_map<PatternID, float>& utilities
    );

    /// Prune a specific pattern if safe
    ///
    /// @param id Pattern ID to prune
    /// @param pattern The pattern node
    /// @param pattern_db Pattern database
    /// @param assoc_matrix Association matrix
    /// @param utility Pattern's utility score
    /// @return true if pruned, false if kept
    bool PrunePattern(
        PatternID id,
        const PatternNode& pattern,
        PatternDatabase& pattern_db,
        AssociationMatrix& assoc_matrix,
        float utility
    );

    // ========================================================================
    // Safety Checks
    // ========================================================================

    /// Check if a pattern is safe to prune
    ///
    /// Returns false if:
    /// - Pattern is a hub (has many associations)
    /// - Pattern was recently created
    /// - Pattern has strong associations
    ///
    /// @param id Pattern ID
    /// @param pattern The pattern node
    /// @param assoc_matrix Association matrix for checking associations
    /// @param utility Pattern's utility score
    /// @return true if safe to prune, false otherwise
    bool IsSafeToPrune(
        PatternID id,
        const PatternNode& pattern,
        const AssociationMatrix& assoc_matrix,
        float utility
    ) const;

    /// Check if pattern is a hub (many connections)
    ///
    /// @param id Pattern ID
    /// @param assoc_matrix Association matrix
    /// @return true if pattern is a hub
    bool IsHub(PatternID id, const AssociationMatrix& assoc_matrix) const;

    /// Check if pattern is recently created
    ///
    /// @param pattern The pattern node
    /// @return true if pattern is too young to prune
    bool IsRecentlyCreated(const PatternNode& pattern) const;

    /// Check if pattern has strong associations
    ///
    /// @param id Pattern ID
    /// @param assoc_matrix Association matrix
    /// @return true if pattern has strong associations
    bool HasStrongAssociations(PatternID id, const AssociationMatrix& assoc_matrix) const;

    // ========================================================================
    // Pattern Merging
    // ========================================================================

    /// Find a candidate pattern to merge with
    ///
    /// Searches for patterns that are highly similar (above merge threshold)
    ///
    /// @param pattern Pattern to find merge candidate for
    /// @param pattern_db Pattern database to search
    /// @return Pattern ID of merge candidate, or nullopt if none found
    std::optional<PatternID> FindMergeCandidate(
        const PatternNode& pattern,
        const PatternDatabase& pattern_db
    ) const;

    /// Merge two patterns
    ///
    /// Transfers all associations from old pattern to new pattern,
    /// then removes old pattern from database
    ///
    /// @param old_pattern Pattern to be merged (will be removed)
    /// @param new_pattern Target pattern (will receive associations)
    /// @param pattern_db Pattern database
    /// @param assoc_matrix Association matrix
    /// @return true if merged successfully
    bool MergePatterns(
        PatternID old_pattern,
        PatternID new_pattern,
        PatternDatabase& pattern_db,
        AssociationMatrix& assoc_matrix
    );

    // ========================================================================
    // Statistics and Configuration
    // ========================================================================

    /// Get current configuration
    const Config& GetConfig() const { return config_; }

    /// Update configuration
    void SetConfig(const Config& config);

private:
    Config config_;

    // Helper methods
    size_t EstimatePatternSize(const PatternNode& pattern) const;
    std::vector<PatternID> SelectPruneCandidates(
        const std::unordered_map<PatternID, float>& utilities
    ) const;
};

} // namespace dpan
