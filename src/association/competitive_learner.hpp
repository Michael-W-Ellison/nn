// File: src/association/competitive_learner.hpp
#pragma once

#include "association/association_matrix.hpp"
#include "core/types.hpp"
#include <vector>

namespace dpan {

/// CompetitiveLearner: Implements winner-take-all competitive learning
///
/// This module implements competitive learning where strong associations
/// are boosted while weaker competing associations are suppressed.
/// This follows the winner-take-all principle where the strongest
/// association receives reinforcement while others are inhibited.
///
/// Algorithm:
/// For competing associations from p1 to {p2, p3, ..., pn}:
/// 1. Find strongest: s_max = max(s_i)
/// 2. Boost winner: s_max = s_max + β × (1 - s_max)
/// 3. Suppress others: s_i = s_i × (1 - β) for i ≠ max
///
/// Where β is the competition factor ∈ [0, 1]
namespace CompetitiveLearner {

    /// Configuration for competitive learning
    struct Config {
        Config() = default;

        /// Competition factor β ∈ [0, 1]
        /// Higher values = stronger competition
        /// 0.0 = no competition, 1.0 = winner takes all
        float competition_factor{0.3f};

        /// Minimum strength threshold - associations below this are not considered
        float min_strength_threshold{0.01f};

        /// Whether to apply competition to all association types or only specific ones
        bool filter_by_type{false};

        /// If filter_by_type is true, only apply to these types
        std::vector<AssociationType> allowed_types;

        /// Minimum number of competing associations required to apply competition
        /// If fewer associations exist, no competition is applied
        size_t min_competing_associations{2};
    };

    // ========================================================================
    // Single Pattern Competition
    // ========================================================================

    /// Apply competitive learning to outgoing associations of a single pattern
    /// @param matrix Association matrix to modify
    /// @param pattern Pattern whose outgoing associations compete
    /// @param config Competition configuration
    /// @return True if competition was applied
    bool ApplyCompetition(
        AssociationMatrix& matrix,
        PatternID pattern,
        const Config& config = Config()
    );

    /// Apply competitive learning to incoming associations of a single pattern
    /// (associations from multiple sources targeting this pattern)
    /// @param matrix Association matrix to modify
    /// @param pattern Target pattern whose incoming associations compete
    /// @param config Competition configuration
    /// @return True if competition was applied
    bool ApplyCompetitionIncoming(
        AssociationMatrix& matrix,
        PatternID pattern,
        const Config& config = Config()
    );

    // ========================================================================
    // Typed Competition
    // ========================================================================

    /// Apply competition only within a specific association type
    /// Only associations of the same type compete with each other
    /// @param matrix Association matrix to modify
    /// @param pattern Source pattern
    /// @param type Association type to apply competition to
    /// @param config Competition configuration
    /// @return True if competition was applied
    bool ApplyTypedCompetition(
        AssociationMatrix& matrix,
        PatternID pattern,
        AssociationType type,
        const Config& config = Config()
    );

    // ========================================================================
    // Batch Competition
    // ========================================================================

    /// Apply competitive learning to multiple patterns
    /// @param matrix Association matrix to modify
    /// @param patterns Patterns to apply competition to
    /// @param config Competition configuration
    /// @return Number of patterns that had competition applied
    size_t ApplyCompetitionBatch(
        AssociationMatrix& matrix,
        const std::vector<PatternID>& patterns,
        const Config& config = Config()
    );

    /// Apply competitive learning to all patterns in the matrix
    /// @param matrix Association matrix to modify
    /// @param config Competition configuration
    /// @return Number of patterns that had competition applied
    size_t ApplyCompetitionAll(
        AssociationMatrix& matrix,
        const Config& config = Config()
    );

    // ========================================================================
    // Utility Functions
    // ========================================================================

    /// Find the strongest association among a set of associations
    /// @param associations Vector of association edge pointers
    /// @return Pointer to strongest association, or nullptr if empty
    const AssociationEdge* FindStrongest(
        const std::vector<const AssociationEdge*>& associations
    );

    /// Calculate new strength for winner (boost)
    /// Formula: s_new = s_old + β × (1 - s_old)
    /// @param current_strength Current strength of winner
    /// @param competition_factor Competition factor β
    /// @return New boosted strength
    float CalculateWinnerStrength(
        float current_strength,
        float competition_factor
    );

    /// Calculate new strength for losers (suppress)
    /// Formula: s_new = s_old × (1 - β)
    /// @param current_strength Current strength of loser
    /// @param competition_factor Competition factor β
    /// @return New suppressed strength
    float CalculateLoserStrength(
        float current_strength,
        float competition_factor
    );

    /// Filter associations by type
    /// @param associations All associations
    /// @param type Type to filter for
    /// @return Filtered associations of specified type
    std::vector<const AssociationEdge*> FilterByType(
        const std::vector<const AssociationEdge*>& associations,
        AssociationType type
    );

    /// Filter associations by minimum strength threshold
    /// @param associations All associations
    /// @param min_threshold Minimum strength threshold
    /// @return Filtered associations above threshold
    std::vector<const AssociationEdge*> FilterByStrength(
        const std::vector<const AssociationEdge*>& associations,
        float min_threshold
    );

    // ========================================================================
    // Statistics
    // ========================================================================

    struct CompetitionStats {
        size_t patterns_processed{0};        // Total patterns examined
        size_t competitions_applied{0};       // Number of times competition was applied
        size_t winners_boosted{0};            // Number of winners boosted
        size_t losers_suppressed{0};          // Number of losers suppressed
        float average_winner_boost{0.0f};     // Average strength increase for winners
        float average_loser_suppression{0.0f}; // Average strength decrease for losers
        float total_strength_before{0.0f};    // Total strength before competition
        float total_strength_after{0.0f};     // Total strength after competition
    };

    /// Analyze the effects of competition on a pattern
    /// @param matrix Association matrix
    /// @param pattern Pattern to analyze
    /// @param config Competition configuration
    /// @return Statistics about competition effects
    CompetitionStats AnalyzeCompetition(
        const AssociationMatrix& matrix,
        PatternID pattern,
        const Config& config = Config()
    );

    /// Apply competition and return statistics
    /// @param matrix Association matrix to modify
    /// @param pattern Pattern to apply competition to
    /// @param config Competition configuration
    /// @return Statistics about the competition that was applied
    CompetitionStats ApplyCompetitionWithStats(
        AssociationMatrix& matrix,
        PatternID pattern,
        const Config& config = Config()
    );

} // namespace CompetitiveLearner

} // namespace dpan
