// File: src/association/strength_normalizer.hpp
#pragma once

#include "association/association_matrix.hpp"
#include "core/types.hpp"
#include <vector>

namespace dpan {

/// StrengthNormalizer: Utility functions for normalizing association strengths
///
/// Provides normalization operations to prevent strength inflation by ensuring
/// that outgoing association strengths from a pattern sum to 1.0.
///
/// Normalization preserves the relative ordering of associations while constraining
/// the total strength budget.
namespace StrengthNormalizer {

    /// Normalization mode
    enum class NormalizationMode {
        /// Normalize outgoing associations (default)
        OUTGOING,
        /// Normalize incoming associations
        INCOMING,
        /// Normalize both outgoing and incoming
        BIDIRECTIONAL
    };

    /// Configuration for normalization
    struct Config {
        Config() = default;

        /// Minimum strength threshold - edges below this are not normalized
        float min_strength_threshold{0.01f};

        /// Whether to preserve zero-strength edges (don't remove them)
        bool preserve_zeros{false};

        /// Normalization mode
        NormalizationMode mode{NormalizationMode::OUTGOING};
    };

    // ========================================================================
    // Single Pattern Normalization
    // ========================================================================

    /// Normalize outgoing association strengths for a pattern
    /// Ensures Σ(outgoing strengths) = 1.0
    /// @param matrix Association matrix to modify
    /// @param pattern Pattern whose outgoing edges to normalize
    /// @param config Normalization configuration
    /// @return True if normalization was applied
    bool NormalizeOutgoing(
        AssociationMatrix& matrix,
        PatternID pattern,
        const Config& config = Config()
    );

    /// Normalize incoming association strengths for a pattern
    /// Ensures Σ(incoming strengths) = 1.0
    /// @param matrix Association matrix to modify
    /// @param pattern Pattern whose incoming edges to normalize
    /// @param config Normalization configuration
    /// @return True if normalization was applied
    bool NormalizeIncoming(
        AssociationMatrix& matrix,
        PatternID pattern,
        const Config& config = Config()
    );

    /// Normalize both outgoing and incoming associations
    /// @param matrix Association matrix to modify
    /// @param pattern Pattern to normalize
    /// @param config Normalization configuration
    /// @return Pair of (outgoing_normalized, incoming_normalized)
    std::pair<bool, bool> NormalizeBidirectional(
        AssociationMatrix& matrix,
        PatternID pattern,
        const Config& config = Config()
    );

    // ========================================================================
    // Batch Normalization
    // ========================================================================

    /// Normalize outgoing associations for multiple patterns
    /// @param matrix Association matrix to modify
    /// @param patterns Patterns to normalize
    /// @param config Normalization configuration
    /// @return Number of patterns successfully normalized
    size_t NormalizeOutgoingBatch(
        AssociationMatrix& matrix,
        const std::vector<PatternID>& patterns,
        const Config& config = Config()
    );

    /// Normalize entire matrix (all patterns)
    /// @param matrix Association matrix to modify
    /// @param config Normalization configuration
    /// @return Number of patterns successfully normalized
    size_t NormalizeAll(
        AssociationMatrix& matrix,
        const Config& config = Config()
    );

    // ========================================================================
    // Utility Functions
    // ========================================================================

    /// Compute current sum of outgoing strengths
    /// @param matrix Association matrix
    /// @param pattern Pattern to query
    /// @return Sum of outgoing association strengths
    float GetOutgoingStrengthSum(
        const AssociationMatrix& matrix,
        PatternID pattern
    );

    /// Compute current sum of incoming strengths
    /// @param matrix Association matrix
    /// @param pattern Pattern to query
    /// @return Sum of incoming association strengths
    float GetIncomingStrengthSum(
        const AssociationMatrix& matrix,
        PatternID pattern
    );

    /// Check if pattern's outgoing strengths are normalized
    /// @param matrix Association matrix
    /// @param pattern Pattern to check
    /// @param tolerance Acceptable deviation from 1.0
    /// @return True if sum is within tolerance of 1.0
    bool IsNormalized(
        const AssociationMatrix& matrix,
        PatternID pattern,
        float tolerance = 0.01f
    );

    /// Get normalization factor needed for a pattern
    /// @param matrix Association matrix
    /// @param pattern Pattern to analyze
    /// @return Factor to multiply strengths by (1.0 / sum)
    float GetNormalizationFactor(
        const AssociationMatrix& matrix,
        PatternID pattern
    );

    // ========================================================================
    // Statistics
    // ========================================================================

    struct NormalizationStats {
        size_t patterns_processed{0};        // Total patterns examined
        size_t patterns_normalized{0};       // Patterns that were normalized
        size_t edges_updated{0};             // Total edges modified
        float average_strength_sum{0.0f};    // Average sum before normalization
        float max_strength_sum{0.0f};        // Maximum sum encountered
        float min_strength_sum{0.0f};        // Minimum sum encountered
    };

    /// Analyze normalization state of matrix
    /// @param matrix Association matrix to analyze
    /// @param config Analysis configuration
    /// @return Statistics about normalization state
    NormalizationStats AnalyzeNormalization(
        const AssociationMatrix& matrix,
        const Config& config = Config()
    );

} // namespace StrengthNormalizer

} // namespace dpan
