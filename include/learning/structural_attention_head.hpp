// File: include/learning/structural_attention_head.hpp
//
// Structural Attention Head for DPAN
//
// Implements structure-based attention for composite patterns using sub-pattern
// overlap and Jaccard similarity.
//
// This attention mechanism is particularly useful for:
// - Hierarchical pattern matching
// - Composite pattern similarity
// - Structure-aware pattern retrieval
//
// Example usage:
// @code
//   StructuralAttentionConfig config;
//   config.jaccard_weight = 0.7f;
//   config.size_weight = 0.3f;
//   config.temperature = 1.0f;
//
//   auto structural_head = std::make_unique<StructuralAttentionHead>(config);
//   structural_head->SetPatternDatabase(db);
//
//   auto weights = structural_head->ComputeAttention(query, candidates, context);
//   // Patterns with similar structure will have higher weights
// @endcode

#ifndef DPAN_LEARNING_STRUCTURAL_ATTENTION_HEAD_HPP
#define DPAN_LEARNING_STRUCTURAL_ATTENTION_HEAD_HPP

#include "learning/attention_mechanism.hpp"
#include <memory>
#include <string>
#include <map>
#include <set>
#include <mutex>

namespace dpan {
namespace attention {

/// Configuration for structural attention head
struct StructuralAttentionConfig {
    /// Weight for Jaccard similarity component (0.0 to 1.0)
    /// Controls how much sub-pattern overlap matters
    /// Default: 0.8f
    float jaccard_weight = 0.8f;

    /// Weight for structure size similarity (0.0 to 1.0)
    /// Controls how much similar sub-pattern count matters
    /// Default: 0.2f
    float size_weight = 0.2f;

    /// Temperature for softmax normalization
    /// Lower temperature = sharper distribution (winner-take-all)
    /// Higher temperature = softer distribution (more uniform)
    /// Default: 1.0f
    float temperature = 1.0f;

    /// Minimum structural similarity threshold
    /// Patterns below this threshold get zero weight
    /// Default: 0.0f (no threshold)
    float similarity_threshold = 0.0f;

    /// Penalty for atomic patterns (patterns without sub-patterns)
    /// When comparing to composite patterns
    /// Default: 0.5f (half the score)
    float atomic_penalty = 0.5f;

    /// Enable caching of structural scores
    /// Default: true
    bool enable_caching = true;

    /// Cache size (if caching is enabled)
    /// Default: 1000
    size_t cache_size = 1000;

    /// Enable debug logging
    /// Default: false
    bool debug_logging = false;

    /// Validate configuration
    bool Validate() const {
        if (jaccard_weight < 0.0f || jaccard_weight > 1.0f) return false;
        if (size_weight < 0.0f || size_weight > 1.0f) return false;
        if (std::abs((jaccard_weight + size_weight) - 1.0f) > 0.01f) return false;  // Should sum to 1.0
        if (temperature <= 0.0f) return false;
        if (similarity_threshold < 0.0f || similarity_threshold > 1.0f) return false;
        if (atomic_penalty < 0.0f || atomic_penalty > 1.0f) return false;
        return true;
    }
};

/// Structural Attention Head
///
/// Implements structure-based attention for composite patterns.
/// Uses Jaccard similarity for sub-pattern overlap.
///
/// Scoring formula:
///   For composite patterns:
///     jaccard = |A ∩ B| / |A ∪ B|
///     size_sim = 1 - |size(A) - size(B)| / max(size(A), size(B))
///     raw_score = jaccard_weight * jaccard + size_weight * size_sim
///
///   For atomic patterns:
///     raw_score = atomic_penalty (if both are atomic: 1.0)
///
///   final_weight = softmax(raw_score / temperature)
///
/// Properties:
/// - Structure-aware: Focuses on pattern composition
/// - Handles hierarchies: Works with composite patterns
/// - Configurable weighting: Balance Jaccard vs size similarity
/// - Normalized: Outputs proper probability distribution
class StructuralAttentionHead : public AttentionMechanism {
public:
    /// Constructor
    ///
    /// @param config Structural attention configuration
    explicit StructuralAttentionHead(
        const StructuralAttentionConfig& config = StructuralAttentionConfig());

    /// Destructor
    ~StructuralAttentionHead() override = default;

    // ========================================================================
    // AttentionMechanism Interface Implementation
    // ========================================================================

    /// Compute structural attention weights
    ///
    /// @param query The query pattern
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector (not used in structural attention)
    /// @return Map from pattern ID to structural attention weight
    std::map<PatternID, float> ComputeAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Compute detailed structural attention scores
    ///
    /// @param query The query pattern
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector
    /// @return Vector of detailed attention scores with structural components
    std::vector<AttentionScore> ComputeDetailedAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Apply structural attention to weight and rank predictions
    ///
    /// @param query The query pattern
    /// @param predictions Initial predictions
    /// @param context Current context vector
    /// @return Weighted and re-ranked predictions (sorted by structural score)
    std::vector<std::pair<PatternID, float>> ApplyAttention(
        PatternID query,
        const std::vector<PatternID>& predictions,
        const ContextVector& context
    ) override;

    /// Set pattern database
    ///
    /// @param db Pointer to pattern database
    void SetPatternDatabase(PatternDatabase* db) override;

    /// Get current configuration (returns base AttentionConfig)
    ///
    /// @return Reference to attention configuration
    const AttentionConfig& GetConfig() const override;

    /// Update configuration
    ///
    /// @param config New attention configuration
    void SetConfig(const AttentionConfig& config) override;

    /// Clear structural score cache
    void ClearCache() override;

    /// Get statistics about structural attention usage
    ///
    /// @return Map of statistic name to value
    std::map<std::string, float> GetStatistics() const override;

    // ========================================================================
    // Structural Attention Specific Methods
    // ========================================================================

    /// Get structural attention specific configuration
    ///
    /// @return Reference to structural attention configuration
    const StructuralAttentionConfig& GetStructuralConfig() const;

    /// Set structural attention specific configuration
    ///
    /// @param config New structural attention configuration
    void SetStructuralConfig(const StructuralAttentionConfig& config);

protected:
    /// Compute structural similarity scores for candidates
    ///
    /// Computes Jaccard similarity and size similarity
    ///
    /// @param query Query pattern ID
    /// @param candidates List of candidate pattern IDs
    /// @return Vector of structural scores (same order as candidates)
    std::vector<float> ComputeStructuralScores(
        PatternID query,
        const std::vector<PatternID>& candidates) const;

    /// Compute Jaccard similarity between two pattern structures
    ///
    /// @param query_subpatterns Sub-patterns of query
    /// @param candidate_subpatterns Sub-patterns of candidate
    /// @return Jaccard similarity (0.0 to 1.0)
    float ComputeJaccardSimilarity(
        const std::vector<PatternID>& query_subpatterns,
        const std::vector<PatternID>& candidate_subpatterns) const;

    /// Compute size similarity between two patterns
    ///
    /// @param query_size Number of sub-patterns in query
    /// @param candidate_size Number of sub-patterns in candidate
    /// @return Size similarity (0.0 to 1.0)
    float ComputeSizeSimilarity(
        size_t query_size,
        size_t candidate_size) const;

    /// Normalize structural scores to attention weights
    ///
    /// Applies softmax with temperature scaling
    ///
    /// @param scores Raw structural scores
    /// @return Normalized attention weights
    std::vector<float> NormalizeScores(const std::vector<float>& scores) const;

    /// Log debug message if debug logging is enabled
    ///
    /// @param message Debug message
    void LogDebug(const std::string& message) const;

private:
    /// Structural attention configuration
    StructuralAttentionConfig config_;

    /// Base attention configuration (for interface compatibility)
    AttentionConfig base_config_;

    /// Pattern database pointer
    PatternDatabase* pattern_db_;

    /// Cache for structural scores (pair<pattern_id, pattern_id> -> score)
    mutable std::map<std::pair<PatternID, PatternID>, float> structural_cache_;

    /// Mutex for thread-safe cache access
    mutable std::mutex cache_mutex_;

    /// Statistics counters
    mutable size_t attention_computations_;
    mutable size_t structural_computations_;
    mutable size_t cache_hits_;
    mutable size_t cache_misses_;
};

} // namespace attention
} // namespace dpan

#endif // DPAN_LEARNING_STRUCTURAL_ATTENTION_HEAD_HPP
