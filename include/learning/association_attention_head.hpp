// File: include/learning/association_attention_head.hpp
//
// Association Attention Head for DPAN
//
// Implements attention using existing association strengths from the
// association matrix. This provides a baseline that directly reflects
// learned associations.
//
// This attention mechanism is particularly useful for:
// - Baseline comparison for other attention heads
// - Leveraging existing association learning
// - Pattern prediction based on learned relationships
//
// Example usage:
// @code
//   AssociationAttentionConfig config;
//   config.use_contextual_strength = true;
//   config.temperature = 1.0f;
//
//   auto assoc_head = std::make_unique<AssociationAttentionHead>(config);
//   assoc_head->SetAssociationMatrix(matrix);
//
//   auto weights = assoc_head->ComputeAttention(query, candidates, context);
//   // Candidates with stronger associations will have higher weights
// @endcode

#ifndef DPAN_LEARNING_ASSOCIATION_ATTENTION_HEAD_HPP
#define DPAN_LEARNING_ASSOCIATION_ATTENTION_HEAD_HPP

#include "learning/attention_mechanism.hpp"
#include "association/association_matrix.hpp"
#include <memory>
#include <string>
#include <map>
#include <mutex>

namespace dpan {
namespace attention {

/// Configuration for association attention head
struct AssociationAttentionConfig {
    /// Temperature for softmax normalization
    /// Lower temperature = sharper distribution (winner-take-all)
    /// Higher temperature = softer distribution (more uniform)
    /// Default: 1.0f
    float temperature = 1.0f;

    /// Use contextual strength instead of base strength
    /// If true, modulates strength based on current context similarity
    /// Default: false
    bool use_contextual_strength = false;

    /// Minimum association strength threshold
    /// Associations below this threshold get zero weight
    /// Default: 0.0f (no threshold)
    float strength_threshold = 0.0f;

    /// Default strength for missing associations
    /// Used when query->candidate association doesn't exist
    /// Default: 0.1f
    float default_strength = 0.1f;

    /// Enable caching of association lookups
    /// Default: false (association matrix is already fast)
    bool enable_caching = false;

    /// Cache size (if caching is enabled)
    /// Default: 100
    size_t cache_size = 100;

    /// Enable debug logging
    /// Default: false
    bool debug_logging = false;

    /// Validate configuration
    bool Validate() const {
        if (temperature <= 0.0f) return false;
        if (strength_threshold < 0.0f || strength_threshold > 1.0f) return false;
        if (default_strength < 0.0f || default_strength > 1.0f) return false;
        return true;
    }
};

/// Association Attention Head
///
/// Implements attention using existing association strengths.
/// Provides a baseline that directly reflects learned associations.
///
/// Scoring formula:
///   For each candidate:
///     if association(query -> candidate) exists:
///       raw_score = association.GetStrength()  (or GetContextualStrength())
///     else:
///       raw_score = default_strength
///
///   if raw_score < strength_threshold:
///       raw_score = 0.0
///
///   final_weight = softmax(raw_score / temperature)
///
/// Properties:
/// - Direct: Uses learned association strengths directly
/// - Fast: Simple lookup in association matrix
/// - Baseline: Good comparison for other attention mechanisms
/// - Context-aware: Optional contextual strength modulation
class AssociationAttentionHead : public AttentionMechanism {
public:
    /// Constructor
    ///
    /// @param config Association attention configuration
    explicit AssociationAttentionHead(
        const AssociationAttentionConfig& config = AssociationAttentionConfig());

    /// Destructor
    ~AssociationAttentionHead() override = default;

    // ========================================================================
    // AttentionMechanism Interface Implementation
    // ========================================================================

    /// Compute association-based attention weights
    ///
    /// @param query The query pattern
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector
    /// @return Map from pattern ID to association attention weight
    std::map<PatternID, float> ComputeAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Compute detailed association attention scores
    ///
    /// @param query The query pattern
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector
    /// @return Vector of detailed attention scores with association components
    std::vector<AttentionScore> ComputeDetailedAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Apply association attention to weight and rank predictions
    ///
    /// @param query The query pattern
    /// @param predictions Initial predictions
    /// @param context Current context vector
    /// @return Weighted and re-ranked predictions (sorted by association score)
    std::vector<std::pair<PatternID, float>> ApplyAttention(
        PatternID query,
        const std::vector<PatternID>& predictions,
        const ContextVector& context
    ) override;

    /// Set pattern database (not used by this head, but required by interface)
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

    /// Clear association lookup cache
    void ClearCache() override;

    /// Get statistics about association attention usage
    ///
    /// @return Map of statistic name to value
    std::map<std::string, float> GetStatistics() const override;

    // ========================================================================
    // Association Attention Specific Methods
    // ========================================================================

    /// Get association attention specific configuration
    ///
    /// @return Reference to association attention configuration
    const AssociationAttentionConfig& GetAssociationConfig() const;

    /// Set association attention specific configuration
    ///
    /// @param config New association attention configuration
    void SetAssociationConfig(const AssociationAttentionConfig& config);

    /// Set association matrix
    ///
    /// @param matrix Pointer to association matrix
    void SetAssociationMatrix(AssociationMatrix* matrix);

protected:
    /// Compute association strength scores for candidates
    ///
    /// Looks up association strengths from matrix
    ///
    /// @param query Query pattern ID
    /// @param candidates List of candidate pattern IDs
    /// @param context Current context (used if use_contextual_strength=true)
    /// @return Vector of association scores (same order as candidates)
    std::vector<float> ComputeAssociationScores(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context) const;

    /// Normalize association scores to attention weights
    ///
    /// Applies softmax with temperature scaling
    ///
    /// @param scores Raw association scores
    /// @return Normalized attention weights
    std::vector<float> NormalizeScores(const std::vector<float>& scores) const;

    /// Log debug message if debug logging is enabled
    ///
    /// @param message Debug message
    void LogDebug(const std::string& message) const;

private:
    /// Association attention configuration
    AssociationAttentionConfig config_;

    /// Base attention configuration (for interface compatibility)
    AttentionConfig base_config_;

    /// Association matrix pointer
    AssociationMatrix* association_matrix_;

    /// Pattern database pointer (not used, but kept for interface)
    PatternDatabase* pattern_db_;

    /// Cache for association scores (pair<pattern_id, pattern_id> -> score)
    mutable std::map<std::pair<PatternID, PatternID>, float> association_cache_;

    /// Mutex for thread-safe cache access
    mutable std::mutex cache_mutex_;

    /// Statistics counters
    mutable size_t attention_computations_;
    mutable size_t association_lookups_;
    mutable size_t cache_hits_;
    mutable size_t cache_misses_;
    mutable size_t missing_associations_;
};

} // namespace attention
} // namespace dpan

#endif // DPAN_LEARNING_ASSOCIATION_ATTENTION_HEAD_HPP
