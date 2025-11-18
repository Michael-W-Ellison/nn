// File: include/learning/context_aware_attention.hpp
//
// ContextAwareAttention: Context-sensitive attention mechanism
//
// This class extends BasicAttentionMechanism to incorporate context similarity
// into attention computation. It maintains a history of contexts where each
// pattern has been activated, and uses this history to boost attention for
// patterns that have appeared in similar contexts.
//
// Key Features:
// - Historical context storage (circular buffer per pattern)
// - Context similarity computation using cosine similarity
// - Combined semantic + context scoring with configurable weights
// - Thread-safe context recording
//
// Usage:
// @code
//   ContextAwareAttention attention(config);
//   attention.SetPatternDatabase(pattern_db);
//
//   // Record pattern activations with their contexts
//   attention.RecordActivation(pattern_id, current_context);
//
//   // Compute context-aware attention
//   auto weights = attention.ComputeAttention(query, candidates, context);
//   // Patterns with similar historical contexts get higher weights
// @endcode

#ifndef DPAN_LEARNING_CONTEXT_AWARE_ATTENTION_HPP
#define DPAN_LEARNING_CONTEXT_AWARE_ATTENTION_HPP

#include "learning/basic_attention.hpp"
#include "core/types.hpp"
#include <deque>
#include <map>
#include <mutex>
#include <vector>

namespace dpan {
namespace attention {

/// Configuration for context-aware attention
struct ContextAwareConfig {
    /// Maximum number of historical contexts to store per pattern
    size_t max_context_history = 10;

    /// Weight for semantic similarity (pattern features)
    float semantic_weight = 0.5f;

    /// Weight for context similarity (historical context matching)
    float context_weight = 0.5f;

    /// Validate configuration
    bool Validate() const {
        if (max_context_history == 0) return false;
        if (semantic_weight < 0.0f || context_weight < 0.0f) return false;

        // Weights should sum to approximately 1.0
        float sum = semantic_weight + context_weight;
        return std::abs(sum - 1.0f) < 0.01f;
    }

    /// Normalize weights to sum to 1.0
    void Normalize() {
        float sum = semantic_weight + context_weight;
        if (sum > 0.0f) {
            semantic_weight /= sum;
            context_weight /= sum;
        }
    }
};

/// ContextAwareAttention: Attention mechanism with context sensitivity
///
/// This class enhances basic attention by considering not just pattern similarity,
/// but also context similarity. For each pattern, it maintains a history of contexts
/// where the pattern has been activated. When computing attention, it combines:
///
/// 1. **Semantic Similarity**: Pattern feature similarity (from BasicAttentionMechanism)
/// 2. **Context Similarity**: How similar the current context is to the pattern's
///    historical activation contexts
///
/// The final attention score is a weighted combination:
/// ```
/// score = semantic_weight * semantic_sim + context_weight * context_sim
/// ```
///
/// **Context History:**
/// - Circular buffer per pattern (configurable size, default 10)
/// - Most recent contexts preserved
/// - Thread-safe updates
///
/// **Context Similarity:**
/// - Cosine similarity between current context and historical contexts
/// - Maximum similarity across all historical contexts is used
/// - Returns 0.5 if no history exists (neutral score)
class ContextAwareAttention : public BasicAttentionMechanism {
public:
    /// Constructor with attention and context-aware configuration
    ///
    /// @param attn_config Attention configuration (temperature, caching, etc.)
    /// @param ctx_config Context-aware configuration (weights, history size)
    explicit ContextAwareAttention(
        const AttentionConfig& attn_config = {},
        const ContextAwareConfig& ctx_config = {}
    );

    /// Destructor
    ~ContextAwareAttention() override = default;

    // ========================================================================
    // Overridden AttentionMechanism Methods
    // ========================================================================

    /// Compute context-aware attention weights
    ///
    /// Combines semantic similarity (pattern features) with context similarity
    /// (historical context matching) to compute attention weights.
    ///
    /// @param query Query pattern ID
    /// @param candidates Candidate pattern IDs
    /// @param context Current context vector
    /// @return Map from pattern ID to attention weight (sum to 1.0)
    std::map<PatternID, float> ComputeAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    // ========================================================================
    // Context History Management
    // ========================================================================

    /// Record a pattern activation with its context
    ///
    /// Stores the context in the pattern's activation history for future
    /// context similarity computations. Uses a circular buffer to maintain
    /// only the most recent contexts.
    ///
    /// @param pattern_id Pattern that was activated
    /// @param context Context in which the pattern was activated
    void RecordActivation(PatternID pattern_id, const ContextVector& context);

    /// Get historical contexts for a pattern
    ///
    /// @param pattern_id Pattern to query
    /// @return Vector of historical contexts (most recent first)
    std::vector<ContextVector> GetContextHistory(PatternID pattern_id) const;

    /// Clear all context history
    ///
    /// Removes all stored historical contexts for all patterns.
    void ClearContextHistory();

    /// Clear context history for a specific pattern
    ///
    /// @param pattern_id Pattern to clear history for
    void ClearContextHistory(PatternID pattern_id);

    // ========================================================================
    // Context Similarity Computation
    // ========================================================================

    /// Compute context similarity between query context and pattern's history
    ///
    /// Retrieves the pattern's historical contexts and computes cosine similarity
    /// with the query context. Returns the maximum similarity across all historical
    /// contexts.
    ///
    /// @param query_context Current context vector
    /// @param candidate_pattern Pattern to compute similarity for
    /// @return Maximum cosine similarity [0, 1], or 0.5 if no history
    float ComputeContextSimilarity(
        const ContextVector& query_context,
        PatternID candidate_pattern
    ) const;

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Set context-aware configuration
    ///
    /// @param config New configuration (will be validated and normalized)
    void SetContextConfig(const ContextAwareConfig& config);

    /// Get context-aware configuration
    ///
    /// @return Current configuration
    const ContextAwareConfig& GetContextConfig() const;

    /// Get statistics about context-aware attention
    ///
    /// @return Map of statistic names to values
    std::map<std::string, float> GetStatistics() const override;

private:
    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Compute raw context similarity scores for all candidates
    ///
    /// @param query_context Current context vector
    /// @param candidates Candidate pattern IDs
    /// @return Vector of context similarity scores (same order as candidates)
    std::vector<float> ComputeContextScores(
        const ContextVector& query_context,
        const std::vector<PatternID>& candidates
    ) const;

    /// Combine semantic and context scores
    ///
    /// @param semantic_scores Scores from pattern feature similarity
    /// @param context_scores Scores from context similarity
    /// @return Combined scores
    std::vector<float> CombineScores(
        const std::vector<float>& semantic_scores,
        const std::vector<float>& context_scores
    ) const;

    // ========================================================================
    // Member Variables
    // ========================================================================

    /// Context-aware configuration
    ContextAwareConfig ctx_config_;

    /// Context history storage
    /// Maps pattern ID to circular buffer of historical contexts
    mutable std::map<PatternID, std::deque<ContextVector>> context_history_;

    /// Mutex for thread-safe context history access
    mutable std::mutex history_mutex_;

    /// Statistics counters
    mutable size_t context_similarity_computations_;
    mutable size_t context_activations_recorded_;
};

} // namespace attention
} // namespace dpan

#endif // DPAN_LEARNING_CONTEXT_AWARE_ATTENTION_HPP
