// File: include/learning/semantic_attention_head.hpp
//
// Semantic Attention Head for DPAN
//
// Specialized attention mechanism that focuses on content-based similarity
// using pattern data metrics like edit distance, feature overlap, etc.
//
// This head is particularly appropriate for:
// - Text patterns (using edit distance, string similarity)
// - Data patterns (using feature overlap, cosine similarity)
// - Structured patterns (using structural similarity)
//
// The semantic head computes attention weights based on how similar
// the content of patterns is, rather than just using learned features.

#ifndef DPAN_LEARNING_SEMANTIC_ATTENTION_HEAD_HPP
#define DPAN_LEARNING_SEMANTIC_ATTENTION_HEAD_HPP

#include "learning/attention_mechanism.hpp"
#include "similarity/similarity_metric.hpp"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <mutex>

namespace dpan {

// Forward declarations
class PatternDatabase;

namespace attention {

/// Configuration for semantic attention head
struct SemanticAttentionConfig {
    /// Temperature parameter for softmax normalization
    /// Higher values (>1.0) make distribution more uniform
    /// Lower values (<1.0) make distribution more peaked
    /// Default: 1.0 (standard softmax)
    float temperature = 1.0f;

    /// Minimum similarity threshold for considering a pattern
    /// Patterns with similarity below this are given zero weight
    /// Range: [0.0, 1.0], Default: 0.0 (no threshold)
    float similarity_threshold = 0.0f;

    /// Enable caching of similarity computations
    /// Default: true
    bool enable_caching = true;

    /// Maximum cache size (LRU eviction)
    /// Default: 1000
    size_t cache_size = 1000;

    /// Enable debug logging
    /// Default: false
    bool debug_logging = false;

    /// Validate configuration
    bool Validate() const {
        if (temperature <= 0.0f) return false;
        if (similarity_threshold < 0.0f || similarity_threshold > 1.0f) return false;
        return true;
    }
};

/// Semantic Attention Head
///
/// Computes attention weights based on content similarity using
/// configurable similarity metrics. This allows the attention mechanism
/// to focus on patterns that are semantically similar in their content.
///
/// Example usage:
/// @code
///   // Create semantic attention head with cosine similarity
///   auto similarity_metric = std::make_shared<CosineSimilarity>();
///   SemanticAttentionConfig config;
///   config.temperature = 0.8f;  // Sharper distribution
///
///   auto semantic_head = std::make_unique<SemanticAttentionHead>(
///       config, similarity_metric);
///   semantic_head->SetPatternDatabase(pattern_db);
///
///   // Compute content-based attention
///   auto weights = semantic_head->ComputeAttention(query, candidates, context);
/// @endcode
class SemanticAttentionHead : public AttentionMechanism {
public:
    /// Constructor
    ///
    /// @param config Semantic attention configuration
    /// @param similarity_metric Similarity metric to use for pattern comparison
    explicit SemanticAttentionHead(
        const SemanticAttentionConfig& config = SemanticAttentionConfig(),
        std::shared_ptr<SimilarityMetric> similarity_metric = nullptr
    );

    /// Destructor
    ~SemanticAttentionHead() override = default;

    // ========================================================================
    // AttentionMechanism Interface Implementation
    // ========================================================================

    /// Compute attention weights based on content similarity
    ///
    /// @param query The query pattern
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector (may be used for context-aware similarity)
    /// @return Map from pattern ID to attention weight [0.0, 1.0]
    std::map<PatternID, float> ComputeAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Compute detailed attention scores with component breakdown
    ///
    /// @param query The query pattern
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector
    /// @return Vector of detailed attention scores
    std::vector<AttentionScore> ComputeDetailedAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Apply attention to weight and rank predictions
    ///
    /// @param query The query pattern
    /// @param predictions Initial predictions
    /// @param context Current context vector
    /// @return Weighted and re-ranked predictions
    std::vector<std::pair<PatternID, float>> ApplyAttention(
        PatternID query,
        const std::vector<PatternID>& predictions,
        const ContextVector& context
    ) override;

    /// Set pattern database for retrieving pattern information
    ///
    /// @param db Pointer to pattern database
    void SetPatternDatabase(PatternDatabase* db) override;

    /// Get current configuration (converted to base AttentionConfig)
    ///
    /// @return Reference to attention configuration
    const AttentionConfig& GetConfig() const override;

    /// Update configuration
    ///
    /// @param config New attention configuration
    void SetConfig(const AttentionConfig& config) override;

    /// Clear cached similarity computations
    void ClearCache() override;

    /// Get statistics about attention usage
    ///
    /// @return Map of statistic name to value
    std::map<std::string, float> GetStatistics() const override;

    // ========================================================================
    // Semantic Attention Specific Methods
    // ========================================================================

    /// Set similarity metric
    ///
    /// @param metric Similarity metric to use
    void SetSimilarityMetric(std::shared_ptr<SimilarityMetric> metric);

    /// Get similarity metric
    ///
    /// @return Pointer to current similarity metric
    std::shared_ptr<SimilarityMetric> GetSimilarityMetric() const;

    /// Get semantic attention configuration
    ///
    /// @return Reference to semantic attention configuration
    const SemanticAttentionConfig& GetSemanticConfig() const;

    /// Set semantic attention configuration
    ///
    /// @param config New semantic attention configuration
    void SetSemanticConfig(const SemanticAttentionConfig& config);

protected:
    /// Compute content similarity scores for candidates
    ///
    /// @param query Query pattern ID
    /// @param candidates Candidate pattern IDs
    /// @return Vector of similarity scores [0.0, 1.0]
    std::vector<float> ComputeSimilarityScores(
        PatternID query,
        const std::vector<PatternID>& candidates
    ) const;

    /// Apply temperature scaling and softmax normalization
    ///
    /// @param scores Raw similarity scores
    /// @return Normalized attention weights
    std::vector<float> NormalizeScores(const std::vector<float>& scores) const;

    /// Log debug information if debug_logging is enabled
    ///
    /// @param message Debug message
    void LogDebug(const std::string& message) const;

private:
    /// Semantic attention configuration
    SemanticAttentionConfig config_;

    /// Base attention configuration (for interface compatibility)
    AttentionConfig base_config_;

    /// Similarity metric for computing pattern similarity
    std::shared_ptr<SimilarityMetric> similarity_metric_;

    /// Pattern database pointer
    PatternDatabase* pattern_db_;

    /// Similarity computation cache
    /// Maps (query_id, candidate_id) -> similarity score
    mutable std::map<std::pair<PatternID, PatternID>, float> similarity_cache_;

    /// Mutex for thread-safe cache access
    mutable std::mutex cache_mutex_;

    /// Statistics counters
    mutable size_t attention_computations_;
    mutable size_t similarity_computations_;
    mutable size_t cache_hits_;
    mutable size_t cache_misses_;
};

} // namespace attention
} // namespace dpan

#endif // DPAN_LEARNING_SEMANTIC_ATTENTION_HEAD_HPP
