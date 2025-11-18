// File: include/learning/basic_attention.hpp
//
// BasicAttentionMechanism: Simple dot-product attention implementation
//
// This class implements a basic attention mechanism using dot-product similarity
// between pattern feature vectors. It supports:
// - Dot-product attention with softmax normalization
// - Temperature scaling for controlling distribution sharpness
// - LRU caching for performance optimization
// - Optional debug logging for inspection
// - Integration with PatternDatabase for feature extraction
//
// Usage:
// @code
//   AttentionConfig config;
//   config.temperature = 1.0f;
//   config.enable_caching = true;
//
//   BasicAttentionMechanism attention(config);
//   attention.SetPatternDatabase(pattern_db);
//
//   auto weights = attention.ComputeAttention(query_id, candidate_ids, context);
//   // weights contains normalized attention weights for each candidate
// @endcode

#ifndef DPAN_LEARNING_BASIC_ATTENTION_HPP
#define DPAN_LEARNING_BASIC_ATTENTION_HPP

#include "learning/attention_mechanism.hpp"
#include "learning/attention_utils.hpp"
#include "core/types.hpp"
#include "storage/pattern_database.hpp"
#include <map>
#include <vector>
#include <list>
#include <mutex>
#include <optional>
#include <iostream>

namespace dpan {
namespace attention {

/// Cache key for attention computation
/// Combines query, candidates, and context into a comparable key
struct CacheKey {
    PatternID query;
    std::vector<PatternID> candidates;
    ContextVector context;

    // Equality comparison for cache lookup
    bool operator==(const CacheKey& other) const;

    // Less-than comparison for std::map
    bool operator<(const CacheKey& other) const;

    // Hash function for use in unordered_map (future optimization)
    struct Hash {
        size_t operator()(const CacheKey& key) const;
    };
};

/// BasicAttentionMechanism: Dot-product attention with caching and debugging
///
/// This class implements the AttentionMechanism interface using scaled dot-product
/// attention (similar to Transformer attention). The attention score between a
/// query pattern and candidate pattern is computed as:
///
/// 1. Extract feature vectors for query and candidates
/// 2. Compute dot-product similarity: score_i = dot(query_features, candidate_i_features)
/// 3. Apply temperature scaling: scaled_i = score_i / temperature
/// 4. Normalize with softmax: weight_i = softmax(scaled_scores)
///
/// **Features:**
/// - **Feature Extraction**: Configurable extraction of pattern features
/// - **Temperature Scaling**: Control distribution sharpness
/// - **LRU Caching**: Cache recent attention computations
/// - **Debug Logging**: Optional logging of intermediate values
/// - **Thread Safety**: Mutex-protected cache operations
///
/// **Performance:**
/// - Caching provides ~10-100x speedup for repeated queries
/// - Feature extraction is the main bottleneck
/// - Scales linearly with number of candidates
class BasicAttentionMechanism : public AttentionMechanism {
public:
    /// Constructor with configuration
    ///
    /// @param config Attention configuration (temperature, caching, etc.)
    explicit BasicAttentionMechanism(const AttentionConfig& config = {});

    /// Destructor
    ~BasicAttentionMechanism() override = default;

    // ========================================================================
    // AttentionMechanism Interface Implementation
    // ========================================================================

    /// Compute attention weights for candidate patterns
    ///
    /// @param query Query pattern ID
    /// @param candidates Candidate pattern IDs to compute attention over
    /// @param context Current context vector
    /// @return Map from pattern ID to attention weight (sum to 1.0)
    std::map<PatternID, float> ComputeAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Compute detailed attention scores with component breakdown
    ///
    /// @param query Query pattern ID
    /// @param candidates Candidate pattern IDs
    /// @param context Current context vector
    /// @return Vector of AttentionScore with detailed components
    std::vector<AttentionScore> ComputeDetailedAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Apply attention to reweight predictions
    ///
    /// Combines attention weights with existing association scores using
    /// configurable weighting: final = attention_weight * attention_score +
    ///                                  association_weight * association_score
    ///
    /// @param query Query pattern ID
    /// @param predictions Initial predictions with scores
    /// @param context Current context vector
    /// @return Reweighted predictions sorted by final score
    std::vector<std::pair<PatternID, float>> ApplyAttention(
        PatternID query,
        const std::vector<PatternID>& predictions,
        const ContextVector& context
    ) override;

    /// Set pattern database for feature extraction
    ///
    /// @param db Pointer to pattern database (not owned)
    void SetPatternDatabase(PatternDatabase* db) override;

    /// Get current configuration
    ///
    /// @return Reference to configuration
    const AttentionConfig& GetConfig() const override;

    /// Set new configuration
    ///
    /// @param config New configuration to use
    /// @note Clears cache when configuration changes
    void SetConfig(const AttentionConfig& config) override;

    /// Clear attention cache
    ///
    /// Forces recomputation of all attention weights.
    /// Useful after major database updates.
    void ClearCache() override;

    /// Get statistics about attention computation
    ///
    /// @return Map of statistic names to values
    std::map<std::string, float> GetStatistics() const override;

    // ========================================================================
    // Additional Configuration
    // ========================================================================

    /// Set feature extraction configuration
    ///
    /// Controls which metadata features are included in attention computation.
    ///
    /// @param config Feature extraction configuration
    void SetFeatureConfig(const FeatureExtractionConfig& config);

    /// Get feature extraction configuration
    ///
    /// @return Current feature extraction configuration
    const FeatureExtractionConfig& GetFeatureConfig() const;

    /// Set debug output stream
    ///
    /// When debug_logging is enabled in config, debug information is
    /// written to this stream.
    ///
    /// @param os Output stream (e.g., std::cout, file stream)
    void SetDebugStream(std::ostream* os);

private:
    // ========================================================================
    // Core Attention Computation
    // ========================================================================

    /// Compute raw attention scores using dot-product similarity
    ///
    /// @param query_features Query pattern features
    /// @param candidate_features Vector of candidate pattern features
    /// @return Raw similarity scores (not normalized)
    std::vector<float> ComputeRawScores(
        const std::vector<float>& query_features,
        const std::vector<std::vector<float>>& candidate_features
    ) const;

    /// Extract features for multiple patterns
    ///
    /// @param pattern_ids Pattern IDs to extract features for
    /// @return Vector of feature vectors (same order as input)
    std::vector<std::vector<float>> ExtractMultipleFeatures(
        const std::vector<PatternID>& pattern_ids
    ) const;

    // ========================================================================
    // Caching
    // ========================================================================

    /// Check if attention weights are cached
    ///
    /// @param key Cache key
    /// @return Cached weights if found, std::nullopt otherwise
    std::optional<std::map<PatternID, float>> GetCachedAttention(
        const CacheKey& key
    );

    /// Store attention weights in cache
    ///
    /// @param key Cache key
    /// @param weights Attention weights to cache
    void CacheAttention(
        const CacheKey& key,
        const std::map<PatternID, float>& weights
    );

    // ========================================================================
    // Debug Logging
    // ========================================================================

    /// Log debug information if debug_logging is enabled
    ///
    /// @param message Debug message
    void LogDebug(const std::string& message) const;

    /// Log attention computation details
    ///
    /// @param query Query pattern ID
    /// @param raw_scores Raw similarity scores
    /// @param weights Final attention weights
    void LogAttentionDetails(
        PatternID query,
        const std::vector<float>& raw_scores,
        const std::map<PatternID, float>& weights
    ) const;

    // ========================================================================
    // Member Variables
    // ========================================================================

    /// Attention configuration
    AttentionConfig config_;

    /// Feature extraction configuration
    FeatureExtractionConfig feature_config_;

    /// Pattern database for feature extraction (not owned)
    PatternDatabase* pattern_db_;

    /// LRU cache for attention weights
    /// Maps cache key to (weights, timestamp)
    std::map<CacheKey, std::map<PatternID, float>> cache_;

    /// LRU access order (most recent at back)
    std::list<CacheKey> cache_order_;

    /// Mutex for thread-safe cache access
    mutable std::mutex cache_mutex_;

    /// Debug output stream
    std::ostream* debug_stream_;

    /// Statistics counters
    mutable size_t cache_hits_;
    mutable size_t cache_misses_;
    mutable size_t total_computations_;
};

} // namespace attention
} // namespace dpan

#endif // DPAN_LEARNING_BASIC_ATTENTION_HPP
