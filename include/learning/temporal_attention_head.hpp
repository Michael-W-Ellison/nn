// File: include/learning/temporal_attention_head.hpp
//
// Temporal Attention Head for DPAN
//
// Implements recency-based attention that favors recently activated patterns.
// Uses exponential decay: score = exp(-time_since_last_activation / decay_constant)
//
// This attention mechanism is particularly useful for:
// - Sequential pattern prediction
// - Time-series data
// - Recency-biased context matching
//
// Example usage:
// @code
//   TemporalAttentionConfig config;
//   config.decay_constant_ms = 1000.0f;  // 1 second decay
//   config.temperature = 1.0f;
//
//   auto temporal_head = std::make_unique<TemporalAttentionHead>(config);
//   temporal_head->SetPatternDatabase(db);
//
//   auto weights = temporal_head->ComputeAttention(query, candidates, context);
//   // Recent patterns will have higher weights
// @endcode

#ifndef DPAN_LEARNING_TEMPORAL_ATTENTION_HEAD_HPP
#define DPAN_LEARNING_TEMPORAL_ATTENTION_HEAD_HPP

#include "learning/attention_mechanism.hpp"
#include <memory>
#include <string>
#include <map>
#include <mutex>

namespace dpan {
namespace attention {

/// Configuration for temporal attention head
struct TemporalAttentionConfig {
    /// Decay constant in milliseconds
    /// Controls how quickly old patterns are discounted
    /// Smaller values = faster decay (favor more recent patterns)
    /// Larger values = slower decay (more gradual temporal falloff)
    /// Default: 1000.0f (1 second)
    float decay_constant_ms = 1000.0f;

    /// Temperature for softmax normalization
    /// Lower temperature = sharper distribution (winner-take-all)
    /// Higher temperature = softer distribution (more uniform)
    /// Default: 1.0f
    float temperature = 1.0f;

    /// Minimum age threshold in milliseconds
    /// Patterns accessed more recently than this are considered "current"
    /// Default: 0.0f (no threshold)
    float min_age_threshold_ms = 0.0f;

    /// Enable caching of temporal scores
    /// Caching is less useful for temporal attention since scores change over time
    /// Default: false
    bool enable_caching = false;

    /// Cache size (if caching is enabled)
    /// Default: 100
    size_t cache_size = 100;

    /// Enable debug logging
    /// Default: false
    bool debug_logging = false;

    /// Validate configuration
    bool Validate() const {
        if (decay_constant_ms <= 0.0f) return false;
        if (temperature <= 0.0f) return false;
        if (min_age_threshold_ms < 0.0f) return false;
        return true;
    }
};

/// Temporal Attention Head
///
/// Implements recency-based attention using exponential decay.
/// Recently activated patterns receive higher attention weights.
///
/// Scoring formula:
///   time_delta = current_time - last_accessed_time
///   raw_score = exp(-time_delta / decay_constant)
///   final_weight = softmax(raw_score / temperature)
///
/// Properties:
/// - Time-aware: Automatically adjusts based on current time
/// - Configurable decay: Control how quickly old patterns are discounted
/// - Normalized: Outputs proper probability distribution
/// - Efficient: Simple exponential computation
class TemporalAttentionHead : public AttentionMechanism {
public:
    /// Constructor
    ///
    /// @param config Temporal attention configuration
    explicit TemporalAttentionHead(const TemporalAttentionConfig& config = TemporalAttentionConfig());

    /// Destructor
    ~TemporalAttentionHead() override = default;

    // ========================================================================
    // AttentionMechanism Interface Implementation
    // ========================================================================

    /// Compute temporal attention weights
    ///
    /// @param query The query pattern (not used in temporal attention)
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector (not used in temporal attention)
    /// @return Map from pattern ID to temporal attention weight
    std::map<PatternID, float> ComputeAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Compute detailed temporal attention scores
    ///
    /// @param query The query pattern
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector
    /// @return Vector of detailed attention scores with temporal components
    std::vector<AttentionScore> ComputeDetailedAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Apply temporal attention to weight and rank predictions
    ///
    /// @param query The query pattern
    /// @param predictions Initial predictions
    /// @param context Current context vector
    /// @return Weighted and re-ranked predictions (sorted by temporal score)
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

    /// Clear temporal score cache
    void ClearCache() override;

    /// Get statistics about temporal attention usage
    ///
    /// @return Map of statistic name to value
    std::map<std::string, float> GetStatistics() const override;

    // ========================================================================
    // Temporal Attention Specific Methods
    // ========================================================================

    /// Get temporal attention specific configuration
    ///
    /// @return Reference to temporal attention configuration
    const TemporalAttentionConfig& GetTemporalConfig() const;

    /// Set temporal attention specific configuration
    ///
    /// @param config New temporal attention configuration
    void SetTemporalConfig(const TemporalAttentionConfig& config);

    /// Get current time (for testing/debugging)
    ///
    /// @return Current timestamp
    static Timestamp GetCurrentTime();

protected:
    /// Compute temporal scores for candidates
    ///
    /// Computes exponential decay scores based on last access time
    ///
    /// @param candidates List of candidate pattern IDs
    /// @return Vector of temporal scores (same order as candidates)
    std::vector<float> ComputeTemporalScores(
        const std::vector<PatternID>& candidates) const;

    /// Normalize temporal scores to attention weights
    ///
    /// Applies softmax with temperature scaling
    ///
    /// @param scores Raw temporal scores
    /// @return Normalized attention weights
    std::vector<float> NormalizeScores(const std::vector<float>& scores) const;

    /// Get time since last access for a pattern
    ///
    /// @param pattern_id Pattern to query
    /// @return Time since last access in milliseconds, or -1 if unknown
    float GetTimeSinceLastAccess(PatternID pattern_id) const;

    /// Log debug message if debug logging is enabled
    ///
    /// @param message Debug message
    void LogDebug(const std::string& message) const;

private:
    /// Temporal attention configuration
    TemporalAttentionConfig config_;

    /// Base attention configuration (for interface compatibility)
    AttentionConfig base_config_;

    /// Pattern database pointer
    PatternDatabase* pattern_db_;

    /// Cache for temporal scores (pattern_id -> (timestamp, score))
    mutable std::map<PatternID, std::pair<uint64_t, float>> temporal_cache_;

    /// Mutex for thread-safe cache access
    mutable std::mutex cache_mutex_;

    /// Statistics counters
    mutable size_t attention_computations_;
    mutable size_t temporal_computations_;
    mutable size_t cache_hits_;
    mutable size_t cache_misses_;
};

} // namespace attention
} // namespace dpan

#endif // DPAN_LEARNING_TEMPORAL_ATTENTION_HEAD_HPP
