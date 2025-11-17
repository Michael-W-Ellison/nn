// File: src/association/spatial_learner.hpp
#pragma once

#include "core/types.hpp"
#include <vector>
#include <unordered_map>
#include <deque>
#include <optional>

namespace dpan {

/// SpatialLearner: Analyzes spatial relationships between patterns
///
/// Learns spatial associations by tracking the contexts in which patterns
/// appear and identifying patterns that occur in similar spatial contexts.
///
/// Spatial similarity is computed using cosine similarity between average
/// context vectors. Patterns with high context similarity are considered
/// spatially related.
///
/// Thread-safety: Not thread-safe. External synchronization required.
class SpatialLearner {
public:
    /// Spatial context observation for a pattern
    struct SpatialContext {
        ContextVector context;                   // Context at time of observation
        Timestamp timestamp;                     // When observed
        std::vector<PatternID> co_occurring_patterns;  // Patterns active at same time
    };

    /// Aggregated spatial statistics for a pattern
    struct SpatialStats {
        ContextVector average_context;           // Average context across observations
        uint32_t observation_count{0};           // Number of observations
        Timestamp last_observed{};               // Last observation time
    };

    /// Configuration for spatial learning
    struct Config {
        Config() = default;
        /// Minimum similarity to consider patterns spatially related
        float min_similarity_threshold{0.7f};
        /// Minimum observations needed for reliable statistics
        uint32_t min_observations{3};
        /// Maximum context history to keep per pattern
        size_t max_history{1000};
        /// Learning rate for updating average context
        float learning_rate{0.1f};
    };

    // ========================================================================
    // Construction
    // ========================================================================

    SpatialLearner();
    explicit SpatialLearner(const Config& config);
    ~SpatialLearner() = default;

    // ========================================================================
    // Recording Spatial Context
    // ========================================================================

    /// Record spatial context for a pattern activation
    /// @param pattern Pattern that was activated
    /// @param context Context vector describing spatial properties
    /// @param timestamp Time of observation (defaults to now)
    void RecordSpatialContext(
        PatternID pattern,
        const ContextVector& context,
        Timestamp timestamp = Timestamp::Now()
    );

    /// Record spatial context with co-occurring patterns
    /// @param pattern Primary pattern
    /// @param context Context vector
    /// @param co_occurring Other patterns active at same time
    /// @param timestamp Time of observation
    void RecordSpatialContext(
        PatternID pattern,
        const ContextVector& context,
        const std::vector<PatternID>& co_occurring,
        Timestamp timestamp = Timestamp::Now()
    );

    // ========================================================================
    // Querying Spatial Relationships
    // ========================================================================

    /// Check if two patterns are spatially related
    /// @param p1 First pattern
    /// @param p2 Second pattern
    /// @param threshold Similarity threshold (defaults to config value)
    /// @return True if context similarity >= threshold
    bool AreSpatiallyRelated(
        PatternID p1,
        PatternID p2,
        float threshold = -1.0f  // -1 means use config default
    ) const;

    /// Get average context vector for a pattern
    /// @param pattern Query pattern
    /// @return Average context, or empty vector if no observations
    ContextVector GetAverageContext(PatternID pattern) const;

    /// Get spatial statistics for a pattern
    /// @param pattern Query pattern
    /// @return Optional statistics (nullopt if no observations)
    std::optional<SpatialStats> GetSpatialStats(PatternID pattern) const;

    /// Get spatial similarity between two patterns
    /// @param p1 First pattern
    /// @param p2 Second pattern
    /// @return Cosine similarity of average contexts [0,1], or 0.0 if insufficient data
    float GetSpatialSimilarity(PatternID p1, PatternID p2) const;

    /// Get patterns with similar spatial profiles
    /// @param pattern Query pattern
    /// @param min_similarity Minimum similarity threshold
    /// @return Vector of (pattern, similarity) pairs sorted by similarity
    std::vector<std::pair<PatternID, float>> GetSpatiallySimilar(
        PatternID pattern,
        float min_similarity = 0.0f
    ) const;

    /// Get all context observations for a pattern
    /// @param pattern Query pattern
    /// @return Vector of spatial contexts (may be empty)
    std::vector<SpatialContext> GetContextHistory(PatternID pattern) const;

    // ========================================================================
    // Maintenance
    // ========================================================================

    /// Remove old observations for a pattern
    /// @param pattern Pattern to prune
    /// @param max_to_keep Maximum observations to keep (oldest removed first)
    void PruneHistory(PatternID pattern, size_t max_to_keep);

    /// Clear all tracked data
    void Clear();

    /// Clear data for a specific pattern
    /// @param pattern Pattern to clear
    void ClearPattern(PatternID pattern);

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get total number of observations across all patterns
    size_t GetTotalObservations() const;

    /// Get number of unique patterns tracked
    size_t GetPatternCount() const { return spatial_stats_.size(); }

    /// Get number of observations for a specific pattern
    /// @param pattern Query pattern
    /// @return Observation count, or 0 if pattern not tracked
    size_t GetObservationCount(PatternID pattern) const;

    /// Get configuration
    const Config& GetConfig() const { return config_; }
    void SetConfig(const Config& config) { config_ = config; }

private:
    Config config_;

    // Spatial statistics per pattern
    std::unordered_map<PatternID, SpatialStats> spatial_stats_;

    // Context history per pattern (for detailed analysis)
    std::unordered_map<PatternID, std::deque<SpatialContext>> context_history_;

    // Helper methods

    /// Update average context for a pattern using exponential moving average
    void UpdateAverageContext(
        PatternID pattern,
        const ContextVector& observed_context,
        Timestamp timestamp
    );

    /// Check if pattern has sufficient observations
    bool HasSufficientObservations(PatternID pattern) const;
};

} // namespace dpan
