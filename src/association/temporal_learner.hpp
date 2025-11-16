// File: src/association/temporal_learner.hpp
#pragma once

#include "core/types.hpp"
#include "association/association_matrix.hpp"  // For PatternPairHash
#include <vector>
#include <deque>
#include <unordered_map>
#include <optional>

namespace dpan {

/// TemporalLearner: Analyzes temporal sequences to detect causal relationships
///
/// Tracks time delays between pattern activations and computes:
/// - Mean delay (μ): average time between pattern occurrences
/// - Standard deviation (σ): variability in delays
/// - Temporal correlation (τ): consistency measure = 1 / (1 + σ/μ)
///
/// Thread-safety: Not thread-safe. External synchronization required.
class TemporalLearner {
public:
    /// Temporal statistics for a pattern pair
    struct TemporalStats {
        uint32_t occurrence_count{0};        // Number of times p1->p2 observed
        int64_t mean_delay_micros{0};        // Mean delay in microseconds
        int64_t stddev_delay_micros{0};      // Standard deviation in microseconds
        float correlation{0.0f};             // Temporal correlation [0,1]
        Timestamp last_updated{};            // Last time stats were updated
    };

    /// Configuration for temporal learning
    struct Config {
        Config() = default;
        /// Maximum delay to consider for temporal correlation (default 10s)
        Timestamp::Duration max_delay{std::chrono::seconds(10)};
        /// Minimum occurrences to compute reliable statistics
        uint32_t min_occurrences{3};
        /// Correlation threshold to consider patterns causally related
        float min_correlation{0.5f};
        /// Window size for tracking activations
        Timestamp::Duration tracking_window{std::chrono::minutes(5)};
    };

    // ========================================================================
    // Construction
    // ========================================================================

    TemporalLearner();
    explicit TemporalLearner(const Config& config);
    ~TemporalLearner() = default;

    // ========================================================================
    // Recording Activations
    // ========================================================================

    /// Record pattern activation at specific time
    /// @param pattern Pattern that was activated
    /// @param timestamp Activation time (defaults to now)
    void RecordActivation(PatternID pattern, Timestamp timestamp = Timestamp::Now());

    /// Record sequence of pattern activations
    /// @param sequence Vector of (timestamp, pattern) pairs
    void RecordSequence(const std::vector<std::pair<Timestamp, PatternID>>& sequence);

    // ========================================================================
    // Querying Temporal Statistics
    // ========================================================================

    /// Get temporal statistics for pattern pair
    /// @param p1 First pattern (predecessor)
    /// @param p2 Second pattern (successor)
    /// @return Optional statistics (nullopt if insufficient data)
    std::optional<TemporalStats> GetTemporalStats(PatternID p1, PatternID p2) const;

    /// Get temporal correlation coefficient for pattern pair
    /// @param p1 First pattern
    /// @param p2 Second pattern
    /// @return Correlation in [0,1], or 0.0 if insufficient data
    float GetTemporalCorrelation(PatternID p1, PatternID p2) const;

    /// Check if two patterns are temporally correlated (causal relationship)
    /// @param p1 First pattern (potential cause)
    /// @param p2 Second pattern (potential effect)
    /// @return True if correlation >= min_correlation threshold
    bool IsTemporallyCorrelated(PatternID p1, PatternID p2) const;

    /// Get mean delay between pattern activations
    /// @param p1 First pattern
    /// @param p2 Second pattern
    /// @return Mean delay in microseconds, or 0 if no data
    int64_t GetMeanDelay(PatternID p1, PatternID p2) const;

    /// Get all patterns that follow a given pattern
    /// @param pattern Query pattern
    /// @param min_correlation Minimum correlation threshold
    /// @return Vector of (pattern, correlation) pairs sorted by correlation
    std::vector<std::pair<PatternID, float>> GetSuccessors(
        PatternID pattern,
        float min_correlation = 0.0f
    ) const;

    /// Get all patterns that precede a given pattern
    /// @param pattern Query pattern
    /// @param min_correlation Minimum correlation threshold
    /// @return Vector of (pattern, correlation) pairs sorted by correlation
    std::vector<std::pair<PatternID, float>> GetPredecessors(
        PatternID pattern,
        float min_correlation = 0.0f
    ) const;

    // ========================================================================
    // Maintenance
    // ========================================================================

    /// Remove old activations outside tracking window
    /// @param cutoff_time Remove activations before this time
    void PruneOldActivations(Timestamp cutoff_time);

    /// Clear all tracked data
    void Clear();

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get total number of recorded activations
    size_t GetActivationCount() const { return activations_.size(); }

    /// Get number of unique patterns tracked
    size_t GetUniquePatternCount() const;

    /// Get number of pattern pairs with statistics
    size_t GetPairCount() const { return temporal_stats_.size(); }

    /// Get configuration
    const Config& GetConfig() const { return config_; }
    void SetConfig(const Config& config) { config_ = config; }

private:
    Config config_;

    // Activation history: sorted by timestamp
    std::deque<std::pair<Timestamp, PatternID>> activations_;

    // Temporal statistics for pattern pairs: (p1, p2) -> stats
    // Key format: (predecessor, successor)
    std::unordered_map<std::pair<PatternID, PatternID>, TemporalStats,
                       PatternPairHash> temporal_stats_;

    // Helper methods

    /// Update statistics for a pattern pair
    void UpdateStats(PatternID p1, PatternID p2, Timestamp t1, Timestamp t2);

    /// Compute temporal correlation from mean and stddev
    /// Formula: τ = 1 / (1 + σ/μ)
    float ComputeCorrelation(int64_t mean_micros, int64_t stddev_micros) const;

    /// Create ordered pair key for consistent lookup
    std::pair<PatternID, PatternID> MakeKey(PatternID p1, PatternID p2) const {
        return std::make_pair(p1, p2);
    }
};

} // namespace dpan
