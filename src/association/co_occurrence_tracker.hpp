// File: src/association/co_occurrence_tracker.hpp
#pragma once

#include "core/types.hpp"
#include "association/association_matrix.hpp"  // For PatternPairHash
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dpan {

/// CoOccurrenceTracker: Tracks pattern co-occurrences within temporal windows
///
/// Maintains sliding window of pattern activations and computes statistical
/// co-occurrence metrics including:
/// - Raw co-occurrence counts
/// - Co-occurrence probabilities
/// - Chi-squared significance testing
///
/// Thread-safety: Not thread-safe. External synchronization required.
class CoOccurrenceTracker {
public:
    /// Configuration for co-occurrence tracking
    struct Config {
        Config() = default;
        /// Size of temporal window for co-occurrence
        Timestamp::Duration window_size{std::chrono::seconds(10)};
        /// Minimum co-occurrences to form association
        uint32_t min_co_occurrences{3};
        /// Chi-squared p-value threshold for significance
        float significance_threshold{0.05f};
    };

    // ========================================================================
    // Construction
    // ========================================================================

    CoOccurrenceTracker();
    explicit CoOccurrenceTracker(const Config& config);
    ~CoOccurrenceTracker() = default;

    // ========================================================================
    // Recording Activations
    // ========================================================================

    /// Record single pattern activation at given time
    /// @param pattern Pattern that was activated
    /// @param timestamp Activation time (defaults to now)
    void RecordActivation(PatternID pattern, Timestamp timestamp = Timestamp::Now());

    /// Record multiple patterns activated simultaneously
    /// @param patterns Patterns activated in same window
    /// @param timestamp Activation time (defaults to now)
    void RecordActivations(const std::vector<PatternID>& patterns,
                          Timestamp timestamp = Timestamp::Now());

    // ========================================================================
    // Querying Co-occurrences
    // ========================================================================

    /// Get number of times two patterns co-occurred
    /// @return Co-occurrence count
    uint32_t GetCoOccurrenceCount(PatternID p1, PatternID p2) const;

    /// Get probability of co-occurrence: P(p1, p2)
    /// @return Probability in [0,1]
    float GetCoOccurrenceProbability(PatternID p1, PatternID p2) const;

    /// Test if co-occurrence is statistically significant (chi-squared test)
    /// @return True if chi-squared > 3.841 (p < 0.05, df=1)
    bool IsSignificant(PatternID p1, PatternID p2) const;

    /// Get chi-squared statistic for co-occurrence
    /// @return Chi-squared value
    float GetChiSquared(PatternID p1, PatternID p2) const;

    /// Get all patterns that co-occur with given pattern
    /// @param pattern Query pattern
    /// @param min_count Minimum co-occurrence count (default 0)
    /// @return Vector of (pattern, count) pairs sorted by count (descending)
    std::vector<std::pair<PatternID, uint32_t>> GetCoOccurringPatterns(
        PatternID pattern,
        uint32_t min_count = 0
    ) const;

    // ========================================================================
    // Maintenance
    // ========================================================================

    /// Remove activations older than cutoff time
    /// @param cutoff_time Timestamp before which to remove activations
    void PruneOldActivations(Timestamp cutoff_time);

    /// Clear all tracked data
    void Clear();

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get total number of recorded activations
    size_t GetActivationCount() const { return activations_.size(); }

    /// Get number of unique patterns seen
    size_t GetUniquePatternCount() const { return pattern_counts_.size(); }

    /// Get total number of windows processed
    uint64_t GetTotalWindows() const { return total_windows_; }

    /// Get number of unique co-occurring pairs
    size_t GetCoOccurrencePairCount() const { return co_occurrence_counts_.size(); }

private:
    Config config_;

    // Activation history: (timestamp, pattern) sorted by timestamp
    std::deque<std::pair<Timestamp, PatternID>> activations_;

    // Co-occurrence counts: (p1, p2) -> count (p1 < p2 always)
    std::unordered_map<std::pair<PatternID, PatternID>, uint32_t, PatternPairHash> co_occurrence_counts_;

    // Individual pattern activation counts
    std::unordered_map<PatternID, uint32_t> pattern_counts_;

    // Total number of windows processed
    uint64_t total_windows_{0};

    // Helper methods
    /// Get all patterns activated within time range
    std::vector<PatternID> GetPatternsInWindow(Timestamp start, Timestamp end) const;

    /// Update co-occurrence counts for patterns in same window
    void UpdateCoOccurrences(const std::vector<PatternID>& patterns_in_window);
};

} // namespace dpan
