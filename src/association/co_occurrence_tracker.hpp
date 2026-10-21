// File: src/association/co_occurrence_tracker.hpp
#pragma once

#include "core/types.hpp"
#include "association/association_matrix.hpp"  // For PatternPairHash
#include <deque>
#include <shared_mutex>
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
/// Thread-safety: All public methods are thread-safe (internal shared_mutex).
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
    /// Counts as one observation window: co-occurrences with every other
    /// pattern activated in [timestamp - window_size, timestamp]
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
    size_t GetActivationCount() const;

    /// Get number of unique patterns seen
    size_t GetUniquePatternCount() const;

    /// Get total number of windows processed
    uint64_t GetTotalWindows() const;

    /// Get number of unique co-occurring pairs
    size_t GetCoOccurrencePairCount() const;

    /// Get all tracked patterns (patterns that have been activated)
    /// @return Vector of pattern IDs that have been recorded
    std::vector<PatternID> GetTrackedPatterns() const;

private:
    Config config_;

    // Activation history: (timestamp, pattern) sorted by timestamp
    std::deque<std::pair<Timestamp, PatternID>> activations_;

    /// Open-addressing map (p1, p2) -> count with p1 < p2. Each activation
    /// updates one count per pattern in its window, so this is the hot path;
    /// a flat table avoids the pointer chasing of std::unordered_map.
    class PairCounts {
    public:
        void Increment(PatternID p1, PatternID p2);
        uint32_t Get(PatternID p1, PatternID p2) const;
        size_t Size() const { return size_; }
        void Clear();

        template <typename Fn>
        void ForEach(Fn fn) const {
            for (const auto& slot : slots_) {
                if (slot.first != 0) {
                    fn(PatternID(slot.first), PatternID(slot.second), slot.count);
                }
            }
        }

    private:
        struct Slot {
            uint64_t first{0};   // 0 = empty (PatternID 0 is invalid)
            uint64_t second{0};
            uint32_t count{0};
        };
        std::vector<Slot> slots_;
        size_t size_{0};

        size_t FindSlot(uint64_t first, uint64_t second) const;
        void Grow();
    };

    // Co-occurrence counts: (p1, p2) -> count (p1 < p2 always)
    PairCounts co_occurrence_counts_;

    // Individual pattern activation counts
    std::unordered_map<PatternID, uint32_t> pattern_counts_;

    // Total number of windows processed
    uint64_t total_windows_{0};

    // Sliding window [latest_timestamp_ - window_size, latest_timestamp_]:
    // activations_[window_begin_..] lie inside it, and window_counts_ holds
    // the number of those activations per pattern (no zero entries).
    std::unordered_map<PatternID, uint32_t> window_counts_;
    size_t window_begin_{0};
    Timestamp latest_timestamp_{};
    bool has_latest_{false};

    mutable std::shared_mutex mutex_;

    // Helper methods (caller must hold mutex_)
    /// Get all patterns activated within time range
    std::vector<PatternID> GetPatternsInWindow(Timestamp start, Timestamp end) const;

    /// Update co-occurrence counts for patterns in same window
    void UpdateCoOccurrences(const std::vector<PatternID>& patterns_in_window);

    /// Increment the co-occurrence count of an unordered pair
    void IncrementPair(PatternID p1, PatternID p2);

    /// Insert an activation keeping activations_ sorted and the window current
    void InsertActivation(PatternID pattern, Timestamp timestamp);

    /// Drop activations that fell out of the sliding window from window_counts_
    void AdvanceWindow();

    void DecrementWindowCount(PatternID pattern);

    uint32_t GetCoOccurrenceCountUnlocked(PatternID p1, PatternID p2) const;
    float GetChiSquaredUnlocked(PatternID p1, PatternID p2) const;
    void PruneOldActivationsUnlocked(Timestamp cutoff_time);
};

} // namespace dpan
