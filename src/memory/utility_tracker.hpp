// File: src/memory/utility_tracker.hpp
//
// Utility Tracker for Periodic Pattern and Association Utility Calculation
//
// This module provides periodic recalculation of utility scores for all patterns
// and associations, maintaining history for trend analysis. It runs on a background
// thread and integrates with UtilityCalculator and AccessTracker.
//
// Key Features:
//   - Background thread for scheduled updates
//   - Batch processing for efficiency (>10K patterns/sec)
//   - Utility history with sliding window
//   - Trend detection (increasing/decreasing/stable)
//   - Top-K utility patterns for quick access
//   - Thread-safe concurrent operations

#pragma once

#include "memory/utility_calculator.hpp"
#include "memory/adaptive_thresholds.hpp"
#include "core/types.hpp"
#include "association/association_matrix.hpp"
#include <chrono>
#include <thread>
#include <atomic>
#include <shared_mutex>
#include <unordered_map>
#include <deque>
#include <vector>
#include <functional>

namespace dpan {

// Forward declarations
class AssociationMatrix;

/// Trend direction for utility scores
enum class UtilityTrend {
    INCREASING,   ///< Utility is going up
    DECREASING,   ///< Utility is going down
    STABLE        ///< Utility is relatively constant
};

/// Historical utility record
struct UtilityRecord {
    float utility{0.0f};           ///< Utility score at this time
    Timestamp timestamp;            ///< When this score was recorded

    UtilityRecord() = default;
    UtilityRecord(float u, Timestamp t) : utility(u), timestamp(t) {}
};

/// Utility history for a pattern/association (sliding window)
class UtilityHistory {
public:
    explicit UtilityHistory(size_t max_history_size = 100);

    /// Add a new utility record
    void AddRecord(float utility, Timestamp timestamp = Timestamp::Now());

    /// Get most recent utility score
    float GetCurrentUtility() const;

    /// Get all historical records (oldest to newest)
    std::vector<UtilityRecord> GetHistory() const;

    /// Detect trend over recent history
    /// @param window_size Number of recent records to analyze (0 = all)
    UtilityTrend DetectTrend(size_t window_size = 10) const;

    /// Get average utility over recent history
    float GetAverageUtility(size_t window_size = 10) const;

    /// Get utility change rate (delta per hour)
    float GetChangeRate() const;

    /// Get number of records
    size_t GetRecordCount() const { return records_.size(); }

    /// Clear all history
    void Clear();

private:
    size_t max_history_size_;
    std::deque<UtilityRecord> records_;  ///< Sliding window of records
    mutable std::shared_mutex mutex_;
};

/// Utility Tracker for periodic recalculation and trend analysis
class UtilityTracker {
public:
    /// Configuration for utility tracking
    struct Config {
        /// Interval between automatic updates
        std::chrono::seconds update_interval{std::chrono::seconds(60)};

        /// Maximum number of historical records to keep per pattern
        size_t max_history_size{100};

        /// Number of top-utility patterns to track
        size_t top_k_size{1000};

        /// Batch size for processing (for efficiency)
        size_t batch_size{1000};

        /// Enable automatic background updates
        bool enable_auto_update{true};

        /// Threshold for trend detection (% change to be considered trend)
        float trend_detection_threshold{0.1f};  ///< 10% change

        /// Validate configuration
        bool IsValid() const;
    };

    /// Construct with dependencies
    ///
    /// @param calculator Utility calculator to use
    /// @param access_tracker Access statistics tracker
    /// @param association_matrix Association graph
    /// @param config Configuration (optional)
    UtilityTracker(
        const UtilityCalculator& calculator,
        const AccessTracker& access_tracker,
        const AssociationMatrix& association_matrix,
        const Config& config
    );

    /// Construct with dependencies using default configuration
    ///
    /// @param calculator Utility calculator to use
    /// @param access_tracker Access statistics tracker
    /// @param association_matrix Association graph
    UtilityTracker(
        const UtilityCalculator& calculator,
        const AccessTracker& access_tracker,
        const AssociationMatrix& association_matrix
    );

    /// Destructor - stops background thread
    ~UtilityTracker();

    // Disable copy/move (has background thread)
    UtilityTracker(const UtilityTracker&) = delete;
    UtilityTracker& operator=(const UtilityTracker&) = delete;

    /// Start background updates (if enabled in config)
    void Start();

    /// Stop background updates
    void Stop();

    /// Trigger immediate update of all utilities
    ///
    /// @return Number of patterns updated
    size_t UpdateAllUtilities();

    /// Update utility for a specific pattern
    ///
    /// @param pattern Pattern to update
    /// @return Updated utility score
    float UpdatePatternUtility(PatternID pattern);

    /// Get current utility for a pattern
    ///
    /// @param pattern Pattern ID
    /// @return Current utility score (0.0 if not tracked)
    float GetPatternUtility(PatternID pattern) const;

    /// Get utility history for a pattern
    ///
    /// @param pattern Pattern ID
    /// @return Pointer to history (nullptr if not tracked)
    const UtilityHistory* GetPatternHistory(PatternID pattern) const;

    /// Get detected trend for a pattern
    ///
    /// @param pattern Pattern ID
    /// @param window_size Number of recent records to analyze
    /// @return Detected trend
    UtilityTrend GetPatternTrend(PatternID pattern, size_t window_size = 10) const;

    /// Get top-K patterns by utility score
    ///
    /// @param k Number of top patterns to return (0 = use config.top_k_size)
    /// @return Vector of (PatternID, utility) pairs, sorted descending
    std::vector<std::pair<PatternID, float>> GetTopKPatterns(size_t k = 0) const;

    /// Get patterns with increasing utility
    ///
    /// @param min_change_rate Minimum change rate (utility/hour)
    /// @return Patterns with increasing trend
    std::vector<PatternID> GetIncreasingPatterns(float min_change_rate = 0.01f) const;

    /// Get patterns with decreasing utility
    ///
    /// @param max_change_rate Maximum change rate (utility/hour)
    /// @return Patterns with decreasing trend
    std::vector<PatternID> GetDecreasingPatterns(float max_change_rate = -0.01f) const;

    /// Get number of tracked patterns
    size_t GetTrackedPatternCount() const;

    /// Clear all tracking data
    void Clear();

    /// Statistics for monitoring
    struct Statistics {
        size_t total_tracked_patterns{0};
        size_t total_updates_performed{0};
        Timestamp last_update_time;
        float average_utility{0.0f};
        float max_utility{0.0f};
        float min_utility{1.0f};
        size_t patterns_increasing{0};
        size_t patterns_decreasing{0};
        size_t patterns_stable{0};
    };

    /// Get current statistics
    Statistics GetStatistics() const;

    /// Update configuration
    void SetConfig(const Config& config);

    /// Get current configuration
    const Config& GetConfig() const { return config_; }

private:
    Config config_;

    // Dependencies (references, not owned)
    const UtilityCalculator& calculator_;
    const AccessTracker& access_tracker_;
    const AssociationMatrix& association_matrix_;

    // Tracking data
    std::unordered_map<PatternID, UtilityHistory> pattern_utilities_;
    mutable std::shared_mutex utilities_mutex_;

    // Top-K tracking (for efficient queries)
    std::vector<std::pair<PatternID, float>> top_k_patterns_;
    mutable std::shared_mutex top_k_mutex_;

    // Background thread management
    std::thread update_thread_;
    std::atomic<bool> running_{false};

    // Statistics
    std::atomic<size_t> total_updates_{0};
    Timestamp last_update_time_;

    // Private methods
    void BackgroundUpdateLoop();
    void UpdateTopK();
    float CalculateUtilityForPattern(PatternID pattern);
};

} // namespace dpan
