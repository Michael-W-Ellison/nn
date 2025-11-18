// File: src/memory/sleep_consolidator.hpp
#pragma once

#include "core/types.hpp"
#include <chrono>
#include <vector>
#include <atomic>
#include <mutex>

namespace dpan {

/**
 * @brief Manages sleep-like memory consolidation during low-activity periods
 *
 * Inspired by sleep consolidation in biological systems, this component:
 * - Monitors system activity levels
 * - Detects low-activity ("sleep") periods
 * - Strengthens important patterns during consolidation
 * - Triggers memory reorganization
 *
 * This simulates how memories are consolidated during sleep in biological brains.
 */
class SleepConsolidator {
public:
    /**
     * @brief Configuration for sleep consolidation
     */
    struct Config {
        // Activity monitoring
        std::chrono::seconds activity_window{60};          // Window for activity tracking
        float low_activity_threshold{0.1f};                 // Activity rate threshold for sleep
        std::chrono::seconds min_sleep_duration{30};       // Minimum time in sleep state

        // Consolidation triggers
        bool enable_automatic_consolidation{true};          // Auto-trigger during sleep
        std::chrono::seconds consolidation_interval{300};  // Time between consolidations

        // Pattern strengthening
        float strengthening_factor{0.1f};                   // How much to boost important patterns
        size_t top_patterns_to_strengthen{100};            // Number of top patterns to strengthen
        float min_utility_for_strengthening{0.6f};         // Min utility to be strengthened

        // Pruning aggressiveness
        float sleep_pruning_multiplier{1.5f};              // More aggressive pruning during sleep

        /**
         * @brief Validate configuration
         */
        bool IsValid() const {
            return activity_window.count() > 0 &&
                   low_activity_threshold >= 0.0f && low_activity_threshold <= 1.0f &&
                   min_sleep_duration.count() > 0 &&
                   consolidation_interval.count() > 0 &&
                   strengthening_factor >= 0.0f && strengthening_factor <= 1.0f &&
                   top_patterns_to_strengthen > 0 &&
                   min_utility_for_strengthening >= 0.0f && min_utility_for_strengthening <= 1.0f &&
                   sleep_pruning_multiplier >= 1.0f;
        }
    };

    /**
     * @brief System activity state
     */
    enum class ActivityState {
        ACTIVE,        // Normal activity level
        LOW_ACTIVITY,  // Activity dropping
        SLEEP,         // In consolidation sleep state
    };

    /**
     * @brief Activity measurement data point
     */
    struct ActivityMeasurement {
        Timestamp timestamp;
        size_t operations_count;      // Number of operations in this measurement
        ActivityState state;
    };

    /**
     * @brief Information about patterns to strengthen
     */
    struct StrengtheningInfo {
        PatternID pattern_id;
        float current_utility;
        float boost_amount;
        float new_utility;
    };

    /**
     * @brief Result of a consolidation cycle
     */
    struct ConsolidationCycleResult {
        Timestamp start_time;
        Timestamp end_time;
        std::chrono::milliseconds duration;

        size_t patterns_strengthened;
        size_t patterns_pruned;
        size_t associations_reorganized;

        float average_utility_change;
        size_t memory_freed_bytes;

        bool was_successful;
        std::string error_message;
    };

    /**
     * @brief Statistics about sleep consolidation
     */
    struct Statistics {
        size_t total_consolidation_cycles{0};
        size_t total_sleep_periods{0};
        std::chrono::seconds total_sleep_time{0};

        size_t total_patterns_strengthened{0};
        size_t total_patterns_pruned{0};

        float average_cycle_duration_ms{0.0f};
        Timestamp last_consolidation_time;

        // Activity tracking
        float current_activity_rate{0.0f};
        ActivityState current_state{ActivityState::ACTIVE};
        Timestamp state_entered_time;
    };

    /**
     * @brief Construct with default configuration
     */
    SleepConsolidator();

    /**
     * @brief Construct with custom configuration
     */
    explicit SleepConsolidator(const Config& config);

    /**
     * @brief Record an operation (for activity monitoring)
     *
     * Call this whenever a significant operation occurs (pattern access,
     * association lookup, etc.) to track system activity.
     */
    void RecordOperation();

    /**
     * @brief Record multiple operations at once
     */
    void RecordOperations(size_t count);

    /**
     * @brief Update activity state based on recent operations
     *
     * Call periodically to update the activity state machine.
     * Returns true if state changed.
     */
    bool UpdateActivityState();

    /**
     * @brief Get current activity state
     */
    ActivityState GetActivityState() const;

    /**
     * @brief Get current activity rate (operations per second)
     */
    float GetActivityRate() const;

    /**
     * @brief Check if system is in sleep state
     */
    bool IsInSleepState() const;

    /**
     * @brief Check if consolidation should be triggered
     *
     * Returns true if:
     * - In sleep state
     * - Sufficient time since last consolidation
     * - Automatic consolidation enabled
     */
    bool ShouldTriggerConsolidation() const;

    /**
     * @brief Manually trigger a consolidation cycle
     *
     * This bypasses the automatic triggering logic and runs
     * consolidation immediately.
     *
     * @return Result of consolidation cycle
     */
    ConsolidationCycleResult TriggerConsolidation();

    /**
     * @brief Identify patterns that should be strengthened
     *
     * Selects important patterns based on:
     * - High utility scores
     * - Strong associations
     * - Recent access patterns
     *
     * @param pattern_utilities Map of pattern utilities
     * @return List of patterns to strengthen
     */
    std::vector<StrengtheningInfo> IdentifyPatternsToStrengthen(
        const std::unordered_map<PatternID, float>& pattern_utilities
    ) const;

    /**
     * @brief Calculate strengthening amount for a pattern
     *
     * Boost = strengthening_factor × (1 - current_utility)
     *
     * This gives more boost to medium-utility patterns that can benefit,
     * while not over-boosting already-strong patterns.
     *
     * @param current_utility Current utility value [0.0, 1.0]
     * @return Boost amount to add to utility
     */
    float CalculateStrengtheningBoost(float current_utility) const;

    /**
     * @brief Force system into sleep state
     *
     * Useful for testing or manual control
     */
    void EnterSleepState();

    /**
     * @brief Wake system from sleep state
     */
    void WakeFromSleep();

    /**
     * @brief Get configuration
     */
    const Config& GetConfig() const { return config_; }

    /**
     * @brief Set configuration
     */
    void SetConfig(const Config& config);

    /**
     * @brief Get statistics
     */
    Statistics GetStatistics() const;

    /**
     * @brief Reset statistics
     */
    void ResetStatistics();

    /**
     * @brief Get activity history
     *
     * @param count Number of recent measurements to return
     * @return Recent activity measurements
     */
    std::vector<ActivityMeasurement> GetActivityHistory(size_t count = 100) const;

private:
    Config config_;

    // Activity tracking
    mutable std::mutex activity_mutex_;
    std::vector<ActivityMeasurement> activity_history_;
    std::atomic<size_t> operations_since_last_measurement_{0};
    Timestamp last_measurement_time_;

    // State tracking
    std::atomic<ActivityState> current_state_{ActivityState::ACTIVE};
    Timestamp state_entered_time_;
    Timestamp last_consolidation_time_;

    // Statistics
    mutable std::mutex stats_mutex_;
    Statistics stats_;

    // Helper methods
    void UpdateActivityHistory();
    float ComputeActivityRate() const;
    void TransitionToState(ActivityState new_state);
    void UpdateStatistics(const ConsolidationCycleResult& result);
};

} // namespace dpan
