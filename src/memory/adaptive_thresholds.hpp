// File: src/memory/adaptive_thresholds.hpp
//
// Adaptive Threshold Management for Memory Pressure
//
// This module implements dynamic threshold adjustment based on memory pressure,
// allowing the system to automatically tighten or relax pruning criteria as
// memory availability changes.
//
// Mathematical Foundation:
//   T(p) = T_base × (1 + pressure_factor × P)
//
// Where:
//   P = (M_used - M_target) / M_target  (memory pressure)
//   T_base = baseline threshold
//   pressure_factor = sensitivity to pressure changes

#pragma once

#include "core/types.hpp"
#include <vector>
#include <algorithm>

namespace dpan {

/// Adaptive threshold management based on memory pressure
///
/// Automatically adjusts utility thresholds based on current memory usage
/// relative to target limits, enabling graceful degradation under pressure.
class AdaptiveThresholdManager {
public:
    /// Configuration for adaptive threshold behavior
    struct Config {
        // Baseline threshold settings
        float baseline_threshold{0.3f};    ///< Base utility threshold when at target memory
        float pressure_factor{2.0f};       ///< How aggressively pressure affects threshold
        float min_threshold{0.1f};         ///< Minimum threshold (most lenient)
        float max_threshold{0.9f};         ///< Maximum threshold (most strict)

        // Memory pressure targets
        size_t target_memory_bytes{8ULL * 1024 * 1024 * 1024};  ///< Target: 8GB
        float pressure_update_interval{60.0f};  ///< Update frequency (seconds)

        // Percentile-based thresholds
        bool use_percentile{false};         ///< Use percentile instead of pressure
        float target_eviction_rate{0.2f};   ///< Target eviction rate (20%)

        // Smoothing
        float smoothing_factor{0.3f};       ///< EMA smoothing for threshold changes

        /// Validate configuration
        bool IsValid() const;
    };

    /// Construct with default configuration
    AdaptiveThresholdManager();

    /// Construct with custom configuration
    explicit AdaptiveThresholdManager(const Config& config);

    /// Update threshold based on current memory usage
    ///
    /// @param current_memory_bytes Current memory consumption
    /// @param pattern_count Current number of patterns (optional, for stats)
    void UpdateThreshold(size_t current_memory_bytes, size_t pattern_count = 0);

    /// Update threshold based on utility distribution (percentile mode)
    ///
    /// @param utilities Vector of utility scores for all patterns
    void UpdateThresholdFromUtilities(const std::vector<float>& utilities);

    /// Get current threshold for pruning decisions
    ///
    /// @return Current utility threshold
    float GetCurrentThreshold() const { return current_threshold_; }

    /// Compute threshold from utility distribution (percentile-based)
    ///
    /// @param utilities Utility scores to analyze
    /// @return Percentile-based threshold
    float ComputePercentileThreshold(const std::vector<float>& utilities) const;

    /// Calculate memory pressure
    ///
    /// @param current_bytes Current memory usage
    /// @return Pressure value (negative = under-utilized, positive = over-utilized)
    float ComputeMemoryPressure(size_t current_bytes) const;

    /// Set new configuration
    void SetConfig(const Config& config);

    /// Get current configuration
    const Config& GetConfig() const { return config_; }

    /// Statistics about threshold adaptation
    struct ThresholdStats {
        float current_threshold{0.0f};      ///< Current active threshold
        float memory_pressure{0.0f};        ///< Current memory pressure
        float baseline_threshold{0.0f};     ///< Configured baseline
        size_t current_memory_bytes{0};     ///< Last observed memory usage
        size_t target_memory_bytes{0};      ///< Target memory limit
        Timestamp last_update;              ///< When last updated
        size_t pattern_count{0};            ///< Number of patterns tracked
    };

    /// Get current statistics
    ThresholdStats GetStats() const;

    /// Reset to baseline
    void Reset();

private:
    Config config_;
    float current_threshold_;
    float current_pressure_;
    Timestamp last_update_;
    size_t last_memory_bytes_;
    size_t last_pattern_count_;

    /// Apply exponential moving average smoothing to threshold changes
    ///
    /// @param new_threshold New calculated threshold
    void SmoothThresholdUpdate(float new_threshold);

    /// Validate and clamp threshold to valid range
    float ClampThreshold(float threshold) const;
};

} // namespace dpan
