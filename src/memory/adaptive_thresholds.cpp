// File: src/memory/adaptive_thresholds.cpp
//
// Implementation of Adaptive Threshold Management

#include "memory/adaptive_thresholds.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace dpan {

// ============================================================================
// AdaptiveThresholdManager::Config
// ============================================================================

bool AdaptiveThresholdManager::Config::IsValid() const {
    // Threshold bounds must be valid
    if (min_threshold < 0.0f || min_threshold > 1.0f) {
        return false;
    }
    if (max_threshold < 0.0f || max_threshold > 1.0f) {
        return false;
    }
    if (min_threshold > max_threshold) {
        return false;
    }

    // Baseline must be within bounds
    if (baseline_threshold < min_threshold || baseline_threshold > max_threshold) {
        return false;
    }

    // Pressure factor must be positive
    if (pressure_factor <= 0.0f) {
        return false;
    }

    // Target memory must be positive
    if (target_memory_bytes == 0) {
        return false;
    }

    // Update interval must be positive
    if (pressure_update_interval <= 0.0f) {
        return false;
    }

    // Eviction rate must be in [0, 1]
    if (target_eviction_rate < 0.0f || target_eviction_rate > 1.0f) {
        return false;
    }

    // Smoothing factor must be in [0, 1]
    if (smoothing_factor < 0.0f || smoothing_factor > 1.0f) {
        return false;
    }

    return true;
}

// ============================================================================
// AdaptiveThresholdManager
// ============================================================================

AdaptiveThresholdManager::AdaptiveThresholdManager()
    : config_(Config{}),
      current_threshold_(config_.baseline_threshold),
      current_pressure_(0.0f),
      last_update_(Timestamp::Now()),
      last_memory_bytes_(0),
      last_pattern_count_(0) {
}

AdaptiveThresholdManager::AdaptiveThresholdManager(const Config& config)
    : config_(config),
      current_threshold_(config.baseline_threshold),
      current_pressure_(0.0f),
      last_update_(Timestamp::Now()),
      last_memory_bytes_(0),
      last_pattern_count_(0) {

    if (!config_.IsValid()) {
        throw std::invalid_argument("Invalid AdaptiveThresholdManager configuration");
    }
}

void AdaptiveThresholdManager::UpdateThreshold(
    size_t current_memory_bytes,
    size_t pattern_count) {

    // Check if enough time has passed since last update
    auto elapsed = Timestamp::Now() - last_update_;
    auto elapsed_secs = std::chrono::duration<float>(elapsed).count();

    if (elapsed_secs < config_.pressure_update_interval) {
        // Update tracking but don't recalculate threshold yet
        last_memory_bytes_ = current_memory_bytes;
        last_pattern_count_ = pattern_count;
        return;
    }

    // Compute current memory pressure
    current_pressure_ = ComputeMemoryPressure(current_memory_bytes);

    // Calculate new threshold based on pressure
    // T(p) = T_base × (1 + pressure_factor × P)
    float pressure_adjusted_threshold =
        config_.baseline_threshold * (1.0f + config_.pressure_factor * current_pressure_);

    // Clamp to valid range
    pressure_adjusted_threshold = ClampThreshold(pressure_adjusted_threshold);

    // Apply smoothing to avoid rapid oscillations
    SmoothThresholdUpdate(pressure_adjusted_threshold);

    // Update tracking
    last_update_ = Timestamp::Now();
    last_memory_bytes_ = current_memory_bytes;
    last_pattern_count_ = pattern_count;
}

void AdaptiveThresholdManager::UpdateThresholdFromUtilities(
    const std::vector<float>& utilities) {

    if (!config_.use_percentile) {
        // Not in percentile mode, ignore
        return;
    }

    float percentile_threshold = ComputePercentileThreshold(utilities);

    // Clamp to valid range
    percentile_threshold = ClampThreshold(percentile_threshold);

    // Apply smoothing
    SmoothThresholdUpdate(percentile_threshold);

    // Update timestamp
    last_update_ = Timestamp::Now();
}

float AdaptiveThresholdManager::ComputeMemoryPressure(size_t current_bytes) const {
    // P = (M_used - M_target) / M_target

    if (current_bytes <= config_.target_memory_bytes) {
        // Under-utilized: negative pressure (lenient)
        float ratio = static_cast<float>(current_bytes) / static_cast<float>(config_.target_memory_bytes);
        return ratio - 1.0f;  // Will be in [-1, 0]
    } else {
        // Over-utilized: positive pressure (strict)
        float excess = static_cast<float>(current_bytes - config_.target_memory_bytes);
        return excess / static_cast<float>(config_.target_memory_bytes);  // Will be > 0
    }
}

float AdaptiveThresholdManager::ComputePercentileThreshold(
    const std::vector<float>& utilities) const {

    if (utilities.empty()) {
        return config_.baseline_threshold;
    }

    // Create sorted copy
    std::vector<float> sorted_utilities = utilities;
    std::sort(sorted_utilities.begin(), sorted_utilities.end());

    // Find k-th percentile where k = target_eviction_rate
    // To evict bottom 20%, we want the 20th percentile
    size_t index = static_cast<size_t>(config_.target_eviction_rate * sorted_utilities.size());
    index = std::min(index, sorted_utilities.size() - 1);

    return sorted_utilities[index];
}

void AdaptiveThresholdManager::SetConfig(const Config& config) {
    if (!config.IsValid()) {
        throw std::invalid_argument("Invalid AdaptiveThresholdManager configuration");
    }

    config_ = config;

    // Reset to baseline with new config
    current_threshold_ = config_.baseline_threshold;
    current_pressure_ = 0.0f;
}

AdaptiveThresholdManager::ThresholdStats AdaptiveThresholdManager::GetStats() const {
    ThresholdStats stats;
    stats.current_threshold = current_threshold_;
    stats.memory_pressure = current_pressure_;
    stats.baseline_threshold = config_.baseline_threshold;
    stats.current_memory_bytes = last_memory_bytes_;
    stats.target_memory_bytes = config_.target_memory_bytes;
    stats.last_update = last_update_;
    stats.pattern_count = last_pattern_count_;
    return stats;
}

void AdaptiveThresholdManager::Reset() {
    current_threshold_ = config_.baseline_threshold;
    current_pressure_ = 0.0f;
    last_update_ = Timestamp::Now();
    last_memory_bytes_ = 0;
    last_pattern_count_ = 0;
}

// ============================================================================
// Private Methods
// ============================================================================

void AdaptiveThresholdManager::SmoothThresholdUpdate(float new_threshold) {
    // Exponential moving average (EMA)
    // T_new = alpha * new_value + (1 - alpha) * old_value
    float alpha = config_.smoothing_factor;
    current_threshold_ = alpha * new_threshold + (1.0f - alpha) * current_threshold_;

    // Ensure it's still in valid range after smoothing
    current_threshold_ = ClampThreshold(current_threshold_);
}

float AdaptiveThresholdManager::ClampThreshold(float threshold) const {
    return std::clamp(threshold, config_.min_threshold, config_.max_threshold);
}

} // namespace dpan
