// File: src/memory/sleep_consolidator.cpp
#include "memory/sleep_consolidator.hpp"
#include <algorithm>
#include <stdexcept>

namespace dpan {

SleepConsolidator::SleepConsolidator()
    : config_(), last_measurement_time_(Timestamp::Now()),
      state_entered_time_(Timestamp::Now()), last_consolidation_time_(Timestamp::Now()) {
}

SleepConsolidator::SleepConsolidator(const Config& config)
    : config_(config), last_measurement_time_(Timestamp::Now()),
      state_entered_time_(Timestamp::Now()), last_consolidation_time_(Timestamp::Now()) {
    if (!config_.IsValid()) {
        throw std::invalid_argument("Invalid SleepConsolidator configuration");
    }
}

void SleepConsolidator::SetConfig(const Config& config) {
    if (!config.IsValid()) {
        throw std::invalid_argument("Invalid SleepConsolidator configuration");
    }
    config_ = config;
}

void SleepConsolidator::RecordOperation() {
    operations_since_last_measurement_.fetch_add(1, std::memory_order_relaxed);
}

void SleepConsolidator::RecordOperations(size_t count) {
    operations_since_last_measurement_.fetch_add(count, std::memory_order_relaxed);
}

bool SleepConsolidator::UpdateActivityState() {
    UpdateActivityHistory();

    float activity_rate = ComputeActivityRate();
    ActivityState old_state = current_state_.load(std::memory_order_relaxed);
    ActivityState new_state = old_state;

    // State machine transitions
    switch (old_state) {
        case ActivityState::ACTIVE:
            if (activity_rate < config_.low_activity_threshold) {
                new_state = ActivityState::LOW_ACTIVITY;
            }
            break;

        case ActivityState::LOW_ACTIVITY: {
            if (activity_rate >= config_.low_activity_threshold) {
                // Activity increased, return to active
                new_state = ActivityState::ACTIVE;
            } else {
                // Check if low activity sustained long enough for sleep
                auto time_in_low = Timestamp::Now() - state_entered_time_;
                if (time_in_low >= config_.min_sleep_duration) {
                    new_state = ActivityState::SLEEP;
                }
            }
            break;
        }

        case ActivityState::SLEEP:
            if (activity_rate >= config_.low_activity_threshold) {
                // Waking up from sleep
                new_state = ActivityState::ACTIVE;

                // Update statistics
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.total_sleep_periods++;
                auto sleep_duration = std::chrono::duration_cast<std::chrono::seconds>(
                    Timestamp::Now() - state_entered_time_
                );
                stats_.total_sleep_time += sleep_duration;
            }
            break;
    }

    if (new_state != old_state) {
        TransitionToState(new_state);
        return true;
    }

    return false;
}

SleepConsolidator::ActivityState SleepConsolidator::GetActivityState() const {
    return current_state_.load(std::memory_order_relaxed);
}

float SleepConsolidator::GetActivityRate() const {
    return ComputeActivityRate();
}

bool SleepConsolidator::IsInSleepState() const {
    return current_state_.load(std::memory_order_relaxed) == ActivityState::SLEEP;
}

bool SleepConsolidator::ShouldTriggerConsolidation() const {
    if (!config_.enable_automatic_consolidation) {
        return false;
    }

    if (!IsInSleepState()) {
        return false;
    }

    // Check if enough time has passed since last consolidation
    auto time_since_last = Timestamp::Now() - last_consolidation_time_;
    return time_since_last >= config_.consolidation_interval;
}

SleepConsolidator::ConsolidationCycleResult SleepConsolidator::TriggerConsolidation() {
    ConsolidationCycleResult result;
    result.start_time = Timestamp::Now();
    result.was_successful = true;

    try {
        // Note: In a real implementation, this would:
        // 1. Call MemoryConsolidator for pattern merging
        // 2. Call PatternPruner for aggressive pruning
        // 3. Call utility strengthening for important patterns
        //
        // For now, we simulate the process

        // Simulated consolidation work
        result.patterns_strengthened = config_.top_patterns_to_strengthen;
        result.patterns_pruned = 0;  // Would be determined by actual pruning
        result.associations_reorganized = 0;  // Would be determined by actual consolidation
        result.average_utility_change = config_.strengthening_factor;
        result.memory_freed_bytes = 0;  // Would be calculated from actual operations

    } catch (const std::exception& e) {
        result.was_successful = false;
        result.error_message = e.what();
    }

    result.end_time = Timestamp::Now();
    result.duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        result.end_time - result.start_time
    );

    // Update last consolidation time
    last_consolidation_time_ = result.end_time;

    // Update statistics
    UpdateStatistics(result);

    return result;
}

std::vector<SleepConsolidator::StrengtheningInfo>
SleepConsolidator::IdentifyPatternsToStrengthen(
    const std::unordered_map<PatternID, float>& pattern_utilities
) const {
    std::vector<StrengtheningInfo> patterns_to_strengthen;

    // Create list of candidates (patterns above min utility threshold)
    std::vector<std::pair<PatternID, float>> candidates;
    for (const auto& [pattern_id, utility] : pattern_utilities) {
        if (utility >= config_.min_utility_for_strengthening) {
            candidates.emplace_back(pattern_id, utility);
        }
    }

    // Sort by utility (descending)
    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        }
    );

    // Take top N patterns
    size_t num_to_strengthen = std::min(
        candidates.size(),
        config_.top_patterns_to_strengthen
    );

    for (size_t i = 0; i < num_to_strengthen; ++i) {
        const auto& [pattern_id, current_utility] = candidates[i];

        StrengtheningInfo info;
        info.pattern_id = pattern_id;
        info.current_utility = current_utility;
        info.boost_amount = CalculateStrengtheningBoost(current_utility);
        info.new_utility = std::min(1.0f, current_utility + info.boost_amount);

        patterns_to_strengthen.push_back(info);
    }

    return patterns_to_strengthen;
}

float SleepConsolidator::CalculateStrengtheningBoost(float current_utility) const {
    // Boost = strengthening_factor × (1 - current_utility)
    // This gives more boost to patterns that have room to grow
    float room_to_grow = 1.0f - current_utility;
    return config_.strengthening_factor * room_to_grow;
}

void SleepConsolidator::EnterSleepState() {
    TransitionToState(ActivityState::SLEEP);
}

void SleepConsolidator::WakeFromSleep() {
    if (current_state_.load(std::memory_order_relaxed) == ActivityState::SLEEP) {
        TransitionToState(ActivityState::ACTIVE);

        // Update sleep statistics
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_sleep_periods++;
        auto sleep_duration = std::chrono::duration_cast<std::chrono::seconds>(
            Timestamp::Now() - state_entered_time_
        );
        stats_.total_sleep_time += sleep_duration;
    }
}

SleepConsolidator::Statistics SleepConsolidator::GetStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    Statistics stats_copy = stats_;

    // Update current state info
    stats_copy.current_activity_rate = ComputeActivityRate();
    stats_copy.current_state = current_state_.load(std::memory_order_relaxed);
    stats_copy.state_entered_time = state_entered_time_;

    return stats_copy;
}

void SleepConsolidator::ResetStatistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = Statistics{};
}

std::vector<SleepConsolidator::ActivityMeasurement>
SleepConsolidator::GetActivityHistory(size_t count) const {
    std::lock_guard<std::mutex> lock(activity_mutex_);

    size_t start_idx = 0;
    if (activity_history_.size() > count) {
        start_idx = activity_history_.size() - count;
    }

    return std::vector<ActivityMeasurement>(
        activity_history_.begin() + start_idx,
        activity_history_.end()
    );
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void SleepConsolidator::UpdateActivityHistory() {
    std::lock_guard<std::mutex> lock(activity_mutex_);

    Timestamp now = Timestamp::Now();

    // Create measurement
    ActivityMeasurement measurement;
    measurement.timestamp = now;
    measurement.operations_count = operations_since_last_measurement_.exchange(
        0,
        std::memory_order_relaxed
    );
    measurement.state = current_state_.load(std::memory_order_relaxed);

    activity_history_.push_back(measurement);

    // Keep history bounded (last 1000 measurements)
    if (activity_history_.size() > 1000) {
        activity_history_.erase(activity_history_.begin());
    }

    last_measurement_time_ = now;
}

float SleepConsolidator::ComputeActivityRate() const {
    std::lock_guard<std::mutex> lock(activity_mutex_);

    if (activity_history_.empty()) {
        return 0.0f;
    }

    // Look at measurements within activity window
    Timestamp cutoff = Timestamp::Now() - config_.activity_window;

    size_t total_operations = 0;
    Timestamp earliest_time = Timestamp::Now();

    for (const auto& measurement : activity_history_) {
        if (measurement.timestamp >= cutoff) {
            total_operations += measurement.operations_count;
            if (measurement.timestamp < earliest_time) {
                earliest_time = measurement.timestamp;
            }
        }
    }

    // Calculate rate (operations per second)
    auto duration = Timestamp::Now() - earliest_time;
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();

    if (seconds == 0) {
        return 0.0f;
    }

    return static_cast<float>(total_operations) / seconds;
}

void SleepConsolidator::TransitionToState(ActivityState new_state) {
    current_state_.store(new_state, std::memory_order_relaxed);
    state_entered_time_ = Timestamp::Now();
}

void SleepConsolidator::UpdateStatistics(const ConsolidationCycleResult& result) {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    stats_.total_consolidation_cycles++;
    stats_.total_patterns_strengthened += result.patterns_strengthened;
    stats_.total_patterns_pruned += result.patterns_pruned;

    // Update average cycle duration
    float total_duration = stats_.average_cycle_duration_ms * (stats_.total_consolidation_cycles - 1) +
                          result.duration.count();
    stats_.average_cycle_duration_ms = total_duration / stats_.total_consolidation_cycles;

    stats_.last_consolidation_time = result.end_time;
}

} // namespace dpan
