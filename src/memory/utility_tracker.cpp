// File: src/memory/utility_tracker.cpp
//
// Implementation of Utility Tracker

#include "memory/utility_tracker.hpp"
#include "core/pattern_data.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace dpan {

// ============================================================================
// UtilityHistory
// ============================================================================

UtilityHistory::UtilityHistory(size_t max_history_size)
    : max_history_size_(max_history_size) {
}

void UtilityHistory::AddRecord(float utility, Timestamp timestamp) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    records_.push_back(UtilityRecord{utility, timestamp});

    // Maintain sliding window
    if (records_.size() > max_history_size_) {
        records_.pop_front();
    }
}

float UtilityHistory::GetCurrentUtility() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    if (records_.empty()) {
        return 0.0f;
    }

    return records_.back().utility;
}

std::vector<UtilityRecord> UtilityHistory::GetHistory() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return std::vector<UtilityRecord>(records_.begin(), records_.end());
}

UtilityTrend UtilityHistory::DetectTrend(size_t window_size) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    if (records_.size() < 2) {
        return UtilityTrend::STABLE;
    }

    // Use specified window or all records
    size_t n = (window_size == 0 || window_size > records_.size())
               ? records_.size()
               : window_size;

    if (n < 2) {
        return UtilityTrend::STABLE;
    }

    // Calculate linear regression slope
    // y = mx + b, where m = slope indicates trend
    std::vector<float> x_values(n);
    std::vector<float> y_values(n);

    auto start_it = records_.end() - n;
    for (size_t i = 0; i < n; ++i) {
        x_values[i] = static_cast<float>(i);
        y_values[i] = (start_it + i)->utility;
    }

    // Calculate means
    float x_mean = std::accumulate(x_values.begin(), x_values.end(), 0.0f) / n;
    float y_mean = std::accumulate(y_values.begin(), y_values.end(), 0.0f) / n;

    // Calculate slope: m = Σ((x - x̄)(y - ȳ)) / Σ((x - x̄)²)
    float numerator = 0.0f;
    float denominator = 0.0f;

    for (size_t i = 0; i < n; ++i) {
        float x_diff = x_values[i] - x_mean;
        float y_diff = y_values[i] - y_mean;
        numerator += x_diff * y_diff;
        denominator += x_diff * x_diff;
    }

    if (std::abs(denominator) < 1e-6f) {
        return UtilityTrend::STABLE;
    }

    float slope = numerator / denominator;

    // Classify based on slope magnitude
    // Threshold: 0.01 per sample (configurable via config in tracker)
    const float threshold = 0.01f;

    if (slope > threshold) {
        return UtilityTrend::INCREASING;
    } else if (slope < -threshold) {
        return UtilityTrend::DECREASING;
    } else {
        return UtilityTrend::STABLE;
    }
}

float UtilityHistory::GetAverageUtility(size_t window_size) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    if (records_.empty()) {
        return 0.0f;
    }

    size_t n = (window_size == 0 || window_size > records_.size())
               ? records_.size()
               : window_size;

    auto start_it = records_.end() - n;
    float sum = 0.0f;

    for (auto it = start_it; it != records_.end(); ++it) {
        sum += it->utility;
    }

    return sum / static_cast<float>(n);
}

float UtilityHistory::GetChangeRate() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    if (records_.size() < 2) {
        return 0.0f;
    }

    // Calculate change rate between first and last record
    const auto& first = records_.front();
    const auto& last = records_.back();

    float utility_change = last.utility - first.utility;
    auto time_diff = last.timestamp - first.timestamp;
    auto hours = std::chrono::duration_cast<std::chrono::hours>(time_diff);

    if (hours.count() == 0) {
        return 0.0f;
    }

    return utility_change / static_cast<float>(hours.count());
}

void UtilityHistory::Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    records_.clear();
}

// ============================================================================
// UtilityTracker::Config
// ============================================================================

bool UtilityTracker::Config::IsValid() const {
    if (update_interval.count() <= 0) {
        return false;
    }

    if (max_history_size == 0 || max_history_size > 10000) {
        return false;
    }

    if (top_k_size == 0 || top_k_size > 1000000) {
        return false;
    }

    if (batch_size == 0 || batch_size > 100000) {
        return false;
    }

    if (trend_detection_threshold < 0.0f || trend_detection_threshold > 1.0f) {
        return false;
    }

    return true;
}

// ============================================================================
// UtilityTracker
// ============================================================================

UtilityTracker::UtilityTracker(
    const UtilityCalculator& calculator,
    const AccessTracker& access_tracker,
    const AssociationMatrix& association_matrix,
    const Config& config)
    : config_(config),
      calculator_(calculator),
      access_tracker_(access_tracker),
      association_matrix_(association_matrix),
      last_update_time_(Timestamp::Now()) {

    if (!config_.IsValid()) {
        throw std::invalid_argument("Invalid UtilityTracker configuration");
    }
}

UtilityTracker::UtilityTracker(
    const UtilityCalculator& calculator,
    const AccessTracker& access_tracker,
    const AssociationMatrix& association_matrix)
    : UtilityTracker(calculator, access_tracker, association_matrix, Config{}) {
}

UtilityTracker::~UtilityTracker() {
    Stop();
}

void UtilityTracker::Start() {
    if (running_.load()) {
        return;  // Already running
    }

    if (!config_.enable_auto_update) {
        return;  // Auto-update disabled
    }

    running_.store(true);
    update_thread_ = std::thread(&UtilityTracker::BackgroundUpdateLoop, this);
}

void UtilityTracker::Stop() {
    if (!running_.load()) {
        return;  // Not running
    }

    running_.store(false);

    if (update_thread_.joinable()) {
        update_thread_.join();
    }
}

size_t UtilityTracker::UpdateAllUtilities() {
    // Get all tracked pattern IDs from AccessTracker
    std::vector<PatternID> patterns_to_update;

    // Since AccessTracker doesn't expose GetAllPatterns(), we update
    // patterns we're already tracking and any new ones encountered
    {
        std::shared_lock<std::shared_mutex> lock(utilities_mutex_);
        patterns_to_update.reserve(pattern_utilities_.size());
        for (const auto& [pattern_id, _] : pattern_utilities_) {
            patterns_to_update.push_back(pattern_id);
        }
    }

    // Batch process patterns
    size_t updated_count = 0;
    Timestamp now = Timestamp::Now();

    for (const auto& pattern_id : patterns_to_update) {
        float utility = CalculateUtilityForPattern(pattern_id);

        // Update history
        {
            std::unique_lock<std::shared_mutex> lock(utilities_mutex_);
            auto it = pattern_utilities_.find(pattern_id);
            if (it == pattern_utilities_.end()) {
                // Create new history using piecewise_construct
                it = pattern_utilities_.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(pattern_id),
                    std::forward_as_tuple(config_.max_history_size)
                ).first;
            }
            it->second.AddRecord(utility, now);
        }

        updated_count++;
    }

    // Update top-K
    UpdateTopK();

    // Update statistics
    last_update_time_ = now;
    total_updates_.fetch_add(1);

    return updated_count;
}

float UtilityTracker::UpdatePatternUtility(PatternID pattern) {
    float utility = CalculateUtilityForPattern(pattern);

    // Update history
    {
        std::unique_lock<std::shared_mutex> lock(utilities_mutex_);
        auto it = pattern_utilities_.find(pattern);
        if (it == pattern_utilities_.end()) {
            // Create new history using piecewise_construct
            it = pattern_utilities_.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(pattern),
                std::forward_as_tuple(config_.max_history_size)
            ).first;
        }
        it->second.AddRecord(utility);
    }

    // Note: Top-K not updated here for efficiency (batch updates only)

    return utility;
}

float UtilityTracker::GetPatternUtility(PatternID pattern) const {
    std::shared_lock<std::shared_mutex> lock(utilities_mutex_);

    auto it = pattern_utilities_.find(pattern);
    if (it == pattern_utilities_.end()) {
        return 0.0f;
    }

    return it->second.GetCurrentUtility();
}

const UtilityHistory* UtilityTracker::GetPatternHistory(PatternID pattern) const {
    std::shared_lock<std::shared_mutex> lock(utilities_mutex_);

    auto it = pattern_utilities_.find(pattern);
    if (it == pattern_utilities_.end()) {
        return nullptr;
    }

    return &it->second;
}

UtilityTrend UtilityTracker::GetPatternTrend(PatternID pattern, size_t window_size) const {
    std::shared_lock<std::shared_mutex> lock(utilities_mutex_);

    auto it = pattern_utilities_.find(pattern);
    if (it == pattern_utilities_.end()) {
        return UtilityTrend::STABLE;
    }

    return it->second.DetectTrend(window_size);
}

std::vector<std::pair<PatternID, float>> UtilityTracker::GetTopKPatterns(size_t k) const {
    std::shared_lock<std::shared_mutex> lock(top_k_mutex_);

    if (k == 0) {
        k = config_.top_k_size;
    }

    size_t result_size = std::min(k, top_k_patterns_.size());
    return std::vector<std::pair<PatternID, float>>(
        top_k_patterns_.begin(),
        top_k_patterns_.begin() + result_size
    );
}

std::vector<PatternID> UtilityTracker::GetIncreasingPatterns(float min_change_rate) const {
    std::shared_lock<std::shared_mutex> lock(utilities_mutex_);

    std::vector<PatternID> result;

    for (const auto& [pattern_id, history] : pattern_utilities_) {
        float change_rate = history.GetChangeRate();
        if (change_rate >= min_change_rate) {
            result.push_back(pattern_id);
        }
    }

    return result;
}

std::vector<PatternID> UtilityTracker::GetDecreasingPatterns(float max_change_rate) const {
    std::shared_lock<std::shared_mutex> lock(utilities_mutex_);

    std::vector<PatternID> result;

    for (const auto& [pattern_id, history] : pattern_utilities_) {
        float change_rate = history.GetChangeRate();
        if (change_rate <= max_change_rate) {
            result.push_back(pattern_id);
        }
    }

    return result;
}

size_t UtilityTracker::GetTrackedPatternCount() const {
    std::shared_lock<std::shared_mutex> lock(utilities_mutex_);
    return pattern_utilities_.size();
}

void UtilityTracker::Clear() {
    {
        std::unique_lock<std::shared_mutex> lock(utilities_mutex_);
        pattern_utilities_.clear();
    }

    {
        std::unique_lock<std::shared_mutex> lock(top_k_mutex_);
        top_k_patterns_.clear();
    }

    total_updates_.store(0);
}

UtilityTracker::Statistics UtilityTracker::GetStatistics() const {
    std::shared_lock<std::shared_mutex> lock(utilities_mutex_);

    Statistics stats;
    stats.total_tracked_patterns = pattern_utilities_.size();
    stats.total_updates_performed = total_updates_.load();
    stats.last_update_time = last_update_time_;

    if (pattern_utilities_.empty()) {
        return stats;
    }

    // Calculate aggregate statistics
    float total_utility = 0.0f;
    float max_util = 0.0f;
    float min_util = 1.0f;

    for (const auto& [pattern_id, history] : pattern_utilities_) {
        float utility = history.GetCurrentUtility();
        total_utility += utility;
        max_util = std::max(max_util, utility);
        min_util = std::min(min_util, utility);

        // Count trends
        UtilityTrend trend = history.DetectTrend(10);
        switch (trend) {
            case UtilityTrend::INCREASING:
                stats.patterns_increasing++;
                break;
            case UtilityTrend::DECREASING:
                stats.patterns_decreasing++;
                break;
            case UtilityTrend::STABLE:
                stats.patterns_stable++;
                break;
        }
    }

    stats.average_utility = total_utility / static_cast<float>(pattern_utilities_.size());
    stats.max_utility = max_util;
    stats.min_utility = min_util;

    return stats;
}

void UtilityTracker::SetConfig(const Config& config) {
    if (!config.IsValid()) {
        throw std::invalid_argument("Invalid UtilityTracker configuration");
    }

    config_ = config;

    // If auto-update setting changed, restart thread
    if (config_.enable_auto_update && !running_.load()) {
        Start();
    } else if (!config_.enable_auto_update && running_.load()) {
        Stop();
    }
}

// ============================================================================
// Private Methods
// ============================================================================

void UtilityTracker::BackgroundUpdateLoop() {
    while (running_.load()) {
        // Sleep for update interval
        std::this_thread::sleep_for(config_.update_interval);

        if (!running_.load()) {
            break;
        }

        // Perform update
        UpdateAllUtilities();
    }
}

void UtilityTracker::UpdateTopK() {
    std::vector<std::pair<PatternID, float>> all_utilities;

    // Collect all current utilities
    {
        std::shared_lock<std::shared_mutex> lock(utilities_mutex_);
        all_utilities.reserve(pattern_utilities_.size());

        for (const auto& [pattern_id, history] : pattern_utilities_) {
            all_utilities.emplace_back(pattern_id, history.GetCurrentUtility());
        }
    }

    // Sort by utility (descending)
    std::sort(all_utilities.begin(), all_utilities.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    // Keep top-K
    size_t k = std::min(config_.top_k_size, all_utilities.size());
    all_utilities.resize(k);

    // Update top-K storage
    {
        std::unique_lock<std::shared_mutex> lock(top_k_mutex_);
        top_k_patterns_ = std::move(all_utilities);
    }
}

float UtilityTracker::CalculateUtilityForPattern(PatternID pattern) {
    // Get access statistics
    const AccessStats* stats = access_tracker_.GetPatternStats(pattern);
    if (!stats) {
        // No access stats yet - return low utility
        return 0.0f;
    }

    // Get associations for this pattern
    auto association_ptrs = association_matrix_.GetOutgoingAssociations(pattern);

    // Convert pointers to values for utility calculation
    // Since AssociationEdge is not copyable, we reconstruct from data
    std::vector<AssociationEdge> associations;
    associations.reserve(association_ptrs.size());
    for (const auto* edge_ptr : association_ptrs) {
        if (edge_ptr) {
            // Reconstruct association edge with current data
            AssociationEdge edge(
                edge_ptr->GetSource(),
                edge_ptr->GetTarget(),
                edge_ptr->GetType(),
                edge_ptr->GetStrength()
            );
            associations.push_back(std::move(edge));
        }
    }

    // Create a minimal PatternNode for utility calculation
    // In a real system, we'd retrieve the actual pattern from storage
    // Using minimal/empty data for calculation
    FeatureVector empty_features;
    PatternData minimal_data = PatternData::FromFeatures(empty_features, DataModality::NUMERIC);
    PatternNode pattern_node(pattern, minimal_data, PatternType::ATOMIC);

    // Calculate utility
    float utility = calculator_.CalculatePatternUtility(pattern_node, *stats, associations);

    return utility;
}

} // namespace dpan
