// File: src/storage/indices/temporal_index.cpp
#include "storage/indices/temporal_index.hpp"
#include <algorithm>

namespace dpan {

void TemporalIndex::Insert(PatternID id, Timestamp timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Remove old entry if exists
    auto it = pattern_to_time_.find(id);
    if (it != pattern_to_time_.end()) {
        // Find and remove from time index
        auto range = time_to_pattern_.equal_range(it->second);
        for (auto time_it = range.first; time_it != range.second; ++time_it) {
            if (time_it->second == id) {
                time_to_pattern_.erase(time_it);
                break;
            }
        }
    }

    // Insert new entry
    time_to_pattern_.insert({timestamp, id});
    pattern_to_time_[id] = timestamp;
}

bool TemporalIndex::Remove(PatternID id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pattern_to_time_.find(id);
    if (it == pattern_to_time_.end()) {
        return false;
    }

    Timestamp timestamp = it->second;

    // Remove from both maps
    pattern_to_time_.erase(it);

    auto range = time_to_pattern_.equal_range(timestamp);
    for (auto time_it = range.first; time_it != range.second; ++time_it) {
        if (time_it->second == id) {
            time_to_pattern_.erase(time_it);
            break;
        }
    }

    return true;
}

std::vector<PatternID> TemporalIndex::FindInRange(
        Timestamp start,
        Timestamp end,
        size_t max_results) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<PatternID> results;
    results.reserve(std::min(max_results, time_to_pattern_.size()));

    auto it = time_to_pattern_.lower_bound(start);
    auto end_it = time_to_pattern_.upper_bound(end);

    for (; it != end_it && results.size() < max_results; ++it) {
        results.push_back(it->second);
    }

    return results;
}

std::vector<PatternID> TemporalIndex::FindBefore(
        Timestamp timestamp,
        size_t max_results) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<PatternID> results;
    results.reserve(std::min(max_results, time_to_pattern_.size()));

    auto end_it = time_to_pattern_.lower_bound(timestamp);

    // Iterate from the end backwards
    if (end_it == time_to_pattern_.begin()) {
        return results;  // Nothing before
    }

    auto it = end_it;
    --it;  // Move to the last element before timestamp

    // Collect results in reverse order, then reverse the vector
    while (results.size() < max_results) {
        results.push_back(it->second);

        if (it == time_to_pattern_.begin()) {
            break;
        }
        --it;
    }

    std::reverse(results.begin(), results.end());
    return results;
}

std::vector<PatternID> TemporalIndex::FindAfter(
        Timestamp timestamp,
        size_t max_results) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<PatternID> results;
    results.reserve(std::min(max_results, time_to_pattern_.size()));

    auto it = time_to_pattern_.upper_bound(timestamp);

    for (; it != time_to_pattern_.end() && results.size() < max_results; ++it) {
        results.push_back(it->second);
    }

    return results;
}

std::vector<PatternID> TemporalIndex::FindMostRecent(size_t n) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<PatternID> results;
    results.reserve(std::min(n, time_to_pattern_.size()));

    if (time_to_pattern_.empty()) {
        return results;
    }

    auto it = time_to_pattern_.rbegin();  // Start from the most recent

    for (; it != time_to_pattern_.rend() && results.size() < n; ++it) {
        results.push_back(it->second);
    }

    return results;
}

std::vector<PatternID> TemporalIndex::FindOldest(size_t n) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<PatternID> results;
    results.reserve(std::min(n, time_to_pattern_.size()));

    auto it = time_to_pattern_.begin();

    for (; it != time_to_pattern_.end() && results.size() < n; ++it) {
        results.push_back(it->second);
    }

    return results;
}

std::optional<Timestamp> TemporalIndex::GetTimestamp(PatternID id) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = pattern_to_time_.find(id);
    if (it != pattern_to_time_.end()) {
        return it->second;
    }

    return std::nullopt;
}

size_t TemporalIndex::Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pattern_to_time_.size();
}

void TemporalIndex::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    time_to_pattern_.clear();
    pattern_to_time_.clear();
}

TemporalIndex::Stats TemporalIndex::GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    Stats stats;
    stats.total_patterns = pattern_to_time_.size();

    if (!time_to_pattern_.empty()) {
        stats.earliest = time_to_pattern_.begin()->first;
        stats.latest = time_to_pattern_.rbegin()->first;

        // Calculate average patterns per second
        int64_t time_span_micros = stats.latest.ToMicros() - stats.earliest.ToMicros();
        if (time_span_micros > 0) {
            double time_span_seconds = time_span_micros / 1000000.0;
            stats.avg_patterns_per_second = stats.total_patterns / time_span_seconds;
        }
    }

    return stats;
}

} // namespace dpan
