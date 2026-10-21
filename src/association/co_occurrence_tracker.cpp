// File: src/association/co_occurrence_tracker.cpp
#include "association/co_occurrence_tracker.hpp"
#include <algorithm>
#include <cmath>

namespace dpan {

// ============================================================================
// Construction
// ============================================================================

CoOccurrenceTracker::CoOccurrenceTracker()
    : CoOccurrenceTracker(Config())
{
}

CoOccurrenceTracker::CoOccurrenceTracker(const Config& config)
    : config_(config)
{
}

// ============================================================================
// Recording Activations
// ============================================================================

void CoOccurrenceTracker::RecordActivation(PatternID pattern, Timestamp timestamp) {
    std::unique_lock lock(mutex_);

    // Pair the new activation with every other pattern activated in
    // [timestamp - window_size, timestamp]
    if (!has_latest_ || !(timestamp < latest_timestamp_)) {
        // In-order activation: the sliding window is exactly the query range
        latest_timestamp_ = timestamp;
        has_latest_ = true;
        AdvanceWindow();
        for (const auto& [other, count] : window_counts_) {
            if (other != pattern) {
                IncrementPair(pattern, other);
            }
        }
    } else {
        // Out-of-order activation: scan the sorted history
        auto in_window = GetPatternsInWindow(timestamp - config_.window_size, timestamp);
        std::unordered_set<PatternID> unique(in_window.begin(), in_window.end());
        unique.erase(pattern);
        for (const auto& other : unique) {
            IncrementPair(pattern, other);
        }
    }

    InsertActivation(pattern, timestamp);
    pattern_counts_[pattern]++;

    // Prune old activations (older than 2x window size to save memory)
    PruneOldActivationsUnlocked(latest_timestamp_ - (config_.window_size * 2));
}

void CoOccurrenceTracker::RecordActivations(
    const std::vector<PatternID>& patterns,
    Timestamp timestamp
) {
    std::unique_lock lock(mutex_);

    if (!has_latest_ || !(timestamp < latest_timestamp_)) {
        latest_timestamp_ = timestamp;
        has_latest_ = true;
        AdvanceWindow();
    }

    // Add all patterns with same timestamp
    for (const auto& pattern : patterns) {
        InsertActivation(pattern, timestamp);
        pattern_counts_[pattern]++;
    }

    // Update co-occurrences for this window
    UpdateCoOccurrences(patterns);
    total_windows_++;
}

// ============================================================================
// Pair Count Table
// ============================================================================

namespace {

inline uint64_t MixPair(uint64_t first, uint64_t second) {
    uint64_t x = first * 0x9e3779b97f4a7c15ULL ^ (second + 0x632be59bd9b4e019ULL);
    x ^= x >> 31;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 29;
    return x;
}

}  // namespace

size_t CoOccurrenceTracker::PairCounts::FindSlot(uint64_t first, uint64_t second) const {
    size_t mask = slots_.size() - 1;
    size_t index = static_cast<size_t>(MixPair(first, second)) & mask;
    while (slots_[index].first != 0 &&
           (slots_[index].first != first || slots_[index].second != second)) {
        index = (index + 1) & mask;  // Linear probing
    }
    return index;
}

void CoOccurrenceTracker::PairCounts::Grow() {
    std::vector<Slot> old = std::move(slots_);
    slots_.assign(old.empty() ? 64 : old.size() * 2, Slot{});
    for (const auto& slot : old) {
        if (slot.first != 0) {
            slots_[FindSlot(slot.first, slot.second)] = slot;
        }
    }
}

void CoOccurrenceTracker::PairCounts::Increment(PatternID p1, PatternID p2) {
    // Keep load factor <= 1/2
    if ((size_ + 1) * 2 > slots_.size()) {
        Grow();
    }
    Slot& slot = slots_[FindSlot(p1.value(), p2.value())];
    if (slot.first == 0) {
        slot.first = p1.value();
        slot.second = p2.value();
        size_++;
    }
    slot.count++;
}

uint32_t CoOccurrenceTracker::PairCounts::Get(PatternID p1, PatternID p2) const {
    if (slots_.empty()) {
        return 0;
    }
    const Slot& slot = slots_[FindSlot(p1.value(), p2.value())];
    return slot.first != 0 ? slot.count : 0;
}

void CoOccurrenceTracker::PairCounts::Clear() {
    slots_.clear();
    size_ = 0;
}

// ============================================================================
// Helper Methods
// ============================================================================

void CoOccurrenceTracker::InsertActivation(PatternID pattern, Timestamp timestamp) {
    // Keep activations_ sorted by timestamp
    auto pos = std::upper_bound(
        activations_.begin(),
        activations_.end(),
        timestamp,
        [](Timestamp time, const auto& activation) {
            return time < activation.first;
        }
    );
    size_t index = static_cast<size_t>(pos - activations_.begin());
    activations_.insert(pos, {timestamp, pattern});

    if (timestamp < latest_timestamp_ - config_.window_size) {
        // Older than the sliding window
        if (index <= window_begin_) {
            window_begin_++;
        }
    } else {
        window_counts_[pattern]++;
    }
}

void CoOccurrenceTracker::AdvanceWindow() {
    Timestamp window_start = latest_timestamp_ - config_.window_size;
    while (window_begin_ < activations_.size() &&
           activations_[window_begin_].first < window_start) {
        DecrementWindowCount(activations_[window_begin_].second);
        window_begin_++;
    }
}

void CoOccurrenceTracker::DecrementWindowCount(PatternID pattern) {
    auto it = window_counts_.find(pattern);
    if (it != window_counts_.end() && --it->second == 0) {
        window_counts_.erase(it);
    }
}

void CoOccurrenceTracker::IncrementPair(PatternID p1, PatternID p2) {
    // Always store in consistent order (smaller ID first)
    if (p2 < p1) {
        std::swap(p1, p2);
    }
    co_occurrence_counts_.Increment(p1, p2);
}

std::vector<PatternID> CoOccurrenceTracker::GetPatternsInWindow(
    Timestamp start,
    Timestamp end
) const {
    std::vector<PatternID> result;

    // Binary search for start position
    auto it_start = std::lower_bound(
        activations_.begin(),
        activations_.end(),
        start,
        [](const auto& activation, Timestamp time) {
            return activation.first < time;
        }
    );

    // Collect patterns in window [start, end]
    for (auto it = it_start; it != activations_.end() && it->first <= end; ++it) {
        result.push_back(it->second);
    }

    return result;
}

void CoOccurrenceTracker::UpdateCoOccurrences(const std::vector<PatternID>& patterns_in_window) {
    // Create unique set to avoid counting duplicates
    std::unordered_set<PatternID> unique_patterns(
        patterns_in_window.begin(),
        patterns_in_window.end()
    );

    // Convert to vector for iteration
    std::vector<PatternID> pattern_vec(unique_patterns.begin(), unique_patterns.end());

    // Update co-occurrence count for all unique pairs
    for (size_t i = 0; i < pattern_vec.size(); ++i) {
        for (size_t j = i + 1; j < pattern_vec.size(); ++j) {
            IncrementPair(pattern_vec[i], pattern_vec[j]);
        }
    }
}

// ============================================================================
// Querying Co-occurrences
// ============================================================================

uint32_t CoOccurrenceTracker::GetCoOccurrenceCount(PatternID p1, PatternID p2) const {
    std::shared_lock lock(mutex_);
    return GetCoOccurrenceCountUnlocked(p1, p2);
}

uint32_t CoOccurrenceTracker::GetCoOccurrenceCountUnlocked(PatternID p1, PatternID p2) const {
    // Ensure consistent ordering
    if (p2 < p1) {
        std::swap(p1, p2);
    }

    return co_occurrence_counts_.Get(p1, p2);
}

float CoOccurrenceTracker::GetCoOccurrenceProbability(PatternID p1, PatternID p2) const {
    std::shared_lock lock(mutex_);
    uint32_t co_count = GetCoOccurrenceCountUnlocked(p1, p2);

    if (co_count == 0 || total_windows_ == 0) {
        return 0.0f;
    }

    // P(p1, p2) = count(p1 AND p2) / total_windows
    return static_cast<float>(co_count) / total_windows_;
}

bool CoOccurrenceTracker::IsSignificant(PatternID p1, PatternID p2) const {
    std::shared_lock lock(mutex_);
    uint32_t count = GetCoOccurrenceCountUnlocked(p1, p2);

    // First check minimum count threshold
    if (count < config_.min_co_occurrences) {
        return false;
    }

    // Then check chi-squared significance
    float chi_squared = GetChiSquaredUnlocked(p1, p2);

    // Chi-squared critical value for df=1, alpha=0.05 is 3.841
    return chi_squared > 3.841f;
}

float CoOccurrenceTracker::GetChiSquared(PatternID p1, PatternID p2) const {
    std::shared_lock lock(mutex_);
    return GetChiSquaredUnlocked(p1, p2);
}

float CoOccurrenceTracker::GetChiSquaredUnlocked(PatternID p1, PatternID p2) const {
    if (total_windows_ == 0) {
        return 0.0f;
    }

    // Contingency table for chi-squared test:
    // |       |  p2   | !p2   |
    // |-------|-------|-------|
    // |  p1   |   a   |   b   |
    // | !p1   |   c   |   d   |

    uint32_t a = GetCoOccurrenceCountUnlocked(p1, p2);  // Both present

    auto it1 = pattern_counts_.find(p1);
    auto it2 = pattern_counts_.find(p2);

    if (it1 == pattern_counts_.end() || it2 == pattern_counts_.end()) {
        return 0.0f;
    }

    uint32_t p1_count = it1->second;
    uint32_t p2_count = it2->second;

    // Single activations (RecordActivation) also count toward pattern and
    // pair totals without adding windows, so clamp cells at zero
    uint64_t b = p1_count > a ? p1_count - a : 0;  // p1 without p2
    uint64_t c = p2_count > a ? p2_count - a : 0;  // p2 without p1
    uint64_t used = static_cast<uint64_t>(a) + b + c;
    uint64_t d = total_windows_ > used ? total_windows_ - used : 0;  // Neither

    double n = static_cast<double>(total_windows_);

    // Chi-squared formula: χ² = n(ad - bc)² / [(a+b)(c+d)(a+c)(b+d)]
    // Computed in double: the 64-bit integer products overflow for large counts
    double diff = static_cast<double>(a) * static_cast<double>(d) -
                  static_cast<double>(b) * static_cast<double>(c);

    double denominator = static_cast<double>(a + b) *
                         static_cast<double>(c + d) *
                         static_cast<double>(a + c) *
                         static_cast<double>(b + d);

    if (denominator == 0.0) {
        return 0.0f;
    }

    return static_cast<float>(n * diff * diff / denominator);
}

std::vector<std::pair<PatternID, uint32_t>> CoOccurrenceTracker::GetCoOccurringPatterns(
    PatternID pattern,
    uint32_t min_count
) const {
    std::vector<std::pair<PatternID, uint32_t>> results;

    std::shared_lock lock(mutex_);
    co_occurrence_counts_.ForEach([&](PatternID first, PatternID second, uint32_t count) {
        if (count < min_count) {
            return;
        }

        // Check if pattern is in the pair
        if (first == pattern) {
            results.emplace_back(second, count);
        } else if (second == pattern) {
            results.emplace_back(first, count);
        }
    });

    // Sort by count (descending)
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    return results;
}

// ============================================================================
// Maintenance
// ============================================================================

void CoOccurrenceTracker::PruneOldActivations(Timestamp cutoff_time) {
    std::unique_lock lock(mutex_);
    PruneOldActivationsUnlocked(cutoff_time);
}

void CoOccurrenceTracker::PruneOldActivationsUnlocked(Timestamp cutoff_time) {
    // Remove activations older than cutoff
    // Since deque is sorted, remove from front
    while (!activations_.empty() && activations_.front().first < cutoff_time) {
        if (window_begin_ > 0) {
            window_begin_--;
        } else {
            DecrementWindowCount(activations_.front().second);
        }
        activations_.pop_front();
    }
}

void CoOccurrenceTracker::Clear() {
    std::unique_lock lock(mutex_);
    activations_.clear();
    co_occurrence_counts_.Clear();
    pattern_counts_.clear();
    window_counts_.clear();
    window_begin_ = 0;
    has_latest_ = false;
    latest_timestamp_ = Timestamp();
    total_windows_ = 0;
}

size_t CoOccurrenceTracker::GetActivationCount() const {
    std::shared_lock lock(mutex_);
    return activations_.size();
}

size_t CoOccurrenceTracker::GetUniquePatternCount() const {
    std::shared_lock lock(mutex_);
    return pattern_counts_.size();
}

uint64_t CoOccurrenceTracker::GetTotalWindows() const {
    std::shared_lock lock(mutex_);
    return total_windows_;
}

size_t CoOccurrenceTracker::GetCoOccurrencePairCount() const {
    std::shared_lock lock(mutex_);
    return co_occurrence_counts_.Size();
}

std::vector<PatternID> CoOccurrenceTracker::GetTrackedPatterns() const {
    std::shared_lock lock(mutex_);
    std::vector<PatternID> patterns;
    patterns.reserve(pattern_counts_.size());

    for (const auto& [pattern, count] : pattern_counts_) {
        patterns.push_back(pattern);
    }

    return patterns;
}

} // namespace dpan
