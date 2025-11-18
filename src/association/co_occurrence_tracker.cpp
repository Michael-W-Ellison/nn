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
    // Add to activation history (must maintain sorted order)
    activations_.push_back({timestamp, pattern});
    pattern_counts_[pattern]++;

    // Get all patterns in the current window
    Timestamp window_start = timestamp - config_.window_size;
    auto patterns_in_window = GetPatternsInWindow(window_start, timestamp);

    // Update co-occurrences
    UpdateCoOccurrences(patterns_in_window);

    // Prune old activations (older than 2x window size to save memory)
    Timestamp cutoff = timestamp - (config_.window_size * 2);
    PruneOldActivations(cutoff);
}

void CoOccurrenceTracker::RecordActivations(
    const std::vector<PatternID>& patterns,
    Timestamp timestamp
) {
    // Add all patterns with same timestamp
    for (const auto& pattern : patterns) {
        activations_.push_back({timestamp, pattern});
        pattern_counts_[pattern]++;
    }

    // Update co-occurrences for this window
    UpdateCoOccurrences(patterns);
    total_windows_++;
}

// ============================================================================
// Helper Methods
// ============================================================================

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
            PatternID p1 = pattern_vec[i];
            PatternID p2 = pattern_vec[j];

            // Always store in consistent order (smaller ID first)
            if (p2 < p1) {
                std::swap(p1, p2);
            }

            auto key = std::make_pair(p1, p2);
            co_occurrence_counts_[key]++;
        }
    }
}

// ============================================================================
// Querying Co-occurrences
// ============================================================================

uint32_t CoOccurrenceTracker::GetCoOccurrenceCount(PatternID p1, PatternID p2) const {
    // Ensure consistent ordering
    if (p2 < p1) {
        std::swap(p1, p2);
    }

    auto key = std::make_pair(p1, p2);
    auto it = co_occurrence_counts_.find(key);

    return it != co_occurrence_counts_.end() ? it->second : 0;
}

float CoOccurrenceTracker::GetCoOccurrenceProbability(PatternID p1, PatternID p2) const {
    uint32_t co_count = GetCoOccurrenceCount(p1, p2);

    if (co_count == 0 || total_windows_ == 0) {
        return 0.0f;
    }

    // P(p1, p2) = count(p1 AND p2) / total_windows
    return static_cast<float>(co_count) / total_windows_;
}

bool CoOccurrenceTracker::IsSignificant(PatternID p1, PatternID p2) const {
    uint32_t count = GetCoOccurrenceCount(p1, p2);

    // First check minimum count threshold
    if (count < config_.min_co_occurrences) {
        return false;
    }

    // Then check chi-squared significance
    float chi_squared = GetChiSquared(p1, p2);

    // Chi-squared critical value for df=1, alpha=0.05 is 3.841
    return chi_squared > 3.841f;
}

float CoOccurrenceTracker::GetChiSquared(PatternID p1, PatternID p2) const {
    if (total_windows_ == 0) {
        return 0.0f;
    }

    // Contingency table for chi-squared test:
    // |       |  p2   | !p2   |
    // |-------|-------|-------|
    // |  p1   |   a   |   b   |
    // | !p1   |   c   |   d   |

    uint32_t a = GetCoOccurrenceCount(p1, p2);  // Both present

    auto it1 = pattern_counts_.find(p1);
    auto it2 = pattern_counts_.find(p2);

    if (it1 == pattern_counts_.end() || it2 == pattern_counts_.end()) {
        return 0.0f;
    }

    uint32_t p1_count = it1->second;
    uint32_t p2_count = it2->second;

    uint32_t b = p1_count - a;          // p1 without p2
    uint32_t c = p2_count - a;          // p2 without p1
    uint32_t d = total_windows_ - a - b - c;  // Neither

    uint64_t n = total_windows_;

    // Chi-squared formula: χ² = n(ad - bc)² / [(a+b)(c+d)(a+c)(b+d)]
    // Use uint64_t to avoid overflow
    uint64_t ad = static_cast<uint64_t>(a) * d;
    uint64_t bc = static_cast<uint64_t>(b) * c;
    int64_t diff = static_cast<int64_t>(ad) - static_cast<int64_t>(bc);

    uint64_t numerator = n * static_cast<uint64_t>(diff) * static_cast<uint64_t>(diff);

    uint64_t denominator = static_cast<uint64_t>(a + b) *
                           static_cast<uint64_t>(c + d) *
                           static_cast<uint64_t>(a + c) *
                           static_cast<uint64_t>(b + d);

    if (denominator == 0) {
        return 0.0f;
    }

    return static_cast<float>(numerator) / static_cast<float>(denominator);
}

std::vector<std::pair<PatternID, uint32_t>> CoOccurrenceTracker::GetCoOccurringPatterns(
    PatternID pattern,
    uint32_t min_count
) const {
    std::vector<std::pair<PatternID, uint32_t>> results;

    for (const auto& [key, count] : co_occurrence_counts_) {
        if (count < min_count) {
            continue;
        }

        // Check if pattern is in the pair
        if (key.first == pattern) {
            results.emplace_back(key.second, count);
        } else if (key.second == pattern) {
            results.emplace_back(key.first, count);
        }
    }

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
    // Remove activations older than cutoff
    // Since deque is sorted, remove from front
    while (!activations_.empty() && activations_.front().first < cutoff_time) {
        activations_.pop_front();
    }
}

void CoOccurrenceTracker::Clear() {
    activations_.clear();
    co_occurrence_counts_.clear();
    pattern_counts_.clear();
    total_windows_ = 0;
}

std::vector<PatternID> CoOccurrenceTracker::GetTrackedPatterns() const {
    std::vector<PatternID> patterns;
    patterns.reserve(pattern_counts_.size());

    for (const auto& [pattern, count] : pattern_counts_) {
        patterns.push_back(pattern);
    }

    return patterns;
}

} // namespace dpan
