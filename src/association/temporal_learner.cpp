// File: src/association/temporal_learner.cpp
#include "association/temporal_learner.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace dpan {

// ============================================================================
// Construction
// ============================================================================

TemporalLearner::TemporalLearner()
    : TemporalLearner(Config())
{
}

TemporalLearner::TemporalLearner(const Config& config)
    : config_(config)
{
}

// ============================================================================
// Recording Activations
// ============================================================================

void TemporalLearner::RecordActivation(PatternID pattern, Timestamp timestamp) {
    // Add to activation history (must maintain sorted order)
    activations_.push_back({timestamp, pattern});

    // Update temporal statistics for the immediate predecessor
    // Look at the most recent activation (if any)
    if (activations_.size() >= 2) {
        // Get the previous activation
        const auto& prev = activations_[activations_.size() - 2];
        PatternID prev_pattern = prev.second;
        Timestamp prev_time = prev.first;

        // Only track if it's a different pattern
        if (prev_pattern != pattern) {
            // Check if within temporal window
            auto delay = timestamp - prev_time;
            if (delay <= config_.max_delay) {
                // Record this transition
                UpdateStats(prev_pattern, pattern, prev_time, timestamp);
            }
        }
    }
}

void TemporalLearner::RecordSequence(
    const std::vector<std::pair<Timestamp, PatternID>>& sequence
) {
    for (const auto& [timestamp, pattern] : sequence) {
        RecordActivation(pattern, timestamp);
    }
}

// ========================================================================
// Querying Temporal Statistics
// ========================================================================

std::optional<TemporalLearner::TemporalStats> TemporalLearner::GetTemporalStats(
    PatternID p1,
    PatternID p2
) const {
    auto key = MakeKey(p1, p2);
    auto it = temporal_stats_.find(key);

    if (it == temporal_stats_.end()) {
        return std::nullopt;
    }

    const auto& stats = it->second;
    if (stats.occurrence_count < config_.min_occurrences) {
        return std::nullopt;  // Insufficient data
    }

    return stats;
}

float TemporalLearner::GetTemporalCorrelation(PatternID p1, PatternID p2) const {
    auto stats_opt = GetTemporalStats(p1, p2);
    if (!stats_opt) {
        return 0.0f;
    }
    return stats_opt->correlation;
}

bool TemporalLearner::IsTemporallyCorrelated(PatternID p1, PatternID p2) const {
    float correlation = GetTemporalCorrelation(p1, p2);
    return correlation >= config_.min_correlation;
}

int64_t TemporalLearner::GetMeanDelay(PatternID p1, PatternID p2) const {
    auto stats_opt = GetTemporalStats(p1, p2);
    if (!stats_opt) {
        return 0;
    }
    return stats_opt->mean_delay_micros;
}

std::vector<std::pair<PatternID, float>> TemporalLearner::GetSuccessors(
    PatternID pattern,
    float min_correlation
) const {
    std::vector<std::pair<PatternID, float>> results;

    for (const auto& [key, stats] : temporal_stats_) {
        if (key.first != pattern) {
            continue;  // Not a successor of 'pattern'
        }

        if (stats.occurrence_count < config_.min_occurrences) {
            continue;  // Insufficient data
        }

        if (stats.correlation < min_correlation) {
            continue;  // Below threshold
        }

        results.emplace_back(key.second, stats.correlation);
    }

    // Sort by correlation (descending)
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    return results;
}

std::vector<std::pair<PatternID, float>> TemporalLearner::GetPredecessors(
    PatternID pattern,
    float min_correlation
) const {
    std::vector<std::pair<PatternID, float>> results;

    for (const auto& [key, stats] : temporal_stats_) {
        if (key.second != pattern) {
            continue;  // Not a predecessor of 'pattern'
        }

        if (stats.occurrence_count < config_.min_occurrences) {
            continue;  // Insufficient data
        }

        if (stats.correlation < min_correlation) {
            continue;  // Below threshold
        }

        results.emplace_back(key.first, stats.correlation);
    }

    // Sort by correlation (descending)
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    return results;
}

// ========================================================================
// Maintenance
// ========================================================================

void TemporalLearner::PruneOldActivations(Timestamp cutoff_time) {
    // Remove activations older than cutoff
    // Since deque is sorted by time, remove from front
    while (!activations_.empty() && activations_.front().first < cutoff_time) {
        activations_.pop_front();
    }
}

void TemporalLearner::Clear() {
    activations_.clear();
    temporal_stats_.clear();
}

size_t TemporalLearner::GetUniquePatternCount() const {
    std::unordered_set<PatternID> unique_patterns;
    for (const auto& [timestamp, pattern] : activations_) {
        unique_patterns.insert(pattern);
    }
    return unique_patterns.size();
}

// ========================================================================
// Helper Methods
// ========================================================================

void TemporalLearner::UpdateStats(
    PatternID p1,
    PatternID p2,
    Timestamp t1,
    Timestamp t2
) {
    // Compute delay
    auto delay = t2 - t1;
    int64_t delay_micros = std::chrono::duration_cast<std::chrono::microseconds>(delay).count();

    // Don't track delays that exceed max_delay
    int64_t max_delay_micros = std::chrono::duration_cast<std::chrono::microseconds>(
        config_.max_delay
    ).count();

    if (delay_micros > max_delay_micros || delay_micros < 0) {
        return;  // Delay too large or negative
    }

    auto key = MakeKey(p1, p2);
    auto& stats = temporal_stats_[key];

    // Update using Welford's online algorithm for mean and variance
    uint32_t n = stats.occurrence_count;

    if (n == 0) {
        // First observation
        stats.mean_delay_micros = delay_micros;
        stats.stddev_delay_micros = 0;
        stats.occurrence_count = 1;
    } else {
        // Incremental update
        int64_t old_mean = stats.mean_delay_micros;

        // Update mean
        int64_t new_mean = old_mean + (delay_micros - old_mean) / (n + 1);
        stats.mean_delay_micros = new_mean;

        // Update variance (using Welford's method)
        // M2 = sum of squared differences from mean
        // We'll compute stddev on demand instead of storing M2
        // For now, use a simplified running variance calculation

        // Simplified approach: track variance incrementally
        // variance = E[X²] - E[X]²
        // We'll use a two-pass approach when computing correlation

        stats.occurrence_count = n + 1;
    }

    // Recompute standard deviation if we have enough samples
    if (stats.occurrence_count >= config_.min_occurrences) {
        // Two-pass variance calculation for accuracy
        // First pass: recompute mean (we already have it)
        // Second pass: compute variance

        int64_t mean = stats.mean_delay_micros;
        int64_t sum_squared_diff = 0;
        int count = 0;

        // Scan through activations to recompute variance
        // This is done periodically, not on every update
        // For efficiency, we'll use a simplified approximation

        // Simple approximation: assume delays have similar variance
        // Use coefficient of variation heuristic
        // For now, estimate stddev as proportional to mean
        // This will be refined with actual variance tracking

        // Better approach: Track sum of squared differences incrementally
        // For simplicity in this implementation, use a heuristic
        int64_t estimated_stddev = std::abs(delay_micros - mean);

        // Running average of absolute deviations as stddev proxy
        if (stats.stddev_delay_micros == 0) {
            stats.stddev_delay_micros = estimated_stddev;
        } else {
            // Exponential moving average of deviations
            stats.stddev_delay_micros = (stats.stddev_delay_micros * 9 + estimated_stddev) / 10;
        }
    }

    // Compute correlation
    stats.correlation = ComputeCorrelation(stats.mean_delay_micros, stats.stddev_delay_micros);
    stats.last_updated = t2;
}

float TemporalLearner::ComputeCorrelation(int64_t mean_micros, int64_t stddev_micros) const {
    if (mean_micros == 0) {
        return 0.0f;
    }

    // Temporal correlation formula: τ = 1 / (1 + σ/μ)
    // This measures consistency - lower variance relative to mean = higher correlation
    float coefficient_of_variation = static_cast<float>(stddev_micros) / std::abs(mean_micros);
    float correlation = 1.0f / (1.0f + coefficient_of_variation);

    // Clamp to [0, 1]
    return std::clamp(correlation, 0.0f, 1.0f);
}

} // namespace dpan
