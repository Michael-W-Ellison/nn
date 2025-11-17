// File: src/memory/utility_calculator.cpp
//
// Implementation of Utility Calculator

#include "memory/utility_calculator.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <istream>

namespace dpan {

// ============================================================================
// UtilityCalculator::Config
// ============================================================================

bool UtilityCalculator::Config::IsValid() const {
    // Weights must sum to approximately 1.0
    float sum = frequency_weight + recency_weight + association_weight + confidence_weight;
    if (std::abs(sum - 1.0f) > 0.01f) {
        return false;
    }

    // All weights must be non-negative
    if (frequency_weight < 0.0f || recency_weight < 0.0f ||
        association_weight < 0.0f || confidence_weight < 0.0f) {
        return false;
    }

    // Decay constants must be positive
    if (frequency_decay <= 0.0f || recency_decay <= 0.0f) {
        return false;
    }

    // Max access count must be positive
    if (max_access_count <= 0.0f) {
        return false;
    }

    return true;
}

// ============================================================================
// UtilityCalculator
// ============================================================================

UtilityCalculator::UtilityCalculator()
    : config_(Config{}) {
    ValidateWeights();
}

UtilityCalculator::UtilityCalculator(const Config& config)
    : config_(config) {
    ValidateWeights();
}

void UtilityCalculator::ValidateWeights() const {
    if (!config_.IsValid()) {
        throw std::invalid_argument("Invalid UtilityCalculator configuration");
    }
}

float UtilityCalculator::CalculatePatternUtility(
    const PatternNode& pattern,
    const AccessStats& stats,
    const std::vector<AssociationEdge>& associations) const {

    float f_score = CalculateFrequencyScore(stats.access_count);
    float r_score = CalculateRecencyScore(stats.TimeSinceLastAccess());
    float a_score = CalculateAssociationScore(associations);
    float c_score = CalculateConfidenceScore(pattern);

    float utility = config_.frequency_weight * f_score +
                   config_.recency_weight * r_score +
                   config_.association_weight * a_score +
                   config_.confidence_weight * c_score;

    return std::clamp(utility, 0.0f, 1.0f);
}

float UtilityCalculator::CalculateAssociationUtility(
    const AssociationEdge& edge,
    const AccessStats& source_stats,
    const AccessStats& target_stats) const {

    // Association utility is based on:
    // 1. Edge strength (primary factor)
    // 2. Access frequency of both endpoints
    // 3. Recency of both endpoints

    float strength_score = edge.GetStrength();

    // Average frequency score of endpoints
    float freq_source = CalculateFrequencyScore(source_stats.access_count);
    float freq_target = CalculateFrequencyScore(target_stats.access_count);
    float freq_score = (freq_source + freq_target) / 2.0f;

    // Average recency score of endpoints
    float rec_source = CalculateRecencyScore(source_stats.TimeSinceLastAccess());
    float rec_target = CalculateRecencyScore(target_stats.TimeSinceLastAccess());
    float rec_score = (rec_source + rec_target) / 2.0f;

    // Weighted combination
    float utility = 0.5f * strength_score +
                   0.25f * freq_score +
                   0.25f * rec_score;

    return std::clamp(utility, 0.0f, 1.0f);
}

UtilityCalculator::UtilityBreakdown UtilityCalculator::GetUtilityBreakdown(
    const PatternNode& pattern,
    const AccessStats& stats,
    const std::vector<AssociationEdge>& associations) const {

    UtilityBreakdown breakdown;
    breakdown.frequency_score = CalculateFrequencyScore(stats.access_count);
    breakdown.recency_score = CalculateRecencyScore(stats.TimeSinceLastAccess());
    breakdown.association_score = CalculateAssociationScore(associations);
    breakdown.confidence_score = CalculateConfidenceScore(pattern);

    breakdown.total = config_.frequency_weight * breakdown.frequency_score +
                     config_.recency_weight * breakdown.recency_score +
                     config_.association_weight * breakdown.association_score +
                     config_.confidence_weight * breakdown.confidence_score;

    breakdown.total = std::clamp(breakdown.total, 0.0f, 1.0f);

    return breakdown;
}

void UtilityCalculator::SetConfig(const Config& config) {
    config_ = config;
    ValidateWeights();
}

// ============================================================================
// Private: Component Calculations
// ============================================================================

float UtilityCalculator::CalculateFrequencyScore(uint64_t access_count) const {
    // F(p) = 1 - exp(-λ_f × access_count)
    // This saturates at ~1.0 for frequently accessed patterns
    float normalized_count = static_cast<float>(access_count);
    float score = 1.0f - std::exp(-config_.frequency_decay * normalized_count);
    return std::clamp(score, 0.0f, 1.0f);
}

float UtilityCalculator::CalculateRecencyScore(Timestamp::Duration time_since_access) const {
    // R(p) = exp(-λ_r × Δt)
    // Where Δt is in hours
    // Half-life: t_1/2 = ln(2)/λ_r ≈ 14 hours (with default λ_r = 0.05)

    auto hours = std::chrono::duration_cast<std::chrono::hours>(time_since_access);
    float hours_since_access = static_cast<float>(hours.count());

    float score = std::exp(-config_.recency_decay * hours_since_access);
    return std::clamp(score, 0.0f, 1.0f);
}

float UtilityCalculator::CalculateAssociationScore(
    const std::vector<AssociationEdge>& associations) const {
    // A(p) = (Σ strengths) / (num_associations + 1)
    // Average strength of all associations, normalized

    if (associations.empty()) {
        return 0.0f;
    }

    float total_strength = 0.0f;
    for (const auto& edge : associations) {
        total_strength += edge.GetStrength();
    }

    // Average strength, but cap at 1.0
    float avg_strength = total_strength / static_cast<float>(associations.size());
    return std::clamp(avg_strength, 0.0f, 1.0f);
}

float UtilityCalculator::CalculateConfidenceScore(const PatternNode& pattern) const {
    // C(p) = pattern confidence
    // For now, use a default value since PatternNode doesn't have confidence yet
    // In future, this could be based on:
    // - Pattern match quality
    // - Number of successful predictions
    // - Validation metrics

    // Default confidence of 0.5 for all patterns
    // TODO: Integrate with pattern quality metrics from Phase 1
    return 0.5f;
}

// ============================================================================
// AccessStats
// ============================================================================

void AccessStats::RecordAccess(Timestamp timestamp) {
    if (access_count == 0) {
        // First access
        creation_time = timestamp;
        last_access = timestamp;
        access_count = 1;
        avg_access_interval = 0.0f;
        return;
    }

    // Update exponential moving average of access interval
    auto interval = timestamp - last_access;
    float interval_seconds = std::chrono::duration<float>(interval).count();

    if (access_count == 1) {
        avg_access_interval = interval_seconds;
    } else {
        // EMA with alpha = 0.3 (more weight to recent intervals)
        const float alpha = 0.3f;
        avg_access_interval = alpha * interval_seconds + (1.0f - alpha) * avg_access_interval;
    }

    last_access = timestamp;
    access_count++;
}

Timestamp::Duration AccessStats::TimeSinceLastAccess() const {
    return Timestamp::Now() - last_access;
}

Timestamp::Duration AccessStats::Age() const {
    return Timestamp::Now() - creation_time;
}

void AccessStats::Serialize(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&access_count), sizeof(access_count));

    // Store as microseconds (Timestamp uses microseconds internally)
    int64_t last_access_micros = last_access.ToMicros();
    out.write(reinterpret_cast<const char*>(&last_access_micros), sizeof(last_access_micros));

    int64_t creation_micros = creation_time.ToMicros();
    out.write(reinterpret_cast<const char*>(&creation_micros), sizeof(creation_micros));

    out.write(reinterpret_cast<const char*>(&avg_access_interval), sizeof(avg_access_interval));
}

AccessStats AccessStats::Deserialize(std::istream& in) {
    AccessStats stats;

    in.read(reinterpret_cast<char*>(&stats.access_count), sizeof(stats.access_count));

    int64_t last_access_micros;
    in.read(reinterpret_cast<char*>(&last_access_micros), sizeof(last_access_micros));
    stats.last_access = Timestamp::FromMicros(last_access_micros);

    int64_t creation_micros;
    in.read(reinterpret_cast<char*>(&creation_micros), sizeof(creation_micros));
    stats.creation_time = Timestamp::FromMicros(creation_micros);

    in.read(reinterpret_cast<char*>(&stats.avg_access_interval), sizeof(stats.avg_access_interval));

    return stats;
}

// ============================================================================
// AccessTracker
// ============================================================================

void AccessTracker::RecordPatternAccess(PatternID pattern, Timestamp timestamp) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto& stats = pattern_stats_[pattern];
    stats.RecordAccess(timestamp);
}

void AccessTracker::RecordAssociationAccess(PatternID source, PatternID target, Timestamp timestamp) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    auto& stats = association_stats_[key];
    stats.RecordAccess(timestamp);
}

const AccessStats* AccessTracker::GetPatternStats(PatternID pattern) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto it = pattern_stats_.find(pattern);
    if (it == pattern_stats_.end()) {
        return nullptr;
    }

    return &it->second;
}

const AccessStats* AccessTracker::GetAssociationStats(PatternID source, PatternID target) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    auto it = association_stats_.find(key);
    if (it == association_stats_.end()) {
        return nullptr;
    }

    return &it->second;
}

size_t AccessTracker::PruneOldStats(Timestamp cutoff_time) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    size_t removed = 0;

    // Prune pattern stats
    for (auto it = pattern_stats_.begin(); it != pattern_stats_.end();) {
        if (it->second.last_access < cutoff_time) {
            it = pattern_stats_.erase(it);
            removed++;
        } else {
            ++it;
        }
    }

    // Prune association stats
    for (auto it = association_stats_.begin(); it != association_stats_.end();) {
        if (it->second.last_access < cutoff_time) {
            it = association_stats_.erase(it);
            removed++;
        } else {
            ++it;
        }
    }

    return removed;
}

void AccessTracker::Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    pattern_stats_.clear();
    association_stats_.clear();
}

size_t AccessTracker::GetTrackedPatternCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return pattern_stats_.size();
}

size_t AccessTracker::GetTrackedAssociationCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return association_stats_.size();
}

} // namespace dpan
