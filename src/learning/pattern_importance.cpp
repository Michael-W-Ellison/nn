// File: src/learning/pattern_importance.cpp
//
// Implementation of PatternImportanceCalculator
//
// Key implementation details:
// - Log scaling for frequency to prevent outlier dominance
// - Thread-safe success rate tracking with mutex
// - Decay factor for recency bias in success rates
// - Graceful handling of missing patterns

#include "learning/pattern_importance.hpp"
#include "core/pattern_node.hpp"
#include <algorithm>
#include <cmath>

namespace dpan {
namespace attention {

namespace {
    // Constants
    constexpr float kDefaultScore = 0.5f;  // Default score for unknown patterns
    constexpr float kEpsilon = 1e-10f;     // Small value to prevent log(0)
    constexpr uint32_t kDefaultMaxAccessCount = 10000;  // Default max for normalization
}

PatternImportanceCalculator::PatternImportanceCalculator(
    PatternDatabase* db,
    const ImportanceWeights& weights)
    : pattern_db_(db)
    , weights_(weights)
    , max_access_count_(kDefaultMaxAccessCount)
    , importance_calculations_(0)
    , success_recordings_(0)
{
    // Validate and normalize weights if needed
    if (!weights_.Validate()) {
        weights_.Normalize();
    }
}

// ============================================================================
// Individual Scoring Methods
// ============================================================================

float PatternImportanceCalculator::ComputeFrequencyScore(PatternID pattern_id) const {
    if (!pattern_db_) {
        return kDefaultScore;
    }

    auto pattern_opt = pattern_db_->Retrieve(pattern_id);
    if (!pattern_opt) {
        return 0.0f;  // Pattern not found
    }

    uint32_t access_count = pattern_opt->GetAccessCount();

    // Log scaling: score = log(1 + count) / log(1 + max_count)
    // This prevents patterns with very high access counts from dominating
    float log_count = std::log(1.0f + static_cast<float>(access_count));
    float log_max = std::log(1.0f + static_cast<float>(max_access_count_));

    if (log_max < kEpsilon) {
        return 0.0f;
    }

    float score = log_count / log_max;

    // Clamp to [0, 1]
    return std::min(std::max(score, 0.0f), 1.0f);
}

float PatternImportanceCalculator::ComputeConfidenceScore(PatternID pattern_id) const {
    if (!pattern_db_) {
        return kDefaultScore;
    }

    auto pattern_opt = pattern_db_->Retrieve(pattern_id);
    if (!pattern_opt) {
        return kDefaultScore;  // Pattern not found, return neutral score
    }

    float confidence = pattern_opt->GetConfidenceScore();

    // Confidence should already be in [0, 1], but clamp to be safe
    return std::min(std::max(confidence, 0.0f), 1.0f);
}

float PatternImportanceCalculator::ComputeAssociationScore(PatternID pattern_id) const {
    // Placeholder implementation until AssociationSystem is integrated
    // In a full implementation, this would:
    // 1. Query AssociationSystem for all associations involving this pattern
    // 2. Count associations with strength > 0.5
    // 3. Normalize using log scaling similar to frequency
    //
    // For now, return neutral score
    (void)pattern_id;  // Unused parameter
    return kDefaultScore;
}

float PatternImportanceCalculator::ComputeSuccessRateScore(PatternID pattern_id) const {
    std::lock_guard<std::mutex> lock(success_mutex_);

    auto it = success_rates_.find(pattern_id);
    if (it == success_rates_.end()) {
        return kDefaultScore;  // No history, return neutral score
    }

    return it->second.GetRate();
}

// ============================================================================
// Combined Importance Scoring
// ============================================================================

float PatternImportanceCalculator::ComputeImportance(PatternID pattern_id) const {
    ++importance_calculations_;

    // Compute individual scores
    float freq_score = ComputeFrequencyScore(pattern_id);
    float conf_score = ComputeConfidenceScore(pattern_id);
    float assoc_score = ComputeAssociationScore(pattern_id);
    float success_score = ComputeSuccessRateScore(pattern_id);

    // Weighted combination
    float importance =
        weights_.frequency * freq_score +
        weights_.confidence * conf_score +
        weights_.association * assoc_score +
        weights_.success_rate * success_score;

    // Should already be in [0, 1] due to normalized weights and scores,
    // but clamp to be absolutely sure
    return std::min(std::max(importance, 0.0f), 1.0f);
}

std::map<PatternID, float> PatternImportanceCalculator::ComputeImportanceBatch(
    const std::vector<PatternID>& pattern_ids) const {

    std::map<PatternID, float> results;

    for (const auto& pattern_id : pattern_ids) {
        results[pattern_id] = ComputeImportance(pattern_id);
    }

    return results;
}

// ============================================================================
// Success Rate Tracking
// ============================================================================

void PatternImportanceCalculator::RecordPrediction(PatternID pattern_id, bool success) {
    std::lock_guard<std::mutex> lock(success_mutex_);

    ++success_recordings_;

    // Get or create success rate data
    auto& data = success_rates_[pattern_id];
    data.RecordPrediction(success);
}

SuccessRateData PatternImportanceCalculator::GetSuccessRateData(PatternID pattern_id) const {
    std::lock_guard<std::mutex> lock(success_mutex_);

    auto it = success_rates_.find(pattern_id);
    if (it == success_rates_.end()) {
        return SuccessRateData();  // Return default data
    }

    return it->second;
}

void PatternImportanceCalculator::ClearSuccessRateData() {
    std::lock_guard<std::mutex> lock(success_mutex_);
    success_rates_.clear();
}

// ============================================================================
// Configuration
// ============================================================================

void PatternImportanceCalculator::SetWeights(const ImportanceWeights& weights) {
    weights_ = weights;

    // Validate and normalize if needed
    if (!weights_.Validate()) {
        weights_.Normalize();
    }
}

const ImportanceWeights& PatternImportanceCalculator::GetWeights() const {
    return weights_;
}

void PatternImportanceCalculator::SetPatternDatabase(PatternDatabase* db) {
    pattern_db_ = db;
}

void PatternImportanceCalculator::SetMaxAccessCount(uint32_t max_count) {
    max_access_count_ = max_count;
}

std::map<std::string, float> PatternImportanceCalculator::GetStatistics() const {
    std::map<std::string, float> stats;

    stats["importance_calculations"] = static_cast<float>(importance_calculations_);
    stats["success_recordings"] = static_cast<float>(success_recordings_);

    {
        std::lock_guard<std::mutex> lock(success_mutex_);
        stats["tracked_patterns"] = static_cast<float>(success_rates_.size());

        // Calculate average success rate
        if (!success_rates_.empty()) {
            float total_rate = 0.0f;
            for (const auto& [id, data] : success_rates_) {
                total_rate += data.GetRate();
            }
            stats["avg_success_rate"] = total_rate / success_rates_.size();
        } else {
            stats["avg_success_rate"] = 0.0f;
        }
    }

    return stats;
}

} // namespace attention
} // namespace dpan
