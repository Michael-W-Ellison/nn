// File: src/learning/temporal_attention_head.cpp
//
// Implementation of TemporalAttentionHead

#include "learning/temporal_attention_head.hpp"
#include "learning/attention_utils.hpp"
#include "storage/pattern_database.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace dpan {
namespace attention {

TemporalAttentionHead::TemporalAttentionHead(const TemporalAttentionConfig& config)
    : config_(config)
    , pattern_db_(nullptr)
    , attention_computations_(0)
    , temporal_computations_(0)
    , cache_hits_(0)
    , cache_misses_(0)
{
    if (!config_.Validate()) {
        throw std::invalid_argument("Invalid TemporalAttentionConfig");
    }

    // Initialize base_config_ for interface compatibility
    base_config_.temperature = config_.temperature;
    base_config_.enable_caching = config_.enable_caching;
    base_config_.cache_size = config_.cache_size;
    base_config_.debug_logging = config_.debug_logging;
}

// ============================================================================
// AttentionMechanism Interface Implementation
// ============================================================================

std::map<PatternID, float> TemporalAttentionHead::ComputeAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    ++attention_computations_;

    // Handle empty candidates
    if (candidates.empty()) {
        LogDebug("TemporalAttentionHead: No candidates provided");
        return {};
    }

    // Handle single candidate
    if (candidates.size() == 1) {
        LogDebug("TemporalAttentionHead: Single candidate, returning weight 1.0");
        return {{candidates[0], 1.0f}};
    }

    LogDebug("TemporalAttentionHead: Computing temporal attention for " +
             std::to_string(candidates.size()) + " candidates");

    // Compute temporal scores
    auto temporal_scores = ComputeTemporalScores(candidates);

    // Normalize scores to attention weights
    auto normalized_weights = NormalizeScores(temporal_scores);

    // Build result map
    std::map<PatternID, float> weights;
    for (size_t i = 0; i < candidates.size(); ++i) {
        weights[candidates[i]] = normalized_weights[i];
    }

    return weights;
}

std::vector<AttentionScore> TemporalAttentionHead::ComputeDetailedAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    // Compute basic attention weights
    auto weights = ComputeAttention(query, candidates, context);

    // Compute raw temporal scores for detailed view
    auto temporal_scores = ComputeTemporalScores(candidates);

    // Convert to AttentionScore vector
    std::vector<AttentionScore> scores;
    scores.reserve(weights.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        AttentionScore score;
        score.pattern_id = candidates[i];
        score.weight = weights[candidates[i]];
        score.raw_score = temporal_scores[i];

        // Fill in component breakdown
        score.components.temporal_score = temporal_scores[i];
        // Other components are zero for pure temporal attention
        score.components.semantic_similarity = 0.0f;
        score.components.context_similarity = 0.0f;
        score.components.importance_score = 0.0f;
        score.components.structural_score = 0.0f;

        scores.push_back(score);
    }

    // Sort by weight descending
    std::sort(scores.begin(), scores.end(),
              [](const AttentionScore& a, const AttentionScore& b) {
                  return a.weight > b.weight;
              });

    return scores;
}

std::vector<std::pair<PatternID, float>> TemporalAttentionHead::ApplyAttention(
    PatternID query,
    const std::vector<PatternID>& predictions,
    const ContextVector& context) {

    // Compute attention weights
    auto attention_weights = ComputeAttention(query, predictions, context);

    // Convert to sorted vector
    std::vector<std::pair<PatternID, float>> result;
    result.reserve(attention_weights.size());

    for (const auto& [pattern_id, weight] : attention_weights) {
        result.emplace_back(pattern_id, weight);
    }

    // Sort by weight descending (most recent first)
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return result;
}

void TemporalAttentionHead::SetPatternDatabase(PatternDatabase* db) {
    pattern_db_ = db;
    LogDebug("Pattern database set");
}

const AttentionConfig& TemporalAttentionHead::GetConfig() const {
    return base_config_;
}

void TemporalAttentionHead::SetConfig(const AttentionConfig& config) {
    if (!config.Validate()) {
        throw std::invalid_argument("Invalid AttentionConfig");
    }

    // Update both configs
    base_config_ = config;
    config_.temperature = config.temperature;
    config_.enable_caching = config.enable_caching;
    config_.cache_size = config.cache_size;
    config_.debug_logging = config.debug_logging;

    // Clear cache if caching was disabled
    if (!config_.enable_caching) {
        ClearCache();
    }

    LogDebug("Configuration updated");
}

void TemporalAttentionHead::ClearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    temporal_cache_.clear();
    LogDebug("Cache cleared");
}

std::map<std::string, float> TemporalAttentionHead::GetStatistics() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    std::map<std::string, float> stats;
    stats["attention_computations"] = static_cast<float>(attention_computations_);
    stats["temporal_computations"] = static_cast<float>(temporal_computations_);
    stats["cache_hits"] = static_cast<float>(cache_hits_);
    stats["cache_misses"] = static_cast<float>(cache_misses_);
    stats["cache_size"] = static_cast<float>(temporal_cache_.size());

    // Calculate cache hit rate
    float total_lookups = static_cast<float>(cache_hits_ + cache_misses_);
    stats["cache_hit_rate"] = total_lookups > 0.0f
        ? static_cast<float>(cache_hits_) / total_lookups
        : 0.0f;

    return stats;
}

// ============================================================================
// Temporal Attention Specific Methods
// ============================================================================

const TemporalAttentionConfig& TemporalAttentionHead::GetTemporalConfig() const {
    return config_;
}

void TemporalAttentionHead::SetTemporalConfig(const TemporalAttentionConfig& config) {
    if (!config.Validate()) {
        throw std::invalid_argument("Invalid TemporalAttentionConfig");
    }

    config_ = config;

    // Update base config as well
    base_config_.temperature = config_.temperature;
    base_config_.enable_caching = config_.enable_caching;
    base_config_.cache_size = config_.cache_size;
    base_config_.debug_logging = config_.debug_logging;

    LogDebug("Temporal attention configuration updated");
}

Timestamp TemporalAttentionHead::GetCurrentTime() {
    return Timestamp::Now();
}

// ============================================================================
// Protected Helper Methods
// ============================================================================

std::vector<float> TemporalAttentionHead::ComputeTemporalScores(
    const std::vector<PatternID>& candidates) const {

    std::vector<float> scores;
    scores.reserve(candidates.size());

    if (!pattern_db_) {
        // Return uniform scores if we can't compute temporal information
        scores.resize(candidates.size(), 1.0f);
        return scores;
    }

    Timestamp current_time = GetCurrentTime();

    // Compute temporal score for each candidate
    for (const auto& candidate_id : candidates) {
        float score = 0.0f;

        // Check cache first if caching is enabled
        if (config_.enable_caching) {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto it = temporal_cache_.find(candidate_id);

            if (it != temporal_cache_.end()) {
                // Check if cached score is still valid (within 100ms)
                uint64_t cached_time = it->second.first;
                uint64_t current_micros = static_cast<uint64_t>(current_time.ToMicros());

                if (current_micros - cached_time < 100000) {  // 100ms threshold
                    score = it->second.second;
                    ++cache_hits_;
                } else {
                    ++cache_misses_;
                    // Recompute score
                    float time_delta_ms = GetTimeSinceLastAccess(candidate_id);
                    if (time_delta_ms >= 0.0f) {
                        score = std::exp(-time_delta_ms / config_.decay_constant_ms);
                        ++temporal_computations_;
                    }

                    // Update cache
                    temporal_cache_[candidate_id] = std::make_pair(current_micros, score);
                }
            } else {
                ++cache_misses_;

                // Compute temporal score
                float time_delta_ms = GetTimeSinceLastAccess(candidate_id);
                if (time_delta_ms >= 0.0f) {
                    score = std::exp(-time_delta_ms / config_.decay_constant_ms);
                    ++temporal_computations_;
                }

                // Cache the result (with LRU eviction if needed)
                if (temporal_cache_.size() >= config_.cache_size) {
                    // Simple eviction: remove first element
                    temporal_cache_.erase(temporal_cache_.begin());
                }
                temporal_cache_[candidate_id] = std::make_pair(static_cast<uint64_t>(current_time.ToMicros()), score);
            }
        } else {
            // No caching - always compute
            float time_delta_ms = GetTimeSinceLastAccess(candidate_id);
            if (time_delta_ms >= 0.0f) {
                score = std::exp(-time_delta_ms / config_.decay_constant_ms);
                ++temporal_computations_;
            }
        }

        scores.push_back(score);
    }

    return scores;
}

std::vector<float> TemporalAttentionHead::NormalizeScores(
    const std::vector<float>& scores) const {

    if (scores.empty()) {
        return {};
    }

    // Apply temperature scaling and softmax
    return Softmax(scores, config_.temperature);
}

float TemporalAttentionHead::GetTimeSinceLastAccess(PatternID pattern_id) const {
    if (!pattern_db_) {
        return -1.0f;
    }

    // Retrieve pattern from database
    auto pattern_opt = pattern_db_->Retrieve(pattern_id);
    if (!pattern_opt) {
        LogDebug("WARNING: Pattern not found: " + pattern_id.ToString());
        return -1.0f;
    }

    const auto& pattern = *pattern_opt;

    // Get last accessed time
    Timestamp last_accessed = pattern.GetLastAccessed();

    // If never accessed, use creation time
    if (last_accessed.ToMicros() == 0) {
        last_accessed = pattern.GetCreationTime();
    }

    // Calculate time delta
    Timestamp current_time = GetCurrentTime();
    auto delta = current_time - last_accessed;

    // Convert to milliseconds (delta is in microseconds)
    float delta_ms = static_cast<float>(delta.count()) / 1000.0f;

    return delta_ms;
}

void TemporalAttentionHead::LogDebug(const std::string& message) const {
    if (config_.debug_logging) {
        std::cout << "[TemporalAttentionHead] " << message << std::endl;
    }
}

} // namespace attention
} // namespace dpan
