// File: src/learning/structural_attention_head.cpp
//
// Implementation of StructuralAttentionHead

#include "learning/structural_attention_head.hpp"
#include "learning/attention_utils.hpp"
#include "storage/pattern_database.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <set>

namespace dpan {
namespace attention {

StructuralAttentionHead::StructuralAttentionHead(
    const StructuralAttentionConfig& config)
    : config_(config)
    , pattern_db_(nullptr)
    , attention_computations_(0)
    , structural_computations_(0)
    , cache_hits_(0)
    , cache_misses_(0)
{
    if (!config_.Validate()) {
        throw std::invalid_argument("Invalid StructuralAttentionConfig");
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

std::map<PatternID, float> StructuralAttentionHead::ComputeAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& /* context */) {

    ++attention_computations_;

    // Handle empty candidates
    if (candidates.empty()) {
        LogDebug("StructuralAttentionHead: No candidates provided");
        return {};
    }

    // Handle single candidate
    if (candidates.size() == 1) {
        LogDebug("StructuralAttentionHead: Single candidate, returning weight 1.0");
        return {{candidates[0], 1.0f}};
    }

    LogDebug("StructuralAttentionHead: Computing structural attention for " +
             std::to_string(candidates.size()) + " candidates");

    // Compute structural scores
    auto structural_scores = ComputeStructuralScores(query, candidates);

    // Normalize scores to attention weights
    auto normalized_weights = NormalizeScores(structural_scores);

    // Build result map
    std::map<PatternID, float> weights;
    for (size_t i = 0; i < candidates.size(); ++i) {
        weights[candidates[i]] = normalized_weights[i];
    }

    return weights;
}

std::vector<AttentionScore> StructuralAttentionHead::ComputeDetailedAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    // Compute basic attention weights
    auto weights = ComputeAttention(query, candidates, context);

    // Compute raw structural scores for detailed view
    auto structural_scores = ComputeStructuralScores(query, candidates);

    // Convert to AttentionScore vector
    std::vector<AttentionScore> scores;
    scores.reserve(weights.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        AttentionScore score;
        score.pattern_id = candidates[i];
        score.weight = weights[candidates[i]];
        score.raw_score = structural_scores[i];

        // Fill in component breakdown
        score.components.structural_score = structural_scores[i];
        // Other components are zero for pure structural attention
        score.components.semantic_similarity = 0.0f;
        score.components.context_similarity = 0.0f;
        score.components.importance_score = 0.0f;
        score.components.temporal_score = 0.0f;

        scores.push_back(score);
    }

    // Sort by weight descending
    std::sort(scores.begin(), scores.end(),
              [](const AttentionScore& a, const AttentionScore& b) {
                  return a.weight > b.weight;
              });

    return scores;
}

std::vector<std::pair<PatternID, float>> StructuralAttentionHead::ApplyAttention(
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

    // Sort by weight descending (most structurally similar first)
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return result;
}

void StructuralAttentionHead::SetPatternDatabase(PatternDatabase* db) {
    pattern_db_ = db;
    LogDebug("Pattern database set");
}

const AttentionConfig& StructuralAttentionHead::GetConfig() const {
    return base_config_;
}

void StructuralAttentionHead::SetConfig(const AttentionConfig& config) {
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

void StructuralAttentionHead::ClearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    structural_cache_.clear();
    LogDebug("Cache cleared");
}

std::map<std::string, float> StructuralAttentionHead::GetStatistics() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    std::map<std::string, float> stats;
    stats["attention_computations"] = static_cast<float>(attention_computations_);
    stats["structural_computations"] = static_cast<float>(structural_computations_);
    stats["cache_hits"] = static_cast<float>(cache_hits_);
    stats["cache_misses"] = static_cast<float>(cache_misses_);
    stats["cache_size"] = static_cast<float>(structural_cache_.size());

    // Calculate cache hit rate
    float total_lookups = static_cast<float>(cache_hits_ + cache_misses_);
    stats["cache_hit_rate"] = total_lookups > 0.0f
        ? static_cast<float>(cache_hits_) / total_lookups
        : 0.0f;

    return stats;
}

// ============================================================================
// Structural Attention Specific Methods
// ============================================================================

const StructuralAttentionConfig& StructuralAttentionHead::GetStructuralConfig() const {
    return config_;
}

void StructuralAttentionHead::SetStructuralConfig(const StructuralAttentionConfig& config) {
    if (!config.Validate()) {
        throw std::invalid_argument("Invalid StructuralAttentionConfig");
    }

    config_ = config;

    // Update base config as well
    base_config_.temperature = config_.temperature;
    base_config_.enable_caching = config_.enable_caching;
    base_config_.cache_size = config_.cache_size;
    base_config_.debug_logging = config_.debug_logging;

    LogDebug("Structural attention configuration updated");
}

// ============================================================================
// Protected Helper Methods
// ============================================================================

std::vector<float> StructuralAttentionHead::ComputeStructuralScores(
    PatternID query,
    const std::vector<PatternID>& candidates) const {

    std::vector<float> scores;
    scores.reserve(candidates.size());

    if (!pattern_db_) {
        // Return uniform scores if we can't compute structural information
        scores.resize(candidates.size(), 1.0f);
        return scores;
    }

    // Get query pattern and its sub-patterns
    auto query_pattern_opt = pattern_db_->Retrieve(query);
    if (!query_pattern_opt) {
        LogDebug("WARNING: Query pattern not found: " + query.ToString());
        scores.resize(candidates.size(), 1.0f / candidates.size());
        return scores;
    }

    const auto& query_pattern = *query_pattern_opt;
    auto query_subpatterns = query_pattern.GetSubPatterns();
    bool query_is_composite = !query_subpatterns.empty();

    // Compute structural score for each candidate
    for (const auto& candidate_id : candidates) {
        float score = 0.0f;

        // Check cache first if caching is enabled
        if (config_.enable_caching) {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto cache_key = std::make_pair(query, candidate_id);
            auto it = structural_cache_.find(cache_key);

            if (it != structural_cache_.end()) {
                score = it->second;
                ++cache_hits_;
            } else {
                ++cache_misses_;

                // Compute structural score
                auto candidate_pattern_opt = pattern_db_->Retrieve(candidate_id);
                if (candidate_pattern_opt) {
                    auto candidate_subpatterns = candidate_pattern_opt->GetSubPatterns();
                    bool candidate_is_composite = !candidate_subpatterns.empty();

                    // Handle different pattern type combinations
                    if (query_is_composite && candidate_is_composite) {
                        // Both composite: compute full structural similarity
                        float jaccard = ComputeJaccardSimilarity(
                            query_subpatterns, candidate_subpatterns);
                        float size_sim = ComputeSizeSimilarity(
                            query_subpatterns.size(), candidate_subpatterns.size());

                        score = config_.jaccard_weight * jaccard +
                                config_.size_weight * size_sim;
                    } else if (!query_is_composite && !candidate_is_composite) {
                        // Both atomic: perfect structural match
                        score = 1.0f;
                    } else {
                        // One atomic, one composite: apply penalty
                        score = config_.atomic_penalty;
                    }

                    ++structural_computations_;
                }

                // Cache the result (with LRU eviction if needed)
                if (structural_cache_.size() >= config_.cache_size) {
                    // Simple eviction: remove first element
                    structural_cache_.erase(structural_cache_.begin());
                }
                structural_cache_[cache_key] = score;
            }
        } else {
            // No caching - always compute
            auto candidate_pattern_opt = pattern_db_->Retrieve(candidate_id);
            if (candidate_pattern_opt) {
                auto candidate_subpatterns = candidate_pattern_opt->GetSubPatterns();
                bool candidate_is_composite = !candidate_subpatterns.empty();

                // Handle different pattern type combinations
                if (query_is_composite && candidate_is_composite) {
                    // Both composite: compute full structural similarity
                    float jaccard = ComputeJaccardSimilarity(
                        query_subpatterns, candidate_subpatterns);
                    float size_sim = ComputeSizeSimilarity(
                        query_subpatterns.size(), candidate_subpatterns.size());

                    score = config_.jaccard_weight * jaccard +
                            config_.size_weight * size_sim;
                } else if (!query_is_composite && !candidate_is_composite) {
                    // Both atomic: perfect structural match
                    score = 1.0f;
                } else {
                    // One atomic, one composite: apply penalty
                    score = config_.atomic_penalty;
                }

                ++structural_computations_;
            }
        }

        // Apply similarity threshold
        if (score < config_.similarity_threshold) {
            score = 0.0f;
        }

        scores.push_back(score);
    }

    return scores;
}

float StructuralAttentionHead::ComputeJaccardSimilarity(
    const std::vector<PatternID>& query_subpatterns,
    const std::vector<PatternID>& candidate_subpatterns) const {

    if (query_subpatterns.empty() && candidate_subpatterns.empty()) {
        return 1.0f;  // Both empty: perfect similarity
    }

    if (query_subpatterns.empty() || candidate_subpatterns.empty()) {
        return 0.0f;  // One empty: no similarity
    }

    // Convert to sets for efficient intersection/union
    std::set<PatternID> query_set(query_subpatterns.begin(), query_subpatterns.end());
    std::set<PatternID> candidate_set(candidate_subpatterns.begin(), candidate_subpatterns.end());

    // Compute intersection
    std::vector<PatternID> intersection;
    std::set_intersection(
        query_set.begin(), query_set.end(),
        candidate_set.begin(), candidate_set.end(),
        std::back_inserter(intersection));

    // Compute union
    std::vector<PatternID> union_vec;
    std::set_union(
        query_set.begin(), query_set.end(),
        candidate_set.begin(), candidate_set.end(),
        std::back_inserter(union_vec));

    // Jaccard similarity = |A ∩ B| / |A ∪ B|
    float jaccard = static_cast<float>(intersection.size()) /
                    static_cast<float>(union_vec.size());

    return jaccard;
}

float StructuralAttentionHead::ComputeSizeSimilarity(
    size_t query_size,
    size_t candidate_size) const {

    if (query_size == 0 && candidate_size == 0) {
        return 1.0f;  // Both empty: perfect similarity
    }

    size_t max_size = std::max(query_size, candidate_size);
    if (max_size == 0) {
        return 1.0f;
    }

    // Size similarity = 1 - |size_diff| / max_size
    size_t size_diff = (query_size > candidate_size)
        ? (query_size - candidate_size)
        : (candidate_size - query_size);

    float similarity = 1.0f - (static_cast<float>(size_diff) /
                               static_cast<float>(max_size));

    return similarity;
}

std::vector<float> StructuralAttentionHead::NormalizeScores(
    const std::vector<float>& scores) const {

    if (scores.empty()) {
        return {};
    }

    // Apply temperature scaling and softmax
    return Softmax(scores, config_.temperature);
}

void StructuralAttentionHead::LogDebug(const std::string& message) const {
    if (config_.debug_logging) {
        std::cout << "[StructuralAttentionHead] " << message << std::endl;
    }
}

} // namespace attention
} // namespace dpan
