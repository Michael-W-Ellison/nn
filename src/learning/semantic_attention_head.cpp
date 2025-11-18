// File: src/learning/semantic_attention_head.cpp
//
// Implementation of SemanticAttentionHead

#include "learning/semantic_attention_head.hpp"
#include "learning/attention_utils.hpp"
#include "storage/pattern_database.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace dpan {
namespace attention {

SemanticAttentionHead::SemanticAttentionHead(
    const SemanticAttentionConfig& config,
    std::shared_ptr<SimilarityMetric> similarity_metric)
    : config_(config)
    , similarity_metric_(similarity_metric)
    , pattern_db_(nullptr)
    , attention_computations_(0)
    , similarity_computations_(0)
    , cache_hits_(0)
    , cache_misses_(0)
{
    if (!config_.Validate()) {
        throw std::invalid_argument("Invalid SemanticAttentionConfig");
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

std::map<PatternID, float> SemanticAttentionHead::ComputeAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    ++attention_computations_;

    // Handle empty candidates
    if (candidates.empty()) {
        LogDebug("SemanticAttentionHead: No candidates provided");
        return {};
    }

    // Handle single candidate
    if (candidates.size() == 1) {
        LogDebug("SemanticAttentionHead: Single candidate, returning weight 1.0");
        return {{candidates[0], 1.0f}};
    }

    // Verify we have a similarity metric
    if (!similarity_metric_) {
        LogDebug("WARNING: No similarity metric configured, using uniform weights");
        std::map<PatternID, float> uniform_weights;
        float weight = 1.0f / candidates.size();
        for (const auto& candidate : candidates) {
            uniform_weights[candidate] = weight;
        }
        return uniform_weights;
    }

    LogDebug("SemanticAttentionHead: Computing content-based attention for " +
             std::to_string(candidates.size()) + " candidates");

    // Compute similarity scores
    auto similarity_scores = ComputeSimilarityScores(query, candidates);

    // Apply threshold if configured
    if (config_.similarity_threshold > 0.0f) {
        for (auto& score : similarity_scores) {
            if (score < config_.similarity_threshold) {
                score = 0.0f;
            }
        }
    }

    // Normalize scores to attention weights
    auto normalized_weights = NormalizeScores(similarity_scores);

    // Build result map
    std::map<PatternID, float> weights;
    for (size_t i = 0; i < candidates.size(); ++i) {
        weights[candidates[i]] = normalized_weights[i];
    }

    return weights;
}

std::vector<AttentionScore> SemanticAttentionHead::ComputeDetailedAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    // Compute basic attention weights
    auto weights = ComputeAttention(query, candidates, context);

    // Compute raw similarity scores for detailed view
    auto similarity_scores = ComputeSimilarityScores(query, candidates);

    // Convert to AttentionScore vector
    std::vector<AttentionScore> scores;
    scores.reserve(weights.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        AttentionScore score;
        score.pattern_id = candidates[i];
        score.weight = weights[candidates[i]];
        score.raw_score = similarity_scores[i];

        // Fill in component breakdown
        score.components.semantic_similarity = similarity_scores[i];
        // Other components are zero for pure semantic attention
        score.components.context_similarity = 0.0f;
        score.components.importance_score = 0.0f;
        score.components.temporal_score = 0.0f;
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

std::vector<std::pair<PatternID, float>> SemanticAttentionHead::ApplyAttention(
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

    // Sort by weight descending
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return result;
}

void SemanticAttentionHead::SetPatternDatabase(PatternDatabase* db) {
    pattern_db_ = db;
    LogDebug("Pattern database set");
}

const AttentionConfig& SemanticAttentionHead::GetConfig() const {
    return base_config_;
}

void SemanticAttentionHead::SetConfig(const AttentionConfig& config) {
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

void SemanticAttentionHead::ClearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    similarity_cache_.clear();
    LogDebug("Cache cleared");
}

std::map<std::string, float> SemanticAttentionHead::GetStatistics() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    std::map<std::string, float> stats;
    stats["attention_computations"] = static_cast<float>(attention_computations_);
    stats["similarity_computations"] = static_cast<float>(similarity_computations_);
    stats["cache_hits"] = static_cast<float>(cache_hits_);
    stats["cache_misses"] = static_cast<float>(cache_misses_);
    stats["cache_size"] = static_cast<float>(similarity_cache_.size());

    // Calculate cache hit rate
    float total_lookups = static_cast<float>(cache_hits_ + cache_misses_);
    stats["cache_hit_rate"] = total_lookups > 0.0f
        ? static_cast<float>(cache_hits_) / total_lookups
        : 0.0f;

    return stats;
}

// ============================================================================
// Semantic Attention Specific Methods
// ============================================================================

void SemanticAttentionHead::SetSimilarityMetric(
    std::shared_ptr<SimilarityMetric> metric) {
    similarity_metric_ = metric;
    // Clear cache since similarity computation has changed
    ClearCache();
    LogDebug("Similarity metric updated");
}

std::shared_ptr<SimilarityMetric> SemanticAttentionHead::GetSimilarityMetric() const {
    return similarity_metric_;
}

const SemanticAttentionConfig& SemanticAttentionHead::GetSemanticConfig() const {
    return config_;
}

void SemanticAttentionHead::SetSemanticConfig(const SemanticAttentionConfig& config) {
    if (!config.Validate()) {
        throw std::invalid_argument("Invalid SemanticAttentionConfig");
    }

    config_ = config;

    // Update base config as well
    base_config_.temperature = config_.temperature;
    base_config_.enable_caching = config_.enable_caching;
    base_config_.cache_size = config_.cache_size;
    base_config_.debug_logging = config_.debug_logging;

    LogDebug("Semantic attention configuration updated");
}

// ============================================================================
// Protected Helper Methods
// ============================================================================

std::vector<float> SemanticAttentionHead::ComputeSimilarityScores(
    PatternID query,
    const std::vector<PatternID>& candidates) const {

    std::vector<float> scores;
    scores.reserve(candidates.size());

    if (!pattern_db_ || !similarity_metric_) {
        // Return uniform scores if we can't compute similarity
        scores.resize(candidates.size(), 1.0f / candidates.size());
        return scores;
    }

    // Get query pattern
    auto query_pattern_opt = pattern_db_->Retrieve(query);
    if (!query_pattern_opt) {
        LogDebug("WARNING: Query pattern not found");
        scores.resize(candidates.size(), 1.0f / candidates.size());
        return scores;
    }

    const auto& query_pattern = *query_pattern_opt;
    auto query_data = query_pattern.GetData();

    // Compute similarity for each candidate
    for (const auto& candidate_id : candidates) {
        float similarity = 0.0f;

        // Check cache first if caching is enabled
        if (config_.enable_caching) {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto cache_key = std::make_pair(query, candidate_id);
            auto it = similarity_cache_.find(cache_key);

            if (it != similarity_cache_.end()) {
                similarity = it->second;
                ++cache_hits_;
            } else {
                ++cache_misses_;

                // Compute similarity
                auto candidate_pattern_opt = pattern_db_->Retrieve(candidate_id);
                if (candidate_pattern_opt) {
                    auto candidate_data = candidate_pattern_opt->GetData();
                    similarity = similarity_metric_->Compute(query_data, candidate_data);
                    ++similarity_computations_;

                    // Cache the result (with LRU eviction if needed)
                    if (similarity_cache_.size() >= config_.cache_size) {
                        // Simple eviction: remove first element (could be improved with LRU)
                        similarity_cache_.erase(similarity_cache_.begin());
                    }
                    similarity_cache_[cache_key] = similarity;
                }
            }
        } else {
            // No caching
            auto candidate_pattern_opt = pattern_db_->Retrieve(candidate_id);
            if (candidate_pattern_opt) {
                auto candidate_data = candidate_pattern_opt->GetData();
                similarity = similarity_metric_->Compute(query_data, candidate_data);
                ++similarity_computations_;
            }
        }

        scores.push_back(similarity);
    }

    return scores;
}

std::vector<float> SemanticAttentionHead::NormalizeScores(
    const std::vector<float>& scores) const {

    if (scores.empty()) {
        return {};
    }

    // Apply temperature scaling and softmax
    return Softmax(scores, config_.temperature);
}

void SemanticAttentionHead::LogDebug(const std::string& message) const {
    if (config_.debug_logging) {
        std::cout << "[SemanticAttentionHead] " << message << std::endl;
    }
}

} // namespace attention
} // namespace dpan
