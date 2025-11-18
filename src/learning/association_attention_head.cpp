// File: src/learning/association_attention_head.cpp
//
// Implementation of AssociationAttentionHead

#include "learning/association_attention_head.hpp"
#include "learning/attention_utils.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace dpan {
namespace attention {

AssociationAttentionHead::AssociationAttentionHead(
    const AssociationAttentionConfig& config)
    : config_(config)
    , association_matrix_(nullptr)
    , pattern_db_(nullptr)
    , attention_computations_(0)
    , association_lookups_(0)
    , cache_hits_(0)
    , cache_misses_(0)
    , missing_associations_(0)
{
    if (!config_.Validate()) {
        throw std::invalid_argument("Invalid AssociationAttentionConfig");
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

std::map<PatternID, float> AssociationAttentionHead::ComputeAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    ++attention_computations_;

    // Handle empty candidates
    if (candidates.empty()) {
        LogDebug("AssociationAttentionHead: No candidates provided");
        return {};
    }

    // Handle single candidate
    if (candidates.size() == 1) {
        LogDebug("AssociationAttentionHead: Single candidate, returning weight 1.0");
        return {{candidates[0], 1.0f}};
    }

    LogDebug("AssociationAttentionHead: Computing association attention for " +
             std::to_string(candidates.size()) + " candidates");

    // Compute association scores
    auto association_scores = ComputeAssociationScores(query, candidates, context);

    // Normalize scores to attention weights
    auto normalized_weights = NormalizeScores(association_scores);

    // Build result map
    std::map<PatternID, float> weights;
    for (size_t i = 0; i < candidates.size(); ++i) {
        weights[candidates[i]] = normalized_weights[i];
    }

    return weights;
}

std::vector<AttentionScore> AssociationAttentionHead::ComputeDetailedAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    // Compute basic attention weights
    auto weights = ComputeAttention(query, candidates, context);

    // Compute raw association scores for detailed view
    auto association_scores = ComputeAssociationScores(query, candidates, context);

    // Convert to AttentionScore vector
    std::vector<AttentionScore> scores;
    scores.reserve(weights.size());

    for (size_t i = 0; i < candidates.size(); ++i) {
        AttentionScore score;
        score.pattern_id = candidates[i];
        score.weight = weights[candidates[i]];
        score.raw_score = association_scores[i];

        // Fill in component breakdown
        // For association attention, we put the score in importance_score
        // since it reflects the importance of the learned association
        score.components.importance_score = association_scores[i];
        // Other components are zero for pure association attention
        score.components.semantic_similarity = 0.0f;
        score.components.context_similarity = 0.0f;
        score.components.structural_score = 0.0f;
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

std::vector<std::pair<PatternID, float>> AssociationAttentionHead::ApplyAttention(
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

    // Sort by weight descending (strongest associations first)
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return result;
}

void AssociationAttentionHead::SetPatternDatabase(PatternDatabase* db) {
    pattern_db_ = db;
    LogDebug("Pattern database set (not used by association head)");
}

const AttentionConfig& AssociationAttentionHead::GetConfig() const {
    return base_config_;
}

void AssociationAttentionHead::SetConfig(const AttentionConfig& config) {
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

void AssociationAttentionHead::ClearCache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    association_cache_.clear();
    LogDebug("Cache cleared");
}

std::map<std::string, float> AssociationAttentionHead::GetStatistics() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);

    std::map<std::string, float> stats;
    stats["attention_computations"] = static_cast<float>(attention_computations_);
    stats["association_lookups"] = static_cast<float>(association_lookups_);
    stats["cache_hits"] = static_cast<float>(cache_hits_);
    stats["cache_misses"] = static_cast<float>(cache_misses_);
    stats["missing_associations"] = static_cast<float>(missing_associations_);
    stats["cache_size"] = static_cast<float>(association_cache_.size());

    // Calculate cache hit rate
    float total_lookups = static_cast<float>(cache_hits_ + cache_misses_);
    stats["cache_hit_rate"] = total_lookups > 0.0f
        ? static_cast<float>(cache_hits_) / total_lookups
        : 0.0f;

    return stats;
}

// ============================================================================
// Association Attention Specific Methods
// ============================================================================

const AssociationAttentionConfig& AssociationAttentionHead::GetAssociationConfig() const {
    return config_;
}

void AssociationAttentionHead::SetAssociationConfig(const AssociationAttentionConfig& config) {
    if (!config.Validate()) {
        throw std::invalid_argument("Invalid AssociationAttentionConfig");
    }

    config_ = config;

    // Update base config as well
    base_config_.temperature = config_.temperature;
    base_config_.enable_caching = config_.enable_caching;
    base_config_.cache_size = config_.cache_size;
    base_config_.debug_logging = config_.debug_logging;

    LogDebug("Association attention configuration updated");
}

void AssociationAttentionHead::SetAssociationMatrix(AssociationMatrix* matrix) {
    association_matrix_ = matrix;
    LogDebug("Association matrix set");
}

// ============================================================================
// Protected Helper Methods
// ============================================================================

std::vector<float> AssociationAttentionHead::ComputeAssociationScores(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) const {

    std::vector<float> scores;
    scores.reserve(candidates.size());

    if (!association_matrix_) {
        // Return uniform scores if we don't have an association matrix
        LogDebug("WARNING: No association matrix available, using uniform scores");
        scores.resize(candidates.size(), 1.0f);
        return scores;
    }

    // Compute association score for each candidate
    for (const auto& candidate_id : candidates) {
        float score = config_.default_strength;

        // Check cache first if caching is enabled
        if (config_.enable_caching) {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            auto cache_key = std::make_pair(query, candidate_id);
            auto it = association_cache_.find(cache_key);

            if (it != association_cache_.end()) {
                score = it->second;
                ++cache_hits_;
            } else {
                ++cache_misses_;

                // Look up association in matrix
                const AssociationEdge* edge = association_matrix_->GetAssociation(query, candidate_id);
                ++association_lookups_;

                if (edge) {
                    // Use contextual or base strength
                    if (config_.use_contextual_strength) {
                        score = edge->GetContextualStrength(context);
                    } else {
                        score = edge->GetStrength();
                    }
                } else {
                    // No association exists, use default
                    ++missing_associations_;
                }

                // Apply strength threshold
                if (score < config_.strength_threshold) {
                    score = 0.0f;
                }

                // Cache the result (with LRU eviction if needed)
                if (association_cache_.size() >= config_.cache_size) {
                    // Simple eviction: remove first element
                    association_cache_.erase(association_cache_.begin());
                }
                association_cache_[cache_key] = score;
            }
        } else {
            // No caching - always look up
            const AssociationEdge* edge = association_matrix_->GetAssociation(query, candidate_id);
            ++association_lookups_;

            if (edge) {
                // Use contextual or base strength
                if (config_.use_contextual_strength) {
                    score = edge->GetContextualStrength(context);
                } else {
                    score = edge->GetStrength();
                }
            } else {
                // No association exists, use default
                ++missing_associations_;
            }

            // Apply strength threshold
            if (score < config_.strength_threshold) {
                score = 0.0f;
            }
        }

        scores.push_back(score);
    }

    return scores;
}

std::vector<float> AssociationAttentionHead::NormalizeScores(
    const std::vector<float>& scores) const {

    if (scores.empty()) {
        return {};
    }

    // Apply temperature scaling and softmax
    return Softmax(scores, config_.temperature);
}

void AssociationAttentionHead::LogDebug(const std::string& message) const {
    if (config_.debug_logging) {
        std::cout << "[AssociationAttentionHead] " << message << std::endl;
    }
}

} // namespace attention
} // namespace dpan
