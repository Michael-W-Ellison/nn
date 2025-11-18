// File: src/learning/multi_head_attention.cpp
//
// Implementation of MultiHeadAttention
//
// Key implementation details:
// - Thread-safe head management with mutex
// - Weighted combination of head outputs
// - Automatic weight normalization
// - Statistics aggregation across all heads

#include "learning/multi_head_attention.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace dpan {
namespace attention {

MultiHeadAttention::MultiHeadAttention(const MultiHeadConfig& config)
    : config_(config)
    , pattern_db_(nullptr)
    , attention_computations_(0)
    , head_combinations_(0)
{
    if (!config_.Validate()) {
        throw std::invalid_argument("Invalid MultiHeadConfig");
    }
}

// ============================================================================
// AttentionMechanism Interface Implementation
// ============================================================================

std::map<PatternID, float> MultiHeadAttention::ComputeAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    std::lock_guard<std::mutex> lock(heads_mutex_);

    ++attention_computations_;

    // Handle empty candidates
    if (candidates.empty()) {
        LogDebug("MultiHeadAttention: No candidates provided");
        return {};
    }

    // Handle no heads
    if (heads_.empty()) {
        LogDebug("MultiHeadAttention: No heads configured, returning uniform weights");
        std::map<PatternID, float> weights;
        float uniform = 1.0f / candidates.size();
        for (const auto& candidate : candidates) {
            weights[candidate] = uniform;
        }
        return weights;
    }

    // Handle single candidate
    if (candidates.size() == 1) {
        LogDebug("MultiHeadAttention: Single candidate, returning weight 1.0");
        return {{candidates[0], 1.0f}};
    }

    LogDebug("MultiHeadAttention: Computing attention with " +
             std::to_string(heads_.size()) + " heads");

    // Compute attention for each head
    std::vector<std::map<PatternID, float>> head_weights;
    head_weights.reserve(heads_.size());

    for (const auto& head : heads_) {
        if (!head.mechanism) {
            LogDebug("WARNING: Head '" + head.name + "' has null mechanism, skipping");
            continue;
        }

        auto weights = head.mechanism->ComputeAttention(query, candidates, context);

        if (weights.empty()) {
            LogDebug("WARNING: Head '" + head.name + "' returned empty weights");
            // Use uniform distribution as fallback
            std::map<PatternID, float> uniform_weights;
            float uniform = 1.0f / candidates.size();
            for (const auto& candidate : candidates) {
                uniform_weights[candidate] = uniform;
            }
            head_weights.push_back(uniform_weights);
        } else {
            head_weights.push_back(weights);
        }
    }

    // Combine head outputs
    auto combined = CombineHeadWeights(head_weights);

    ++head_combinations_;

    return combined;
}

std::vector<AttentionScore> MultiHeadAttention::ComputeDetailedAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    // Compute basic attention weights
    auto weights = ComputeAttention(query, candidates, context);

    // Convert to AttentionScore vector
    std::vector<AttentionScore> scores;
    scores.reserve(weights.size());

    for (const auto& [pattern_id, weight] : weights) {
        AttentionScore score;
        score.pattern_id = pattern_id;
        score.weight = weight;
        score.raw_score = weight;  // For multi-head, raw and normalized are same
        scores.push_back(score);
    }

    // Sort by weight descending
    std::sort(scores.begin(), scores.end(),
              [](const AttentionScore& a, const AttentionScore& b) {
                  return a.weight > b.weight;
              });

    return scores;
}

std::vector<std::pair<PatternID, float>> MultiHeadAttention::ApplyAttention(
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

void MultiHeadAttention::SetPatternDatabase(PatternDatabase* db) {
    std::lock_guard<std::mutex> lock(heads_mutex_);

    pattern_db_ = db;

    // Propagate to all heads
    for (auto& head : heads_) {
        if (head.mechanism) {
            head.mechanism->SetPatternDatabase(db);
        }
    }

    LogDebug("Pattern database set for all heads");
}

const AttentionConfig& MultiHeadAttention::GetConfig() const {
    return base_config_;
}

void MultiHeadAttention::SetConfig(const AttentionConfig& config) {
    if (!config.Validate()) {
        throw std::invalid_argument("Invalid AttentionConfig");
    }

    std::lock_guard<std::mutex> lock(heads_mutex_);

    base_config_ = config;

    // Propagate to all heads
    for (auto& head : heads_) {
        if (head.mechanism) {
            head.mechanism->SetConfig(config);
        }
    }

    LogDebug("Configuration updated for all heads");
}

void MultiHeadAttention::ClearCache() {
    std::lock_guard<std::mutex> lock(heads_mutex_);

    // Clear cache for all heads
    for (auto& head : heads_) {
        if (head.mechanism) {
            head.mechanism->ClearCache();
        }
    }

    LogDebug("Cache cleared for all heads");
}

std::map<std::string, float> MultiHeadAttention::GetStatistics() const {
    std::lock_guard<std::mutex> lock(heads_mutex_);

    std::map<std::string, float> stats;

    // Multi-head specific statistics
    stats["num_heads"] = static_cast<float>(heads_.size());
    stats["attention_computations"] = static_cast<float>(attention_computations_);
    stats["head_combinations"] = static_cast<float>(head_combinations_);

    // Aggregate statistics from all heads
    for (const auto& head : heads_) {
        if (head.mechanism) {
            auto head_stats = head.mechanism->GetStatistics();
            for (const auto& [key, value] : head_stats) {
                std::string prefixed_key = "head_" + head.name + "_" + key;
                stats[prefixed_key] = value;
            }
        }
    }

    return stats;
}

// ============================================================================
// Multi-Head Specific Methods
// ============================================================================

bool MultiHeadAttention::AddHead(const std::string& name,
                                  std::shared_ptr<AttentionMechanism> mechanism,
                                  float weight) {
    if (name.empty() || !mechanism) {
        return false;
    }

    if (weight < 0.0f || weight > 1.0f) {
        return false;
    }

    std::lock_guard<std::mutex> lock(heads_mutex_);

    // Check if head with this name already exists
    for (const auto& head : heads_) {
        if (head.name == name) {
            LogDebug("Head '" + name + "' already exists");
            return false;
        }
    }

    // Create and add head
    AttentionHead head(name, mechanism, weight);

    // Set pattern database if we have one
    if (pattern_db_) {
        mechanism->SetPatternDatabase(pattern_db_);
    }

    heads_.push_back(head);

    // Auto-normalize weights if enabled
    if (config_.auto_normalize_weights) {
        NormalizeWeightsUnsafe();
    }

    LogDebug("Added head '" + name + "' with weight " + std::to_string(weight));

    return true;
}

bool MultiHeadAttention::RemoveHead(const std::string& name) {
    std::lock_guard<std::mutex> lock(heads_mutex_);

    auto it = std::find_if(heads_.begin(), heads_.end(),
                          [&name](const AttentionHead& head) {
                              return head.name == name;
                          });

    if (it == heads_.end()) {
        LogDebug("Head '" + name + "' not found");
        return false;
    }

    heads_.erase(it);

    // Auto-normalize weights if enabled
    if (config_.auto_normalize_weights && !heads_.empty()) {
        NormalizeWeightsUnsafe();
    }

    LogDebug("Removed head '" + name + "'");

    return true;
}

const AttentionHead* MultiHeadAttention::GetHead(const std::string& name) const {
    std::lock_guard<std::mutex> lock(heads_mutex_);

    for (const auto& head : heads_) {
        if (head.name == name) {
            return &head;
        }
    }

    return nullptr;
}

const std::vector<AttentionHead>& MultiHeadAttention::GetHeads() const {
    return heads_;
}

bool MultiHeadAttention::SetHeadWeight(const std::string& name, float weight) {
    if (weight < 0.0f || weight > 1.0f) {
        return false;
    }

    std::lock_guard<std::mutex> lock(heads_mutex_);

    for (auto& head : heads_) {
        if (head.name == name) {
            head.weight = weight;

            // Auto-normalize weights if enabled
            if (config_.auto_normalize_weights) {
                NormalizeWeightsUnsafe();
            }

            LogDebug("Updated weight for head '" + name + "' to " +
                     std::to_string(weight));

            return true;
        }
    }

    LogDebug("Head '" + name + "' not found");
    return false;
}

size_t MultiHeadAttention::GetNumHeads() const {
    std::lock_guard<std::mutex> lock(heads_mutex_);
    return heads_.size();
}

void MultiHeadAttention::NormalizeWeights() {
    std::lock_guard<std::mutex> lock(heads_mutex_);
    NormalizeWeightsUnsafe();
}

void MultiHeadAttention::NormalizeWeightsUnsafe() {
    // Note: Caller must hold heads_mutex_

    if (heads_.empty()) {
        return;
    }

    // Calculate sum of weights
    float sum = 0.0f;
    for (const auto& head : heads_) {
        sum += head.weight;
    }

    // Avoid division by zero
    if (sum <= 0.0f) {
        // Set equal weights
        float equal_weight = 1.0f / heads_.size();
        for (auto& head : heads_) {
            head.weight = equal_weight;
        }
        LogDebug("Weights were zero or negative, set to equal distribution");
        return;
    }

    // Normalize
    for (auto& head : heads_) {
        head.weight /= sum;
    }

    LogDebug("Normalized head weights (sum=" + std::to_string(sum) + ")");
}

bool MultiHeadAttention::ValidateHeads() const {
    std::lock_guard<std::mutex> lock(heads_mutex_);

    if (heads_.empty()) {
        return true;  // Empty is valid
    }

    // Check each head
    for (const auto& head : heads_) {
        if (!head.Validate()) {
            return false;
        }
    }

    // Check weight sum (should be close to 1.0)
    float sum = 0.0f;
    for (const auto& head : heads_) {
        sum += head.weight;
    }

    // Allow small floating-point error
    const float epsilon = 1e-5f;
    if (std::abs(sum - 1.0f) > epsilon) {
        return false;
    }

    return true;
}

const MultiHeadConfig& MultiHeadAttention::GetMultiHeadConfig() const {
    return config_;
}

void MultiHeadAttention::SetMultiHeadConfig(const MultiHeadConfig& config) {
    if (!config.Validate()) {
        throw std::invalid_argument("Invalid MultiHeadConfig");
    }

    config_ = config;

    // Re-normalize weights if auto-normalize changed to true
    if (config_.auto_normalize_weights) {
        NormalizeWeights();
    }

    LogDebug("Multi-head configuration updated");
}

// ============================================================================
// Protected Helper Methods
// ============================================================================

std::map<PatternID, float> MultiHeadAttention::CombineHeadWeights(
    const std::vector<std::map<PatternID, float>>& head_weights) const {

    // Note: Caller should hold heads_mutex_

    if (head_weights.empty()) {
        return {};
    }

    if (head_weights.size() == 1) {
        return head_weights[0];
    }

    // Initialize combined weights
    std::map<PatternID, float> combined;

    // Weighted average of all head outputs
    for (size_t i = 0; i < head_weights.size(); ++i) {
        float head_weight = (i < heads_.size()) ? heads_[i].weight : 1.0f;

        for (const auto& [pattern_id, weight] : head_weights[i]) {
            combined[pattern_id] += head_weight * weight;
        }
    }

    // Note: Weights should already be normalized if heads are normalized
    // But we'll verify and renormalize to be safe

    float sum = 0.0f;
    for (const auto& [_, weight] : combined) {
        sum += weight;
    }

    if (sum > 0.0f) {
        for (auto& [_, weight] : combined) {
            weight /= sum;
        }
    }

    return combined;
}

void MultiHeadAttention::LogDebug(const std::string& message) const {
    if (config_.debug_logging) {
        std::cout << "[MultiHeadAttention] " << message << std::endl;
    }
}

} // namespace attention
} // namespace dpan
