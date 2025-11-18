// File: src/learning/context_aware_attention.cpp
//
// Implementation of ContextAwareAttention
//
// Key implementation details:
// - Circular buffer for context history (std::deque with size limit)
// - Thread-safe context recording and retrieval
// - Combined semantic + context scoring before softmax
// - Maximum similarity across historical contexts

#include "learning/context_aware_attention.hpp"
#include "learning/attention_utils.hpp"
#include <algorithm>
#include <cmath>

namespace dpan {
namespace attention {

namespace {
    constexpr float kDefaultContextSimilarity = 0.5f;  // Neutral score for no history
}

ContextAwareAttention::ContextAwareAttention(
    const AttentionConfig& attn_config,
    const ContextAwareConfig& ctx_config)
    : BasicAttentionMechanism(attn_config)
    , ctx_config_(ctx_config)
    , context_similarity_computations_(0)
    , context_activations_recorded_(0)
{
    // Validate and normalize context config
    if (!ctx_config_.Validate()) {
        ctx_config_.Normalize();
    }
}

// ============================================================================
// Overridden AttentionMechanism Methods
// ============================================================================

std::map<PatternID, float> ContextAwareAttention::ComputeAttention(
    PatternID query,
    const std::vector<PatternID>& candidates,
    const ContextVector& context) {

    // Handle empty candidates
    if (candidates.empty()) {
        LogDebug("ContextAwareAttention: No candidates provided");
        return {};
    }

    // Handle single candidate
    if (candidates.size() == 1) {
        LogDebug("ContextAwareAttention: Single candidate, returning weight 1.0");
        return {{candidates[0], 1.0f}};
    }

    // Check cache if enabled (using base class cache)
    // Note: We can't access protected members directly from base class,
    // so we skip caching for now in this override
    // (could be enhanced with friend declaration or public interface)

    LogDebug("ContextAwareAttention: Computing context-aware attention");

    // Validate pattern database
    PatternDatabase* db = nullptr;
    // We need to get pattern_db_ from base class - it's protected
    // For now, we'll use the method interface

    // Extract features for query and candidates (semantic similarity)
    // We'll need to use public methods or make pattern_db_ accessible

    // For now, let's implement using composition approach
    // We'll compute semantic similarity separately

    // Since we can't easily access base class protected members,
    // we'll implement a simplified version that computes both scores

    // Actually, looking at the base class, pattern_db_ is protected
    // so we can access it in derived class

    if (!pattern_db_) {
        LogDebug("ERROR: Pattern database not set");
        // Return uniform distribution
        float uniform = 1.0f / candidates.size();
        std::map<PatternID, float> weights;
        for (const auto& candidate : candidates) {
            weights[candidate] = uniform;
        }
        return weights;
    }

    // Extract features for query and candidates (semantic similarity)
    auto query_features = ExtractFeatures(query, pattern_db_, GetFeatureConfig());
    if (query_features.empty()) {
        LogDebug("ERROR: Failed to extract query features");
        float uniform = 1.0f / candidates.size();
        std::map<PatternID, float> weights;
        for (const auto& candidate : candidates) {
            weights[candidate] = uniform;
        }
        return weights;
    }

    auto candidate_features = ExtractMultipleFeatures(candidates);

    // Compute semantic similarity scores
    auto semantic_scores = ComputeRawScores(query_features, candidate_features);

    // Compute context similarity scores
    auto context_scores = ComputeContextScores(context, candidates);

    // Combine semantic and context scores
    auto combined_scores = CombineScores(semantic_scores, context_scores);

    // Apply temperature scaling and softmax normalization
    auto normalized_weights = Softmax(combined_scores, GetConfig().temperature);

    // Build result map
    std::map<PatternID, float> weights;
    for (size_t i = 0; i < candidates.size(); ++i) {
        weights[candidates[i]] = normalized_weights[i];
    }

    // Log debug information
    if (GetConfig().debug_logging) {
        LogAttentionDetails(query, combined_scores, weights);
    }

    return weights;
}

// ============================================================================
// Context History Management
// ============================================================================

void ContextAwareAttention::RecordActivation(
    PatternID pattern_id,
    const ContextVector& context) {

    std::lock_guard<std::mutex> lock(history_mutex_);

    ++context_activations_recorded_;

    // Get or create context history for this pattern
    auto& history = context_history_[pattern_id];

    // Add new context to the front (most recent)
    history.push_front(context);

    // Maintain maximum history size (circular buffer behavior)
    if (history.size() > ctx_config_.max_context_history) {
        history.pop_back();  // Remove oldest
    }

    LogDebug("Recorded activation for pattern " +
             std::to_string(pattern_id.value()) +
             " (history size: " + std::to_string(history.size()) + ")");
}

std::vector<ContextVector> ContextAwareAttention::GetContextHistory(
    PatternID pattern_id) const {

    std::lock_guard<std::mutex> lock(history_mutex_);

    auto it = context_history_.find(pattern_id);
    if (it == context_history_.end()) {
        return {};  // No history for this pattern
    }

    // Convert deque to vector (most recent first)
    return std::vector<ContextVector>(it->second.begin(), it->second.end());
}

void ContextAwareAttention::ClearContextHistory() {
    std::lock_guard<std::mutex> lock(history_mutex_);
    context_history_.clear();
    LogDebug("Cleared all context history");
}

void ContextAwareAttention::ClearContextHistory(PatternID pattern_id) {
    std::lock_guard<std::mutex> lock(history_mutex_);
    context_history_.erase(pattern_id);
    LogDebug("Cleared context history for pattern " +
             std::to_string(pattern_id.value()));
}

// ============================================================================
// Context Similarity Computation
// ============================================================================

float ContextAwareAttention::ComputeContextSimilarity(
    const ContextVector& query_context,
    PatternID candidate_pattern) const {

    std::lock_guard<std::mutex> lock(history_mutex_);

    ++context_similarity_computations_;

    // Get historical contexts for this pattern
    auto it = context_history_.find(candidate_pattern);
    if (it == context_history_.end() || it->second.empty()) {
        // No history - return neutral score
        return kDefaultContextSimilarity;
    }

    const auto& history = it->second;

    // Compute similarity with each historical context
    // Return maximum similarity (best match)
    float max_similarity = -1.0f;  // Cosine similarity range: [-1, 1]

    for (const auto& historical_context : history) {
        float similarity = query_context.CosineSimilarity(historical_context);
        max_similarity = std::max(max_similarity, similarity);
    }

    // Normalize to [0, 1] range
    // Cosine similarity is in [-1, 1], so: (sim + 1) / 2
    float normalized_similarity = (max_similarity + 1.0f) / 2.0f;

    // Clamp to [0, 1] to be safe
    return std::min(std::max(normalized_similarity, 0.0f), 1.0f);
}

// ============================================================================
// Configuration
// ============================================================================

void ContextAwareAttention::SetContextConfig(const ContextAwareConfig& config) {
    ctx_config_ = config;

    // Validate and normalize if needed
    if (!ctx_config_.Validate()) {
        ctx_config_.Normalize();
    }

    LogDebug("Context-aware configuration updated");
}

const ContextAwareConfig& ContextAwareAttention::GetContextConfig() const {
    return ctx_config_;
}

std::map<std::string, float> ContextAwareAttention::GetStatistics() const {
    // Get base class statistics
    auto stats = BasicAttentionMechanism::GetStatistics();

    // Add context-aware statistics
    stats["context_similarity_computations"] =
        static_cast<float>(context_similarity_computations_);
    stats["context_activations_recorded"] =
        static_cast<float>(context_activations_recorded_);

    {
        std::lock_guard<std::mutex> lock(history_mutex_);
        stats["patterns_with_history"] =
            static_cast<float>(context_history_.size());

        // Calculate average history size
        if (!context_history_.empty()) {
            size_t total_size = 0;
            for (const auto& [id, history] : context_history_) {
                total_size += history.size();
            }
            stats["avg_history_size"] =
                static_cast<float>(total_size) / context_history_.size();
        } else {
            stats["avg_history_size"] = 0.0f;
        }
    }

    return stats;
}

// ============================================================================
// Helper Methods
// ============================================================================

std::vector<float> ContextAwareAttention::ComputeContextScores(
    const ContextVector& query_context,
    const std::vector<PatternID>& candidates) const {

    std::vector<float> scores;
    scores.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        float similarity = ComputeContextSimilarity(query_context, candidate);
        scores.push_back(similarity);
    }

    return scores;
}

std::vector<float> ContextAwareAttention::CombineScores(
    const std::vector<float>& semantic_scores,
    const std::vector<float>& context_scores) const {

    if (semantic_scores.size() != context_scores.size()) {
        LogDebug("ERROR: Score vector size mismatch");
        return semantic_scores;  // Fallback to semantic only
    }

    std::vector<float> combined;
    combined.reserve(semantic_scores.size());

    for (size_t i = 0; i < semantic_scores.size(); ++i) {
        float combined_score =
            ctx_config_.semantic_weight * semantic_scores[i] +
            ctx_config_.context_weight * context_scores[i];

        combined.push_back(combined_score);
    }

    return combined;
}

} // namespace attention
} // namespace dpan
