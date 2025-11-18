// File: src/learning/self_attention.cpp
//
// Implementation of SelfAttention
//
// Key implementation details:
// - Computes N×N attention matrix efficiently
// - Supports multiple normalization modes
// - Optional diagonal masking for preventing self-attention
// - Caching for repeated computations
// - Analysis utilities for finding important patterns

#include "learning/self_attention.hpp"
#include "learning/attention_utils.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace dpan {
namespace attention {

SelfAttention::SelfAttention(const SelfAttentionConfig& config)
    : config_(config)
    , pattern_db_(nullptr)
    , similarity_metric_(nullptr)  // Will use simple dot product if not set
    , matrix_computations_(0)
    , cache_hits_(0)
    , cache_misses_(0)
{
    if (!config_.Validate()) {
        throw std::invalid_argument("Invalid SelfAttentionConfig");
    }
}

// ============================================================================
// Core Methods
// ============================================================================

std::map<std::pair<PatternID, PatternID>, float> SelfAttention::ComputeAttentionMatrix(
    const std::vector<PatternID>& patterns,
    const ContextVector& context) {

    if (patterns.empty()) {
        return {};
    }

    // Check cache first
    std::string cache_key = GenerateCacheKey(patterns);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (config_.enable_caching && cache_.count(cache_key) > 0) {
            ++cache_hits_;
            const auto& cached_matrix = cache_[cache_key];

            // Convert dense matrix to sparse map
            std::map<std::pair<PatternID, PatternID>, float> result;
            for (size_t i = 0; i < patterns.size(); ++i) {
                for (size_t j = 0; j < patterns.size(); ++j) {
                    if (cached_matrix[i][j] > 0.0f) {
                        result[{patterns[i], patterns[j]}] = cached_matrix[i][j];
                    }
                }
            }
            return result;
        }
        ++cache_misses_;
    }

    // Compute attention matrix
    auto dense_matrix = ComputeAttentionMatrixDense(patterns, context);

    // Cache if enabled
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (config_.enable_caching) {
            // Simple LRU: remove oldest if cache is full
            if (cache_.size() >= config_.cache_size) {
                cache_.erase(cache_.begin());
            }
            cache_[cache_key] = dense_matrix;
        }
    }

    // Convert to sparse map representation
    std::map<std::pair<PatternID, PatternID>, float> result;
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = 0; j < patterns.size(); ++j) {
            if (dense_matrix[i][j] > 0.0f) {
                result[{patterns[i], patterns[j]}] = dense_matrix[i][j];
            }
        }
    }

    ++matrix_computations_;
    return result;
}

std::vector<std::vector<float>> SelfAttention::ComputeAttentionMatrixDense(
    const std::vector<PatternID>& patterns,
    const ContextVector& context) {

    size_t n = patterns.size();
    if (n == 0) {
        return {};
    }

    // Compute similarity matrix
    auto similarity_scores = ComputeSimilarityMatrix(patterns, context);

    // Apply normalization
    auto attention = ApplySoftmax(similarity_scores);

    // Apply threshold if configured
    if (config_.attention_threshold > 0.0f) {
        ApplyThreshold(attention);
    }

    return attention;
}

std::map<PatternID, float> SelfAttention::GetQueryAttention(
    PatternID query,
    const std::vector<PatternID>& patterns,
    const ContextVector& context) {

    // Find query index
    auto it = std::find(patterns.begin(), patterns.end(), query);
    if (it == patterns.end()) {
        return {};  // Query not in patterns
    }
    size_t query_idx = std::distance(patterns.begin(), it);

    // Compute full attention matrix
    auto matrix = ComputeAttentionMatrixDense(patterns, context);

    // Extract query row
    std::map<PatternID, float> result;
    for (size_t j = 0; j < patterns.size(); ++j) {
        if (matrix[query_idx][j] > 0.0f) {
            result[patterns[j]] = matrix[query_idx][j];
        }
    }

    return result;
}

// ============================================================================
// Configuration
// ============================================================================

void SelfAttention::SetPatternDatabase(PatternDatabase* db) {
    std::lock_guard<std::mutex> lock(mutex_);
    pattern_db_ = db;
}

void SelfAttention::SetSimilarityMetric(std::shared_ptr<SimilarityMetric> metric) {
    std::lock_guard<std::mutex> lock(mutex_);
    similarity_metric_ = metric;
}

const SelfAttentionConfig& SelfAttention::GetConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

void SelfAttention::SetConfig(const SelfAttentionConfig& config) {
    if (!config.Validate()) {
        throw std::invalid_argument("Invalid SelfAttentionConfig");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    // Clear cache when config changes
    cache_.clear();
}

void SelfAttention::ClearCache() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

// ============================================================================
// Analysis and Utilities
// ============================================================================

std::vector<std::pair<PatternID, float>> SelfAttention::FindMostAttendedPatterns(
    const std::vector<PatternID>& patterns,
    size_t top_k,
    const ContextVector& context) {

    auto matrix = ComputeAttentionMatrixDense(patterns, context);
    size_t n = patterns.size();

    // Compute average attention received (column sums / n)
    std::vector<std::pair<PatternID, float>> scores;
    for (size_t j = 0; j < n; ++j) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum += matrix[i][j];
        }
        float avg = sum / static_cast<float>(n);
        scores.push_back({patterns[j], avg});
    }

    // Sort by attention (descending)
    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Return top k
    if (scores.size() > top_k) {
        scores.resize(top_k);
    }
    return scores;
}

std::vector<std::pair<PatternID, float>> SelfAttention::FindMostAttentivePatterns(
    const std::vector<PatternID>& patterns,
    size_t top_k,
    const ContextVector& context) {

    auto matrix = ComputeAttentionMatrixDense(patterns, context);
    size_t n = patterns.size();

    // Compute average attention given (row sums / n)
    std::vector<std::pair<PatternID, float>> scores;
    for (size_t i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            sum += matrix[i][j];
        }
        float avg = sum / static_cast<float>(n);
        scores.push_back({patterns[i], avg});
    }

    // Sort by attention (descending)
    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    // Return top k
    if (scores.size() > top_k) {
        scores.resize(top_k);
    }
    return scores;
}

std::map<PatternID, float> SelfAttention::ComputeAttentionEntropy(
    const std::vector<PatternID>& patterns,
    const ContextVector& context) {

    auto matrix = ComputeAttentionMatrixDense(patterns, context);
    size_t n = patterns.size();

    std::map<PatternID, float> entropy_map;
    for (size_t i = 0; i < n; ++i) {
        float entropy = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            float p = matrix[i][j];
            if (p > 0.0f) {
                entropy -= p * std::log2(p);
            }
        }
        entropy_map[patterns[i]] = entropy;
    }

    return entropy_map;
}

std::map<std::string, float> SelfAttention::GetStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::map<std::string, float> stats;
    stats["matrix_computations"] = static_cast<float>(matrix_computations_);
    stats["cache_hits"] = static_cast<float>(cache_hits_);
    stats["cache_misses"] = static_cast<float>(cache_misses_);

    if (cache_hits_ + cache_misses_ > 0) {
        float hit_rate = static_cast<float>(cache_hits_) /
                        static_cast<float>(cache_hits_ + cache_misses_);
        stats["cache_hit_rate"] = hit_rate;
    }

    stats["cache_size"] = static_cast<float>(cache_.size());

    return stats;
}

// ============================================================================
// Protected Methods
// ============================================================================

std::vector<std::vector<float>> SelfAttention::ComputeSimilarityMatrix(
    const std::vector<PatternID>& patterns,
    const ContextVector& context) {

    size_t n = patterns.size();
    std::vector<std::vector<float>> scores(n, std::vector<float>(n, 0.0f));

    // Compute pairwise similarities
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (i == j && config_.mask_diagonal) {
                scores[i][j] = config_.mask_value;
            } else {
                scores[i][j] = GetSimilarity(patterns[i], patterns[j], context);
            }
        }
    }

    return scores;
}

std::vector<std::vector<float>> SelfAttention::ApplySoftmax(
    const std::vector<std::vector<float>>& scores) {

    switch (config_.normalization) {
        case NormalizationMode::ROW_WISE:
            return ApplyRowWiseSoftmax(scores);

        case NormalizationMode::COLUMN_WISE:
            return ApplyColumnWiseSoftmax(scores);

        case NormalizationMode::BIDIRECTIONAL:
            return ApplyBidirectionalNormalization(scores);

        default:
            return ApplyRowWiseSoftmax(scores);
    }
}

std::vector<std::vector<float>> SelfAttention::ApplyRowWiseSoftmax(
    const std::vector<std::vector<float>>& scores) {

    size_t n = scores.size();
    std::vector<std::vector<float>> attention(n, std::vector<float>(n, 0.0f));

    for (size_t i = 0; i < n; ++i) {
        // Apply temperature scaling
        std::vector<float> scaled_scores(n);
        for (size_t j = 0; j < n; ++j) {
            scaled_scores[j] = scores[i][j] / config_.temperature;
        }

        // Apply softmax from attention_utils
        auto softmax_weights = Softmax(scaled_scores);

        // Store in attention matrix
        for (size_t j = 0; j < n; ++j) {
            attention[i][j] = softmax_weights[j];
        }
    }

    return attention;
}

std::vector<std::vector<float>> SelfAttention::ApplyColumnWiseSoftmax(
    const std::vector<std::vector<float>>& scores) {

    size_t n = scores.size();
    std::vector<std::vector<float>> attention(n, std::vector<float>(n, 0.0f));

    for (size_t j = 0; j < n; ++j) {
        // Extract column
        std::vector<float> column(n);
        for (size_t i = 0; i < n; ++i) {
            column[i] = scores[i][j] / config_.temperature;
        }

        // Apply softmax from attention_utils
        auto softmax_weights = Softmax(column);

        // Store in attention matrix
        for (size_t i = 0; i < n; ++i) {
            attention[i][j] = softmax_weights[i];
        }
    }

    return attention;
}

std::vector<std::vector<float>> SelfAttention::ApplyBidirectionalNormalization(
    const std::vector<std::vector<float>>& scores) {

    // First apply row-wise normalization
    auto row_normalized = ApplyRowWiseSoftmax(scores);

    // Then apply column-wise normalization to the result
    return ApplyColumnWiseSoftmax(row_normalized);
}

void SelfAttention::ApplyThreshold(std::vector<std::vector<float>>& attention) {
    size_t n = attention.size();

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (attention[i][j] < config_.attention_threshold) {
                attention[i][j] = 0.0f;
            }
        }

        // Re-normalize row after thresholding
        float row_sum = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            row_sum += attention[i][j];
        }

        if (row_sum > 0.0f) {
            for (size_t j = 0; j < n; ++j) {
                attention[i][j] /= row_sum;
            }
        }
    }
}

float SelfAttention::GetSimilarity(
    PatternID p1,
    PatternID p2,
    const ContextVector& context) {

    if (!pattern_db_) {
        throw std::runtime_error("PatternDatabase not set");
    }

    auto pattern1_opt = pattern_db_->Retrieve(p1);
    auto pattern2_opt = pattern_db_->Retrieve(p2);

    if (!pattern1_opt.has_value() || !pattern2_opt.has_value()) {
        return 0.0f;
    }

    // If similarity metric is set, use it
    if (similarity_metric_) {
        const auto& pattern1 = pattern1_opt.value();
        const auto& pattern2 = pattern2_opt.value();
        auto data1 = pattern1.GetData();
        auto data2 = pattern2.GetData();

        // Get feature vectors for similarity computation
        auto features1 = data1.GetFeatures();
        auto features2 = data2.GetFeatures();

        return similarity_metric_->ComputeFromFeatures(features1, features2);
    }

    // Otherwise, use simple data-based similarity (cosine similarity)
    const auto& pattern1 = pattern1_opt.value();
    const auto& pattern2 = pattern2_opt.value();

    auto data1 = pattern1.GetData();
    auto data2 = pattern2.GetData();

    if (data1.IsEmpty() || data2.IsEmpty()) {
        return 0.5f;  // Default similarity for patterns without data
    }

    // Get feature vectors and compute cosine similarity
    auto features1 = data1.GetFeatures();
    auto features2 = data2.GetFeatures();

    // Use CosineSimilarity from attention_utils
    return CosineSimilarity(features1.Data(), features2.Data());
}

std::string SelfAttention::GenerateCacheKey(const std::vector<PatternID>& patterns) const {
    std::ostringstream oss;
    for (const auto& pattern_id : patterns) {
        oss << pattern_id.ToString() << "|";
    }
    return oss.str();
}

} // namespace attention
} // namespace dpan
