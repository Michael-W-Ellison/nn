// File: src/learning/attention_utils.cpp
//
// Implementation of Attention Utility Functions
//
// Key implementation details:
// - Softmax uses max-subtraction for numerical stability
// - All functions handle edge cases (empty, zero, NaN/inf)
// - Optimized for performance with minimal allocations

#include "learning/attention_utils.hpp"
#include "core/pattern_node.hpp"
#include "core/pattern_data.hpp"
#include "storage/pattern_database.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace dpan {
namespace attention {

namespace {
    // Constants for numerical stability
    constexpr float kEpsilon = 1e-10f;
    constexpr float kMaxFloat = std::numeric_limits<float>::max();
    constexpr float kMinFloat = std::numeric_limits<float>::lowest();
}

/// Apply softmax normalization to vector
std::vector<float> Softmax(
    const std::vector<float>& scores,
    float temperature) {

    // Handle edge cases
    if (scores.empty()) {
        return {};
    }

    if (temperature <= 0.0f) {
        temperature = 1.0f;  // Fallback to standard softmax
    }

    // Find max score for numerical stability
    // Subtracting max prevents overflow in exp()
    float max_score = kMinFloat;
    bool has_valid_score = false;

    for (float score : scores) {
        if (IsValid(score)) {
            max_score = std::max(max_score, score);
            has_valid_score = true;
        }
    }

    // If all scores are invalid, return uniform distribution
    if (!has_valid_score) {
        float uniform_weight = 1.0f / scores.size();
        return std::vector<float>(scores.size(), uniform_weight);
    }

    // Apply temperature scaling and compute exp(score)
    std::vector<float> exp_scores;
    exp_scores.reserve(scores.size());

    float sum_exp = 0.0f;

    for (float score : scores) {
        float scaled_score;
        if (IsValid(score)) {
            // Numerical stability: exp(score - max) instead of exp(score)
            // This prevents overflow while maintaining relative differences
            scaled_score = (score - max_score) / temperature;
        } else {
            // Invalid scores get very low probability
            scaled_score = kMinFloat / temperature;
        }

        // Compute exponential
        float exp_val = std::exp(scaled_score);

        // Clamp to prevent numerical issues
        if (!IsValid(exp_val) || exp_val < kEpsilon) {
            exp_val = kEpsilon;
        }

        exp_scores.push_back(exp_val);
        sum_exp += exp_val;
    }

    // Normalize to sum to 1.0
    std::vector<float> weights;
    weights.reserve(scores.size());

    if (sum_exp < kEpsilon) {
        // Fallback to uniform distribution if sum is too small
        float uniform_weight = 1.0f / scores.size();
        weights.assign(scores.size(), uniform_weight);
    } else {
        for (float exp_val : exp_scores) {
            weights.push_back(exp_val / sum_exp);
        }
    }

    return weights;
}

/// Template implementation for map-based softmax
/// (Defined in header as template)

/// Compute dot product between two vectors
float DotProduct(
    const std::vector<float>& a,
    const std::vector<float>& b) {

    if (a.empty() || b.empty()) {
        return 0.0f;
    }

    // Use minimum length to handle different-sized vectors
    size_t len = std::min(a.size(), b.size());

    float dot = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        if (IsValid(a[i]) && IsValid(b[i])) {
            dot += a[i] * b[i];
        }
    }

    return dot;
}

/// Compute scaled dot product
float ScaledDotProduct(
    const std::vector<float>& query,
    const std::vector<float>& key,
    bool scale_by_dim) {

    float dot = DotProduct(query, key);

    if (!scale_by_dim) {
        return dot;
    }

    // Scale by 1/sqrt(dimension) for numerical stability
    size_t dim = std::min(query.size(), key.size());
    if (dim == 0) {
        return 0.0f;
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    return dot * scale;
}

/// Compute L2 norm
float L2Norm(const std::vector<float>& vec) {
    if (vec.empty()) {
        return 0.0f;
    }

    float sum_squares = 0.0f;
    for (float val : vec) {
        if (IsValid(val)) {
            sum_squares += val * val;
        }
    }

    return std::sqrt(sum_squares);
}

/// Normalize vector to unit length
std::vector<float> NormalizeL2(const std::vector<float>& vec) {
    float norm = L2Norm(vec);

    if (norm < kEpsilon) {
        // Zero vector or too small - return as-is
        return vec;
    }

    std::vector<float> normalized;
    normalized.reserve(vec.size());

    for (float val : vec) {
        normalized.push_back(val / norm);
    }

    return normalized;
}

/// Compute cosine similarity
float CosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b) {

    if (a.empty() || b.empty()) {
        return 0.0f;
    }

    float dot = DotProduct(a, b);
    float norm_a = L2Norm(a);
    float norm_b = L2Norm(b);

    // Avoid division by zero
    if (norm_a < kEpsilon || norm_b < kEpsilon) {
        return 0.0f;
    }

    float similarity = dot / (norm_a * norm_b);

    // Clamp to valid range [-1, 1] (numerical errors can cause slight overshoot)
    return Clamp(similarity, -1.0f, 1.0f);
}

/// Combine two scores with weighted averaging
float CombineScores(
    float score_a,
    float score_b,
    float weight_a,
    float weight_b) {

    return weight_a * score_a + weight_b * score_b;
}

/// Clamp value to range
float Clamp(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

/// Apply temperature scaling
std::vector<float> ApplyTemperature(
    const std::vector<float>& scores,
    float temperature) {

    if (temperature <= 0.0f) {
        return scores;  // Invalid temperature, return unchanged
    }

    std::vector<float> scaled;
    scaled.reserve(scores.size());

    for (float score : scores) {
        scaled.push_back(score / temperature);
    }

    return scaled;
}

/// Check if value is valid
bool IsValid(float value) {
    return std::isfinite(value);
}

/// Safe division with fallback
float SafeDivide(float numerator, float denominator, float fallback) {
    if (std::abs(denominator) < kEpsilon) {
        return fallback;
    }

    float result = numerator / denominator;

    if (!IsValid(result)) {
        return fallback;
    }

    return result;
}

// ============================================================================
// Feature Extraction
// ============================================================================

/// Extract features from PatternNode
std::vector<float> ExtractFeatures(
    const PatternNode& node,
    const FeatureExtractionConfig& config) {

    std::vector<float> features;

    // 1. Extract base features from pattern data
    const auto& pattern_data = node.GetData();
    auto base_features = pattern_data.GetFeatures();

    // Copy base features to result
    const auto& base_data = base_features.Data();
    features.reserve(
        base_data.size() +
        (config.include_confidence ? 1 : 0) +
        (config.include_access_count ? 1 : 0) +
        (config.include_age ? 1 : 0) +
        (config.include_type ? 3 : 0)
    );

    features.insert(features.end(), base_data.begin(), base_data.end());

    // 2. Add confidence score if requested
    if (config.include_confidence) {
        float confidence = node.GetConfidenceScore();
        // Confidence should already be in [0, 1] but clamp to be safe
        features.push_back(Clamp(confidence, 0.0f, 1.0f));
    }

    // 3. Add normalized access count if requested
    if (config.include_access_count) {
        uint32_t access_count = node.GetAccessCount();
        float normalized_count = static_cast<float>(access_count) /
                                static_cast<float>(config.max_access_count);
        // Clamp to [0, 1]
        features.push_back(Clamp(normalized_count, 0.0f, 1.0f));
    }

    // 4. Add normalized age if requested
    if (config.include_age) {
        auto age = node.GetAge();
        // Convert microseconds to seconds
        float age_seconds = static_cast<float>(age.count()) / 1000000.0f;
        float normalized_age = age_seconds / config.max_age_seconds;
        // Clamp to [0, 1]
        features.push_back(Clamp(normalized_age, 0.0f, 1.0f));
    }

    // 5. Add pattern type encoding if requested (one-hot)
    if (config.include_type) {
        PatternType type = node.GetType();

        // One-hot encoding: [ATOMIC, COMPOSITE, META]
        features.push_back(type == PatternType::ATOMIC ? 1.0f : 0.0f);
        features.push_back(type == PatternType::COMPOSITE ? 1.0f : 0.0f);
        features.push_back(type == PatternType::META ? 1.0f : 0.0f);
    }

    return features;
}

/// Extract features from pattern by ID
std::vector<float> ExtractFeatures(
    PatternID pattern_id,
    PatternDatabase* db,
    const FeatureExtractionConfig& config) {

    if (!db) {
        return {};
    }

    // Retrieve pattern from database
    auto pattern_opt = db->Retrieve(pattern_id);
    if (!pattern_opt) {
        return {};  // Pattern not found
    }

    // Extract features from retrieved pattern
    return ExtractFeatures(*pattern_opt, config);
}

/// Compute feature dimensionality
size_t GetFeatureDimension(
    size_t base_dimension,
    const FeatureExtractionConfig& config) {

    size_t total = base_dimension;

    if (config.include_confidence) {
        total += 1;
    }

    if (config.include_access_count) {
        total += 1;
    }

    if (config.include_age) {
        total += 1;
    }

    if (config.include_type) {
        total += 3;  // One-hot encoding for 3 pattern types
    }

    return total;
}

} // namespace attention
} // namespace dpan
