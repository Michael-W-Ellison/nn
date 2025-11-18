// File: include/learning/attention_utils.hpp
//
// Utility Functions for Attention Mechanisms
// Provides common mathematical operations needed for attention computation
//
// Key functions:
// - Softmax normalization (converting scores to probabilities)
// - Dot product similarity (query-key matching)
// - Cosine similarity (normalized vector similarity)
// - Temperature scaling (controlling distribution sharpness)
// - Score combination (weighted averaging)
//
// These utilities support various attention types (dot-product, additive,
// multiplicative) and enable flexible attention computation.

#ifndef DPAN_LEARNING_ATTENTION_UTILS_HPP
#define DPAN_LEARNING_ATTENTION_UTILS_HPP

#include <vector>
#include <map>
#include <cmath>

namespace dpan {
namespace attention {

/// Apply softmax normalization to convert scores to probability distribution
///
/// Softmax ensures all weights are positive and sum to 1.0, creating a
/// valid probability distribution. The temperature parameter controls
/// the "sharpness" of the distribution:
/// - temperature > 1.0: More uniform (softer) distribution
/// - temperature = 1.0: Standard softmax
/// - temperature < 1.0: More peaked (sharper) distribution
///
/// Formula: softmax(x_i) = exp(x_i / T) / sum(exp(x_j / T))
/// where T is the temperature parameter.
///
/// @param scores Input scores (can be any real numbers)
/// @param temperature Temperature parameter for scaling (must be > 0)
/// @return Normalized weights that sum to 1.0
///
/// @note Empty input returns empty output
/// @note All NaN or infinite scores return uniform distribution
/// @note Temperature must be positive; defaults to 1.0 if invalid
///
/// Example:
/// @code
///   std::vector<float> scores = {2.0, 1.0, 0.1};
///   auto weights = Softmax(scores, 1.0);
///   // weights ≈ [0.659, 0.242, 0.099] (sum = 1.0)
/// @endcode
std::vector<float> Softmax(
    const std::vector<float>& scores,
    float temperature = 1.0f
);

/// Apply softmax normalization to map values
///
/// Similar to vector version but operates on a map structure.
/// Useful when scores are associated with keys (e.g., PatternIDs).
///
/// @param scores Map from keys to scores
/// @param temperature Temperature parameter for scaling (must be > 0)
/// @return Map from keys to normalized weights (sum to 1.0)
///
/// @note Empty input returns empty output
/// @note Preserves all keys from input
///
/// Example:
/// @code
///   std::map<PatternID, float> scores = {{1, 2.0}, {2, 1.0}, {3, 0.1}};
///   auto weights = Softmax(scores, 1.0);
///   // weights[1] ≈ 0.659, weights[2] ≈ 0.242, weights[3] ≈ 0.099
/// @endcode
template<typename KeyType>
std::map<KeyType, float> Softmax(
    const std::map<KeyType, float>& scores,
    float temperature = 1.0f
);

/// Compute dot product between two vectors
///
/// Dot product measures similarity by summing element-wise products.
/// Higher values indicate greater similarity/alignment.
/// This is the core operation in scaled dot-product attention.
///
/// Formula: dot(a, b) = sum(a_i * b_i)
///
/// @param a First vector
/// @param b Second vector
/// @return Dot product value
///
/// @note If vectors have different lengths, uses minimum length
/// @note Empty vectors return 0.0
/// @note Result is unbounded (can be positive or negative)
///
/// Example:
/// @code
///   std::vector<float> a = {1.0, 2.0, 3.0};
///   std::vector<float> b = {4.0, 5.0, 6.0};
///   float dot = DotProduct(a, b);  // dot = 1*4 + 2*5 + 3*6 = 32.0
/// @endcode
float DotProduct(
    const std::vector<float>& a,
    const std::vector<float>& b
);

/// Compute scaled dot product (Transformer-style attention)
///
/// Applies scaling factor of 1/sqrt(dimension) to prevent very large
/// dot products that would cause softmax saturation. This is the
/// standard scaling used in Transformer models.
///
/// Formula: scaled_dot(q, k) = dot(q, k) / sqrt(d_k)
/// where d_k is the dimension of the key vectors.
///
/// @param query Query vector
/// @param key Key vector
/// @param scale_by_dim If true, scales by 1/sqrt(dim); if false, no scaling
/// @return Scaled dot product
///
/// @note Scaling prevents gradient vanishing in deep networks
/// @note If vectors are empty, returns 0.0
///
/// Example:
/// @code
///   std::vector<float> query = {1.0, 2.0, 3.0};
///   std::vector<float> key = {4.0, 5.0, 6.0};
///   float score = ScaledDotProduct(query, key, true);
///   // score = 32.0 / sqrt(3) ≈ 18.48
/// @endcode
float ScaledDotProduct(
    const std::vector<float>& query,
    const std::vector<float>& key,
    bool scale_by_dim = true
);

/// Compute cosine similarity between two vectors
///
/// Cosine similarity measures the angle between vectors, normalized to [-1, 1].
/// Unlike dot product, it's independent of vector magnitude.
/// - 1.0: Vectors point in same direction (identical patterns)
/// - 0.0: Vectors are orthogonal (unrelated patterns)
/// - -1.0: Vectors point in opposite directions (opposing patterns)
///
/// Formula: cosine(a, b) = dot(a, b) / (||a|| * ||b||)
/// where ||a|| is the L2 norm (magnitude) of vector a.
///
/// @param a First vector
/// @param b Second vector
/// @return Cosine similarity in range [-1.0, 1.0]
///
/// @note If either vector has zero magnitude, returns 0.0
/// @note If vectors have different lengths, uses minimum length
/// @note Result is always bounded in [-1, 1]
///
/// Example:
/// @code
///   std::vector<float> a = {1.0, 2.0, 3.0};
///   std::vector<float> b = {2.0, 4.0, 6.0};  // Same direction, different magnitude
///   float sim = CosineSimilarity(a, b);  // sim = 1.0 (identical direction)
/// @endcode
float CosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b
);

/// Compute L2 norm (Euclidean magnitude) of a vector
///
/// The L2 norm is the square root of the sum of squared elements.
/// Used for vector normalization and distance calculations.
///
/// Formula: ||v|| = sqrt(sum(v_i^2))
///
/// @param vec Input vector
/// @return L2 norm (always non-negative)
///
/// @note Empty vector returns 0.0
/// @note Used internally by CosineSimilarity
///
/// Example:
/// @code
///   std::vector<float> v = {3.0, 4.0};
///   float norm = L2Norm(v);  // norm = sqrt(9 + 16) = 5.0
/// @endcode
float L2Norm(const std::vector<float>& vec);

/// Normalize vector to unit length (L2 normalization)
///
/// Divides each element by the L2 norm so the result has magnitude 1.0.
/// Useful for creating direction vectors independent of scale.
///
/// @param vec Input vector
/// @return Normalized vector with L2 norm = 1.0
///
/// @note Zero-magnitude vectors return zero vector
/// @note Preserves direction, normalizes magnitude
///
/// Example:
/// @code
///   std::vector<float> v = {3.0, 4.0};
///   auto normalized = NormalizeL2(v);  // normalized = {0.6, 0.8}
///   float norm = L2Norm(normalized);  // norm = 1.0
/// @endcode
std::vector<float> NormalizeL2(const std::vector<float>& vec);

/// Combine two scores with weighted averaging
///
/// Computes: weight_a * score_a + weight_b * score_b
///
/// Commonly used to combine attention scores with association strengths.
/// Weights should typically sum to 1.0 for proper normalization.
///
/// @param score_a First score
/// @param score_b Second score
/// @param weight_a Weight for first score
/// @param weight_b Weight for second score
/// @return Combined weighted score
///
/// @note No validation that weights sum to 1.0
/// @note Negative weights are allowed (though unusual)
///
/// Example:
/// @code
///   float attention_score = 0.8;
///   float association_score = 0.6;
///   float combined = CombineScores(attention_score, association_score, 0.4, 0.6);
///   // combined = 0.4 * 0.8 + 0.6 * 0.6 = 0.68
/// @endcode
float CombineScores(
    float score_a,
    float score_b,
    float weight_a,
    float weight_b
);

/// Clamp value to range [min_val, max_val]
///
/// Ensures value stays within specified bounds.
/// Useful for preventing numerical overflow/underflow.
///
/// @param value Input value
/// @param min_val Minimum allowed value
/// @param max_val Maximum allowed value
/// @return Clamped value in [min_val, max_val]
///
/// @note If min_val > max_val, behavior is undefined
///
/// Example:
/// @code
///   float clamped = Clamp(1.5, 0.0, 1.0);  // clamped = 1.0
///   float clamped2 = Clamp(-0.5, 0.0, 1.0);  // clamped2 = 0.0
/// @endcode
float Clamp(float value, float min_val, float max_val);

/// Apply temperature scaling to scores
///
/// Divides all scores by temperature parameter.
/// Higher temperature makes distribution more uniform.
/// Lower temperature makes distribution more peaked.
///
/// @param scores Input scores
/// @param temperature Temperature parameter (must be > 0)
/// @return Temperature-scaled scores
///
/// @note Temperature must be positive; if not, returns scores unchanged
/// @note Typically followed by Softmax normalization
///
/// Example:
/// @code
///   std::vector<float> scores = {2.0, 1.0, 0.0};
///   auto scaled = ApplyTemperature(scores, 2.0);
///   // scaled = {1.0, 0.5, 0.0} (each divided by 2.0)
/// @endcode
std::vector<float> ApplyTemperature(
    const std::vector<float>& scores,
    float temperature
);

/// Check if floating point value is valid (not NaN or infinite)
///
/// @param value Value to check
/// @return true if value is finite and not NaN, false otherwise
///
/// Example:
/// @code
///   bool valid = IsValid(1.0f);  // true
///   bool invalid = IsValid(NAN);  // false
///   bool invalid2 = IsValid(INFINITY);  // false
/// @endcode
bool IsValid(float value);

/// Safe division with fallback value
///
/// Performs division but returns fallback if divisor is zero or result is invalid.
///
/// @param numerator Numerator value
/// @param denominator Denominator value
/// @param fallback Value to return if division is invalid
/// @return numerator / denominator, or fallback if invalid
///
/// Example:
/// @code
///   float result = SafeDivide(10.0, 2.0, 0.0);  // result = 5.0
///   float safe = SafeDivide(10.0, 0.0, 1.0);  // safe = 1.0 (fallback)
/// @endcode
float SafeDivide(float numerator, float denominator, float fallback = 0.0f);

// ============================================================================
// Template Implementations
// ============================================================================

/// Template implementation for map-based softmax
template<typename KeyType>
std::map<KeyType, float> Softmax(
    const std::map<KeyType, float>& scores,
    float temperature) {

    if (scores.empty()) {
        return {};
    }

    // Extract values into vector for processing
    std::vector<float> values;
    values.reserve(scores.size());

    for (const auto& [key, score] : scores) {
        values.push_back(score);
    }

    // Apply vector softmax
    auto weights = Softmax(values, temperature);

    // Map weights back to keys
    std::map<KeyType, float> result;
    auto key_it = scores.begin();
    auto weight_it = weights.begin();

    while (key_it != scores.end() && weight_it != weights.end()) {
        result[key_it->first] = *weight_it;
        ++key_it;
        ++weight_it;
    }

    return result;
}

} // namespace attention
} // namespace dpan

#endif // DPAN_LEARNING_ATTENTION_UTILS_HPP

