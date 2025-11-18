// File: include/learning/attention_mechanism.hpp
//
// Attention Mechanism Interface for DPAN
// Enables context-aware, importance-weighted pattern selection
//
// The attention mechanism computes dynamic weights for patterns based on:
// - Query-key similarity (how relevant is this pattern?)
// - Pattern importance (intrinsic value based on usage, confidence, etc.)
// - Context alignment (how well does it fit current context?)
// - Multi-head perspectives (multiple attention viewpoints)
//
// This improves prediction quality by focusing on the most relevant patterns
// rather than treating all patterns equally.

#ifndef DPAN_LEARNING_ATTENTION_MECHANISM_HPP
#define DPAN_LEARNING_ATTENTION_MECHANISM_HPP

#include "core/types.hpp"
#include <vector>
#include <map>
#include <string>
#include <memory>

namespace dpan {

// Forward declarations
class PatternDatabase;

/// Configuration for attention mechanisms
struct AttentionConfig {
    /// Number of attention heads for multi-head attention
    /// More heads provide diverse perspectives but increase computation
    /// Default: 4 (semantic, temporal, structural, association)
    size_t num_heads = 4;

    /// Temperature parameter for softmax normalization
    /// Higher values (>1.0) make distribution more uniform
    /// Lower values (<1.0) make distribution more peaked
    /// Default: 1.0 (standard softmax)
    float temperature = 1.0f;

    /// Enable context-aware attention scoring
    /// When true, considers context similarity in attention weights
    /// Default: true
    bool use_context = true;

    /// Enable pattern importance weighting
    /// When true, considers intrinsic pattern importance
    /// (frequency, confidence, association richness, success rate)
    /// Default: true
    bool use_importance = true;

    /// Type of attention computation
    /// Options:
    /// - "dot_product": Scaled dot-product attention (fast, Transformer-style)
    /// - "additive": Additive attention (Bahdanau-style, more parameters)
    /// - "multiplicative": Multiplicative attention (Luong-style)
    /// Default: "dot_product"
    std::string attention_type = "dot_product";

    /// Weight for combining association strength and attention scores
    /// Final score = association_weight * assoc_strength + attention_weight * attention
    /// Should sum to 1.0 for proper normalization
    /// Default: 0.6 (favor associations) + 0.4 (attention boost)
    float association_weight = 0.6f;
    float attention_weight = 0.4f;

    /// Enable caching of attention computations
    /// Significant speedup for repeated queries
    /// Default: true
    bool enable_caching = true;

    /// Maximum size of attention cache (in number of cached computations)
    /// Uses LRU eviction policy
    /// Default: 1000
    size_t cache_size = 1000;

    /// Enable debug logging for attention computation
    /// Shows intermediate values, useful for debugging
    /// Default: false (minimal performance impact when disabled)
    bool debug_logging = false;

    /// Validate configuration
    bool Validate() const {
        if (num_heads == 0) return false;
        if (temperature <= 0.0f) return false;
        if (association_weight < 0.0f || attention_weight < 0.0f) return false;
        if (association_weight + attention_weight <= 0.0f) return false;
        if (attention_type != "dot_product" &&
            attention_type != "additive" &&
            attention_type != "multiplicative") {
            return false;
        }
        return true;
    }
};

/// Result of attention computation for a single candidate pattern
struct AttentionScore {
    /// Pattern ID being scored
    PatternID pattern_id;

    /// Final attention weight [0.0, 1.0]
    /// After softmax normalization, all weights sum to 1.0
    float weight;

    /// Raw attention score (before normalization)
    /// Useful for debugging and analysis
    float raw_score;

    /// Breakdown of score components (for debugging/explanation)
    struct Components {
        float semantic_similarity = 0.0f;  // Content-based similarity
        float context_similarity = 0.0f;   // Context alignment
        float importance_score = 0.0f;     // Intrinsic pattern importance
        float temporal_score = 0.0f;       // Recency-based score
        float structural_score = 0.0f;     // Structure-based similarity
    } components;

    /// Constructor
    AttentionScore() : pattern_id(0), weight(0.0f), raw_score(0.0f) {}

    AttentionScore(PatternID id, float w, float rs = 0.0f)
        : pattern_id(id), weight(w), raw_score(rs) {}
};

/// Abstract base class for attention mechanisms
///
/// Attention mechanisms compute dynamic importance weights for patterns
/// to improve prediction quality through context-aware selection.
///
/// Implementations should:
/// 1. Compute attention scores for candidate patterns
/// 2. Normalize scores to valid probability distribution
/// 3. Optionally combine with existing association strengths
/// 4. Return weighted, ranked predictions
///
/// Example usage:
/// @code
///   // Create attention mechanism
///   auto attention = std::make_unique<BasicAttentionMechanism>(config);
///   attention->SetPatternDatabase(pattern_db);
///
///   // Compute attention for candidates
///   std::vector<PatternID> candidates = {id1, id2, id3};
///   auto weights = attention->ComputeAttention(query, candidates, context);
///
///   // Apply to predictions
///   auto weighted = attention->ApplyAttention(query, predictions, context);
/// @endcode
class AttentionMechanism {
public:
    virtual ~AttentionMechanism() = default;

    /// Compute attention weights for candidate patterns
    ///
    /// Given a query pattern and a set of candidate patterns, computes
    /// attention weights that represent the relevance of each candidate.
    /// Weights are normalized to sum to 1.0 (probability distribution).
    ///
    /// @param query The query pattern (current input/context)
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector (optional, can be empty)
    /// @return Map from pattern ID to attention weight [0.0, 1.0]
    ///
    /// @note Returned weights sum to 1.0 across all candidates
    /// @note Empty candidates returns empty map
    /// @note Invalid query returns uniform weights (1/N for each)
    virtual std::map<PatternID, float> ComputeAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) = 0;

    /// Compute detailed attention scores with component breakdown
    ///
    /// Similar to ComputeAttention() but returns full AttentionScore
    /// objects with detailed breakdown of score components.
    /// Useful for explanation, debugging, and visualization.
    ///
    /// @param query The query pattern
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector
    /// @return Vector of detailed attention scores
    ///
    /// @note Scores are sorted by weight (descending)
    virtual std::vector<AttentionScore> ComputeDetailedAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) = 0;

    /// Apply attention to weight and rank predictions
    ///
    /// Combines attention weights with existing association strengths
    /// to produce final weighted predictions. The combination uses
    /// configurable weights (typically 60% association, 40% attention).
    ///
    /// @param query The query pattern
    /// @param predictions Initial predictions (typically from association matrix)
    /// @param context Current context vector
    /// @return Weighted and re-ranked predictions (top-k by combined score)
    ///
    /// @note If config.association_weight is 0, uses pure attention
    /// @note If config.attention_weight is 0, returns predictions unchanged
    virtual std::vector<std::pair<PatternID, float>> ApplyAttention(
        PatternID query,
        const std::vector<PatternID>& predictions,
        const ContextVector& context
    ) = 0;

    /// Set pattern database for retrieving pattern information
    ///
    /// The attention mechanism needs access to pattern data to compute
    /// features, confidence scores, and other metadata.
    ///
    /// @param db Pointer to pattern database (must outlive this object)
    virtual void SetPatternDatabase(PatternDatabase* db) = 0;

    /// Get current configuration
    ///
    /// @return Reference to configuration struct
    virtual const AttentionConfig& GetConfig() const = 0;

    /// Update configuration
    ///
    /// Allows dynamic adjustment of attention parameters.
    /// Changes take effect immediately.
    ///
    /// @param config New configuration
    /// @throws std::invalid_argument if config.Validate() fails
    virtual void SetConfig(const AttentionConfig& config) = 0;

    /// Clear any cached attention computations
    ///
    /// Useful when pattern database changes significantly or
    /// when memory pressure requires cache eviction.
    virtual void ClearCache() = 0;

    /// Get statistics about attention usage
    ///
    /// Useful for monitoring and debugging.
    ///
    /// @return Map of statistic name to value
    virtual std::map<std::string, float> GetStatistics() const = 0;
};

} // namespace dpan

#endif // DPAN_LEARNING_ATTENTION_MECHANISM_HPP
