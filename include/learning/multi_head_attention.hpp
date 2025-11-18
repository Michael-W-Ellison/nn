// File: include/learning/multi_head_attention.hpp
//
// Multi-Head Attention Mechanism for DPAN
//
// Combines multiple attention mechanisms (heads) with weighted averaging
// to capture different perspectives on pattern relevance.
//
// Each head can use a different attention strategy:
// - Semantic head: Focus on content similarity
// - Temporal head: Focus on recency and temporal patterns
// - Context head: Focus on contextual alignment
// - Structural head: Focus on structural similarity
//
// The final attention weights are computed as a weighted combination
// of all head outputs, allowing the mechanism to balance multiple
// relevance factors.

#ifndef DPAN_LEARNING_MULTI_HEAD_ATTENTION_HPP
#define DPAN_LEARNING_MULTI_HEAD_ATTENTION_HPP

#include "learning/attention_mechanism.hpp"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <mutex>

namespace dpan {
namespace attention {

/// Represents a single attention head in multi-head attention
struct AttentionHead {
    /// Human-readable name for this head (e.g., "semantic", "temporal")
    std::string name;

    /// The attention mechanism for this head
    /// Each head can use a different attention strategy
    std::shared_ptr<AttentionMechanism> mechanism;

    /// Weight for combining this head's output with others
    /// Should be in range [0.0, 1.0]
    /// All head weights should sum to 1.0 for proper normalization
    float weight;

    /// Constructor
    AttentionHead() : weight(0.0f) {}

    AttentionHead(const std::string& n,
                  std::shared_ptr<AttentionMechanism> m,
                  float w = 1.0f)
        : name(n), mechanism(m), weight(w) {}

    /// Validate head configuration
    bool Validate() const {
        if (name.empty()) return false;
        if (!mechanism) return false;
        if (weight < 0.0f || weight > 1.0f) return false;
        return true;
    }
};

/// Configuration for multi-head attention
struct MultiHeadConfig {
    /// Whether to automatically normalize head weights to sum to 1.0
    /// Default: true
    bool auto_normalize_weights = true;

    /// Whether to enable parallel computation of heads
    /// When true, heads are computed concurrently (future optimization)
    /// Default: false (sequential computation)
    bool parallel_heads = false;

    /// Temperature for final softmax (if combining raw scores)
    /// Default: 1.0
    float temperature = 1.0f;

    /// Enable debug logging
    /// Default: false
    bool debug_logging = false;

    /// Validate configuration
    bool Validate() const {
        if (temperature <= 0.0f) return false;
        return true;
    }
};

/// Multi-Head Attention Mechanism
///
/// Combines multiple attention mechanisms (heads) to capture diverse
/// perspectives on pattern relevance. Each head computes attention weights
/// independently, and the final weights are a weighted combination of all
/// head outputs.
///
/// Example usage:
/// @code
///   // Create multi-head attention
///   MultiHeadConfig config;
///   auto multi_head = std::make_unique<MultiHeadAttention>(config);
///
///   // Add heads with different strategies
///   auto semantic = std::make_shared<BasicAttentionMechanism>(semantic_config);
///   auto context = std::make_shared<ContextAwareAttention>(context_config);
///
///   multi_head->AddHead("semantic", semantic, 0.6f);
///   multi_head->AddHead("context", context, 0.4f);
///
///   // Compute combined attention
///   auto weights = multi_head->ComputeAttention(query, candidates, context);
/// @endcode
class MultiHeadAttention : public AttentionMechanism {
public:
    /// Constructor
    ///
    /// @param config Multi-head attention configuration
    explicit MultiHeadAttention(const MultiHeadConfig& config = MultiHeadConfig());

    /// Destructor
    ~MultiHeadAttention() override = default;

    // ========================================================================
    // AttentionMechanism Interface Implementation
    // ========================================================================

    /// Compute attention weights by combining all heads
    ///
    /// @param query The query pattern
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector
    /// @return Map from pattern ID to combined attention weight
    std::map<PatternID, float> ComputeAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Compute detailed attention scores with per-head breakdown
    ///
    /// @param query The query pattern
    /// @param candidates List of candidate patterns to score
    /// @param context Current context vector
    /// @return Vector of detailed attention scores
    std::vector<AttentionScore> ComputeDetailedAttention(
        PatternID query,
        const std::vector<PatternID>& candidates,
        const ContextVector& context
    ) override;

    /// Apply attention to weight and rank predictions
    ///
    /// @param query The query pattern
    /// @param predictions Initial predictions
    /// @param context Current context vector
    /// @return Weighted and re-ranked predictions
    std::vector<std::pair<PatternID, float>> ApplyAttention(
        PatternID query,
        const std::vector<PatternID>& predictions,
        const ContextVector& context
    ) override;

    /// Set pattern database for all heads
    ///
    /// @param db Pointer to pattern database
    void SetPatternDatabase(PatternDatabase* db) override;

    /// Get current configuration (returns base AttentionConfig)
    ///
    /// @return Reference to attention configuration
    const AttentionConfig& GetConfig() const override;

    /// Update configuration for all heads
    ///
    /// @param config New attention configuration
    void SetConfig(const AttentionConfig& config) override;

    /// Clear caches for all heads
    void ClearCache() override;

    /// Get statistics about multi-head attention usage
    ///
    /// Includes statistics from all heads plus combination metrics
    ///
    /// @return Map of statistic name to value
    std::map<std::string, float> GetStatistics() const override;

    // ========================================================================
    // Multi-Head Specific Methods
    // ========================================================================

    /// Add an attention head
    ///
    /// @param name Unique name for this head
    /// @param mechanism Attention mechanism for this head
    /// @param weight Weight for combining this head's output [0.0, 1.0]
    /// @return true if head was added, false if name already exists
    bool AddHead(const std::string& name,
                 std::shared_ptr<AttentionMechanism> mechanism,
                 float weight = 1.0f);

    /// Remove an attention head
    ///
    /// @param name Name of head to remove
    /// @return true if head was removed, false if not found
    bool RemoveHead(const std::string& name);

    /// Get an attention head by name
    ///
    /// @param name Name of head to retrieve
    /// @return Pointer to head if found, nullptr otherwise
    const AttentionHead* GetHead(const std::string& name) const;

    /// Get all attention heads
    ///
    /// @return Vector of all heads
    const std::vector<AttentionHead>& GetHeads() const;

    /// Update weight for a specific head
    ///
    /// @param name Name of head to update
    /// @param weight New weight value
    /// @return true if head was found and updated, false otherwise
    bool SetHeadWeight(const std::string& name, float weight);

    /// Get number of heads
    ///
    /// @return Number of attention heads
    size_t GetNumHeads() const;

    /// Normalize head weights to sum to 1.0
    ///
    /// Adjusts all head weights proportionally so they sum to 1.0
    void NormalizeWeights();

    /// Validate that all heads are properly configured
    ///
    /// @return true if all heads are valid and weights are positive
    bool ValidateHeads() const;

    /// Get multi-head specific configuration
    ///
    /// @return Reference to multi-head configuration
    const MultiHeadConfig& GetMultiHeadConfig() const;

    /// Set multi-head specific configuration
    ///
    /// @param config New multi-head configuration
    void SetMultiHeadConfig(const MultiHeadConfig& config);

protected:
    /// Combine attention weights from multiple heads
    ///
    /// @param head_weights Vector of weight maps, one per head
    /// @return Combined weight map
    std::map<PatternID, float> CombineHeadWeights(
        const std::vector<std::map<PatternID, float>>& head_weights
    ) const;

    /// Log debug information if debug_logging is enabled
    ///
    /// @param message Debug message
    void LogDebug(const std::string& message) const;

    /// Normalize weights without locking (caller must hold mutex)
    void NormalizeWeightsUnsafe();

private:
    /// Multi-head configuration
    MultiHeadConfig config_;

    /// Base attention configuration (shared across heads)
    AttentionConfig base_config_;

    /// Vector of attention heads
    std::vector<AttentionHead> heads_;

    /// Pattern database pointer (shared with all heads)
    PatternDatabase* pattern_db_;

    /// Mutex for thread-safe head management
    mutable std::mutex heads_mutex_;

    /// Statistics counters
    mutable size_t attention_computations_;
    mutable size_t head_combinations_;
};

} // namespace attention
} // namespace dpan

#endif // DPAN_LEARNING_MULTI_HEAD_ATTENTION_HPP
