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
#include <set>
#include <mutex>
#include <variant>

namespace dpan {

// Forward declarations
class AssociationMatrix;

namespace attention {

/// Supported attention head types
enum class AttentionHeadType {
    SEMANTIC,     ///< Content-based similarity attention
    TEMPORAL,     ///< Recency-based attention
    STRUCTURAL,   ///< Pattern structure similarity attention
    ASSOCIATION,  ///< Association strength-based attention
    BASIC,        ///< Basic attention mechanism
    CONTEXT       ///< Context-aware attention
};

/// Convert head type to string
inline std::string HeadTypeToString(AttentionHeadType type) {
    switch (type) {
        case AttentionHeadType::SEMANTIC:    return "semantic";
        case AttentionHeadType::TEMPORAL:    return "temporal";
        case AttentionHeadType::STRUCTURAL:  return "structural";
        case AttentionHeadType::ASSOCIATION: return "association";
        case AttentionHeadType::BASIC:       return "basic";
        case AttentionHeadType::CONTEXT:     return "context";
        default:                              return "unknown";
    }
}

/// Convert string to head type
inline bool StringToHeadType(const std::string& str, AttentionHeadType& type) {
    if (str == "semantic") {
        type = AttentionHeadType::SEMANTIC;
        return true;
    } else if (str == "temporal") {
        type = AttentionHeadType::TEMPORAL;
        return true;
    } else if (str == "structural") {
        type = AttentionHeadType::STRUCTURAL;
        return true;
    } else if (str == "association") {
        type = AttentionHeadType::ASSOCIATION;
        return true;
    } else if (str == "basic") {
        type = AttentionHeadType::BASIC;
        return true;
    } else if (str == "context") {
        type = AttentionHeadType::CONTEXT;
        return true;
    }
    return false;
}

/// Configuration for a single attention head
struct HeadConfig {
    /// Unique name for this head
    std::string name;

    /// Type of attention mechanism for this head
    AttentionHeadType type;

    /// Weight for combining this head's output [0.0, 1.0]
    float weight = 1.0f;

    /// Head-specific parameters
    /// Common parameters across heads:
    /// - "temperature": Softmax temperature (all heads)
    /// - "enable_caching": Enable/disable caching (all heads)
    /// - "cache_size": Cache size (all heads)
    ///
    /// Semantic-specific:
    /// - "similarity_threshold": Minimum similarity threshold
    /// - "similarity_metric": Similarity metric type
    ///
    /// Temporal-specific:
    /// - "decay_constant_ms": Temporal decay constant in milliseconds
    /// - "min_age_threshold_ms": Minimum age threshold
    ///
    /// Structural-specific:
    /// - "jaccard_weight": Weight for Jaccard similarity
    /// - "size_weight": Weight for size similarity
    /// - "similarity_threshold": Minimum similarity threshold
    /// - "atomic_penalty": Penalty for atomic vs composite comparison
    ///
    /// Association-specific:
    /// - "use_contextual_strength": Use contextual strength (0=false, 1=true)
    /// - "strength_threshold": Minimum association strength
    /// - "default_strength": Default strength for missing associations
    std::map<std::string, float> parameters;

    /// Validate head configuration
    bool Validate() const {
        if (name.empty()) return false;
        if (weight < 0.0f || weight > 1.0f) return false;
        // Type is always valid as it's an enum
        return true;
    }
};

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

    /// Head configurations for automatic initialization
    /// If non-empty, heads will be created from these configs
    std::vector<HeadConfig> head_configs;

    /// Validate configuration
    bool Validate() const {
        if (temperature <= 0.0f) return false;

        // Validate all head configs
        for (const auto& head_config : head_configs) {
            if (!head_config.Validate()) return false;
        }

        // Check for duplicate head names
        std::set<std::string> names;
        for (const auto& head_config : head_configs) {
            if (names.count(head_config.name) > 0) {
                return false;  // Duplicate name
            }
            names.insert(head_config.name);
        }

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

    /// Initialize heads from configuration
    ///
    /// Creates and adds attention heads based on head configurations.
    /// This method clears existing heads and creates new ones from the config.
    ///
    /// @param head_configs Vector of head configurations
    /// @param pattern_db Pattern database (required for all heads)
    /// @param association_matrix Association matrix (required for association heads)
    /// @return true if all heads were successfully created, false otherwise
    bool InitializeHeadsFromConfig(
        const std::vector<HeadConfig>& head_configs,
        PatternDatabase* pattern_db,
        AssociationMatrix* association_matrix = nullptr);

    /// Create a single attention head from configuration
    ///
    /// Factory method that creates the appropriate head type based on config.
    ///
    /// @param config Head configuration
    /// @param pattern_db Pattern database (required for all heads)
    /// @param association_matrix Association matrix (required for association heads)
    /// @return Shared pointer to created head, or nullptr if creation failed
    std::shared_ptr<AttentionMechanism> CreateHeadFromConfig(
        const HeadConfig& config,
        PatternDatabase* pattern_db,
        AssociationMatrix* association_matrix = nullptr);

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
