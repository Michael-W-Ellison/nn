// File: include/learning/self_attention.hpp
//
// Self-Attention Mechanism for DPAN
//
// Computes N×N attention matrix where each pattern attends to all other
// patterns in a set. Useful for discovering implicit relationships and
// dependencies between patterns.
//
// Unlike standard attention (single query → multiple candidates), self-attention
// treats all patterns as both queries and keys/values, creating a full
// attention matrix that captures all pairwise relationships.
//
// Example use cases:
// - Discovering clusters of related patterns
// - Finding implicit dependencies in pattern sequences
// - Computing pattern similarity graphs
// - Identifying central/hub patterns in a collection

#ifndef DPAN_LEARNING_SELF_ATTENTION_HPP
#define DPAN_LEARNING_SELF_ATTENTION_HPP

#include "core/types.hpp"
#include "storage/pattern_database.hpp"
#include "similarity/similarity_metric.hpp"
#include <memory>
#include <vector>
#include <map>
#include <mutex>

namespace dpan {

// Forward declaration
class AssociationMatrix;

namespace attention {

/// Normalization mode for attention matrix
enum class NormalizationMode {
    ROW_WISE,     ///< Normalize each row (standard self-attention)
    COLUMN_WISE,  ///< Normalize each column
    BIDIRECTIONAL ///< Normalize both rows and columns
};

/// Configuration for self-attention mechanism
struct SelfAttentionConfig {
    /// Temperature for softmax normalization
    /// Higher values (>1.0) make distribution more uniform
    /// Lower values (<1.0) make distribution more peaked
    /// Default: 1.0
    float temperature = 1.0f;

    /// Whether to mask diagonal (prevent self-attention)
    /// When true, patterns cannot attend to themselves
    /// Default: false
    bool mask_diagonal = false;

    /// Value to use for masked entries (should be very negative for softmax)
    /// Default: -1e9f
    float mask_value = -1e9f;

    /// Normalization mode for attention matrix
    /// Default: ROW_WISE (standard self-attention)
    NormalizationMode normalization = NormalizationMode::ROW_WISE;

    /// Minimum attention threshold (set to 0 if below)
    /// Helps create sparse attention matrices
    /// Default: 0.0 (no thresholding)
    float attention_threshold = 0.0f;

    /// Enable caching of attention matrices
    /// Default: false (recompute each time)
    bool enable_caching = false;

    /// Cache size (number of attention matrices to cache)
    /// Default: 10
    size_t cache_size = 10;

    /// Validate configuration
    bool Validate() const {
        if (temperature <= 0.0f) return false;
        if (attention_threshold < 0.0f || attention_threshold >= 1.0f) return false;
        return true;
    }
};

/// Discovered relationship via self-attention
struct DiscoveredRelationship {
    /// Related pattern ID
    PatternID pattern;

    /// Attention weight (strength of relationship)
    float attention_weight;

    /// Whether this relationship exists in explicit associations
    bool has_explicit_association;

    /// Type of explicit association (if it exists)
    /// Only valid if has_explicit_association is true
    AssociationType explicit_type;

    /// Strength of explicit association (if it exists)
    /// Only valid if has_explicit_association is true
    float explicit_strength;

    /// Whether this is a novel relationship (high attention, no explicit association)
    bool is_novel() const {
        return !has_explicit_association;
    }

    /// Whether this is a confirmed relationship (both implicit and explicit)
    bool is_confirmed() const {
        return has_explicit_association;
    }
};

/// Result of relationship discovery
struct RelationshipDiscoveryResult {
    /// Query pattern
    PatternID query;

    /// All discovered relationships, sorted by attention weight (descending)
    std::vector<DiscoveredRelationship> relationships;

    /// Number of novel relationships (not in explicit associations)
    size_t novel_count() const {
        size_t count = 0;
        for (const auto& rel : relationships) {
            if (rel.is_novel()) ++count;
        }
        return count;
    }

    /// Number of confirmed relationships (in both implicit and explicit)
    size_t confirmed_count() const {
        size_t count = 0;
        for (const auto& rel : relationships) {
            if (rel.is_confirmed()) ++count;
        }
        return count;
    }

    /// Get only novel relationships
    std::vector<DiscoveredRelationship> get_novel_relationships() const {
        std::vector<DiscoveredRelationship> novel;
        for (const auto& rel : relationships) {
            if (rel.is_novel()) {
                novel.push_back(rel);
            }
        }
        return novel;
    }

    /// Get only confirmed relationships
    std::vector<DiscoveredRelationship> get_confirmed_relationships() const {
        std::vector<DiscoveredRelationship> confirmed;
        for (const auto& rel : relationships) {
            if (rel.is_confirmed()) {
                confirmed.push_back(rel);
            }
        }
        return confirmed;
    }
};

/// Self-attention mechanism
///
/// Computes N×N attention matrix for a set of patterns, where entry (i,j)
/// represents how much pattern i attends to pattern j.
///
/// Example usage:
/// @code
///   SelfAttentionConfig config;
///   config.mask_diagonal = true;  // Prevent self-attention
///   auto self_attn = std::make_unique<SelfAttention>(config);
///   self_attn->SetPatternDatabase(pattern_db);
///
///   std::vector<PatternID> patterns = {p1, p2, p3, p4};
///   auto matrix = self_attn->ComputeAttentionMatrix(patterns);
///
///   // Access attention from pattern i to pattern j
///   float attention = matrix[{p_i, p_j}];
/// @endcode
class SelfAttention {
public:
    /// Constructor
    ///
    /// @param config Self-attention configuration
    explicit SelfAttention(const SelfAttentionConfig& config = SelfAttentionConfig());

    /// Destructor
    ~SelfAttention() = default;

    // ========================================================================
    // Core Methods
    // ========================================================================

    /// Compute attention matrix for a set of patterns
    ///
    /// Returns N×N attention matrix where entry (i,j) represents
    /// how much pattern i attends to pattern j.
    ///
    /// @param patterns Set of patterns to compute attention for
    /// @param context Current context vector (optional)
    /// @return Attention matrix as map from (query, key) pairs to attention weights
    std::map<std::pair<PatternID, PatternID>, float> ComputeAttentionMatrix(
        const std::vector<PatternID>& patterns,
        const ContextVector& context = ContextVector()
    );

    /// Compute attention matrix and return as 2D vector
    ///
    /// More efficient for dense matrices. Returns vector where
    /// result[i][j] is attention from patterns[i] to patterns[j].
    ///
    /// @param patterns Set of patterns to compute attention for
    /// @param context Current context vector (optional)
    /// @return 2D vector of attention weights [N x N]
    std::vector<std::vector<float>> ComputeAttentionMatrixDense(
        const std::vector<PatternID>& patterns,
        const ContextVector& context = ContextVector()
    );

    /// Get attention weights for a specific query pattern
    ///
    /// Extracts a single row from the attention matrix.
    ///
    /// @param query Query pattern
    /// @param patterns Set of patterns (including query)
    /// @param context Current context vector (optional)
    /// @return Map from key patterns to attention weights
    std::map<PatternID, float> GetQueryAttention(
        PatternID query,
        const std::vector<PatternID>& patterns,
        const ContextVector& context = ContextVector()
    );

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Set pattern database
    ///
    /// @param db Pointer to pattern database
    void SetPatternDatabase(PatternDatabase* db);

    /// Set similarity metric
    ///
    /// @param metric Shared pointer to similarity metric
    void SetSimilarityMetric(std::shared_ptr<SimilarityMetric> metric);

    /// Set association matrix for comparing with explicit associations
    ///
    /// @param matrix Pointer to association matrix
    void SetAssociationMatrix(AssociationMatrix* matrix);

    /// Get current configuration
    ///
    /// @return Reference to self-attention configuration
    const SelfAttentionConfig& GetConfig() const;

    /// Set configuration
    ///
    /// @param config New self-attention configuration
    void SetConfig(const SelfAttentionConfig& config);

    /// Clear attention matrix cache
    void ClearCache();

    // ========================================================================
    // Relationship Discovery
    // ========================================================================

    /// Discover related patterns using self-attention
    ///
    /// Uses self-attention to find patterns related to the query pattern.
    /// Compares discovered relationships with explicit associations to
    /// identify novel relationships.
    ///
    /// @param query_pattern Query pattern to find relationships for
    /// @param candidate_patterns Set of candidate patterns to consider
    /// @param top_k Number of top relationships to return
    /// @param context Current context vector (optional)
    /// @return Relationship discovery result with novel and confirmed relationships
    RelationshipDiscoveryResult DiscoverRelatedPatterns(
        PatternID query_pattern,
        const std::vector<PatternID>& candidate_patterns,
        size_t top_k,
        const ContextVector& context = ContextVector()
    );

    // ========================================================================
    // Analysis and Utilities
    // ========================================================================

    /// Find most attended patterns (highest average attention received)
    ///
    /// @param patterns Set of patterns
    /// @param top_k Number of top patterns to return
    /// @param context Current context vector (optional)
    /// @return Vector of (pattern, avg_attention) pairs, sorted by attention
    std::vector<std::pair<PatternID, float>> FindMostAttendedPatterns(
        const std::vector<PatternID>& patterns,
        size_t top_k,
        const ContextVector& context = ContextVector()
    );

    /// Find most attentive patterns (highest average attention given)
    ///
    /// @param patterns Set of patterns
    /// @param top_k Number of top patterns to return
    /// @param context Current context vector (optional)
    /// @return Vector of (pattern, avg_attention) pairs, sorted by attention
    std::vector<std::pair<PatternID, float>> FindMostAttentivePatterns(
        const std::vector<PatternID>& patterns,
        size_t top_k,
        const ContextVector& context = ContextVector()
    );

    /// Compute attention entropy for each pattern
    ///
    /// High entropy = pattern attends to many patterns uniformly
    /// Low entropy = pattern focuses attention on few patterns
    ///
    /// @param patterns Set of patterns
    /// @param context Current context vector (optional)
    /// @return Map from pattern to entropy value
    std::map<PatternID, float> ComputeAttentionEntropy(
        const std::vector<PatternID>& patterns,
        const ContextVector& context = ContextVector()
    );

    /// Get statistics about attention computations
    ///
    /// @return Map of statistic name to value
    std::map<std::string, float> GetStatistics() const;

protected:
    /// Compute pairwise similarity scores
    ///
    /// @param patterns Set of patterns
    /// @param context Current context vector
    /// @return 2D vector of similarity scores [N x N]
    std::vector<std::vector<float>> ComputeSimilarityMatrix(
        const std::vector<PatternID>& patterns,
        const ContextVector& context
    );

    /// Apply softmax normalization to attention matrix
    ///
    /// @param scores Raw similarity scores [N x N]
    /// @return Normalized attention weights [N x N]
    std::vector<std::vector<float>> ApplySoftmax(
        const std::vector<std::vector<float>>& scores
    );

    /// Apply row-wise softmax
    ///
    /// @param scores Raw scores [N x N]
    /// @return Row-normalized weights [N x N]
    std::vector<std::vector<float>> ApplyRowWiseSoftmax(
        const std::vector<std::vector<float>>& scores
    );

    /// Apply column-wise softmax
    ///
    /// @param scores Raw scores [N x N]
    /// @return Column-normalized weights [N x N]
    std::vector<std::vector<float>> ApplyColumnWiseSoftmax(
        const std::vector<std::vector<float>>& scores
    );

    /// Apply bidirectional normalization (row then column)
    ///
    /// @param scores Raw scores [N x N]
    /// @return Bidirectionally normalized weights [N x N]
    std::vector<std::vector<float>> ApplyBidirectionalNormalization(
        const std::vector<std::vector<float>>& scores
    );

    /// Apply attention threshold (sparsify matrix)
    ///
    /// @param attention Attention weights [N x N]
    void ApplyThreshold(std::vector<std::vector<float>>& attention);

    /// Get similarity between two patterns
    ///
    /// @param p1 First pattern
    /// @param p2 Second pattern
    /// @param context Current context vector
    /// @return Similarity score
    float GetSimilarity(PatternID p1, PatternID p2, const ContextVector& context);

    /// Generate cache key for patterns
    ///
    /// @param patterns Set of patterns
    /// @return Cache key string
    std::string GenerateCacheKey(const std::vector<PatternID>& patterns) const;

private:
    /// Self-attention configuration
    SelfAttentionConfig config_;

    /// Pattern database pointer
    PatternDatabase* pattern_db_;

    /// Association matrix pointer (for comparing with explicit associations)
    AssociationMatrix* association_matrix_;

    /// Similarity metric for computing pairwise similarities
    std::shared_ptr<SimilarityMetric> similarity_metric_;

    /// Cache of computed attention matrices
    /// Maps from cache key to attention matrix
    std::map<std::string, std::vector<std::vector<float>>> cache_;

    /// Mutex for thread-safe operations
    mutable std::mutex mutex_;

    /// Statistics counters
    mutable size_t matrix_computations_;
    mutable size_t cache_hits_;
    mutable size_t cache_misses_;
};

} // namespace attention
} // namespace dpan

#endif // DPAN_LEARNING_SELF_ATTENTION_HPP
