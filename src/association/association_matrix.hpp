// File: src/association/association_matrix.hpp
#pragma once

#include "association/association_edge.hpp"
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <optional>

namespace dpan {

/// Hash function for (PatternID, PatternID) pairs
#ifndef DPAN_PATTERN_PAIR_HASH_DEFINED
#define DPAN_PATTERN_PAIR_HASH_DEFINED
struct PatternPairHash {
    size_t operator()(const std::pair<PatternID, PatternID>& p) const {
        size_t h1 = PatternID::Hash()(p.first);
        size_t h2 = PatternID::Hash()(p.second);
        return h1 ^ (h2 << 1);  // Combine hashes
    }
};
#endif // DPAN_PATTERN_PAIR_HASH_DEFINED

/// AssociationMatrix: Sparse directed graph of pattern associations
///
/// Storage uses a hybrid approach combining:
/// - CSR-like row indices for efficient outgoing edge lookup
/// - Reverse index for incoming edge lookup
/// - Direct hash-based (source, target) lookup
/// - Type index for filtering by association type
///
/// Thread-safe with reader-writer locking (std::shared_mutex)
class AssociationMatrix {
public:
    /// Configuration for matrix behavior
    struct Config {
        Config() = default;
        size_t initial_capacity{10000};
        bool enable_reverse_lookup{true};
        bool enable_type_index{true};
        float load_factor_threshold{0.75f};
    };

    /// Activation propagation result
    struct ActivationResult {
        PatternID pattern;
        float activation;
    };

    // ========================================================================
    // Construction
    // ========================================================================

    AssociationMatrix();
    explicit AssociationMatrix(const Config& config);
    ~AssociationMatrix() = default;

    // ========================================================================
    // Add/Update/Remove Operations
    // ========================================================================

    /// Add new association (returns false if already exists)
    bool AddAssociation(const AssociationEdge& edge);

    /// Update existing association (returns false if doesn't exist)
    bool UpdateAssociation(PatternID source, PatternID target, const AssociationEdge& edge);

    /// Remove association (returns false if doesn't exist)
    bool RemoveAssociation(PatternID source, PatternID target);

    // ========================================================================
    // Lookup Operations
    // ========================================================================

    /// Get specific association (O(1) lookup)
    /// @return Pointer to edge (nullptr if not found). Pointer valid while matrix unchanged.
    const AssociationEdge* GetAssociation(PatternID source, PatternID target) const;

    /// Check if association exists (O(1) lookup)
    bool HasAssociation(PatternID source, PatternID target) const;

    /// Get all outgoing associations from source pattern
    /// @return Vector of pointers to edges (valid while matrix unchanged)
    std::vector<const AssociationEdge*> GetOutgoingAssociations(PatternID source) const;

    /// Get all incoming associations to target pattern
    /// @return Vector of pointers to edges (valid while matrix unchanged)
    std::vector<const AssociationEdge*> GetIncomingAssociations(PatternID target) const;

    /// Get all associations of a specific type
    /// @return Vector of pointers to edges (valid while matrix unchanged)
    std::vector<const AssociationEdge*> GetAssociationsByType(AssociationType type) const;

    /// Get neighbor patterns (outgoing=true for successors, false for predecessors)
    std::vector<PatternID> GetNeighbors(PatternID pattern, bool outgoing = true) const;

    /// Get mutual neighbors (patterns that are both predecessors and successors)
    std::vector<PatternID> GetMutualNeighbors(PatternID pattern) const;

    // ========================================================================
    // Strength Operations
    // ========================================================================

    /// Strengthen association by amount (bounded to [0,1])
    bool StrengthenAssociation(PatternID source, PatternID target, float amount);

    /// Weaken association by amount (bounded to [0,1])
    bool WeakenAssociation(PatternID source, PatternID target, float amount);

    /// Apply time-based decay to all associations
    void ApplyDecayAll(Timestamp::Duration elapsed_time);

    /// Apply decay only to associations involving specific pattern
    void ApplyDecayPattern(PatternID pattern, Timestamp::Duration elapsed_time);

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get total number of associations (excluding deleted)
    size_t GetAssociationCount() const;

    /// Get number of unique patterns with associations
    size_t GetPatternCount() const;

    /// Get average degree (average number of outgoing edges per pattern)
    float GetAverageDegree() const;

    /// Get average association strength
    float GetAverageStrength() const;

    /// Get graph density (fraction of possible edges that exist)
    float GetDensity() const;

    // ========================================================================
    // Graph Properties
    // ========================================================================

    /// Get degree of specific pattern (number of edges)
    size_t GetDegree(PatternID pattern, bool outgoing = true) const;

    /// Get patterns with no associations
    std::vector<PatternID> GetIsolatedPatterns() const;

    /// Get all patterns with associations (source or target)
    /// @return Vector of all pattern IDs
    std::vector<PatternID> GetAllPatterns() const;

    // ========================================================================
    // Activation Propagation
    // ========================================================================

    /// Propagate activation through association graph using BFS
    /// @param source Starting pattern
    /// @param initial_activation Initial activation strength [0,1]
    /// @param max_hops Maximum propagation distance
    /// @param min_activation Minimum activation threshold to continue propagation
    /// @param context Optional context for context-aware strength modulation
    /// @return Sorted list of activated patterns (descending by activation)
    std::vector<ActivationResult> PropagateActivation(
        PatternID source,
        float initial_activation,
        size_t max_hops = 3,
        float min_activation = 0.01f,
        const ContextVector* context = nullptr
    ) const;

    // ========================================================================
    // Serialization
    // ========================================================================

    /// Serialize matrix to output stream
    void Serialize(std::ostream& out) const;

    /// Deserialize matrix from input stream
    /// @return Unique pointer to deserialized matrix
    static std::unique_ptr<AssociationMatrix> Deserialize(std::istream& in);

    // ========================================================================
    // Memory Management
    // ========================================================================

    /// Compact storage by removing deleted edges and rebuilding indices
    void Compact();

    /// Clear all associations
    void Clear();

    /// Estimate total memory usage in bytes
    size_t EstimateMemoryUsage() const;

    // ========================================================================
    // Debugging
    // ========================================================================

    /// Print statistics to output stream
    void PrintStatistics(std::ostream& out) const;

    /// Get string representation
    std::string ToString() const;

private:
    Config config_;

    // Thread safety (reader-writer lock)
    mutable std::shared_mutex mutex_;

    // Main edge storage (using unique_ptr since AssociationEdge is not copyable)
    std::vector<std::unique_ptr<AssociationEdge>> edges_;

    // Outgoing index: source -> edge indices (CSR-like)
    std::unordered_map<PatternID, std::vector<size_t>> outgoing_index_;

    // Incoming index: target -> edge indices (reverse CSR)
    std::unordered_map<PatternID, std::vector<size_t>> incoming_index_;

    // Direct lookup: (source, target) -> edge index (O(1) access)
    std::unordered_map<std::pair<PatternID, PatternID>, size_t, PatternPairHash> edge_lookup_;

    // Type index: type -> edge indices
    std::unordered_map<AssociationType, std::vector<size_t>> type_index_;

    // Deleted edge indices for reuse
    std::vector<size_t> deleted_indices_;

    // Helper methods
    size_t AllocateEdgeIndex();
    void ReleaseEdgeIndex(size_t index);
    void UpdateIndices(size_t edge_index, bool add);
    void RebuildIndices();
};

} // namespace dpan
