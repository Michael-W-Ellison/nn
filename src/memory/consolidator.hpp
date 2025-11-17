// File: src/memory/consolidator.hpp
#pragma once

#include "core/types.hpp"
#include "core/pattern_node.hpp"
#include "storage/pattern_database.hpp"
#include "association/association_matrix.hpp"
#include "similarity/similarity_metric.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>

namespace dpan {

/// MemoryConsolidator: Consolidate patterns and compress association graphs
///
/// Implements memory consolidation strategies:
/// - Pattern merging: Combine similar patterns into single representatives
/// - Hierarchy formation: Group related patterns into clusters with parent nodes
/// - Association compression: Create shortcut edges for frequently traversed paths
///
/// These operations reduce memory usage while preserving system knowledge
class MemoryConsolidator {
public:
    /// Configuration for consolidation behavior
    struct Config {
        // Pattern merging settings
        float merge_similarity_threshold{0.95f};  // High similarity required
        size_t max_merge_batch{100};              // Patterns to merge per operation
        bool enable_pattern_merging{true};        // Enable pattern merging

        // Hierarchy formation settings
        float cluster_similarity_threshold{0.7f};  // Similarity for clustering
        size_t min_cluster_size{3};                // Minimum patterns in cluster
        size_t max_cluster_size{50};               // Maximum patterns in cluster
        bool enable_hierarchy_formation{true};     // Enable clustering

        // Association compression settings
        size_t min_path_traversals{10};            // Min traversals to create shortcut
        float path_compression_threshold{0.6f};    // Min path strength for compression
        size_t max_path_length{3};                 // Max hops to consider
        bool enable_association_compression{true}; // Enable path compression

        // Safety settings
        bool preserve_original_patterns{false};    // Keep originals after merge
        float min_pattern_confidence{0.5f};        // Only merge high-confidence patterns
    };

    /// Result of pattern merging operation
    struct MergeResult {
        std::vector<std::pair<PatternID, PatternID>> merged_pairs;  // (old, new)
        size_t patterns_removed{0};
        size_t associations_transferred{0};
        size_t patterns_preserved{0};
    };

    /// Result of hierarchy formation
    struct HierarchyResult {
        struct Cluster {
            PatternID parent_id;                    // Parent pattern representing cluster
            std::vector<PatternID> members;         // Child patterns in cluster
            float avg_internal_similarity{0.0f};    // Average similarity within cluster
        };

        std::vector<Cluster> clusters;
        size_t total_patterns_clustered{0};
        size_t hierarchies_created{0};
    };

    /// Result of association compression
    struct CompressionResult {
        std::vector<std::tuple<PatternID, PatternID, float>> shortcuts_created;  // (src, tgt, strength)
        std::vector<std::pair<PatternID, PatternID>> edges_weakened;
        size_t total_shortcuts{0};
        size_t graph_edges_before{0};
        size_t graph_edges_after{0};
    };

    /// Combined consolidation result
    struct ConsolidationResult {
        MergeResult merge_result;
        HierarchyResult hierarchy_result;
        CompressionResult compression_result;

        Timestamp timestamp;
        size_t memory_freed_bytes{0};
    };

    // ========================================================================
    // Construction
    // ========================================================================

    /// Construct with default configuration
    MemoryConsolidator() = default;

    /// Construct with custom configuration
    /// @throws std::invalid_argument if config is invalid
    explicit MemoryConsolidator(const Config& config);

    // ========================================================================
    // Main Consolidation Operations
    // ========================================================================

    /// Perform complete consolidation (merge + hierarchy + compression)
    /// @param pattern_db Pattern database to consolidate
    /// @param assoc_matrix Association matrix
    /// @param similarity_metric Similarity metric for pattern comparison
    /// @return Result with details of all consolidation operations
    ConsolidationResult Consolidate(
        PatternDatabase& pattern_db,
        AssociationMatrix& assoc_matrix,
        const SimilarityMetric& similarity_metric
    );

    /// Merge similar patterns into representatives
    /// @param pattern_db Pattern database
    /// @param assoc_matrix Association matrix
    /// @param similarity_metric Similarity metric for finding similar patterns
    /// @return Result with details of merged patterns
    MergeResult MergePatterns(
        PatternDatabase& pattern_db,
        AssociationMatrix& assoc_matrix,
        const SimilarityMetric& similarity_metric
    );

    /// Form hierarchies by clustering related patterns
    /// @param pattern_db Pattern database
    /// @param assoc_matrix Association matrix
    /// @param similarity_metric Similarity metric for clustering
    /// @return Result with details of formed hierarchies
    HierarchyResult FormHierarchies(
        PatternDatabase& pattern_db,
        AssociationMatrix& assoc_matrix,
        const SimilarityMetric& similarity_metric
    );

    /// Compress association graph by creating shortcuts
    /// @param assoc_matrix Association matrix
    /// @param access_stats Access statistics to identify frequent paths
    /// @return Result with details of compression
    CompressionResult CompressAssociations(
        AssociationMatrix& assoc_matrix,
        const std::unordered_map<std::pair<PatternID, PatternID>, size_t, PatternPairHash>& access_stats
    );

    // ========================================================================
    // Helper Operations
    // ========================================================================

    /// Find patterns that are candidates for merging
    /// @param pattern_db Pattern database
    /// @param similarity_metric Similarity metric
    /// @return Vector of (pattern1, pattern2, similarity) tuples
    std::vector<std::tuple<PatternID, PatternID, float>> FindMergeCandidates(
        PatternDatabase& pattern_db,
        const SimilarityMetric& similarity_metric
    );

    /// Merge two specific patterns
    /// @param old_pattern Pattern to be merged (will be removed or marked)
    /// @param new_pattern Pattern to merge into (will receive associations)
    /// @param pattern_db Pattern database
    /// @param assoc_matrix Association matrix
    /// @return true if merge successful
    bool MergeTwoPatterns(
        PatternID old_pattern,
        PatternID new_pattern,
        PatternDatabase& pattern_db,
        AssociationMatrix& assoc_matrix
    );

    /// Find clusters of related patterns
    /// @param patterns List of pattern IDs to cluster
    /// @param pattern_db Pattern database
    /// @param similarity_metric Similarity metric
    /// @return Vector of pattern clusters
    std::vector<std::vector<PatternID>> FindClusters(
        const std::vector<PatternID>& patterns,
        PatternDatabase& pattern_db,
        const SimilarityMetric& similarity_metric
    );

    /// Create parent pattern for a cluster
    /// @param cluster Member patterns in the cluster
    /// @param pattern_db Pattern database
    /// @return ID of created parent pattern
    PatternID CreateClusterParent(
        const std::vector<PatternID>& cluster,
        PatternDatabase& pattern_db
    );

    /// Find frequently traversed paths in graph
    /// @param assoc_matrix Association matrix
    /// @param access_stats Access statistics
    /// @return Vector of (source, intermediate, target, traversal_count) tuples
    std::vector<std::tuple<PatternID, PatternID, PatternID, size_t>> FindFrequentPaths(
        const AssociationMatrix& assoc_matrix,
        const std::unordered_map<std::pair<PatternID, PatternID>, size_t, PatternPairHash>& access_stats
    );

    /// Create shortcut edge for a path
    /// @param source Source pattern
    /// @param target Target pattern
    /// @param strength Strength for the shortcut
    /// @param assoc_matrix Association matrix
    /// @return true if shortcut created
    bool CreateShortcut(
        PatternID source,
        PatternID target,
        float strength,
        AssociationMatrix& assoc_matrix
    );

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Set configuration (validates before applying)
    /// @throws std::invalid_argument if config is invalid
    void SetConfig(const Config& config);

    /// Get current configuration
    const Config& GetConfig() const { return config_; }

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get statistics about consolidation operations
    struct Statistics {
        size_t total_consolidation_operations{0};
        size_t total_patterns_merged{0};
        size_t total_hierarchies_created{0};
        size_t total_shortcuts_created{0};
        size_t total_memory_freed_bytes{0};
        Timestamp last_consolidation;
    };

    /// Get consolidation statistics
    const Statistics& GetStatistics() const { return stats_; }

    /// Reset statistics
    void ResetStatistics();

private:
    Config config_;
    Statistics stats_;

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Validate configuration
    /// @throws std::invalid_argument if invalid
    void ValidateConfig() const;

    /// Transfer associations from old pattern to new pattern
    /// @param old_pattern Source pattern
    /// @param new_pattern Destination pattern
    /// @param assoc_matrix Association matrix
    /// @return Number of associations transferred
    size_t TransferAssociations(
        PatternID old_pattern,
        PatternID new_pattern,
        AssociationMatrix& assoc_matrix
    );

    /// Calculate centroid (average) of patterns in cluster
    /// @param patterns Patterns in cluster
    /// @param pattern_db Pattern database
    /// @return Centroid pattern data
    PatternData CalculateCentroid(
        const std::vector<PatternID>& patterns,
        PatternDatabase& pattern_db
    );

    /// Greedy clustering algorithm
    /// @param patterns Patterns to cluster
    /// @param similarity_matrix Pairwise similarity matrix
    /// @return Clusters of pattern IDs
    std::vector<std::vector<PatternID>> GreedyClustering(
        const std::vector<PatternID>& patterns,
        const std::unordered_map<std::pair<PatternID, PatternID>, float, PatternPairHash>& similarity_matrix
    );
};

} // namespace dpan
