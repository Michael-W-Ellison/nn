// File: src/association/categorical_learner.hpp
#pragma once

#include "core/types.hpp"
#include "core/pattern_data.hpp"
#include <vector>
#include <unordered_map>
#include <optional>

namespace dpan {

/// CategoricalLearner: Clusters patterns based on feature similarity
///
/// Learns categorical relationships by grouping patterns with similar
/// features using k-means clustering. Patterns within the same cluster
/// are considered categorically related.
///
/// Thread-safety: Not thread-safe. External synchronization required.
class CategoricalLearner {
public:
    /// Cluster information
    struct ClusterInfo {
        size_t cluster_id;                      // Cluster identifier
        FeatureVector centroid;                 // Cluster centroid
        std::vector<PatternID> members;         // Patterns in cluster
        float average_similarity{0.0f};         // Average intra-cluster similarity
    };

    /// Pattern cluster assignment
    struct PatternCluster {
        size_t cluster_id;                      // Assigned cluster
        float distance_to_centroid{0.0f};       // Distance from centroid
        float similarity_to_centroid{0.0f};     // Cosine similarity to centroid
    };

    /// Configuration for categorical learning
    struct Config {
        Config() = default;
        /// Number of clusters for k-means
        size_t num_clusters{5};
        /// Maximum iterations for k-means convergence
        size_t max_iterations{100};
        /// Convergence threshold (centroid change magnitude)
        float convergence_threshold{0.001f};
        /// Minimum similarity to consider patterns categorically related
        float min_categorical_similarity{0.7f};
        /// Whether to auto-recompute clusters when patterns are added
        bool auto_recompute{false};
    };

    // ========================================================================
    // Construction
    // ========================================================================

    CategoricalLearner();
    explicit CategoricalLearner(const Config& config);
    ~CategoricalLearner() = default;

    // ========================================================================
    // Pattern Management
    // ========================================================================

    /// Add pattern with its feature vector
    /// @param pattern Pattern ID
    /// @param features Feature vector for the pattern
    void AddPattern(PatternID pattern, const FeatureVector& features);

    /// Remove pattern from learner
    /// @param pattern Pattern to remove
    void RemovePattern(PatternID pattern);

    /// Check if pattern is tracked
    /// @param pattern Pattern to check
    /// @return True if pattern is tracked
    bool HasPattern(PatternID pattern) const;

    /// Get feature vector for a pattern
    /// @param pattern Pattern ID
    /// @return Optional feature vector (nullopt if not tracked)
    std::optional<FeatureVector> GetFeatures(PatternID pattern) const;

    // ========================================================================
    // Clustering
    // ========================================================================

    /// Compute clusters using k-means algorithm
    /// @param k_clusters Number of clusters (uses config default if 0)
    /// @return True if clustering succeeded
    bool ComputeClusters(size_t k_clusters = 0);

    /// Get number of clusters
    size_t GetNumClusters() const { return centroids_.size(); }

    /// Get cluster information
    /// @param cluster_id Cluster identifier
    /// @return Optional cluster info (nullopt if cluster doesn't exist)
    std::optional<ClusterInfo> GetClusterInfo(size_t cluster_id) const;

    /// Get all clusters
    /// @return Vector of cluster information
    std::vector<ClusterInfo> GetAllClusters() const;

    /// Clear all clusters (keeps patterns)
    void ClearClusters();

    // ========================================================================
    // Categorical Queries
    // ========================================================================

    /// Check if two patterns belong to same category (cluster)
    /// @param p1 First pattern
    /// @param p2 Second pattern
    /// @return True if patterns are in same cluster
    bool AreCategoricallyRelated(PatternID p1, PatternID p2) const;

    /// Get cluster ID for a pattern
    /// @param pattern Pattern to query
    /// @return Optional cluster ID (nullopt if not clustered)
    std::optional<size_t> GetClusterID(PatternID pattern) const;

    /// Get pattern's cluster assignment details
    /// @param pattern Pattern to query
    /// @return Optional cluster assignment (nullopt if not clustered)
    std::optional<PatternCluster> GetPatternCluster(PatternID pattern) const;

    /// Get all patterns in the same cluster
    /// @param pattern Query pattern
    /// @return Vector of patterns in same cluster (excludes query pattern)
    std::vector<PatternID> GetClusterMembers(PatternID pattern) const;

    /// Get patterns categorically similar to query pattern
    /// @param pattern Query pattern
    /// @param min_similarity Minimum similarity threshold
    /// @return Vector of (pattern, similarity) pairs sorted by similarity
    std::vector<std::pair<PatternID, float>> GetCategoriallyimilar(
        PatternID pattern,
        float min_similarity = 0.0f
    ) const;

    // ========================================================================
    // Feature Similarity
    // ========================================================================

    /// Compute feature similarity between two patterns
    /// @param p1 First pattern
    /// @param p2 Second pattern
    /// @return Cosine similarity [0,1], or 0.0 if either pattern not tracked
    float ComputeFeatureSimilarity(PatternID p1, PatternID p2) const;

    // ========================================================================
    // Maintenance
    // ========================================================================

    /// Clear all data (patterns and clusters)
    void Clear();

    // ========================================================================
    // Statistics
    // ========================================================================

    /// Get number of patterns tracked
    size_t GetPatternCount() const { return pattern_features_.size(); }

    /// Get clustering statistics
    struct ClusteringStats {
        size_t num_patterns{0};
        size_t num_clusters{0};
        size_t num_unassigned{0};
        float average_cluster_size{0.0f};
        float average_intra_cluster_similarity{0.0f};
    };

    /// Get clustering statistics
    ClusteringStats GetClusteringStats() const;

    /// Get configuration
    const Config& GetConfig() const { return config_; }
    void SetConfig(const Config& config) { config_ = config; }

private:
    Config config_;

    // Pattern features
    std::unordered_map<PatternID, FeatureVector> pattern_features_;

    // Cluster centroids
    std::vector<FeatureVector> centroids_;

    // Pattern to cluster assignments
    std::unordered_map<PatternID, PatternCluster> pattern_to_cluster_;

    // Helper methods

    /// Initialize centroids using k-means++ algorithm
    void InitializeCentroids(size_t k);

    /// Assign patterns to nearest centroid
    /// @return True if any assignments changed
    bool AssignPatternsToClusters();

    /// Update centroids based on cluster assignments
    void UpdateCentroids();

    /// Compute distance between feature vectors
    float ComputeDistance(const FeatureVector& v1, const FeatureVector& v2) const;

    /// Find nearest centroid for a feature vector
    /// @return Cluster ID of nearest centroid
    size_t FindNearestCentroid(const FeatureVector& features) const;

    /// Check if clustering has converged
    bool HasConverged(const std::vector<FeatureVector>& old_centroids) const;
};

} // namespace dpan
