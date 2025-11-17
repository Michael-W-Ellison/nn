// File: src/association/categorical_learner.cpp
#include "association/categorical_learner.hpp"
#include <algorithm>
#include <random>
#include <limits>
#include <cmath>

namespace dpan {

// ============================================================================
// Construction
// ============================================================================

CategoricalLearner::CategoricalLearner()
    : config_()
{
}

CategoricalLearner::CategoricalLearner(const Config& config)
    : config_(config)
{
}

// ============================================================================
// Pattern Management
// ============================================================================

void CategoricalLearner::AddPattern(PatternID pattern, const FeatureVector& features) {
    pattern_features_[pattern] = features;

    // Auto-recompute if enabled and we have enough patterns
    if (config_.auto_recompute && pattern_features_.size() >= config_.num_clusters) {
        ComputeClusters();
    }
}

void CategoricalLearner::RemovePattern(PatternID pattern) {
    pattern_features_.erase(pattern);
    pattern_to_cluster_.erase(pattern);
}

bool CategoricalLearner::HasPattern(PatternID pattern) const {
    return pattern_features_.find(pattern) != pattern_features_.end();
}

std::optional<FeatureVector> CategoricalLearner::GetFeatures(PatternID pattern) const {
    auto it = pattern_features_.find(pattern);
    if (it == pattern_features_.end()) {
        return std::nullopt;
    }
    return it->second;
}

// ============================================================================
// Clustering
// ============================================================================

bool CategoricalLearner::ComputeClusters(size_t k_clusters) {
    // Use config default if not specified
    if (k_clusters == 0) {
        k_clusters = config_.num_clusters;
    }

    // Need at least k patterns to create k clusters
    if (pattern_features_.size() < k_clusters) {
        return false;
    }

    // Initialize centroids using k-means++
    InitializeCentroids(k_clusters);

    // K-means iteration
    std::vector<FeatureVector> old_centroids;
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        // Save old centroids for convergence check
        old_centroids = centroids_;

        // Assign patterns to nearest centroids
        bool assignments_changed = AssignPatternsToClusters();

        // Update centroids
        UpdateCentroids();

        // Check for convergence
        if (!assignments_changed && HasConverged(old_centroids)) {
            return true;
        }
    }

    // Max iterations reached
    return true;
}

std::optional<CategoricalLearner::ClusterInfo> CategoricalLearner::GetClusterInfo(
    size_t cluster_id
) const {
    if (cluster_id >= centroids_.size()) {
        return std::nullopt;
    }

    ClusterInfo info;
    info.cluster_id = cluster_id;
    info.centroid = centroids_[cluster_id];

    // Collect members and compute average similarity
    float total_similarity = 0.0f;
    for (const auto& [pattern, assignment] : pattern_to_cluster_) {
        if (assignment.cluster_id == cluster_id) {
            info.members.push_back(pattern);
            total_similarity += assignment.similarity_to_centroid;
        }
    }

    if (!info.members.empty()) {
        info.average_similarity = total_similarity / info.members.size();
    }

    return info;
}

std::vector<CategoricalLearner::ClusterInfo> CategoricalLearner::GetAllClusters() const {
    std::vector<ClusterInfo> clusters;
    clusters.reserve(centroids_.size());

    for (size_t i = 0; i < centroids_.size(); ++i) {
        auto info = GetClusterInfo(i);
        if (info) {
            clusters.push_back(*info);
        }
    }

    return clusters;
}

void CategoricalLearner::ClearClusters() {
    centroids_.clear();
    pattern_to_cluster_.clear();
}

// ============================================================================
// Categorical Queries
// ============================================================================

bool CategoricalLearner::AreCategoricallyRelated(PatternID p1, PatternID p2) const {
    auto cluster1 = GetClusterID(p1);
    auto cluster2 = GetClusterID(p2);

    if (!cluster1 || !cluster2) {
        return false;
    }

    return *cluster1 == *cluster2;
}

std::optional<size_t> CategoricalLearner::GetClusterID(PatternID pattern) const {
    auto it = pattern_to_cluster_.find(pattern);
    if (it == pattern_to_cluster_.end()) {
        return std::nullopt;
    }
    return it->second.cluster_id;
}

std::optional<CategoricalLearner::PatternCluster> CategoricalLearner::GetPatternCluster(
    PatternID pattern
) const {
    auto it = pattern_to_cluster_.find(pattern);
    if (it == pattern_to_cluster_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::vector<PatternID> CategoricalLearner::GetClusterMembers(PatternID pattern) const {
    auto cluster_id = GetClusterID(pattern);
    if (!cluster_id) {
        return {};
    }

    std::vector<PatternID> members;
    for (const auto& [other_pattern, assignment] : pattern_to_cluster_) {
        if (assignment.cluster_id == *cluster_id && other_pattern != pattern) {
            members.push_back(other_pattern);
        }
    }

    return members;
}

std::vector<std::pair<PatternID, float>> CategoricalLearner::GetCategoriallyimilar(
    PatternID pattern,
    float min_similarity
) const {
    auto it = pattern_features_.find(pattern);
    if (it == pattern_features_.end()) {
        return {};
    }

    const FeatureVector& query_features = it->second;
    std::vector<std::pair<PatternID, float>> results;

    // Compute similarity with all other patterns
    for (const auto& [other_pattern, other_features] : pattern_features_) {
        if (other_pattern == pattern) {
            continue;
        }

        float similarity = query_features.CosineSimilarity(other_features);
        if (similarity >= min_similarity) {
            results.push_back({other_pattern, similarity});
        }
    }

    // Sort by similarity (descending)
    std::sort(results.begin(), results.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });

    return results;
}

// ============================================================================
// Feature Similarity
// ============================================================================

float CategoricalLearner::ComputeFeatureSimilarity(PatternID p1, PatternID p2) const {
    auto it1 = pattern_features_.find(p1);
    auto it2 = pattern_features_.find(p2);

    if (it1 == pattern_features_.end() || it2 == pattern_features_.end()) {
        return 0.0f;
    }

    return it1->second.CosineSimilarity(it2->second);
}

// ============================================================================
// Maintenance
// ============================================================================

void CategoricalLearner::Clear() {
    pattern_features_.clear();
    centroids_.clear();
    pattern_to_cluster_.clear();
}

// ============================================================================
// Statistics
// ============================================================================

CategoricalLearner::ClusteringStats CategoricalLearner::GetClusteringStats() const {
    ClusteringStats stats;
    stats.num_patterns = pattern_features_.size();
    stats.num_clusters = centroids_.size();
    stats.num_unassigned = pattern_features_.size() - pattern_to_cluster_.size();

    if (stats.num_clusters > 0) {
        // Compute average cluster size
        std::vector<size_t> cluster_sizes(stats.num_clusters, 0);
        float total_similarity = 0.0f;

        for (const auto& [pattern, assignment] : pattern_to_cluster_) {
            cluster_sizes[assignment.cluster_id]++;
            total_similarity += assignment.similarity_to_centroid;
        }

        // Average cluster size
        if (!cluster_sizes.empty()) {
            size_t total_size = 0;
            for (size_t size : cluster_sizes) {
                total_size += size;
            }
            stats.average_cluster_size = static_cast<float>(total_size) / cluster_sizes.size();
        }

        // Average intra-cluster similarity
        if (!pattern_to_cluster_.empty()) {
            stats.average_intra_cluster_similarity = total_similarity / pattern_to_cluster_.size();
        }
    }

    return stats;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

void CategoricalLearner::InitializeCentroids(size_t k) {
    centroids_.clear();
    centroids_.reserve(k);

    if (pattern_features_.empty()) {
        return;
    }

    // Use k-means++ initialization
    std::random_device rd;
    std::mt19937 gen(rd());

    // Collect all patterns
    std::vector<PatternID> patterns;
    patterns.reserve(pattern_features_.size());
    for (const auto& [pattern, _] : pattern_features_) {
        patterns.push_back(pattern);
    }

    // Choose first centroid randomly
    std::uniform_int_distribution<size_t> dist(0, patterns.size() - 1);
    size_t first_idx = dist(gen);
    centroids_.push_back(pattern_features_[patterns[first_idx]]);

    // Choose remaining centroids
    for (size_t i = 1; i < k; ++i) {
        // Compute squared distances to nearest centroid for each pattern
        std::vector<float> distances;
        distances.reserve(patterns.size());

        for (const auto& pattern : patterns) {
            const auto& features = pattern_features_[pattern];

            // Find minimum distance to existing centroids
            float min_dist = std::numeric_limits<float>::max();
            for (const auto& centroid : centroids_) {
                float dist = ComputeDistance(features, centroid);
                min_dist = std::min(min_dist, dist);
            }

            distances.push_back(min_dist * min_dist);  // Squared distance
        }

        // Choose next centroid with probability proportional to squared distance
        std::discrete_distribution<size_t> weighted_dist(distances.begin(), distances.end());
        size_t next_idx = weighted_dist(gen);
        centroids_.push_back(pattern_features_[patterns[next_idx]]);
    }
}

bool CategoricalLearner::AssignPatternsToClusters() {
    bool changed = false;

    for (const auto& [pattern, features] : pattern_features_) {
        // Find nearest centroid
        size_t nearest = FindNearestCentroid(features);

        // Compute distance and similarity to nearest centroid
        float distance = ComputeDistance(features, centroids_[nearest]);
        float similarity = features.CosineSimilarity(centroids_[nearest]);

        // Create assignment
        PatternCluster assignment;
        assignment.cluster_id = nearest;
        assignment.distance_to_centroid = distance;
        assignment.similarity_to_centroid = similarity;

        // Check if assignment changed
        auto it = pattern_to_cluster_.find(pattern);
        if (it == pattern_to_cluster_.end() || it->second.cluster_id != nearest) {
            changed = true;
        }

        pattern_to_cluster_[pattern] = assignment;
    }

    return changed;
}

void CategoricalLearner::UpdateCentroids() {
    // Reset centroids
    std::vector<FeatureVector> new_centroids(centroids_.size());
    std::vector<size_t> cluster_counts(centroids_.size(), 0);

    // Initialize centroids with correct dimensions
    size_t feature_dim = 0;
    if (!pattern_features_.empty()) {
        feature_dim = pattern_features_.begin()->second.Dimension();
    }

    for (auto& centroid : new_centroids) {
        centroid = FeatureVector(feature_dim);
    }

    // Sum features for each cluster
    for (const auto& [pattern, assignment] : pattern_to_cluster_) {
        size_t cluster_id = assignment.cluster_id;
        const auto& features = pattern_features_[pattern];

        for (size_t i = 0; i < feature_dim; ++i) {
            new_centroids[cluster_id][i] += features[i];
        }
        cluster_counts[cluster_id]++;
    }

    // Compute averages
    for (size_t i = 0; i < new_centroids.size(); ++i) {
        if (cluster_counts[i] > 0) {
            for (size_t j = 0; j < feature_dim; ++j) {
                new_centroids[i][j] /= cluster_counts[i];
            }
            centroids_[i] = new_centroids[i];
        }
        // If cluster is empty, keep old centroid
    }
}

float CategoricalLearner::ComputeDistance(
    const FeatureVector& v1,
    const FeatureVector& v2
) const {
    return v1.EuclideanDistance(v2);
}

size_t CategoricalLearner::FindNearestCentroid(const FeatureVector& features) const {
    if (centroids_.empty()) {
        return 0;
    }

    size_t nearest = 0;
    float min_distance = ComputeDistance(features, centroids_[0]);

    for (size_t i = 1; i < centroids_.size(); ++i) {
        float distance = ComputeDistance(features, centroids_[i]);
        if (distance < min_distance) {
            min_distance = distance;
            nearest = i;
        }
    }

    return nearest;
}

bool CategoricalLearner::HasConverged(
    const std::vector<FeatureVector>& old_centroids
) const {
    if (old_centroids.size() != centroids_.size()) {
        return false;
    }

    for (size_t i = 0; i < centroids_.size(); ++i) {
        float distance = ComputeDistance(centroids_[i], old_centroids[i]);
        if (distance > config_.convergence_threshold) {
            return false;
        }
    }

    return true;
}

} // namespace dpan
