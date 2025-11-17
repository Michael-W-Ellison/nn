// File: src/memory/consolidator.cpp
#include "memory/consolidator.hpp"
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <cmath>

namespace dpan {

// ============================================================================
// Construction
// ============================================================================

MemoryConsolidator::MemoryConsolidator(const Config& config)
    : config_(config)
{
    ValidateConfig();
}

void MemoryConsolidator::ValidateConfig() const {
    if (config_.merge_similarity_threshold < 0.0f || config_.merge_similarity_threshold > 1.0f) {
        throw std::invalid_argument(
            "merge_similarity_threshold must be in [0,1]"
        );
    }

    if (config_.cluster_similarity_threshold < 0.0f || config_.cluster_similarity_threshold > 1.0f) {
        throw std::invalid_argument(
            "cluster_similarity_threshold must be in [0,1]"
        );
    }

    if (config_.path_compression_threshold < 0.0f || config_.path_compression_threshold > 1.0f) {
        throw std::invalid_argument(
            "path_compression_threshold must be in [0,1]"
        );
    }

    if (config_.min_cluster_size == 0) {
        throw std::invalid_argument(
            "min_cluster_size must be > 0"
        );
    }

    if (config_.max_cluster_size < config_.min_cluster_size) {
        throw std::invalid_argument(
            "max_cluster_size must be >= min_cluster_size"
        );
    }

    if (config_.max_merge_batch == 0) {
        throw std::invalid_argument(
            "max_merge_batch must be > 0"
        );
    }

    if (config_.max_path_length == 0) {
        throw std::invalid_argument(
            "max_path_length must be > 0"
        );
    }

    if (config_.min_pattern_confidence < 0.0f || config_.min_pattern_confidence > 1.0f) {
        throw std::invalid_argument(
            "min_pattern_confidence must be in [0,1]"
        );
    }
}

// ============================================================================
// Main Consolidation Operations
// ============================================================================

MemoryConsolidator::ConsolidationResult MemoryConsolidator::Consolidate(
    PatternDatabase& pattern_db,
    AssociationMatrix& assoc_matrix,
    const SimilarityMetric& similarity_metric
) {
    ConsolidationResult result;
    result.timestamp = Timestamp::Now();

    size_t memory_before = pattern_db.GetStats().memory_usage_bytes +
                          assoc_matrix.GetAssociationCount() * sizeof(AssociationEdge);

    // Phase 1: Pattern Merging
    if (config_.enable_pattern_merging) {
        result.merge_result = MergePatterns(pattern_db, assoc_matrix, similarity_metric);
    }

    // Phase 2: Hierarchy Formation
    if (config_.enable_hierarchy_formation) {
        result.hierarchy_result = FormHierarchies(pattern_db, assoc_matrix, similarity_metric);
    }

    // Phase 3: Association Compression
    if (config_.enable_association_compression) {
        // Need access stats - for now use empty map
        std::unordered_map<std::pair<PatternID, PatternID>, size_t, PatternPairHash> empty_stats;
        result.compression_result = CompressAssociations(assoc_matrix, empty_stats);
    }

    size_t memory_after = pattern_db.GetStats().memory_usage_bytes +
                         assoc_matrix.GetAssociationCount() * sizeof(AssociationEdge);

    result.memory_freed_bytes = (memory_before > memory_after) ? (memory_before - memory_after) : 0;

    // Update statistics
    stats_.total_consolidation_operations++;
    stats_.total_patterns_merged += result.merge_result.patterns_removed;
    stats_.total_hierarchies_created += result.hierarchy_result.hierarchies_created;
    stats_.total_shortcuts_created += result.compression_result.total_shortcuts;
    stats_.total_memory_freed_bytes += result.memory_freed_bytes;
    stats_.last_consolidation = result.timestamp;

    return result;
}

MemoryConsolidator::MergeResult MemoryConsolidator::MergePatterns(
    PatternDatabase& pattern_db,
    AssociationMatrix& assoc_matrix,
    const SimilarityMetric& similarity_metric
) {
    MergeResult result;

    // Find merge candidates
    auto candidates = FindMergeCandidates(pattern_db, similarity_metric);

    // Limit batch size
    if (candidates.size() > config_.max_merge_batch) {
        candidates.resize(config_.max_merge_batch);
    }

    // Process each merge candidate
    for (const auto& [pattern1, pattern2, similarity] : candidates) {
        // Choose which pattern to keep (prefer higher confidence)
        auto opt_p1 = pattern_db.Retrieve(pattern1);
        auto opt_p2 = pattern_db.Retrieve(pattern2);

        if (!opt_p1 || !opt_p2) {
            continue;  // One was already removed
        }

        // Keep the pattern with higher confidence
        PatternID to_keep = (opt_p1->GetConfidenceScore() >= opt_p2->GetConfidenceScore()) ? pattern1 : pattern2;
        PatternID to_remove = (to_keep == pattern1) ? pattern2 : pattern1;

        // Perform merge
        if (MergeTwoPatterns(to_remove, to_keep, pattern_db, assoc_matrix)) {
            result.merged_pairs.push_back({to_remove, to_keep});
            result.patterns_removed++;

            // Count associations transferred
            result.associations_transferred += assoc_matrix.GetDegree(to_keep, true) +
                                              assoc_matrix.GetDegree(to_keep, false);

            // Optionally preserve original
            if (config_.preserve_original_patterns) {
                result.patterns_preserved++;
            }
        }
    }

    return result;
}

MemoryConsolidator::HierarchyResult MemoryConsolidator::FormHierarchies(
    PatternDatabase& pattern_db,
    AssociationMatrix& assoc_matrix,
    const SimilarityMetric& similarity_metric
) {
    HierarchyResult result;

    // Get all patterns
    QueryOptions opts;
    opts.max_results = 10000;  // Reasonable limit
    auto all_pattern_ids = pattern_db.FindAll(opts);

    // Find clusters
    auto clusters = FindClusters(all_pattern_ids, pattern_db, similarity_metric);

    // Create parent for each cluster
    for (const auto& cluster : clusters) {
        if (cluster.size() < config_.min_cluster_size) {
            continue;  // Too small
        }

        // Create parent pattern
        PatternID parent_id = CreateClusterParent(cluster, pattern_db);

        // Calculate average internal similarity
        float avg_similarity = 0.0f;
        size_t pair_count = 0;

        for (size_t i = 0; i < cluster.size(); ++i) {
            for (size_t j = i + 1; j < cluster.size(); ++j) {
                auto opt_p1 = pattern_db.Retrieve(cluster[i]);
                auto opt_p2 = pattern_db.Retrieve(cluster[j]);

                if (opt_p1 && opt_p2) {
                    float sim = similarity_metric.Compute(
                        opt_p1->GetData(), opt_p2->GetData()
                    );
                    avg_similarity += sim;
                    pair_count++;
                }
            }
        }

        if (pair_count > 0) {
            avg_similarity /= pair_count;
        }

        // Add cluster to result
        HierarchyResult::Cluster cluster_info;
        cluster_info.parent_id = parent_id;
        cluster_info.members = cluster;
        cluster_info.avg_internal_similarity = avg_similarity;

        result.clusters.push_back(cluster_info);
        result.total_patterns_clustered += cluster.size();
        result.hierarchies_created++;

        // Create hierarchical associations (parent -> children)
        for (PatternID child : cluster) {
            AssociationEdge edge(parent_id, child, AssociationType::COMPOSITIONAL, 0.9f);
            assoc_matrix.AddAssociation(edge);
        }
    }

    return result;
}

MemoryConsolidator::CompressionResult MemoryConsolidator::CompressAssociations(
    AssociationMatrix& assoc_matrix,
    const std::unordered_map<std::pair<PatternID, PatternID>, size_t, PatternPairHash>& access_stats
) {
    CompressionResult result;

    result.graph_edges_before = assoc_matrix.GetAssociationCount();

    // Find frequently traversed paths
    auto frequent_paths = FindFrequentPaths(assoc_matrix, access_stats);

    // Create shortcuts for frequent paths
    for (const auto& [source, intermediate, target, traversals] : frequent_paths) {
        if (traversals < config_.min_path_traversals) {
            continue;  // Not frequent enough
        }

        // Calculate shortcut strength based on path
        const AssociationEdge* edge1 = assoc_matrix.GetAssociation(source, intermediate);
        const AssociationEdge* edge2 = assoc_matrix.GetAssociation(intermediate, target);

        if (!edge1 || !edge2) {
            continue;  // Path no longer exists
        }

        // Shortcut strength = geometric mean of edge strengths
        float shortcut_strength = std::sqrt(edge1->GetStrength() * edge2->GetStrength());

        // Only create if strong enough
        if (shortcut_strength < config_.path_compression_threshold) {
            continue;
        }

        // Check if shortcut already exists
        if (assoc_matrix.HasAssociation(source, target)) {
            // Strengthen existing edge
            assoc_matrix.StrengthenAssociation(source, target, 0.1f);
        } else {
            // Create new shortcut
            if (CreateShortcut(source, target, shortcut_strength, assoc_matrix)) {
                result.shortcuts_created.push_back({source, target, shortcut_strength});
                result.total_shortcuts++;
            }
        }

        // Optionally weaken intermediate edges
        if (result.total_shortcuts > 0) {
            assoc_matrix.WeakenAssociation(source, intermediate, 0.05f);
            assoc_matrix.WeakenAssociation(intermediate, target, 0.05f);
            result.edges_weakened.push_back({source, intermediate});
            result.edges_weakened.push_back({intermediate, target});
        }
    }

    result.graph_edges_after = assoc_matrix.GetAssociationCount();

    return result;
}

// ============================================================================
// Helper Operations
// ============================================================================

std::vector<std::tuple<PatternID, PatternID, float>> MemoryConsolidator::FindMergeCandidates(
    PatternDatabase& pattern_db,
    const SimilarityMetric& similarity_metric
) {
    std::vector<std::tuple<PatternID, PatternID, float>> candidates;

    // Get all patterns
    QueryOptions opts;
    opts.max_results = 1000;  // Reasonable limit for efficiency
    auto all_patterns = pattern_db.FindAll(opts);

    // Compare pairs to find similar patterns
    for (size_t i = 0; i < all_patterns.size(); ++i) {
        for (size_t j = i + 1; j < all_patterns.size(); ++j) {
            auto opt_p1 = pattern_db.Retrieve(all_patterns[i]);
            auto opt_p2 = pattern_db.Retrieve(all_patterns[j]);

            if (!opt_p1 || !opt_p2) {
                continue;
            }

            // Check confidence threshold
            if (opt_p1->GetConfidenceScore() < config_.min_pattern_confidence ||
                opt_p2->GetConfidenceScore() < config_.min_pattern_confidence) {
                continue;
            }

            // Calculate similarity
            float similarity = similarity_metric.Compute(
                opt_p1->GetData(), opt_p2->GetData()
            );

            // Check merge threshold
            if (similarity >= config_.merge_similarity_threshold) {
                candidates.push_back({all_patterns[i], all_patterns[j], similarity});
            }
        }
    }

    // Sort by similarity (highest first)
    std::sort(candidates.begin(), candidates.end(),
        [](const auto& a, const auto& b) {
            return std::get<2>(a) > std::get<2>(b);
        });

    return candidates;
}

bool MemoryConsolidator::MergeTwoPatterns(
    PatternID old_pattern,
    PatternID new_pattern,
    PatternDatabase& pattern_db,
    AssociationMatrix& assoc_matrix
) {
    // Transfer all associations from old to new
    size_t transferred = TransferAssociations(old_pattern, new_pattern, assoc_matrix);

    // Remove old pattern (or mark as merged)
    if (!config_.preserve_original_patterns) {
        pattern_db.Delete(old_pattern);
    }

    return transferred > 0 || true;  // Consider successful even if no associations
}

std::vector<std::vector<PatternID>> MemoryConsolidator::FindClusters(
    const std::vector<PatternID>& patterns,
    PatternDatabase& pattern_db,
    const SimilarityMetric& similarity_metric
) {
    if (patterns.size() < config_.min_cluster_size) {
        return {};  // Not enough patterns to cluster
    }

    // Build similarity matrix
    std::unordered_map<std::pair<PatternID, PatternID>, float, PatternPairHash> similarity_matrix;

    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = i + 1; j < patterns.size(); ++j) {
            auto opt_p1 = pattern_db.Retrieve(patterns[i]);
            auto opt_p2 = pattern_db.Retrieve(patterns[j]);

            if (opt_p1 && opt_p2) {
                float sim = similarity_metric.Compute(
                    opt_p1->GetData(), opt_p2->GetData()
                );
                similarity_matrix[{patterns[i], patterns[j]}] = sim;
                similarity_matrix[{patterns[j], patterns[i]}] = sim;
            }
        }
    }

    // Perform greedy clustering
    return GreedyClustering(patterns, similarity_matrix);
}

PatternID MemoryConsolidator::CreateClusterParent(
    const std::vector<PatternID>& cluster,
    PatternDatabase& pattern_db
) {
    // Calculate centroid of cluster members
    PatternData centroid = CalculateCentroid(cluster, pattern_db);

    // Create new parent pattern
    PatternID parent_id = PatternID::Generate();
    PatternNode parent(parent_id, centroid, PatternType::COMPOSITE);

    // Store parent
    pattern_db.Store(parent);

    return parent_id;
}

std::vector<std::tuple<PatternID, PatternID, PatternID, size_t>> MemoryConsolidator::FindFrequentPaths(
    const AssociationMatrix& assoc_matrix,
    const std::unordered_map<std::pair<PatternID, PatternID>, size_t, PatternPairHash>& access_stats
) {
    std::vector<std::tuple<PatternID, PatternID, PatternID, size_t>> frequent_paths;

    // This is a simplified implementation
    // In a full implementation, we would:
    // 1. Iterate through all two-hop paths in the graph
    // 2. Count how many times each path is traversed (from access_stats)
    // 3. Return paths that exceed min_path_traversals threshold

    // For now, return empty vector as we don't have access_stats populated
    return frequent_paths;
}

bool MemoryConsolidator::CreateShortcut(
    PatternID source,
    PatternID target,
    float strength,
    AssociationMatrix& assoc_matrix
) {
    AssociationEdge shortcut(source, target, AssociationType::CAUSAL, strength);
    return assoc_matrix.AddAssociation(shortcut);
}

// ============================================================================
// Configuration
// ============================================================================

void MemoryConsolidator::SetConfig(const Config& config) {
    config_ = config;
    ValidateConfig();
}

void MemoryConsolidator::ResetStatistics() {
    stats_ = Statistics{};
}

// ============================================================================
// Helper Methods
// ============================================================================

size_t MemoryConsolidator::TransferAssociations(
    PatternID old_pattern,
    PatternID new_pattern,
    AssociationMatrix& assoc_matrix
) {
    size_t transferred = 0;

    // Transfer outgoing associations
    auto outgoing = assoc_matrix.GetOutgoingAssociations(old_pattern);
    for (const AssociationEdge* edge : outgoing) {
        if (!edge) continue;

        PatternID target = edge->GetTarget();

        // Skip self-loop to old pattern
        if (target == old_pattern) continue;

        // Create new edge from new_pattern to target
        AssociationEdge new_edge(new_pattern, target, edge->GetType(), edge->GetStrength());

        if (assoc_matrix.AddAssociation(new_edge) ||
            assoc_matrix.StrengthenAssociation(new_pattern, target, edge->GetStrength() * 0.5f)) {
            transferred++;
        }
    }

    // Transfer incoming associations
    auto incoming = assoc_matrix.GetIncomingAssociations(old_pattern);
    for (const AssociationEdge* edge : incoming) {
        if (!edge) continue;

        PatternID source = edge->GetSource();

        // Skip self-loop to old pattern
        if (source == old_pattern) continue;

        // Create new edge from source to new_pattern
        AssociationEdge new_edge(source, new_pattern, edge->GetType(), edge->GetStrength());

        if (assoc_matrix.AddAssociation(new_edge) ||
            assoc_matrix.StrengthenAssociation(source, new_pattern, edge->GetStrength() * 0.5f)) {
            transferred++;
        }
    }

    // Remove old associations
    assoc_matrix.RemoveAssociation(old_pattern, new_pattern);

    return transferred;
}

PatternData MemoryConsolidator::CalculateCentroid(
    const std::vector<PatternID>& patterns,
    PatternDatabase& pattern_db
) {
    if (patterns.empty()) {
        return PatternData();  // Empty centroid
    }

    // Get first pattern to initialize
    auto opt_first = pattern_db.Retrieve(patterns[0]);
    if (!opt_first) {
        return PatternData();
    }

    PatternData centroid = opt_first->GetData();

    // Average the features
    const FeatureVector& first_features = opt_first->GetData().GetFeatures();
    std::vector<float> sum_features(first_features.Dimension(), 0.0f);

    // Sum all features
    for (PatternID pid : patterns) {
        auto opt_pattern = pattern_db.Retrieve(pid);
        if (!opt_pattern) continue;

        const FeatureVector& features = opt_pattern->GetData().GetFeatures();
        for (size_t i = 0; i < features.Dimension() && i < sum_features.size(); ++i) {
            sum_features[i] += features[i];
        }
    }

    // Calculate average
    for (float& val : sum_features) {
        val /= patterns.size();
    }

    // Create centroid from averaged features
    FeatureVector centroid_features(sum_features);
    return PatternData::FromFeatures(centroid_features, centroid.GetModality());
}

std::vector<std::vector<PatternID>> MemoryConsolidator::GreedyClustering(
    const std::vector<PatternID>& patterns,
    const std::unordered_map<std::pair<PatternID, PatternID>, float, PatternPairHash>& similarity_matrix
) {
    std::vector<std::vector<PatternID>> clusters;
    std::unordered_set<PatternID> assigned;

    for (PatternID seed : patterns) {
        if (assigned.count(seed) > 0) {
            continue;  // Already in a cluster
        }

        // Start new cluster with seed
        std::vector<PatternID> cluster = {seed};
        assigned.insert(seed);

        // Greedily add similar patterns
        for (PatternID candidate : patterns) {
            if (assigned.count(candidate) > 0) {
                continue;  // Already assigned
            }

            if (cluster.size() >= config_.max_cluster_size) {
                break;  // Cluster full
            }

            // Check similarity to all cluster members
            float avg_similarity = 0.0f;
            size_t count = 0;

            for (PatternID member : cluster) {
                auto it = similarity_matrix.find({candidate, member});
                if (it != similarity_matrix.end()) {
                    avg_similarity += it->second;
                    count++;
                }
            }

            if (count > 0) {
                avg_similarity /= count;
            }

            // Add if similar enough
            if (avg_similarity >= config_.cluster_similarity_threshold) {
                cluster.push_back(candidate);
                assigned.insert(candidate);
            }
        }

        // Only keep cluster if it meets minimum size
        if (cluster.size() >= config_.min_cluster_size) {
            clusters.push_back(cluster);
        }
    }

    return clusters;
}

} // namespace dpan
