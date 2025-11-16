// File: src/discovery/pattern_refiner.hpp
#pragma once

#include "core/pattern_node.hpp"
#include "storage/pattern_database.hpp"
#include <memory>
#include <vector>

namespace dpan {

/// PatternRefiner - Refines and maintains patterns through updates, splitting, and merging
///
/// Handles:
/// - Updating patterns with new data
/// - Adjusting confidence based on match results
/// - Splitting overly general patterns
/// - Merging similar patterns to reduce redundancy
class PatternRefiner {
public:
    /// Result from splitting a pattern
    struct SplitResult {
        std::vector<PatternID> new_pattern_ids;
        bool success;
    };

    /// Result from merging patterns
    struct MergeResult {
        PatternID merged_id;
        bool success;
    };

    /// Constructor
    /// @param database Pattern database to work with
    explicit PatternRefiner(std::shared_ptr<PatternDatabase> database);

    /// Update existing pattern with new data
    /// @param id Pattern ID to update
    /// @param new_data New pattern data
    /// @return true if update succeeded, false otherwise
    bool UpdatePattern(PatternID id, const PatternData& new_data);

    /// Adjust confidence based on match results
    /// @param id Pattern ID
    /// @param matched_correctly true if pattern matched correctly, false if it was a mismatch
    void AdjustConfidence(PatternID id, bool matched_correctly);

    /// Split a pattern into multiple sub-patterns
    /// @param id Pattern ID to split
    /// @param num_clusters Number of clusters to create (default: 2)
    /// @return SplitResult containing new pattern IDs
    SplitResult SplitPattern(PatternID id, size_t num_clusters = 2);

    /// Merge multiple patterns into one
    /// @param pattern_ids IDs of patterns to merge
    /// @return MergeResult containing merged pattern ID
    MergeResult MergePatterns(const std::vector<PatternID>& pattern_ids);

    /// Check if pattern needs splitting
    /// @param id Pattern ID to check
    /// @return true if pattern should be split
    bool NeedsSplitting(PatternID id) const;

    /// Check if two patterns should be merged
    /// @param id1 First pattern ID
    /// @param id2 Second pattern ID
    /// @return true if patterns should be merged
    bool ShouldMerge(PatternID id1, PatternID id2) const;

    /// Set variance threshold for splitting
    /// @param threshold Variance threshold [0, 1]
    void SetVarianceThreshold(float threshold);

    /// Set minimum instances required for splitting
    /// @param min_instances Minimum instance count
    void SetMinInstancesForSplit(size_t min_instances);

    /// Set similarity threshold for merging
    /// @param threshold Similarity threshold [0, 1]
    void SetMergeSimilarityThreshold(float threshold);

    /// Get variance threshold
    float GetVarianceThreshold() const { return variance_threshold_; }

    /// Get minimum instances for split
    size_t GetMinInstancesForSplit() const { return min_instances_for_split_; }

    /// Get merge similarity threshold
    float GetMergeSimilarityThreshold() const { return merge_similarity_threshold_; }

    /// Set confidence adjustment rate
    /// @param rate Adjustment rate (0, 1] - smaller means slower adjustment
    void SetConfidenceAdjustmentRate(float rate);

    /// Get confidence adjustment rate
    float GetConfidenceAdjustmentRate() const { return confidence_adjustment_rate_; }

private:
    std::shared_ptr<PatternDatabase> database_;

    // Splitting criteria
    float variance_threshold_{0.5f};
    size_t min_instances_for_split_{10};

    // Merging criteria
    float merge_similarity_threshold_{0.95f};

    // Confidence adjustment
    float confidence_adjustment_rate_{0.1f};  // How much to adjust per update

    /// Cluster pattern instances for splitting
    /// @param instances Pattern data instances
    /// @param num_clusters Number of clusters to create
    /// @return Vector of clusters, each containing pattern data instances
    std::vector<std::vector<PatternData>> ClusterInstances(
        const std::vector<PatternData>& instances,
        size_t num_clusters
    ) const;

    /// Compute variance of pattern instances
    /// @param instances Pattern data instances
    /// @return Variance value
    float ComputeVariance(const std::vector<PatternData>& instances) const;

    /// Compute centroid of pattern instances
    /// @param instances Pattern data instances
    /// @return Centroid pattern data
    PatternData ComputeCentroid(const std::vector<PatternData>& instances) const;

    /// Compute distance between two pattern data instances
    /// @param data1 First pattern data
    /// @param data2 Second pattern data
    /// @return Distance value
    float ComputeDistance(const PatternData& data1, const PatternData& data2) const;

    /// Generate a new unique pattern ID
    /// @return New pattern ID
    PatternID GenerateNewPatternID() const;
};

} // namespace dpan
