// File: src/similarity/similarity_search.hpp
#pragma once

#include "similarity_metric.hpp"
#include "storage/pattern_database.hpp"
#include "core/pattern_node.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <queue>

namespace dpan {

/// Search result containing pattern ID and similarity score
struct SearchResult {
    PatternID pattern_id;
    float similarity;

    SearchResult(PatternID id, float sim) : pattern_id(id), similarity(sim) {}

    /// Comparison for sorting (higher similarity first)
    bool operator<(const SearchResult& other) const {
        return similarity > other.similarity;  // Reversed for max-heap
    }
};

/// Search configuration
struct SearchConfig {
    /// Maximum number of results to return
    size_t max_results{10};

    /// Minimum similarity threshold (0.0 to 1.0)
    float min_similarity{0.0f};

    /// Whether to include the query pattern in results
    bool include_query{false};

    /// Optional filter function (returns true if pattern should be included)
    std::function<bool(const PatternNode&)> filter;

    /// Default configuration
    static SearchConfig Default() {
        return SearchConfig{};
    }

    /// Top-K configuration
    static SearchConfig TopK(size_t k) {
        SearchConfig config;
        config.max_results = k;
        return config;
    }

    /// Threshold configuration
    static SearchConfig WithThreshold(float threshold, size_t max_results = 100) {
        SearchConfig config;
        config.min_similarity = threshold;
        config.max_results = max_results;
        return config;
    }
};

/// Similarity Search Engine
///
/// Provides efficient similarity search over pattern collections.
/// Supports multiple similarity metrics, filtering, and top-k retrieval.
class SimilaritySearch {
public:
    /// Constructor
    /// @param database Pattern database to search
    /// @param metric Similarity metric to use
    explicit SimilaritySearch(std::shared_ptr<PatternDatabase> database,
                             std::shared_ptr<SimilarityMetric> metric);

    /// Search for similar patterns by PatternData
    /// @param query Query pattern data
    /// @param config Search configuration
    /// @return Sorted search results (highest similarity first)
    std::vector<SearchResult> Search(const PatternData& query,
                                     const SearchConfig& config = SearchConfig::Default()) const;

    /// Search for similar patterns by FeatureVector
    /// @param query Query feature vector
    /// @param config Search configuration
    /// @return Sorted search results (highest similarity first)
    std::vector<SearchResult> SearchByFeatures(const FeatureVector& query,
                                               const SearchConfig& config = SearchConfig::Default()) const;

    /// Search for similar patterns to an existing pattern
    /// @param query_id ID of query pattern
    /// @param config Search configuration
    /// @return Sorted search results (highest similarity first)
    std::vector<SearchResult> SearchById(PatternID query_id,
                                         const SearchConfig& config = SearchConfig::Default()) const;

    /// Batch search for multiple queries
    /// @param queries Vector of query patterns
    /// @param config Search configuration
    /// @return Vector of result vectors (one per query)
    std::vector<std::vector<SearchResult>> SearchBatch(
        const std::vector<PatternData>& queries,
        const SearchConfig& config = SearchConfig::Default()) const;

    /// Get similarity metric
    std::shared_ptr<SimilarityMetric> GetMetric() const { return metric_; }

    /// Set similarity metric
    void SetMetric(std::shared_ptr<SimilarityMetric> metric);

    /// Get pattern database
    std::shared_ptr<PatternDatabase> GetDatabase() const { return database_; }

    /// Statistics
    struct Stats {
        size_t patterns_evaluated{0};
        size_t patterns_filtered{0};
        size_t results_returned{0};
        float min_similarity_found{1.0f};
        float max_similarity_found{0.0f};
        float avg_similarity_found{0.0f};
    };

    /// Get statistics from last search
    const Stats& GetLastSearchStats() const { return last_stats_; }

private:
    std::shared_ptr<PatternDatabase> database_;
    std::shared_ptr<SimilarityMetric> metric_;
    mutable Stats last_stats_;

    /// Core search implementation
    std::vector<SearchResult> SearchImpl(
        const std::function<float(const PatternData&)>& similarity_fn,
        const SearchConfig& config,
        PatternID exclude_id = PatternID(0)) const;

    /// Update statistics
    void UpdateStats(const std::vector<SearchResult>& results) const;
};

/// Approximate Nearest Neighbor Search (simplified version)
///
/// Uses a simple bucketing strategy for faster approximate search.
/// Useful for large-scale pattern collections where exact search is too slow.
class ApproximateSearch {
public:
    /// Constructor
    /// @param database Pattern database to search
    /// @param metric Similarity metric to use
    /// @param num_buckets Number of buckets for hashing
    explicit ApproximateSearch(std::shared_ptr<PatternDatabase> database,
                              std::shared_ptr<SimilarityMetric> metric,
                              size_t num_buckets = 100);

    /// Build index for approximate search
    void BuildIndex();

    /// Approximate search
    /// @param query Query pattern
    /// @param config Search configuration
    /// @return Approximate search results
    std::vector<SearchResult> Search(const PatternData& query,
                                     const SearchConfig& config = SearchConfig::Default()) const;

    /// Check if index is built
    bool IsIndexBuilt() const { return index_built_; }

private:
    std::shared_ptr<PatternDatabase> database_;
    std::shared_ptr<SimilarityMetric> metric_;
    size_t num_buckets_;
    bool index_built_{false};

    /// Bucket index: bucket_id -> pattern_ids
    std::vector<std::vector<PatternID>> buckets_;

    /// Compute bucket ID for a pattern
    size_t ComputeBucket(const FeatureVector& features) const;
};

/// Multi-metric search
///
/// Combines multiple metrics with weights for more sophisticated search.
class MultiMetricSearch {
public:
    /// Constructor
    /// @param database Pattern database to search
    MultiMetricSearch(std::shared_ptr<PatternDatabase> database);

    /// Add a metric with weight
    void AddMetric(std::shared_ptr<SimilarityMetric> metric, float weight);

    /// Clear all metrics
    void Clear();

    /// Search using combined metrics
    std::vector<SearchResult> Search(const PatternData& query,
                                     const SearchConfig& config = SearchConfig::Default()) const;

    /// Get number of metrics
    size_t GetMetricCount() const { return metrics_.size(); }

private:
    std::shared_ptr<PatternDatabase> database_;
    std::vector<std::pair<std::shared_ptr<SimilarityMetric>, float>> metrics_;
    std::vector<float> normalized_weights_;

    void NormalizeWeights();
};

} // namespace dpan
