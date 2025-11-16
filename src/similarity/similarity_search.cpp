// File: src/similarity/similarity_search.cpp
#include "similarity_search.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace dpan {

// ============================================================================
// SimilaritySearch Implementation
// ============================================================================

SimilaritySearch::SimilaritySearch(std::shared_ptr<PatternDatabase> database,
                                   std::shared_ptr<SimilarityMetric> metric)
    : database_(database), metric_(metric) {
    if (!database_) {
        throw std::invalid_argument("Database cannot be null");
    }
    if (!metric_) {
        throw std::invalid_argument("Metric cannot be null");
    }
}

std::vector<SearchResult> SimilaritySearch::Search(const PatternData& query,
                                                   const SearchConfig& config) const {
    auto similarity_fn = [this, &query](const PatternData& candidate) {
        return metric_->Compute(query, candidate);
    };

    return SearchImpl(similarity_fn, config);
}

std::vector<SearchResult> SimilaritySearch::SearchByFeatures(const FeatureVector& query,
                                                             const SearchConfig& config) const {
    auto similarity_fn = [this, &query](const PatternData& candidate) {
        return metric_->ComputeFromFeatures(query, candidate.GetFeatures());
    };

    return SearchImpl(similarity_fn, config);
}

std::vector<SearchResult> SimilaritySearch::SearchById(PatternID query_id,
                                                       const SearchConfig& config) const {
    // Get the query pattern
    auto query_node_opt = database_->Retrieve(query_id);
    if (!query_node_opt) {
        return {};
    }

    const PatternData& query_data = query_node_opt->GetData();
    auto similarity_fn = [this, &query_data](const PatternData& candidate) {
        return metric_->Compute(query_data, candidate);
    };

    return SearchImpl(similarity_fn, config, query_id);
}

std::vector<std::vector<SearchResult>> SimilaritySearch::SearchBatch(
    const std::vector<PatternData>& queries,
    const SearchConfig& config) const {

    std::vector<std::vector<SearchResult>> results;
    results.reserve(queries.size());

    for (const auto& query : queries) {
        results.push_back(Search(query, config));
    }

    return results;
}

void SimilaritySearch::SetMetric(std::shared_ptr<SimilarityMetric> metric) {
    if (!metric) {
        throw std::invalid_argument("Metric cannot be null");
    }
    metric_ = metric;
}

std::vector<SearchResult> SimilaritySearch::SearchImpl(
    const std::function<float(const PatternData&)>& similarity_fn,
    const SearchConfig& config,
    PatternID exclude_id) const {

    // Reset statistics
    last_stats_ = Stats{};

    // Priority queue for top-k results (min-heap)
    std::priority_queue<SearchResult> top_k;

    // Get all pattern IDs
    auto all_ids = database_->FindAll();
    last_stats_.patterns_evaluated = all_ids.size();

    for (const auto& pattern_id : all_ids) {
        // Skip excluded pattern (e.g., query pattern)
        if (!config.include_query && pattern_id == exclude_id) {
            last_stats_.patterns_filtered++;
            continue;
        }

        // Get pattern node
        auto node_opt = database_->Retrieve(pattern_id);
        if (!node_opt) {
            continue;
        }

        // Apply custom filter if provided
        if (config.filter && !config.filter(*node_opt)) {
            last_stats_.patterns_filtered++;
            continue;
        }

        // Compute similarity
        float similarity = similarity_fn(node_opt->GetData());

        // Check threshold
        if (similarity < config.min_similarity) {
            last_stats_.patterns_filtered++;
            continue;
        }

        // Add to top-k
        top_k.emplace(pattern_id, similarity);

        // Keep only top-k results
        if (top_k.size() > config.max_results) {
            top_k.pop();
        }
    }

    // Extract results from priority queue
    std::vector<SearchResult> results;
    results.reserve(top_k.size());

    while (!top_k.empty()) {
        results.push_back(top_k.top());
        top_k.pop();
    }

    // Reverse to get highest similarity first
    std::reverse(results.begin(), results.end());

    // Update statistics
    UpdateStats(results);

    return results;
}

void SimilaritySearch::UpdateStats(const std::vector<SearchResult>& results) const {
    last_stats_.results_returned = results.size();

    if (!results.empty()) {
        last_stats_.min_similarity_found = results.back().similarity;
        last_stats_.max_similarity_found = results.front().similarity;

        float sum = 0.0f;
        for (const auto& result : results) {
            sum += result.similarity;
        }
        last_stats_.avg_similarity_found = sum / results.size();
    }
}

// ============================================================================
// ApproximateSearch Implementation
// ============================================================================

ApproximateSearch::ApproximateSearch(std::shared_ptr<PatternDatabase> database,
                                     std::shared_ptr<SimilarityMetric> metric,
                                     size_t num_buckets)
    : database_(database), metric_(metric), num_buckets_(num_buckets) {
    if (!database_) {
        throw std::invalid_argument("Database cannot be null");
    }
    if (!metric_) {
        throw std::invalid_argument("Metric cannot be null");
    }
    if (num_buckets_ == 0) {
        num_buckets_ = 1;
    }

    buckets_.resize(num_buckets_);
}

void ApproximateSearch::BuildIndex() {
    // Clear existing buckets
    for (auto& bucket : buckets_) {
        bucket.clear();
    }

    // Get all patterns and assign to buckets
    auto all_ids = database_->FindAll();

    for (const auto& pattern_id : all_ids) {
        auto node_opt = database_->Retrieve(pattern_id);
        if (!node_opt) {
            continue;
        }

        size_t bucket_id = ComputeBucket(node_opt->GetData().GetFeatures());
        buckets_[bucket_id].push_back(pattern_id);
    }

    index_built_ = true;
}

std::vector<SearchResult> ApproximateSearch::Search(const PatternData& query,
                                                    const SearchConfig& config) const {
    if (!index_built_) {
        throw std::runtime_error("Index not built. Call BuildIndex() first.");
    }

    // Compute query bucket
    size_t query_bucket = ComputeBucket(query.GetFeatures());

    // Search in the query bucket and neighboring buckets
    std::priority_queue<SearchResult> top_k;

    // Search in query bucket
    for (const auto& pattern_id : buckets_[query_bucket]) {
        auto node_opt = database_->Retrieve(pattern_id);
        if (!node_opt) {
            continue;
        }

        if (config.filter && !config.filter(*node_opt)) {
            continue;
        }

        float similarity = metric_->Compute(query, node_opt->GetData());

        if (similarity >= config.min_similarity) {
            top_k.emplace(pattern_id, similarity);
            if (top_k.size() > config.max_results) {
                top_k.pop();
            }
        }
    }

    // Also search neighboring buckets for better recall
    std::vector<size_t> neighbor_buckets;
    if (query_bucket > 0) {
        neighbor_buckets.push_back(query_bucket - 1);
    }
    if (query_bucket < num_buckets_ - 1) {
        neighbor_buckets.push_back(query_bucket + 1);
    }

    for (size_t bucket_id : neighbor_buckets) {
        for (const auto& pattern_id : buckets_[bucket_id]) {
            auto node_opt = database_->Retrieve(pattern_id);
            if (!node_opt) {
                continue;
            }

            if (config.filter && !config.filter(*node_opt)) {
                continue;
            }

            float similarity = metric_->Compute(query, node_opt->GetData());

            if (similarity >= config.min_similarity) {
                top_k.emplace(pattern_id, similarity);
                if (top_k.size() > config.max_results) {
                    top_k.pop();
                }
            }
        }
    }

    // Extract and sort results
    std::vector<SearchResult> results;
    results.reserve(top_k.size());

    while (!top_k.empty()) {
        results.push_back(top_k.top());
        top_k.pop();
    }

    std::reverse(results.begin(), results.end());

    return results;
}

size_t ApproximateSearch::ComputeBucket(const FeatureVector& features) const {
    if (features.Dimension() == 0) {
        return 0;
    }

    // Simple hash function: sum of feature values mod num_buckets
    float sum = 0.0f;
    for (size_t i = 0; i < features.Dimension(); ++i) {
        sum += features[i];
    }

    // Use absolute value and modulo to get bucket ID
    size_t bucket_id = static_cast<size_t>(std::abs(sum)) % num_buckets_;
    return bucket_id;
}

// ============================================================================
// MultiMetricSearch Implementation
// ============================================================================

MultiMetricSearch::MultiMetricSearch(std::shared_ptr<PatternDatabase> database)
    : database_(database) {
    if (!database_) {
        throw std::invalid_argument("Database cannot be null");
    }
}

void MultiMetricSearch::AddMetric(std::shared_ptr<SimilarityMetric> metric, float weight) {
    if (!metric) {
        return;
    }

    metrics_.emplace_back(metric, weight);
    NormalizeWeights();
}

void MultiMetricSearch::Clear() {
    metrics_.clear();
    normalized_weights_.clear();
}

std::vector<SearchResult> MultiMetricSearch::Search(const PatternData& query,
                                                    const SearchConfig& config) const {
    if (metrics_.empty()) {
        return {};
    }

    // Compute combined similarity for all patterns
    std::priority_queue<SearchResult> top_k;

    auto all_ids = database_->FindAll();

    for (const auto& pattern_id : all_ids) {
        auto node_opt = database_->Retrieve(pattern_id);
        if (!node_opt) {
            continue;
        }

        if (config.filter && !config.filter(*node_opt)) {
            continue;
        }

        // Compute weighted combination of similarities
        float combined_similarity = 0.0f;
        for (size_t i = 0; i < metrics_.size(); ++i) {
            float sim = metrics_[i].first->Compute(query, node_opt->GetData());
            combined_similarity += normalized_weights_[i] * sim;
        }

        if (combined_similarity >= config.min_similarity) {
            top_k.emplace(pattern_id, combined_similarity);
            if (top_k.size() > config.max_results) {
                top_k.pop();
            }
        }
    }

    // Extract and sort results
    std::vector<SearchResult> results;
    results.reserve(top_k.size());

    while (!top_k.empty()) {
        results.push_back(top_k.top());
        top_k.pop();
    }

    std::reverse(results.begin(), results.end());

    return results;
}

void MultiMetricSearch::NormalizeWeights() {
    normalized_weights_.clear();

    float total_weight = 0.0f;
    for (const auto& [metric, weight] : metrics_) {
        total_weight += weight;
    }

    if (total_weight > 1e-10f) {
        for (const auto& [metric, weight] : metrics_) {
            normalized_weights_.push_back(weight / total_weight);
        }
    } else {
        // Uniform distribution
        float uniform_weight = 1.0f / metrics_.size();
        for (size_t i = 0; i < metrics_.size(); ++i) {
            normalized_weights_.push_back(uniform_weight);
        }
    }
}

} // namespace dpan
