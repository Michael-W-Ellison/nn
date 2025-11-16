// File: src/discovery/pattern_refiner.cpp
#include "pattern_refiner.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace dpan {

// ============================================================================
// PatternRefiner Implementation
// ============================================================================

PatternRefiner::PatternRefiner(std::shared_ptr<PatternDatabase> database)
    : database_(database) {
    if (!database_) {
        throw std::invalid_argument("PatternRefiner requires non-null database");
    }
}

bool PatternRefiner::UpdatePattern(PatternID id, const PatternData& new_data) {
    // Retrieve existing pattern
    auto node_opt = database_->Retrieve(id);
    if (!node_opt.has_value()) {
        return false;
    }

    const auto& node = node_opt.value();

    // Create new node with updated data
    PatternNode updated_node(id, new_data, node.GetType());

    // Preserve existing statistics
    updated_node.SetActivationThreshold(node.GetActivationThreshold());
    updated_node.SetConfidenceScore(node.GetConfidenceScore());
    updated_node.SetBaseActivation(node.GetBaseActivation());

    // Preserve sub-patterns for composite/meta patterns
    for (const auto& sub_id : node.GetSubPatterns()) {
        updated_node.AddSubPattern(sub_id);
    }

    // Update the pattern in database
    return database_->Update(updated_node);
}

void PatternRefiner::AdjustConfidence(PatternID id, bool matched_correctly) {
    // Retrieve pattern
    auto node_opt = database_->Retrieve(id);
    if (!node_opt.has_value()) {
        return;
    }

    // Use move semantics to avoid copy
    auto node = std::move(node_opt.value());
    float current_confidence = node.GetConfidenceScore();

    // Adjust confidence
    float adjustment = matched_correctly ? confidence_adjustment_rate_ : -confidence_adjustment_rate_;
    float new_confidence = std::clamp(current_confidence + adjustment, 0.0f, 1.0f);

    node.SetConfidenceScore(new_confidence);

    // Update the pattern in database
    database_->Update(std::move(node));
}

PatternRefiner::SplitResult PatternRefiner::SplitPattern(
    PatternID id,
    size_t num_clusters) {

    SplitResult result{};
    result.success = false;

    if (num_clusters < 2) {
        return result;
    }

    // Retrieve pattern
    auto node_opt = database_->Retrieve(id);
    if (!node_opt.has_value()) {
        return result;
    }

    const auto& node = node_opt.value();

    // For demonstration, we'll create a simplified splitting approach
    // In a real implementation, you would collect actual pattern instances/activations
    // Here we'll create synthetic instances based on the pattern's feature vector

    std::vector<PatternData> instances;

    // Get the pattern's data
    const auto& pattern_data = node.GetData();
    const auto& features = pattern_data.GetFeatures();

    if (features.Dimension() == 0) {
        return result;
    }

    // Create synthetic variations for splitting
    // In practice, these would be actual recorded activations
    // For now, create num_clusters variations by perturbing the original features
    for (size_t i = 0; i < num_clusters; ++i) {
        std::vector<float> perturbed_values;
        float perturbation = (static_cast<float>(i) / num_clusters) - 0.5f; // Range: -0.5 to +0.5

        for (size_t dim = 0; dim < features.Dimension(); ++dim) {
            perturbed_values.push_back(features[dim] + perturbation);
        }

        FeatureVector perturbed_features(perturbed_values);
        instances.push_back(PatternData::FromFeatures(perturbed_features, pattern_data.GetModality()));
    }

    // Cluster the synthetic instances
    auto clusters = ClusterInstances(instances, num_clusters);

    if (clusters.empty()) {
        return result;
    }

    // Create new patterns for each cluster
    for (const auto& cluster : clusters) {
        if (!cluster.empty()) {
            // Compute centroid for this cluster
            PatternData centroid = ComputeCentroid(cluster);

            // Create new pattern
            PatternID new_id = GenerateNewPatternID();
            PatternNode new_node(new_id, centroid, node.GetType());

            // Set similar parameters to original
            new_node.SetActivationThreshold(node.GetActivationThreshold());
            new_node.SetConfidenceScore(node.GetConfidenceScore() * 0.8f); // Slightly lower confidence for split patterns
            new_node.SetBaseActivation(0.0f);

            // Store new pattern
            if (database_->Store(new_node)) {
                result.new_pattern_ids.push_back(new_id);
            }
        }
    }

    result.success = !result.new_pattern_ids.empty();
    return result;
}

PatternRefiner::MergeResult PatternRefiner::MergePatterns(
    const std::vector<PatternID>& pattern_ids) {

    MergeResult result{};
    result.success = false;

    if (pattern_ids.size() < 2) {
        return result;
    }

    // Retrieve all patterns and verify they exist and have same type
    std::vector<PatternData> data_instances;
    PatternType first_type = PatternType::ATOMIC;
    bool first = true;

    float avg_threshold = 0.0f;
    float avg_confidence = 0.0f;
    float avg_activation = 0.0f;
    std::vector<PatternID> all_sub_patterns;

    for (const auto& id : pattern_ids) {
        auto node_opt = database_->Retrieve(id);
        if (!node_opt.has_value()) {
            return result;  // If any pattern doesn't exist, fail
        }

        const auto& pattern = node_opt.value();

        if (first) {
            first_type = pattern.GetType();
            first = false;
        } else if (pattern.GetType() != first_type) {
            return result;  // Can't merge patterns of different types
        }

        // Collect data
        data_instances.push_back(pattern.GetData());

        // Accumulate parameters
        avg_threshold += pattern.GetActivationThreshold();
        avg_confidence += pattern.GetConfidenceScore();
        avg_activation += pattern.GetBaseActivation();

        // Collect sub-patterns for composite/meta patterns
        if (first_type == PatternType::COMPOSITE || first_type == PatternType::META) {
            for (const auto& sub_id : pattern.GetSubPatterns()) {
                all_sub_patterns.push_back(sub_id);
            }
        }
    }

    // Compute merged centroid
    PatternData merged_data = ComputeCentroid(data_instances);

    // Create merged pattern
    PatternID merged_id = GenerateNewPatternID();
    PatternNode merged_node(merged_id, merged_data, first_type);

    // Set parameters as average of merged patterns
    size_t count = pattern_ids.size();
    merged_node.SetActivationThreshold(avg_threshold / count);
    merged_node.SetConfidenceScore(avg_confidence / count);
    merged_node.SetBaseActivation(avg_activation / count);

    // If patterns are composite/meta, merge their sub-patterns
    if (first_type == PatternType::COMPOSITE || first_type == PatternType::META) {
        // Remove duplicates
        std::sort(all_sub_patterns.begin(), all_sub_patterns.end(),
                  [](const PatternID& a, const PatternID& b) { return a.value() < b.value(); });
        all_sub_patterns.erase(
            std::unique(all_sub_patterns.begin(), all_sub_patterns.end(),
                       [](const PatternID& a, const PatternID& b) { return a.value() == b.value(); }),
            all_sub_patterns.end()
        );

        for (const auto& sub_id : all_sub_patterns) {
            merged_node.AddSubPattern(sub_id);
        }
    }

    // Store merged pattern
    if (database_->Store(merged_node)) {
        result.merged_id = merged_id;
        result.success = true;
    }

    return result;
}

bool PatternRefiner::NeedsSplitting(PatternID id) const {
    // Retrieve pattern
    auto node_opt = database_->Retrieve(id);
    if (!node_opt.has_value()) {
        return false;
    }

    // For this implementation, we'll use a simple heuristic:
    // A pattern needs splitting if it has low confidence
    // In practice, you would collect activation instances and compute variance

    const auto& node = node_opt.value();
    float confidence = node.GetConfidenceScore();

    // If confidence is low, pattern might be too general
    return confidence < 0.3f;
}

bool PatternRefiner::ShouldMerge(PatternID id1, PatternID id2) const {
    // Retrieve both patterns
    auto node1_opt = database_->Retrieve(id1);
    auto node2_opt = database_->Retrieve(id2);

    if (!node1_opt.has_value() || !node2_opt.has_value()) {
        return false;
    }

    const auto& node1 = node1_opt.value();
    const auto& node2 = node2_opt.value();

    // Can only merge patterns of the same type
    if (node1.GetType() != node2.GetType()) {
        return false;
    }

    // Compute similarity between patterns
    const auto& data1 = node1.GetData();
    const auto& data2 = node2.GetData();

    float distance = ComputeDistance(data1, data2);

    // Convert distance to similarity (inverse relationship)
    // If distance is small, similarity is high
    float similarity = 1.0f / (1.0f + distance);

    return similarity >= merge_similarity_threshold_;
}

void PatternRefiner::SetVarianceThreshold(float threshold) {
    if (threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("variance_threshold must be in range [0.0, 1.0]");
    }
    variance_threshold_ = threshold;
}

void PatternRefiner::SetMinInstancesForSplit(size_t min_instances) {
    min_instances_for_split_ = min_instances;
}

void PatternRefiner::SetMergeSimilarityThreshold(float threshold) {
    if (threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("merge_similarity_threshold must be in range [0.0, 1.0]");
    }
    merge_similarity_threshold_ = threshold;
}

void PatternRefiner::SetConfidenceAdjustmentRate(float rate) {
    if (rate <= 0.0f || rate > 1.0f) {
        throw std::invalid_argument("confidence_adjustment_rate must be in range (0.0, 1.0]");
    }
    confidence_adjustment_rate_ = rate;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

std::vector<std::vector<PatternData>> PatternRefiner::ClusterInstances(
    const std::vector<PatternData>& instances,
    size_t num_clusters) const {

    if (instances.empty() || num_clusters == 0) {
        return {};
    }

    // For simplicity, use a basic k-means-like approach
    // In practice, you'd use a more sophisticated clustering algorithm

    std::vector<std::vector<PatternData>> clusters(num_clusters);

    // If we have fewer instances than clusters, put each in its own cluster
    if (instances.size() <= num_clusters) {
        for (size_t i = 0; i < instances.size(); ++i) {
            clusters[i].push_back(instances[i]);
        }
        return clusters;
    }

    // Initialize centroids (use first k instances)
    std::vector<PatternData> centroids;
    for (size_t i = 0; i < num_clusters && i < instances.size(); ++i) {
        centroids.push_back(instances[i]);
    }

    // Simple k-means iteration (just one iteration for simplicity)
    for (const auto& instance : instances) {
        // Find closest centroid
        size_t closest_cluster = 0;
        float min_distance = std::numeric_limits<float>::max();

        for (size_t i = 0; i < centroids.size(); ++i) {
            float dist = ComputeDistance(instance, centroids[i]);
            if (dist < min_distance) {
                min_distance = dist;
                closest_cluster = i;
            }
        }

        // Assign to closest cluster
        clusters[closest_cluster].push_back(instance);
    }

    return clusters;
}

float PatternRefiner::ComputeVariance(const std::vector<PatternData>& instances) const {
    if (instances.empty()) {
        return 0.0f;
    }

    // Compute centroid
    PatternData centroid = ComputeCentroid(instances);

    // Compute variance as average squared distance from centroid
    float variance = 0.0f;
    for (const auto& instance : instances) {
        float dist = ComputeDistance(instance, centroid);
        variance += dist * dist;
    }

    return variance / instances.size();
}

PatternData PatternRefiner::ComputeCentroid(const std::vector<PatternData>& instances) const {
    if (instances.empty()) {
        throw std::invalid_argument("Cannot compute centroid of empty instances");
    }

    // Get feature dimension from first instance
    const auto& first_features = instances[0].GetFeatures();
    size_t dim = first_features.Dimension();

    if (dim == 0) {
        // Return copy of first instance if no features
        return instances[0];
    }

    // Compute mean for each dimension
    std::vector<float> mean_values(dim, 0.0f);

    for (const auto& instance : instances) {
        const auto& features = instance.GetFeatures();
        if (features.Dimension() != dim) {
            throw std::invalid_argument("All instances must have same feature dimension");
        }

        for (size_t i = 0; i < dim; ++i) {
            mean_values[i] += features[i];
        }
    }

    for (size_t i = 0; i < dim; ++i) {
        mean_values[i] /= instances.size();
    }

    // Create centroid pattern data
    FeatureVector centroid_features(mean_values);
    return PatternData::FromFeatures(centroid_features, instances[0].GetModality());
}

float PatternRefiner::ComputeDistance(const PatternData& data1, const PatternData& data2) const {
    const auto& f1 = data1.GetFeatures();
    const auto& f2 = data2.GetFeatures();

    if (f1.Dimension() != f2.Dimension()) {
        // If dimensions don't match, return large distance
        return std::numeric_limits<float>::max();
    }

    if (f1.Dimension() == 0) {
        return 0.0f;
    }

    // Euclidean distance
    float sum_squared_diff = 0.0f;
    for (size_t i = 0; i < f1.Dimension(); ++i) {
        float diff = f1[i] - f2[i];
        sum_squared_diff += diff * diff;
    }

    return std::sqrt(sum_squared_diff);
}

PatternID PatternRefiner::GenerateNewPatternID() const {
    // Get all existing pattern IDs
    auto all_ids = database_->FindAll();

    if (all_ids.empty()) {
        return PatternID(1);
    }

    // Find maximum ID
    uint64_t max_id = 0;
    for (const auto& id : all_ids) {
        max_id = std::max(max_id, id.value());
    }

    return PatternID(max_id + 1);
}

} // namespace dpan
