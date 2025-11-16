// File: src/similarity/contextual_similarity.cpp
#include "contextual_similarity.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace dpan {

// ============================================================================
// ContextVectorSimilarity Implementation
// ============================================================================

float ContextVectorSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    // PatternData doesn't contain ContextVector, so return default
    // This method should be called with ComputeFromContext instead
    return 0.0f;
}

float ContextVectorSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    // FeatureVector doesn't represent ContextVector, use dense vector similarity
    return a.CosineSimilarity(b);
}

float ContextVectorSimilarity::ComputeFromContext(const ContextVector& a, const ContextVector& b) const {
    return CosineSimilarity(a, b);
}

float ContextVectorSimilarity::CosineSimilarity(const ContextVector& a, const ContextVector& b) {
    // Use built-in CosineSimilarity method
    return a.CosineSimilarity(b);
}

// ============================================================================
// TemporalSimilarity Implementation
// ============================================================================

float TemporalSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    // PatternData doesn't contain timestamps
    return 0.0f;
}

float TemporalSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    // FeatureVector doesn't contain timestamps
    return 0.0f;
}

float TemporalSimilarity::ComputeFromTimestamps(Timestamp t1, Timestamp t2) const {
    // Compute time difference in milliseconds
    auto diff = t1 > t2 ? t1 - t2 : t2 - t1;
    int64_t diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();

    if (diff_ms == 0) {
        return 1.0f;
    }

    // Exponential decay: similarity = exp(-diff / window)
    float normalized_diff = static_cast<float>(diff_ms) / time_window_ms_;
    return std::exp(-normalized_diff);
}

// ============================================================================
// HierarchicalSimilarity Implementation
// ============================================================================

float HierarchicalSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    // PatternData doesn't contain sub-patterns
    return 0.0f;
}

float HierarchicalSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    // FeatureVector doesn't contain sub-patterns
    return 0.0f;
}

float HierarchicalSimilarity::ComputeFromSubPatterns(const std::vector<PatternID>& a,
                                                     const std::vector<PatternID>& b) const {
    std::set<PatternID> set_a(a.begin(), a.end());
    std::set<PatternID> set_b(b.begin(), b.end());

    return JaccardSimilarity(set_a, set_b);
}

float HierarchicalSimilarity::JaccardSimilarity(const std::set<PatternID>& a,
                                                const std::set<PatternID>& b) {
    if (a.empty() && b.empty()) {
        return 1.0f;
    }

    if (a.empty() || b.empty()) {
        return 0.0f;
    }

    // Compute intersection
    std::vector<PatternID> intersection;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                         std::back_inserter(intersection));

    // Compute union
    std::vector<PatternID> union_set;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                  std::back_inserter(union_set));

    if (union_set.empty()) {
        return 1.0f;
    }

    return static_cast<float>(intersection.size()) / union_set.size();
}

// ============================================================================
// StatisticalProfileSimilarity Implementation
// ============================================================================

StatisticalProfileSimilarity::Profile StatisticalProfileSimilarity::Profile::FromNode(const PatternNode& node) {
    Profile profile;
    profile.access_count = node.GetAccessCount();
    profile.confidence_score = node.GetConfidenceScore();
    profile.base_activation = node.GetBaseActivation();
    profile.age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        node.GetAge()).count();
    return profile;
}

StatisticalProfileSimilarity::Profile StatisticalProfileSimilarity::Profile::Create(
    uint32_t access_count, float confidence, float activation, int64_t age_ms) {
    Profile profile;
    profile.access_count = access_count;
    profile.confidence_score = confidence;
    profile.base_activation = activation;
    profile.age_ms = age_ms;
    return profile;
}

void StatisticalProfileSimilarity::NormalizeWeights() {
    float sum = std::accumulate(weights_.begin(), weights_.end(), 0.0f);
    if (sum > 1e-10f) {
        for (auto& w : weights_) {
            w /= sum;
        }
    } else {
        std::fill(weights_.begin(), weights_.end(), 0.25f);
    }
}

float StatisticalProfileSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    // PatternData doesn't contain statistical profiles
    return 0.0f;
}

float StatisticalProfileSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    // FeatureVector doesn't contain statistical profiles
    return 0.0f;
}

float StatisticalProfileSimilarity::ComputeFromProfiles(const Profile& a, const Profile& b) const {
    return CompareProfiles(a, b, weights_);
}

float StatisticalProfileSimilarity::CompareProfiles(const Profile& a, const Profile& b,
                                                   const std::vector<float>& weights) {
    std::vector<float> diffs(4);

    // Access count similarity (normalized)
    uint32_t max_access = std::max(a.access_count, b.access_count);
    if (max_access > 0) {
        uint32_t min_access = std::min(a.access_count, b.access_count);
        diffs[0] = static_cast<float>(min_access) / max_access;
    } else {
        diffs[0] = 1.0f;
    }

    // Confidence similarity
    diffs[1] = 1.0f - std::abs(a.confidence_score - b.confidence_score);

    // Activation similarity
    diffs[2] = 1.0f - std::min(1.0f, std::abs(a.base_activation - b.base_activation));

    // Age similarity (exponential decay)
    int64_t age_diff = std::abs(a.age_ms - b.age_ms);
    if (age_diff == 0) {
        diffs[3] = 1.0f;
    } else {
        // Use 1 day (86400000 ms) as reference window
        float normalized_age_diff = static_cast<float>(age_diff) / 86400000.0f;
        diffs[3] = std::exp(-normalized_age_diff);
    }

    // Weighted average
    float similarity = 0.0f;
    for (size_t i = 0; i < 4; ++i) {
        similarity += weights[i] * diffs[i];
    }

    return similarity;
}

// ============================================================================
// TypeSimilarity Implementation
// ============================================================================

float TypeSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    // PatternData doesn't contain PatternType
    return 0.0f;
}

float TypeSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    // FeatureVector doesn't contain PatternType
    return 0.0f;
}

float TypeSimilarity::ComputeFromTypes(PatternType t1, PatternType t2) const {
    if (t1 == t2) {
        return 1.0f;
    }

    if (strict_) {
        return 0.0f;
    }

    // Non-strict: check if types are related
    return AreRelated(t1, t2) ? 0.5f : 0.0f;
}

bool TypeSimilarity::AreRelated(PatternType t1, PatternType t2) {
    // ATOMIC is fundamentally different from COMPOSITE and META
    if (t1 == PatternType::ATOMIC && (t2 == PatternType::COMPOSITE || t2 == PatternType::META)) {
        return false;
    }
    if ((t1 == PatternType::COMPOSITE || t1 == PatternType::META) && t2 == PatternType::ATOMIC) {
        return false;
    }

    // COMPOSITE and META are related (both are hierarchical)
    if ((t1 == PatternType::COMPOSITE || t1 == PatternType::META) &&
        (t2 == PatternType::COMPOSITE || t2 == PatternType::META)) {
        return true;
    }

    return false;
}

// ============================================================================
// MetadataSimilarity Implementation
// ============================================================================

MetadataSimilarity::MetadataSimilarity() {
    // Default: use all metrics with equal weight
    AddMetric(std::make_shared<ContextVectorSimilarity>(), 1.0f);
    AddMetric(std::make_shared<TemporalSimilarity>(), 1.0f);
    AddMetric(std::make_shared<HierarchicalSimilarity>(), 1.0f);
    AddMetric(std::make_shared<StatisticalProfileSimilarity>(), 1.0f);
    AddMetric(std::make_shared<TypeSimilarity>(), 1.0f);
}

MetadataSimilarity::MetadataSimilarity(bool use_context, bool use_temporal, bool use_hierarchical,
                                      bool use_statistical, bool use_type) {
    if (use_context) {
        AddMetric(std::make_shared<ContextVectorSimilarity>(), 1.0f);
    }
    if (use_temporal) {
        AddMetric(std::make_shared<TemporalSimilarity>(), 1.0f);
    }
    if (use_hierarchical) {
        AddMetric(std::make_shared<HierarchicalSimilarity>(), 1.0f);
    }
    if (use_statistical) {
        AddMetric(std::make_shared<StatisticalProfileSimilarity>(), 1.0f);
    }
    if (use_type) {
        AddMetric(std::make_shared<TypeSimilarity>(), 1.0f);
    }
}

float MetadataSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float MetadataSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (metrics_.empty()) {
        return 0.0f;
    }

    float total_similarity = 0.0f;
    for (size_t i = 0; i < metrics_.size(); ++i) {
        float sim = metrics_[i].first->ComputeFromFeatures(a, b);
        total_similarity += normalized_weights_[i] * sim;
    }

    return total_similarity;
}

void MetadataSimilarity::AddMetric(std::shared_ptr<SimilarityMetric> metric, float weight) {
    if (!metric) {
        return;
    }

    metrics_.emplace_back(metric, weight);
    NormalizeWeights();
}

void MetadataSimilarity::Clear() {
    metrics_.clear();
    normalized_weights_.clear();
}

void MetadataSimilarity::NormalizeWeights() {
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
