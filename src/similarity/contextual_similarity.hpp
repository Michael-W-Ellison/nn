// File: src/similarity/contextual_similarity.hpp
#pragma once

#include "similarity_metric.hpp"
#include "core/types.hpp"
#include "core/pattern_node.hpp"
#include <set>
#include <cmath>

namespace dpan {

/// ContextVector Similarity
///
/// Compares sparse context vectors using cosine similarity.
/// ContextVectors are maps of string dimensions to float values.
///
/// Use cases:
/// - Semantic similarity based on metadata
/// - Tag-based similarity
/// - Feature-based context matching
class ContextVectorSimilarity : public SimilarityMetric {
public:
    ContextVectorSimilarity() = default;

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;

    /// Compute similarity between two ContextVectors
    float ComputeFromContext(const ContextVector& a, const ContextVector& b) const;

    std::string GetName() const override { return "ContextVector"; }
    bool IsSymmetric() const override { return true; }

private:
    /// Compute cosine similarity between sparse vectors
    static float CosineSimilarity(const ContextVector& a, const ContextVector& b);
};

/// Temporal Proximity Similarity
///
/// Compares patterns based on their temporal proximity.
/// Useful for finding patterns that occur close in time.
///
/// Use cases:
/// - Event correlation
/// - Temporal pattern mining
/// - Time-series analysis
class TemporalSimilarity : public SimilarityMetric {
public:
    /// Constructor
    /// @param time_window_ms Maximum time window for full similarity (milliseconds)
    explicit TemporalSimilarity(int64_t time_window_ms = 1000)
        : time_window_ms_(time_window_ms) {}

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;

    /// Compute temporal similarity between timestamps
    float ComputeFromTimestamps(Timestamp t1, Timestamp t2) const;

    std::string GetName() const override { return "Temporal"; }
    bool IsSymmetric() const override { return true; }

private:
    int64_t time_window_ms_;
};

/// Hierarchical Similarity
///
/// Compares patterns based on their hierarchical structure.
/// Uses Jaccard similarity on sub-pattern sets.
///
/// Use cases:
/// - Structural pattern matching
/// - Compositional similarity
/// - Hierarchy-based clustering
class HierarchicalSimilarity : public SimilarityMetric {
public:
    HierarchicalSimilarity() = default;

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;

    /// Compute Jaccard similarity between sub-pattern sets
    float ComputeFromSubPatterns(const std::vector<PatternID>& a,
                                 const std::vector<PatternID>& b) const;

    std::string GetName() const override { return "Hierarchical"; }
    bool IsSymmetric() const override { return true; }

private:
    /// Compute Jaccard similarity: |A ∩ B| / |A ∪ B|
    static float JaccardSimilarity(const std::set<PatternID>& a,
                                  const std::set<PatternID>& b);
};

/// Statistical Profile Similarity
///
/// Compares patterns based on their usage statistics.
/// Considers access counts, confidence scores, and other metadata.
///
/// Use cases:
/// - Usage pattern matching
/// - Popularity-based similarity
/// - Quality-based filtering
class StatisticalProfileSimilarity : public SimilarityMetric {
public:
    /// Statistical profile of a pattern
    struct Profile {
        uint32_t access_count{0};
        float confidence_score{0.5f};
        float base_activation{0.0f};
        int64_t age_ms{0};

        /// Create profile from PatternNode
        static Profile FromNode(const PatternNode& node);

        /// Create profile from values
        static Profile Create(uint32_t access_count, float confidence,
                            float activation, int64_t age_ms);
    };

    /// Constructor
    /// @param weights Weights for [access, confidence, activation, age]
    explicit StatisticalProfileSimilarity(const std::vector<float>& weights = {1.0f, 1.0f, 0.5f, 0.5f})
        : weights_(weights) {
        if (weights_.size() != 4) {
            weights_ = {1.0f, 1.0f, 0.5f, 0.5f};
        }
        NormalizeWeights();
    }

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;

    /// Compute similarity between statistical profiles
    float ComputeFromProfiles(const Profile& a, const Profile& b) const;

    std::string GetName() const override { return "StatisticalProfile"; }
    bool IsSymmetric() const override { return true; }

private:
    std::vector<float> weights_;

    void NormalizeWeights();

    /// Compare two profiles
    static float CompareProfiles(const Profile& a, const Profile& b,
                                const std::vector<float>& weights);
};

/// Type Similarity
///
/// Compares patterns based on their types.
/// Returns 1.0 for identical types, 0.0 for different types.
/// Can optionally consider type hierarchy.
///
/// Use cases:
/// - Type-based filtering
/// - Categorical pattern matching
/// - Type hierarchy navigation
class TypeSimilarity : public SimilarityMetric {
public:
    /// Constructor
    /// @param strict If true, only exact type matches return 1.0
    explicit TypeSimilarity(bool strict = true) : strict_(strict) {}

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;

    /// Compute similarity between pattern types
    float ComputeFromTypes(PatternType t1, PatternType t2) const;

    std::string GetName() const override { return "Type"; }
    bool IsSymmetric() const override { return true; }

private:
    bool strict_;

    /// Check if types are related in hierarchy
    static bool AreRelated(PatternType t1, PatternType t2);
};

/// Metadata Similarity (Composite)
///
/// Combines multiple contextual metrics into a single score.
/// Useful for holistic context-based comparison.
///
/// Use cases:
/// - Multi-faceted similarity search
/// - Context-aware pattern matching
/// - Comprehensive pattern comparison
class MetadataSimilarity : public SimilarityMetric {
public:
    /// Constructor with default weights
    MetadataSimilarity();

    /// Constructor with custom component weights
    /// @param use_context Include ContextVector similarity
    /// @param use_temporal Include temporal similarity
    /// @param use_hierarchical Include hierarchical similarity
    /// @param use_statistical Include statistical profile similarity
    /// @param use_type Include type similarity
    MetadataSimilarity(bool use_context, bool use_temporal, bool use_hierarchical,
                      bool use_statistical, bool use_type);

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;

    std::string GetName() const override { return "Metadata"; }
    bool IsSymmetric() const override { return true; }

    /// Add a contextual metric with weight
    void AddMetric(std::shared_ptr<SimilarityMetric> metric, float weight);

    /// Clear all metrics
    void Clear();

private:
    std::vector<std::pair<std::shared_ptr<SimilarityMetric>, float>> metrics_;
    std::vector<float> normalized_weights_;

    void NormalizeWeights();
};

} // namespace dpan
