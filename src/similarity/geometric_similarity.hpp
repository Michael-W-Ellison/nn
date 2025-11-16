// File: src/similarity/geometric_similarity.hpp
#pragma once

#include "similarity_metric.hpp"
#include <vector>
#include <cmath>

namespace dpan {

/// Point in N-dimensional space
template<size_t N>
struct Point {
    float coords[N];

    Point() {
        for (size_t i = 0; i < N; ++i) {
            coords[i] = 0.0f;
        }
    }

    explicit Point(const float* data) {
        for (size_t i = 0; i < N; ++i) {
            coords[i] = data[i];
        }
    }

    float& operator[](size_t i) { return coords[i]; }
    const float& operator[](size_t i) const { return coords[i]; }

    // Euclidean distance to another point
    float DistanceTo(const Point<N>& other) const {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float diff = coords[i] - other.coords[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // Squared distance (faster, no sqrt)
    float SquaredDistanceTo(const Point<N>& other) const {
        float sum = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            float diff = coords[i] - other.coords[i];
            sum += diff * diff;
        }
        return sum;
    }
};

/// Point set extracted from feature vector
template<size_t N>
struct PointSet {
    std::vector<Point<N>> points;

    size_t Size() const { return points.size(); }

    // Extract point set from feature vector
    // Interprets consecutive N values as N-dimensional points
    static PointSet<N> FromFeatureVector(const FeatureVector& features) {
        PointSet<N> result;
        const auto& data = features.Data();

        size_t num_points = data.size() / N;
        result.points.reserve(num_points);

        for (size_t i = 0; i < num_points; ++i) {
            result.points.emplace_back(&data[i * N]);
        }

        return result;
    }
};

/// Hausdorff Distance
///
/// Measures the maximum distance from any point in one set
/// to its nearest neighbor in the other set.
/// Lower values = more similar (0.0 = identical)
///
/// This is converted to similarity by normalization:
/// similarity = 1.0 / (1.0 + hausdorff_distance)
class HausdorffSimilarity : public SimilarityMetric {
public:
    HausdorffSimilarity() = default;

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "Hausdorff"; }
    bool IsSymmetric() const override { return true; }
    bool IsMetric() const override { return true; }

private:
    template<size_t N>
    static float ComputeHausdorff(const PointSet<N>& a, const PointSet<N>& b);
};

/// Chamfer Distance
///
/// Measures the average distance from points in one set
/// to their nearest neighbors in the other set.
/// Lower values = more similar (0.0 = identical)
///
/// Converted to similarity: similarity = 1.0 / (1.0 + chamfer_distance)
class ChamferSimilarity : public SimilarityMetric {
public:
    ChamferSimilarity() = default;

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "Chamfer"; }
    bool IsSymmetric() const override { return true; }
    bool IsMetric() const override { return false; }  // Chamfer is not a true metric

private:
    template<size_t N>
    static float ComputeChamfer(const PointSet<N>& a, const PointSet<N>& b);
};

/// Modified Hausdorff Distance
///
/// Uses average instead of maximum, making it more robust to outliers.
/// Lower values = more similar (0.0 = identical)
///
/// Converted to similarity: similarity = 1.0 / (1.0 + distance)
class ModifiedHausdorffSimilarity : public SimilarityMetric {
public:
    ModifiedHausdorffSimilarity() = default;

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "ModifiedHausdorff"; }
    bool IsSymmetric() const override { return true; }
    bool IsMetric() const override { return false; }

private:
    template<size_t N>
    static float ComputeModifiedHausdorff(const PointSet<N>& a, const PointSet<N>& b);
};

/// Procrustes Distance
///
/// Measures shape similarity after optimal alignment (translation, rotation, scaling).
/// Uses simple centroid alignment and scaling normalization.
/// Lower values = more similar (0.0 = identical shape)
///
/// Converted to similarity: similarity = 1.0 / (1.0 + distance)
class ProcrusteSimilarity : public SimilarityMetric {
public:
    ProcrusteSimilarity() = default;

    float Compute(const PatternData& a, const PatternData& b) const override;
    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override;
    std::string GetName() const override { return "Procrustes"; }
    bool IsSymmetric() const override { return true; }
    bool IsMetric() const override { return false; }

private:
    template<size_t N>
    static float ComputeProcrustes(const PointSet<N>& a, const PointSet<N>& b);

    template<size_t N>
    static Point<N> ComputeCentroid(const PointSet<N>& points);

    template<size_t N>
    static PointSet<N> CenterAndNormalize(const PointSet<N>& points);
};

} // namespace dpan
