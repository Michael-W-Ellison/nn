// File: src/similarity/geometric_similarity.cpp
#include "geometric_similarity.hpp"
#include <algorithm>
#include <limits>
#include <cmath>

namespace dpan {

// ============================================================================
// Hausdorff Similarity
// ============================================================================

template<size_t N>
float HausdorffSimilarity::ComputeHausdorff(const PointSet<N>& a, const PointSet<N>& b) {
    if (a.Size() == 0 || b.Size() == 0) {
        return std::numeric_limits<float>::infinity();
    }

    // Hausdorff distance: max(directed_hausdorff(a,b), directed_hausdorff(b,a))

    // Directed Hausdorff from a to b
    float max_min_dist_ab = 0.0f;
    for (const auto& point_a : a.points) {
        float min_dist = std::numeric_limits<float>::infinity();
        for (const auto& point_b : b.points) {
            float dist = point_a.DistanceTo(point_b);
            min_dist = std::min(min_dist, dist);
        }
        max_min_dist_ab = std::max(max_min_dist_ab, min_dist);
    }

    // Directed Hausdorff from b to a
    float max_min_dist_ba = 0.0f;
    for (const auto& point_b : b.points) {
        float min_dist = std::numeric_limits<float>::infinity();
        for (const auto& point_a : a.points) {
            float dist = point_b.DistanceTo(point_a);
            min_dist = std::min(min_dist, dist);
        }
        max_min_dist_ba = std::max(max_min_dist_ba, min_dist);
    }

    return std::max(max_min_dist_ab, max_min_dist_ba);
}

float HausdorffSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float HausdorffSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    // Determine point dimensionality based on feature vector size
    // For simplicity, use 2D points if dimension is even and >= 2
    // Otherwise use 1D points
    size_t dim = a.Dimension();

    float distance;
    if (dim >= 2 && dim % 2 == 0) {
        // Use 2D points
        auto points_a = PointSet<2>::FromFeatureVector(a);
        auto points_b = PointSet<2>::FromFeatureVector(b);
        distance = ComputeHausdorff(points_a, points_b);
    } else {
        // Use 1D points
        auto points_a = PointSet<1>::FromFeatureVector(a);
        auto points_b = PointSet<1>::FromFeatureVector(b);
        distance = ComputeHausdorff(points_a, points_b);
    }

    // Convert distance to similarity: similarity = 1.0 / (1.0 + distance)
    if (std::isinf(distance)) {
        return 0.0f;
    }
    return 1.0f / (1.0f + distance);
}

// ============================================================================
// Chamfer Similarity
// ============================================================================

template<size_t N>
float ChamferSimilarity::ComputeChamfer(const PointSet<N>& a, const PointSet<N>& b) {
    if (a.Size() == 0 || b.Size() == 0) {
        return std::numeric_limits<float>::infinity();
    }

    // Chamfer distance: average of directed chamfer distances

    // Directed Chamfer from a to b
    float sum_ab = 0.0f;
    for (const auto& point_a : a.points) {
        float min_dist = std::numeric_limits<float>::infinity();
        for (const auto& point_b : b.points) {
            float dist = point_a.DistanceTo(point_b);
            min_dist = std::min(min_dist, dist);
        }
        sum_ab += min_dist;
    }
    float avg_ab = sum_ab / a.Size();

    // Directed Chamfer from b to a
    float sum_ba = 0.0f;
    for (const auto& point_b : b.points) {
        float min_dist = std::numeric_limits<float>::infinity();
        for (const auto& point_a : a.points) {
            float dist = point_b.DistanceTo(point_a);
            min_dist = std::min(min_dist, dist);
        }
        sum_ba += min_dist;
    }
    float avg_ba = sum_ba / b.Size();

    return (avg_ab + avg_ba) / 2.0f;
}

float ChamferSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float ChamferSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    size_t dim = a.Dimension();

    float distance;
    if (dim >= 2 && dim % 2 == 0) {
        auto points_a = PointSet<2>::FromFeatureVector(a);
        auto points_b = PointSet<2>::FromFeatureVector(b);
        distance = ComputeChamfer(points_a, points_b);
    } else {
        auto points_a = PointSet<1>::FromFeatureVector(a);
        auto points_b = PointSet<1>::FromFeatureVector(b);
        distance = ComputeChamfer(points_a, points_b);
    }

    if (std::isinf(distance)) {
        return 0.0f;
    }
    return 1.0f / (1.0f + distance);
}

// ============================================================================
// Modified Hausdorff Similarity
// ============================================================================

template<size_t N>
float ModifiedHausdorffSimilarity::ComputeModifiedHausdorff(const PointSet<N>& a, const PointSet<N>& b) {
    if (a.Size() == 0 || b.Size() == 0) {
        return std::numeric_limits<float>::infinity();
    }

    // Modified Hausdorff: average of minimum distances instead of maximum

    // From a to b
    float sum_ab = 0.0f;
    for (const auto& point_a : a.points) {
        float min_dist = std::numeric_limits<float>::infinity();
        for (const auto& point_b : b.points) {
            float dist = point_a.DistanceTo(point_b);
            min_dist = std::min(min_dist, dist);
        }
        sum_ab += min_dist;
    }
    float avg_ab = sum_ab / a.Size();

    // From b to a
    float sum_ba = 0.0f;
    for (const auto& point_b : b.points) {
        float min_dist = std::numeric_limits<float>::infinity();
        for (const auto& point_a : a.points) {
            float dist = point_b.DistanceTo(point_a);
            min_dist = std::min(min_dist, dist);
        }
        sum_ba += min_dist;
    }
    float avg_ba = sum_ba / b.Size();

    return std::max(avg_ab, avg_ba);
}

float ModifiedHausdorffSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float ModifiedHausdorffSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0) {
        return 0.0f;
    }

    size_t dim = a.Dimension();

    float distance;
    if (dim >= 2 && dim % 2 == 0) {
        auto points_a = PointSet<2>::FromFeatureVector(a);
        auto points_b = PointSet<2>::FromFeatureVector(b);
        distance = ComputeModifiedHausdorff(points_a, points_b);
    } else {
        auto points_a = PointSet<1>::FromFeatureVector(a);
        auto points_b = PointSet<1>::FromFeatureVector(b);
        distance = ComputeModifiedHausdorff(points_a, points_b);
    }

    if (std::isinf(distance)) {
        return 0.0f;
    }
    return 1.0f / (1.0f + distance);
}

// ============================================================================
// Procrustes Similarity
// ============================================================================

template<size_t N>
Point<N> ProcrusteSimilarity::ComputeCentroid(const PointSet<N>& points) {
    Point<N> centroid;
    for (size_t i = 0; i < N; ++i) {
        centroid[i] = 0.0f;
    }

    if (points.Size() == 0) {
        return centroid;
    }

    for (const auto& point : points.points) {
        for (size_t i = 0; i < N; ++i) {
            centroid[i] += point[i];
        }
    }

    for (size_t i = 0; i < N; ++i) {
        centroid[i] /= points.Size();
    }

    return centroid;
}

template<size_t N>
PointSet<N> ProcrusteSimilarity::CenterAndNormalize(const PointSet<N>& points) {
    PointSet<N> result;
    result.points.reserve(points.Size());

    // Compute centroid
    Point<N> centroid = ComputeCentroid(points);

    // Center points
    for (const auto& point : points.points) {
        Point<N> centered;
        for (size_t i = 0; i < N; ++i) {
            centered[i] = point[i] - centroid[i];
        }
        result.points.push_back(centered);
    }

    // Compute scale (root mean square distance from origin)
    float scale = 0.0f;
    for (const auto& point : result.points) {
        for (size_t i = 0; i < N; ++i) {
            scale += point[i] * point[i];
        }
    }
    scale = std::sqrt(scale / result.points.size());

    // Normalize by scale
    if (scale > 1e-6f) {
        for (auto& point : result.points) {
            for (size_t i = 0; i < N; ++i) {
                point[i] /= scale;
            }
        }
    }

    return result;
}

template<size_t N>
float ProcrusteSimilarity::ComputeProcrustes(const PointSet<N>& a, const PointSet<N>& b) {
    if (a.Size() == 0 || b.Size() == 0 || a.Size() != b.Size()) {
        return std::numeric_limits<float>::infinity();
    }

    // Center and normalize both point sets
    auto a_normalized = CenterAndNormalize(a);
    auto b_normalized = CenterAndNormalize(b);

    // Compute sum of squared distances
    // (Note: full Procrustes would include optimal rotation via SVD,
    //  but we use simplified version with just centering and scaling)
    float sum_sq_dist = 0.0f;
    for (size_t i = 0; i < a_normalized.Size(); ++i) {
        sum_sq_dist += a_normalized.points[i].SquaredDistanceTo(b_normalized.points[i]);
    }

    return std::sqrt(sum_sq_dist / a_normalized.Size());
}

float ProcrusteSimilarity::Compute(const PatternData& a, const PatternData& b) const {
    return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
}

float ProcrusteSimilarity::ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const {
    if (a.Dimension() == 0 || b.Dimension() == 0 || a.Dimension() != b.Dimension()) {
        return 0.0f;
    }

    size_t dim = a.Dimension();

    float distance;
    if (dim >= 2 && dim % 2 == 0) {
        auto points_a = PointSet<2>::FromFeatureVector(a);
        auto points_b = PointSet<2>::FromFeatureVector(b);
        distance = ComputeProcrustes(points_a, points_b);
    } else {
        auto points_a = PointSet<1>::FromFeatureVector(a);
        auto points_b = PointSet<1>::FromFeatureVector(b);
        distance = ComputeProcrustes(points_a, points_b);
    }

    if (std::isinf(distance)) {
        return 0.0f;
    }
    return 1.0f / (1.0f + distance);
}

} // namespace dpan
