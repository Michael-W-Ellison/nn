// File: src/memory/utility_calculator.hpp
//
// Utility Calculator for Pattern and Association Importance
//
// This module implements a utility scoring system to determine which patterns
// and associations are most valuable to keep in fast memory. The utility score
// combines multiple factors including access frequency, recency, association
// strength, and confidence.
//
// Mathematical Foundation:
//   U(p) = w_f × F(p) + w_r × R(p) + w_a × A(p) + w_c × C(p)
//
// Where:
//   F(p) = access frequency score ∈ [0,1]
//   R(p) = recency score ∈ [0,1]
//   A(p) = association strength score ∈ [0,1]
//   C(p) = confidence score ∈ [0,1]
//   w_f, w_r, w_a, w_c = weights (sum to 1.0)

#pragma once

#include "core/pattern_node.hpp"
#include "core/types.hpp"
#include "association/association_edge.hpp"
#include <chrono>
#include <unordered_map>
#include <shared_mutex>
#include <vector>

namespace dpan {

// Forward declarations
struct AccessStats;
class AccessTracker;

/// Utility calculation for patterns and associations
///
/// Calculates importance scores based on multiple factors to guide
/// memory management decisions (pruning, tier placement, etc.)
class UtilityCalculator {
public:
    /// Configuration for utility calculation
    struct Config {
        // Weight parameters (must sum to 1.0)
        float frequency_weight{0.3f};      ///< Weight for access frequency
        float recency_weight{0.3f};        ///< Weight for recency
        float association_weight{0.25f};   ///< Weight for association strength
        float confidence_weight{0.15f};    ///< Weight for pattern confidence

        // Decay constants
        float frequency_decay{0.01f};      ///< λ_f for frequency saturation
        float recency_decay{0.05f};        ///< λ_r for recency decay (per hour)

        // Normalization parameters
        float max_access_count{1000.0f};   ///< For access count normalization

        /// Validate configuration
        bool IsValid() const;
    };

    /// Construct with default configuration
    UtilityCalculator();

    /// Construct with custom configuration
    explicit UtilityCalculator(const Config& config);

    /// Calculate utility score for a pattern
    ///
    /// @param pattern Pattern node to evaluate
    /// @param stats Access statistics for the pattern
    /// @param associations Associations involving this pattern
    /// @return Utility score in [0,1]
    float CalculatePatternUtility(
        const PatternNode& pattern,
        const AccessStats& stats,
        const std::vector<AssociationEdge>& associations
    ) const;

    /// Calculate utility score for an association
    ///
    /// @param edge Association edge to evaluate
    /// @param source_stats Access stats for source pattern
    /// @param target_stats Access stats for target pattern
    /// @return Utility score in [0,1]
    float CalculateAssociationUtility(
        const AssociationEdge& edge,
        const AccessStats& source_stats,
        const AccessStats& target_stats
    ) const;

    /// Detailed breakdown of utility score components (for debugging/analysis)
    struct UtilityBreakdown {
        float frequency_score{0.0f};
        float recency_score{0.0f};
        float association_score{0.0f};
        float confidence_score{0.0f};
        float total{0.0f};
    };

    /// Get detailed breakdown of utility components
    ///
    /// @param pattern Pattern to evaluate
    /// @param stats Access statistics
    /// @param associations Associated edges
    /// @return Breakdown of all score components
    UtilityBreakdown GetUtilityBreakdown(
        const PatternNode& pattern,
        const AccessStats& stats,
        const std::vector<AssociationEdge>& associations
    ) const;

    /// Update configuration
    void SetConfig(const Config& config);

    /// Get current configuration
    const Config& GetConfig() const { return config_; }

private:
    Config config_;

    // Individual component calculations
    float CalculateFrequencyScore(uint64_t access_count) const;
    float CalculateRecencyScore(Timestamp::Duration time_since_access) const;
    float CalculateAssociationScore(const std::vector<AssociationEdge>& associations) const;
    float CalculateConfidenceScore(const PatternNode& pattern) const;

    // Validation
    void ValidateWeights() const;
};

/// Statistics for tracking pattern/association access patterns
struct AccessStats {
    uint64_t access_count{0};           ///< Total number of accesses
    Timestamp last_access;              ///< Timestamp of last access
    Timestamp creation_time;            ///< When first tracked

    // Exponential moving average of access interval
    float avg_access_interval{0.0f};    ///< Average time between accesses (seconds)

    /// Record a new access
    void RecordAccess(Timestamp timestamp = Timestamp::Now());

    /// Get time since last access
    Timestamp::Duration TimeSinceLastAccess() const;

    /// Get age of the pattern/association
    Timestamp::Duration Age() const;

    /// Serialize to output stream
    void Serialize(std::ostream& out) const;

    /// Deserialize from input stream
    static AccessStats Deserialize(std::istream& in);
};

/// Centralized tracking of access statistics for patterns and associations
///
/// Thread-safe tracker for maintaining access history to support
/// utility-based memory management decisions.
class AccessTracker {
public:
    /// Record access to a pattern
    ///
    /// @param pattern Pattern ID accessed
    /// @param timestamp Time of access (default: now)
    void RecordPatternAccess(PatternID pattern, Timestamp timestamp = Timestamp::Now());

    /// Record access to an association
    ///
    /// @param source Source pattern ID
    /// @param target Target pattern ID
    /// @param timestamp Time of access (default: now)
    void RecordAssociationAccess(PatternID source, PatternID target,
                                  Timestamp timestamp = Timestamp::Now());

    /// Get access statistics for a pattern
    ///
    /// @param pattern Pattern ID to query
    /// @return Pointer to stats (nullptr if not tracked)
    const AccessStats* GetPatternStats(PatternID pattern) const;

    /// Get access statistics for an association
    ///
    /// @param source Source pattern ID
    /// @param target Target pattern ID
    /// @return Pointer to stats (nullptr if not tracked)
    const AccessStats* GetAssociationStats(PatternID source, PatternID target) const;

    /// Remove statistics older than cutoff time
    ///
    /// @param cutoff_time Remove stats for items not accessed since this time
    /// @return Number of entries removed
    size_t PruneOldStats(Timestamp cutoff_time);

    /// Clear all statistics
    void Clear();

    /// Get number of tracked patterns
    size_t GetTrackedPatternCount() const;

    /// Get number of tracked associations
    size_t GetTrackedAssociationCount() const;

private:
    std::unordered_map<PatternID, AccessStats> pattern_stats_;

    // Use custom hash for pattern pairs
    struct PatternPairHash {
        size_t operator()(const std::pair<PatternID, PatternID>& p) const {
            size_t h1 = std::hash<PatternID>{}(p.first);
            size_t h2 = std::hash<PatternID>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    std::unordered_map<std::pair<PatternID, PatternID>, AccessStats, PatternPairHash> association_stats_;

    mutable std::shared_mutex mutex_;
};

} // namespace dpan
