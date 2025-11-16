// File: src/storage/indices/temporal_index.hpp
#pragma once

#include "core/types.hpp"
#include <map>
#include <vector>
#include <mutex>
#include <optional>

namespace dpan {

/// Temporal index for fast time-based pattern lookups
///
/// Uses a Red-Black tree (std::map) to maintain patterns sorted by timestamp,
/// enabling efficient range queries and temporal pattern discovery.
///
/// Thread-safe with mutex protection for concurrent access.
class TemporalIndex {
public:
    /// Default constructor
    TemporalIndex() = default;

    /// Insert a pattern with its timestamp
    /// @param id Pattern identifier
    /// @param timestamp Pattern creation or access time
    void Insert(PatternID id, Timestamp timestamp);

    /// Remove a pattern from the index
    /// @param id Pattern identifier
    /// @return true if removed, false if not found
    bool Remove(PatternID id);

    /// Find patterns within a time range
    /// @param start Start timestamp (inclusive)
    /// @param end End timestamp (inclusive)
    /// @param max_results Maximum number of results to return
    /// @return Vector of pattern IDs in chronological order
    std::vector<PatternID> FindInRange(
        Timestamp start,
        Timestamp end,
        size_t max_results = 1000) const;

    /// Find patterns before a specific time
    /// @param timestamp Upper bound (exclusive)
    /// @param max_results Maximum number of results to return
    /// @return Vector of pattern IDs in chronological order
    std::vector<PatternID> FindBefore(
        Timestamp timestamp,
        size_t max_results = 1000) const;

    /// Find patterns after a specific time
    /// @param timestamp Lower bound (exclusive)
    /// @param max_results Maximum number of results to return
    /// @return Vector of pattern IDs in chronological order
    std::vector<PatternID> FindAfter(
        Timestamp timestamp,
        size_t max_results = 1000) const;

    /// Find the N most recent patterns
    /// @param n Number of patterns to return
    /// @return Vector of pattern IDs in reverse chronological order
    std::vector<PatternID> FindMostRecent(size_t n) const;

    /// Find the N oldest patterns
    /// @param n Number of patterns to return
    /// @return Vector of pattern IDs in chronological order
    std::vector<PatternID> FindOldest(size_t n) const;

    /// Get timestamp for a specific pattern
    /// @param id Pattern identifier
    /// @return Timestamp if found, nullopt otherwise
    std::optional<Timestamp> GetTimestamp(PatternID id) const;

    /// Get total number of indexed patterns
    /// @return Number of patterns
    size_t Size() const;

    /// Clear all entries from the index
    void Clear();

    /// Get statistics about the index
    struct Stats {
        size_t total_patterns{0};
        Timestamp earliest;
        Timestamp latest;
        double avg_patterns_per_second{0.0};
    };

    Stats GetStats() const;

private:
    /// Map from timestamp to pattern ID (allows multiple patterns per timestamp)
    std::multimap<Timestamp, PatternID> time_to_pattern_;

    /// Map from pattern ID to timestamp (for removal and lookup)
    std::map<PatternID, Timestamp> pattern_to_time_;

    /// Mutex for thread safety
    mutable std::mutex mutex_;
};

} // namespace dpan
