// File: src/storage/pattern_database.hpp
#pragma once

#include "core/pattern_node.hpp"
#include <memory>
#include <vector>
#include <optional>
#include <string>

namespace dpan {

/// Storage statistics for monitoring database performance
struct StorageStats {
    /// Total number of patterns stored in the database
    size_t total_patterns{0};

    /// Total memory usage in bytes (for in-memory storage)
    size_t memory_usage_bytes{0};

    /// Total disk usage in bytes (for persistent storage)
    size_t disk_usage_bytes{0};

    /// Average lookup time in milliseconds
    float avg_lookup_time_ms{0.0f};

    /// Cache hit rate (0.0 to 1.0)
    float cache_hit_rate{0.0f};
};

/// Query options for database searches
struct QueryOptions {
    /// Maximum number of results to return
    size_t max_results{100};

    /// Minimum similarity threshold for similarity-based queries (0.0 to 1.0)
    float similarity_threshold{0.5f};

    /// Whether to use caching for this query
    bool use_cache{true};

    /// Minimum timestamp for time-range queries (optional)
    std::optional<Timestamp> min_timestamp;

    /// Maximum timestamp for time-range queries (optional)
    std::optional<Timestamp> max_timestamp;
};

/// Abstract interface for pattern storage backends
///
/// This interface provides a generic API for storing, retrieving, and querying
/// pattern nodes. Concrete implementations can use different storage backends
/// such as in-memory hash maps, memory-mapped files, or persistent databases.
///
/// Thread Safety: All methods must be thread-safe. Implementations should use
/// appropriate synchronization mechanisms (e.g., shared_mutex for read/write locks).
class PatternDatabase {
public:
    virtual ~PatternDatabase() = default;

    // ========================================================================
    // Core CRUD Operations
    // ========================================================================

    /// Store a new pattern in the database
    /// @param node The pattern node to store
    /// @return true if stored successfully, false if pattern already exists
    virtual bool Store(const PatternNode& node) = 0;

    /// Retrieve a pattern by its ID
    /// @param id The pattern ID to retrieve
    /// @return The pattern node if found, std::nullopt otherwise
    virtual std::optional<PatternNode> Retrieve(PatternID id) = 0;

    /// Update an existing pattern in the database
    /// @param node The pattern node with updated data
    /// @return true if updated successfully, false if pattern doesn't exist
    virtual bool Update(const PatternNode& node) = 0;

    /// Delete a pattern from the database
    /// @param id The pattern ID to delete
    /// @return true if deleted successfully, false if pattern doesn't exist
    virtual bool Delete(PatternID id) = 0;

    /// Check if a pattern exists in the database
    /// @param id The pattern ID to check
    /// @return true if pattern exists, false otherwise
    virtual bool Exists(PatternID id) const = 0;

    // ========================================================================
    // Batch Operations
    // ========================================================================

    /// Store multiple patterns in a single operation
    /// @param nodes Vector of pattern nodes to store
    /// @return Number of patterns successfully stored
    virtual size_t StoreBatch(const std::vector<PatternNode>& nodes) = 0;

    /// Retrieve multiple patterns in a single operation
    /// @param ids Vector of pattern IDs to retrieve
    /// @return Vector of pattern nodes (may be smaller than ids if some not found)
    virtual std::vector<PatternNode> RetrieveBatch(const std::vector<PatternID>& ids) = 0;

    /// Delete multiple patterns in a single operation
    /// @param ids Vector of pattern IDs to delete
    /// @return Number of patterns successfully deleted
    virtual size_t DeleteBatch(const std::vector<PatternID>& ids) = 0;

    // ========================================================================
    // Query Operations
    // ========================================================================

    /// Find all patterns of a specific type
    /// @param type The pattern type to search for
    /// @param options Query options (max_results, cache usage, etc.)
    /// @return Vector of pattern IDs matching the criteria
    virtual std::vector<PatternID> FindByType(
        PatternType type,
        const QueryOptions& options = {}) = 0;

    /// Find all patterns created within a time range
    /// @param start Start of the time range (inclusive)
    /// @param end End of the time range (inclusive)
    /// @param options Query options (max_results, cache usage, etc.)
    /// @return Vector of pattern IDs matching the criteria
    virtual std::vector<PatternID> FindByTimeRange(
        Timestamp start,
        Timestamp end,
        const QueryOptions& options = {}) = 0;

    /// Find all patterns in the database
    /// @param options Query options (max_results, cache usage, etc.)
    /// @return Vector of all pattern IDs
    virtual std::vector<PatternID> FindAll(const QueryOptions& options = {}) = 0;

    // ========================================================================
    // Statistics and Monitoring
    // ========================================================================

    /// Get the total number of patterns in the database
    /// @return Number of patterns
    virtual size_t Count() const = 0;

    /// Get detailed storage statistics
    /// @return StorageStats structure with current statistics
    virtual StorageStats GetStats() const = 0;

    // ========================================================================
    // Maintenance Operations
    // ========================================================================

    /// Flush any pending writes to disk (for persistent backends)
    /// For in-memory backends, this may be a no-op
    virtual void Flush() = 0;

    /// Compact the database to reclaim space
    /// For in-memory backends, this may defragment memory
    /// For persistent backends, this may merge database files
    virtual void Compact() = 0;

    /// Clear all patterns from the database
    /// WARNING: This operation cannot be undone
    virtual void Clear() = 0;

    // ========================================================================
    // Snapshot and Restore
    // ========================================================================

    /// Create a snapshot of the database at the specified path
    /// @param path File path where the snapshot should be saved
    /// @return true if snapshot created successfully, false otherwise
    virtual bool CreateSnapshot(const std::string& path) = 0;

    /// Restore the database from a snapshot
    /// @param path File path of the snapshot to restore
    /// @return true if restored successfully, false otherwise
    virtual bool RestoreSnapshot(const std::string& path) = 0;
};

/// Factory function to create a pattern database from configuration
///
/// The configuration file should specify the backend type and parameters.
/// Example config (JSON format):
/// {
///   "backend": "memory",  // or "rocksdb", "mmap"
///   "memory": {
///     "initial_capacity": 10000,
///     "enable_cache": true,
///     "cache_size": 1000
///   }
/// }
///
/// @param config_path Path to the configuration file
/// @return Unique pointer to a PatternDatabase implementation
std::unique_ptr<PatternDatabase> CreatePatternDatabase(const std::string& config_path);

} // namespace dpan
