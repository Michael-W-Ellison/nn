// File: src/storage/memory_backend.hpp
#pragma once

#include "storage/pattern_database.hpp"
#include <unordered_map>
#include <shared_mutex>
#include <atomic>

namespace dpan {

/// In-memory pattern storage backend using hash map
///
/// This implementation provides fast in-memory storage with optional
/// memory-mapped file backing for persistence. Thread-safe with
/// shared_mutex for concurrent read access.
///
/// Features:
/// - Fast O(1) lookup, insert, delete
/// - Thread-safe with shared_mutex (multiple readers, single writer)
/// - Statistics tracking for performance monitoring
/// - Optional memory-mapped file persistence
/// - Snapshot/restore for data backup
class MemoryBackend : public PatternDatabase {
public:
    /// Configuration for MemoryBackend
    struct Config {
        /// Whether to use memory-mapped file for persistence
        bool use_mmap{false};

        /// Path to memory-mapped file (required if use_mmap is true)
        std::string mmap_path;

        /// Initial capacity for the hash map (pre-allocation)
        size_t initial_capacity{10000};

        /// Whether to enable caching (future extension)
        bool enable_cache{true};

        /// Cache size in number of patterns (future extension)
        size_t cache_size{1000};
    };

    /// Construct MemoryBackend with configuration
    /// @param config Configuration options
    explicit MemoryBackend(const Config& config);

    /// Destructor - saves to mmap if enabled
    ~MemoryBackend() override;

    // ========================================================================
    // PatternDatabase Interface Implementation
    // ========================================================================

    bool Store(const PatternNode& node) override;
    std::optional<PatternNode> Retrieve(PatternID id) override;
    bool Update(const PatternNode& node) override;
    bool Delete(PatternID id) override;
    bool Exists(PatternID id) const override;

    size_t StoreBatch(const std::vector<PatternNode>& nodes) override;
    std::vector<PatternNode> RetrieveBatch(const std::vector<PatternID>& ids) override;
    size_t DeleteBatch(const std::vector<PatternID>& ids) override;

    std::vector<PatternID> FindByType(
        PatternType type,
        const QueryOptions& options) override;

    std::vector<PatternID> FindByTimeRange(
        Timestamp start,
        Timestamp end,
        const QueryOptions& options) override;

    std::vector<PatternID> FindAll(const QueryOptions& options) override;

    size_t Count() const override;
    StorageStats GetStats() const override;

    void Flush() override;
    void Compact() override;
    void Clear() override;

    bool CreateSnapshot(const std::string& path) override;
    bool RestoreSnapshot(const std::string& path) override;

private:
    // Configuration
    Config config_;

    // Thread synchronization (shared_mutex allows multiple readers, single writer)
    mutable std::shared_mutex mutex_;

    // Main storage: hash map from PatternID to PatternNode
    std::unordered_map<PatternID, PatternNode> patterns_;

    // Statistics tracking (atomics for lock-free updates)
    mutable std::atomic<uint64_t> total_lookups_{0};
    mutable std::atomic<uint64_t> cache_hits_{0};
    mutable std::atomic<uint64_t> total_lookup_time_ns_{0};

    // Memory-mapped file support (if enabled)
    void* mmap_ptr_{nullptr};
    size_t mmap_size_{0};
    int mmap_fd_{-1};

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Update performance statistics
    /// @param lookup_time_ns Lookup time in nanoseconds
    /// @param cache_hit Whether this was a cache hit
    void UpdateStats(uint64_t lookup_time_ns, bool cache_hit);

    /// Load patterns from memory-mapped file
    void LoadFromMmap();

    /// Save patterns to memory-mapped file
    void SaveToMmap();

    /// Unmap the memory-mapped file
    void UnmapFile();

    /// Helper to serialize a single pattern node to a stream
    static void SerializePattern(std::ostream& out, const PatternNode& node);

    /// Helper to deserialize a single pattern node from a stream
    static PatternNode DeserializePattern(std::istream& in);
};

} // namespace dpan
