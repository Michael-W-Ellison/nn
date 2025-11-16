// File: src/storage/memory_backend.cpp
#include "storage/memory_backend.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace dpan {

// ============================================================================
// Constructor and Destructor
// ============================================================================

MemoryBackend::MemoryBackend(const Config& config)
    : config_(config) {
    // Pre-allocate hash map capacity
    patterns_.reserve(config_.initial_capacity);

    // Load from mmap if enabled
    if (config_.use_mmap && !config_.mmap_path.empty()) {
        LoadFromMmap();
    }
}

MemoryBackend::~MemoryBackend() {
    // Save to mmap if enabled
    if (config_.use_mmap && !config_.mmap_path.empty()) {
        SaveToMmap();
    }

    // Unmap file if mapped
    UnmapFile();
}

// ============================================================================
// Core CRUD Operations
// ============================================================================

bool MemoryBackend::Store(const PatternNode& node) {
    auto start = std::chrono::steady_clock::now();

    // Exclusive lock for writing
    std::unique_lock<std::shared_mutex> lock(mutex_);

    PatternID id = node.GetID();

    // Check if already exists
    if (patterns_.find(id) != patterns_.end()) {
        return false;
    }

    // Clone the node to preserve all state
    patterns_.emplace(id, node.Clone());

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    UpdateStats(duration.count(), false);

    return true;
}

std::optional<PatternNode> MemoryBackend::Retrieve(PatternID id) {
    auto start = std::chrono::steady_clock::now();

    // Shared lock for reading
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto it = patterns_.find(id);

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    UpdateStats(duration.count(), false);

    if (it != patterns_.end()) {
        // Clone to return with all state preserved
        return it->second.Clone();
    }

    return std::nullopt;
}

bool MemoryBackend::Update(const PatternNode& node) {
    // Exclusive lock for writing
    std::unique_lock<std::shared_mutex> lock(mutex_);

    PatternID id = node.GetID();
    auto it = patterns_.find(id);

    if (it == patterns_.end()) {
        return false;
    }

    // Erase and re-insert (since PatternNode has deleted move assignment)
    patterns_.erase(it);
    patterns_.emplace(id, PatternNode(node.GetID(), node.GetData(), node.GetType()));
    return true;
}

bool MemoryBackend::Delete(PatternID id) {
    // Exclusive lock for writing
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto it = patterns_.find(id);
    if (it == patterns_.end()) {
        return false;
    }

    patterns_.erase(it);
    return true;
}

bool MemoryBackend::Exists(PatternID id) const {
    // Shared lock for reading
    std::shared_lock<std::shared_mutex> lock(mutex_);

    return patterns_.find(id) != patterns_.end();
}

// ============================================================================
// Batch Operations
// ============================================================================

size_t MemoryBackend::StoreBatch(const std::vector<PatternNode>& nodes) {
    // Exclusive lock for writing
    std::unique_lock<std::shared_mutex> lock(mutex_);

    size_t stored_count = 0;

    for (const auto& node : nodes) {
        PatternID id = node.GetID();

        // Skip if already exists
        if (patterns_.find(id) != patterns_.end()) {
            continue;
        }

        patterns_.emplace(id, PatternNode(node.GetID(), node.GetData(), node.GetType()));
        ++stored_count;
    }

    return stored_count;
}

std::vector<PatternNode> MemoryBackend::RetrieveBatch(const std::vector<PatternID>& ids) {
    // Shared lock for reading
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<PatternNode> results;
    results.reserve(ids.size());

    for (const auto& id : ids) {
        auto it = patterns_.find(id);
        if (it != patterns_.end()) {
            results.emplace_back(it->second.GetID(), it->second.GetData(), it->second.GetType());
        }
    }

    return results;
}

size_t MemoryBackend::DeleteBatch(const std::vector<PatternID>& ids) {
    // Exclusive lock for writing
    std::unique_lock<std::shared_mutex> lock(mutex_);

    size_t deleted_count = 0;

    for (const auto& id : ids) {
        auto it = patterns_.find(id);
        if (it != patterns_.end()) {
            patterns_.erase(it);
            ++deleted_count;
        }
    }

    return deleted_count;
}

// ============================================================================
// Query Operations
// ============================================================================

std::vector<PatternID> MemoryBackend::FindByType(
        PatternType type,
        const QueryOptions& options) {
    // Shared lock for reading
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<PatternID> results;

    for (const auto& [id, node] : patterns_) {
        if (node.GetType() == type) {
            results.push_back(id);

            // Check max results limit
            if (results.size() >= options.max_results) {
                break;
            }
        }
    }

    return results;
}

std::vector<PatternID> MemoryBackend::FindByTimeRange(
        Timestamp start,
        Timestamp end,
        const QueryOptions& options) {
    // Shared lock for reading
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<PatternID> results;

    for (const auto& [id, node] : patterns_) {
        Timestamp creation_time = node.GetCreationTime();

        if (creation_time >= start && creation_time <= end) {
            results.push_back(id);

            // Check max results limit
            if (results.size() >= options.max_results) {
                break;
            }
        }
    }

    return results;
}

std::vector<PatternID> MemoryBackend::FindAll(const QueryOptions& options) {
    // Shared lock for reading
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<PatternID> results;
    results.reserve(std::min(patterns_.size(), options.max_results));

    for (const auto& [id, node] : patterns_) {
        results.push_back(id);

        // Check max results limit
        if (results.size() >= options.max_results) {
            break;
        }
    }

    return results;
}

// ============================================================================
// Statistics and Monitoring
// ============================================================================

size_t MemoryBackend::Count() const {
    // Shared lock for reading
    std::shared_lock<std::shared_mutex> lock(mutex_);

    return patterns_.size();
}

StorageStats MemoryBackend::GetStats() const {
    // Shared lock for reading
    std::shared_lock<std::shared_mutex> lock(mutex_);

    StorageStats stats;
    stats.total_patterns = patterns_.size();

    // Estimate memory usage
    size_t estimated_memory = 0;
    for (const auto& [id, node] : patterns_) {
        estimated_memory += node.EstimateMemoryUsage();
    }
    stats.memory_usage_bytes = estimated_memory;

    // Disk usage (only if mmap is enabled)
    stats.disk_usage_bytes = config_.use_mmap ? mmap_size_ : 0;

    // Calculate average lookup time
    uint64_t total_lookups = total_lookups_.load(std::memory_order_relaxed);
    uint64_t total_time_ns = total_lookup_time_ns_.load(std::memory_order_relaxed);

    if (total_lookups > 0) {
        stats.avg_lookup_time_ms = static_cast<float>(total_time_ns) / total_lookups / 1000000.0f;
    }

    // Calculate cache hit rate
    uint64_t cache_hits = cache_hits_.load(std::memory_order_relaxed);
    if (total_lookups > 0) {
        stats.cache_hit_rate = static_cast<float>(cache_hits) / total_lookups;
    }

    return stats;
}

// ============================================================================
// Maintenance Operations
// ============================================================================

void MemoryBackend::Flush() {
    if (config_.use_mmap && !config_.mmap_path.empty()) {
        SaveToMmap();
    }
}

void MemoryBackend::Compact() {
    // For hash map, compaction is minimal
    // We can rehash to reduce bucket count if load factor is low

    std::unique_lock<std::shared_mutex> lock(mutex_);

    float load_factor = patterns_.load_factor();
    size_t bucket_count = patterns_.bucket_count();

    // If load factor < 0.5, rehash to reduce memory
    if (load_factor < 0.5f && patterns_.size() < bucket_count / 2) {
        std::unordered_map<PatternID, PatternNode> compacted;
        compacted.reserve(patterns_.size());

        for (auto& [id, node] : patterns_) {
            compacted.emplace(id, std::move(node));
        }

        patterns_ = std::move(compacted);
    }
}

void MemoryBackend::Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    patterns_.clear();

    // Reset statistics
    total_lookups_.store(0, std::memory_order_relaxed);
    cache_hits_.store(0, std::memory_order_relaxed);
    total_lookup_time_ns_.store(0, std::memory_order_relaxed);
}

// ============================================================================
// Snapshot and Restore
// ============================================================================

bool MemoryBackend::CreateSnapshot(const std::string& path) {
    try {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Shared lock for reading
        std::shared_lock<std::shared_mutex> lock(mutex_);

        // Write header: version and pattern count
        uint32_t version = 1;
        uint64_t count = patterns_.size();

        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));

        // Write each pattern
        for (const auto& [id, node] : patterns_) {
            node.Serialize(file);
        }

        file.close();
        return true;
    } catch (...) {
        return false;
    }
}

bool MemoryBackend::RestoreSnapshot(const std::string& path) {
    try {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }

        // Read header
        uint32_t version;
        uint64_t count;

        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        file.read(reinterpret_cast<char*>(&count), sizeof(count));

        if (version != 1) {
            return false;  // Unsupported version
        }

        // Exclusive lock for writing
        std::unique_lock<std::shared_mutex> lock(mutex_);

        // Clear existing patterns
        patterns_.clear();
        patterns_.reserve(count);

        // Read each pattern
        for (uint64_t i = 0; i < count; ++i) {
            PatternNode node = PatternNode::Deserialize(file);
            patterns_.emplace(node.GetID(), std::move(node));
        }

        file.close();
        return true;
    } catch (...) {
        return false;
    }
}

// ============================================================================
// Helper Methods
// ============================================================================

void MemoryBackend::UpdateStats(uint64_t lookup_time_ns, bool cache_hit) {
    total_lookups_.fetch_add(1, std::memory_order_relaxed);
    total_lookup_time_ns_.fetch_add(lookup_time_ns, std::memory_order_relaxed);

    if (cache_hit) {
        cache_hits_.fetch_add(1, std::memory_order_relaxed);
    }
}

void MemoryBackend::LoadFromMmap() {
    // Open the memory-mapped file
    mmap_fd_ = open(config_.mmap_path.c_str(), O_RDONLY);
    if (mmap_fd_ == -1) {
        return;  // File doesn't exist yet
    }

    // Get file size
    struct stat sb;
    if (fstat(mmap_fd_, &sb) == -1) {
        close(mmap_fd_);
        mmap_fd_ = -1;
        return;
    }

    mmap_size_ = sb.st_size;

    if (mmap_size_ == 0) {
        close(mmap_fd_);
        mmap_fd_ = -1;
        return;
    }

    // Map the file
    mmap_ptr_ = mmap(nullptr, mmap_size_, PROT_READ, MAP_PRIVATE, mmap_fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
        close(mmap_fd_);
        mmap_fd_ = -1;
        mmap_ptr_ = nullptr;
        return;
    }

    // Load patterns from the mapped memory
    // For simplicity, treat it as a snapshot file
    RestoreSnapshot(config_.mmap_path);

    // Unmap (we've loaded the data)
    UnmapFile();
}

void MemoryBackend::SaveToMmap() {
    // Simply save as a snapshot
    CreateSnapshot(config_.mmap_path);
}

void MemoryBackend::UnmapFile() {
    if (mmap_ptr_ != nullptr && mmap_ptr_ != MAP_FAILED) {
        munmap(mmap_ptr_, mmap_size_);
        mmap_ptr_ = nullptr;
        mmap_size_ = 0;
    }

    if (mmap_fd_ != -1) {
        close(mmap_fd_);
        mmap_fd_ = -1;
    }
}

} // namespace dpan
