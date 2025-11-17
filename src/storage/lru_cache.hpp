// File: src/storage/lru_cache.hpp
#pragma once

#include <unordered_map>
#include <list>
#include <optional>
#include <mutex>
#include <atomic>

namespace dpan {

/// LRU (Least Recently Used) Cache implementation
///
/// Template-based cache with O(1) get and put operations.
/// Thread-safe with mutex protection.
/// Automatically evicts least recently used items when capacity is reached.
///
/// @tparam Key Key type (must be hashable)
/// @tparam Value Value type (must be copyable or movable)
template<typename Key, typename Value>
class LRUCache {
public:
    /// Construct LRU cache with specified capacity
    /// @param capacity Maximum number of items to cache
    explicit LRUCache(size_t capacity)
        : capacity_(capacity) {
        if (capacity_ == 0) {
            capacity_ = 1;  // Minimum capacity
        }
    }

    /// Get value from cache
    /// If found, moves item to front (most recently used)
    /// @param key Key to lookup
    /// @return Value if found, std::nullopt otherwise
    std::optional<Value> Get(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto map_it = map_.find(key);
        if (map_it == map_.end()) {
            misses_.fetch_add(1, std::memory_order_relaxed);
            return std::nullopt;
        }

        hits_.fetch_add(1, std::memory_order_relaxed);

        // Move to front (most recently used)
        items_.splice(items_.begin(), items_, map_it->second);

        return map_it->second->second;
    }

    /// Put value into cache
    /// If key exists, updates value and moves to front
    /// If cache is full, evicts least recently used item
    /// @param key Key to store
    /// @param value Value to store
    void Put(const Key& key, const Value& value) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto map_it = map_.find(key);

        // Key already exists, update and move to front
        if (map_it != map_.end()) {
            map_it->second->second = value;
            items_.splice(items_.begin(), items_, map_it->second);
            return;
        }

        // Cache is full, evict LRU item
        if (items_.size() >= capacity_) {
            auto& lru = items_.back();
            map_.erase(lru.first);
            items_.pop_back();
            evictions_.fetch_add(1, std::memory_order_relaxed);
        }

        // Insert new item at front
        items_.emplace_front(key, value);
        map_[key] = items_.begin();
    }

    /// Remove item from cache
    /// @param key Key to remove
    /// @return true if removed, false if not found
    bool Remove(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto map_it = map_.find(key);
        if (map_it == map_.end()) {
            return false;
        }

        items_.erase(map_it->second);
        map_.erase(map_it);
        return true;
    }

    /// Clear all items from cache
    void Clear() {
        std::lock_guard<std::mutex> lock(mutex_);

        items_.clear();
        map_.clear();

        // Reset statistics
        hits_.store(0, std::memory_order_relaxed);
        misses_.store(0, std::memory_order_relaxed);
        evictions_.store(0, std::memory_order_relaxed);
    }

    /// Get current number of items in cache
    /// @return Number of cached items
    size_t Size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return items_.size();
    }

    /// Get cache capacity
    /// @return Maximum capacity
    size_t Capacity() const {
        return capacity_;
    }

    /// Check if cache contains key
    /// @param key Key to check
    /// @return true if key exists
    bool Contains(const Key& key) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return map_.find(key) != map_.end();
    }

    /// Get cache hit rate
    /// @return Hit rate [0.0, 1.0]
    float HitRate() const {
        uint64_t total_hits = hits_.load(std::memory_order_relaxed);
        uint64_t total_misses = misses_.load(std::memory_order_relaxed);
        uint64_t total = total_hits + total_misses;

        if (total == 0) {
            return 0.0f;
        }

        return static_cast<float>(total_hits) / static_cast<float>(total);
    }

    /// Get total number of cache hits
    /// @return Hit count
    uint64_t Hits() const {
        return hits_.load(std::memory_order_relaxed);
    }

    /// Get total number of cache misses
    /// @return Miss count
    uint64_t Misses() const {
        return misses_.load(std::memory_order_relaxed);
    }

    /// Get total number of evictions
    /// @return Eviction count
    uint64_t Evictions() const {
        return evictions_.load(std::memory_order_relaxed);
    }

    /// Statistics structure
    struct Stats {
        size_t size{0};
        size_t capacity{0};
        uint64_t hits{0};
        uint64_t misses{0};
        uint64_t evictions{0};
        float hit_rate{0.0f};
        float utilization{0.0f};  // size / capacity
    };

    /// Get comprehensive statistics
    /// @return Stats structure
    Stats GetStats() const {
        std::lock_guard<std::mutex> lock(mutex_);

        Stats stats;
        stats.size = items_.size();
        stats.capacity = capacity_;
        stats.hits = hits_.load(std::memory_order_relaxed);
        stats.misses = misses_.load(std::memory_order_relaxed);
        stats.evictions = evictions_.load(std::memory_order_relaxed);
        stats.hit_rate = HitRate();

        if (capacity_ > 0) {
            stats.utilization = static_cast<float>(stats.size) / static_cast<float>(capacity_);
        }

        return stats;
    }

private:
    /// Maximum capacity
    size_t capacity_;

    /// Doubly-linked list of (key, value) pairs
    /// Front = most recently used, Back = least recently used
    std::list<std::pair<Key, Value>> items_;

    /// Hash map from key to iterator in list (for O(1) lookup)
    std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator> map_;

    /// Mutex for thread safety
    mutable std::mutex mutex_;

    /// Statistics (atomics for lock-free access)
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> evictions_{0};
};

} // namespace dpan
