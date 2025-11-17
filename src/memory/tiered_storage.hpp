// File: src/memory/tiered_storage.hpp
//
// Tiered Storage - Transparent Access Layer
//
// Provides unified interface for accessing patterns across all memory tiers
// with automatic tier lookup, LRU caching, and intelligent prefetching.
//
// Key Features:
//   - Automatic tier lookup (Active → Warm → Cold → Archive)
//   - LRU cache for recently accessed patterns
//   - Transparent promotion on access
//   - Prefetching based on association graphs
//   - Cache statistics and monitoring

#pragma once

#include "memory/tier_manager.hpp"
#include "storage/lru_cache.hpp"
#include "association/association_matrix.hpp"
#include "core/pattern_node.hpp"
#include <memory>
#include <vector>
#include <atomic>
#include <unordered_set>

namespace dpan {

/// Transparent access layer for tiered pattern storage
class TieredStorage {
public:
    /// Configuration for tiered storage
    struct Config {
        /// LRU cache capacity (number of patterns)
        size_t cache_capacity{10000};

        /// Enable automatic promotion on access
        bool enable_auto_promotion{true};

        /// Promotion threshold (access count before promotion)
        size_t promotion_access_threshold{3};

        /// Enable prefetching of associated patterns
        bool enable_prefetching{true};

        /// Maximum depth for prefetching (0 = disabled)
        size_t prefetch_max_depth{1};

        /// Maximum patterns to prefetch per operation
        size_t prefetch_max_patterns{10};

        /// Validate configuration
        bool IsValid() const;
    };

    /// Cache statistics
    struct CacheStats {
        size_t hits{0};
        size_t misses{0};
        size_t evictions{0};
        size_t promotions{0};
        size_t prefetch_requests{0};
        size_t prefetch_patterns_loaded{0};

        /// Calculate hit rate [0,1]
        float GetHitRate() const {
            size_t total = hits + misses;
            return total > 0 ? static_cast<float>(hits) / total : 0.0f;
        }
    };

    /// Construct with tier manager and config
    ///
    /// @param tier_manager Tier manager for accessing patterns across tiers
    /// @param association_matrix Association matrix for prefetching (optional)
    /// @param config Configuration
    TieredStorage(
        TierManager& tier_manager,
        const AssociationMatrix* association_matrix,
        const Config& config
    );

    // ========================================================================
    // Pattern Access
    // ========================================================================

    /// Get pattern from any tier (transparent lookup)
    ///
    /// Checks cache first, then searches tiers: Active → Warm → Cold → Archive
    ///
    /// @param id Pattern ID
    /// @return Pattern if found, nullopt otherwise
    std::optional<PatternNode> GetPattern(PatternID id);

    /// Get pattern with automatic promotion
    ///
    /// Like GetPattern, but promotes frequently accessed patterns to higher tiers
    ///
    /// @param id Pattern ID
    /// @return Pattern if found, nullopt otherwise
    std::optional<PatternNode> GetPatternWithPromotion(PatternID id);

    /// Store pattern in specified tier
    ///
    /// @param pattern Pattern to store
    /// @param tier Target tier
    /// @return true if stored successfully
    bool StorePattern(const PatternNode& pattern, MemoryTier tier);

    /// Store pattern in active tier (convenience)
    ///
    /// @param pattern Pattern to store
    /// @return true if stored successfully
    bool StorePattern(const PatternNode& pattern);

    /// Remove pattern from all tiers and cache
    ///
    /// @param id Pattern ID
    /// @return true if removed
    bool RemovePattern(PatternID id);

    /// Check if pattern exists in any tier
    ///
    /// @param id Pattern ID
    /// @return true if pattern exists
    bool HasPattern(PatternID id);

    /// Get tier containing pattern
    ///
    /// @param id Pattern ID
    /// @return Tier if found, nullopt otherwise
    std::optional<MemoryTier> GetPatternTier(PatternID id);

    // ========================================================================
    // Prefetching
    // ========================================================================

    /// Prefetch associated patterns
    ///
    /// Loads patterns associated with the given pattern into cache
    ///
    /// @param id Pattern ID
    /// @param max_depth Maximum traversal depth (0 = only direct associations)
    void PrefetchAssociations(PatternID id, size_t max_depth = 1);

    /// Prefetch patterns by IDs
    ///
    /// @param ids Pattern IDs to prefetch
    void PrefetchPatterns(const std::vector<PatternID>& ids);

    // ========================================================================
    // Cache Management
    // ========================================================================

    /// Clear cache (does not affect tier storage)
    void ClearCache();

    /// Get cache statistics
    CacheStats GetCacheStats() const;

    /// Get cache size
    size_t GetCacheSize() const;

    /// Get cache capacity
    size_t GetCacheCapacity() const;

    /// Set cache capacity (clears cache)
    void SetCacheCapacity(size_t capacity);

    // ========================================================================
    // Configuration
    // ========================================================================

    /// Get current configuration
    const Config& GetConfig() const { return config_; }

    /// Update configuration
    void SetConfig(const Config& config);

private:
    Config config_;

    // Dependencies
    TierManager& tier_manager_;
    const AssociationMatrix* association_matrix_;

    // LRU cache for recently accessed patterns (using shared_ptr for non-copyable PatternNode)
    LRUCache<PatternID, std::shared_ptr<PatternNode>> cache_;

    // Access tracking for promotion decisions
    std::unordered_map<PatternID, size_t> access_counts_;
    mutable std::shared_mutex access_mutex_;

    // Statistics
    std::atomic<size_t> promotions_{0};
    std::atomic<size_t> prefetch_requests_{0};
    std::atomic<size_t> prefetch_patterns_loaded_{0};

    // Helper methods
    std::optional<PatternNode> LoadFromTiers(PatternID id);
    void RecordAccess(PatternID id);
    bool ShouldPromote(PatternID id);
    void PromotePattern(PatternID id, const PatternNode& pattern);
    void PrefetchAssociationsRecursive(
        PatternID id,
        size_t current_depth,
        size_t max_depth,
        std::unordered_set<PatternID>& visited
    );
};

} // namespace dpan
