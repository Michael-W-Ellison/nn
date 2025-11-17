// File: src/memory/tiered_storage.cpp
//
// Implementation of Tiered Storage

#include "memory/tiered_storage.hpp"
#include <stdexcept>
#include <algorithm>

namespace dpan {

// ============================================================================
// Config
// ============================================================================

bool TieredStorage::Config::IsValid() const {
    if (cache_capacity == 0 || cache_capacity > 10000000) {
        return false;  // Cache capacity should be reasonable
    }

    if (promotion_access_threshold == 0) {
        return false;
    }

    if (prefetch_max_depth > 5) {
        return false;  // Prevent excessive recursion
    }

    if (prefetch_max_patterns == 0 || prefetch_max_patterns > 1000) {
        return false;
    }

    return true;
}

// ============================================================================
// Constructor
// ============================================================================

TieredStorage::TieredStorage(
    TierManager& tier_manager,
    const AssociationMatrix* association_matrix,
    const Config& config)
    : config_(config),
      tier_manager_(tier_manager),
      association_matrix_(association_matrix),
      cache_(config.cache_capacity) {

    if (!config_.IsValid()) {
        throw std::invalid_argument("Invalid TieredStorage configuration");
    }
}

// ============================================================================
// Pattern Access
// ============================================================================

std::optional<PatternNode> TieredStorage::GetPattern(PatternID id) {
    // Check cache first
    auto cached = cache_.Get(id);
    if (cached && *cached) {
        RecordAccess(id);
        // Clone from cached shared_ptr
        return (*cached)->Clone();
    }

    // Load from tiers
    auto pattern = LoadFromTiers(id);
    if (pattern) {
        // Store in cache
        cache_.Put(id, std::make_shared<PatternNode>(std::move(*pattern)));
        RecordAccess(id);
        // Return from cache (which we just populated)
        auto cached_pattern = cache_.Get(id);
        if (cached_pattern && *cached_pattern) {
            return (*cached_pattern)->Clone();
        }
    }

    return std::nullopt;
}

std::optional<PatternNode> TieredStorage::GetPatternWithPromotion(PatternID id) {
    auto pattern = GetPattern(id);

    if (!pattern) {
        return std::nullopt;
    }

    // Check if pattern should be promoted
    if (config_.enable_auto_promotion && ShouldPromote(id)) {
        PromotePattern(id, *pattern);
    }

    // Prefetch if enabled
    if (config_.enable_prefetching && association_matrix_) {
        PrefetchAssociations(id, config_.prefetch_max_depth);
    }

    return pattern;
}

bool TieredStorage::StorePattern(const PatternNode& pattern, MemoryTier tier) {
    bool success = tier_manager_.StorePattern(pattern, tier);

    if (success) {
        // Update cache (clone pattern for cache)
        cache_.Put(pattern.GetID(), std::make_shared<PatternNode>(pattern.Clone()));
    }

    return success;
}

bool TieredStorage::StorePattern(const PatternNode& pattern) {
    return StorePattern(pattern, MemoryTier::ACTIVE);
}

bool TieredStorage::RemovePattern(PatternID id) {
    // Remove from cache
    cache_.Remove(id);

    // Remove from access tracking
    {
        std::unique_lock<std::shared_mutex> lock(access_mutex_);
        access_counts_.erase(id);
    }

    // Remove from tier manager
    return tier_manager_.RemovePattern(id);
}

bool TieredStorage::HasPattern(PatternID id) {
    // Check cache first (fast path)
    auto cached = cache_.Get(id);
    if (cached && *cached) {
        return true;
    }

    // Check tier manager
    return tier_manager_.GetPatternTier(id).has_value();
}

std::optional<MemoryTier> TieredStorage::GetPatternTier(PatternID id) {
    return tier_manager_.GetPatternTier(id);
}

// ============================================================================
// Prefetching
// ============================================================================

void TieredStorage::PrefetchAssociations(PatternID id, size_t max_depth) {
    if (!association_matrix_ || max_depth == 0) {
        return;
    }

    prefetch_requests_.fetch_add(1);

    std::unordered_set<PatternID> visited;
    visited.insert(id);  // Don't prefetch the pattern itself

    PrefetchAssociationsRecursive(id, 0, max_depth, visited);
}

void TieredStorage::PrefetchPatterns(const std::vector<PatternID>& ids) {
    size_t patterns_loaded = 0;

    for (const auto& id : ids) {
        if (patterns_loaded >= config_.prefetch_max_patterns) {
            break;
        }

        // Skip if already in cache
        auto cached = cache_.Get(id);
        if (cached && *cached) {
            continue;
        }

        // Load from tiers
        auto pattern = LoadFromTiers(id);
        if (pattern) {
            cache_.Put(id, std::make_shared<PatternNode>(std::move(*pattern)));
            patterns_loaded++;
        }
    }

    prefetch_patterns_loaded_.fetch_add(patterns_loaded);
}

void TieredStorage::PrefetchAssociationsRecursive(
    PatternID id,
    size_t current_depth,
    size_t max_depth,
    std::unordered_set<PatternID>& visited) {

    if (current_depth >= max_depth) {
        return;
    }

    // Get outgoing associations
    auto associations = association_matrix_->GetOutgoingAssociations(id);

    std::vector<PatternID> to_prefetch;
    to_prefetch.reserve(std::min(associations.size(), config_.prefetch_max_patterns));

    for (const auto* edge : associations) {
        if (!edge) continue;

        PatternID target = edge->GetTarget();

        // Skip if already visited or in cache
        auto cached = cache_.Get(target);
        if (visited.count(target) > 0 || (cached && *cached)) {
            continue;
        }

        visited.insert(target);
        to_prefetch.push_back(target);

        if (to_prefetch.size() >= config_.prefetch_max_patterns) {
            break;
        }
    }

    // Prefetch patterns
    PrefetchPatterns(to_prefetch);

    // Recursively prefetch from loaded patterns
    if (current_depth + 1 < max_depth) {
        for (const auto& target_id : to_prefetch) {
            PrefetchAssociationsRecursive(target_id, current_depth + 1, max_depth, visited);
        }
    }
}

// ========================================================================
// Cache Management
// ========================================================================

void TieredStorage::ClearCache() {
    cache_.Clear();
}

TieredStorage::CacheStats TieredStorage::GetCacheStats() const {
    CacheStats stats;
    stats.hits = cache_.Hits();
    stats.misses = cache_.Misses();
    stats.evictions = cache_.Evictions();
    stats.promotions = promotions_.load();
    stats.prefetch_requests = prefetch_requests_.load();
    stats.prefetch_patterns_loaded = prefetch_patterns_loaded_.load();
    return stats;
}

size_t TieredStorage::GetCacheSize() const {
    return cache_.Size();
}

size_t TieredStorage::GetCacheCapacity() const {
    return cache_.Capacity();
}

void TieredStorage::SetCacheCapacity(size_t capacity) {
    // Note: LRUCache capacity is set at construction and cannot be changed.
    // We clear the cache and update the config, but the actual capacity
    // remains the same until a new TieredStorage is created.
    cache_.Clear();
    config_.cache_capacity = capacity;

    // TODO: To properly change capacity, we would need to reconstruct
    // the entire TieredStorage object or make LRUCache support dynamic resizing
}

// ============================================================================
// Configuration
// ============================================================================

void TieredStorage::SetConfig(const Config& config) {
    if (!config.IsValid()) {
        throw std::invalid_argument("Invalid TieredStorage configuration");
    }

    // Update cache capacity if changed
    if (config.cache_capacity != config_.cache_capacity) {
        SetCacheCapacity(config.cache_capacity);
    }

    config_ = config;
}

// ============================================================================
// Helper Methods
// ============================================================================

std::optional<PatternNode> TieredStorage::LoadFromTiers(PatternID id) {
    // Try loading from tier manager (it knows which tier)
    return tier_manager_.LoadPattern(id);
}

void TieredStorage::RecordAccess(PatternID id) {
    std::unique_lock<std::shared_mutex> lock(access_mutex_);
    access_counts_[id]++;
}

bool TieredStorage::ShouldPromote(PatternID id) {
    std::shared_lock<std::shared_mutex> lock(access_mutex_);

    auto it = access_counts_.find(id);
    if (it == access_counts_.end()) {
        return false;
    }

    return it->second >= config_.promotion_access_threshold;
}

void TieredStorage::PromotePattern(PatternID id, const PatternNode& pattern) {
    auto current_tier = tier_manager_.GetPatternTier(id);
    if (!current_tier) {
        return;
    }

    // Determine target tier (one level up)
    MemoryTier target_tier;
    switch (*current_tier) {
        case MemoryTier::WARM:
            target_tier = MemoryTier::ACTIVE;
            break;
        case MemoryTier::COLD:
            target_tier = MemoryTier::WARM;
            break;
        case MemoryTier::ARCHIVE:
            target_tier = MemoryTier::COLD;
            break;
        default:
            return;  // Already in active tier
    }

    // Promote using tier manager
    if (tier_manager_.PromotePattern(id, target_tier)) {
        promotions_.fetch_add(1);

        // Reset access count after promotion
        {
            std::unique_lock<std::shared_mutex> lock(access_mutex_);
            access_counts_[id] = 0;
        }
    }
}

} // namespace dpan
