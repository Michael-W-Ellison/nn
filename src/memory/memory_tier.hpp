// File: src/memory/memory_tier.hpp
//
// Memory Tier Interface for Multi-Tier Storage System
//
// This module defines the abstract interface for memory tiers in the DPAN
// memory hierarchy. Tiers provide transparent storage for patterns and
// associations with different performance characteristics.
//
// Tier Structure:
//   1. Active (RAM): Hot patterns, <100ns access
//   2. Warm (SSD): Recently used, <10µs access
//   3. Cold (HDD): Rarely accessed, <1ms access
//   4. Archive (Compressed): Long-term storage, <10ms access

#pragma once

#include "core/pattern_node.hpp"
#include "association/association_edge.hpp"
#include "core/types.hpp"
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <functional>

namespace dpan {

/// Memory tier levels
enum class MemoryTier {
    ACTIVE = 0,   ///< RAM-based, fastest (<100ns)
    WARM = 1,     ///< SSD-based, fast (<10µs)
    COLD = 2,     ///< HDD-based, slow (<1ms)
    ARCHIVE = 3   ///< Compressed disk, slowest (<10ms)
};

/// Convert tier to string
std::string TierToString(MemoryTier tier);

/// Convert string to tier
std::optional<MemoryTier> StringToTier(const std::string& str);

/// Abstract interface for a memory tier
class IMemoryTier {
public:
    virtual ~IMemoryTier() = default;

    // ========================================================================
    // Pattern Operations
    // ========================================================================

    /// Store a pattern in this tier
    ///
    /// @param pattern Pattern to store
    /// @return true if successfully stored
    virtual bool StorePattern(const PatternNode& pattern) = 0;

    /// Load a pattern from this tier
    ///
    /// @param id Pattern ID to load
    /// @return Pattern if found, nullopt otherwise
    virtual std::optional<PatternNode> LoadPattern(PatternID id) = 0;

    /// Remove a pattern from this tier
    ///
    /// @param id Pattern ID to remove
    /// @return true if pattern was removed
    virtual bool RemovePattern(PatternID id) = 0;

    /// Check if pattern exists in this tier
    ///
    /// @param id Pattern ID to check
    /// @return true if pattern exists
    virtual bool HasPattern(PatternID id) const = 0;

    // ========================================================================
    // Association Operations
    // ========================================================================

    /// Store an association in this tier
    ///
    /// @param edge Association to store
    /// @return true if successfully stored
    virtual bool StoreAssociation(const AssociationEdge& edge) = 0;

    /// Load an association from this tier
    ///
    /// @param source Source pattern ID
    /// @param target Target pattern ID
    /// @return Association if found, nullopt otherwise
    virtual std::optional<AssociationEdge> LoadAssociation(
        PatternID source, PatternID target) = 0;

    /// Remove an association from this tier
    ///
    /// @param source Source pattern ID
    /// @param target Target pattern ID
    /// @return true if association was removed
    virtual bool RemoveAssociation(PatternID source, PatternID target) = 0;

    /// Check if association exists in this tier
    ///
    /// @param source Source pattern ID
    /// @param target Target pattern ID
    /// @return true if association exists
    virtual bool HasAssociation(PatternID source, PatternID target) const = 0;

    // ========================================================================
    // Batch Operations (more efficient than individual operations)
    // ========================================================================

    /// Store multiple patterns
    ///
    /// @param patterns Patterns to store
    /// @return Number of patterns successfully stored
    virtual size_t StorePatternsBatch(const std::vector<PatternNode>& patterns) = 0;

    /// Load multiple patterns
    ///
    /// @param ids Pattern IDs to load
    /// @return Patterns that were found (may be fewer than requested)
    virtual std::vector<PatternNode> LoadPatternsBatch(
        const std::vector<PatternID>& ids) = 0;

    /// Remove multiple patterns
    ///
    /// @param ids Pattern IDs to remove
    /// @return Number of patterns successfully removed
    virtual size_t RemovePatternsBatch(const std::vector<PatternID>& ids) = 0;

    /// Store multiple associations
    ///
    /// @param edges Associations to store
    /// @return Number of associations successfully stored
    virtual size_t StoreAssociationsBatch(
        const std::vector<AssociationEdge>& edges) = 0;

    // ========================================================================
    // Statistics and Information
    // ========================================================================

    /// Get number of patterns in this tier
    virtual size_t GetPatternCount() const = 0;

    /// Get number of associations in this tier
    virtual size_t GetAssociationCount() const = 0;

    /// Estimate memory/disk usage in bytes
    virtual size_t EstimateMemoryUsage() const = 0;

    /// Get tier level
    virtual MemoryTier GetTierLevel() const = 0;

    /// Get tier name
    virtual std::string GetTierName() const = 0;

    // ========================================================================
    // Maintenance Operations
    // ========================================================================

    /// Compact storage (reduce fragmentation, optimize layout)
    virtual void Compact() = 0;

    /// Clear all data from this tier
    virtual void Clear() = 0;

    /// Flush any pending writes to disk (no-op for in-memory tiers)
    virtual void Flush() = 0;
};

// ============================================================================
// Factory Functions
// ============================================================================

/// Create an Active tier (RAM-based, in-memory)
///
/// @param config_path Optional configuration (unused for now)
/// @return Unique pointer to active tier instance
std::unique_ptr<IMemoryTier> CreateActiveTier(const std::string& config_path = "");

/// Create a Warm tier (SSD-based, file-backed)
///
/// @param storage_path Path to storage directory
/// @return Unique pointer to warm tier instance
std::unique_ptr<IMemoryTier> CreateWarmTier(const std::string& storage_path);

/// Create a Cold tier (HDD-based, file-backed)
///
/// @param storage_path Path to storage directory
/// @return Unique pointer to cold tier instance
std::unique_ptr<IMemoryTier> CreateColdTier(const std::string& storage_path);

/// Create an Archive tier (Compressed disk storage)
///
/// @param storage_path Path to storage directory
/// @return Unique pointer to archive tier instance
std::unique_ptr<IMemoryTier> CreateArchiveTier(const std::string& storage_path);

// ============================================================================
// Utilities
// ============================================================================

/// Hash function for PatternID pairs (for unordered_map)
#ifndef DPAN_PATTERN_PAIR_HASH_DEFINED
#define DPAN_PATTERN_PAIR_HASH_DEFINED
struct PatternPairHash {
    size_t operator()(const std::pair<PatternID, PatternID>& pair) const {
        // Combine hashes using boost-style hash combine
        size_t seed = std::hash<PatternID>{}(pair.first);
        seed ^= std::hash<PatternID>{}(pair.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};
#endif // DPAN_PATTERN_PAIR_HASH_DEFINED

} // namespace dpan
