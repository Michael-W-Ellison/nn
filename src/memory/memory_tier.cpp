// File: src/memory/memory_tier.cpp
//
// Implementation of Memory Tier utilities and Active Tier

#include "memory/memory_tier.hpp"
#include <unordered_map>
#include <shared_mutex>
#include <stdexcept>

namespace dpan {

// ============================================================================
// Utility Functions
// ============================================================================

std::string TierToString(MemoryTier tier) {
    switch (tier) {
        case MemoryTier::ACTIVE:
            return "Active";
        case MemoryTier::WARM:
            return "Warm";
        case MemoryTier::COLD:
            return "Cold";
        case MemoryTier::ARCHIVE:
            return "Archive";
        default:
            return "Unknown";
    }
}

std::optional<MemoryTier> StringToTier(const std::string& str) {
    if (str == "Active" || str == "ACTIVE") {
        return MemoryTier::ACTIVE;
    } else if (str == "Warm" || str == "WARM") {
        return MemoryTier::WARM;
    } else if (str == "Cold" || str == "COLD") {
        return MemoryTier::COLD;
    } else if (str == "Archive" || str == "ARCHIVE") {
        return MemoryTier::ARCHIVE;
    }
    return std::nullopt;
}

// ============================================================================
// Active Tier Implementation (RAM-based, in-memory)
// ============================================================================

class ActiveTier : public IMemoryTier {
public:
    ActiveTier() = default;

    // Pattern operations
    bool StorePattern(const PatternNode& pattern) override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto id = pattern.GetID();
        // Erase existing pattern if it exists
        patterns_.erase(id);
        // Insert cloned pattern
        patterns_.emplace(id, pattern.Clone());
        return true;
    }

    std::optional<PatternNode> LoadPattern(PatternID id) override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = patterns_.find(id);
        if (it != patterns_.end()) {
            return it->second.Clone();
        }
        return std::nullopt;
    }

    bool RemovePattern(PatternID id) override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return patterns_.erase(id) > 0;
    }

    bool HasPattern(PatternID id) const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return patterns_.find(id) != patterns_.end();
    }

    // Association operations
    bool StoreAssociation(const AssociationEdge& edge) override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto key = std::make_pair(edge.GetSource(), edge.GetTarget());

        // Create new edge since AssociationEdge is not copyable
        AssociationEdge new_edge(
            edge.GetSource(),
            edge.GetTarget(),
            edge.GetType(),
            edge.GetStrength()
        );

        associations_[key] = std::move(new_edge);
        return true;
    }

    std::optional<AssociationEdge> LoadAssociation(
        PatternID source, PatternID target) override {

        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto key = std::make_pair(source, target);
        auto it = associations_.find(key);

        if (it != associations_.end()) {
            // Return a copy (reconstruct since AssociationEdge is not copyable)
            return AssociationEdge(
                it->second.GetSource(),
                it->second.GetTarget(),
                it->second.GetType(),
                it->second.GetStrength()
            );
        }
        return std::nullopt;
    }

    bool RemoveAssociation(PatternID source, PatternID target) override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        auto key = std::make_pair(source, target);
        return associations_.erase(key) > 0;
    }

    bool HasAssociation(PatternID source, PatternID target) const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto key = std::make_pair(source, target);
        return associations_.find(key) != associations_.end();
    }

    // Batch operations
    size_t StorePatternsBatch(const std::vector<PatternNode>& patterns) override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        size_t count = 0;
        for (const auto& pattern : patterns) {
            auto id = pattern.GetID();
            // Erase existing pattern if it exists
            patterns_.erase(id);
            // Insert cloned pattern
            patterns_.emplace(id, pattern.Clone());
            count++;
        }
        return count;
    }

    std::vector<PatternNode> LoadPatternsBatch(
        const std::vector<PatternID>& ids) override {

        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<PatternNode> result;
        result.reserve(ids.size());

        for (const auto& id : ids) {
            auto it = patterns_.find(id);
            if (it != patterns_.end()) {
                result.push_back(it->second.Clone());
            }
        }

        return result;
    }

    size_t RemovePatternsBatch(const std::vector<PatternID>& ids) override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        size_t count = 0;
        for (const auto& id : ids) {
            count += patterns_.erase(id);
        }
        return count;
    }

    size_t StoreAssociationsBatch(
        const std::vector<AssociationEdge>& edges) override {

        std::unique_lock<std::shared_mutex> lock(mutex_);
        size_t count = 0;

        for (const auto& edge : edges) {
            auto key = std::make_pair(edge.GetSource(), edge.GetTarget());

            // Create new edge
            AssociationEdge new_edge(
                edge.GetSource(),
                edge.GetTarget(),
                edge.GetType(),
                edge.GetStrength()
            );

            associations_[key] = std::move(new_edge);
            count++;
        }

        return count;
    }

    // Statistics
    size_t GetPatternCount() const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return patterns_.size();
    }

    size_t GetAssociationCount() const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return associations_.size();
    }

    size_t EstimateMemoryUsage() const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        // Rough estimate: pattern + association storage
        size_t pattern_size = patterns_.size() * sizeof(PatternNode);
        size_t assoc_size = associations_.size() * sizeof(AssociationEdge);
        return pattern_size + assoc_size;
    }

    // Tier information
    MemoryTier GetTierLevel() const override {
        return MemoryTier::ACTIVE;
    }

    std::string GetTierName() const override {
        return "Active";
    }

    // Maintenance
    void Compact() override {
        // No-op for in-memory tier
    }

    void Clear() override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        patterns_.clear();
        associations_.clear();
    }

    void Flush() override {
        // No-op for in-memory tier
    }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<PatternID, PatternNode> patterns_;
    std::unordered_map<std::pair<PatternID, PatternID>, AssociationEdge, PatternPairHash> associations_;
};

// ============================================================================
// Factory Functions
// ============================================================================

std::unique_ptr<IMemoryTier> CreateActiveTier(const std::string& config_path) {
    (void)config_path;  // Unused for now
    return std::make_unique<ActiveTier>();
}

} // namespace dpan
