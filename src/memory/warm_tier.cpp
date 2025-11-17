// File: src/memory/warm_tier.cpp
//
// Warm Tier Implementation (File-based storage for SSD)
//
// Uses simple file-based storage with one file per pattern/association.
// Optimized for SSD access patterns (<10µs latency target).

#include "memory/memory_tier.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace dpan {
namespace fs = std::filesystem;

class WarmTier : public IMemoryTier {
public:
    explicit WarmTier(const std::string& storage_path)
        : storage_path_(storage_path) {

        // Create storage directories
        fs::create_directories(storage_path_);
        patterns_dir_ = storage_path_ / "patterns";
        associations_dir_ = storage_path_ / "associations";
        fs::create_directories(patterns_dir_);
        fs::create_directories(associations_dir_);

        // Build index of existing files
        RebuildIndex();
    }

    // Pattern operations
    bool StorePattern(const PatternNode& pattern) override {
        try {
            fs::path filepath = patterns_dir_ / (pattern.GetID().ToString() + ".pat");
            std::ofstream file(filepath, std::ios::binary);
            if (!file) {
                return false;
            }

            pattern.Serialize(file);

            // Update index
            {
                std::unique_lock<std::shared_mutex> lock(mutex_);
                pattern_index_.insert(pattern.GetID());
            }

            return true;
        } catch (...) {
            return false;
        }
    }

    std::optional<PatternNode> LoadPattern(PatternID id) override {
        try {
            fs::path filepath = patterns_dir_ / (id.ToString() + ".pat");

            if (!fs::exists(filepath)) {
                return std::nullopt;
            }

            std::ifstream file(filepath, std::ios::binary);
            if (!file) {
                return std::nullopt;
            }

            return PatternNode::Deserialize(file);
        } catch (...) {
            return std::nullopt;
        }
    }

    bool RemovePattern(PatternID id) override {
        try {
            fs::path filepath = patterns_dir_ / (id.ToString() + ".pat");

            if (fs::exists(filepath)) {
                fs::remove(filepath);

                // Update index
                {
                    std::unique_lock<std::shared_mutex> lock(mutex_);
                    pattern_index_.erase(id);
                }

                return true;
            }

            return false;
        } catch (...) {
            return false;
        }
    }

    bool HasPattern(PatternID id) const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return pattern_index_.find(id) != pattern_index_.end();
    }

    // Association operations
    bool StoreAssociation(const AssociationEdge& edge) override {
        try {
            std::string filename = edge.GetSource().ToString() + "_" +
                                 edge.GetTarget().ToString() + ".assoc";
            fs::path filepath = associations_dir_ / filename;

            std::ofstream file(filepath, std::ios::binary);
            if (!file) {
                return false;
            }

            edge.Serialize(file);

            // Update index
            {
                std::unique_lock<std::shared_mutex> lock(mutex_);
                auto key = std::make_pair(edge.GetSource(), edge.GetTarget());
                association_index_.insert(key);
            }

            return true;
        } catch (...) {
            return false;
        }
    }

    std::optional<AssociationEdge> LoadAssociation(
        PatternID source, PatternID target) override {

        try {
            std::string filename = source.ToString() + "_" +
                                 target.ToString() + ".assoc";
            fs::path filepath = associations_dir_ / filename;

            if (!fs::exists(filepath)) {
                return std::nullopt;
            }

            std::ifstream file(filepath, std::ios::binary);
            if (!file) {
                return std::nullopt;
            }

            auto edge_ptr = AssociationEdge::Deserialize(file);
            if (edge_ptr) {
                return std::move(*edge_ptr);
            }
            return std::nullopt;
        } catch (...) {
            return std::nullopt;
        }
    }

    bool RemoveAssociation(PatternID source, PatternID target) override {
        try {
            std::string filename = source.ToString() + "_" +
                                 target.ToString() + ".assoc";
            fs::path filepath = associations_dir_ / filename;

            if (fs::exists(filepath)) {
                fs::remove(filepath);

                // Update index
                {
                    std::unique_lock<std::shared_mutex> lock(mutex_);
                    auto key = std::make_pair(source, target);
                    association_index_.erase(key);
                }

                return true;
            }

            return false;
        } catch (...) {
            return false;
        }
    }

    bool HasAssociation(PatternID source, PatternID target) const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto key = std::make_pair(source, target);
        return association_index_.find(key) != association_index_.end();
    }

    // Batch operations
    size_t StorePatternsBatch(const std::vector<PatternNode>& patterns) override {
        size_t count = 0;
        for (const auto& pattern : patterns) {
            if (StorePattern(pattern)) {
                count++;
            }
        }
        return count;
    }

    std::vector<PatternNode> LoadPatternsBatch(
        const std::vector<PatternID>& ids) override {

        std::vector<PatternNode> result;
        result.reserve(ids.size());

        for (const auto& id : ids) {
            auto pattern = LoadPattern(id);
            if (pattern) {
                result.push_back(std::move(*pattern));
            }
        }

        return result;
    }

    size_t RemovePatternsBatch(const std::vector<PatternID>& ids) override {
        size_t count = 0;
        for (const auto& id : ids) {
            if (RemovePattern(id)) {
                count++;
            }
        }
        return count;
    }

    size_t StoreAssociationsBatch(
        const std::vector<AssociationEdge>& edges) override {

        size_t count = 0;
        for (const auto& edge : edges) {
            if (StoreAssociation(edge)) {
                count++;
            }
        }
        return count;
    }

    // Statistics
    size_t GetPatternCount() const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return pattern_index_.size();
    }

    size_t GetAssociationCount() const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return association_index_.size();
    }

    size_t EstimateMemoryUsage() const override {
        size_t total = 0;

        try {
            for (const auto& entry : fs::directory_iterator(patterns_dir_)) {
                if (entry.is_regular_file()) {
                    total += entry.file_size();
                }
            }

            for (const auto& entry : fs::directory_iterator(associations_dir_)) {
                if (entry.is_regular_file()) {
                    total += entry.file_size();
                }
            }
        } catch (...) {
            // Return best estimate
        }

        return total;
    }

    // Tier information
    MemoryTier GetTierLevel() const override {
        return MemoryTier::WARM;
    }

    std::string GetTierName() const override {
        return "Warm";
    }

    // Maintenance
    void Compact() override {
        // Could implement defragmentation here
        // For now, no-op
    }

    void Clear() override {
        try {
            // Remove all pattern files
            for (const auto& entry : fs::directory_iterator(patterns_dir_)) {
                if (entry.is_regular_file()) {
                    fs::remove(entry.path());
                }
            }

            // Remove all association files
            for (const auto& entry : fs::directory_iterator(associations_dir_)) {
                if (entry.is_regular_file()) {
                    fs::remove(entry.path());
                }
            }

            // Clear indices
            {
                std::unique_lock<std::shared_mutex> lock(mutex_);
                pattern_index_.clear();
                association_index_.clear();
            }
        } catch (...) {
            // Best effort
        }
    }

    void Flush() override {
        // File I/O is synchronous, so no buffering to flush
    }

private:
    fs::path storage_path_;
    fs::path patterns_dir_;
    fs::path associations_dir_;

    // In-memory indices for fast lookups
    std::unordered_set<PatternID> pattern_index_;
    std::unordered_set<std::pair<PatternID, PatternID>, PatternPairHash> association_index_;
    mutable std::shared_mutex mutex_;

    void RebuildIndex() {
        // TODO: Implement index rebuilding once PatternID::FromString() is available
        // For now, indices are built incrementally as patterns are stored
        try {
            // Scan pattern files
            // for (const auto& entry : fs::directory_iterator(patterns_dir_)) {
            //     if (entry.is_regular_file() && entry.path().extension() == ".pat") {
            //         std::string filename = entry.path().stem().string();
            //         auto id = PatternID::FromString(filename);
            //         if (id) {
            //             pattern_index_.insert(*id);
            //         }
            //     }
            // }

            // Scan association files
            // for (const auto& entry : fs::directory_iterator(associations_dir_)) {
            //     if (entry.is_regular_file() && entry.path().extension() == ".assoc") {
            //         std::string filename = entry.path().stem().string();
            //         size_t underscore_pos = filename.find('_');
            //         if (underscore_pos != std::string::npos) {
            //             auto source = PatternID::FromString(filename.substr(0, underscore_pos));
            //             auto target = PatternID::FromString(filename.substr(underscore_pos + 1));
            //             if (source && target) {
            //                 association_index_.insert(std::make_pair(*source, *target));
            //             }
            //         }
            //     }
            // }
        } catch (...) {
            // Best effort
        }
    }
};

// Factory function
std::unique_ptr<IMemoryTier> CreateWarmTier(const std::string& storage_path) {
    return std::make_unique<WarmTier>(storage_path);
}

} // namespace dpan
