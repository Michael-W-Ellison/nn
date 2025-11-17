// File: src/memory/archive_tier.cpp
//
// Archive Tier Implementation (Compressed long-term storage)
//
// Uses file-based storage with compression for long-term archival.
// Target latency: <10ms
//
// NOTE: Compression not yet implemented - placeholder for future enhancement

#include "memory/memory_tier.hpp"
#include <fstream>
#include <filesystem>
#include <shared_mutex>
#include <unordered_set>

namespace dpan {
namespace fs = std::filesystem;

class ArchiveTier : public IMemoryTier {
public:
    explicit ArchiveTier(const std::string& storage_path)
        : storage_path_(storage_path) {

        fs::create_directories(storage_path_);
        patterns_dir_ = storage_path_ / "patterns";
        associations_dir_ = storage_path_ / "associations";
        fs::create_directories(patterns_dir_);
        fs::create_directories(associations_dir_);

        RebuildIndex();
    }

    bool StorePattern(const PatternNode& pattern) override {
        try {
            fs::path filepath = patterns_dir_ / (pattern.GetID().ToString() + ".arc");
            std::ofstream file(filepath, std::ios::binary);
            if (!file) {
                return false;
            }

            // TODO: Add compression here
            pattern.Serialize(file);

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
            fs::path filepath = patterns_dir_ / (id.ToString() + ".arc");

            if (!fs::exists(filepath)) {
                return std::nullopt;
            }

            std::ifstream file(filepath, std::ios::binary);
            if (!file) {
                return std::nullopt;
            }

            // TODO: Add decompression here
            return PatternNode::Deserialize(file);
        } catch (...) {
            return std::nullopt;
        }
    }

    bool RemovePattern(PatternID id) override {
        try {
            fs::path filepath = patterns_dir_ / (id.ToString() + ".arc");

            if (fs::exists(filepath)) {
                fs::remove(filepath);

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

    bool StoreAssociation(const AssociationEdge& edge) override {
        try {
            std::string filename = edge.GetSource().ToString() + "_" +
                                 edge.GetTarget().ToString() + ".arc";
            fs::path filepath = associations_dir_ / filename;

            std::ofstream file(filepath, std::ios::binary);
            if (!file) {
                return false;
            }

            // TODO: Add compression here
            edge.Serialize(file);

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
                                 target.ToString() + ".arc";
            fs::path filepath = associations_dir_ / filename;

            if (!fs::exists(filepath)) {
                return std::nullopt;
            }

            std::ifstream file(filepath, std::ios::binary);
            if (!file) {
                return std::nullopt;
            }

            // TODO: Add decompression here
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
                                 target.ToString() + ".arc";
            fs::path filepath = associations_dir_ / filename;

            if (fs::exists(filepath)) {
                fs::remove(filepath);

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
            // Best effort
        }

        return total;
    }

    MemoryTier GetTierLevel() const override {
        return MemoryTier::ARCHIVE;
    }

    std::string GetTierName() const override {
        return "Archive";
    }

    void Compact() override {
        // Could implement compression optimization here
    }

    void Clear() override {
        try {
            for (const auto& entry : fs::directory_iterator(patterns_dir_)) {
                if (entry.is_regular_file()) {
                    fs::remove(entry.path());
                }
            }

            for (const auto& entry : fs::directory_iterator(associations_dir_)) {
                if (entry.is_regular_file()) {
                    fs::remove(entry.path());
                }
            }

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
        // Synchronous I/O, no buffering
    }

private:
    fs::path storage_path_;
    fs::path patterns_dir_;
    fs::path associations_dir_;

    std::unordered_set<PatternID> pattern_index_;
    std::unordered_set<std::pair<PatternID, PatternID>, PatternPairHash> association_index_;
    mutable std::shared_mutex mutex_;

    void RebuildIndex() {
        // TODO: Implement index rebuilding once PatternID::FromString() is available
        // For now, indices are built incrementally as patterns are stored
        try {
            // for (const auto& entry : fs::directory_iterator(patterns_dir_)) {
            //     if (entry.is_regular_file() && entry.path().extension() == ".arc") {
            //         std::string filename = entry.path().stem().string();
            //         auto id = PatternID::FromString(filename);
            //         if (id) {
            //             pattern_index_.insert(*id);
            //         }
            //     }
            // }

            // for (const auto& entry : fs::directory_iterator(associations_dir_)) {
            //     if (entry.is_regular_file() && entry.path().extension() == ".arc") {
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

std::unique_ptr<IMemoryTier> CreateArchiveTier(const std::string& storage_path) {
    return std::make_unique<ArchiveTier>(storage_path);
}

} // namespace dpan
