// File: src/memory/archive_tier.cpp
//
// Archive Tier Implementation (Compressed long-term storage)
//
// Uses file-based storage with zlib compression for long-term archival.
// Target latency: <10ms
//
// File format: "DPZ1" magic, uint64 uncompressed size, zlib stream.
// Files without the magic are read as uncompressed (pre-compression format).

#include "memory/memory_tier.hpp"
#include <zlib.h>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iterator>
#include <shared_mutex>
#include <sstream>
#include <unordered_set>

namespace dpan {
namespace fs = std::filesystem;

namespace {

constexpr char kArchiveMagic[4] = {'D', 'P', 'Z', '1'};

/// Compress serialized bytes and write them to `filepath`
bool WriteCompressed(const fs::path& filepath, const std::string& raw) {
    uLongf compressed_size = compressBound(static_cast<uLong>(raw.size()));
    std::string compressed(compressed_size, '\0');
    if (compress2(reinterpret_cast<Bytef*>(compressed.data()), &compressed_size,
                  reinterpret_cast<const Bytef*>(raw.data()),
                  static_cast<uLong>(raw.size()), Z_BEST_COMPRESSION) != Z_OK) {
        return false;
    }

    std::ofstream file(filepath, std::ios::binary | std::ios::trunc);
    if (!file) {
        return false;
    }
    uint64_t raw_size = raw.size();
    file.write(kArchiveMagic, sizeof(kArchiveMagic));
    file.write(reinterpret_cast<const char*>(&raw_size), sizeof(raw_size));
    file.write(compressed.data(), static_cast<std::streamsize>(compressed_size));
    return file.good();
}

/// Read `filepath` and return its decompressed bytes (std::nullopt on error)
std::optional<std::string> ReadCompressed(const fs::path& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }
    std::string contents((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    constexpr size_t kHeaderSize = sizeof(kArchiveMagic) + sizeof(uint64_t);
    if (contents.size() < kHeaderSize ||
        std::memcmp(contents.data(), kArchiveMagic, sizeof(kArchiveMagic)) != 0) {
        return contents;  // Uncompressed legacy file
    }

    uint64_t raw_size;
    std::memcpy(&raw_size, contents.data() + sizeof(kArchiveMagic), sizeof(raw_size));

    std::string raw(raw_size, '\0');
    uLongf dest_size = static_cast<uLongf>(raw_size);
    if (uncompress(reinterpret_cast<Bytef*>(raw.data()), &dest_size,
                   reinterpret_cast<const Bytef*>(contents.data() + kHeaderSize),
                   static_cast<uLong>(contents.size() - kHeaderSize)) != Z_OK ||
        dest_size != raw_size) {
        return std::nullopt;
    }
    return raw;
}

}  // namespace

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
            std::ostringstream buffer(std::ios::binary);
            pattern.Serialize(buffer);
            if (!WriteCompressed(filepath, buffer.str())) {
                return false;
            }

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

            auto raw = ReadCompressed(filepath);
            if (!raw) {
                return std::nullopt;
            }

            std::istringstream buffer(*raw, std::ios::binary);
            return PatternNode::Deserialize(buffer);
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

            std::ostringstream buffer(std::ios::binary);
            edge.Serialize(buffer);
            if (!WriteCompressed(filepath, buffer.str())) {
                return false;
            }

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

            auto raw = ReadCompressed(filepath);
            if (!raw) {
                return std::nullopt;
            }

            std::istringstream buffer(*raw, std::ios::binary);
            auto edge_ptr = AssociationEdge::Deserialize(buffer);
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
        // Reconcile the in-memory indices with the files on disk (drops
        // entries whose files were removed externally, adds files written
        // by other instances)
        std::unique_lock<std::shared_mutex> lock(mutex_);
        pattern_index_.clear();
        association_index_.clear();
        RebuildIndex();
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
        // Recover indices from files written by previous instances
        // (pattern files: "<id>.arc", association files: "<source>_<target>.arc")
        try {
            for (const auto& entry : fs::directory_iterator(patterns_dir_)) {
                if (entry.is_regular_file() && entry.path().extension() == ".arc") {
                    auto id = PatternID::FromString(entry.path().stem().string());
                    if (id) {
                        pattern_index_.insert(*id);
                    }
                }
            }

            for (const auto& entry : fs::directory_iterator(associations_dir_)) {
                if (entry.is_regular_file() && entry.path().extension() == ".arc") {
                    std::string filename = entry.path().stem().string();
                    size_t underscore_pos = filename.find('_');
                    if (underscore_pos != std::string::npos) {
                        auto source = PatternID::FromString(filename.substr(0, underscore_pos));
                        auto target = PatternID::FromString(filename.substr(underscore_pos + 1));
                        if (source && target) {
                            association_index_.insert(std::make_pair(*source, *target));
                        }
                    }
                }
            }
        } catch (const fs::filesystem_error&) {
            // Best effort: unreadable directories leave the index partial
        }
    }
};

std::unique_ptr<IMemoryTier> CreateArchiveTier(const std::string& storage_path) {
    return std::make_unique<ArchiveTier>(storage_path);
}

} // namespace dpan
