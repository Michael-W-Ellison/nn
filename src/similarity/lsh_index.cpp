// File: src/similarity/lsh_index.cpp
#include "similarity/lsh_index.hpp"

#include <algorithm>
#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace dpan {

LshIndex::LshIndex() : LshIndex(Config{}) {}

LshIndex::LshIndex(const Config& config) : config_(config) {
    if (!config_.IsValid()) {
        throw std::invalid_argument("LshIndex: invalid configuration");
    }
}

LshIndex::DimensionIndex& LshIndex::GetOrCreateDimension(size_t dimension) {
    auto it = dimensions_.find(dimension);
    if (it != dimensions_.end()) {
        return it->second;
    }

    DimensionIndex index;
    // Seed per dimension so results do not depend on insertion order
    std::mt19937 rng(config_.seed ^ static_cast<uint32_t>(dimension * 2654435761u));
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    index.hyperplanes.resize(config_.num_tables * config_.bits_per_table);
    for (auto& plane : index.hyperplanes) {
        plane.resize(dimension);
        for (auto& value : plane) {
            value = gaussian(rng);
        }
    }
    index.tables.resize(config_.num_tables);
    return dimensions_.emplace(dimension, std::move(index)).first->second;
}

std::vector<uint64_t> LshIndex::HashKeys(const DimensionIndex& index,
                                         const FeatureVector& features) const {
    std::vector<uint64_t> keys(config_.num_tables, 0);
    const auto& values = features.Data();
    for (size_t t = 0; t < config_.num_tables; ++t) {
        uint64_t key = 0;
        for (size_t b = 0; b < config_.bits_per_table; ++b) {
            const auto& plane = index.hyperplanes[t * config_.bits_per_table + b];
            float dot = 0.0f;
            for (size_t d = 0; d < values.size(); ++d) {
                dot += plane[d] * values[d];
            }
            if (dot >= 0.0f) {
                key |= uint64_t{1} << b;
            }
        }
        keys[t] = key;
    }
    return keys;
}

void LshIndex::Insert(PatternID id, const FeatureVector& features) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    RemoveLocked(id);

    DimensionIndex& index = GetOrCreateDimension(features.Dimension());
    Entry entry{features.Dimension(), HashKeys(index, features)};
    for (size_t t = 0; t < config_.num_tables; ++t) {
        index.tables[t][entry.keys[t]].push_back(id);
    }
    entries_.emplace(id, std::move(entry));
}

void LshIndex::RemoveLocked(PatternID id) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        return;
    }
    DimensionIndex& index = dimensions_.at(it->second.dimension);
    for (size_t t = 0; t < config_.num_tables; ++t) {
        auto bucket = index.tables[t].find(it->second.keys[t]);
        if (bucket == index.tables[t].end()) {
            continue;
        }
        auto& ids = bucket->second;
        ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
        if (ids.empty()) {
            index.tables[t].erase(bucket);
        }
    }
    entries_.erase(it);
}

bool LshIndex::Remove(PatternID id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    bool present = entries_.count(id) > 0;
    RemoveLocked(id);
    return present;
}

bool LshIndex::Contains(PatternID id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return entries_.count(id) > 0;
}

size_t LshIndex::Size() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return entries_.size();
}

void LshIndex::Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    entries_.clear();
    dimensions_.clear();
}

std::vector<PatternID> LshIndex::Ids() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<PatternID> ids;
    ids.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
        ids.push_back(id);
    }
    return ids;
}

std::vector<PatternID> LshIndex::Candidates(const FeatureVector& query) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto dim_it = dimensions_.find(query.Dimension());
    if (dim_it == dimensions_.end()) {
        return {};
    }
    const DimensionIndex& index = dim_it->second;
    std::vector<uint64_t> keys = HashKeys(index, query);

    std::unordered_set<PatternID> seen;
    std::vector<PatternID> candidates;
    auto collect = [&](size_t table, uint64_t key) {
        auto bucket = index.tables[table].find(key);
        if (bucket == index.tables[table].end()) {
            return;
        }
        for (const auto& id : bucket->second) {
            if (seen.insert(id).second) {
                candidates.push_back(id);
            }
        }
    };

    for (size_t t = 0; t < config_.num_tables; ++t) {
        collect(t, keys[t]);
        if (config_.multi_probe) {
            for (size_t b = 0; b < config_.bits_per_table; ++b) {
                collect(t, keys[t] ^ (uint64_t{1} << b));
            }
        }
    }
    return candidates;
}

}  // namespace dpan
