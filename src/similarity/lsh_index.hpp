// File: src/similarity/lsh_index.hpp
//
// Locality-sensitive hashing index for approximate nearest-neighbour
// candidate retrieval over feature vectors.
//
// Random-hyperplane hashing (SimHash): each of `num_tables` tables hashes a
// vector to `bits_per_table` sign bits of its dot products with random
// Gaussian hyperplanes. Vectors at a small angle share buckets with high
// probability, so candidates approximate the nearest neighbours by cosine
// similarity. Callers re-rank candidates with their actual metric.
//
// Vectors of different dimensions are indexed separately (a query only
// returns vectors of its own dimension).
//
// Thread safety: all methods are thread-safe (internal shared_mutex).
#pragma once

#include "core/pattern_data.hpp"
#include "core/types.hpp"

#include <cstdint>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace dpan {

class LshIndex {
public:
    struct Config {
        /// Independent hash tables; more tables raise recall and memory use
        size_t num_tables{8};

        /// Sign bits per table (1..64); more bits make buckets smaller
        /// (faster queries, lower recall)
        size_t bits_per_table{12};

        /// Also probe buckets that differ from the query's in one bit
        /// (raises recall at the cost of bits_per_table extra lookups per table)
        bool multi_probe{true};

        uint32_t seed{42};

        bool IsValid() const {
            return num_tables > 0 && bits_per_table > 0 && bits_per_table <= 64;
        }
    };

    LshIndex();
    explicit LshIndex(const Config& config);

    /// Add or replace the vector stored for `id`
    void Insert(PatternID id, const FeatureVector& features);

    /// @return true if `id` was indexed
    bool Remove(PatternID id);

    bool Contains(PatternID id) const;
    size_t Size() const;
    void Clear();

    /// IDs of indexed vectors sharing a probed bucket with `query`
    /// (unordered, without duplicates)
    std::vector<PatternID> Candidates(const FeatureVector& query) const;

    /// All indexed IDs
    std::vector<PatternID> Ids() const;

    const Config& GetConfig() const { return config_; }

private:
    struct DimensionIndex {
        /// num_tables * bits_per_table hyperplanes of `dimension` values
        std::vector<std::vector<float>> hyperplanes;
        /// Per table: hash key -> IDs
        std::vector<std::unordered_map<uint64_t, std::vector<PatternID>>> tables;
    };

    struct Entry {
        size_t dimension;
        std::vector<uint64_t> keys;  // One per table
    };

    Config config_;
    std::unordered_map<size_t, DimensionIndex> dimensions_;
    std::unordered_map<PatternID, Entry> entries_;
    mutable std::shared_mutex mutex_;

    DimensionIndex& GetOrCreateDimension(size_t dimension);
    std::vector<uint64_t> HashKeys(const DimensionIndex& index, const FeatureVector& features) const;
    void RemoveLocked(PatternID id);
};

}  // namespace dpan
