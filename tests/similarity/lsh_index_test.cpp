// File: tests/similarity/lsh_index_test.cpp
#include "similarity/lsh_index.hpp"
#include "similarity/geometric_similarity.hpp"
#include "similarity/similarity_search.hpp"
#include "storage/memory_backend.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <thread>

namespace dpan {
namespace {

FeatureVector RandomVector(std::mt19937& rng, size_t dimension) {
    std::normal_distribution<float> gaussian(0.0f, 1.0f);
    std::vector<float> values(dimension);
    for (auto& v : values) {
        v = gaussian(rng);
    }
    return FeatureVector(values);
}

FeatureVector Perturb(const FeatureVector& base, std::mt19937& rng, float noise) {
    std::normal_distribution<float> gaussian(0.0f, noise);
    std::vector<float> values = base.Data();
    for (auto& v : values) {
        v += gaussian(rng);
    }
    return FeatureVector(values);
}

TEST(LshIndexTest, InsertRemoveAndCandidates) {
    LshIndex index;
    PatternID a = PatternID::Generate();
    PatternID b = PatternID::Generate();
    FeatureVector va(std::vector<float>{1.0f, 0.2f, 0.1f});
    FeatureVector vb(std::vector<float>{-1.0f, -0.2f, -0.1f});

    index.Insert(a, va);
    index.Insert(b, vb);
    EXPECT_EQ(2u, index.Size());
    EXPECT_TRUE(index.Contains(a));

    // Identical vector always shares its buckets; the opposite vector never
    // does (every sign bit differs)
    auto candidates = index.Candidates(va);
    EXPECT_NE(candidates.end(), std::find(candidates.begin(), candidates.end(), a));
    EXPECT_EQ(candidates.end(), std::find(candidates.begin(), candidates.end(), b));

    // Re-inserting replaces; removing drops it
    index.Insert(a, vb);
    EXPECT_EQ(2u, index.Size());
    candidates = index.Candidates(vb);
    EXPECT_NE(candidates.end(), std::find(candidates.begin(), candidates.end(), a));

    EXPECT_TRUE(index.Remove(a));
    EXPECT_FALSE(index.Remove(a));
    EXPECT_FALSE(index.Contains(a));
    EXPECT_EQ(1u, index.Size());

    index.Clear();
    EXPECT_EQ(0u, index.Size());
    EXPECT_TRUE(index.Candidates(va).empty());
}

TEST(LshIndexTest, SeparatesDimensions) {
    LshIndex index;
    PatternID two = PatternID::Generate();
    PatternID three = PatternID::Generate();
    index.Insert(two, FeatureVector(std::vector<float>{1.0f, 1.0f}));
    index.Insert(three, FeatureVector(std::vector<float>{1.0f, 1.0f, 1.0f}));

    auto candidates = index.Candidates(FeatureVector(std::vector<float>{1.0f, 1.0f}));
    ASSERT_EQ(1u, candidates.size());
    EXPECT_EQ(two, candidates[0]);
    EXPECT_TRUE(index.Candidates(FeatureVector(std::vector<float>{1.0f})).empty());
}

TEST(LshIndexTest, RejectsInvalidConfig) {
    LshIndex::Config config;
    config.bits_per_table = 0;
    EXPECT_THROW(LshIndex{config}, std::invalid_argument);
    config.bits_per_table = 65;
    EXPECT_THROW(LshIndex{config}, std::invalid_argument);
    config.bits_per_table = 8;
    config.num_tables = 0;
    EXPECT_THROW(LshIndex{config}, std::invalid_argument);
}

TEST(LshIndexTest, HighRecallWithFewCandidates) {
    std::mt19937 rng(11);
    const size_t n = 5000;
    const size_t dimension = 32;

    LshIndex index;
    std::vector<FeatureVector> stored;
    std::vector<PatternID> ids;
    for (size_t i = 0; i < n; ++i) {
        stored.push_back(RandomVector(rng, dimension));
        ids.push_back(PatternID::Generate());
        index.Insert(ids.back(), stored.back());
    }

    // Queries are small perturbations of stored vectors; the stored vector
    // is the true nearest neighbour
    size_t found = 0;
    size_t total_candidates = 0;
    const size_t queries = 200;
    for (size_t q = 0; q < queries; ++q) {
        size_t target = (q * 37) % n;
        auto candidates = index.Candidates(Perturb(stored[target], rng, 0.15f));
        total_candidates += candidates.size();
        if (std::find(candidates.begin(), candidates.end(), ids[target]) != candidates.end()) {
            found++;
        }
    }

    EXPECT_GE(static_cast<float>(found) / queries, 0.95f);
    // Far fewer candidates than an exhaustive scan
    EXPECT_LT(static_cast<float>(total_candidates) / queries, 0.2f * n);
}

TEST(LshIndexTest, ConcurrentInsertAndQuery) {
    LshIndex index;
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&index, t]() {
            std::mt19937 rng(t);
            for (int i = 0; i < 300; ++i) {
                PatternID id = PatternID::Generate();
                FeatureVector v = RandomVector(rng, 8);
                index.Insert(id, v);
                index.Candidates(v);
                if (i % 3 == 0) {
                    index.Remove(id);
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_EQ(4u * 200u, index.Size());
}

TEST(ApproximateSearchTest, FindsNearestNeighboursAndTracksChanges) {
    auto db = std::make_shared<MemoryBackend>(MemoryBackend::Config{});
    std::mt19937 rng(3);
    std::vector<PatternID> ids;
    std::vector<FeatureVector> stored;
    for (int i = 0; i < 2000; ++i) {
        stored.push_back(RandomVector(rng, 16));
        PatternNode node(PatternID::Generate(), PatternData::FromFeatures(stored.back(), DataModality::NUMERIC),
                         PatternType::ATOMIC);
        ids.push_back(node.GetID());
        db->Store(node);
    }

    ApproximateSearch search(db, std::make_shared<CosineSimilarity>());
    search.BuildIndex();
    EXPECT_EQ(2000u, search.IndexedCount());

    auto query = PatternData::FromFeatures(Perturb(stored[5], rng, 0.05f), DataModality::NUMERIC);
    auto results = search.Search(query, SearchConfig::TopK(3));
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(ids[5], results[0].pattern_id);
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i - 1].similarity, results[i].similarity);
    }

    // Incremental maintenance
    PatternNode added(PatternID::Generate(), PatternData::FromFeatures(stored[5], DataModality::NUMERIC),
                      PatternType::ATOMIC);
    db->Store(added);
    EXPECT_TRUE(search.Insert(added.GetID()));
    EXPECT_FALSE(search.Insert(PatternID::Generate()));  // Not in database
    EXPECT_TRUE(search.Remove(ids[5]));

    results = search.Search(query, SearchConfig::TopK(1));
    ASSERT_EQ(1u, results.size());
    EXPECT_EQ(added.GetID(), results[0].pattern_id);
}

}  // namespace
}  // namespace dpan
