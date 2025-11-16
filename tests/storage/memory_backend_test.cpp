// File: tests/storage/memory_backend_test.cpp
#include "storage/memory_backend.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <ctime>

namespace dpan {
namespace {

// ============================================================================
// Helper Functions
// ============================================================================

PatternNode CreateTestPattern(PatternID id = PatternID::Generate()) {
    FeatureVector features(3);
    features[0] = 1.0f;
    features[1] = 2.0f;
    features[2] = 3.0f;

    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    return PatternNode(id, data, PatternType::ATOMIC);
}

// ============================================================================
// Constructor and Configuration Tests
// ============================================================================

TEST(MemoryBackendTest, DefaultConstructorWorks) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    EXPECT_EQ(0u, backend.Count());
}

TEST(MemoryBackendTest, ConfigSetsInitialCapacity) {
    MemoryBackend::Config config;
    config.initial_capacity = 100;

    MemoryBackend backend(config);
    EXPECT_EQ(0u, backend.Count());  // Empty, but capacity is set
}

// ============================================================================
// Store Tests
// ============================================================================

TEST(MemoryBackendTest, StoreNewPattern) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternNode node = CreateTestPattern();
    bool result = backend.Store(node);

    EXPECT_TRUE(result);
    EXPECT_EQ(1u, backend.Count());
}

TEST(MemoryBackendTest, StoreDuplicatePatternFails) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id = PatternID::Generate();
    PatternNode node1 = CreateTestPattern(id);
    PatternNode node2 = CreateTestPattern(id);

    EXPECT_TRUE(backend.Store(node1));
    EXPECT_FALSE(backend.Store(node2));  // Duplicate should fail
    EXPECT_EQ(1u, backend.Count());
}

TEST(MemoryBackendTest, StoreMultiplePatterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    for (int i = 0; i < 10; ++i) {
        PatternNode node = CreateTestPattern();
        EXPECT_TRUE(backend.Store(node));
    }

    EXPECT_EQ(10u, backend.Count());
}

// ============================================================================
// Retrieve Tests
// ============================================================================

TEST(MemoryBackendTest, RetrieveExistingPattern) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id = PatternID::Generate();
    PatternNode node = CreateTestPattern(id);

    backend.Store(node);

    std::optional<PatternNode> retrieved = backend.Retrieve(id);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(id, retrieved->GetID());
}

TEST(MemoryBackendTest, RetrieveNonExistentPattern) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id = PatternID::Generate();
    std::optional<PatternNode> retrieved = backend.Retrieve(id);

    EXPECT_FALSE(retrieved.has_value());
}

TEST(MemoryBackendTest, RetrieveAfterStore) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    std::vector<PatternID> ids;
    for (int i = 0; i < 5; ++i) {
        PatternID id = PatternID::Generate();
        ids.push_back(id);
        PatternNode node = CreateTestPattern(id);
        backend.Store(node);
    }

    // Retrieve all patterns
    for (const auto& id : ids) {
        std::optional<PatternNode> retrieved = backend.Retrieve(id);
        EXPECT_TRUE(retrieved.has_value());
        EXPECT_EQ(id, retrieved->GetID());
    }
}

// ============================================================================
// Update Tests
// ============================================================================

TEST(MemoryBackendTest, UpdateExistingPattern) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id = PatternID::Generate();
    PatternNode node1 = CreateTestPattern(id);
    backend.Store(node1);

    // Create updated pattern with different data
    FeatureVector new_features(3);
    new_features[0] = 10.0f;
    new_features[1] = 20.0f;
    new_features[2] = 30.0f;
    PatternData new_data = PatternData::FromFeatures(new_features, DataModality::NUMERIC);
    PatternNode node2(id, new_data, PatternType::COMPOSITE);

    bool result = backend.Update(node2);
    EXPECT_TRUE(result);

    // Verify update
    std::optional<PatternNode> retrieved = backend.Retrieve(id);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(PatternType::COMPOSITE, retrieved->GetType());
}

TEST(MemoryBackendTest, UpdateNonExistentPatternFails) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternNode node = CreateTestPattern();
    bool result = backend.Update(node);

    EXPECT_FALSE(result);
}

// ============================================================================
// Delete Tests
// ============================================================================

TEST(MemoryBackendTest, DeleteExistingPattern) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id = PatternID::Generate();
    PatternNode node = CreateTestPattern(id);
    backend.Store(node);

    EXPECT_EQ(1u, backend.Count());

    bool result = backend.Delete(id);
    EXPECT_TRUE(result);
    EXPECT_EQ(0u, backend.Count());
}

TEST(MemoryBackendTest, DeleteNonExistentPatternFails) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id = PatternID::Generate();
    bool result = backend.Delete(id);

    EXPECT_FALSE(result);
}

TEST(MemoryBackendTest, DeleteAndRetrieveFails) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id = PatternID::Generate();
    PatternNode node = CreateTestPattern(id);
    backend.Store(node);
    backend.Delete(id);

    std::optional<PatternNode> retrieved = backend.Retrieve(id);
    EXPECT_FALSE(retrieved.has_value());
}

// ============================================================================
// Exists Tests
// ============================================================================

TEST(MemoryBackendTest, ExistsReturnsTrueForStoredPattern) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id = PatternID::Generate();
    PatternNode node = CreateTestPattern(id);
    backend.Store(node);

    EXPECT_TRUE(backend.Exists(id));
}

TEST(MemoryBackendTest, ExistsReturnsFalseForNonExistent) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id = PatternID::Generate();
    EXPECT_FALSE(backend.Exists(id));
}

// ============================================================================
// Batch Operations Tests
// ============================================================================

TEST(MemoryBackendTest, StoreBatchMultiplePatterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    std::vector<PatternNode> nodes;
    for (int i = 0; i < 10; ++i) {
        nodes.push_back(CreateTestPattern());
    }

    size_t stored = backend.StoreBatch(nodes);
    EXPECT_EQ(10u, stored);
    EXPECT_EQ(10u, backend.Count());
}

TEST(MemoryBackendTest, StoreBatchSkipsDuplicates) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id = PatternID::Generate();
    PatternNode node1 = CreateTestPattern(id);
    backend.Store(node1);

    std::vector<PatternNode> nodes;
    nodes.push_back(CreateTestPattern(id));  // Duplicate
    nodes.push_back(CreateTestPattern());    // New
    nodes.push_back(CreateTestPattern());    // New

    size_t stored = backend.StoreBatch(nodes);
    EXPECT_EQ(2u, stored);  // Only 2 new patterns
    EXPECT_EQ(3u, backend.Count());  // Total 3
}

TEST(MemoryBackendTest, RetrieveBatchMultiplePatterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    std::vector<PatternID> ids;
    for (int i = 0; i < 5; ++i) {
        PatternID id = PatternID::Generate();
        ids.push_back(id);
        backend.Store(CreateTestPattern(id));
    }

    std::vector<PatternNode> retrieved = backend.RetrieveBatch(ids);
    EXPECT_EQ(5u, retrieved.size());
}

TEST(MemoryBackendTest, RetrieveBatchSkipsMissing) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id1 = PatternID::Generate();
    PatternID id2 = PatternID::Generate();
    PatternID id3 = PatternID::Generate();

    backend.Store(CreateTestPattern(id1));
    backend.Store(CreateTestPattern(id3));

    std::vector<PatternID> ids = {id1, id2, id3};  // id2 doesn't exist
    std::vector<PatternNode> retrieved = backend.RetrieveBatch(ids);

    EXPECT_EQ(2u, retrieved.size());  // Only id1 and id3
}

TEST(MemoryBackendTest, DeleteBatchMultiplePatterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    std::vector<PatternID> ids;
    for (int i = 0; i < 5; ++i) {
        PatternID id = PatternID::Generate();
        ids.push_back(id);
        backend.Store(CreateTestPattern(id));
    }

    EXPECT_EQ(5u, backend.Count());

    size_t deleted = backend.DeleteBatch(ids);
    EXPECT_EQ(5u, deleted);
    EXPECT_EQ(0u, backend.Count());
}

TEST(MemoryBackendTest, DeleteBatchSkipsMissing) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    PatternID id1 = PatternID::Generate();
    PatternID id2 = PatternID::Generate();
    PatternID id3 = PatternID::Generate();

    backend.Store(CreateTestPattern(id1));
    backend.Store(CreateTestPattern(id3));

    std::vector<PatternID> ids = {id1, id2, id3};  // id2 doesn't exist
    size_t deleted = backend.DeleteBatch(ids);

    EXPECT_EQ(2u, deleted);  // Only id1 and id3 deleted
    EXPECT_EQ(0u, backend.Count());
}

// ============================================================================
// Query Tests
// ============================================================================

TEST(MemoryBackendTest, FindByTypeReturnsMatchingPatterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store patterns of different types
    for (int i = 0; i < 3; ++i) {
        PatternID id = PatternID::Generate();
        FeatureVector features(3);
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        PatternNode node(id, data, PatternType::ATOMIC);
        backend.Store(node);
    }

    for (int i = 0; i < 2; ++i) {
        PatternID id = PatternID::Generate();
        FeatureVector features(3);
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        PatternNode node(id, data, PatternType::COMPOSITE);
        backend.Store(node);
    }

    QueryOptions options;
    std::vector<PatternID> atomic = backend.FindByType(PatternType::ATOMIC, options);
    std::vector<PatternID> composite = backend.FindByType(PatternType::COMPOSITE, options);

    EXPECT_EQ(3u, atomic.size());
    EXPECT_EQ(2u, composite.size());
}

TEST(MemoryBackendTest, FindByTypeRespectsMaxResults) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store 10 ATOMIC patterns
    for (int i = 0; i < 10; ++i) {
        backend.Store(CreateTestPattern());
    }

    QueryOptions options;
    options.max_results = 5;

    std::vector<PatternID> results = backend.FindByType(PatternType::ATOMIC, options);
    EXPECT_LE(results.size(), 5u);
}

TEST(MemoryBackendTest, FindByTimeRangeReturnsMatchingPatterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    Timestamp start = Timestamp::Now();

    // Store some patterns
    for (int i = 0; i < 5; ++i) {
        backend.Store(CreateTestPattern());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    Timestamp end = Timestamp::Now();

    QueryOptions options;
    std::vector<PatternID> results = backend.FindByTimeRange(start, end, options);

    EXPECT_EQ(5u, results.size());
}

TEST(MemoryBackendTest, FindAllReturnsAllPatterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    for (int i = 0; i < 7; ++i) {
        backend.Store(CreateTestPattern());
    }

    QueryOptions options;
    options.max_results = 100;

    std::vector<PatternID> results = backend.FindAll(options);
    EXPECT_EQ(7u, results.size());
}

TEST(MemoryBackendTest, FindAllRespectsMaxResults) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    for (int i = 0; i < 20; ++i) {
        backend.Store(CreateTestPattern());
    }

    QueryOptions options;
    options.max_results = 10;

    std::vector<PatternID> results = backend.FindAll(options);
    EXPECT_LE(results.size(), 10u);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(MemoryBackendTest, CountReturnsCorrectNumber) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    EXPECT_EQ(0u, backend.Count());

    backend.Store(CreateTestPattern());
    EXPECT_EQ(1u, backend.Count());

    backend.Store(CreateTestPattern());
    EXPECT_EQ(2u, backend.Count());
}

TEST(MemoryBackendTest, GetStatsReturnsValidStats) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    for (int i = 0; i < 5; ++i) {
        backend.Store(CreateTestPattern());
    }

    StorageStats stats = backend.GetStats();

    EXPECT_EQ(5u, stats.total_patterns);
    EXPECT_GT(stats.memory_usage_bytes, 0u);
}

// ============================================================================
// Maintenance Tests
// ============================================================================

TEST(MemoryBackendTest, ClearRemovesAllPatterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    for (int i = 0; i < 10; ++i) {
        backend.Store(CreateTestPattern());
    }

    EXPECT_EQ(10u, backend.Count());

    backend.Clear();

    EXPECT_EQ(0u, backend.Count());
}

TEST(MemoryBackendTest, CompactDoesntLoseData) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    std::vector<PatternID> ids;
    for (int i = 0; i < 10; ++i) {
        PatternID id = PatternID::Generate();
        ids.push_back(id);
        backend.Store(CreateTestPattern(id));
    }

    backend.Compact();

    EXPECT_EQ(10u, backend.Count());

    // Verify all patterns still exist
    for (const auto& id : ids) {
        EXPECT_TRUE(backend.Exists(id));
    }
}

TEST(MemoryBackendTest, FlushDoesntCrash) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    for (int i = 0; i < 5; ++i) {
        backend.Store(CreateTestPattern());
    }

    EXPECT_NO_THROW(backend.Flush());
}

// ============================================================================
// Snapshot and Restore Tests
// ============================================================================

TEST(MemoryBackendTest, CreateSnapshotSucceeds) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    for (int i = 0; i < 5; ++i) {
        backend.Store(CreateTestPattern());
    }

    std::string snapshot_path = "/tmp/test_snapshot_" + std::to_string(std::time(nullptr)) + ".bin";
    bool result = backend.CreateSnapshot(snapshot_path);

    EXPECT_TRUE(result);

    // Clean up
    std::remove(snapshot_path.c_str());
}

TEST(MemoryBackendTest, SnapshotAndRestorePreservesData) {
    std::string snapshot_path = "/tmp/test_snapshot_" + std::to_string(std::time(nullptr)) + ".bin";

    std::vector<PatternID> ids;

    // Create backend and store patterns
    {
        MemoryBackend::Config config;
        MemoryBackend backend(config);

        for (int i = 0; i < 5; ++i) {
            PatternID id = PatternID::Generate();
            ids.push_back(id);
            backend.Store(CreateTestPattern(id));
        }

        backend.CreateSnapshot(snapshot_path);
    }

    // Create new backend and restore
    {
        MemoryBackend::Config config;
        MemoryBackend backend(config);

        bool result = backend.RestoreSnapshot(snapshot_path);
        EXPECT_TRUE(result);
        EXPECT_EQ(5u, backend.Count());

        // Verify all patterns exist
        for (const auto& id : ids) {
            EXPECT_TRUE(backend.Exists(id));
        }
    }

    // Clean up
    std::remove(snapshot_path.c_str());
}

TEST(MemoryBackendTest, RestoreNonExistentSnapshotFails) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    bool result = backend.RestoreSnapshot("/tmp/nonexistent_snapshot.bin");
    EXPECT_FALSE(result);
}

// ============================================================================
// Concurrency Tests
// ============================================================================

TEST(MemoryBackendTest, ConcurrentStoreIsSafe) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    const int num_threads = 10;
    const int patterns_per_thread = 100;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&backend, patterns_per_thread]() {
            for (int i = 0; i < patterns_per_thread; ++i) {
                backend.Store(CreateTestPattern());
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(num_threads * patterns_per_thread, backend.Count());
}

TEST(MemoryBackendTest, ConcurrentRetrieveIsSafe) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store patterns
    std::vector<PatternID> ids;
    for (int i = 0; i < 100; ++i) {
        PatternID id = PatternID::Generate();
        ids.push_back(id);
        backend.Store(CreateTestPattern(id));
    }

    const int num_threads = 10;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&backend, &ids]() {
            for (const auto& id : ids) {
                std::optional<PatternNode> node = backend.Retrieve(id);
                EXPECT_TRUE(node.has_value());
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

TEST(MemoryBackendTest, ConcurrentMixedOperationsIsSafe) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Pre-populate
    std::vector<PatternID> ids;
    for (int i = 0; i < 50; ++i) {
        PatternID id = PatternID::Generate();
        ids.push_back(id);
        backend.Store(CreateTestPattern(id));
    }

    const int num_threads = 8;
    std::vector<std::thread> threads;

    // Mix of readers and writers
    for (int t = 0; t < num_threads; ++t) {
        if (t % 2 == 0) {
            // Reader threads
            threads.emplace_back([&backend, &ids]() {
                for (int i = 0; i < 100; ++i) {
                    backend.Retrieve(ids[i % ids.size()]);
                }
            });
        } else {
            // Writer threads
            threads.emplace_back([&backend]() {
                for (int i = 0; i < 20; ++i) {
                    backend.Store(CreateTestPattern());
                }
            });
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // At least original patterns + some new ones
    EXPECT_GE(backend.Count(), 50u);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(MemoryBackendTest, SingleLookupPerformance) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store 1000 patterns
    std::vector<PatternID> ids;
    for (int i = 0; i < 1000; ++i) {
        PatternID id = PatternID::Generate();
        ids.push_back(id);
        backend.Store(CreateTestPattern(id));
    }

    // Measure lookup time
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        backend.Retrieve(ids[i % ids.size()]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    float avg_lookup_us = static_cast<float>(duration.count()) / 1000.0f;

    // Should be less than 10 microseconds per lookup on average
    EXPECT_LT(avg_lookup_us, 10.0f);
}

TEST(MemoryBackendTest, BatchLookupPerformance) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store patterns
    std::vector<PatternID> ids;
    for (int i = 0; i < 1000; ++i) {
        PatternID id = PatternID::Generate();
        ids.push_back(id);
        backend.Store(CreateTestPattern(id));
    }

    // Measure batch lookup time for 100 patterns
    std::vector<PatternID> batch_ids(ids.begin(), ids.begin() + 100);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<PatternNode> results = backend.RetrieveBatch(batch_ids);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    EXPECT_EQ(100u, results.size());
    EXPECT_LT(duration.count(), 5000);  // Less than 5ms
}

} // namespace
} // namespace dpan
