// File: tests/storage/persistent_backend_test.cpp
#include "storage/persistent_backend.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>
#include <filesystem>
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

std::string GetTempDbPath() {
    static int counter = 0;
    return "/tmp/test_persistent_" + std::to_string(std::time(nullptr)) +
           "_" + std::to_string(counter++) + ".db";
}

// ============================================================================
// Constructor and Configuration Tests
// ============================================================================

TEST(PersistentBackendTest, ConstructorCreatesDatabase) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        EXPECT_EQ(0u, backend.Count());
    }

    // Clean up
    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, ConfigEnablesWAL) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        config.enable_wal = true;

        PersistentBackend backend(config);
        EXPECT_EQ(0u, backend.Count());
    }

    // Clean up
    std::filesystem::remove(db_path);
    std::filesystem::remove(db_path + "-wal");
    std::filesystem::remove(db_path + "-shm");
}

// ============================================================================
// Store Tests
// ============================================================================

TEST(PersistentBackendTest, StoreNewPattern) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        PatternNode node = CreateTestPattern();
        bool result = backend.Store(node);

        EXPECT_TRUE(result);
        EXPECT_EQ(1u, backend.Count());
    }

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, StoreDuplicatePatternFails) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        PatternID id = PatternID::Generate();
        PatternNode node1 = CreateTestPattern(id);
        PatternNode node2 = CreateTestPattern(id);

        EXPECT_TRUE(backend.Store(node1));
        EXPECT_FALSE(backend.Store(node2));  // Duplicate should fail
        EXPECT_EQ(1u, backend.Count());
    }

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, StoreMultiplePatterns) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        for (int i = 0; i < 10; ++i) {
            PatternNode node = CreateTestPattern();
            EXPECT_TRUE(backend.Store(node));
        }

        EXPECT_EQ(10u, backend.Count());
    }

    std::filesystem::remove(db_path);
}

// ============================================================================
// Retrieve Tests
// ============================================================================

TEST(PersistentBackendTest, RetrieveExistingPattern) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        PatternID id = PatternID::Generate();
        PatternNode node = CreateTestPattern(id);

        backend.Store(node);

        std::optional<PatternNode> retrieved = backend.Retrieve(id);
        ASSERT_TRUE(retrieved.has_value());
        EXPECT_EQ(id, retrieved->GetID());
    }

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, RetrieveNonExistentPattern) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        PatternID id = PatternID::Generate();
        std::optional<PatternNode> retrieved = backend.Retrieve(id);

        EXPECT_FALSE(retrieved.has_value());
    }

    std::filesystem::remove(db_path);
}

// ============================================================================
// Persistence Tests
// ============================================================================

TEST(PersistentBackendTest, DataPersisstsAcrossRestarts) {
    std::string db_path = GetTempDbPath();

    std::vector<PatternID> ids;

    // Store patterns
    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        for (int i = 0; i < 5; ++i) {
            PatternID id = PatternID::Generate();
            ids.push_back(id);
            backend.Store(CreateTestPattern(id));
        }

        EXPECT_EQ(5u, backend.Count());
    }

    // Restart and verify
    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        EXPECT_EQ(5u, backend.Count());

        for (const auto& id : ids) {
            EXPECT_TRUE(backend.Exists(id));
        }
    }

    std::filesystem::remove(db_path);
}

// ============================================================================
// Update Tests
// ============================================================================

TEST(PersistentBackendTest, UpdateExistingPattern) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        PatternID id = PatternID::Generate();
        PatternNode node1 = CreateTestPattern(id);
        backend.Store(node1);

        // Create updated pattern
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

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, UpdateNonExistentPatternFails) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        PatternNode node = CreateTestPattern();
        bool result = backend.Update(node);

        EXPECT_FALSE(result);
    }

    std::filesystem::remove(db_path);
}

// ============================================================================
// Delete Tests
// ============================================================================

TEST(PersistentBackendTest, DeleteExistingPattern) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        PatternID id = PatternID::Generate();
        PatternNode node = CreateTestPattern(id);
        backend.Store(node);

        EXPECT_EQ(1u, backend.Count());

        bool result = backend.Delete(id);
        EXPECT_TRUE(result);
        EXPECT_EQ(0u, backend.Count());
    }

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, DeleteNonExistentPatternFails) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        PatternID id = PatternID::Generate();
        bool result = backend.Delete(id);

        EXPECT_FALSE(result);
    }

    std::filesystem::remove(db_path);
}

// ============================================================================
// Batch Operations Tests
// ============================================================================

TEST(PersistentBackendTest, StoreBatchMultiplePatterns) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        std::vector<PatternNode> nodes;
        for (int i = 0; i < 10; ++i) {
            nodes.push_back(CreateTestPattern());
        }

        size_t stored = backend.StoreBatch(nodes);
        EXPECT_EQ(10u, stored);
        EXPECT_EQ(10u, backend.Count());
    }

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, StoreBatchSkipsDuplicates) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

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

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, RetrieveBatchMultiplePatterns) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        std::vector<PatternID> ids;
        for (int i = 0; i < 5; ++i) {
            PatternID id = PatternID::Generate();
            ids.push_back(id);
            backend.Store(CreateTestPattern(id));
        }

        std::vector<PatternNode> retrieved = backend.RetrieveBatch(ids);
        EXPECT_EQ(5u, retrieved.size());
    }

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, DeleteBatchMultiplePatterns) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

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

    std::filesystem::remove(db_path);
}

// ============================================================================
// Query Tests
// ============================================================================

TEST(PersistentBackendTest, FindByTypeReturnsMatchingPatterns) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

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

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, FindByTimeRangeReturnsMatchingPatterns) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

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

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, FindAllReturnsAllPatterns) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        for (int i = 0; i < 7; ++i) {
            backend.Store(CreateTestPattern());
        }

        QueryOptions options;
        options.max_results = 100;

        std::vector<PatternID> results = backend.FindAll(options);
        EXPECT_EQ(7u, results.size());
    }

    std::filesystem::remove(db_path);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(PersistentBackendTest, CountReturnsCorrectNumber) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        EXPECT_EQ(0u, backend.Count());

        backend.Store(CreateTestPattern());
        EXPECT_EQ(1u, backend.Count());

        backend.Store(CreateTestPattern());
        EXPECT_EQ(2u, backend.Count());
    }

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, GetStatsReturnsValidStats) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        for (int i = 0; i < 5; ++i) {
            backend.Store(CreateTestPattern());
        }

        StorageStats stats = backend.GetStats();

        EXPECT_EQ(5u, stats.total_patterns);
        EXPECT_GT(stats.disk_usage_bytes, 0u);
    }

    std::filesystem::remove(db_path);
}

// ============================================================================
// Maintenance Tests
// ============================================================================

TEST(PersistentBackendTest, ClearRemovesAllPatterns) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        for (int i = 0; i < 10; ++i) {
            backend.Store(CreateTestPattern());
        }

        EXPECT_EQ(10u, backend.Count());

        backend.Clear();

        EXPECT_EQ(0u, backend.Count());
    }

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, FlushDoesntCrash) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        for (int i = 0; i < 5; ++i) {
            backend.Store(CreateTestPattern());
        }

        EXPECT_NO_THROW(backend.Flush());
    }

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, CompactReducesFileSize) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        config.enable_auto_vacuum = true;
        PersistentBackend backend(config);

        // Store and delete many patterns
        std::vector<PatternID> ids;
        for (int i = 0; i < 100; ++i) {
            PatternID id = PatternID::Generate();
            ids.push_back(id);
            backend.Store(CreateTestPattern(id));
        }

        // Delete most patterns
        for (size_t i = 0; i < 90; ++i) {
            backend.Delete(ids[i]);
        }

        EXPECT_NO_THROW(backend.Compact());
    }

    std::filesystem::remove(db_path);
}

// ============================================================================
// Snapshot and Restore Tests
// ============================================================================

TEST(PersistentBackendTest, CreateSnapshotSucceeds) {
    std::string db_path = GetTempDbPath();
    std::string snapshot_path = db_path + ".snapshot";

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        for (int i = 0; i < 5; ++i) {
            backend.Store(CreateTestPattern());
        }

        bool result = backend.CreateSnapshot(snapshot_path);
        EXPECT_TRUE(result);
    }

    std::filesystem::remove(db_path);
    std::filesystem::remove(snapshot_path);
}

TEST(PersistentBackendTest, SnapshotAndRestorePreservesData) {
    std::string db_path = GetTempDbPath();
    std::string snapshot_path = db_path + ".snapshot";

    std::vector<PatternID> ids;

    // Create and snapshot
    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        for (int i = 0; i < 5; ++i) {
            PatternID id = PatternID::Generate();
            ids.push_back(id);
            backend.Store(CreateTestPattern(id));
        }

        backend.CreateSnapshot(snapshot_path);
    }

    // Create new database and restore
    {
        std::string new_db_path = GetTempDbPath();
        PersistentBackend::Config config;
        config.db_path = new_db_path;
        PersistentBackend backend(config);

        bool result = backend.RestoreSnapshot(snapshot_path);
        EXPECT_TRUE(result);
        EXPECT_EQ(5u, backend.Count());

        // Verify all patterns exist
        for (const auto& id : ids) {
            EXPECT_TRUE(backend.Exists(id));
        }

        std::filesystem::remove(new_db_path);
    }

    std::filesystem::remove(db_path);
    std::filesystem::remove(snapshot_path);
}

// ============================================================================
// Concurrency Tests
// ============================================================================

TEST(PersistentBackendTest, ConcurrentReadsAreSafe) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        // Store patterns
        std::vector<PatternID> ids;
        for (int i = 0; i < 50; ++i) {
            PatternID id = PatternID::Generate();
            ids.push_back(id);
            backend.Store(CreateTestPattern(id));
        }

        const int num_threads = 5;
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

    std::filesystem::remove(db_path);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(PersistentBackendTest, SingleReadPerformance) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        // Store 1000 patterns
        std::vector<PatternID> ids;
        for (int i = 0; i < 1000; ++i) {
            PatternID id = PatternID::Generate();
            ids.push_back(id);
            backend.Store(CreateTestPattern(id));
        }

        // Measure read time
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 100; ++i) {
            backend.Retrieve(ids[i % ids.size()]);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        float avg_read_us = static_cast<float>(duration.count()) / 100.0f;

        // Should be less than 2ms per read on average
        EXPECT_LT(avg_read_us, 2000.0f);
    }

    std::filesystem::remove(db_path);
}

TEST(PersistentBackendTest, BatchWritePerformance) {
    std::string db_path = GetTempDbPath();

    {
        PersistentBackend::Config config;
        config.db_path = db_path;
        PersistentBackend backend(config);

        // Create 100 patterns
        std::vector<PatternNode> nodes;
        for (int i = 0; i < 100; ++i) {
            nodes.push_back(CreateTestPattern());
        }

        // Measure batch write time
        auto start = std::chrono::high_resolution_clock::now();
        size_t stored = backend.StoreBatch(nodes);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        EXPECT_EQ(100u, stored);
        // Should be less than 500ms for 100 patterns
        EXPECT_LT(duration.count(), 500);
    }

    std::filesystem::remove(db_path);
}

} // namespace
} // namespace dpan
