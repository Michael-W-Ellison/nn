// File: tests/benchmarks/storage_benchmarks.cpp
//
// Performance benchmarks for Storage module

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include "storage/memory_backend.hpp"
#include "storage/pattern_database.hpp"
#include "core/pattern_node.hpp"

using namespace dpan;
using namespace std::chrono;

// ============================================================================
// Benchmark Helper
// ============================================================================

struct BenchmarkTimer {
    using TimePoint = high_resolution_clock::time_point;
    TimePoint start;

    BenchmarkTimer() : start(high_resolution_clock::now()) {}

    double ElapsedMs() const {
        auto end = high_resolution_clock::now();
        return duration_cast<duration<double, std::milli>>(end - start).count();
    }
};

PatternNode CreateTestPattern(size_t size = 10) {
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i) / static_cast<float>(size);
    }
    FeatureVector features(data);
    PatternData pattern_data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    return PatternNode(PatternID::Generate(), pattern_data, PatternType::ATOMIC);
}

// ============================================================================
// MemoryBackend Benchmarks
// ============================================================================

TEST(MemoryBackendBenchmark, Store_10000_Patterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    BenchmarkTimer timer;
    std::vector<PatternID> ids;

    for (size_t i = 0; i < 10000; ++i) {
        auto pattern = CreateTestPattern(10);
        bool result = backend.Store(pattern);
        ASSERT_TRUE(result);
        ids.push_back(pattern.GetID());
    }

    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (10000.0 / elapsed) * 1000.0;

    std::cout << "MemoryBackend Store (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 1000.0); // Should complete in < 1s
}

TEST(MemoryBackendBenchmark, Retrieve_10000_Patterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store patterns first
    std::vector<PatternID> ids;
    for (size_t i = 0; i < 1000; ++i) {
        auto pattern = CreateTestPattern(10);
        bool result = backend.Store(pattern);
        ASSERT_TRUE(result);
        ids.push_back(pattern.GetID());
    }

    // Benchmark retrieval
    BenchmarkTimer timer;
    for (size_t i = 0; i < 10000; ++i) {
        size_t idx = i % ids.size();
        auto pattern = backend.Retrieve(ids[idx]);
        EXPECT_TRUE(pattern.has_value());
    }

    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (10000.0 / elapsed) * 1000.0;

    std::cout << "MemoryBackend Retrieve (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 100.0); // Should complete in < 100ms
}

TEST(MemoryBackendBenchmark, Update_10000_Patterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store patterns first
    std::vector<PatternID> ids;
    for (size_t i = 0; i < 1000; ++i) {
        auto pattern = CreateTestPattern(10);
        bool result = backend.Store(pattern);
        ASSERT_TRUE(result);
        ids.push_back(pattern.GetID());
    }

    // Benchmark updates
    BenchmarkTimer timer;
    for (size_t i = 0; i < 10000; ++i) {
        size_t idx = i % ids.size();
        std::vector<float> data(15);
        for (size_t j = 0; j < 15; ++j) {
            data[j] = static_cast<float>(j) / 15.0f;
        }
        FeatureVector features(data);
        PatternData pattern_data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        PatternNode pattern(ids[idx], pattern_data, PatternType::ATOMIC);
        backend.Update(pattern);
    }

    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (10000.0 / elapsed) * 1000.0;

    std::cout << "MemoryBackend Update (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 200.0); // Should complete in < 200ms
}

TEST(MemoryBackendBenchmark, Delete_10000_Patterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store patterns first
    std::vector<PatternID> ids;
    for (size_t i = 0; i < 10000; ++i) {
        auto pattern = CreateTestPattern(10);
        bool result = backend.Store(pattern);
        ASSERT_TRUE(result);
        ids.push_back(pattern.GetID());
    }

    // Benchmark deletions
    BenchmarkTimer timer;
    for (const auto& id : ids) {
        backend.Delete(id);
    }

    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (10000.0 / elapsed) * 1000.0;

    std::cout << "MemoryBackend Delete (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 500.0); // Should complete in < 500ms
}

TEST(MemoryBackendBenchmark, BatchStore_1000x10) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    BenchmarkTimer timer;

    // Store in batches of 10
    for (size_t batch = 0; batch < 1000; ++batch) {
        std::vector<PatternNode> patterns;
        for (size_t i = 0; i < 10; ++i) {
            patterns.push_back(CreateTestPattern(10));
        }
        auto results = backend.StoreBatch(patterns);
        EXPECT_EQ(10u, results);
    }

    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (10000.0 / elapsed) * 1000.0;

    std::cout << "MemoryBackend BatchStore (1000x10): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 1000.0); // Should complete in < 1s
}

TEST(MemoryBackendBenchmark, BatchRetrieve_1000x10) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store patterns first
    std::vector<PatternID> ids;
    for (size_t i = 0; i < 1000; ++i) {
        auto pattern = CreateTestPattern(10);
        bool result = backend.Store(pattern);
        ASSERT_TRUE(result);
        ids.push_back(pattern.GetID());
    }

    // Benchmark batch retrieval
    BenchmarkTimer timer;

    for (size_t batch = 0; batch < 1000; ++batch) {
        std::vector<PatternID> batch_ids;
        for (size_t i = 0; i < 10; ++i) {
            batch_ids.push_back(ids[(batch * 10 + i) % ids.size()]);
        }
        auto results = backend.RetrieveBatch(batch_ids);
        EXPECT_EQ(10u, results.size());
    }

    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (10000.0 / elapsed) * 1000.0;

    std::cout << "MemoryBackend BatchRetrieve (1000x10): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 200.0); // Should complete in < 200ms
}

TEST(MemoryBackendBenchmark, GetStats) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store some patterns
    for (size_t i = 0; i < 1000; ++i) {
        auto pattern = CreateTestPattern(10);
        backend.Store(pattern);
    }

    // Benchmark GetStats
    BenchmarkTimer timer;
    for (size_t i = 0; i < 1000; ++i) {
        auto stats = backend.GetStats();
        EXPECT_EQ(1000u, stats.total_patterns);
    }

    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (1000.0 / elapsed) * 1000.0;

    std::cout << "MemoryBackend GetStats (1000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 50.0); // Should complete in < 50ms
}

// ============================================================================
// Large Scale Storage Benchmarks
// ============================================================================

TEST(StorageScalabilityBenchmark, Store_100k_Patterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    BenchmarkTimer timer;

    for (size_t i = 0; i < 100000; ++i) {
        auto pattern = CreateTestPattern(10);
        bool result = backend.Store(pattern);
        EXPECT_TRUE(result);
    }

    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (100000.0 / elapsed) * 1000.0;

    std::cout << "Large Scale Store (100k patterns): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;
    std::cout << "  Average per pattern: " << (elapsed / 100000.0) << "ms" << std::endl;

    EXPECT_LT(elapsed, 10000.0); // Should complete in < 10s
}

TEST(StorageScalabilityBenchmark, Retrieve_After_100k_Store) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store 100k patterns
    std::vector<PatternID> ids;
    ids.reserve(100000);

    for (size_t i = 0; i < 100000; ++i) {
        auto pattern = CreateTestPattern(10);
        bool result = backend.Store(pattern);
        ASSERT_TRUE(result);
        ids.push_back(pattern.GetID());
    }

    // Benchmark random retrieval from large dataset
    BenchmarkTimer timer;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> dist(0, ids.size() - 1);

    for (size_t i = 0; i < 10000; ++i) {
        size_t idx = dist(rng);
        auto pattern = backend.Retrieve(ids[idx]);
        EXPECT_TRUE(pattern.has_value());
    }

    double elapsed = timer.ElapsedMs();
    double ops_per_sec = (10000.0 / elapsed) * 1000.0;

    std::cout << "Random Retrieve from 100k dataset (10000 queries): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 1000.0); // Should still be fast
}
