// File: tests/storage/indices/temporal_index_test.cpp
#include "storage/indices/temporal_index.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <algorithm>

namespace dpan {
namespace {

// ============================================================================
// Basic Operations Tests
// ============================================================================

TEST(TemporalIndexTest, DefaultConstructorCreatesEmpty) {
    TemporalIndex index;
    EXPECT_EQ(0u, index.Size());
}

TEST(TemporalIndexTest, InsertSinglePattern) {
    TemporalIndex index;

    PatternID id = PatternID::Generate();
    Timestamp ts = Timestamp::Now();

    index.Insert(id, ts);

    EXPECT_EQ(1u, index.Size());
}

TEST(TemporalIndexTest, InsertMultiplePatterns) {
    TemporalIndex index;

    for (int i = 0; i < 10; ++i) {
        PatternID id = PatternID::Generate();
        Timestamp ts = Timestamp::Now();
        index.Insert(id, ts);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    EXPECT_EQ(10u, index.Size());
}

TEST(TemporalIndexTest, InsertDuplicateUpdatesTimestamp) {
    TemporalIndex index;

    PatternID id = PatternID::Generate();
    Timestamp ts1 = Timestamp::Now();
    index.Insert(id, ts1);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    Timestamp ts2 = Timestamp::Now();
    index.Insert(id, ts2);

    EXPECT_EQ(1u, index.Size());  // Still only one pattern

    auto retrieved_ts = index.GetTimestamp(id);
    ASSERT_TRUE(retrieved_ts.has_value());
    EXPECT_EQ(ts2, *retrieved_ts);  // Timestamp should be updated
}

TEST(TemporalIndexTest, RemoveExistingPattern) {
    TemporalIndex index;

    PatternID id = PatternID::Generate();
    Timestamp ts = Timestamp::Now();

    index.Insert(id, ts);
    EXPECT_EQ(1u, index.Size());

    bool result = index.Remove(id);
    EXPECT_TRUE(result);
    EXPECT_EQ(0u, index.Size());
}

TEST(TemporalIndexTest, RemoveNonExistentPatternFails) {
    TemporalIndex index;

    PatternID id = PatternID::Generate();
    bool result = index.Remove(id);

    EXPECT_FALSE(result);
}

TEST(TemporalIndexTest, GetTimestampReturnsCorrectValue) {
    TemporalIndex index;

    PatternID id = PatternID::Generate();
    Timestamp ts = Timestamp::Now();

    index.Insert(id, ts);

    auto retrieved = index.GetTimestamp(id);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(ts, *retrieved);
}

TEST(TemporalIndexTest, GetTimestampForNonExistentReturnsNullopt) {
    TemporalIndex index;

    PatternID id = PatternID::Generate();
    auto retrieved = index.GetTimestamp(id);

    EXPECT_FALSE(retrieved.has_value());
}

// ============================================================================
// Range Query Tests
// ============================================================================

TEST(TemporalIndexTest, FindInRangeReturnsMatchingPatterns) {
    TemporalIndex index;

    Timestamp start = Timestamp::Now();

    std::vector<PatternID> inserted_ids;
    for (int i = 0; i < 5; ++i) {
        PatternID id = PatternID::Generate();
        inserted_ids.push_back(id);
        index.Insert(id, Timestamp::Now());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    Timestamp end = Timestamp::Now();

    auto results = index.FindInRange(start, end);

    EXPECT_EQ(5u, results.size());
}

TEST(TemporalIndexTest, FindInRangeRespectsMaxResults) {
    TemporalIndex index;

    Timestamp start = Timestamp::Now();

    for (int i = 0; i < 10; ++i) {
        PatternID id = PatternID::Generate();
        index.Insert(id, Timestamp::Now());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    Timestamp end = Timestamp::Now();

    auto results = index.FindInRange(start, end, 5);

    EXPECT_EQ(5u, results.size());
}

TEST(TemporalIndexTest, FindInRangeReturnsChronologicalOrder) {
    TemporalIndex index;

    std::vector<std::pair<PatternID, Timestamp>> patterns;

    for (int i = 0; i < 5; ++i) {
        PatternID id = PatternID::Generate();
        Timestamp ts = Timestamp::Now();
        patterns.push_back({id, ts});
        index.Insert(id, ts);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto results = index.FindInRange(patterns.front().second, patterns.back().second);

    EXPECT_EQ(5u, results.size());

    // Verify chronological order
    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_EQ(patterns[i].first, results[i]);
    }
}

TEST(TemporalIndexTest, FindBeforeReturnsOlderPatterns) {
    TemporalIndex index;

    std::vector<PatternID> ids;
    std::vector<Timestamp> timestamps;

    for (int i = 0; i < 5; ++i) {
        PatternID id = PatternID::Generate();
        Timestamp ts = Timestamp::Now();
        ids.push_back(id);
        timestamps.push_back(ts);
        index.Insert(id, ts);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Find patterns before the last one
    auto results = index.FindBefore(timestamps.back());

    EXPECT_EQ(4u, results.size());  // Should get first 4 patterns
}

TEST(TemporalIndexTest, FindAfterReturnsNewerPatterns) {
    TemporalIndex index;

    std::vector<PatternID> ids;
    std::vector<Timestamp> timestamps;

    for (int i = 0; i < 5; ++i) {
        PatternID id = PatternID::Generate();
        Timestamp ts = Timestamp::Now();
        ids.push_back(id);
        timestamps.push_back(ts);
        index.Insert(id, ts);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Find patterns after the first one
    auto results = index.FindAfter(timestamps.front());

    EXPECT_EQ(4u, results.size());  // Should get last 4 patterns
}

TEST(TemporalIndexTest, FindMostRecentReturnsLatestPatterns) {
    TemporalIndex index;

    std::vector<PatternID> ids;

    for (int i = 0; i < 10; ++i) {
        PatternID id = PatternID::Generate();
        ids.push_back(id);
        index.Insert(id, Timestamp::Now());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    auto results = index.FindMostRecent(5);

    EXPECT_EQ(5u, results.size());

    // Most recent should be last inserted
    EXPECT_EQ(ids.back(), results.front());
}

TEST(TemporalIndexTest, FindOldestReturnsEarliestPatterns) {
    TemporalIndex index;

    std::vector<PatternID> ids;

    for (int i = 0; i < 10; ++i) {
        PatternID id = PatternID::Generate();
        ids.push_back(id);
        index.Insert(id, Timestamp::Now());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    auto results = index.FindOldest(5);

    EXPECT_EQ(5u, results.size());

    // Oldest should be first inserted
    EXPECT_EQ(ids.front(), results.front());
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(TemporalIndexTest, GetStatsReturnsValidData) {
    TemporalIndex index;

    for (int i = 0; i < 5; ++i) {
        PatternID id = PatternID::Generate();
        index.Insert(id, Timestamp::Now());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    auto stats = index.GetStats();

    EXPECT_EQ(5u, stats.total_patterns);
    EXPECT_GT(stats.latest.ToMicros(), stats.earliest.ToMicros());
    EXPECT_GT(stats.avg_patterns_per_second, 0.0);
}

TEST(TemporalIndexTest, ClearRemovesAllPatterns) {
    TemporalIndex index;

    for (int i = 0; i < 10; ++i) {
        PatternID id = PatternID::Generate();
        index.Insert(id, Timestamp::Now());
    }

    EXPECT_EQ(10u, index.Size());

    index.Clear();

    EXPECT_EQ(0u, index.Size());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(TemporalIndexTest, MultiplePatternsWithSameTimestamp) {
    TemporalIndex index;

    Timestamp ts = Timestamp::Now();

    std::vector<PatternID> ids;
    for (int i = 0; i < 5; ++i) {
        PatternID id = PatternID::Generate();
        ids.push_back(id);
        index.Insert(id, ts);
    }

    EXPECT_EQ(5u, index.Size());

    auto results = index.FindInRange(ts, ts);
    EXPECT_EQ(5u, results.size());
}

TEST(TemporalIndexTest, QueryOnEmptyIndex) {
    TemporalIndex index;

    Timestamp start = Timestamp::Now();
    Timestamp end = Timestamp::Now();

    auto results = index.FindInRange(start, end);
    EXPECT_EQ(0u, results.size());

    auto recent = index.FindMostRecent(10);
    EXPECT_EQ(0u, recent.size());

    auto oldest = index.FindOldest(10);
    EXPECT_EQ(0u, oldest.size());
}

TEST(TemporalIndexTest, RemoveAndReinsert) {
    TemporalIndex index;

    PatternID id = PatternID::Generate();
    Timestamp ts1 = Timestamp::Now();

    index.Insert(id, ts1);
    index.Remove(id);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    Timestamp ts2 = Timestamp::Now();
    index.Insert(id, ts2);

    auto retrieved = index.GetTimestamp(id);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(ts2, *retrieved);
}

// ============================================================================
// Concurrency Tests
// ============================================================================

TEST(TemporalIndexTest, ConcurrentInsertsAreSafe) {
    TemporalIndex index;

    const int num_threads = 5;
    const int patterns_per_thread = 100;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&index, patterns_per_thread]() {
            for (int i = 0; i < patterns_per_thread; ++i) {
                PatternID id = PatternID::Generate();
                index.Insert(id, Timestamp::Now());
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(num_threads * patterns_per_thread, index.Size());
}

TEST(TemporalIndexTest, ConcurrentReadsAreSafe) {
    TemporalIndex index;

    // Pre-populate
    for (int i = 0; i < 100; ++i) {
        PatternID id = PatternID::Generate();
        index.Insert(id, Timestamp::Now());
    }

    const int num_threads = 5;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&index]() {
            for (int i = 0; i < 50; ++i) {
                Timestamp start = Timestamp::Now();
                Timestamp end = Timestamp::Now();
                auto results = index.FindInRange(start, end);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

TEST(TemporalIndexTest, ConcurrentMixedOperationsAreSafe) {
    TemporalIndex index;

    const int num_threads = 4;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        if (t % 2 == 0) {
            // Writer threads
            threads.emplace_back([&index]() {
                for (int i = 0; i < 50; ++i) {
                    PatternID id = PatternID::Generate();
                    index.Insert(id, Timestamp::Now());
                }
            });
        } else {
            // Reader threads
            threads.emplace_back([&index]() {
                for (int i = 0; i < 50; ++i) {
                    index.FindMostRecent(10);
                    index.Size();
                }
            });
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_GT(index.Size(), 0u);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(TemporalIndexTest, InsertPerformance) {
    TemporalIndex index;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        PatternID id = PatternID::Generate();
        index.Insert(id, Timestamp::Now());
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    float avg_insert_us = static_cast<float>(duration.count()) / 1000.0f;

    // Should be less than 10 microseconds per insert on average
    EXPECT_LT(avg_insert_us, 10.0f);
}

TEST(TemporalIndexTest, RangeQueryPerformance) {
    TemporalIndex index;

    Timestamp start = Timestamp::Now();

    // Insert 10000 patterns
    for (int i = 0; i < 10000; ++i) {
        PatternID id = PatternID::Generate();
        index.Insert(id, Timestamp::Now());
    }

    Timestamp end = Timestamp::Now();

    auto query_start = std::chrono::high_resolution_clock::now();
    auto results = index.FindInRange(start, end, 1000);
    auto query_end = std::chrono::high_resolution_clock::now();

    auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - query_start);

    EXPECT_EQ(1000u, results.size());
    // Range query should be fast (< 10ms for 10k entries)
    EXPECT_LT(query_duration.count(), 10000);
}

} // namespace
} // namespace dpan
