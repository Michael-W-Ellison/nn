// File: tests/storage/lru_cache_test.cpp
#include "storage/lru_cache.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <string>

namespace dpan {
namespace {

// ============================================================================
// Basic Operations Tests
// ============================================================================

TEST(LRUCacheTest, ConstructorSetsCapacity) {
    LRUCache<int, std::string> cache(10);
    EXPECT_EQ(10u, cache.Capacity());
    EXPECT_EQ(0u, cache.Size());
}

TEST(LRUCacheTest, ZeroCapacitySetToOne) {
    LRUCache<int, int> cache(0);
    EXPECT_EQ(1u, cache.Capacity());
}

TEST(LRUCacheTest, PutAndGetSingleItem) {
    LRUCache<int, std::string> cache(5);

    cache.Put(1, "one");

    auto result = cache.Get(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ("one", *result);
}

TEST(LRUCacheTest, GetNonExistentReturnsNullopt) {
    LRUCache<int, std::string> cache(5);

    auto result = cache.Get(99);
    EXPECT_FALSE(result.has_value());
}

TEST(LRUCacheTest, PutMultipleItems) {
    LRUCache<int, std::string> cache(5);

    cache.Put(1, "one");
    cache.Put(2, "two");
    cache.Put(3, "three");

    EXPECT_EQ(3u, cache.Size());

    EXPECT_EQ("one", *cache.Get(1));
    EXPECT_EQ("two", *cache.Get(2));
    EXPECT_EQ("three", *cache.Get(3));
}

TEST(LRUCacheTest, UpdateExistingKey) {
    LRUCache<int, std::string> cache(5);

    cache.Put(1, "one");
    cache.Put(1, "ONE");

    auto result = cache.Get(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ("ONE", *result);
    EXPECT_EQ(1u, cache.Size());  // Should still be 1 item
}

TEST(LRUCacheTest, RemoveExistingItem) {
    LRUCache<int, std::string> cache(5);

    cache.Put(1, "one");
    EXPECT_EQ(1u, cache.Size());

    bool removed = cache.Remove(1);
    EXPECT_TRUE(removed);
    EXPECT_EQ(0u, cache.Size());

    auto result = cache.Get(1);
    EXPECT_FALSE(result.has_value());
}

TEST(LRUCacheTest, RemoveNonExistentReturnsFalse) {
    LRUCache<int, std::string> cache(5);

    bool removed = cache.Remove(99);
    EXPECT_FALSE(removed);
}

TEST(LRUCacheTest, ContainsReturnsTrueForExisting) {
    LRUCache<int, std::string> cache(5);

    cache.Put(1, "one");

    EXPECT_TRUE(cache.Contains(1));
    EXPECT_FALSE(cache.Contains(2));
}

TEST(LRUCacheTest, ClearRemovesAllItems) {
    LRUCache<int, std::string> cache(5);

    cache.Put(1, "one");
    cache.Put(2, "two");
    cache.Put(3, "three");

    EXPECT_EQ(3u, cache.Size());

    cache.Clear();

    EXPECT_EQ(0u, cache.Size());
    EXPECT_FALSE(cache.Get(1).has_value());
}

// ============================================================================
// LRU Eviction Tests
// ============================================================================

TEST(LRUCacheTest, EvictsLRUWhenFull) {
    LRUCache<int, std::string> cache(3);

    cache.Put(1, "one");
    cache.Put(2, "two");
    cache.Put(3, "three");

    EXPECT_EQ(3u, cache.Size());

    // Cache is full, adding a new item should evict the LRU (1)
    cache.Put(4, "four");

    EXPECT_EQ(3u, cache.Size());
    EXPECT_FALSE(cache.Get(1).has_value());  // 1 was evicted
    EXPECT_TRUE(cache.Get(2).has_value());
    EXPECT_TRUE(cache.Get(3).has_value());
    EXPECT_TRUE(cache.Get(4).has_value());
}

TEST(LRUCacheTest, AccessMakesItemMostRecent) {
    LRUCache<int, std::string> cache(3);

    cache.Put(1, "one");
    cache.Put(2, "two");
    cache.Put(3, "three");

    // Access 1 to make it most recent
    cache.Get(1);

    // Add new item, should evict 2 (oldest since 1 was accessed)
    cache.Put(4, "four");

    EXPECT_TRUE(cache.Get(1).has_value());   // Still there
    EXPECT_FALSE(cache.Get(2).has_value());  // Evicted
    EXPECT_TRUE(cache.Get(3).has_value());   // Still there
    EXPECT_TRUE(cache.Get(4).has_value());   // New item
}

TEST(LRUCacheTest, UpdateMakesItemMostRecent) {
    LRUCache<int, std::string> cache(3);

    cache.Put(1, "one");
    cache.Put(2, "two");
    cache.Put(3, "three");

    // Update 1 to make it most recent
    cache.Put(1, "ONE");

    // Add new item, should evict 2
    cache.Put(4, "four");

    EXPECT_TRUE(cache.Get(1).has_value());   // Still there
    EXPECT_FALSE(cache.Get(2).has_value());  // Evicted
    EXPECT_TRUE(cache.Get(3).has_value());   // Still there
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(LRUCacheTest, HitsAndMissesTracked) {
    LRUCache<int, std::string> cache(5);

    cache.Put(1, "one");

    cache.Get(1);  // Hit
    cache.Get(2);  // Miss
    cache.Get(1);  // Hit
    cache.Get(3);  // Miss

    EXPECT_EQ(2u, cache.Hits());
    EXPECT_EQ(2u, cache.Misses());
}

TEST(LRUCacheTest, HitRateCalculatedCorrectly) {
    LRUCache<int, std::string> cache(5);

    cache.Put(1, "one");
    cache.Put(2, "two");

    cache.Get(1);  // Hit
    cache.Get(2);  // Hit
    cache.Get(3);  // Miss
    cache.Get(4);  // Miss

    float hit_rate = cache.HitRate();
    EXPECT_FLOAT_EQ(0.5f, hit_rate);  // 2 hits out of 4 total
}

TEST(LRUCacheTest, HitRateZeroWhenNoAccess) {
    LRUCache<int, std::string> cache(5);

    EXPECT_FLOAT_EQ(0.0f, cache.HitRate());
}

TEST(LRUCacheTest, EvictionsTracked) {
    LRUCache<int, std::string> cache(2);

    cache.Put(1, "one");
    cache.Put(2, "two");

    EXPECT_EQ(0u, cache.Evictions());

    cache.Put(3, "three");  // Evicts 1

    EXPECT_EQ(1u, cache.Evictions());

    cache.Put(4, "four");   // Evicts 2

    EXPECT_EQ(2u, cache.Evictions());
}

TEST(LRUCacheTest, GetStatsReturnsComprehensiveInfo) {
    LRUCache<int, std::string> cache(5);

    cache.Put(1, "one");
    cache.Put(2, "two");
    cache.Put(3, "three");

    cache.Get(1);  // Hit
    cache.Get(4);  // Miss

    auto stats = cache.GetStats();

    EXPECT_EQ(3u, stats.size);
    EXPECT_EQ(5u, stats.capacity);
    EXPECT_EQ(1u, stats.hits);
    EXPECT_EQ(1u, stats.misses);
    EXPECT_EQ(0u, stats.evictions);
    EXPECT_FLOAT_EQ(0.5f, stats.hit_rate);
    EXPECT_FLOAT_EQ(0.6f, stats.utilization);  // 3/5
}

TEST(LRUCacheTest, ClearResetsStatistics) {
    LRUCache<int, std::string> cache(5);

    cache.Put(1, "one");
    cache.Get(1);
    cache.Get(2);

    cache.Clear();

    EXPECT_EQ(0u, cache.Hits());
    EXPECT_EQ(0u, cache.Misses());
    EXPECT_EQ(0u, cache.Evictions());
}

// ============================================================================
// Different Types Tests
// ============================================================================

TEST(LRUCacheTest, WorksWithStringKeys) {
    LRUCache<std::string, int> cache(5);

    cache.Put("one", 1);
    cache.Put("two", 2);

    auto result = cache.Get("one");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(1, *result);
}

TEST(LRUCacheTest, WorksWithComplexValues) {
    struct ComplexValue {
        int x;
        std::string y;

        bool operator==(const ComplexValue& other) const {
            return x == other.x && y == other.y;
        }
    };

    LRUCache<int, ComplexValue> cache(5);

    cache.Put(1, {42, "test"});

    auto result = cache.Get(1);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(42, result->x);
    EXPECT_EQ("test", result->y);
}

// ============================================================================
// Concurrency Tests
// ============================================================================

TEST(LRUCacheTest, ConcurrentPutsAreSafe) {
    LRUCache<int, int> cache(1000);

    const int num_threads = 10;
    const int items_per_thread = 100;

    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&cache, t, items_per_thread]() {
            for (int i = 0; i < items_per_thread; ++i) {
                int key = t * items_per_thread + i;
                cache.Put(key, key * 2);
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(1000u, cache.Size());
}

TEST(LRUCacheTest, ConcurrentGetsAreSafe) {
    LRUCache<int, int> cache(100);

    // Pre-populate
    for (int i = 0; i < 100; ++i) {
        cache.Put(i, i * 2);
    }

    const int num_threads = 10;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&cache]() {
            for (int i = 0; i < 100; ++i) {
                auto result = cache.Get(i);
                EXPECT_TRUE(result.has_value());
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

TEST(LRUCacheTest, ConcurrentMixedOperationsAreSafe) {
    LRUCache<int, int> cache(100);

    const int num_threads = 8;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        if (t % 2 == 0) {
            // Writer threads
            threads.emplace_back([&cache, t]() {
                for (int i = 0; i < 50; ++i) {
                    cache.Put(t * 50 + i, i);
                }
            });
        } else {
            // Reader threads
            threads.emplace_back([&cache]() {
                for (int i = 0; i < 50; ++i) {
                    cache.Get(i);
                }
            });
        }
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_GT(cache.Size(), 0u);
}

// ============================================================================
// Performance Tests
// ============================================================================

TEST(LRUCacheTest, PutPerformance) {
    LRUCache<int, int> cache(10000);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; ++i) {
        cache.Put(i, i);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    float avg_put_us = static_cast<float>(duration.count()) / 10000.0f;

    // Should be very fast (< 1 microsecond per put on average)
    EXPECT_LT(avg_put_us, 1.0f);
}

TEST(LRUCacheTest, GetPerformance) {
    LRUCache<int, int> cache(10000);

    // Pre-populate
    for (int i = 0; i < 10000; ++i) {
        cache.Put(i, i);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; ++i) {
        cache.Get(i);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    float avg_get_us = static_cast<float>(duration.count()) / 10000.0f;

    // Should be very fast (< 1 microsecond per get on average)
    EXPECT_LT(avg_get_us, 1.0f);
}

TEST(LRUCacheTest, HighHitRateOnRepeatedAccess) {
    LRUCache<int, int> cache(100);

    // Populate cache
    for (int i = 0; i < 100; ++i) {
        cache.Put(i, i);
    }

    // Access same keys repeatedly
    for (int round = 0; round < 10; ++round) {
        for (int i = 0; i < 100; ++i) {
            cache.Get(i);
        }
    }

    float hit_rate = cache.HitRate();

    // Should have very high hit rate (> 99%)
    EXPECT_GT(hit_rate, 0.99f);
}

} // namespace
} // namespace dpan
