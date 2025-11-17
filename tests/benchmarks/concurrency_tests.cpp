// File: tests/benchmarks/concurrency_tests.cpp
//
// Concurrency and thread safety tests

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <chrono>
#include "association/association_matrix.hpp"
#include "association/association_learning_system.hpp"
#include "association/co_occurrence_tracker.hpp"
#include "storage/memory_backend.hpp"
#include "core/pattern_node.hpp"

using namespace dpan;
using namespace std::chrono;

// ============================================================================
// Helper Functions
// ============================================================================

PatternNode CreateTestPattern(size_t id) {
    std::vector<float> data(10);
    for (size_t i = 0; i < 10; ++i) {
        data[i] = static_cast<float>(id * 10 + i) / 100.0f;
    }
    FeatureVector features(data);
    PatternData pattern_data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    return PatternNode(PatternID::Generate(), pattern_data, PatternType::ATOMIC);
}

// ============================================================================
// AssociationMatrix Concurrency Tests
// ============================================================================

TEST(AssociationMatrixConcurrencyTest, ConcurrentReads) {
    AssociationMatrix matrix;

    // Setup: Add associations
    const size_t num_patterns = 100;
    std::vector<PatternID> patterns;
    for (size_t i = 0; i < num_patterns; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    for (size_t i = 0; i < num_patterns; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            size_t tgt_idx = (i + j + 1) % num_patterns;
            AssociationEdge edge(patterns[i], patterns[tgt_idx],
                               AssociationType::CATEGORICAL, 0.5f);
            matrix.AddAssociation(edge);
        }
    }

    // Test: Concurrent reads from multiple threads
    const size_t num_threads = 10;
    const size_t reads_per_thread = 1000;

    std::atomic<size_t> total_reads{0};
    std::vector<std::thread> threads;

    auto start = high_resolution_clock::now();

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < reads_per_thread; ++i) {
                size_t idx = (t * reads_per_thread + i) % num_patterns;
                auto assocs = matrix.GetOutgoingAssociations(patterns[idx]);
                EXPECT_EQ(5u, assocs.size());
                total_reads++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start).count();

    std::cout << "Concurrent Reads (" << num_threads << " threads, " << reads_per_thread << " reads each):" << std::endl;
    std::cout << "  Time: " << elapsed_ms << "ms" << std::endl;
    std::cout << "  Total reads: " << total_reads << std::endl;
    std::cout << "  Throughput: " << (total_reads * 1000.0 / elapsed_ms) << " reads/sec" << std::endl;

    EXPECT_EQ(num_threads * reads_per_thread, total_reads);
    EXPECT_LT(elapsed_ms, 5000); // Should complete quickly
}

TEST(AssociationMatrixConcurrencyTest, ConcurrentWrites) {
    AssociationMatrix matrix;

    const size_t num_threads = 10;
    const size_t writes_per_thread = 100;

    // Pre-generate patterns for each thread
    std::vector<std::vector<PatternID>> thread_patterns(num_threads);
    for (size_t t = 0; t < num_threads; ++t) {
        for (size_t i = 0; i < writes_per_thread * 2; ++i) {
            thread_patterns[t].push_back(PatternID::Generate());
        }
    }

    std::atomic<size_t> successful_writes{0};
    std::vector<std::thread> threads;

    auto start = high_resolution_clock::now();

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < writes_per_thread; ++i) {
                AssociationEdge edge(thread_patterns[t][i * 2],
                                   thread_patterns[t][i * 2 + 1],
                                   AssociationType::CATEGORICAL, 0.5f);
                if (matrix.AddAssociation(edge)) {
                    successful_writes++;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start).count();

    std::cout << "Concurrent Writes (" << num_threads << " threads, " << writes_per_thread << " writes each):" << std::endl;
    std::cout << "  Time: " << elapsed_ms << "ms" << std::endl;
    std::cout << "  Successful writes: " << successful_writes << std::endl;
    std::cout << "  Throughput: " << (successful_writes * 1000.0 / elapsed_ms) << " writes/sec" << std::endl;

    EXPECT_EQ(num_threads * writes_per_thread, successful_writes);
    EXPECT_LT(elapsed_ms, 5000);
}

TEST(AssociationMatrixConcurrencyTest, MixedReadWrite) {
    AssociationMatrix matrix;

    // Setup initial data
    const size_t initial_patterns = 100;
    std::vector<PatternID> patterns;
    for (size_t i = 0; i < initial_patterns; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    for (size_t i = 0; i < initial_patterns; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            size_t tgt_idx = (i + j + 1) % initial_patterns;
            AssociationEdge edge(patterns[i], patterns[tgt_idx],
                               AssociationType::CATEGORICAL, 0.5f);
            matrix.AddAssociation(edge);
        }
    }

    // Mixed workload: some threads read, some write
    const size_t num_reader_threads = 8;
    const size_t num_writer_threads = 2;
    const size_t ops_per_thread = 500;

    std::atomic<size_t> reads{0}, writes{0};
    std::vector<std::thread> threads;

    auto start = high_resolution_clock::now();

    // Reader threads
    for (size_t t = 0; t < num_reader_threads; ++t) {
        threads.emplace_back([&]() {
            for (size_t i = 0; i < ops_per_thread; ++i) {
                size_t idx = i % patterns.size();
                auto assocs = matrix.GetOutgoingAssociations(patterns[idx]);
                reads++;
            }
        });
    }

    // Writer threads
    for (size_t t = 0; t < num_writer_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < ops_per_thread; ++i) {
                size_t src_idx = (t * ops_per_thread + i) % patterns.size();
                size_t tgt_idx = (src_idx + 1) % patterns.size();
                AssociationEdge edge(patterns[src_idx], patterns[tgt_idx],
                                   AssociationType::CATEGORICAL, 0.7f);
                matrix.UpdateAssociation(patterns[src_idx], patterns[tgt_idx], edge);
                writes++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start).count();

    std::cout << "Mixed Read/Write (" << num_reader_threads << " readers, " << num_writer_threads << " writers):" << std::endl;
    std::cout << "  Time: " << elapsed_ms << "ms" << std::endl;
    std::cout << "  Reads: " << reads << ", Writes: " << writes << std::endl;
    std::cout << "  Total throughput: " << ((reads + writes) * 1000.0 / elapsed_ms) << " ops/sec" << std::endl;

    EXPECT_EQ(num_reader_threads * ops_per_thread, reads);
    EXPECT_EQ(num_writer_threads * ops_per_thread, writes);
}

// ============================================================================
// CoOccurrenceTracker Concurrency Tests
// ============================================================================

TEST(CoOccurrenceTrackerConcurrencyTest, ConcurrentActivations) {
    CoOccurrenceTracker::Config config;
    config.window_size = std::chrono::seconds(10);
    CoOccurrenceTracker tracker(config);

    const size_t num_threads = 10;
    const size_t activations_per_thread = 1000;

    std::vector<PatternID> patterns;
    for (size_t i = 0; i < 100; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    std::atomic<size_t> total_activations{0};
    std::vector<std::thread> threads;

    auto start = high_resolution_clock::now();

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < activations_per_thread; ++i) {
                size_t idx = (t * activations_per_thread + i) % patterns.size();
                tracker.RecordActivation(patterns[idx], Timestamp::Now());
                total_activations++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start).count();

    std::cout << "Concurrent Activations (" << num_threads << " threads):" << std::endl;
    std::cout << "  Time: " << elapsed_ms << "ms" << std::endl;
    std::cout << "  Total activations: " << total_activations << std::endl;
    std::cout << "  Throughput: " << (total_activations * 1000.0 / elapsed_ms) << " activations/sec" << std::endl;

    EXPECT_EQ(num_threads * activations_per_thread, total_activations);
}

// ============================================================================
// Storage Concurrency Tests
// ============================================================================

TEST(MemoryBackendConcurrencyTest, ConcurrentStores) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    const size_t num_threads = 10;
    const size_t stores_per_thread = 100;

    std::atomic<size_t> successful_stores{0};
    std::vector<std::thread> threads;
    std::mutex ids_mutex;
    std::vector<PatternID> all_ids;

    auto start = high_resolution_clock::now();

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            std::vector<PatternID> thread_ids;
            for (size_t i = 0; i < stores_per_thread; ++i) {
                auto pattern = CreateTestPattern(t * stores_per_thread + i);
                bool result = backend.Store(pattern);
                if (result) {
                    thread_ids.push_back(pattern.GetID());
                    successful_stores++;
                }
            }

            std::lock_guard<std::mutex> lock(ids_mutex);
            all_ids.insert(all_ids.end(), thread_ids.begin(), thread_ids.end());
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start).count();

    std::cout << "Concurrent Stores (" << num_threads << " threads, " << stores_per_thread << " each):" << std::endl;
    std::cout << "  Time: " << elapsed_ms << "ms" << std::endl;
    std::cout << "  Successful stores: " << successful_stores << std::endl;
    std::cout << "  Throughput: " << (successful_stores * 1000.0 / elapsed_ms) << " stores/sec" << std::endl;

    EXPECT_EQ(num_threads * stores_per_thread, successful_stores);
    EXPECT_EQ(all_ids.size(), successful_stores);
}

TEST(MemoryBackendConcurrencyTest, ConcurrentRetrievals) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Setup: Store patterns
    std::vector<PatternID> ids;
    for (size_t i = 0; i < 100; ++i) {
        auto pattern = CreateTestPattern(i);
        bool result = backend.Store(pattern);
        ASSERT_TRUE(result);
        ids.push_back(pattern.GetID());
    }

    // Test: Concurrent retrievals
    const size_t num_threads = 10;
    const size_t retrievals_per_thread = 1000;

    std::atomic<size_t> successful_retrievals{0};
    std::vector<std::thread> threads;

    auto start = high_resolution_clock::now();

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            for (size_t i = 0; i < retrievals_per_thread; ++i) {
                size_t idx = i % ids.size();
                auto pattern = backend.Retrieve(ids[idx]);
                if (pattern.has_value()) {
                    successful_retrievals++;
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start).count();

    std::cout << "Concurrent Retrievals (" << num_threads << " threads):" << std::endl;
    std::cout << "  Time: " << elapsed_ms << "ms" << std::endl;
    std::cout << "  Successful retrievals: " << successful_retrievals << std::endl;
    std::cout << "  Throughput: " << (successful_retrievals * 1000.0 / elapsed_ms) << " retrievals/sec" << std::endl;

    EXPECT_EQ(num_threads * retrievals_per_thread, successful_retrievals);
}

TEST(MemoryBackendConcurrencyTest, MixedStorageOperations) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Setup: Store initial patterns
    std::vector<PatternID> ids;
    std::mutex ids_mutex;

    for (size_t i = 0; i < 100; ++i) {
        auto pattern = CreateTestPattern(i);
        bool result = backend.Store(pattern);
        ASSERT_TRUE(result);
        ids.push_back(pattern.GetID());
    }

    // Test: Mixed operations
    const size_t num_threads = 8;
    const size_t ops_per_thread = 100;

    std::atomic<size_t> stores{0}, retrievals{0}, updates{0};
    std::vector<std::thread> threads;

    auto start = high_resolution_clock::now();

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            for (size_t i = 0; i < ops_per_thread; ++i) {
                size_t op = (t + i) % 3;

                if (op == 0) { // Store
                    auto pattern = CreateTestPattern(t * ops_per_thread + i + 1000);
                    bool result = backend.Store(pattern);
                    if (result) {
                        std::lock_guard<std::mutex> lock(ids_mutex);
                        ids.push_back(pattern.GetID());
                        stores++;
                    }
                } else if (op == 1) { // Retrieve
                    std::lock_guard<std::mutex> lock(ids_mutex);
                    if (!ids.empty()) {
                        size_t idx = i % ids.size();
                        auto pattern = backend.Retrieve(ids[idx]);
                        if (pattern.has_value()) {
                            retrievals++;
                        }
                    }
                } else { // Update
                    std::lock_guard<std::mutex> lock(ids_mutex);
                    if (!ids.empty()) {
                        size_t idx = i % ids.size();
                        size_t pattern_id_val = t * ops_per_thread + i + 2000;
                        std::vector<float> data(10);
                        for (size_t j = 0; j < 10; ++j) {
                            data[j] = static_cast<float>(pattern_id_val * 10 + j) / 100.0f;
                        }
                        FeatureVector features(data);
                        PatternData pattern_data = PatternData::FromFeatures(features, DataModality::NUMERIC);
                        PatternNode pattern(ids[idx], pattern_data, PatternType::ATOMIC);
                        backend.Update(pattern);
                        updates++;
                    }
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start).count();

    std::cout << "Mixed Storage Operations (" << num_threads << " threads):" << std::endl;
    std::cout << "  Time: " << elapsed_ms << "ms" << std::endl;
    std::cout << "  Stores: " << stores << ", Retrievals: " << retrievals << ", Updates: " << updates << std::endl;
    std::cout << "  Total throughput: " << ((stores + retrievals + updates) * 1000.0 / elapsed_ms) << " ops/sec" << std::endl;

    EXPECT_GT(stores + retrievals + updates, 0u);
}

// ============================================================================
// Learning System Concurrency Tests
// ============================================================================

TEST(LearningSystemConcurrencyTest, ConcurrentActivationRecording) {
    AssociationLearningSystem::Config config;
    config.enable_auto_maintenance = false;
    AssociationLearningSystem system(config);

    std::vector<PatternID> patterns;
    for (size_t i = 0; i < 100; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    const size_t num_threads = 10;
    const size_t activations_per_thread = 1000;

    std::atomic<size_t> total_activations{0};
    std::vector<std::thread> threads;

    auto start = high_resolution_clock::now();

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            ContextVector context;
            for (size_t i = 0; i < activations_per_thread; ++i) {
                size_t idx = i % patterns.size();
                system.RecordPatternActivation(patterns[idx], context);
                total_activations++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start).count();

    std::cout << "Concurrent Activation Recording (" << num_threads << " threads):" << std::endl;
    std::cout << "  Time: " << elapsed_ms << "ms" << std::endl;
    std::cout << "  Total activations: " << total_activations << std::endl;
    std::cout << "  Throughput: " << (total_activations * 1000.0 / elapsed_ms) << " activations/sec" << std::endl;

    EXPECT_EQ(num_threads * activations_per_thread, total_activations);
}

TEST(LearningSystemConcurrencyTest, ConcurrentPredictions) {
    AssociationLearningSystem system;

    std::vector<PatternID> patterns;
    for (size_t i = 0; i < 100; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    // Setup associations
    auto& matrix = const_cast<AssociationMatrix&>(system.GetAssociationMatrix());
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = 0; j < 5; ++j) {
            size_t tgt_idx = (i + j + 1) % patterns.size();
            AssociationEdge edge(patterns[i], patterns[tgt_idx],
                               AssociationType::CATEGORICAL, 0.5f);
            matrix.AddAssociation(edge);
        }
    }

    // Concurrent predictions
    const size_t num_threads = 10;
    const size_t predictions_per_thread = 1000;

    std::atomic<size_t> total_predictions{0};
    std::vector<std::thread> threads;

    auto start = high_resolution_clock::now();

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
            for (size_t i = 0; i < predictions_per_thread; ++i) {
                size_t idx = i % patterns.size();
                auto preds = system.Predict(patterns[idx], 3);
                total_predictions++;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto end = high_resolution_clock::now();
    auto elapsed_ms = duration_cast<milliseconds>(end - start).count();

    std::cout << "Concurrent Predictions (" << num_threads << " threads):" << std::endl;
    std::cout << "  Time: " << elapsed_ms << "ms" << std::endl;
    std::cout << "  Total predictions: " << total_predictions << std::endl;
    std::cout << "  Throughput: " << (total_predictions * 1000.0 / elapsed_ms) << " predictions/sec" << std::endl;

    EXPECT_EQ(num_threads * predictions_per_thread, total_predictions);
}

// ============================================================================
// Race Condition Detection Tests
// ============================================================================

TEST(RaceConditionTest, NoDataRaceInMatrix) {
    // This test attempts to trigger race conditions
    AssociationMatrix matrix;

    std::vector<PatternID> patterns;
    for (size_t i = 0; i < 50; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    // Add some initial associations
    for (size_t i = 0; i < 25; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            AssociationEdge edge(patterns[i], patterns[i + j + 1],
                               AssociationType::CATEGORICAL, 0.5f);
            matrix.AddAssociation(edge);
        }
    }

    // Concurrent mixed operations on overlapping data
    const size_t num_threads = 20;
    std::vector<std::thread> threads;
    std::atomic<bool> error_detected{false};

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            try {
                for (size_t i = 0; i < 100; ++i) {
                    size_t idx = (t + i) % 25;

                    // Mix of operations
                    auto assocs = matrix.GetOutgoingAssociations(patterns[idx]);

                    if (i % 3 == 0) {
                        AssociationEdge edge(patterns[idx], patterns[(idx + 1) % patterns.size()],
                                           AssociationType::CATEGORICAL, 0.6f);
                        matrix.UpdateAssociation(patterns[idx], patterns[(idx + 1) % patterns.size()], edge);
                    }

                    auto* edge = matrix.GetAssociation(patterns[idx], patterns[(idx + 1) % patterns.size()]);
                    (void)edge;
                }
            } catch (...) {
                error_detected = true;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_FALSE(error_detected) << "Data race or exception detected!";
}
