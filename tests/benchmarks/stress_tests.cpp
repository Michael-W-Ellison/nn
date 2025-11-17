// File: tests/benchmarks/stress_tests.cpp
//
// Stress tests for large-scale data and edge cases

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include "association/association_matrix.hpp"
#include "association/association_learning_system.hpp"
#include "storage/memory_backend.hpp"
#include "core/pattern_node.hpp"

using namespace dpan;
using namespace std::chrono;

// ============================================================================
// Helper Functions
// ============================================================================

struct StressTestTimer {
    using TimePoint = high_resolution_clock::time_point;
    TimePoint start;

    StressTestTimer() : start(high_resolution_clock::now()) {}

    double ElapsedMs() const {
        auto end = high_resolution_clock::now();
        return duration_cast<duration<double, std::milli>>(end - start).count();
    }
};

PatternNode CreateRandomPattern(size_t size, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dist(rng);
    }
    FeatureVector features(data);
    PatternData pattern_data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    return PatternNode(PatternID::Generate(), pattern_data, PatternType::ATOMIC);
}

// ============================================================================
// Association Matrix Stress Tests
// ============================================================================

TEST(AssociationMatrixStressTest, Million_Associations) {
    AssociationMatrix matrix;
    std::mt19937 rng(42);

    // Generate pattern IDs
    const size_t num_patterns = 10000;
    std::vector<PatternID> patterns;
    patterns.reserve(num_patterns);
    for (size_t i = 0; i < num_patterns; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    StressTestTimer timer;
    size_t added = 0;

    // Add 1 million associations (this will take a while)
    std::uniform_int_distribution<size_t> dist(0, patterns.size() - 1);

    for (size_t i = 0; i < 1000000; ++i) {
        size_t src_idx = dist(rng);
        size_t tgt_idx = dist(rng);

        if (src_idx == tgt_idx) continue;

        std::uniform_real_distribution<float> strength_dist(0.1f, 0.9f);
        AssociationEdge edge(patterns[src_idx], patterns[tgt_idx],
                           AssociationType::CATEGORICAL, strength_dist(rng));

        if (matrix.AddAssociation(edge)) {
            added++;
        }

        // Progress report every 100k
        if ((i + 1) % 100000 == 0) {
            std::cout << "  Progress: " << (i + 1) << " / 1000000" << std::endl;
        }
    }

    double elapsed = timer.ElapsedMs();

    std::cout << "Million Association Stress Test:" << std::endl;
    std::cout << "  Time: " << elapsed << "ms (" << (elapsed / 1000.0) << "s)" << std::endl;
    std::cout << "  Added: " << added << " unique associations" << std::endl;
    std::cout << "  Rate: " << (added / elapsed) * 1000.0 << " ops/sec" << std::endl;
    std::cout << "  Matrix size: " << matrix.GetAssociationCount() << std::endl;

    // Verify we can still query efficiently
    StressTestTimer query_timer;
    for (size_t i = 0; i < 1000; ++i) {
        size_t idx = i % patterns.size();
        auto assocs = matrix.GetOutgoingAssociations(patterns[idx]);
        (void)assocs;
    }
    double query_elapsed = query_timer.ElapsedMs();

    std::cout << "  Query performance (1000 queries): " << query_elapsed << "ms" << std::endl;

    EXPECT_GT(added, 900000u); // Should add most associations
    EXPECT_LT(query_elapsed, 500.0); // Queries should still be fast
}

TEST(AssociationMatrixStressTest, DenseConnectivity_1000x1000) {
    AssociationMatrix matrix;

    const size_t num_patterns = 1000;
    std::vector<PatternID> patterns;
    patterns.reserve(num_patterns);
    for (size_t i = 0; i < num_patterns; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    StressTestTimer timer;

    // Create dense connectivity (each pattern connects to many others)
    size_t added = 0;
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = 0; j < patterns.size(); ++j) {
            if (i == j) continue;

            // Connect to every 10th pattern to avoid timeout
            if (j % 10 == 0) {
                float strength = 0.5f;
                AssociationEdge edge(patterns[i], patterns[j],
                                   AssociationType::CATEGORICAL, strength);
                if (matrix.AddAssociation(edge)) {
                    added++;
                }
            }
        }

        if ((i + 1) % 100 == 0) {
            std::cout << "  Progress: " << (i + 1) << " / " << num_patterns << std::endl;
        }
    }

    double elapsed = timer.ElapsedMs();

    std::cout << "Dense Connectivity Stress Test (1000x100 avg):" << std::endl;
    std::cout << "  Time: " << elapsed << "ms" << std::endl;
    std::cout << "  Associations added: " << added << std::endl;
    std::cout << "  Average per pattern: " << (added / (double)num_patterns) << std::endl;

    EXPECT_GT(added, 90000u); // Should have many associations
    EXPECT_LT(elapsed, 10000.0); // Should complete in reasonable time
}

TEST(AssociationMatrixStressTest, HighFrequencyUpdates) {
    AssociationMatrix matrix;

    const size_t num_patterns = 100;
    std::vector<PatternID> patterns;
    for (size_t i = 0; i < num_patterns; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    // Add initial associations
    for (size_t i = 0; i < num_patterns; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            size_t tgt_idx = (i + j + 1) % patterns.size();
            AssociationEdge edge(patterns[i], patterns[tgt_idx],
                               AssociationType::CATEGORICAL, 0.5f);
            matrix.AddAssociation(edge);
        }
    }

    // Stress test with rapid updates
    StressTestTimer timer;
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> idx_dist(0, num_patterns - 1);
    std::uniform_real_distribution<float> strength_dist(0.1f, 0.9f);

    for (size_t i = 0; i < 100000; ++i) {
        size_t src_idx = idx_dist(rng);
        size_t tgt_idx = (src_idx + 1 + (i % 10)) % patterns.size();

        AssociationEdge edge(patterns[src_idx], patterns[tgt_idx],
                           AssociationType::CATEGORICAL, strength_dist(rng));
        matrix.UpdateAssociation(patterns[src_idx], patterns[tgt_idx], edge);
    }

    double elapsed = timer.ElapsedMs();

    std::cout << "High Frequency Updates (100k):" << std::endl;
    std::cout << "  Time: " << elapsed << "ms" << std::endl;
    std::cout << "  Rate: " << (100000.0 / elapsed) * 1000.0 << " updates/sec" << std::endl;

    EXPECT_LT(elapsed, 5000.0); // Should handle rapid updates
}

// ============================================================================
// Storage Stress Tests
// ============================================================================

TEST(StorageStressTest, Store_500k_Patterns) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);
    std::mt19937 rng(42);

    StressTestTimer timer;

    for (size_t i = 0; i < 500000; ++i) {
        auto pattern = CreateRandomPattern(20, rng);
        bool result = backend.Store(pattern);
        EXPECT_TRUE(result);

        if ((i + 1) % 50000 == 0) {
            std::cout << "  Progress: " << (i + 1) << " / 500000" << std::endl;
        }
    }

    double elapsed = timer.ElapsedMs();

    std::cout << "Storage 500k Patterns:" << std::endl;
    std::cout << "  Time: " << elapsed << "ms (" << (elapsed / 1000.0) << "s)" << std::endl;
    std::cout << "  Rate: " << (500000.0 / elapsed) * 1000.0 << " ops/sec" << std::endl;

    auto stats = backend.GetStats();
    std::cout << "  Total patterns: " << stats.total_patterns << std::endl;

    EXPECT_EQ(500000u, stats.total_patterns);
}

TEST(StorageStressTest, MixedOperations_100k) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);
    std::mt19937 rng(42);
    std::vector<PatternID> ids;

    // Store initial patterns
    for (size_t i = 0; i < 10000; ++i) {
        auto pattern = CreateRandomPattern(20, rng);
        bool result = backend.Store(pattern);
        ASSERT_TRUE(result);
        ids.push_back(pattern.GetID());
    }

    // Mixed operations: store, retrieve, update, delete
    StressTestTimer timer;
    std::uniform_int_distribution<int> op_dist(0, 3);
    std::uniform_int_distribution<size_t> idx_dist(0, ids.size() - 1);

    size_t stores = 0, retrieves = 0, updates = 0, deletes = 0;

    for (size_t i = 0; i < 100000; ++i) {
        int op = op_dist(rng);

        switch (op) {
            case 0: { // Store
                auto pattern = CreateRandomPattern(20, rng);
                bool result = backend.Store(pattern);
                if (result) {
                    ids.push_back(pattern.GetID());
                    stores++;
                }
                break;
            }
            case 1: { // Retrieve
                if (!ids.empty()) {
                    size_t idx = idx_dist(rng) % ids.size();
                    auto pattern = backend.Retrieve(ids[idx]);
                    retrieves++;
                }
                break;
            }
            case 2: { // Update
                if (!ids.empty()) {
                    size_t idx = idx_dist(rng) % ids.size();
                    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                    std::vector<float> data(25);
                    for (size_t j = 0; j < 25; ++j) {
                        data[j] = dist(rng);
                    }
                    FeatureVector features(data);
                    PatternData pattern_data = PatternData::FromFeatures(features, DataModality::NUMERIC);
                    PatternNode pattern(ids[idx], pattern_data, PatternType::ATOMIC);
                    backend.Update(pattern);
                    updates++;
                }
                break;
            }
            case 3: { // Delete (occasionally)
                if (ids.size() > 5000) {
                    size_t idx = idx_dist(rng) % ids.size();
                    backend.Delete(ids[idx]);
                    ids.erase(ids.begin() + idx);
                    deletes++;
                }
                break;
            }
        }
    }

    double elapsed = timer.ElapsedMs();

    std::cout << "Mixed Operations Stress Test (100k ops):" << std::endl;
    std::cout << "  Time: " << elapsed << "ms" << std::endl;
    std::cout << "  Operations: stores=" << stores << ", retrieves=" << retrieves
              << ", updates=" << updates << ", deletes=" << deletes << std::endl;
    std::cout << "  Final pattern count: " << backend.GetStats().total_patterns << std::endl;

    EXPECT_LT(elapsed, 10000.0); // Should handle mixed workload
}

// ============================================================================
// Learning System Stress Tests
// ============================================================================

TEST(LearningSystemStressTest, ContinuousLearning_100k_Activations) {
    AssociationLearningSystem::Config config;
    config.enable_auto_maintenance = true;
    config.auto_decay_interval = std::chrono::seconds(5);
    AssociationLearningSystem system(config);

    std::mt19937 rng(42);
    std::vector<PatternID> patterns;

    // Generate patterns
    for (size_t i = 0; i < 1000; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    StressTestTimer timer;
    ContextVector context;

    // Simulate continuous learning
    for (size_t i = 0; i < 100000; ++i) {
        // Record activations
        std::vector<PatternID> batch;
        for (size_t j = 0; j < 5; ++j) {
            batch.push_back(patterns[(i + j) % patterns.size()]);
        }
        system.RecordPatternActivations(batch, context);

        // Occasionally form associations
        if (i % 100 == 0) {
            for (size_t j = 0; j < batch.size() - 1; ++j) {
                auto& matrix = const_cast<AssociationMatrix&>(system.GetAssociationMatrix());
                AssociationEdge edge(batch[j], batch[j + 1],
                                   AssociationType::CAUSAL, 0.5f);
                matrix.AddAssociation(edge);
            }
        }

        // Occasionally reinforce
        if (i % 50 == 0 && i > 0) {
            size_t idx = i % patterns.size();
            size_t next_idx = (i + 1) % patterns.size();
            system.Reinforce(patterns[idx], patterns[next_idx], true);
        }

        if ((i + 1) % 10000 == 0) {
            std::cout << "  Progress: " << (i + 1) << " / 100000, "
                      << "Associations: " << system.GetAssociationCount() << std::endl;
        }
    }

    double elapsed = timer.ElapsedMs();

    auto stats = system.GetStatistics();

    std::cout << "Continuous Learning Stress Test (100k activations):" << std::endl;
    std::cout << "  Time: " << elapsed << "ms (" << (elapsed / 1000.0) << "s)" << std::endl;
    std::cout << "  Total associations: " << stats.total_associations << std::endl;
    std::cout << "  Formations: " << stats.formations_count << std::endl;
    std::cout << "  Reinforcements: " << stats.reinforcements_count << std::endl;

    EXPECT_GT(stats.total_associations, 0u);
    EXPECT_LT(elapsed, 30000.0); // Should complete in reasonable time
}

TEST(LearningSystemStressTest, MassivePredictions) {
    AssociationLearningSystem system;

    std::vector<PatternID> patterns;
    for (size_t i = 0; i < 1000; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    // Create associations
    auto& matrix = const_cast<AssociationMatrix&>(system.GetAssociationMatrix());
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = 0; j < 20; ++j) {
            size_t tgt_idx = (i + j + 1) % patterns.size();
            float strength = 0.5f + (j * 0.02f);
            AssociationEdge edge(patterns[i], patterns[tgt_idx],
                               AssociationType::CATEGORICAL, strength);
            matrix.AddAssociation(edge);
        }
    }

    // Massive prediction workload
    StressTestTimer timer;

    for (size_t i = 0; i < 100000; ++i) {
        size_t idx = i % patterns.size();
        auto predictions = system.Predict(patterns[idx], 5);
        EXPECT_LE(predictions.size(), 20u);
    }

    double elapsed = timer.ElapsedMs();

    std::cout << "Massive Predictions (100k):" << std::endl;
    std::cout << "  Time: " << elapsed << "ms" << std::endl;
    std::cout << "  Rate: " << (100000.0 / elapsed) * 1000.0 << " predictions/sec" << std::endl;

    EXPECT_LT(elapsed, 5000.0); // Should handle high prediction load
}

// ============================================================================
// Memory Pressure Tests
// ============================================================================

TEST(MemoryStressTest, LargePatternData) {
    MemoryBackend::Config config;
    MemoryBackend backend(config);

    // Store patterns with large data vectors
    StressTestTimer timer;

    for (size_t i = 0; i < 10000; ++i) {
        std::vector<float> large_data(1000); // 1000 floats = 4KB per pattern
        for (size_t j = 0; j < large_data.size(); ++j) {
            large_data[j] = static_cast<float>(j) / 1000.0f;
        }

        FeatureVector features(large_data);
        PatternData pattern_data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        PatternNode pattern(PatternID::Generate(), pattern_data, PatternType::ATOMIC);
        bool result = backend.Store(pattern);
        EXPECT_TRUE(result);
    }

    double elapsed = timer.ElapsedMs();

    std::cout << "Large Pattern Data (10k x 4KB):" << std::endl;
    std::cout << "  Time: " << elapsed << "ms" << std::endl;
    std::cout << "  Approximate memory: " << (10000 * 4) / 1024.0 << " MB" << std::endl;

    EXPECT_LT(elapsed, 5000.0);
}
