// File: tests/benchmarks/association_benchmarks.cpp
//
// Performance benchmarks for Association module

#include <gtest/gtest.h>
#include <chrono>
#include <random>
#include "association/association_matrix.hpp"
#include "association/association_learning_system.hpp"
#include "association/co_occurrence_tracker.hpp"
#include "association/competitive_learner.hpp"
#include "core/types.hpp"

using namespace dpan;
using namespace std::chrono;

// ============================================================================
// Benchmark Helper Functions
// ============================================================================

struct BenchmarkTimer {
    using TimePoint = high_resolution_clock::time_point;

    TimePoint start;

    BenchmarkTimer() : start(high_resolution_clock::now()) {}

    double ElapsedMs() const {
        auto end = high_resolution_clock::now();
        return duration_cast<duration<double, std::milli>>(end - start).count();
    }

    static double MeasureOps(size_t iterations, std::function<void()> fn) {
        BenchmarkTimer timer;
        for (size_t i = 0; i < iterations; ++i) {
            fn();
        }
        double elapsed = timer.ElapsedMs();
        return (iterations / elapsed) * 1000.0; // ops per second
    }
};

std::vector<PatternID> GeneratePatterns(size_t count) {
    std::vector<PatternID> patterns;
    patterns.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        patterns.push_back(PatternID::Generate());
    }
    return patterns;
}

// ============================================================================
// AssociationMatrix Benchmarks
// ============================================================================

TEST(AssociationMatrixBenchmark, AddAssociations_1000) {
    AssociationMatrix matrix;
    auto patterns = GeneratePatterns(100);

    BenchmarkTimer timer;
    for (size_t i = 0; i < 1000; ++i) {
        size_t src_idx = i % patterns.size();
        size_t tgt_idx = (i + 1) % patterns.size();
        AssociationEdge edge(patterns[src_idx], patterns[tgt_idx],
                           AssociationType::CATEGORICAL, 0.5f);
        matrix.AddAssociation(edge);
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (1000.0 / elapsed) * 1000.0;
    std::cout << "AddAssociations (1000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 100.0); // Should complete in < 100ms
}

TEST(AssociationMatrixBenchmark, AddAssociations_10000) {
    AssociationMatrix matrix;
    auto patterns = GeneratePatterns(1000);

    BenchmarkTimer timer;
    for (size_t i = 0; i < 10000; ++i) {
        size_t src_idx = i % patterns.size();
        size_t tgt_idx = (i + 1) % patterns.size();
        AssociationEdge edge(patterns[src_idx], patterns[tgt_idx],
                           AssociationType::CATEGORICAL, 0.5f);
        matrix.AddAssociation(edge);
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (10000.0 / elapsed) * 1000.0;
    std::cout << "AddAssociations (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 1000.0); // Should complete in < 1s
}

TEST(AssociationMatrixBenchmark, LookupAssociations_10000) {
    AssociationMatrix matrix;
    auto patterns = GeneratePatterns(100);

    // Add associations
    for (size_t i = 0; i < 1000; ++i) {
        size_t src_idx = i % patterns.size();
        size_t tgt_idx = (i + 1) % patterns.size();
        AssociationEdge edge(patterns[src_idx], patterns[tgt_idx],
                           AssociationType::CATEGORICAL, 0.5f);
        matrix.AddAssociation(edge);
    }

    // Benchmark lookups
    BenchmarkTimer timer;
    for (size_t i = 0; i < 10000; ++i) {
        size_t src_idx = i % patterns.size();
        size_t tgt_idx = (i + 1) % patterns.size();
        auto* assoc = matrix.GetAssociation(patterns[src_idx], patterns[tgt_idx]);
        (void)assoc; // Suppress unused warning
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (10000.0 / elapsed) * 1000.0;
    std::cout << "LookupAssociations (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 50.0); // Should complete in < 50ms
}

TEST(AssociationMatrixBenchmark, UpdateAssociations_10000) {
    AssociationMatrix matrix;
    auto patterns = GeneratePatterns(100);

    // Add associations
    for (size_t i = 0; i < 1000; ++i) {
        size_t src_idx = i % patterns.size();
        size_t tgt_idx = (i + 1) % patterns.size();
        AssociationEdge edge(patterns[src_idx], patterns[tgt_idx],
                           AssociationType::CATEGORICAL, 0.5f);
        matrix.AddAssociation(edge);
    }

    // Benchmark updates
    BenchmarkTimer timer;
    for (size_t i = 0; i < 10000; ++i) {
        size_t src_idx = i % patterns.size();
        size_t tgt_idx = (i + 1) % patterns.size();
        AssociationEdge edge(patterns[src_idx], patterns[tgt_idx],
                           AssociationType::CATEGORICAL, 0.7f);
        matrix.UpdateAssociation(patterns[src_idx], patterns[tgt_idx], edge);
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (10000.0 / elapsed) * 1000.0;
    std::cout << "UpdateAssociations (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 100.0); // Should complete in < 100ms
}

TEST(AssociationMatrixBenchmark, GetOutgoingAssociations_10000) {
    AssociationMatrix matrix;
    auto patterns = GeneratePatterns(100);

    // Add associations (10 outgoing per pattern)
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = 0; j < 10; ++j) {
            size_t tgt_idx = (i + j + 1) % patterns.size();
            AssociationEdge edge(patterns[i], patterns[tgt_idx],
                               AssociationType::CATEGORICAL, 0.5f);
            matrix.AddAssociation(edge);
        }
    }

    // Benchmark getting outgoing associations
    BenchmarkTimer timer;
    for (size_t i = 0; i < 10000; ++i) {
        size_t src_idx = i % patterns.size();
        auto assocs = matrix.GetOutgoingAssociations(patterns[src_idx]);
        EXPECT_EQ(10u, assocs.size());
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (10000.0 / elapsed) * 1000.0;
    std::cout << "GetOutgoingAssociations (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 50.0); // Should complete in < 50ms
}

TEST(AssociationMatrixBenchmark, PropagateActivation_1000) {
    AssociationMatrix matrix;
    auto patterns = GeneratePatterns(100);

    // Create a network with 5 associations per pattern
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = 0; j < 5; ++j) {
            size_t tgt_idx = (i + j + 1) % patterns.size();
            AssociationEdge edge(patterns[i], patterns[tgt_idx],
                               AssociationType::CATEGORICAL, 0.7f);
            matrix.AddAssociation(edge);
        }
    }

    // Benchmark propagation
    BenchmarkTimer timer;
    for (size_t i = 0; i < 1000; ++i) {
        size_t src_idx = i % patterns.size();
        auto results = matrix.PropagateActivation(patterns[src_idx], 1.0f, 3);
        EXPECT_GT(results.size(), 0u);
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (1000.0 / elapsed) * 1000.0;
    std::cout << "PropagateActivation (1000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 500.0); // Should complete in < 500ms
}

// ============================================================================
// CoOccurrenceTracker Benchmarks
// ============================================================================

TEST(CoOccurrenceTrackerBenchmark, RecordActivations_10000) {
    CoOccurrenceTracker::Config config;
    config.window_size = std::chrono::seconds(10);
    CoOccurrenceTracker tracker(config);

    auto patterns = GeneratePatterns(100);

    BenchmarkTimer timer;
    for (size_t i = 0; i < 10000; ++i) {
        size_t idx = i % patterns.size();
        tracker.RecordActivation(patterns[idx], Timestamp::Now());
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (10000.0 / elapsed) * 1000.0;
    std::cout << "RecordActivations (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 100.0); // Should complete in < 100ms
}

TEST(CoOccurrenceTrackerBenchmark, RecordBatchActivations_1000) {
    CoOccurrenceTracker::Config config;
    config.window_size = std::chrono::seconds(10);
    CoOccurrenceTracker tracker(config);

    auto patterns = GeneratePatterns(100);

    BenchmarkTimer timer;
    for (size_t i = 0; i < 1000; ++i) {
        // Record 10 patterns at once
        std::vector<PatternID> batch;
        for (size_t j = 0; j < 10; ++j) {
            batch.push_back(patterns[(i + j) % patterns.size()]);
        }
        tracker.RecordActivations(batch, Timestamp::Now());
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (1000.0 / elapsed) * 1000.0;
    std::cout << "RecordBatchActivations (1000x10): " << elapsed << "ms, "
              << ops_per_sec << " batch-ops/sec" << std::endl;

    EXPECT_LT(elapsed, 200.0); // Should complete in < 200ms
}

TEST(CoOccurrenceTrackerBenchmark, GetCoOccurrenceCount_10000) {
    CoOccurrenceTracker::Config config;
    config.window_size = std::chrono::seconds(10);
    CoOccurrenceTracker tracker(config);

    auto patterns = GeneratePatterns(100);

    // Record co-occurrences
    Timestamp now = Timestamp::Now();
    for (size_t i = 0; i < 1000; ++i) {
        size_t idx1 = i % patterns.size();
        size_t idx2 = (i + 1) % patterns.size();
        tracker.RecordActivation(patterns[idx1], now);
        tracker.RecordActivation(patterns[idx2], now);
    }

    // Benchmark lookups
    BenchmarkTimer timer;
    for (size_t i = 0; i < 10000; ++i) {
        size_t idx1 = i % patterns.size();
        size_t idx2 = (i + 1) % patterns.size();
        auto count = tracker.GetCoOccurrenceCount(patterns[idx1], patterns[idx2]);
        (void)count;
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (10000.0 / elapsed) * 1000.0;
    std::cout << "GetCoOccurrenceCount (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 50.0); // Should complete in < 50ms
}

// ============================================================================
// CompetitiveLearner Benchmarks
// ============================================================================

TEST(CompetitiveLearnerBenchmark, ApplyCompetition_1000Patterns) {
    AssociationMatrix matrix;
    auto patterns = GeneratePatterns(1000);

    // Add 10 associations per pattern
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = 0; j < 10; ++j) {
            size_t tgt_idx = (i + j + 1) % patterns.size();
            float strength = 0.3f + (static_cast<float>(j) / 10.0f) * 0.6f;
            AssociationEdge edge(patterns[i], patterns[tgt_idx],
                               AssociationType::CATEGORICAL, strength);
            matrix.AddAssociation(edge);
        }
    }

    CompetitiveLearner::Config config;
    config.competition_factor = 0.3f;

    // Benchmark competition
    BenchmarkTimer timer;
    size_t applied = 0;
    for (const auto& pattern : patterns) {
        if (CompetitiveLearner::ApplyCompetition(matrix, pattern, config)) {
            applied++;
        }
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (patterns.size() / elapsed) * 1000.0;
    std::cout << "ApplyCompetition (1000 patterns): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec, " << applied << " competed" << std::endl;

    EXPECT_LT(elapsed, 500.0); // Should complete in < 500ms
}

// ============================================================================
// AssociationLearningSystem Benchmarks
// ============================================================================

TEST(AssociationLearningSystemBenchmark, RecordActivations_10000) {
    AssociationLearningSystem::Config config;
    config.enable_auto_maintenance = false;
    AssociationLearningSystem system(config);

    auto patterns = GeneratePatterns(100);
    ContextVector context;

    BenchmarkTimer timer;
    for (size_t i = 0; i < 10000; ++i) {
        size_t idx = i % patterns.size();
        system.RecordPatternActivation(patterns[idx], context);
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (10000.0 / elapsed) * 1000.0;
    std::cout << "System RecordActivations (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 200.0); // Should complete in < 200ms
}

TEST(AssociationLearningSystemBenchmark, Predict_10000) {
    AssociationLearningSystem system;
    auto patterns = GeneratePatterns(100);

    // Create associations
    auto& matrix = const_cast<AssociationMatrix&>(system.GetAssociationMatrix());
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = 0; j < 5; ++j) {
            size_t tgt_idx = (i + j + 1) % patterns.size();
            AssociationEdge edge(patterns[i], patterns[tgt_idx],
                               AssociationType::CATEGORICAL, 0.5f + j * 0.1f);
            matrix.AddAssociation(edge);
        }
    }

    // Benchmark predictions
    BenchmarkTimer timer;
    for (size_t i = 0; i < 10000; ++i) {
        size_t idx = i % patterns.size();
        auto predictions = system.Predict(patterns[idx], 3);
        EXPECT_LE(predictions.size(), 5u);
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (10000.0 / elapsed) * 1000.0;
    std::cout << "System Predict (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 100.0); // Should complete in < 100ms
}

TEST(AssociationLearningSystemBenchmark, Reinforce_10000) {
    AssociationLearningSystem system;
    auto patterns = GeneratePatterns(100);

    // Create associations
    auto& matrix = const_cast<AssociationMatrix&>(system.GetAssociationMatrix());
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = 0; j < 5; ++j) {
            size_t tgt_idx = (i + j + 1) % patterns.size();
            AssociationEdge edge(patterns[i], patterns[tgt_idx],
                               AssociationType::CATEGORICAL, 0.5f);
            matrix.AddAssociation(edge);
        }
    }

    // Benchmark reinforcement
    BenchmarkTimer timer;
    for (size_t i = 0; i < 10000; ++i) {
        size_t src_idx = i % patterns.size();
        size_t tgt_idx = (i + 1) % patterns.size();
        bool correct = (i % 3) != 0;
        system.Reinforce(patterns[src_idx], patterns[tgt_idx], correct);
    }
    double elapsed = timer.ElapsedMs();

    double ops_per_sec = (10000.0 / elapsed) * 1000.0;
    std::cout << "System Reinforce (10000): " << elapsed << "ms, "
              << ops_per_sec << " ops/sec" << std::endl;

    EXPECT_LT(elapsed, 500.0); // Should complete in < 500ms
}

TEST(AssociationLearningSystemBenchmark, PerformMaintenance) {
    AssociationLearningSystem::Config config;
    config.enable_auto_maintenance = false;
    AssociationLearningSystem system(config);

    auto patterns = GeneratePatterns(100);

    // Create associations
    auto& matrix = const_cast<AssociationMatrix&>(system.GetAssociationMatrix());
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = 0; j < 10; ++j) {
            size_t tgt_idx = (i + j + 1) % patterns.size();
            AssociationEdge edge(patterns[i], patterns[tgt_idx],
                               AssociationType::CATEGORICAL, 0.5f);
            matrix.AddAssociation(edge);
        }
    }

    // Benchmark maintenance
    BenchmarkTimer timer;
    auto stats = system.PerformMaintenance();
    double elapsed = timer.ElapsedMs();

    std::cout << "System PerformMaintenance (1000 associations): " << elapsed << "ms" << std::endl;
    std::cout << "  Pruned: " << stats.associations_pruned << std::endl;

    EXPECT_LT(elapsed, 100.0); // Should complete in < 100ms
}

// ============================================================================
// Memory and Scalability Benchmarks
// ============================================================================

TEST(ScalabilityBenchmark, LargeScaleMatrix_100k_Associations) {
    AssociationMatrix matrix;
    auto patterns = GeneratePatterns(10000);

    BenchmarkTimer timer;

    // Add 100k associations
    for (size_t i = 0; i < 100000; ++i) {
        size_t src_idx = i % patterns.size();
        size_t tgt_idx = (i + 1) % patterns.size();
        AssociationEdge edge(patterns[src_idx], patterns[tgt_idx],
                           AssociationType::CATEGORICAL, 0.5f);
        matrix.AddAssociation(edge);
    }

    double add_elapsed = timer.ElapsedMs();

    // Test query performance
    BenchmarkTimer query_timer;
    for (size_t i = 0; i < 1000; ++i) {
        size_t idx = i % patterns.size();
        auto assocs = matrix.GetOutgoingAssociations(patterns[idx]);
        (void)assocs;
    }
    double query_elapsed = query_timer.ElapsedMs();

    std::cout << "Large Scale Matrix (100k associations):" << std::endl;
    std::cout << "  Add time: " << add_elapsed << "ms" << std::endl;
    std::cout << "  Query time (1000 queries): " << query_elapsed << "ms" << std::endl;
    std::cout << "  Total associations: " << matrix.GetAssociationCount() << std::endl;

    EXPECT_LT(add_elapsed, 10000.0); // Should complete in < 10s
    EXPECT_LT(query_elapsed, 100.0); // Queries should be fast
}

TEST(ScalabilityBenchmark, LargeScaleTracker_100k_Activations) {
    CoOccurrenceTracker::Config config;
    config.window_size = std::chrono::seconds(60);
    CoOccurrenceTracker tracker(config);

    auto patterns = GeneratePatterns(1000);

    BenchmarkTimer timer;

    // Record 100k activations
    for (size_t i = 0; i < 100000; ++i) {
        size_t idx = i % patterns.size();
        tracker.RecordActivation(patterns[idx], Timestamp::Now());
    }

    double record_elapsed = timer.ElapsedMs();

    // Test query performance
    BenchmarkTimer query_timer;
    for (size_t i = 0; i < 1000; ++i) {
        size_t idx1 = i % patterns.size();
        size_t idx2 = (i + 1) % patterns.size();
        auto count = tracker.GetCoOccurrenceCount(patterns[idx1], patterns[idx2]);
        (void)count;
    }
    double query_elapsed = query_timer.ElapsedMs();

    std::cout << "Large Scale Tracker (100k activations):" << std::endl;
    std::cout << "  Record time: " << record_elapsed << "ms" << std::endl;
    std::cout << "  Query time (1000 queries): " << query_elapsed << "ms" << std::endl;
    std::cout << "  Co-occurrence pairs: " << tracker.GetCoOccurrencePairCount() << std::endl;

    EXPECT_LT(record_elapsed, 5000.0); // Should complete in < 5s
    EXPECT_LT(query_elapsed, 100.0); // Queries should be fast
}
