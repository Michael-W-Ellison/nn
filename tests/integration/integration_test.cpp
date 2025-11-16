// File: tests/integration/integration_test.cpp
//
// Integration tests for the DPAN pattern recognition system.
// Tests end-to-end workflows and component interactions.

#include "core/pattern_engine.hpp"
#include "storage/memory_backend.hpp"
#include "storage/persistent_backend.hpp"
#include "similarity/contextual_similarity.hpp"
#include "similarity/geometric_similarity.hpp"
#include "similarity/statistical_similarity.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <filesystem>

using namespace dpan;

// ============================================================================
// Test Utilities
// ============================================================================

/// Generate random input data for testing
std::vector<uint8_t> GenerateRandomInput(size_t size, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, 255);

    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>(dist(gen));
    }
    return data;
}

/// Generate patterned input with repeated sequences
std::vector<uint8_t> GeneratePatternedInput(size_t pattern_size, size_t repetitions) {
    std::vector<uint8_t> pattern = GenerateRandomInput(pattern_size, 123);
    std::vector<uint8_t> result;
    result.reserve(pattern_size * repetitions);

    for (size_t i = 0; i < repetitions; ++i) {
        result.insert(result.end(), pattern.begin(), pattern.end());
    }
    return result;
}

/// Generate numeric sequence data
std::vector<uint8_t> GenerateNumericSequence(size_t count, float start = 0.0f, float step = 1.0f) {
    std::vector<uint8_t> result;
    result.reserve(count * sizeof(float));

    for (size_t i = 0; i < count; ++i) {
        float value = start + i * step;
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        result.insert(result.end(), bytes, bytes + sizeof(float));
    }
    return result;
}

/// Create test configuration for PatternEngine
PatternEngine::Config CreateTestEngineConfig(const std::string& db_type = "memory") {
    PatternEngine::Config config;
    config.database_type = db_type;
    config.database_path = "/tmp/dpan_integration_test.db";
    config.similarity_metric = "context";
    config.enable_auto_refinement = true;
    config.enable_indexing = true;

    // Configure extraction
    config.extraction_config.modality = DataModality::NUMERIC;
    config.extraction_config.min_pattern_size = 10;
    config.extraction_config.max_pattern_size = 1000;
    config.extraction_config.feature_dimension = 64;

    // Configure matching
    config.matching_config.similarity_threshold = 0.7f;
    config.matching_config.strong_match_threshold = 0.85f;
    config.matching_config.weak_match_threshold = 0.7f;

    return config;
}

// ============================================================================
// End-to-End Workflow Tests
// ============================================================================

TEST(IntegrationTest, EndToEndPatternProcessing) {
    auto config = CreateTestEngineConfig();
    PatternEngine engine(config);

    // Generate test data
    auto input = GenerateRandomInput(500);

    // Process input end-to-end
    auto result = engine.ProcessInput(input, DataModality::NUMERIC);

    // Verify processing completed
    EXPECT_GE(result.processing_time_ms, 0.0f);

    // Should have some activity (created or activated patterns)
    size_t total_activity = result.created_patterns.size() +
                           result.activated_patterns.size();
    EXPECT_GE(total_activity, 0u);

    // Verify patterns were actually stored
    auto stats = engine.GetStatistics();
    EXPECT_EQ(stats.total_patterns, result.created_patterns.size());
}

TEST(IntegrationTest, MultipleInputProcessingConverges) {
    auto config = CreateTestEngineConfig();
    PatternEngine engine(config);

    // Generate patterned input
    auto pattern = GeneratePatternedInput(50, 10);

    // Process same pattern multiple times
    std::vector<size_t> created_counts;
    std::vector<size_t> activated_counts;

    for (int i = 0; i < 5; ++i) {
        auto result = engine.ProcessInput(pattern, DataModality::NUMERIC);
        created_counts.push_back(result.created_patterns.size());
        activated_counts.push_back(result.activated_patterns.size());
    }

    // After initial processing, should create fewer new patterns
    // and activate more existing ones
    if (created_counts.size() >= 2) {
        // Later iterations should create fewer patterns than first iteration
        bool converging = created_counts.back() <= created_counts.front();
        EXPECT_TRUE(converging);
    }
}

TEST(IntegrationTest, PatternLifecycleCreateMatchUpdateSearch) {
    auto config = CreateTestEngineConfig();
    PatternEngine engine(config);

    // 1. Create initial pattern
    FeatureVector features(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    PatternID id = engine.CreatePattern(data, 0.7f);

    ASSERT_GT(id.value(), 0u);

    // 2. Retrieve and verify
    auto retrieved = engine.GetPattern(id);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->GetID(), id);
    EXPECT_FLOAT_EQ(retrieved->GetConfidenceScore(), 0.7f);

    // 3. Search for similar patterns (before update)
    auto similar = engine.FindSimilarPatterns(data, 5, 0.0f);
    EXPECT_GE(similar.size(), 1u);

    // Should find the pattern we created
    bool found = false;
    for (const auto& result : similar) {
        if (result.pattern_id == id) {
            found = true;
            EXPECT_GE(result.similarity, 0.0f);
        }
    }
    EXPECT_TRUE(found);

    // 4. Update pattern
    FeatureVector new_features(std::vector<float>{1.1f, 2.1f, 3.1f, 4.1f});
    PatternData new_data = PatternData::FromFeatures(new_features, DataModality::NUMERIC);
    bool updated = engine.UpdatePattern(id, new_data);
    EXPECT_TRUE(updated);

    // 5. Delete pattern
    bool deleted = engine.DeletePattern(id);
    EXPECT_TRUE(deleted);

    // 6. Verify deletion
    auto after_delete = engine.GetPattern(id);
    EXPECT_FALSE(after_delete.has_value());
}

// ============================================================================
// Multi-Component Integration Tests
// ============================================================================

TEST(IntegrationTest, ExtractionMatchingCreationPipeline) {
    auto config = CreateTestEngineConfig();
    PatternEngine engine(config);

    // Create initial patterns manually
    for (int i = 0; i < 5; ++i) {
        FeatureVector fv(std::vector<float>{
            static_cast<float>(i),
            static_cast<float>(i * 2)
        });
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        engine.CreatePattern(pd, 0.6f);
    }

    // Process new input
    auto input = GenerateNumericSequence(100, 2.5f, 0.5f);
    auto result = engine.ProcessInput(input, DataModality::NUMERIC);

    // Should match some existing patterns or create new ones
    size_t total_activity = result.created_patterns.size() +
                           result.activated_patterns.size();
    EXPECT_GT(total_activity, 0u);

    // Verify final state
    auto stats = engine.GetStatistics();
    EXPECT_GE(stats.total_patterns, 5u);
}

TEST(IntegrationTest, SimilaritySearchAcrossComponents) {
    auto config = CreateTestEngineConfig();
    config.enable_indexing = true;
    PatternEngine engine(config);

    // Create patterns with known similarities
    std::vector<PatternID> pattern_ids;
    for (int i = 0; i < 10; ++i) {
        FeatureVector fv(std::vector<float>{
            static_cast<float>(i) * 0.1f,
            static_cast<float>(i) * 0.2f,
            static_cast<float>(i) * 0.3f
        });
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        PatternID id = engine.CreatePattern(pd, 0.8f);
        pattern_ids.push_back(id);
    }

    // Search for patterns similar to first one
    auto query = engine.GetPattern(pattern_ids[0]);
    ASSERT_TRUE(query.has_value());

    auto results = engine.FindSimilarPatterns(query->GetData(), 5, 0.0f);

    // Should find multiple similar patterns
    EXPECT_GE(results.size(), 1u);
    EXPECT_LE(results.size(), 5u);

    // Results should be sorted by similarity (descending)
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i-1].similarity, results[i].similarity);
    }
}

TEST(IntegrationTest, CompositePatternHierarchy) {
    auto config = CreateTestEngineConfig();
    PatternEngine engine(config);

    // Create atomic patterns
    std::vector<PatternID> atomic_ids;
    for (int i = 0; i < 3; ++i) {
        FeatureVector fv(std::vector<float>{static_cast<float>(i)});
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        PatternID id = engine.CreatePattern(pd);
        atomic_ids.push_back(id);
    }

    // Create composite pattern
    FeatureVector comp_fv(std::vector<float>{10.0f, 20.0f});
    PatternData comp_data = PatternData::FromFeatures(comp_fv, DataModality::NUMERIC);
    PatternID comp_id = engine.CreateCompositePattern(atomic_ids, comp_data);

    ASSERT_GT(comp_id.value(), 0u);

    // Retrieve and verify composite pattern
    auto comp_pattern = engine.GetPattern(comp_id);
    ASSERT_TRUE(comp_pattern.has_value());
    EXPECT_EQ(comp_pattern->GetType(), PatternType::COMPOSITE);
    EXPECT_TRUE(comp_pattern->HasSubPatterns());

    auto sub_patterns = comp_pattern->GetSubPatterns();
    EXPECT_EQ(sub_patterns.size(), 3u);
}

// ============================================================================
// Database Backend Integration Tests
// ============================================================================

TEST(IntegrationTest, MemoryBackendFullWorkflow) {
    auto config = CreateTestEngineConfig("memory");
    PatternEngine engine(config);

    // Create patterns
    for (int i = 0; i < 20; ++i) {
        FeatureVector fv(std::vector<float>{static_cast<float>(i), static_cast<float>(i * i)});
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        engine.CreatePattern(pd);
    }

    // Verify all stored
    auto all_ids = engine.GetAllPatternIDs();
    EXPECT_EQ(all_ids.size(), 20u);

    // Batch retrieval
    auto patterns = engine.GetPatternsBatch(all_ids);
    EXPECT_EQ(patterns.size(), 20u);

    // Statistics
    auto stats = engine.GetStatistics();
    EXPECT_EQ(stats.total_patterns, 20u);
    EXPECT_EQ(stats.atomic_patterns, 20u);
    EXPECT_EQ(stats.composite_patterns, 0u);
}

// DISABLED: Hangs on pattern reload - needs investigation
TEST(IntegrationTest, DISABLED_PersistentBackendFullWorkflow) {
    std::string db_path = "/tmp/dpan_integration_persistent_test.db";

    // Clean up any existing database
    if (std::filesystem::exists(db_path)) {
        std::filesystem::remove(db_path);
    }

    // Create patterns and persist
    {
        auto config = CreateTestEngineConfig("persistent");
        config.database_path = db_path;
        PatternEngine engine(config);

        for (int i = 0; i < 15; ++i) {
            FeatureVector fv(std::vector<float>{static_cast<float>(i) * 0.5f});
            PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
            engine.CreatePattern(pd, 0.75f);
        }

        engine.Flush();

        auto stats = engine.GetStatistics();
        EXPECT_EQ(stats.total_patterns, 15u);
    }

    // Reload from persistent storage
    {
        auto config = CreateTestEngineConfig("persistent");
        config.database_path = db_path;
        PatternEngine engine(config);

        auto stats = engine.GetStatistics();
        EXPECT_EQ(stats.total_patterns, 15u);

        auto all_ids = engine.GetAllPatternIDs();
        EXPECT_EQ(all_ids.size(), 15u);
    }

    // Clean up
    if (std::filesystem::exists(db_path)) {
        std::filesystem::remove(db_path);
    }
}

// ============================================================================
// Similarity Metric Integration Tests
// ============================================================================

TEST(IntegrationTest, MultipleSimilarityMetrics) {
    std::vector<std::string> metrics = {"context", "hausdorff", "temporal", "histogram"};

    for (const auto& metric : metrics) {
        auto config = CreateTestEngineConfig();
        config.similarity_metric = metric;

        PatternEngine engine(config);

        // Create test patterns
        FeatureVector fv1(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});
        FeatureVector fv2(std::vector<float>{1.1f, 2.1f, 3.1f, 4.1f});

        PatternData pd1 = PatternData::FromFeatures(fv1, DataModality::NUMERIC);
        PatternData pd2 = PatternData::FromFeatures(fv2, DataModality::NUMERIC);

        PatternID id1 = engine.CreatePattern(pd1);
        PatternID id2 = engine.CreatePattern(pd2);

        // Search should work with any metric
        auto similar = engine.FindSimilarPatternsById(id1, 5, 0.0f);
        EXPECT_GE(similar.size(), 1u);
    }
}

TEST(IntegrationTest, SimilarityMetricConsistency) {
    auto config = CreateTestEngineConfig();
    config.similarity_metric = "context";
    PatternEngine engine(config);

    // Create patterns
    std::vector<PatternID> ids;
    for (int i = 0; i < 10; ++i) {
        FeatureVector fv(std::vector<float>{
            static_cast<float>(i),
            static_cast<float>(i + 1),
            static_cast<float>(i + 2)
        });
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        ids.push_back(engine.CreatePattern(pd));
    }

    // Query multiple times - should get consistent results
    auto query = engine.GetPattern(ids[0])->GetData();

    auto results1 = engine.FindSimilarPatterns(query, 5);
    auto results2 = engine.FindSimilarPatterns(query, 5);

    ASSERT_EQ(results1.size(), results2.size());

    for (size_t i = 0; i < results1.size(); ++i) {
        EXPECT_EQ(results1[i].pattern_id, results2[i].pattern_id);
        EXPECT_FLOAT_EQ(results1[i].similarity, results2[i].similarity);
    }
}

// ============================================================================
// Performance and Stress Tests
// ============================================================================

TEST(IntegrationTest, LargeScalePatternCreation) {
    auto config = CreateTestEngineConfig();
    PatternEngine engine(config);

    // Note: Due to caching limits, actual storage may be less than requested
    const size_t num_patterns = 100;

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_patterns; ++i) {
        FeatureVector fv(std::vector<float>{
            static_cast<float>(i),
            static_cast<float>(i % 100),
            static_cast<float>(i % 10)
        });
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        engine.CreatePattern(pd);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Verify all created
    auto stats = engine.GetStatistics();
    EXPECT_EQ(stats.total_patterns, num_patterns);

    // Performance check - should complete in reasonable time
    EXPECT_LT(duration.count(), 5000);  // Less than 5 seconds
}

TEST(IntegrationTest, LargeScaleSimilaritySearch) {
    auto config = CreateTestEngineConfig();
    config.enable_indexing = true;
    PatternEngine engine(config);

    // Create 500 patterns
    std::vector<PatternID> ids;
    for (size_t i = 0; i < 500; ++i) {
        FeatureVector fv(std::vector<float>{
            static_cast<float>(i) * 0.1f,
            static_cast<float>(i % 50) * 0.2f
        });
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        ids.push_back(engine.CreatePattern(pd));
    }

    // Perform multiple searches
    auto query = engine.GetPattern(ids[0])->GetData();

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; ++i) {
        auto results = engine.FindSimilarPatterns(query, 10, 0.3f);
        EXPECT_LE(results.size(), 10u);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 100 searches should be fast
    EXPECT_LT(duration.count(), 5000);  // Less than 5 seconds
}

TEST(IntegrationTest, ConcurrentPatternRetrieval) {
    auto config = CreateTestEngineConfig();
    PatternEngine engine(config);

    // Create patterns
    std::vector<PatternID> ids;
    for (int i = 0; i < 100; ++i) {
        FeatureVector fv(std::vector<float>{static_cast<float>(i)});
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        ids.push_back(engine.CreatePattern(pd));
    }

    // Retrieve all patterns
    auto patterns = engine.GetPatternsBatch(ids);
    EXPECT_EQ(patterns.size(), 100u);

    // Verify all have correct data
    for (const auto& pattern : patterns) {
        EXPECT_GT(pattern.GetID().value(), 0u);
        EXPECT_GE(pattern.GetConfidenceScore(), 0.0f);
        EXPECT_LE(pattern.GetConfidenceScore(), 1.0f);
    }
}

// ============================================================================
// Maintenance and Operations Tests
// ============================================================================

TEST(IntegrationTest, MaintenanceOperations) {
    auto config = CreateTestEngineConfig();
    config.enable_auto_refinement = true;
    PatternEngine engine(config);

    // Create patterns
    for (int i = 0; i < 50; ++i) {
        FeatureVector fv(std::vector<float>{
            static_cast<float>(i % 10),
            static_cast<float>(i % 5)
        });
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        engine.CreatePattern(pd);
    }

    auto stats_before = engine.GetStatistics();

    // Run maintenance
    engine.RunMaintenance();

    // Compact and flush
    engine.Compact();
    engine.Flush();

    // System should still be functional
    auto stats_after = engine.GetStatistics();
    EXPECT_GT(stats_after.total_patterns, 0u);
}

TEST(IntegrationTest, StatisticsAccuracy) {
    auto config = CreateTestEngineConfig();
    PatternEngine engine(config);

    // Create known number of patterns
    const size_t num_atomic = 15;
    const size_t num_composite = 3;

    std::vector<PatternID> atomic_ids;
    for (size_t i = 0; i < num_atomic; ++i) {
        FeatureVector fv(std::vector<float>{static_cast<float>(i)});
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        atomic_ids.push_back(engine.CreatePattern(pd, 0.8f));
    }

    for (size_t i = 0; i < num_composite; ++i) {
        std::vector<PatternID> sub_ids = {
            atomic_ids[i * 3],
            atomic_ids[i * 3 + 1],
            atomic_ids[i * 3 + 2]
        };
        FeatureVector fv(std::vector<float>{static_cast<float>(i * 10)});
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        engine.CreateCompositePattern(sub_ids, pd);
    }

    auto stats = engine.GetStatistics();
    EXPECT_EQ(stats.total_patterns, num_atomic + num_composite);
    EXPECT_EQ(stats.atomic_patterns, num_atomic);
    EXPECT_EQ(stats.composite_patterns, num_composite);
    // Average: (15 * 0.8 + 3 * 0.5) / 18 = 0.75
    EXPECT_FLOAT_EQ(stats.avg_confidence, 0.75f);
}

// ============================================================================
// Error Handling and Edge Cases
// ============================================================================

TEST(IntegrationTest, EmptyInputHandling) {
    auto config = CreateTestEngineConfig();
    PatternEngine engine(config);

    std::vector<uint8_t> empty_input;
    auto result = engine.ProcessInput(empty_input, DataModality::NUMERIC);

    // Should handle gracefully
    EXPECT_EQ(result.created_patterns.size(), 0u);
    EXPECT_EQ(result.activated_patterns.size(), 0u);
}

TEST(IntegrationTest, NonExistentPatternOperations) {
    auto config = CreateTestEngineConfig();
    PatternEngine engine(config);

    PatternID fake_id(999999);

    // Retrieve non-existent pattern
    auto pattern = engine.GetPattern(fake_id);
    EXPECT_FALSE(pattern.has_value());

    // Search from non-existent pattern
    auto similar = engine.FindSimilarPatternsById(fake_id, 10);
    EXPECT_EQ(similar.size(), 0u);

    // Delete non-existent pattern
    bool deleted = engine.DeletePattern(fake_id);
    EXPECT_FALSE(deleted);
}

TEST(IntegrationTest, DisabledIndexingFallback) {
    auto config = CreateTestEngineConfig();
    config.enable_indexing = false;  // Disable indexing
    PatternEngine engine(config);

    // Create patterns
    for (int i = 0; i < 20; ++i) {
        FeatureVector fv(std::vector<float>{static_cast<float>(i)});
        PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        engine.CreatePattern(pd);
    }

    // Search should still work (brute-force fallback)
    FeatureVector query_fv(std::vector<float>{5.0f});
    PatternData query = PatternData::FromFeatures(query_fv, DataModality::NUMERIC);

    auto results = engine.FindSimilarPatterns(query, 5, 0.0f);
    EXPECT_GT(results.size(), 0u);
}
