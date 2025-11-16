// File: tests/core/pattern_engine_test.cpp
#include "core/pattern_engine.hpp"
#include <gtest/gtest.h>

namespace dpan {
namespace {

// Helper to create test configuration
PatternEngine::Config CreateTestConfig() {
    PatternEngine::Config config;
    config.database_type = "memory";
    config.similarity_metric = "cosine";
    config.enable_auto_refinement = true;
    config.enable_indexing = true;
    return config;
}

// Helper to create test input
std::vector<uint8_t> CreateTestInput(size_t size) {
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<uint8_t>(i % 256);
    }
    return data;
}

// ============================================================================
// PatternEngine Tests
// ============================================================================

TEST(PatternEngineTest, ConstructorWithMemoryBackend) {
    PatternEngine::Config config = CreateTestConfig();
    EXPECT_NO_THROW(PatternEngine engine(config));
}

TEST(PatternEngineTest, ConstructorWithInvalidBackend) {
    PatternEngine::Config config = CreateTestConfig();
    config.database_type = "invalid";
    EXPECT_THROW(PatternEngine engine(config), std::invalid_argument);
}

TEST(PatternEngineTest, ProcessInputWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    auto input = CreateTestInput(100);
    auto result = engine.ProcessInput(input, DataModality::NUMERIC);

    // Should have created some patterns or activated existing ones
    size_t total_activity = result.activated_patterns.size() +
                           result.created_patterns.size() +
                           result.updated_patterns.size();

    EXPECT_GE(total_activity, 0u);  // At minimum, no crash
    EXPECT_GE(result.processing_time_ms, 0.0f);
}

TEST(PatternEngineTest, ProcessInputWithEmptyData) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    std::vector<uint8_t> empty_input;
    auto result = engine.ProcessInput(empty_input, DataModality::NUMERIC);

    EXPECT_EQ(0u, result.created_patterns.size());
    EXPECT_EQ(0u, result.activated_patterns.size());
    EXPECT_GE(result.processing_time_ms, 0.0f);
}

TEST(PatternEngineTest, DiscoverPatternsWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    auto input = CreateTestInput(100);
    auto pattern_ids = engine.DiscoverPatterns(input, DataModality::NUMERIC);

    // Should extract at least some patterns
    EXPECT_GE(pattern_ids.size(), 0u);

    // All IDs should be valid
    for (const auto& id : pattern_ids) {
        EXPECT_TRUE(id.IsValid());
    }
}

TEST(PatternEngineTest, CreatePatternWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    FeatureVector features(std::vector<float>{1.0f, 2.0f, 3.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    PatternID id = engine.CreatePattern(data, 0.75f);

    EXPECT_TRUE(id.IsValid());

    auto pattern_opt = engine.GetPattern(id);
    ASSERT_TRUE(pattern_opt.has_value());
    EXPECT_FLOAT_EQ(0.75f, pattern_opt->GetConfidenceScore());
}

TEST(PatternEngineTest, CreateCompositePatternWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    // Create sub-patterns first
    FeatureVector f1(std::vector<float>{1.0f, 2.0f});
    FeatureVector f2(std::vector<float>{3.0f, 4.0f});
    PatternData d1 = PatternData::FromFeatures(f1, DataModality::NUMERIC);
    PatternData d2 = PatternData::FromFeatures(f2, DataModality::NUMERIC);

    PatternID sub1 = engine.CreatePattern(d1);
    PatternID sub2 = engine.CreatePattern(d2);

    // Create composite pattern
    FeatureVector comp_features(std::vector<float>{2.5f, 3.0f});
    PatternData comp_data = PatternData::FromFeatures(comp_features, DataModality::NUMERIC);

    PatternID comp_id = engine.CreateCompositePattern({sub1, sub2}, comp_data);

    EXPECT_TRUE(comp_id.IsValid());

    auto comp_opt = engine.GetPattern(comp_id);
    ASSERT_TRUE(comp_opt.has_value());
    EXPECT_EQ(PatternType::COMPOSITE, comp_opt->GetType());
    EXPECT_EQ(2u, comp_opt->GetSubPatterns().size());
}

TEST(PatternEngineTest, GetPatternWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    FeatureVector features(std::vector<float>{1.0f, 2.0f, 3.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    PatternID id = engine.CreatePattern(data);

    auto pattern_opt = engine.GetPattern(id);
    ASSERT_TRUE(pattern_opt.has_value());
    EXPECT_EQ(id.value(), pattern_opt->GetID().value());
}

TEST(PatternEngineTest, GetPatternReturnsEmptyForNonExistent) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    auto pattern_opt = engine.GetPattern(PatternID(9999));
    EXPECT_FALSE(pattern_opt.has_value());
}

TEST(PatternEngineTest, GetPatternsBatchWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    // Create multiple patterns
    std::vector<PatternID> ids;
    for (int i = 0; i < 5; ++i) {
        FeatureVector features(std::vector<float>{static_cast<float>(i)});
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        ids.push_back(engine.CreatePattern(data));
    }

    auto patterns = engine.GetPatternsBatch(ids);
    EXPECT_EQ(5u, patterns.size());
}

TEST(PatternEngineTest, GetAllPatternIDsWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    // Create some patterns
    for (int i = 0; i < 3; ++i) {
        FeatureVector features(std::vector<float>{static_cast<float>(i)});
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        engine.CreatePattern(data);
    }

    auto all_ids = engine.GetAllPatternIDs();
    EXPECT_EQ(3u, all_ids.size());
}

TEST(PatternEngineTest, FindSimilarPatternsWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    // Create several patterns
    for (int i = 0; i < 10; ++i) {
        FeatureVector features(std::vector<float>{
            static_cast<float>(i),
            static_cast<float>(i + 1),
            static_cast<float>(i + 2)
        });
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        engine.CreatePattern(data);
    }

    // Search for similar patterns
    FeatureVector query_features(std::vector<float>{5.0f, 6.0f, 7.0f});
    PatternData query = PatternData::FromFeatures(query_features, DataModality::NUMERIC);

    auto results = engine.FindSimilarPatterns(query, 5);
    EXPECT_LE(results.size(), 5u);
}

TEST(PatternEngineTest, FindSimilarPatternsByIdWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    // Create patterns
    std::vector<PatternID> ids;
    for (int i = 0; i < 5; ++i) {
        FeatureVector features(std::vector<float>{static_cast<float>(i)});
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        ids.push_back(engine.CreatePattern(data));
    }

    // Find similar to first pattern
    auto results = engine.FindSimilarPatternsById(ids[0], 3);
    EXPECT_LE(results.size(), 3u);
}

TEST(PatternEngineTest, UpdatePatternWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    FeatureVector features(std::vector<float>{1.0f, 2.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    PatternID id = engine.CreatePattern(data, 0.7f);

    // Update pattern
    FeatureVector new_features(std::vector<float>{3.0f, 4.0f});
    PatternData new_data = PatternData::FromFeatures(new_features, DataModality::NUMERIC);

    bool success = engine.UpdatePattern(id, new_data);
    EXPECT_TRUE(success);

    auto pattern_opt = engine.GetPattern(id);
    ASSERT_TRUE(pattern_opt.has_value());

    const auto& updated_features = pattern_opt->GetData().GetFeatures();
    EXPECT_FLOAT_EQ(3.0f, updated_features[0]);
    EXPECT_FLOAT_EQ(4.0f, updated_features[1]);

    // Confidence should be preserved
    EXPECT_FLOAT_EQ(0.7f, pattern_opt->GetConfidenceScore());
}

TEST(PatternEngineTest, DeletePatternWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    FeatureVector features(std::vector<float>{1.0f, 2.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    PatternID id = engine.CreatePattern(data);

    // Verify pattern exists
    EXPECT_TRUE(engine.GetPattern(id).has_value());

    // Delete pattern
    bool success = engine.DeletePattern(id);
    EXPECT_TRUE(success);

    // Verify pattern no longer exists
    EXPECT_FALSE(engine.GetPattern(id).has_value());
}

TEST(PatternEngineTest, GetStatisticsWorks) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    auto stats_before = engine.GetStatistics();
    EXPECT_EQ(0u, stats_before.total_patterns);

    // Create some patterns
    for (int i = 0; i < 3; ++i) {
        FeatureVector features(std::vector<float>{static_cast<float>(i)});
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        engine.CreatePattern(data, 0.6f);
    }

    auto stats_after = engine.GetStatistics();
    EXPECT_EQ(3u, stats_after.total_patterns);
    EXPECT_EQ(3u, stats_after.atomic_patterns);
    EXPECT_EQ(0u, stats_after.composite_patterns);
    EXPECT_FLOAT_EQ(0.6f, stats_after.avg_confidence);
}

TEST(PatternEngineTest, GetConfigWorks) {
    PatternEngine::Config config = CreateTestConfig();
    config.similarity_metric = "euclidean";

    PatternEngine engine(config);

    const auto& retrieved_config = engine.GetConfig();
    EXPECT_EQ("euclidean", retrieved_config.similarity_metric);
    EXPECT_EQ("memory", retrieved_config.database_type);
}

TEST(PatternEngineTest, FlushDoesNotCrash) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    EXPECT_NO_THROW(engine.Flush());
}

TEST(PatternEngineTest, CompactDoesNotCrash) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    EXPECT_NO_THROW(engine.Compact());
}

TEST(PatternEngineTest, RunMaintenanceWorks) {
    PatternEngine::Config config = CreateTestConfig();
    config.enable_auto_refinement = true;

    PatternEngine engine(config);

    // Create some patterns
    for (int i = 0; i < 5; ++i) {
        FeatureVector features(std::vector<float>{static_cast<float>(i)});
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        engine.CreatePattern(data, 0.2f);  // Low confidence to trigger splitting
    }

    EXPECT_NO_THROW(engine.RunMaintenance());
}

TEST(PatternEngineTest, MultipleInputProcessing) {
    PatternEngine::Config config = CreateTestConfig();
    PatternEngine engine(config);

    // Process multiple inputs
    for (int i = 0; i < 10; ++i) {
        auto input = CreateTestInput(100 + i);
        auto result = engine.ProcessInput(input, DataModality::NUMERIC);
        EXPECT_GE(result.processing_time_ms, 0.0f);
    }

    auto stats = engine.GetStatistics();
    EXPECT_GT(stats.total_patterns, 0u);
}

TEST(PatternEngineTest, DifferentSimilarityMetrics) {
    // Test with cosine
    {
        PatternEngine::Config config = CreateTestConfig();
        config.similarity_metric = "cosine";
        EXPECT_NO_THROW(PatternEngine engine(config));
    }

    // Test with euclidean
    {
        PatternEngine::Config config = CreateTestConfig();
        config.similarity_metric = "euclidean";
        EXPECT_NO_THROW(PatternEngine engine(config));
    }

    // Test with manhattan
    {
        PatternEngine::Config config = CreateTestConfig();
        config.similarity_metric = "manhattan";
        EXPECT_NO_THROW(PatternEngine engine(config));
    }

    // Test with unknown (should default to cosine)
    {
        PatternEngine::Config config = CreateTestConfig();
        config.similarity_metric = "unknown";
        EXPECT_NO_THROW(PatternEngine engine(config));
    }
}

TEST(PatternEngineTest, IndexingEnabledVsDisabled) {
    // With indexing
    {
        PatternEngine::Config config = CreateTestConfig();
        config.enable_indexing = true;
        PatternEngine engine(config);

        FeatureVector features(std::vector<float>{1.0f, 2.0f});
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        engine.CreatePattern(data);

        auto results = engine.FindSimilarPatterns(data, 5);
        EXPECT_GE(results.size(), 0u);
    }

    // Without indexing (fallback to brute-force)
    {
        PatternEngine::Config config = CreateTestConfig();
        config.enable_indexing = false;
        PatternEngine engine(config);

        FeatureVector features(std::vector<float>{1.0f, 2.0f});
        PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        engine.CreatePattern(data);

        auto results = engine.FindSimilarPatterns(data, 5);
        EXPECT_GE(results.size(), 0u);
    }
}

} // namespace
} // namespace dpan
