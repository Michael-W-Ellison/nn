// File: tests/similarity/similarity_search_test.cpp
#include "similarity/similarity_search.hpp"
#include "similarity/similarity_metric.hpp"
#include "storage/memory_backend.hpp"
#include <gtest/gtest.h>
#include <memory>

namespace dpan {
namespace {

// Mock metric that returns similarity based on feature sum difference
class MockSumSimilarity : public SimilarityMetric {
public:
    float Compute(const PatternData& a, const PatternData& b) const override {
        return ComputeFromFeatures(a.GetFeatures(), b.GetFeatures());
    }

    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override {
        float sum_a = 0.0f, sum_b = 0.0f;
        for (size_t i = 0; i < a.Dimension(); ++i) sum_a += a[i];
        for (size_t i = 0; i < b.Dimension(); ++i) sum_b += b[i];

        float diff = std::abs(sum_a - sum_b);
        return 1.0f / (1.0f + diff);
    }

    std::string GetName() const override { return "MockSum"; }
    bool IsSymmetric() const override { return true; }
};

// Helper to create test database with patterns
std::shared_ptr<PatternDatabase> CreateTestDatabase() {
    auto db = std::make_shared<MemoryBackend>(MemoryBackend::Config{});

    // Create patterns with different feature vectors
    for (int i = 0; i < 10; ++i) {
        FeatureVector fv({static_cast<float>(i), static_cast<float>(i * 2)});
        PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        PatternID id(i + 1);  // IDs starting from 1
        PatternNode node(id, data, PatternType::ATOMIC);
        db->Store(node);
    }

    return db;
}

// ============================================================================
// SimilaritySearch Tests
// ============================================================================

TEST(SimilaritySearchTest, ConstructorRequiresNonNullDatabase) {
    auto metric = std::make_shared<MockSumSimilarity>();
    EXPECT_THROW(SimilaritySearch(nullptr, metric), std::invalid_argument);
}

TEST(SimilaritySearchTest, ConstructorRequiresNonNullMetric) {
    auto db = CreateTestDatabase();
    EXPECT_THROW(SimilaritySearch(db, nullptr), std::invalid_argument);
}

TEST(SimilaritySearchTest, SearchReturnsResults) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    SimilaritySearch search(db, metric);

    FeatureVector query({5.0f, 10.0f});  // Similar to pattern 5
    PatternData query_data = PatternData::FromFeatures(query, DataModality::NUMERIC);

    auto results = search.Search(query_data, SearchConfig::TopK(5));

    EXPECT_EQ(5u, results.size());
    // Results should be sorted by similarity (highest first)
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i-1].similarity, results[i].similarity);
    }
}

TEST(SimilaritySearchTest, SearchByFeaturesWorks) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    SimilaritySearch search(db, metric);

    FeatureVector query({5.0f, 10.0f});
    auto results = search.SearchByFeatures(query, SearchConfig::TopK(3));

    EXPECT_EQ(3u, results.size());
}

TEST(SimilaritySearchTest, SearchByIdWorks) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    SimilaritySearch search(db, metric);

    auto all_ids = db->FindAll();
    ASSERT_FALSE(all_ids.empty());

    auto results = search.SearchById(all_ids[0], SearchConfig::TopK(5));

    EXPECT_LE(results.size(), 5u);
}

TEST(SimilaritySearchTest, SearchByIdExcludesQueryPattern) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    SimilaritySearch search(db, metric);

    auto all_ids = db->FindAll();
    ASSERT_FALSE(all_ids.empty());

    PatternID query_id = all_ids[0];
    auto results = search.SearchById(query_id, SearchConfig::TopK(10));

    // Query pattern should not be in results by default
    for (const auto& result : results) {
        EXPECT_NE(query_id, result.pattern_id);
    }
}

TEST(SimilaritySearchTest, SearchByIdIncludesQueryPatternWhenConfigured) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    SimilaritySearch search(db, metric);

    auto all_ids = db->FindAll();
    ASSERT_FALSE(all_ids.empty());

    PatternID query_id = all_ids[0];
    SearchConfig config = SearchConfig::TopK(10);
    config.include_query = true;

    auto results = search.SearchById(query_id, config);

    // Query pattern should be in results
    bool found_query = false;
    for (const auto& result : results) {
        if (result.pattern_id == query_id) {
            found_query = true;
            EXPECT_FLOAT_EQ(1.0f, result.similarity);  // Should have perfect similarity with itself
            break;
        }
    }
    EXPECT_TRUE(found_query);
}

TEST(SimilaritySearchTest, ThresholdFilteringWorks) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    SimilaritySearch search(db, metric);

    FeatureVector query({5.0f, 10.0f});
    PatternData query_data = PatternData::FromFeatures(query, DataModality::NUMERIC);

    auto results = search.Search(query_data, SearchConfig::WithThreshold(0.8f));

    // All results should have similarity >= 0.8
    for (const auto& result : results) {
        EXPECT_GE(result.similarity, 0.8f);
    }
}

TEST(SimilaritySearchTest, CustomFilterWorks) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    SimilaritySearch search(db, metric);

    FeatureVector query({5.0f, 10.0f});
    PatternData query_data = PatternData::FromFeatures(query, DataModality::NUMERIC);

    SearchConfig config = SearchConfig::TopK(10);
    // Filter to only include ATOMIC patterns
    config.filter = [](const PatternNode& node) {
        return node.GetType() == PatternType::ATOMIC;
    };

    auto results = search.Search(query_data, config);

    // All results should be ATOMIC
    for (const auto& result : results) {
        auto node_opt = db->Retrieve(result.pattern_id);
        ASSERT_TRUE(node_opt.has_value());
        EXPECT_EQ(PatternType::ATOMIC, node_opt->GetType());
    }
}

TEST(SimilaritySearchTest, BatchSearchWorks) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    SimilaritySearch search(db, metric);

    std::vector<PatternData> queries;
    queries.push_back(PatternData::FromFeatures(FeatureVector({1.0f, 2.0f}), DataModality::NUMERIC));
    queries.push_back(PatternData::FromFeatures(FeatureVector({5.0f, 10.0f}), DataModality::NUMERIC));
    queries.push_back(PatternData::FromFeatures(FeatureVector({9.0f, 18.0f}), DataModality::NUMERIC));

    auto results = search.SearchBatch(queries, SearchConfig::TopK(3));

    EXPECT_EQ(3u, results.size());
    for (const auto& query_results : results) {
        EXPECT_EQ(3u, query_results.size());
    }
}

TEST(SimilaritySearchTest, StatisticsAreUpdated) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    SimilaritySearch search(db, metric);

    FeatureVector query({5.0f, 10.0f});
    PatternData query_data = PatternData::FromFeatures(query, DataModality::NUMERIC);

    auto results = search.Search(query_data, SearchConfig::TopK(5));

    const auto& stats = search.GetLastSearchStats();
    EXPECT_GT(stats.patterns_evaluated, 0u);
    EXPECT_EQ(5u, stats.results_returned);
    EXPECT_GE(stats.max_similarity_found, stats.min_similarity_found);
    EXPECT_GE(stats.avg_similarity_found, stats.min_similarity_found);
    EXPECT_LE(stats.avg_similarity_found, stats.max_similarity_found);
}

TEST(SimilaritySearchTest, SetMetricWorks) {
    auto db = CreateTestDatabase();
    auto metric1 = std::make_shared<MockSumSimilarity>();
    SimilaritySearch search(db, metric1);

    auto metric2 = std::make_shared<MockSumSimilarity>();
    search.SetMetric(metric2);

    EXPECT_EQ(metric2, search.GetMetric());
}

TEST(SimilaritySearchTest, SetMetricRejectsNull) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    SimilaritySearch search(db, metric);

    EXPECT_THROW(search.SetMetric(nullptr), std::invalid_argument);
}

// ============================================================================
// ApproximateSearch Tests
// ============================================================================

TEST(ApproximateSearchTest, ConstructorRequiresNonNullDatabase) {
    auto metric = std::make_shared<MockSumSimilarity>();
    EXPECT_THROW(ApproximateSearch(nullptr, metric), std::invalid_argument);
}

TEST(ApproximateSearchTest, ConstructorRequiresNonNullMetric) {
    auto db = CreateTestDatabase();
    EXPECT_THROW(ApproximateSearch(db, nullptr), std::invalid_argument);
}

TEST(ApproximateSearchTest, SearchRequiresBuiltIndex) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    ApproximateSearch search(db, metric);

    FeatureVector query({5.0f, 10.0f});
    PatternData query_data = PatternData::FromFeatures(query, DataModality::NUMERIC);

    EXPECT_FALSE(search.IsIndexBuilt());
    EXPECT_THROW(search.Search(query_data), std::runtime_error);
}

TEST(ApproximateSearchTest, BuildIndexWorks) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    ApproximateSearch search(db, metric);

    search.BuildIndex();
    EXPECT_TRUE(search.IsIndexBuilt());
}

TEST(ApproximateSearchTest, SearchAfterBuildIndexWorks) {
    auto db = CreateTestDatabase();
    auto metric = std::make_shared<MockSumSimilarity>();
    ApproximateSearch search(db, metric, 5);  // 5 buckets

    search.BuildIndex();

    FeatureVector query({5.0f, 10.0f});
    PatternData query_data = PatternData::FromFeatures(query, DataModality::NUMERIC);

    auto results = search.Search(query_data, SearchConfig::TopK(5));

    EXPECT_LE(results.size(), 5u);
    // Results should be sorted
    for (size_t i = 1; i < results.size(); ++i) {
        EXPECT_GE(results[i-1].similarity, results[i].similarity);
    }
}

// ============================================================================
// MultiMetricSearch Tests
// ============================================================================

TEST(MultiMetricSearchTest, ConstructorRequiresNonNullDatabase) {
    EXPECT_THROW(MultiMetricSearch(nullptr), std::invalid_argument);
}

TEST(MultiMetricSearchTest, AddMetricWorks) {
    auto db = CreateTestDatabase();
    MultiMetricSearch search(db);

    auto metric = std::make_shared<MockSumSimilarity>();
    search.AddMetric(metric, 1.0f);

    EXPECT_EQ(1u, search.GetMetricCount());
}

TEST(MultiMetricSearchTest, AddMultipleMetricsWorks) {
    auto db = CreateTestDatabase();
    MultiMetricSearch search(db);

    auto metric1 = std::make_shared<MockSumSimilarity>();
    auto metric2 = std::make_shared<MockSumSimilarity>();

    search.AddMetric(metric1, 1.0f);
    search.AddMetric(metric2, 2.0f);

    EXPECT_EQ(2u, search.GetMetricCount());
}

TEST(MultiMetricSearchTest, ClearWorks) {
    auto db = CreateTestDatabase();
    MultiMetricSearch search(db);

    auto metric = std::make_shared<MockSumSimilarity>();
    search.AddMetric(metric, 1.0f);

    search.Clear();
    EXPECT_EQ(0u, search.GetMetricCount());
}

TEST(MultiMetricSearchTest, SearchWithNoMetricsReturnsEmpty) {
    auto db = CreateTestDatabase();
    MultiMetricSearch search(db);

    FeatureVector query({5.0f, 10.0f});
    PatternData query_data = PatternData::FromFeatures(query, DataModality::NUMERIC);

    auto results = search.Search(query_data);
    EXPECT_TRUE(results.empty());
}

TEST(MultiMetricSearchTest, SearchWithMetricsWorks) {
    auto db = CreateTestDatabase();
    MultiMetricSearch search(db);

    auto metric = std::make_shared<MockSumSimilarity>();
    search.AddMetric(metric, 1.0f);

    FeatureVector query({5.0f, 10.0f});
    PatternData query_data = PatternData::FromFeatures(query, DataModality::NUMERIC);

    auto results = search.Search(query_data, SearchConfig::TopK(5));

    EXPECT_EQ(5u, results.size());
}

TEST(MultiMetricSearchTest, WeightedCombinationWorks) {
    auto db = CreateTestDatabase();
    MultiMetricSearch search(db);

    auto metric1 = std::make_shared<MockSumSimilarity>();
    auto metric2 = std::make_shared<MockSumSimilarity>();

    search.AddMetric(metric1, 2.0f);
    search.AddMetric(metric2, 1.0f);

    FeatureVector query({5.0f, 10.0f});
    PatternData query_data = PatternData::FromFeatures(query, DataModality::NUMERIC);

    auto results = search.Search(query_data, SearchConfig::TopK(3));

    // Should get results based on weighted combination
    EXPECT_LE(results.size(), 3u);
}

// ============================================================================
// SearchConfig Tests
// ============================================================================

TEST(SearchConfigTest, DefaultConfigHasReasonableValues) {
    auto config = SearchConfig::Default();
    EXPECT_EQ(10u, config.max_results);
    EXPECT_FLOAT_EQ(0.0f, config.min_similarity);
    EXPECT_FALSE(config.include_query);
}

TEST(SearchConfigTest, TopKConfigSetsMaxResults) {
    auto config = SearchConfig::TopK(20);
    EXPECT_EQ(20u, config.max_results);
}

TEST(SearchConfigTest, WithThresholdConfigSetsThreshold) {
    auto config = SearchConfig::WithThreshold(0.7f);
    EXPECT_FLOAT_EQ(0.7f, config.min_similarity);
}

TEST(SearchConfigTest, WithThresholdConfigSetsMaxResults) {
    auto config = SearchConfig::WithThreshold(0.7f, 50);
    EXPECT_FLOAT_EQ(0.7f, config.min_similarity);
    EXPECT_EQ(50u, config.max_results);
}

} // namespace
} // namespace dpan
