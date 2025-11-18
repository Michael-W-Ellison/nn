// File: tests/learning/attention_test_fixtures.hpp
//
// Test Fixtures and Utilities for Attention Mechanism Tests
//
// Provides:
// - MockPatternDatabase: Simple in-memory pattern database for testing
// - Pattern factory functions: Create test patterns with known properties
// - Context factory functions: Create test contexts for various scenarios
// - Helper utilities: Common test data and verification functions
//
// Usage:
// @code
//   AttentionTestFixture fixture;
//   auto patterns = fixture.CreateTestPatterns(10);
//   auto context = fixture.CreateSemanticContext();
//   // Use in tests...
// @endcode

#ifndef DPAN_LEARNING_ATTENTION_TEST_FIXTURES_HPP
#define DPAN_LEARNING_ATTENTION_TEST_FIXTURES_HPP

#include "core/types.hpp"
#include "core/pattern_node.hpp"
#include "core/pattern_data.hpp"
#include "storage/pattern_database.hpp"
#include "learning/attention_mechanism.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <map>
#include <random>
#include <string>

namespace dpan {
namespace testing {

/// Simple in-memory mock pattern database for testing
///
/// This mock provides basic CRUD operations without persistence.
/// All data is stored in memory and lost when destroyed.
/// Thread-safe for concurrent read/write access.
class MockPatternDatabase : public PatternDatabase {
public:
    MockPatternDatabase() = default;
    ~MockPatternDatabase() override = default;

    // Core CRUD Operations
    bool Store(const PatternNode& node) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto id = node.GetID();
        if (patterns_.find(id) != patterns_.end()) {
            return false;  // Pattern already exists
        }
        // Can't use assignment due to deleted operators, use emplace
        patterns_.emplace(id, node.Clone());
        return true;
    }

    std::optional<PatternNode> Retrieve(PatternID id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = patterns_.find(id);
        if (it == patterns_.end()) {
            return std::nullopt;
        }
        return it->second.Clone();
    }

    bool Update(const PatternNode& node) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto id = node.GetID();
        if (patterns_.find(id) == patterns_.end()) {
            return false;  // Pattern doesn't exist
        }
        // Erase and re-insert due to deleted assignment operators
        patterns_.erase(id);
        patterns_.emplace(id, node.Clone());
        return true;
    }

    bool Delete(PatternID id) override {
        std::lock_guard<std::mutex> lock(mutex_);
        return patterns_.erase(id) > 0;
    }

    bool Exists(PatternID id) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return patterns_.find(id) != patterns_.end();
    }

    // Batch Operations
    size_t StoreBatch(const std::vector<PatternNode>& nodes) override {
        size_t count = 0;
        for (const auto& node : nodes) {
            if (Store(node)) {
                ++count;
            }
        }
        return count;
    }

    std::vector<PatternNode> RetrieveBatch(const std::vector<PatternID>& ids) override {
        std::vector<PatternNode> result;
        for (const auto& id : ids) {
            auto node = Retrieve(id);
            if (node) {
                result.push_back(std::move(*node));
            }
        }
        return result;
    }

    size_t DeleteBatch(const std::vector<PatternID>& ids) override {
        size_t count = 0;
        for (const auto& id : ids) {
            if (Delete(id)) {
                ++count;
            }
        }
        return count;
    }

    // Query Operations
    std::vector<PatternID> FindByType(
        PatternType type,
        const QueryOptions& options = {}) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<PatternID> result;
        for (const auto& [id, node] : patterns_) {
            if (node.GetType() == type) {
                result.push_back(id);
                if (result.size() >= options.max_results) break;
            }
        }
        return result;
    }

    std::vector<PatternID> FindByTimeRange(
        Timestamp start,
        Timestamp end,
        const QueryOptions& options = {}) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<PatternID> result;
        for (const auto& [id, node] : patterns_) {
            auto created = node.GetCreationTime();
            if (created >= start && created <= end) {
                result.push_back(id);
                if (result.size() >= options.max_results) break;
            }
        }
        return result;
    }

    std::vector<PatternID> FindAll(const QueryOptions& options = {}) override {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<PatternID> result;
        for (const auto& [id, _] : patterns_) {
            result.push_back(id);
            if (result.size() >= options.max_results) break;
        }
        return result;
    }

    // Statistics
    size_t Count() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return patterns_.size();
    }

    StorageStats GetStats() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        StorageStats stats;
        stats.total_patterns = patterns_.size();
        stats.memory_usage_bytes = EstimateMemoryUsage();
        return stats;
    }

    // Maintenance
    void Flush() override { /* No-op for in-memory */ }
    void Compact() override { /* No-op for in-memory */ }

    void Clear() override {
        std::lock_guard<std::mutex> lock(mutex_);
        patterns_.clear();
    }

    // Snapshot/Restore
    bool CreateSnapshot(const std::string& path) override {
        return false;  // Not implemented for mock
    }

    bool RestoreSnapshot(const std::string& path) override {
        return false;  // Not implemented for mock
    }

private:
    size_t EstimateMemoryUsage() const {
        size_t total = 0;
        for (const auto& [id, node] : patterns_) {
            total += node.EstimateMemoryUsage();
        }
        return total;
    }

    mutable std::mutex mutex_;
    std::map<PatternID, PatternNode> patterns_;
};

/// Test fixture for attention mechanism tests
///
/// Provides common test data, patterns, and utility functions.
/// Inherit from this class in your test fixtures:
/// @code
///   class MyAttentionTest : public AttentionTestFixture { };
/// @endcode
class AttentionTestFixture : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock database
        mock_db_ = std::make_unique<MockPatternDatabase>();

        // Initialize random seed for reproducible tests
        rng_.seed(42);
    }

    void TearDown() override {
        mock_db_.reset();
    }

    /// Create a test pattern with specified properties
    ///
    /// @param confidence Confidence score [0.0, 1.0]
    /// @param access_count Number of times accessed
    /// @param age_ms Age in milliseconds (defaults to 0 = just created)
    /// @return Pattern node with specified properties
    PatternNode CreateTestPattern(
        float confidence = 0.5f,
        uint32_t access_count = 0,
        int64_t age_ms = 0) {

        PatternID id = PatternID::Generate();

        // Create simple pattern data (empty for now, can be enhanced)
        PatternData data;

        PatternNode node(id, data, PatternType::ATOMIC);
        node.SetConfidenceScore(confidence);

        for (uint32_t i = 0; i < access_count; ++i) {
            node.IncrementAccessCount();
        }

        return node;
    }

    /// Create multiple test patterns with varying properties
    ///
    /// Creates patterns with:
    /// - Linearly increasing confidence (0.1 to 0.9)
    /// - Exponentially increasing access counts (1, 2, 4, 8, ...)
    ///
    /// @param count Number of patterns to create
    /// @return Vector of pattern IDs (patterns are stored in mock_db_)
    std::vector<PatternID> CreateTestPatterns(size_t count) {
        std::vector<PatternID> ids;

        for (size_t i = 0; i < count; ++i) {
            float confidence = 0.1f + (0.8f * i / std::max(count - 1, size_t(1)));
            uint32_t access_count = 1 << (i % 8);  // 1, 2, 4, 8, 16, 32, 64, 128

            auto node = CreateTestPattern(confidence, access_count);
            PatternID id = node.GetID();

            mock_db_->Store(node);
            ids.push_back(id);
        }

        return ids;
    }

    /// Create test patterns with random properties
    ///
    /// @param count Number of patterns to create
    /// @return Vector of pattern IDs (patterns are stored in mock_db_)
    std::vector<PatternID> CreateRandomPatterns(size_t count) {
        std::vector<PatternID> ids;
        std::uniform_real_distribution<float> conf_dist(0.0f, 1.0f);
        std::uniform_int_distribution<uint32_t> access_dist(0, 1000);

        for (size_t i = 0; i < count; ++i) {
            float confidence = conf_dist(rng_);
            uint32_t access_count = access_dist(rng_);

            auto node = CreateTestPattern(confidence, access_count);
            PatternID id = node.GetID();

            mock_db_->Store(node);
            ids.push_back(id);
        }

        return ids;
    }

    /// Create an empty context vector
    ContextVector CreateEmptyContext() {
        return ContextVector();
    }

    /// Create a simple semantic context
    ///
    /// Context dimensions:
    /// - "semantic": 0.8 (high semantic relevance)
    /// - "domain": 0.6 (moderate domain match)
    ContextVector CreateSemanticContext() {
        ContextVector context;
        context.Set("semantic", 0.8f);
        context.Set("domain", 0.6f);
        return context;
    }

    /// Create a temporal context
    ///
    /// Context dimensions:
    /// - "temporal": 0.9 (high temporal relevance)
    /// - "recency": 0.7 (moderately recent)
    ContextVector CreateTemporalContext() {
        ContextVector context;
        context.Set("temporal", 0.9f);
        context.Set("recency", 0.7f);
        return context;
    }

    /// Create a structural context
    ///
    /// Context dimensions:
    /// - "structural": 0.85 (high structural match)
    /// - "complexity": 0.5 (medium complexity)
    ContextVector CreateStructuralContext() {
        ContextVector context;
        context.Set("structural", 0.85f);
        context.Set("complexity", 0.5f);
        return context;
    }

    /// Create a multi-dimensional context
    ///
    /// Combines semantic, temporal, and structural dimensions
    ContextVector CreateMultiDimensionalContext() {
        ContextVector context;
        context.Set("semantic", 0.8f);
        context.Set("temporal", 0.6f);
        context.Set("structural", 0.7f);
        context.Set("domain", 0.5f);
        context.Set("recency", 0.4f);
        return context;
    }

    /// Create a random context with N dimensions
    ///
    /// @param num_dimensions Number of context dimensions
    /// @return Context with random values [0.0, 1.0]
    ContextVector CreateRandomContext(size_t num_dimensions) {
        ContextVector context;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (size_t i = 0; i < num_dimensions; ++i) {
            std::string dim = "dim_" + std::to_string(i);
            context.Set(dim, dist(rng_));
        }

        return context;
    }

    /// Create default attention configuration
    AttentionConfig CreateDefaultConfig() {
        AttentionConfig config;
        config.num_heads = 4;
        config.temperature = 1.0f;
        config.use_context = true;
        config.use_importance = true;
        config.attention_type = "dot_product";
        config.association_weight = 0.6f;
        config.attention_weight = 0.4f;
        config.enable_caching = true;
        config.cache_size = 1000;
        config.debug_logging = false;
        return config;
    }

    /// Create high-temperature config (softer distribution)
    AttentionConfig CreateHighTemperatureConfig() {
        auto config = CreateDefaultConfig();
        config.temperature = 2.0f;
        return config;
    }

    /// Create low-temperature config (sharper distribution)
    AttentionConfig CreateLowTemperatureConfig() {
        auto config = CreateDefaultConfig();
        config.temperature = 0.5f;
        return config;
    }

    /// Verify that attention weights sum to 1.0 (within tolerance)
    ///
    /// @param weights Map of pattern IDs to weights
    /// @param tolerance Acceptable deviation from 1.0 (default: 1e-5)
    void VerifyWeightsSumToOne(
        const std::map<PatternID, float>& weights,
        float tolerance = 1e-5f) {

        float sum = 0.0f;
        for (const auto& [id, weight] : weights) {
            sum += weight;
        }

        EXPECT_NEAR(sum, 1.0f, tolerance)
            << "Attention weights should sum to 1.0";
    }

    /// Verify that all weights are in valid range [0.0, 1.0]
    ///
    /// @param weights Map of pattern IDs to weights
    void VerifyWeightsInRange(const std::map<PatternID, float>& weights) {
        for (const auto& [id, weight] : weights) {
            EXPECT_GE(weight, 0.0f)
                << "Weight for pattern " << id.value() << " is negative";
            EXPECT_LE(weight, 1.0f)
                << "Weight for pattern " << id.value() << " exceeds 1.0";
        }
    }

    /// Verify attention scores are sorted by weight (descending)
    ///
    /// @param scores Vector of attention scores
    void VerifyScoresSorted(const std::vector<AttentionScore>& scores) {
        for (size_t i = 1; i < scores.size(); ++i) {
            EXPECT_GE(scores[i-1].weight, scores[i].weight)
                << "Scores not sorted at index " << i;
        }
    }

    /// Verify configuration is valid
    ///
    /// @param config Configuration to validate
    void VerifyConfigValid(const AttentionConfig& config) {
        EXPECT_TRUE(config.Validate())
            << "Configuration validation failed";
    }

protected:
    /// Mock pattern database for testing
    std::unique_ptr<MockPatternDatabase> mock_db_;

    /// Random number generator (seeded for reproducibility)
    std::mt19937 rng_;
};

} // namespace testing
} // namespace dpan

#endif // DPAN_LEARNING_ATTENTION_TEST_FIXTURES_HPP
