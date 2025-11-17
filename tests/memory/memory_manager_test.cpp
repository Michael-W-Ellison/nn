// File: tests/memory/memory_manager_test.cpp
#include "memory/memory_manager.hpp"
#include "storage/memory_backend.hpp"
#include "similarity/similarity_metric.hpp"
#include <gtest/gtest.h>
#include <memory>

using namespace dpan;

// ============================================================================
// Mock Similarity Metric
// ============================================================================

class MockSimilarityMetric : public SimilarityMetric {
public:
    float Compute(const PatternData& a, const PatternData& b) const override {
        return 0.5f;
    }

    float ComputeFromFeatures(const FeatureVector& a, const FeatureVector& b) const override {
        return 0.5f;
    }

    std::string GetName() const override {
        return "MockSimilarityMetric";
    }
};

// ============================================================================
// Test Fixtures
// ============================================================================

class MemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create pattern database
        MemoryBackend::Config db_config;
        pattern_db_ = std::make_unique<MemoryBackend>(db_config);

        // Create association matrix
        assoc_matrix_ = std::make_unique<AssociationMatrix>();

        // Create similarity metric
        similarity_metric_ = std::make_shared<MockSimilarityMetric>();

        // Create memory manager with default config
        memory_manager_ = std::make_unique<MemoryManager>();
    }

    std::unique_ptr<PatternDatabase> pattern_db_;
    std::unique_ptr<AssociationMatrix> assoc_matrix_;
    std::shared_ptr<SimilarityMetric> similarity_metric_;
    std::unique_ptr<MemoryManager> memory_manager_;
};

// ============================================================================
// Configuration Tests (3 tests)
// ============================================================================

TEST_F(MemoryManagerTest, DefaultConfiguration) {
    MemoryManager manager;
    auto config = manager.GetConfig();

    EXPECT_TRUE(config.IsValid());
    EXPECT_TRUE(config.enable_automatic_pruning);
    EXPECT_TRUE(config.enable_tier_transitions);
}

TEST_F(MemoryManagerTest, CustomConfiguration) {
    MemoryManager::Config config;
    config.enable_automatic_pruning = false;
    config.enable_tier_transitions = true;
    config.enable_consolidation = false;
    config.maintenance_interval = std::chrono::seconds(60);

    EXPECT_TRUE(config.IsValid());

    MemoryManager manager(config);
    auto retrieved_config = manager.GetConfig();

    EXPECT_FALSE(retrieved_config.enable_automatic_pruning);
    EXPECT_TRUE(retrieved_config.enable_tier_transitions);
    EXPECT_FALSE(retrieved_config.enable_consolidation);
}

TEST_F(MemoryManagerTest, ConfigurationValidation) {
    MemoryManager::Config invalid_config;
    invalid_config.maintenance_interval = std::chrono::seconds(0);  // Invalid

    EXPECT_FALSE(invalid_config.IsValid());
}

// ============================================================================
// Initialization Tests (3 tests)
// ============================================================================

TEST_F(MemoryManagerTest, InitializationRequired) {
    EXPECT_FALSE(memory_manager_->IsInitialized());

    // Should throw if not initialized
    EXPECT_THROW(memory_manager_->PerformMaintenance(), std::runtime_error);
}

TEST_F(MemoryManagerTest, SuccessfulInitialization) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    EXPECT_TRUE(memory_manager_->IsInitialized());
}

TEST_F(MemoryManagerTest, InitializationWithNullPointers) {
    EXPECT_THROW(
        memory_manager_->Initialize(nullptr, assoc_matrix_.get(), similarity_metric_),
        std::invalid_argument
    );

    EXPECT_THROW(
        memory_manager_->Initialize(pattern_db_.get(), nullptr, similarity_metric_),
        std::invalid_argument
    );
}

// ============================================================================
// Statistics Tests (4 tests)
// ============================================================================

TEST_F(MemoryManagerTest, InitialStatistics) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    auto stats = memory_manager_->GetStatistics();

    EXPECT_EQ(0u, stats.total_patterns);
    EXPECT_EQ(0u, stats.total_associations);
    EXPECT_EQ(0u, stats.patterns_pruned_total);
    EXPECT_EQ(0u, stats.associations_pruned_total);
    EXPECT_GE(stats.memory_pressure, 0.0f);
    EXPECT_LE(stats.memory_pressure, 1.0f);
}

TEST_F(MemoryManagerTest, StatisticsAfterMaintenance) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    auto stats_before = memory_manager_->GetStatistics();

    memory_manager_->PerformMaintenance();

    auto stats_after = memory_manager_->GetStatistics();

    // Timestamps should be updated
    EXPECT_GT(stats_after.last_maintenance_time.ToMicros(), 0);
}

TEST_F(MemoryManagerTest, StatisticsPatternCounts) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    // Add some patterns to database
    // (In real test, would add actual patterns)

    auto stats = memory_manager_->GetStatistics();

    EXPECT_EQ(pattern_db_->Count(), stats.total_patterns);
    EXPECT_EQ(assoc_matrix_->GetAssociationCount(), stats.total_associations);
}

TEST_F(MemoryManagerTest, SleepStateInStatistics) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    auto stats = memory_manager_->GetStatistics();

    // Should have valid sleep state
    EXPECT_TRUE(
        stats.sleep_state == SleepConsolidator::ActivityState::ACTIVE ||
        stats.sleep_state == SleepConsolidator::ActivityState::LOW_ACTIVITY ||
        stats.sleep_state == SleepConsolidator::ActivityState::SLEEP
    );
}

// ============================================================================
// Maintenance Operations Tests (5 tests)
// ============================================================================

TEST_F(MemoryManagerTest, PerformMaintenanceCycle) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    EXPECT_NO_THROW(memory_manager_->PerformMaintenance());

    auto stats = memory_manager_->GetStatistics();
    EXPECT_GT(stats.last_maintenance_time.ToMicros(), 0);
}

TEST_F(MemoryManagerTest, PruningOperation) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    // Enable pruning
    MemoryManager::Config config;
    config.enable_automatic_pruning = true;
    memory_manager_->SetConfig(config);
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    EXPECT_NO_THROW(memory_manager_->PerformPruning());

    auto stats = memory_manager_->GetStatistics();
    // Pruning stats should be accessible
    EXPECT_GE(stats.patterns_pruned_last_cycle, 0u);
}

TEST_F(MemoryManagerTest, TierTransitions) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    EXPECT_NO_THROW(memory_manager_->PerformTierTransitions());
}

TEST_F(MemoryManagerTest, ConsolidationOperation) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    EXPECT_NO_THROW(memory_manager_->PerformConsolidation());

    auto stats = memory_manager_->GetStatistics();
    EXPECT_GT(stats.last_consolidation_time.ToMicros(), 0);
}

TEST_F(MemoryManagerTest, ForgettingMechanisms) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    MemoryManager::Config config;
    config.enable_forgetting = true;
    memory_manager_->SetConfig(config);
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    EXPECT_NO_THROW(memory_manager_->ApplyForgetting());
}

// ============================================================================
// Activity Recording Tests (3 tests)
// ============================================================================

TEST_F(MemoryManagerTest, RecordOperation) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    EXPECT_NO_THROW(memory_manager_->RecordOperation());

    // Record multiple operations
    for (int i = 0; i < 10; ++i) {
        memory_manager_->RecordOperation();
    }

    // Should not throw
    EXPECT_NO_THROW(memory_manager_->UpdateSleepState());
}

TEST_F(MemoryManagerTest, SleepStateUpdates) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    auto stats_before = memory_manager_->GetStatistics();
    auto state_before = stats_before.sleep_state;

    memory_manager_->UpdateSleepState();

    auto stats_after = memory_manager_->GetStatistics();
    // State should be valid (may or may not have changed)
    EXPECT_TRUE(
        stats_after.sleep_state == SleepConsolidator::ActivityState::ACTIVE ||
        stats_after.sleep_state == SleepConsolidator::ActivityState::LOW_ACTIVITY ||
        stats_after.sleep_state == SleepConsolidator::ActivityState::SLEEP
    );
}

TEST_F(MemoryManagerTest, SleepConsolidatorAccess) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    // Should be able to access sleep consolidator
    auto* sleep_consolidator = memory_manager_->GetSleepConsolidator();

    ASSERT_NE(sleep_consolidator, nullptr);
    EXPECT_FALSE(sleep_consolidator->IsInSleepState());
}

// ============================================================================
// Subsystem Access Tests (3 tests)
// ============================================================================

TEST_F(MemoryManagerTest, UtilityCalculatorAccess) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    auto& utility_calc = memory_manager_->GetUtilityCalculator();

    // Should be initialized with config
    auto config = utility_calc.GetConfig();
    EXPECT_TRUE(config.IsValid());
}

TEST_F(MemoryManagerTest, TierManagerAccess) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    auto& tier_manager = memory_manager_->GetTierManager();

    // Should be initialized
    auto stats = tier_manager.GetStats();
    EXPECT_GE(stats.active_count, 0u);
}

TEST_F(MemoryManagerTest, SleepConsolidatorConfiguration) {
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    const auto* sleep_consolidator = memory_manager_->GetSleepConsolidator();
    ASSERT_NE(sleep_consolidator, nullptr);
    auto config = sleep_consolidator->GetConfig();

    EXPECT_TRUE(config.IsValid());
}

// ============================================================================
// Integration Test (1 test)
// ============================================================================

TEST_F(MemoryManagerTest, FullIntegrationWorkflow) {
    // Initialize memory manager
    memory_manager_->Initialize(
        pattern_db_.get(),
        assoc_matrix_.get(),
        similarity_metric_
    );

    ASSERT_TRUE(memory_manager_->IsInitialized());

    // Record some operations
    for (int i = 0; i < 20; ++i) {
        memory_manager_->RecordOperation();
    }

    // Update sleep state
    memory_manager_->UpdateSleepState();

    // Perform maintenance
    memory_manager_->PerformMaintenance();

    // Get statistics
    auto stats = memory_manager_->GetStatistics();

    // Verify all subsystems are working
    EXPECT_TRUE(stats.last_maintenance_time.ToMicros() > 0);
    EXPECT_GE(stats.memory_pressure, 0.0f);
    EXPECT_LE(stats.memory_pressure, 1.0f);
    EXPECT_GE(stats.current_utility_threshold, 0.0f);
    EXPECT_LE(stats.current_utility_threshold, 1.0f);

    // Perform individual operations
    EXPECT_NO_THROW(memory_manager_->PerformPruning());
    EXPECT_NO_THROW(memory_manager_->PerformTierTransitions());
    EXPECT_NO_THROW(memory_manager_->PerformConsolidation());
    EXPECT_NO_THROW(memory_manager_->ApplyForgetting());

    // Get final statistics
    auto final_stats = memory_manager_->GetStatistics();

    EXPECT_GT(final_stats.last_pruning_time.ToMicros(), 0);
    EXPECT_GT(final_stats.last_transition_time.ToMicros(), 0);
    EXPECT_GT(final_stats.last_consolidation_time.ToMicros(), 0);
}
