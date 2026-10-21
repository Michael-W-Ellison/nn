// File: tests/memory/memory_manager_test.cpp
#include "memory/memory_manager.hpp"
#include "storage/memory_backend.hpp"
#include "similarity/similarity_metric.hpp"
#include <gtest/gtest.h>
#include "../test_temp_path.hpp"
#include <chrono>
#include <filesystem>
#include <memory>
#include <thread>

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

// ============================================================================
// Access-Driven Utilities, Pruning and Forgetting
// ============================================================================

namespace {

PatternID StoreDistinctPattern(PatternDatabase& db, float seed) {
    FeatureVector fv(3);
    fv[0] = seed;
    fv[1] = -seed * 2.0f;
    fv[2] = seed * seed + 1.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    PatternNode node(PatternID::Generate(), data, PatternType::ATOMIC);
    node.SetConfidenceScore(0.8f);
    PatternID id = node.GetID();
    db.Store(node);
    return id;
}

}  // namespace

TEST_F(MemoryManagerTest, UtilitiesReflectRecordedAccesses) {
    memory_manager_->Initialize(pattern_db_.get(), assoc_matrix_.get(), similarity_metric_);

    PatternID used = StoreDistinctPattern(*pattern_db_, 1.0f);
    PatternID untracked = StoreDistinctPattern(*pattern_db_, 2.0f);
    for (int i = 0; i < 200; ++i) {
        memory_manager_->RecordPatternAccess(used);
    }

    auto utilities = memory_manager_->ComputePatternUtilities();
    ASSERT_EQ(2u, utilities.size());
    // Untracked pattern is scored from its own node: new and never accessed
    EXPECT_GT(utilities[used], utilities[untracked]);
    EXPECT_EQ(1u, memory_manager_->GetAccessTracker().GetTrackedPatternCount());
}

TEST_F(MemoryManagerTest, PruningRemovesUnusedPatterns) {
    MemoryManager::Config config;
    config.pattern_pruner_config.min_pattern_age = std::chrono::milliseconds(1);
    config.pattern_pruner_config.enable_merging = false;
    MemoryManager manager(config);
    manager.Initialize(pattern_db_.get(), assoc_matrix_.get(), similarity_metric_);

    PatternID used = StoreDistinctPattern(*pattern_db_, 1.0f);
    PatternID unused = StoreDistinctPattern(*pattern_db_, 2.0f);
    for (int i = 0; i < 200; ++i) {
        manager.RecordPatternAccess(used);
    }
    // Last used 1000 hours ago: recency (and so utility) is low
    manager.RecordPatternAccess(unused, Timestamp::Now() - std::chrono::hours(1000));
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    manager.PerformPruning();

    EXPECT_TRUE(pattern_db_->Exists(used));
    EXPECT_FALSE(pattern_db_->Exists(unused));
    EXPECT_EQ(1u, manager.GetStatistics().patterns_pruned_last_cycle);
}

TEST_F(MemoryManagerTest, AssociationPruningRemovesWeakAssociations) {
    MemoryManager::Config config;
    config.association_pruner_config.min_association_strength = 0.05f;
    config.association_pruner_config.weak_strength_threshold = 0.1f;
    MemoryManager manager(config);
    manager.Initialize(pattern_db_.get(), assoc_matrix_.get(), similarity_metric_);

    PatternID a = PatternID::Generate();
    PatternID b = PatternID::Generate();
    PatternID c = PatternID::Generate();
    assoc_matrix_->AddAssociation(AssociationEdge(a, b, AssociationType::CAUSAL, 0.01f));
    assoc_matrix_->AddAssociation(AssociationEdge(a, c, AssociationType::CAUSAL, 0.9f));

    manager.PerformPruning();

    EXPECT_FALSE(assoc_matrix_->HasAssociation(a, b));
    EXPECT_TRUE(assoc_matrix_->HasAssociation(a, c));
    EXPECT_EQ(1u, manager.GetStatistics().associations_pruned_last_cycle);
}

TEST_F(MemoryManagerTest, ForgettingDecaysUnusedPatternConfidence) {
    MemoryManager::Config config;
    config.decay_constant = 100000.0f;  // per hour: visible after milliseconds
    MemoryManager manager(config);
    manager.Initialize(pattern_db_.get(), assoc_matrix_.get(), nullptr);

    PatternID id = StoreDistinctPattern(*pattern_db_, 1.0f);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    manager.ApplyForgetting();

    auto node = pattern_db_->Retrieve(id);
    ASSERT_TRUE(node.has_value());
    EXPECT_LT(node->GetConfidenceScore(), 0.8f);
    EXPECT_EQ(1u, manager.GetStatistics().patterns_with_decay);
}

TEST_F(MemoryManagerTest, SleepConsolidationStrengthensUsefulPatterns) {
    MemoryManager::Config config;
    config.sleep_config.min_utility_for_strengthening = 0.3f;
    config.pattern_pruner_config.enable_merging = false;
    MemoryManager manager(config);
    manager.Initialize(pattern_db_.get(), assoc_matrix_.get(), similarity_metric_);

    PatternID used = StoreDistinctPattern(*pattern_db_, 1.0f);
    for (int i = 0; i < 500; ++i) {
        manager.RecordPatternAccess(used);
    }

    auto result = manager.GetSleepConsolidator()->TriggerConsolidation();

    EXPECT_TRUE(result.was_successful);
    EXPECT_EQ(1u, result.patterns_strengthened);
    auto node = pattern_db_->Retrieve(used);
    ASSERT_TRUE(node.has_value());
    EXPECT_GT(node->GetConfidenceScore(), 0.8f);
}

TEST_F(MemoryManagerTest, TierTransitionsRunOnceTiersInitialized) {
    std::filesystem::path base = test::UniqueTempPath("dpan_memory_manager_tiers");
    std::filesystem::remove_all(base);

    memory_manager_->Initialize(pattern_db_.get(), assoc_matrix_.get(), similarity_metric_);
    auto& tiers = memory_manager_->GetTierManager();
    tiers.Initialize(CreateActiveTier(), CreateWarmTier((base / "warm").string()),
                     CreateColdTier((base / "cold").string()),
                     CreateArchiveTier((base / "archive").string()));

    FeatureVector fv(2);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    PatternNode node(PatternID::Generate(), PatternData::FromFeatures(fv, DataModality::NUMERIC),
                     PatternType::ATOMIC);
    ASSERT_TRUE(tiers.StorePattern(node, MemoryTier::ACTIVE));

    memory_manager_->PerformTierTransitions();

    EXPECT_NE(MemoryTier::ACTIVE, tiers.GetPatternTier(node.GetID()));
    EXPECT_EQ(0u, memory_manager_->GetStatistics().active_patterns);

    std::filesystem::remove_all(base);
}
