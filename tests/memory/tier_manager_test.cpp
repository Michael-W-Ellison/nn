// File: tests/memory/tier_manager_test.cpp
//
// Comprehensive unit tests for Tier Manager System
//
// Tests all tier management functionality including:
// - Configuration validation
// - Initialization and setup
// - Pattern operations across tiers
// - Manual tier transitions (promotion/demotion)
// - Automatic tier transitions
// - Pattern selection algorithms
// - Statistics tracking
// - Threshold management
// - Background thread operations
// - Edge cases and error handling

#include "memory/tier_manager.hpp"
#include "memory/memory_tier.hpp"
#include "memory/utility_calculator.hpp"
#include "core/pattern_node.hpp"
#include "core/pattern_data.hpp"
#include "core/types.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>

using namespace dpan;
namespace fs = std::filesystem;

// ============================================================================
// Test Fixtures
// ============================================================================

class TierManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary directories for file-based tiers
        temp_base_dir_ = fs::temp_directory_path() / "dpan_tier_manager_test";
        warm_dir_ = temp_base_dir_ / "warm";
        cold_dir_ = temp_base_dir_ / "cold";
        archive_dir_ = temp_base_dir_ / "archive";

        fs::create_directories(warm_dir_);
        fs::create_directories(cold_dir_);
        fs::create_directories(archive_dir_);
    }

    void TearDown() override {
        // Clean up manager first
        manager_.reset();

        // Clean up temporary directories
        if (fs::exists(temp_base_dir_)) {
            fs::remove_all(temp_base_dir_);
        }
    }

    // Helper to create and initialize a TierManager with default config
    void CreateManager() {
        TierManager::Config config;
        manager_ = std::make_unique<TierManager>(config);
        manager_->Initialize(
            CreateActiveTier(),
            CreateWarmTier(warm_dir_.string()),
            CreateColdTier(cold_dir_.string()),
            CreateArchiveTier(archive_dir_.string())
        );
    }

    // Helper to create and initialize a TierManager with custom config
    void CreateManager(const TierManager::Config& config) {
        manager_ = std::make_unique<TierManager>(config);
        manager_->Initialize(
            CreateActiveTier(),
            CreateWarmTier(warm_dir_.string()),
            CreateColdTier(cold_dir_.string()),
            CreateArchiveTier(archive_dir_.string())
        );
    }

    // Helper to create a test pattern
    PatternNode CreateTestPattern(PatternID id) {
        FeatureVector fv(3);
        fv[0] = 1.0f;
        fv[1] = 2.0f;
        fv[2] = 3.0f;
        PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        return PatternNode(id, data, PatternType::ATOMIC);
    }

    PatternNode CreateTestPattern() {
        return CreateTestPattern(PatternID::Generate());
    }

    // Helper to create utility map
    std::unordered_map<PatternID, float> CreateUtilityMap(
        const std::vector<PatternID>& ids,
        const std::vector<float>& utilities) {
        std::unordered_map<PatternID, float> result;
        for (size_t i = 0; i < ids.size() && i < utilities.size(); ++i) {
            result[ids[i]] = utilities[i];
        }
        return result;
    }

    std::unique_ptr<TierManager> manager_;
    fs::path temp_base_dir_;
    fs::path warm_dir_;
    fs::path cold_dir_;
    fs::path archive_dir_;
};

// ============================================================================
// 1. Config Validation Tests (3+ tests)
// ============================================================================

TEST_F(TierManagerTest, ValidConfig) {
    TierManager::Config config;
    config.active_capacity = 100;
    config.warm_capacity = 1000;
    config.cold_capacity = 10000;
    config.warm_to_active_threshold = 0.8f;
    config.cold_to_warm_threshold = 0.6f;
    config.archive_to_cold_threshold = 0.4f;
    config.active_to_warm_threshold = 0.7f;
    config.warm_to_cold_threshold = 0.4f;
    config.cold_to_archive_threshold = 0.2f;
    config.transition_batch_size = 100;
    config.transition_interval_seconds = 60.0f;

    EXPECT_TRUE(config.IsValid());
    EXPECT_NO_THROW(TierManager manager(config));
}

TEST_F(TierManagerTest, InvalidCapacities_ZeroCapacity) {
    TierManager::Config config;
    config.active_capacity = 0;  // Invalid: zero capacity
    config.warm_capacity = 1000;
    config.cold_capacity = 10000;

    EXPECT_FALSE(config.IsValid());
    EXPECT_THROW(TierManager manager(config), std::invalid_argument);
}

TEST_F(TierManagerTest, InvalidCapacities_DecreasingCapacity) {
    TierManager::Config config;
    config.active_capacity = 10000;  // Invalid: active > warm
    config.warm_capacity = 1000;
    config.cold_capacity = 100;

    EXPECT_FALSE(config.IsValid());
    EXPECT_THROW(TierManager manager(config), std::invalid_argument);
}

TEST_F(TierManagerTest, InvalidThresholds_OutOfRange) {
    TierManager::Config config;
    config.active_capacity = 100;
    config.warm_capacity = 1000;
    config.cold_capacity = 10000;
    config.warm_to_active_threshold = 1.5f;  // Invalid: > 1.0

    EXPECT_FALSE(config.IsValid());
    EXPECT_THROW(TierManager manager(config), std::invalid_argument);
}

TEST_F(TierManagerTest, InvalidThresholds_NoHysteresis) {
    TierManager::Config config;
    config.active_capacity = 100;
    config.warm_capacity = 1000;
    config.cold_capacity = 10000;
    config.warm_to_active_threshold = 0.7f;
    config.active_to_warm_threshold = 0.8f;  // Invalid: promotion <= demotion

    EXPECT_FALSE(config.IsValid());
    EXPECT_THROW(TierManager manager(config), std::invalid_argument);
}

TEST_F(TierManagerTest, InvalidBatchSize) {
    TierManager::Config config;
    config.active_capacity = 100;
    config.warm_capacity = 1000;
    config.cold_capacity = 10000;
    config.transition_batch_size = 0;  // Invalid: zero batch size

    EXPECT_FALSE(config.IsValid());
    EXPECT_THROW(TierManager manager(config), std::invalid_argument);
}

TEST_F(TierManagerTest, InvalidTransitionInterval) {
    TierManager::Config config;
    config.active_capacity = 100;
    config.warm_capacity = 1000;
    config.cold_capacity = 10000;
    config.transition_interval_seconds = -1.0f;  // Invalid: negative interval

    EXPECT_FALSE(config.IsValid());
    EXPECT_THROW(TierManager manager(config), std::invalid_argument);
}

// ============================================================================
// 2. Initialization Tests (2+ tests)
// ============================================================================

TEST_F(TierManagerTest, SuccessfulInitialization) {
    TierManager::Config config;
    manager_ = std::make_unique<TierManager>(config);

    EXPECT_FALSE(manager_->IsInitialized());

    manager_->Initialize(
        CreateActiveTier(),
        CreateWarmTier(warm_dir_.string()),
        CreateColdTier(cold_dir_.string()),
        CreateArchiveTier(archive_dir_.string())
    );

    EXPECT_TRUE(manager_->IsInitialized());
}

TEST_F(TierManagerTest, OperationsFailBeforeInitialization) {
    TierManager::Config config;
    manager_ = std::make_unique<TierManager>(config);

    EXPECT_FALSE(manager_->IsInitialized());

    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    // Operations should fail before initialization
    EXPECT_FALSE(manager_->StorePattern(pattern, MemoryTier::ACTIVE));
    EXPECT_FALSE(manager_->LoadPattern(id).has_value());
    EXPECT_FALSE(manager_->RemovePattern(id));
    EXPECT_FALSE(manager_->GetPatternTier(id).has_value());

    std::unordered_map<PatternID, float> utilities;
    EXPECT_EQ(0u, manager_->PerformTierTransitions(utilities));
}

TEST_F(TierManagerTest, InitializationWithNullTiers) {
    TierManager::Config config;
    manager_ = std::make_unique<TierManager>(config);

    // Should throw when passing null tiers
    EXPECT_THROW(
        manager_->Initialize(nullptr, nullptr, nullptr, nullptr),
        std::invalid_argument
    );
}

// ============================================================================
// 3. Pattern Operations Tests (4+ tests)
// ============================================================================

TEST_F(TierManagerTest, StorePattern_Success) {
    CreateManager();

    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    EXPECT_TRUE(manager_->StorePattern(pattern, MemoryTier::ACTIVE));

    auto tier = manager_->GetPatternTier(id);
    ASSERT_TRUE(tier.has_value());
    EXPECT_EQ(MemoryTier::ACTIVE, *tier);
}

TEST_F(TierManagerTest, LoadPattern_Success) {
    CreateManager();

    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::WARM));

    auto loaded = manager_->LoadPattern(id);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(id, loaded->GetID());
}

TEST_F(TierManagerTest, LoadPattern_NotFound) {
    CreateManager();

    PatternID nonexistent = PatternID::Generate();

    auto loaded = manager_->LoadPattern(nonexistent);
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(TierManagerTest, RemovePattern_Success) {
    CreateManager();

    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::COLD));
    EXPECT_TRUE(manager_->GetPatternTier(id).has_value());

    EXPECT_TRUE(manager_->RemovePattern(id));
    EXPECT_FALSE(manager_->GetPatternTier(id).has_value());
}

TEST_F(TierManagerTest, RemovePattern_NotFound) {
    CreateManager();

    PatternID nonexistent = PatternID::Generate();

    EXPECT_FALSE(manager_->RemovePattern(nonexistent));
}

TEST_F(TierManagerTest, GetPatternTier_AllTiers) {
    CreateManager();

    std::vector<PatternNode> patterns;
    std::vector<PatternID> ids;
    std::vector<MemoryTier> tiers = {
        MemoryTier::ACTIVE,
        MemoryTier::WARM,
        MemoryTier::COLD,
        MemoryTier::ARCHIVE
    };

    for (auto tier : tiers) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        ASSERT_TRUE(manager_->StorePattern(pattern, tier));
    }

    for (size_t i = 0; i < ids.size(); ++i) {
        auto tier = manager_->GetPatternTier(ids[i]);
        ASSERT_TRUE(tier.has_value());
        EXPECT_EQ(tiers[i], *tier);
    }
}

// ============================================================================
// 4. Manual Tier Control Tests (4+ tests)
// ============================================================================

TEST_F(TierManagerTest, PromotePattern_WarmToActive) {
    CreateManager();

    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::WARM));

    EXPECT_TRUE(manager_->PromotePattern(id, MemoryTier::ACTIVE));

    auto tier = manager_->GetPatternTier(id);
    ASSERT_TRUE(tier.has_value());
    EXPECT_EQ(MemoryTier::ACTIVE, *tier);

    // Verify pattern data preserved
    auto loaded = manager_->LoadPattern(id);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(id, loaded->GetID());
}

TEST_F(TierManagerTest, PromotePattern_InvalidTarget) {
    CreateManager();

    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::WARM));

    // Cannot promote to same or lower tier
    EXPECT_FALSE(manager_->PromotePattern(id, MemoryTier::WARM));
    EXPECT_FALSE(manager_->PromotePattern(id, MemoryTier::COLD));
}

TEST_F(TierManagerTest, DemotePattern_ActiveToWarm) {
    CreateManager();

    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::ACTIVE));

    EXPECT_TRUE(manager_->DemotePattern(id, MemoryTier::WARM));

    auto tier = manager_->GetPatternTier(id);
    ASSERT_TRUE(tier.has_value());
    EXPECT_EQ(MemoryTier::WARM, *tier);

    // Verify pattern data preserved
    auto loaded = manager_->LoadPattern(id);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(id, loaded->GetID());
}

TEST_F(TierManagerTest, DemotePattern_InvalidTarget) {
    CreateManager();

    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::WARM));

    // Cannot demote to same or higher tier
    EXPECT_FALSE(manager_->DemotePattern(id, MemoryTier::WARM));
    EXPECT_FALSE(manager_->DemotePattern(id, MemoryTier::ACTIVE));
}

TEST_F(TierManagerTest, PromotePattern_PatternNotFound) {
    CreateManager();

    PatternID nonexistent = PatternID::Generate();

    EXPECT_FALSE(manager_->PromotePattern(nonexistent, MemoryTier::ACTIVE));
}

TEST_F(TierManagerTest, DemotePattern_PatternNotFound) {
    CreateManager();

    PatternID nonexistent = PatternID::Generate();

    EXPECT_FALSE(manager_->DemotePattern(nonexistent, MemoryTier::COLD));
}

// ============================================================================
// 5. Automatic Tier Transitions Tests (5+ tests)
// ============================================================================

TEST_F(TierManagerTest, PerformTierTransitions_PromoteFromWarm) {
    CreateManager();

    std::vector<PatternID> ids;

    // Store patterns in warm tier
    for (int i = 0; i < 5; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::WARM));
    }

    // Create utilities above warm_to_active_threshold (0.8)
    std::vector<float> utilities = {0.9f, 0.85f, 0.82f, 0.75f, 0.70f};
    auto utility_map = CreateUtilityMap(ids, utilities);

    size_t transitions = manager_->PerformTierTransitions(utility_map);
    EXPECT_GT(transitions, 0u);

    // First 3 patterns should be promoted (utility >= 0.8)
    for (int i = 0; i < 3; ++i) {
        auto tier = manager_->GetPatternTier(ids[i]);
        ASSERT_TRUE(tier.has_value());
        EXPECT_EQ(MemoryTier::ACTIVE, *tier);
    }

    // Last 2 should remain in warm (utility < 0.8)
    for (int i = 3; i < 5; ++i) {
        auto tier = manager_->GetPatternTier(ids[i]);
        ASSERT_TRUE(tier.has_value());
        EXPECT_EQ(MemoryTier::WARM, *tier);
    }
}

TEST_F(TierManagerTest, PerformTierTransitions_DemoteFromActive) {
    CreateManager();

    std::vector<PatternID> ids;

    // Store patterns in active tier
    for (int i = 0; i < 5; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::ACTIVE));
    }

    // Create utilities below active_to_warm_threshold (0.7)
    std::vector<float> utilities = {0.5f, 0.6f, 0.65f, 0.75f, 0.80f};
    auto utility_map = CreateUtilityMap(ids, utilities);

    size_t transitions = manager_->PerformTierTransitions(utility_map);
    EXPECT_GT(transitions, 0u);

    // First 3 patterns should be demoted (utility < 0.7)
    for (int i = 0; i < 3; ++i) {
        auto tier = manager_->GetPatternTier(ids[i]);
        ASSERT_TRUE(tier.has_value());
        EXPECT_EQ(MemoryTier::WARM, *tier);
    }

    // Last 2 should remain in active (utility >= 0.7)
    for (int i = 3; i < 5; ++i) {
        auto tier = manager_->GetPatternTier(ids[i]);
        ASSERT_TRUE(tier.has_value());
        EXPECT_EQ(MemoryTier::ACTIVE, *tier);
    }
}

TEST_F(TierManagerTest, PerformTierTransitions_EmptyUtilities) {
    CreateManager();

    // Store some patterns
    for (int i = 0; i < 3; ++i) {
        PatternNode pattern = CreateTestPattern();
        ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::ACTIVE));
    }

    std::unordered_map<PatternID, float> empty_utilities;

    // Should not crash with empty utilities
    size_t transitions = manager_->PerformTierTransitions(empty_utilities);
    EXPECT_EQ(0u, transitions);
}

TEST_F(TierManagerTest, PerformTierTransitions_CapacityEnforcement) {
    TierManager::Config config;
    config.active_capacity = 2;  // Small capacity
    config.warm_capacity = 10;
    config.cold_capacity = 100;
    CreateManager(config);

    std::vector<PatternID> ids;

    // Store 5 patterns in active tier (exceeds capacity)
    for (int i = 0; i < 5; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::ACTIVE));
    }

    // Create low utilities to trigger demotion
    std::vector<float> utilities = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    auto utility_map = CreateUtilityMap(ids, utilities);

    size_t transitions = manager_->PerformTierTransitions(utility_map);
    EXPECT_GT(transitions, 0u);

    // Active tier should be at or below capacity
    auto stats = manager_->GetStats();
    EXPECT_LE(stats.active_count, config.active_capacity);
}

TEST_F(TierManagerTest, PerformTierTransitions_MultiTierPromotions) {
    CreateManager();

    std::vector<PatternID> ids;

    // Store patterns across different tiers
    for (int i = 0; i < 3; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::ARCHIVE));
    }

    for (int i = 0; i < 3; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::COLD));
    }

    // High utilities for all patterns
    std::vector<float> utilities(6, 0.9f);
    auto utility_map = CreateUtilityMap(ids, utilities);

    size_t transitions = manager_->PerformTierTransitions(utility_map);
    EXPECT_GT(transitions, 0u);

    // Some patterns should have been promoted
    auto stats = manager_->GetStats();
    EXPECT_LT(stats.archive_count, 3u);
    EXPECT_LT(stats.cold_count, 3u);
}

TEST_F(TierManagerTest, PerformTierTransitions_VerifyNoDataLoss) {
    CreateManager();

    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::WARM));

    // Load original data
    auto original = manager_->LoadPattern(id);
    ASSERT_TRUE(original.has_value());

    // Promote to active
    std::unordered_map<PatternID, float> utilities;
    utilities[id] = 0.95f;
    manager_->PerformTierTransitions(utilities);

    // Verify data still intact
    auto after_promotion = manager_->LoadPattern(id);
    ASSERT_TRUE(after_promotion.has_value());
    EXPECT_EQ(id, after_promotion->GetID());

    // Demote to cold
    utilities[id] = 0.3f;
    manager_->PerformTierTransitions(utilities);

    // Verify data still intact
    auto after_demotion = manager_->LoadPattern(id);
    ASSERT_TRUE(after_demotion.has_value());
    EXPECT_EQ(id, after_demotion->GetID());
}

// ============================================================================
// 6. Pattern Selection Tests (2+ tests)
// ============================================================================

TEST_F(TierManagerTest, SelectPatternsForPromotion_SortedByUtility) {
    CreateManager();

    std::vector<PatternID> ids;

    // Store patterns in warm tier
    for (int i = 0; i < 10; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::WARM));
    }

    // Create utilities with varying values
    std::vector<float> utilities = {0.95f, 0.85f, 0.82f, 0.81f, 0.79f,
                                   0.75f, 0.70f, 0.65f, 0.60f, 0.55f};
    auto utility_map = CreateUtilityMap(ids, utilities);

    // Perform transitions
    manager_->PerformTierTransitions(utility_map);

    // Check that highest utility patterns were promoted first
    // (utilities >= 0.8 threshold)
    for (int i = 0; i < 4; ++i) {
        auto tier = manager_->GetPatternTier(ids[i]);
        ASSERT_TRUE(tier.has_value());
        EXPECT_EQ(MemoryTier::ACTIVE, *tier);
    }
}

TEST_F(TierManagerTest, SelectPatternsForDemotion_SortedByUtility) {
    CreateManager();

    std::vector<PatternID> ids;

    // Store patterns in active tier
    for (int i = 0; i < 10; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::ACTIVE));
    }

    // Create utilities with varying values
    // Note: PerformTierTransitions applies cascading demotions:
    // - ACTIVE->WARM if < 0.7 (active_to_warm_threshold)
    // - WARM->COLD if < 0.4 (warm_to_cold_threshold)
    // - COLD->ARCHIVE if < 0.2 (cold_to_archive_threshold)
    std::vector<float> utilities = {0.45f, 0.50f, 0.55f, 0.60f, 0.65f,
                                   0.68f, 0.72f, 0.75f, 0.80f, 0.85f};
    auto utility_map = CreateUtilityMap(ids, utilities);

    // Perform transitions
    manager_->PerformTierTransitions(utility_map);

    // Patterns with utility < 0.7 should be demoted to WARM (first 6 patterns: 0.45-0.68)
    // Since all have utility >= 0.4, they stay in WARM (not demoted to COLD)
    for (int i = 0; i < 6; ++i) {
        auto tier = manager_->GetPatternTier(ids[i]);
        ASSERT_TRUE(tier.has_value());
        EXPECT_EQ(MemoryTier::WARM, *tier);
    }

    // Patterns with utility >= 0.7 should remain in ACTIVE (last 4 patterns: 0.72-0.85)
    for (int i = 6; i < 10; ++i) {
        auto tier = manager_->GetPatternTier(ids[i]);
        ASSERT_TRUE(tier.has_value());
        EXPECT_EQ(MemoryTier::ACTIVE, *tier);
    }
}

TEST_F(TierManagerTest, SelectPatternsForPromotion_BatchSizeLimit) {
    TierManager::Config config;
    config.transition_batch_size = 3;  // Small batch size
    CreateManager(config);

    std::vector<PatternID> ids;

    // Store many patterns in warm tier
    for (int i = 0; i < 10; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::WARM));
    }

    // All have high utility
    std::vector<float> utilities(10, 0.95f);
    auto utility_map = CreateUtilityMap(ids, utilities);

    // Perform transitions
    size_t transitions = manager_->PerformTierTransitions(utility_map);

    // Should respect batch size limit
    EXPECT_LE(transitions, config.transition_batch_size);
}

// ============================================================================
// 7. Statistics Tests (2+ tests)
// ============================================================================

TEST_F(TierManagerTest, GetStats_InitialState) {
    CreateManager();

    auto stats = manager_->GetStats();

    EXPECT_EQ(0u, stats.active_count);
    EXPECT_EQ(0u, stats.warm_count);
    EXPECT_EQ(0u, stats.cold_count);
    EXPECT_EQ(0u, stats.archive_count);
    EXPECT_EQ(0u, stats.promotions_count);
    EXPECT_EQ(0u, stats.demotions_count);
}

TEST_F(TierManagerTest, GetStats_PatternCounts) {
    CreateManager();

    // Store patterns in different tiers
    for (int i = 0; i < 5; ++i) {
        ASSERT_TRUE(manager_->StorePattern(CreateTestPattern(), MemoryTier::ACTIVE));
    }
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(manager_->StorePattern(CreateTestPattern(), MemoryTier::WARM));
    }
    for (int i = 0; i < 7; ++i) {
        ASSERT_TRUE(manager_->StorePattern(CreateTestPattern(), MemoryTier::COLD));
    }
    for (int i = 0; i < 2; ++i) {
        ASSERT_TRUE(manager_->StorePattern(CreateTestPattern(), MemoryTier::ARCHIVE));
    }

    auto stats = manager_->GetStats();

    EXPECT_EQ(5u, stats.active_count);
    EXPECT_EQ(3u, stats.warm_count);
    EXPECT_EQ(7u, stats.cold_count);
    EXPECT_EQ(2u, stats.archive_count);
}

TEST_F(TierManagerTest, GetStats_PromotionDemotionCounters) {
    CreateManager();

    PatternNode pattern1 = CreateTestPattern();
    PatternNode pattern2 = CreateTestPattern();
    PatternID id1 = pattern1.GetID();
    PatternID id2 = pattern2.GetID();

    ASSERT_TRUE(manager_->StorePattern(pattern1, MemoryTier::WARM));
    ASSERT_TRUE(manager_->StorePattern(pattern2, MemoryTier::ACTIVE));

    auto stats_before = manager_->GetStats();
    EXPECT_EQ(0u, stats_before.promotions_count);
    EXPECT_EQ(0u, stats_before.demotions_count);

    // Promote pattern1
    ASSERT_TRUE(manager_->PromotePattern(id1, MemoryTier::ACTIVE));

    auto stats_after_promotion = manager_->GetStats();
    EXPECT_EQ(1u, stats_after_promotion.promotions_count);
    EXPECT_EQ(0u, stats_after_promotion.demotions_count);

    // Demote pattern2
    ASSERT_TRUE(manager_->DemotePattern(id2, MemoryTier::COLD));

    auto stats_after_demotion = manager_->GetStats();
    EXPECT_EQ(1u, stats_after_demotion.promotions_count);
    EXPECT_EQ(1u, stats_after_demotion.demotions_count);
}

TEST_F(TierManagerTest, GetStats_LastTransitionTime) {
    CreateManager();

    auto stats_before = manager_->GetStats();
    Timestamp time_before = stats_before.last_transition;

    // Perform a transition
    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::WARM));

    std::unordered_map<PatternID, float> utilities;
    utilities[id] = 0.95f;

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager_->PerformTierTransitions(utilities);

    auto stats_after = manager_->GetStats();

    // Last transition time should be updated if transitions occurred
    if (stats_after.promotions_count > stats_before.promotions_count) {
        EXPECT_GT(stats_after.last_transition.ToMicros(),
                 time_before.ToMicros());
    }
}

// ============================================================================
// 8. Threshold Tests (2+ tests)
// ============================================================================

TEST_F(TierManagerTest, GetPromotionThreshold_AllTiers) {
    CreateManager();

    // Active tier should have no promotion threshold
    EXPECT_LT(manager_->GetPromotionThreshold(MemoryTier::ACTIVE), 0.0f);

    // Other tiers should have valid thresholds
    EXPECT_GE(manager_->GetPromotionThreshold(MemoryTier::WARM), 0.0f);
    EXPECT_LE(manager_->GetPromotionThreshold(MemoryTier::WARM), 1.0f);

    EXPECT_GE(manager_->GetPromotionThreshold(MemoryTier::COLD), 0.0f);
    EXPECT_LE(manager_->GetPromotionThreshold(MemoryTier::COLD), 1.0f);

    EXPECT_GE(manager_->GetPromotionThreshold(MemoryTier::ARCHIVE), 0.0f);
    EXPECT_LE(manager_->GetPromotionThreshold(MemoryTier::ARCHIVE), 1.0f);
}

TEST_F(TierManagerTest, GetDemotionThreshold_AllTiers) {
    CreateManager();

    // Archive tier should have no demotion threshold
    EXPECT_LT(manager_->GetDemotionThreshold(MemoryTier::ARCHIVE), 0.0f);

    // Other tiers should have valid thresholds
    EXPECT_GE(manager_->GetDemotionThreshold(MemoryTier::ACTIVE), 0.0f);
    EXPECT_LE(manager_->GetDemotionThreshold(MemoryTier::ACTIVE), 1.0f);

    EXPECT_GE(manager_->GetDemotionThreshold(MemoryTier::WARM), 0.0f);
    EXPECT_LE(manager_->GetDemotionThreshold(MemoryTier::WARM), 1.0f);

    EXPECT_GE(manager_->GetDemotionThreshold(MemoryTier::COLD), 0.0f);
    EXPECT_LE(manager_->GetDemotionThreshold(MemoryTier::COLD), 1.0f);
}

TEST_F(TierManagerTest, Thresholds_HysteresisVerification) {
    CreateManager();

    // Promotion thresholds should be higher than demotion thresholds
    float warm_promote = manager_->GetPromotionThreshold(MemoryTier::WARM);
    float active_demote = manager_->GetDemotionThreshold(MemoryTier::ACTIVE);
    EXPECT_GT(warm_promote, active_demote);

    float cold_promote = manager_->GetPromotionThreshold(MemoryTier::COLD);
    float warm_demote = manager_->GetDemotionThreshold(MemoryTier::WARM);
    EXPECT_GT(cold_promote, warm_demote);

    float archive_promote = manager_->GetPromotionThreshold(MemoryTier::ARCHIVE);
    float cold_demote = manager_->GetDemotionThreshold(MemoryTier::COLD);
    EXPECT_GT(archive_promote, cold_demote);
}

// ============================================================================
// 9. Background Thread Tests (1+ test)
// ============================================================================

TEST_F(TierManagerTest, BackgroundTransitions_StartStop) {
    // Use shorter transition interval for testing
    TierManager::Config config;
    config.transition_interval_seconds = 0.1f;  // 100ms instead of 300s
    CreateManager(config);

    UtilityCalculator utility_calc;
    AccessTracker access_tracker;

    EXPECT_FALSE(manager_->IsBackgroundRunning());

    // Start background thread
    manager_->StartBackgroundTransitions(&utility_calc, &access_tracker);
    EXPECT_TRUE(manager_->IsBackgroundRunning());

    // Allow some time for thread to start
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Stop background thread
    manager_->StopBackgroundTransitions();
    EXPECT_FALSE(manager_->IsBackgroundRunning());
}

TEST_F(TierManagerTest, BackgroundTransitions_DoubleStart) {
    // Use shorter transition interval for testing
    TierManager::Config config;
    config.transition_interval_seconds = 0.1f;  // 100ms instead of 300s
    CreateManager(config);

    UtilityCalculator utility_calc;
    AccessTracker access_tracker;

    // Start once
    manager_->StartBackgroundTransitions(&utility_calc, &access_tracker);
    EXPECT_TRUE(manager_->IsBackgroundRunning());

    // Starting again should be safe (no-op)
    EXPECT_NO_THROW(
        manager_->StartBackgroundTransitions(&utility_calc, &access_tracker)
    );
    EXPECT_TRUE(manager_->IsBackgroundRunning());

    manager_->StopBackgroundTransitions();
}

TEST_F(TierManagerTest, BackgroundTransitions_DoubleStop) {
    // Use shorter transition interval for testing
    TierManager::Config config;
    config.transition_interval_seconds = 0.1f;  // 100ms instead of 300s
    CreateManager(config);

    UtilityCalculator utility_calc;
    AccessTracker access_tracker;

    manager_->StartBackgroundTransitions(&utility_calc, &access_tracker);
    manager_->StopBackgroundTransitions();
    EXPECT_FALSE(manager_->IsBackgroundRunning());

    // Stopping again should be safe (no-op)
    EXPECT_NO_THROW(manager_->StopBackgroundTransitions());
    EXPECT_FALSE(manager_->IsBackgroundRunning());
}

TEST_F(TierManagerTest, BackgroundTransitions_NullArguments) {
    CreateManager();

    // Should throw with null arguments
    EXPECT_THROW(
        manager_->StartBackgroundTransitions(nullptr, nullptr),
        std::invalid_argument
    );
}

// ============================================================================
// 10. Edge Cases Tests (2+ tests)
// ============================================================================

TEST_F(TierManagerTest, EdgeCase_EmptyTransitions) {
    CreateManager();

    std::unordered_map<PatternID, float> empty_utilities;

    // Should handle empty transitions gracefully
    EXPECT_NO_THROW(manager_->PerformTierTransitions(empty_utilities));

    auto stats = manager_->GetStats();
    EXPECT_EQ(0u, stats.active_count);
    EXPECT_EQ(0u, stats.warm_count);
    EXPECT_EQ(0u, stats.cold_count);
    EXPECT_EQ(0u, stats.archive_count);
}

TEST_F(TierManagerTest, EdgeCase_PatternInMultipleTransitions) {
    CreateManager();

    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::WARM));

    // Multiple transitions in sequence
    std::unordered_map<PatternID, float> utilities_high;
    utilities_high[id] = 0.95f;
    manager_->PerformTierTransitions(utilities_high);

    auto tier1 = manager_->GetPatternTier(id);
    ASSERT_TRUE(tier1.has_value());
    EXPECT_EQ(MemoryTier::ACTIVE, *tier1);

    // Demote back down
    // Use utility 0.5 which will demote from ACTIVE to WARM but not further
    // (< 0.7 active_to_warm_threshold, but >= 0.4 warm_to_cold_threshold)
    std::unordered_map<PatternID, float> utilities_low;
    utilities_low[id] = 0.5f;
    manager_->PerformTierTransitions(utilities_low);

    auto tier2 = manager_->GetPatternTier(id);
    ASSERT_TRUE(tier2.has_value());
    EXPECT_EQ(MemoryTier::WARM, *tier2);
}

TEST_F(TierManagerTest, EdgeCase_LargeNumberOfPatterns) {
    CreateManager();

    const int num_patterns = 100;
    std::vector<PatternID> ids;

    // Store many patterns
    for (int i = 0; i < num_patterns; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::COLD));
    }

    auto stats = manager_->GetStats();
    EXPECT_EQ(static_cast<size_t>(num_patterns), stats.cold_count);

    // Promote all
    std::unordered_map<PatternID, float> utilities;
    for (const auto& id : ids) {
        utilities[id] = 0.95f;
    }

    manager_->PerformTierTransitions(utilities);

    // Verify some were promoted
    auto stats_after = manager_->GetStats();
    EXPECT_LT(stats_after.cold_count, static_cast<size_t>(num_patterns));
}

TEST_F(TierManagerTest, EdgeCase_UtilityAtExactThreshold) {
    CreateManager();

    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    ASSERT_TRUE(manager_->StorePattern(pattern, MemoryTier::WARM));

    // Utility exactly at threshold (0.8)
    std::unordered_map<PatternID, float> utilities;
    utilities[id] = 0.8f;

    manager_->PerformTierTransitions(utilities);

    // Should be promoted (>= threshold)
    auto tier = manager_->GetPatternTier(id);
    ASSERT_TRUE(tier.has_value());
    EXPECT_EQ(MemoryTier::ACTIVE, *tier);
}

TEST_F(TierManagerTest, EdgeCase_ConfigUpdate) {
    CreateManager();

    TierManager::Config new_config;
    new_config.active_capacity = 50;
    new_config.warm_capacity = 500;
    new_config.cold_capacity = 5000;
    new_config.warm_to_active_threshold = 0.9f;
    new_config.active_to_warm_threshold = 0.85f;

    EXPECT_NO_THROW(manager_->SetConfig(new_config));

    auto& config = manager_->GetConfig();
    EXPECT_EQ(50u, config.active_capacity);
    EXPECT_FLOAT_EQ(0.9f, config.warm_to_active_threshold);
}

TEST_F(TierManagerTest, EdgeCase_InvalidConfigUpdate) {
    CreateManager();

    TierManager::Config invalid_config;
    invalid_config.active_capacity = 0;  // Invalid

    EXPECT_THROW(manager_->SetConfig(invalid_config), std::invalid_argument);
}
