// File: tests/memory/pattern_pruner_test.cpp
//
// Comprehensive unit tests for Pattern Pruner system
//
// Tests cover:
// - Config validation (valid/invalid configurations)
// - Safety checks (hub detection, age checks, association strength)
// - Pattern pruning (individual and batch operations)
// - Pattern merging (association transfer, self-loop prevention)
// - Statistics (pruning results, bytes freed)
// - Edge cases (non-existent patterns, safety preservation)

#include "memory/pattern_pruner.hpp"
#include "storage/memory_backend.hpp"
#include "association/association_matrix.hpp"
#include "association/association_edge.hpp"
#include "core/pattern_node.hpp"
#include "core/pattern_data.hpp"
#include "core/types.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <thread>

using namespace dpan;

// ============================================================================
// Test Fixtures
// ============================================================================

class PatternPrunerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create in-memory pattern database
        MemoryBackend::Config db_config;
        db_config.initial_capacity = 1000;
        pattern_db_ = std::make_unique<MemoryBackend>(db_config);

        // Create association matrix
        assoc_matrix_ = std::make_unique<AssociationMatrix>();

        // Create pattern pruner with default config
        PatternPruner::Config pruner_config;
        pruner_config.utility_threshold = 0.2f;
        pruner_config.min_associations_for_hub = 50;
        pruner_config.min_pattern_age = std::chrono::hours(24);
        pruner_config.strong_association_threshold = 0.7f;
        pruner_config.enable_merging = false;  // Disable merging by default for simpler tests
        pruner_config.max_prune_batch = 1000;

        pruner_ = std::make_unique<PatternPruner>(pruner_config);
    }

    void TearDown() override {
        pruner_.reset();
        assoc_matrix_.reset();
        pattern_db_.reset();
    }

    // Helper to create a test pattern with specific timestamp
    PatternNode CreateTestPattern(Timestamp creation_time = Timestamp::Now()) {
        FeatureVector fv(3);
        fv[0] = 1.0f;
        fv[1] = 2.0f;
        fv[2] = 3.0f;
        PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);

        // Note: We can't directly set creation time, so for old patterns
        // we'll need to work around this limitation in tests
        return pattern;
    }

    // Helper to create an old pattern (simulate by using a pattern created long ago)
    PatternID CreateAndStoreOldPattern() {
        auto pattern = CreateTestPattern();
        PatternID id = pattern.GetID();
        pattern_db_->Store(std::move(pattern));

        // Sleep briefly to ensure some time passes
        // Note: For truly old patterns, we'd need to modify creation_timestamp_
        // which is private. For testing, we'll work with the age check logic.
        return id;
    }

    std::unique_ptr<MemoryBackend> pattern_db_;
    std::unique_ptr<AssociationMatrix> assoc_matrix_;
    std::unique_ptr<PatternPruner> pruner_;
};

// ============================================================================
// Config Validation Tests (8 tests)
// ============================================================================

TEST_F(PatternPrunerTest, Config_Valid) {
    PatternPruner::Config config;
    config.utility_threshold = 0.3f;
    config.min_associations_for_hub = 50;
    config.min_pattern_age = std::chrono::hours(24);
    config.strong_association_threshold = 0.7f;
    config.merge_similarity_threshold = 0.95f;
    config.max_prune_batch = 1000;

    EXPECT_TRUE(config.IsValid());
}

TEST_F(PatternPrunerTest, Config_InvalidUtilityThresholdNegative) {
    PatternPruner::Config config;
    config.utility_threshold = -0.1f;
    EXPECT_FALSE(config.IsValid());
}

TEST_F(PatternPrunerTest, Config_InvalidUtilityThresholdTooHigh) {
    PatternPruner::Config config;
    config.utility_threshold = 1.5f;
    EXPECT_FALSE(config.IsValid());
}

TEST_F(PatternPrunerTest, Config_InvalidMinAssociationsZero) {
    PatternPruner::Config config;
    config.min_associations_for_hub = 0;
    EXPECT_FALSE(config.IsValid());
}

TEST_F(PatternPrunerTest, Config_InvalidMinAssociationsTooHigh) {
    PatternPruner::Config config;
    config.min_associations_for_hub = 2000;
    EXPECT_FALSE(config.IsValid());
}

TEST_F(PatternPrunerTest, Config_InvalidPatternAgeNegative) {
    PatternPruner::Config config;
    config.min_pattern_age = std::chrono::hours(-1);
    EXPECT_FALSE(config.IsValid());
}

TEST_F(PatternPrunerTest, Config_InvalidStrongAssociationThreshold) {
    PatternPruner::Config config;
    config.strong_association_threshold = 1.5f;
    EXPECT_FALSE(config.IsValid());
}

TEST_F(PatternPrunerTest, Config_InvalidMaxPruneBatch) {
    PatternPruner::Config config;
    config.max_prune_batch = 0;
    EXPECT_FALSE(config.IsValid());

    config.max_prune_batch = 200000;
    EXPECT_FALSE(config.IsValid());
}

TEST_F(PatternPrunerTest, Config_ConstructorRejectsInvalid) {
    PatternPruner::Config invalid_config;
    invalid_config.utility_threshold = -0.5f;

    EXPECT_THROW(
        PatternPruner pruner(invalid_config),
        std::invalid_argument
    );
}

TEST_F(PatternPrunerTest, Config_SetConfigValid) {
    PatternPruner::Config new_config;
    new_config.utility_threshold = 0.5f;
    new_config.min_associations_for_hub = 100;

    EXPECT_NO_THROW(pruner_->SetConfig(new_config));
    EXPECT_FLOAT_EQ(0.5f, pruner_->GetConfig().utility_threshold);
    EXPECT_EQ(100u, pruner_->GetConfig().min_associations_for_hub);
}

TEST_F(PatternPrunerTest, Config_SetConfigInvalid) {
    PatternPruner::Config invalid_config;
    invalid_config.utility_threshold = 2.0f;

    EXPECT_THROW(pruner_->SetConfig(invalid_config), std::invalid_argument);
}

// ============================================================================
// Safety Checks - IsHub Tests (3 tests)
// ============================================================================

TEST_F(PatternPrunerTest, IsHub_WithHubPattern) {
    PatternID hub_pattern = PatternID::Generate();

    // Create 60 associations (50 outgoing, 10 incoming)
    // This exceeds the default min_associations_for_hub (50)
    for (int i = 0; i < 50; ++i) {
        PatternID target = PatternID::Generate();
        AssociationEdge edge(hub_pattern, target, AssociationType::CAUSAL, 0.5f);
        assoc_matrix_->AddAssociation(edge);
    }

    for (int i = 0; i < 10; ++i) {
        PatternID source = PatternID::Generate();
        AssociationEdge edge(source, hub_pattern, AssociationType::CAUSAL, 0.5f);
        assoc_matrix_->AddAssociation(edge);
    }

    EXPECT_TRUE(pruner_->IsHub(hub_pattern, *assoc_matrix_));
}

TEST_F(PatternPrunerTest, IsHub_WithNonHubPattern) {
    PatternID pattern = PatternID::Generate();

    // Create only 5 associations (well below threshold)
    for (int i = 0; i < 5; ++i) {
        PatternID target = PatternID::Generate();
        AssociationEdge edge(pattern, target, AssociationType::CAUSAL, 0.5f);
        assoc_matrix_->AddAssociation(edge);
    }

    EXPECT_FALSE(pruner_->IsHub(pattern, *assoc_matrix_));
}

TEST_F(PatternPrunerTest, IsHub_ExactlyAtThreshold) {
    // Create pruner with specific hub threshold
    PatternPruner::Config config;
    config.min_associations_for_hub = 10;
    PatternPruner pruner(config);

    PatternID pattern = PatternID::Generate();

    // Create exactly 10 associations
    for (int i = 0; i < 10; ++i) {
        PatternID target = PatternID::Generate();
        AssociationEdge edge(pattern, target, AssociationType::CAUSAL, 0.5f);
        assoc_matrix_->AddAssociation(edge);
    }

    EXPECT_TRUE(pruner.IsHub(pattern, *assoc_matrix_));
}

// ============================================================================
// Safety Checks - IsRecentlyCreated Tests (2 tests)
// ============================================================================

TEST_F(PatternPrunerTest, IsRecentlyCreated_YoungPattern) {
    // Create a fresh pattern
    auto pattern = CreateTestPattern(Timestamp::Now());

    EXPECT_TRUE(pruner_->IsRecentlyCreated(pattern));
}

TEST_F(PatternPrunerTest, IsRecentlyCreated_OldPattern) {
    // Create pruner with very short age requirement for testing
    PatternPruner::Config config;
    config.min_pattern_age = std::chrono::milliseconds(10);
    PatternPruner pruner(config);

    // Create pattern and wait
    auto pattern = CreateTestPattern(Timestamp::Now());
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    EXPECT_FALSE(pruner.IsRecentlyCreated(pattern));
}

// ============================================================================
// Safety Checks - HasStrongAssociations Tests (3 tests)
// ============================================================================

TEST_F(PatternPrunerTest, HasStrongAssociations_WithStrongOutgoing) {
    PatternID pattern = PatternID::Generate();
    PatternID target = PatternID::Generate();

    // Create strong outgoing association (0.8 > 0.7 threshold)
    AssociationEdge edge(pattern, target, AssociationType::CAUSAL, 0.8f);
    assoc_matrix_->AddAssociation(edge);

    EXPECT_TRUE(pruner_->HasStrongAssociations(pattern, *assoc_matrix_));
}

TEST_F(PatternPrunerTest, HasStrongAssociations_WithStrongIncoming) {
    PatternID pattern = PatternID::Generate();
    PatternID source = PatternID::Generate();

    // Create strong incoming association
    AssociationEdge edge(source, pattern, AssociationType::CAUSAL, 0.9f);
    assoc_matrix_->AddAssociation(edge);

    EXPECT_TRUE(pruner_->HasStrongAssociations(pattern, *assoc_matrix_));
}

TEST_F(PatternPrunerTest, HasStrongAssociations_OnlyWeakAssociations) {
    PatternID pattern = PatternID::Generate();

    // Create several weak associations (all below 0.7 threshold)
    for (int i = 0; i < 10; ++i) {
        PatternID target = PatternID::Generate();
        AssociationEdge edge(pattern, target, AssociationType::CAUSAL, 0.5f);
        assoc_matrix_->AddAssociation(edge);
    }

    EXPECT_FALSE(pruner_->HasStrongAssociations(pattern, *assoc_matrix_));
}

// ============================================================================
// Safety Checks - IsSafeToPrune Tests (6 tests)
// ============================================================================

TEST_F(PatternPrunerTest, IsSafeToPrune_LowUtilityNoRestrictions) {
    // Create an old pattern with low utility, no hub status, no strong associations
    PatternPruner::Config config;
    config.utility_threshold = 0.2f;
    config.min_pattern_age = std::chrono::milliseconds(10);
    PatternPruner pruner(config);

    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    // Wait for pattern to age
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    float utility = 0.1f;  // Below threshold

    EXPECT_TRUE(pruner.IsSafeToPrune(id, pattern, *assoc_matrix_, utility));
}

TEST_F(PatternPrunerTest, IsSafeToPrune_HighUtility) {
    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    float utility = 0.8f;  // Above threshold (0.2)

    EXPECT_FALSE(pruner_->IsSafeToPrune(id, pattern, *assoc_matrix_, utility));
}

TEST_F(PatternPrunerTest, IsSafeToPrune_IsHub) {
    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    // Make it a hub
    for (int i = 0; i < 60; ++i) {
        PatternID target = PatternID::Generate();
        AssociationEdge edge(id, target, AssociationType::CAUSAL, 0.5f);
        assoc_matrix_->AddAssociation(edge);
    }

    float utility = 0.1f;  // Low utility

    EXPECT_FALSE(pruner_->IsSafeToPrune(id, pattern, *assoc_matrix_, utility));
}

TEST_F(PatternPrunerTest, IsSafeToPrune_RecentlyCreated) {
    auto pattern = CreateTestPattern(Timestamp::Now());
    PatternID id = pattern.GetID();

    float utility = 0.1f;  // Low utility

    EXPECT_FALSE(pruner_->IsSafeToPrune(id, pattern, *assoc_matrix_, utility));
}

TEST_F(PatternPrunerTest, IsSafeToPrune_HasStrongAssociations) {
    // Use short age requirement for testing
    PatternPruner::Config config;
    config.utility_threshold = 0.2f;
    config.min_pattern_age = std::chrono::milliseconds(10);
    config.strong_association_threshold = 0.7f;
    PatternPruner pruner(config);

    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    PatternID target = PatternID::Generate();

    // Create strong association
    AssociationEdge edge(id, target, AssociationType::CAUSAL, 0.9f);
    assoc_matrix_->AddAssociation(edge);

    // Wait for pattern to age
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    float utility = 0.1f;  // Low utility

    EXPECT_FALSE(pruner.IsSafeToPrune(id, pattern, *assoc_matrix_, utility));
}

TEST_F(PatternPrunerTest, IsSafeToPrune_UtilityAtThreshold) {
    PatternPruner::Config config;
    config.utility_threshold = 0.2f;
    config.min_pattern_age = std::chrono::milliseconds(10);
    PatternPruner pruner(config);

    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    float utility = 0.2f;  // Exactly at threshold

    // Should NOT be safe to prune (utility >= threshold)
    EXPECT_FALSE(pruner.IsSafeToPrune(id, pattern, *assoc_matrix_, utility));
}

// ============================================================================
// Pattern Pruning Tests (6 tests)
// ============================================================================

TEST_F(PatternPrunerTest, PrunePattern_Success) {
    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    pattern_db_->Store(std::move(pattern));

    ASSERT_TRUE(pattern_db_->Exists(id));

    // Retrieve for pruning
    auto pattern_opt = pattern_db_->Retrieve(id);
    ASSERT_TRUE(pattern_opt.has_value());

    bool pruned = pruner_->PrunePattern(id, *pattern_opt, *pattern_db_, *assoc_matrix_, 0.1f);

    EXPECT_TRUE(pruned);
    EXPECT_FALSE(pattern_db_->Exists(id));
}

TEST_F(PatternPrunerTest, PrunePattern_RemovesOutgoingAssociations) {
    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    pattern_db_->Store(std::move(pattern));

    // Create outgoing associations
    std::vector<PatternID> targets;
    for (int i = 0; i < 5; ++i) {
        PatternID target = PatternID::Generate();
        targets.push_back(target);
        AssociationEdge edge(id, target, AssociationType::CAUSAL, 0.5f);
        assoc_matrix_->AddAssociation(edge);
    }

    // Verify associations exist
    for (const auto& target : targets) {
        ASSERT_TRUE(assoc_matrix_->HasAssociation(id, target));
    }

    // Prune pattern
    auto pattern_opt = pattern_db_->Retrieve(id);
    pruner_->PrunePattern(id, *pattern_opt, *pattern_db_, *assoc_matrix_, 0.1f);

    // Verify associations removed
    for (const auto& target : targets) {
        EXPECT_FALSE(assoc_matrix_->HasAssociation(id, target));
    }
}

TEST_F(PatternPrunerTest, PrunePattern_RemovesIncomingAssociations) {
    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    pattern_db_->Store(std::move(pattern));

    // Create incoming associations
    std::vector<PatternID> sources;
    for (int i = 0; i < 5; ++i) {
        PatternID source = PatternID::Generate();
        sources.push_back(source);
        AssociationEdge edge(source, id, AssociationType::CAUSAL, 0.5f);
        assoc_matrix_->AddAssociation(edge);
    }

    // Verify associations exist
    for (const auto& source : sources) {
        ASSERT_TRUE(assoc_matrix_->HasAssociation(source, id));
    }

    // Prune pattern
    auto pattern_opt = pattern_db_->Retrieve(id);
    pruner_->PrunePattern(id, *pattern_opt, *pattern_db_, *assoc_matrix_, 0.1f);

    // Verify associations removed
    for (const auto& source : sources) {
        EXPECT_FALSE(assoc_matrix_->HasAssociation(source, id));
    }
}

TEST_F(PatternPrunerTest, PrunePattern_RemovesBothDirections) {
    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    pattern_db_->Store(std::move(pattern));

    // Create both incoming and outgoing associations
    PatternID target = PatternID::Generate();
    PatternID source = PatternID::Generate();

    AssociationEdge outgoing(id, target, AssociationType::CAUSAL, 0.5f);
    AssociationEdge incoming(source, id, AssociationType::CAUSAL, 0.5f);

    assoc_matrix_->AddAssociation(outgoing);
    assoc_matrix_->AddAssociation(incoming);

    ASSERT_TRUE(assoc_matrix_->HasAssociation(id, target));
    ASSERT_TRUE(assoc_matrix_->HasAssociation(source, id));

    // Prune pattern
    auto pattern_opt = pattern_db_->Retrieve(id);
    pruner_->PrunePattern(id, *pattern_opt, *pattern_db_, *assoc_matrix_, 0.1f);

    // Verify all associations removed
    EXPECT_FALSE(assoc_matrix_->HasAssociation(id, target));
    EXPECT_FALSE(assoc_matrix_->HasAssociation(source, id));
    EXPECT_FALSE(pattern_db_->Exists(id));
}

TEST_F(PatternPrunerTest, PrunePattern_NonExistentPattern) {
    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    // Don't store pattern in database

    bool pruned = pruner_->PrunePattern(id, pattern, *pattern_db_, *assoc_matrix_, 0.1f);

    // Should return false (pattern not found)
    EXPECT_FALSE(pruned);
}

TEST_F(PatternPrunerTest, PrunePattern_MultipleAssociations) {
    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    pattern_db_->Store(std::move(pattern));

    // Create many associations
    for (int i = 0; i < 20; ++i) {
        PatternID target = PatternID::Generate();
        AssociationEdge edge(id, target, AssociationType::CAUSAL, 0.3f + (i * 0.01f));
        assoc_matrix_->AddAssociation(edge);
    }

    size_t initial_count = assoc_matrix_->GetAssociationCount();
    EXPECT_EQ(20u, initial_count);

    // Prune pattern
    auto pattern_opt = pattern_db_->Retrieve(id);
    pruner_->PrunePattern(id, *pattern_opt, *pattern_db_, *assoc_matrix_, 0.1f);

    // All associations should be removed
    EXPECT_EQ(0u, assoc_matrix_->GetAssociationCount());
}

// ============================================================================
// Batch Pruning Tests (5 tests)
// ============================================================================

TEST_F(PatternPrunerTest, PrunePatterns_MultipleCandidates) {
    PatternPruner::Config config;
    config.utility_threshold = 0.3f;
    config.min_pattern_age = std::chrono::milliseconds(10);
    PatternPruner pruner(config);

    // Create multiple low-utility patterns
    std::unordered_map<PatternID, float> utilities;
    std::vector<PatternID> low_utility_ids;

    for (int i = 0; i < 5; ++i) {
        auto pattern = CreateTestPattern();
        PatternID id = pattern.GetID();
        pattern_db_->Store(std::move(pattern));
        utilities[id] = 0.1f;  // Low utility
        low_utility_ids.push_back(id);
    }

    // Wait for patterns to age
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    auto result = pruner.PrunePatterns(*pattern_db_, *assoc_matrix_, utilities);

    EXPECT_EQ(5u, result.pruned_patterns.size());

    // Verify all patterns were pruned
    for (const auto& id : low_utility_ids) {
        EXPECT_FALSE(pattern_db_->Exists(id));
    }
}

TEST_F(PatternPrunerTest, PrunePatterns_BatchSizeLimit) {
    PatternPruner::Config config;
    config.utility_threshold = 0.3f;
    config.min_pattern_age = std::chrono::milliseconds(10);
    config.max_prune_batch = 3;  // Limit to 3 patterns per batch
    PatternPruner pruner(config);

    // Create 10 low-utility patterns
    std::unordered_map<PatternID, float> utilities;

    for (int i = 0; i < 10; ++i) {
        auto pattern = CreateTestPattern();
        PatternID id = pattern.GetID();
        pattern_db_->Store(std::move(pattern));
        utilities[id] = 0.1f;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    auto result = pruner.PrunePatterns(*pattern_db_, *assoc_matrix_, utilities);

    // Should only process 3 patterns (batch size limit)
    EXPECT_LE(result.pruned_patterns.size(), 3u);
}

TEST_F(PatternPrunerTest, PrunePatterns_EmptyUtilities) {
    std::unordered_map<PatternID, float> empty_utilities;

    auto result = pruner_->PrunePatterns(*pattern_db_, *assoc_matrix_, empty_utilities);

    EXPECT_EQ(0u, result.pruned_patterns.size());
    EXPECT_EQ(0u, result.patterns_kept_safe);
}

TEST_F(PatternPrunerTest, PrunePatterns_SafetyChecksPreventPruning) {
    PatternPruner::Config config;
    config.utility_threshold = 0.3f;
    config.min_pattern_age = std::chrono::milliseconds(10);
    config.min_associations_for_hub = 5;
    PatternPruner pruner(config);

    // Create a pattern that's a hub (should not be pruned)
    auto hub_pattern = CreateTestPattern();
    PatternID hub_id = hub_pattern.GetID();
    pattern_db_->Store(std::move(hub_pattern));

    // Make it a hub
    for (int i = 0; i < 10; ++i) {
        PatternID target = PatternID::Generate();
        AssociationEdge edge(hub_id, target, AssociationType::CAUSAL, 0.5f);
        assoc_matrix_->AddAssociation(edge);
    }

    std::unordered_map<PatternID, float> utilities;
    utilities[hub_id] = 0.1f;  // Low utility

    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    auto result = pruner.PrunePatterns(*pattern_db_, *assoc_matrix_, utilities);

    EXPECT_EQ(0u, result.pruned_patterns.size());
    EXPECT_EQ(1u, result.patterns_kept_safe);
    EXPECT_TRUE(pattern_db_->Exists(hub_id));
}

TEST_F(PatternPrunerTest, PrunePatterns_MixedUtilities) {
    PatternPruner::Config config;
    config.utility_threshold = 0.3f;
    config.min_pattern_age = std::chrono::milliseconds(10);
    PatternPruner pruner(config);

    std::unordered_map<PatternID, float> utilities;
    std::vector<PatternID> low_utility_ids;
    std::vector<PatternID> high_utility_ids;

    // Create patterns with low utility
    for (int i = 0; i < 3; ++i) {
        auto pattern = CreateTestPattern();
        PatternID id = pattern.GetID();
        pattern_db_->Store(std::move(pattern));
        utilities[id] = 0.1f;
        low_utility_ids.push_back(id);
    }

    // Create patterns with high utility
    for (int i = 0; i < 3; ++i) {
        auto pattern = CreateTestPattern();
        PatternID id = pattern.GetID();
        pattern_db_->Store(std::move(pattern));
        utilities[id] = 0.8f;
        high_utility_ids.push_back(id);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    auto result = pruner.PrunePatterns(*pattern_db_, *assoc_matrix_, utilities);

    // Only low utility patterns should be pruned
    EXPECT_EQ(3u, result.pruned_patterns.size());

    for (const auto& id : low_utility_ids) {
        EXPECT_FALSE(pattern_db_->Exists(id));
    }

    for (const auto& id : high_utility_ids) {
        EXPECT_TRUE(pattern_db_->Exists(id));
    }
}

// ============================================================================
// Pattern Merging Tests (4 tests)
// ============================================================================

TEST_F(PatternPrunerTest, MergePatterns_TransfersOutgoingAssociations) {
    // Create patterns and store them
    auto old_p = CreateTestPattern();
    auto new_p = CreateTestPattern();
    PatternID old_pattern = old_p.GetID();
    PatternID new_pattern = new_p.GetID();
    PatternID target1 = PatternID::Generate();
    PatternID target2 = PatternID::Generate();

    pattern_db_->Store(std::move(old_p));
    pattern_db_->Store(std::move(new_p));

    // Create outgoing associations from old pattern
    AssociationEdge edge1(old_pattern, target1, AssociationType::CAUSAL, 0.6f);
    AssociationEdge edge2(old_pattern, target2, AssociationType::CATEGORICAL, 0.7f);
    assoc_matrix_->AddAssociation(edge1);
    assoc_matrix_->AddAssociation(edge2);

    ASSERT_TRUE(assoc_matrix_->HasAssociation(old_pattern, target1));
    ASSERT_TRUE(assoc_matrix_->HasAssociation(old_pattern, target2));

    // Merge patterns
    bool merged = pruner_->MergePatterns(old_pattern, new_pattern, *pattern_db_, *assoc_matrix_);

    EXPECT_TRUE(merged);

    // Verify associations transferred
    EXPECT_TRUE(assoc_matrix_->HasAssociation(new_pattern, target1));
    EXPECT_TRUE(assoc_matrix_->HasAssociation(new_pattern, target2));

    // Verify old associations removed
    EXPECT_FALSE(assoc_matrix_->HasAssociation(old_pattern, target1));
    EXPECT_FALSE(assoc_matrix_->HasAssociation(old_pattern, target2));

    // Verify old pattern deleted
    EXPECT_FALSE(pattern_db_->Exists(old_pattern));
}

TEST_F(PatternPrunerTest, MergePatterns_TransfersIncomingAssociations) {
    // Create patterns and store them
    auto old_p = CreateTestPattern();
    auto new_p = CreateTestPattern();
    PatternID old_pattern = old_p.GetID();
    PatternID new_pattern = new_p.GetID();
    PatternID source1 = PatternID::Generate();
    PatternID source2 = PatternID::Generate();

    pattern_db_->Store(std::move(old_p));
    pattern_db_->Store(std::move(new_p));

    // Create incoming associations to old pattern
    AssociationEdge edge1(source1, old_pattern, AssociationType::CAUSAL, 0.6f);
    AssociationEdge edge2(source2, old_pattern, AssociationType::SPATIAL, 0.8f);
    assoc_matrix_->AddAssociation(edge1);
    assoc_matrix_->AddAssociation(edge2);

    // Merge patterns
    bool merged = pruner_->MergePatterns(old_pattern, new_pattern, *pattern_db_, *assoc_matrix_);

    EXPECT_TRUE(merged);

    // Verify associations transferred
    EXPECT_TRUE(assoc_matrix_->HasAssociation(source1, new_pattern));
    EXPECT_TRUE(assoc_matrix_->HasAssociation(source2, new_pattern));

    // Verify old associations removed
    EXPECT_FALSE(assoc_matrix_->HasAssociation(source1, old_pattern));
    EXPECT_FALSE(assoc_matrix_->HasAssociation(source2, old_pattern));
}

TEST_F(PatternPrunerTest, MergePatterns_AvoidsSelfLoops) {
    // Create patterns and store them
    auto old_p = CreateTestPattern();
    auto new_p = CreateTestPattern();
    PatternID old_pattern = old_p.GetID();
    PatternID new_pattern = new_p.GetID();

    pattern_db_->Store(std::move(old_p));
    pattern_db_->Store(std::move(new_p));

    // Create association from old to new (would create self-loop)
    AssociationEdge edge(old_pattern, new_pattern, AssociationType::CAUSAL, 0.6f);
    assoc_matrix_->AddAssociation(edge);

    ASSERT_TRUE(assoc_matrix_->HasAssociation(old_pattern, new_pattern));

    // Merge patterns
    pruner_->MergePatterns(old_pattern, new_pattern, *pattern_db_, *assoc_matrix_);

    // Should NOT create self-loop
    EXPECT_FALSE(assoc_matrix_->HasAssociation(new_pattern, new_pattern));
}

TEST_F(PatternPrunerTest, MergePatterns_PreservesAssociationStrength) {
    // Create patterns and store them
    auto old_p = CreateTestPattern();
    auto new_p = CreateTestPattern();
    PatternID old_pattern = old_p.GetID();
    PatternID new_pattern = new_p.GetID();
    PatternID target = PatternID::Generate();

    pattern_db_->Store(std::move(old_p));
    pattern_db_->Store(std::move(new_p));

    // Create association with specific strength
    float original_strength = 0.75f;
    AssociationEdge edge(old_pattern, target, AssociationType::CAUSAL, original_strength);
    assoc_matrix_->AddAssociation(edge);

    // Merge patterns
    pruner_->MergePatterns(old_pattern, new_pattern, *pattern_db_, *assoc_matrix_);

    // Verify strength preserved
    const auto* transferred_edge = assoc_matrix_->GetAssociation(new_pattern, target);
    ASSERT_NE(nullptr, transferred_edge);
    EXPECT_FLOAT_EQ(original_strength, transferred_edge->GetStrength());
}

// ============================================================================
// Statistics Tests (3 tests)
// ============================================================================

TEST_F(PatternPrunerTest, PruneResult_TracksSuccessfulPruning) {
    PatternPruner::Config config;
    config.utility_threshold = 0.3f;
    config.min_pattern_age = std::chrono::milliseconds(10);
    PatternPruner pruner(config);

    std::unordered_map<PatternID, float> utilities;

    // Create 3 low-utility patterns
    for (int i = 0; i < 3; ++i) {
        auto pattern = CreateTestPattern();
        PatternID id = pattern.GetID();
        pattern_db_->Store(std::move(pattern));
        utilities[id] = 0.1f;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    auto result = pruner.PrunePatterns(*pattern_db_, *assoc_matrix_, utilities);

    EXPECT_EQ(3u, result.pruned_patterns.size());
    EXPECT_GT(result.bytes_freed, 0u);
}

TEST_F(PatternPrunerTest, PruneResult_BytesFreedCalculation) {
    PatternPruner::Config config;
    config.utility_threshold = 0.3f;
    config.min_pattern_age = std::chrono::milliseconds(10);
    PatternPruner pruner(config);

    std::unordered_map<PatternID, float> utilities;

    // Create patterns with different sizes
    for (int i = 0; i < 5; ++i) {
        FeatureVector fv(10 + i * 5);  // Different sizes
        for (size_t j = 0; j < fv.Dimension(); ++j) {
            fv[j] = static_cast<float>(j);
        }
        PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
        PatternNode pattern(PatternID::Generate(), data, PatternType::ATOMIC);
        PatternID id = pattern.GetID();
        utilities[id] = 0.1f;
        pattern_db_->Store(std::move(pattern));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    auto result = pruner.PrunePatterns(*pattern_db_, *assoc_matrix_, utilities);

    // Should track bytes freed
    EXPECT_GT(result.bytes_freed, 0u);
    // Bytes freed should be reasonable (at least base object size * count)
    EXPECT_GT(result.bytes_freed, result.pruned_patterns.size() * sizeof(PatternNode));
}

TEST_F(PatternPrunerTest, PruneResult_TracksPatternsKeptSafe) {
    PatternPruner::Config config;
    config.utility_threshold = 0.3f;
    config.min_pattern_age = std::chrono::hours(24);  // Long age requirement
    PatternPruner pruner(config);

    std::unordered_map<PatternID, float> utilities;

    // Create fresh patterns (will be kept due to age)
    for (int i = 0; i < 3; ++i) {
        auto pattern = CreateTestPattern();
        PatternID id = pattern.GetID();
        pattern_db_->Store(std::move(pattern));
        utilities[id] = 0.1f;
    }

    auto result = pruner.PrunePatterns(*pattern_db_, *assoc_matrix_, utilities);

    EXPECT_EQ(0u, result.pruned_patterns.size());
    EXPECT_EQ(3u, result.patterns_kept_safe);
}

// ============================================================================
// Edge Cases Tests (4 tests)
// ============================================================================

TEST_F(PatternPrunerTest, EdgeCase_NonExistentPattern) {
    std::unordered_map<PatternID, float> utilities;
    PatternID fake_id = PatternID::Generate();
    utilities[fake_id] = 0.1f;

    auto result = pruner_->PrunePatterns(*pattern_db_, *assoc_matrix_, utilities);

    // Should handle gracefully (skip non-existent pattern)
    EXPECT_EQ(0u, result.pruned_patterns.size());
}

TEST_F(PatternPrunerTest, EdgeCase_AllPatternsKeptSafe) {
    std::unordered_map<PatternID, float> utilities;

    // Create patterns with low utility but all recently created
    // (should be candidates but kept safe due to age check)
    for (int i = 0; i < 5; ++i) {
        auto pattern = CreateTestPattern();
        PatternID id = pattern.GetID();
        pattern_db_->Store(std::move(pattern));
        utilities[id] = 0.1f;  // Low utility (below threshold)
    }

    // Don't wait - patterns are freshly created and should be kept safe
    auto result = pruner_->PrunePatterns(*pattern_db_, *assoc_matrix_, utilities);

    // All patterns should be candidates but kept safe due to recent creation
    EXPECT_EQ(0u, result.pruned_patterns.size());
    EXPECT_EQ(5u, result.patterns_kept_safe);
}

TEST_F(PatternPrunerTest, EdgeCase_PatternWithNoAssociations) {
    PatternPruner::Config config;
    config.utility_threshold = 0.3f;
    config.min_pattern_age = std::chrono::milliseconds(10);
    PatternPruner pruner(config);

    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    pattern_db_->Store(std::move(pattern));

    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    std::unordered_map<PatternID, float> utilities;
    utilities[id] = 0.1f;

    auto result = pruner.PrunePatterns(*pattern_db_, *assoc_matrix_, utilities);

    // Should successfully prune pattern with no associations
    EXPECT_EQ(1u, result.pruned_patterns.size());
    EXPECT_FALSE(pattern_db_->Exists(id));
}

TEST_F(PatternPrunerTest, EdgeCase_ZeroUtility) {
    PatternPruner::Config config;
    config.utility_threshold = 0.3f;
    config.min_pattern_age = std::chrono::milliseconds(10);
    PatternPruner pruner(config);

    auto pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    pattern_db_->Store(std::move(pattern));

    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    std::unordered_map<PatternID, float> utilities;
    utilities[id] = 0.0f;  // Zero utility

    auto result = pruner.PrunePatterns(*pattern_db_, *assoc_matrix_, utilities);

    // Should prune pattern with zero utility
    EXPECT_EQ(1u, result.pruned_patterns.size());
    EXPECT_FALSE(pattern_db_->Exists(id));
}
