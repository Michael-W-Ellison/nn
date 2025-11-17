// File: tests/memory/memory_tier_test.cpp
//
// Comprehensive unit tests for Memory Tier system
//
// Tests all 4 tiers (Active, Warm, Cold, Archive) and validates:
// - Tier creation and initialization
// - Pattern operations (Store, Load, Remove, Has)
// - Association operations (Store, Load, Remove, Has)
// - Batch operations
// - Statistics and metrics
// - Tier information
// - Maintenance operations
// - Utility functions

#include "memory/memory_tier.hpp"
#include "core/pattern_node.hpp"
#include "core/pattern_data.hpp"
#include "association/association_edge.hpp"
#include "core/types.hpp"
#include <gtest/gtest.h>
#include <filesystem>
#include <vector>
#include <memory>

using namespace dpan;
namespace fs = std::filesystem;

// ============================================================================
// Test Fixtures
// ============================================================================

class ActiveTierTest : public ::testing::Test {
protected:
    void SetUp() override {
        tier_ = CreateActiveTier();
    }

    void TearDown() override {
        tier_.reset();
    }

    std::unique_ptr<IMemoryTier> tier_;
};

class WarmTierTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary directory for testing
        temp_dir_ = fs::temp_directory_path() / "dpan_warm_tier_test";
        fs::create_directories(temp_dir_);
        tier_ = CreateWarmTier(temp_dir_.string());
    }

    void TearDown() override {
        tier_.reset();
        if (fs::exists(temp_dir_)) {
            fs::remove_all(temp_dir_);
        }
    }

    std::unique_ptr<IMemoryTier> tier_;
    fs::path temp_dir_;
};

class ColdTierTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary directory for testing
        temp_dir_ = fs::temp_directory_path() / "dpan_cold_tier_test";
        fs::create_directories(temp_dir_);
        tier_ = CreateColdTier(temp_dir_.string());
    }

    void TearDown() override {
        tier_.reset();
        if (fs::exists(temp_dir_)) {
            fs::remove_all(temp_dir_);
        }
    }

    std::unique_ptr<IMemoryTier> tier_;
    fs::path temp_dir_;
};

class ArchiveTierTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary directory for testing
        temp_dir_ = fs::temp_directory_path() / "dpan_archive_tier_test";
        fs::create_directories(temp_dir_);
        tier_ = CreateArchiveTier(temp_dir_.string());
    }

    void TearDown() override {
        tier_.reset();
        if (fs::exists(temp_dir_)) {
            fs::remove_all(temp_dir_);
        }
    }

    std::unique_ptr<IMemoryTier> tier_;
    fs::path temp_dir_;
};

// ============================================================================
// Helper Functions
// ============================================================================

// Create a test pattern with specific data
PatternNode CreateTestPattern(PatternID id) {
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;
    PatternData data = PatternData::FromFeatures(fv, DataModality::NUMERIC);
    return PatternNode(id, data, PatternType::ATOMIC);
}

// Create a test pattern with auto-generated ID
PatternNode CreateTestPattern() {
    return CreateTestPattern(PatternID::Generate());
}

// Create a test association
AssociationEdge CreateTestAssociation(PatternID source, PatternID target) {
    return AssociationEdge(source, target, AssociationType::CAUSAL, 0.7f);
}

// ============================================================================
// Tier Creation and Initialization Tests (4 tests)
// ============================================================================

TEST_F(ActiveTierTest, CreationAndInitialization) {
    ASSERT_NE(nullptr, tier_);
    EXPECT_EQ(MemoryTier::ACTIVE, tier_->GetTierLevel());
    EXPECT_EQ("Active", tier_->GetTierName());
    EXPECT_EQ(0u, tier_->GetPatternCount());
    EXPECT_EQ(0u, tier_->GetAssociationCount());
}

TEST_F(WarmTierTest, CreationAndInitialization) {
    ASSERT_NE(nullptr, tier_);
    EXPECT_EQ(MemoryTier::WARM, tier_->GetTierLevel());
    EXPECT_EQ("Warm", tier_->GetTierName());
    EXPECT_EQ(0u, tier_->GetPatternCount());
    EXPECT_EQ(0u, tier_->GetAssociationCount());

    // Verify directories were created
    EXPECT_TRUE(fs::exists(temp_dir_));
}

TEST_F(ColdTierTest, CreationAndInitialization) {
    ASSERT_NE(nullptr, tier_);
    EXPECT_EQ(MemoryTier::COLD, tier_->GetTierLevel());
    EXPECT_EQ("Cold", tier_->GetTierName());
    EXPECT_EQ(0u, tier_->GetPatternCount());
    EXPECT_EQ(0u, tier_->GetAssociationCount());

    // Verify directories were created
    EXPECT_TRUE(fs::exists(temp_dir_));
}

TEST_F(ArchiveTierTest, CreationAndInitialization) {
    ASSERT_NE(nullptr, tier_);
    EXPECT_EQ(MemoryTier::ARCHIVE, tier_->GetTierLevel());
    EXPECT_EQ("Archive", tier_->GetTierName());
    EXPECT_EQ(0u, tier_->GetPatternCount());
    EXPECT_EQ(0u, tier_->GetAssociationCount());

    // Verify directories were created
    EXPECT_TRUE(fs::exists(temp_dir_));
}

// ============================================================================
// Pattern Operation Tests (8+ tests across tiers)
// ============================================================================

TEST_F(ActiveTierTest, StoreAndLoadPattern) {
    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    // Store pattern
    EXPECT_TRUE(tier_->StorePattern(pattern));
    EXPECT_EQ(1u, tier_->GetPatternCount());

    // Load pattern
    auto loaded = tier_->LoadPattern(id);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(id, loaded->GetID());
}

TEST_F(WarmTierTest, StoreAndLoadPattern) {
    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    // Store pattern
    EXPECT_TRUE(tier_->StorePattern(pattern));
    EXPECT_EQ(1u, tier_->GetPatternCount());

    // Load pattern
    auto loaded = tier_->LoadPattern(id);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(id, loaded->GetID());
}

TEST_F(ActiveTierTest, RemovePattern) {
    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    // Store and verify
    EXPECT_TRUE(tier_->StorePattern(pattern));
    EXPECT_TRUE(tier_->HasPattern(id));
    EXPECT_EQ(1u, tier_->GetPatternCount());

    // Remove and verify
    EXPECT_TRUE(tier_->RemovePattern(id));
    EXPECT_FALSE(tier_->HasPattern(id));
    EXPECT_EQ(0u, tier_->GetPatternCount());

    // Removing non-existent pattern should return false
    EXPECT_FALSE(tier_->RemovePattern(id));
}

TEST_F(WarmTierTest, RemovePattern) {
    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    // Store and verify
    EXPECT_TRUE(tier_->StorePattern(pattern));
    EXPECT_TRUE(tier_->HasPattern(id));

    // Remove and verify
    EXPECT_TRUE(tier_->RemovePattern(id));
    EXPECT_FALSE(tier_->HasPattern(id));

    // Removing non-existent pattern should return false
    EXPECT_FALSE(tier_->RemovePattern(id));
}

TEST_F(ActiveTierTest, HasPattern) {
    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    PatternID nonexistent_id = PatternID::Generate();

    // Should not exist initially
    EXPECT_FALSE(tier_->HasPattern(id));
    EXPECT_FALSE(tier_->HasPattern(nonexistent_id));

    // Store pattern
    EXPECT_TRUE(tier_->StorePattern(pattern));

    // Should exist now
    EXPECT_TRUE(tier_->HasPattern(id));
    EXPECT_FALSE(tier_->HasPattern(nonexistent_id));
}

TEST_F(ActiveTierTest, LoadNonexistentPattern) {
    PatternID nonexistent_id = PatternID::Generate();

    auto loaded = tier_->LoadPattern(nonexistent_id);
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(ColdTierTest, StoreAndLoadPattern) {
    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    // Store pattern
    EXPECT_TRUE(tier_->StorePattern(pattern));
    EXPECT_EQ(1u, tier_->GetPatternCount());

    // Load pattern
    auto loaded = tier_->LoadPattern(id);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(id, loaded->GetID());
}

TEST_F(ArchiveTierTest, StoreAndLoadPattern) {
    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();

    // Store pattern
    EXPECT_TRUE(tier_->StorePattern(pattern));
    EXPECT_EQ(1u, tier_->GetPatternCount());

    // Load pattern
    auto loaded = tier_->LoadPattern(id);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(id, loaded->GetID());
}

TEST_F(ActiveTierTest, OverwritePattern) {
    PatternID id = PatternID::Generate();
    PatternNode pattern1 = CreateTestPattern(id);

    // Store first pattern
    EXPECT_TRUE(tier_->StorePattern(pattern1));
    EXPECT_EQ(1u, tier_->GetPatternCount());

    // Overwrite with second pattern (same ID)
    PatternNode pattern2 = CreateTestPattern(id);
    EXPECT_TRUE(tier_->StorePattern(pattern2));
    EXPECT_EQ(1u, tier_->GetPatternCount());  // Should still be 1

    // Load should return the latest version
    auto loaded = tier_->LoadPattern(id);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(id, loaded->GetID());
}

// ============================================================================
// Association Operation Tests (8+ tests across tiers)
// ============================================================================

TEST_F(ActiveTierTest, StoreAndLoadAssociation) {
    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();
    AssociationEdge edge = CreateTestAssociation(source, target);

    // Store association
    EXPECT_TRUE(tier_->StoreAssociation(edge));
    EXPECT_EQ(1u, tier_->GetAssociationCount());

    // Load association
    auto loaded = tier_->LoadAssociation(source, target);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(source, loaded->GetSource());
    EXPECT_EQ(target, loaded->GetTarget());
    EXPECT_EQ(AssociationType::CAUSAL, loaded->GetType());
}

TEST_F(WarmTierTest, StoreAndLoadAssociation) {
    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();
    AssociationEdge edge = CreateTestAssociation(source, target);

    // Store association
    EXPECT_TRUE(tier_->StoreAssociation(edge));
    EXPECT_EQ(1u, tier_->GetAssociationCount());

    // Load association
    auto loaded = tier_->LoadAssociation(source, target);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(source, loaded->GetSource());
    EXPECT_EQ(target, loaded->GetTarget());
}

TEST_F(ActiveTierTest, RemoveAssociation) {
    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();
    AssociationEdge edge = CreateTestAssociation(source, target);

    // Store and verify
    EXPECT_TRUE(tier_->StoreAssociation(edge));
    EXPECT_TRUE(tier_->HasAssociation(source, target));
    EXPECT_EQ(1u, tier_->GetAssociationCount());

    // Remove and verify
    EXPECT_TRUE(tier_->RemoveAssociation(source, target));
    EXPECT_FALSE(tier_->HasAssociation(source, target));
    EXPECT_EQ(0u, tier_->GetAssociationCount());

    // Removing non-existent association should return false
    EXPECT_FALSE(tier_->RemoveAssociation(source, target));
}

TEST_F(WarmTierTest, RemoveAssociation) {
    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();
    AssociationEdge edge = CreateTestAssociation(source, target);

    // Store and verify
    EXPECT_TRUE(tier_->StoreAssociation(edge));
    EXPECT_TRUE(tier_->HasAssociation(source, target));

    // Remove and verify
    EXPECT_TRUE(tier_->RemoveAssociation(source, target));
    EXPECT_FALSE(tier_->HasAssociation(source, target));
}

TEST_F(ActiveTierTest, HasAssociation) {
    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();
    AssociationEdge edge = CreateTestAssociation(source, target);

    // Should not exist initially
    EXPECT_FALSE(tier_->HasAssociation(source, target));

    // Store association
    EXPECT_TRUE(tier_->StoreAssociation(edge));

    // Should exist now
    EXPECT_TRUE(tier_->HasAssociation(source, target));

    // Different direction should not exist
    EXPECT_FALSE(tier_->HasAssociation(target, source));
}

TEST_F(ActiveTierTest, LoadNonexistentAssociation) {
    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();

    auto loaded = tier_->LoadAssociation(source, target);
    EXPECT_FALSE(loaded.has_value());
}

TEST_F(ColdTierTest, StoreAndLoadAssociation) {
    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();
    AssociationEdge edge = CreateTestAssociation(source, target);

    // Store association
    EXPECT_TRUE(tier_->StoreAssociation(edge));
    EXPECT_EQ(1u, tier_->GetAssociationCount());

    // Load association
    auto loaded = tier_->LoadAssociation(source, target);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(source, loaded->GetSource());
    EXPECT_EQ(target, loaded->GetTarget());
}

TEST_F(ArchiveTierTest, StoreAndLoadAssociation) {
    PatternID source = PatternID::Generate();
    PatternID target = PatternID::Generate();
    AssociationEdge edge = CreateTestAssociation(source, target);

    // Store association
    EXPECT_TRUE(tier_->StoreAssociation(edge));
    EXPECT_EQ(1u, tier_->GetAssociationCount());

    // Load association
    auto loaded = tier_->LoadAssociation(source, target);
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(source, loaded->GetSource());
    EXPECT_EQ(target, loaded->GetTarget());
}

TEST_F(ActiveTierTest, MultipleAssociations) {
    PatternID source1 = PatternID::Generate();
    PatternID target1 = PatternID::Generate();
    PatternID source2 = PatternID::Generate();
    PatternID target2 = PatternID::Generate();

    AssociationEdge edge1 = CreateTestAssociation(source1, target1);
    AssociationEdge edge2 = CreateTestAssociation(source2, target2);

    // Store multiple associations
    EXPECT_TRUE(tier_->StoreAssociation(edge1));
    EXPECT_TRUE(tier_->StoreAssociation(edge2));
    EXPECT_EQ(2u, tier_->GetAssociationCount());

    // Verify both exist
    EXPECT_TRUE(tier_->HasAssociation(source1, target1));
    EXPECT_TRUE(tier_->HasAssociation(source2, target2));
}

// ============================================================================
// Batch Operation Tests (4+ tests)
// ============================================================================

TEST_F(ActiveTierTest, StorePatternsBatch) {
    std::vector<PatternNode> patterns;
    std::vector<PatternID> ids;

    // Create multiple patterns
    for (int i = 0; i < 5; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        patterns.push_back(std::move(pattern));
    }

    // Store batch
    size_t stored = tier_->StorePatternsBatch(patterns);
    EXPECT_EQ(5u, stored);
    EXPECT_EQ(5u, tier_->GetPatternCount());

    // Verify all patterns exist
    for (const auto& id : ids) {
        EXPECT_TRUE(tier_->HasPattern(id));
    }
}

TEST_F(ActiveTierTest, LoadPatternsBatch) {
    std::vector<PatternNode> patterns;
    std::vector<PatternID> ids;

    // Create and store patterns
    for (int i = 0; i < 5; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        tier_->StorePattern(pattern);
    }

    // Load batch
    auto loaded = tier_->LoadPatternsBatch(ids);
    EXPECT_EQ(5u, loaded.size());

    // Verify all IDs match
    for (size_t i = 0; i < loaded.size(); ++i) {
        bool found = false;
        for (const auto& id : ids) {
            if (loaded[i].GetID() == id) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found);
    }
}

TEST_F(ActiveTierTest, LoadPatternsBatchPartial) {
    std::vector<PatternID> ids;

    // Store only some patterns
    for (int i = 0; i < 3; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        tier_->StorePattern(pattern);
    }

    // Add non-existent IDs
    ids.push_back(PatternID::Generate());
    ids.push_back(PatternID::Generate());

    // Load batch should return only existing patterns
    auto loaded = tier_->LoadPatternsBatch(ids);
    EXPECT_EQ(3u, loaded.size());
}

TEST_F(ActiveTierTest, RemovePatternsBatch) {
    std::vector<PatternID> ids;

    // Create and store patterns
    for (int i = 0; i < 5; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        tier_->StorePattern(pattern);
    }

    EXPECT_EQ(5u, tier_->GetPatternCount());

    // Remove batch
    size_t removed = tier_->RemovePatternsBatch(ids);
    EXPECT_EQ(5u, removed);
    EXPECT_EQ(0u, tier_->GetPatternCount());

    // Verify all patterns are gone
    for (const auto& id : ids) {
        EXPECT_FALSE(tier_->HasPattern(id));
    }
}

TEST_F(ActiveTierTest, StoreAssociationsBatch) {
    std::vector<AssociationEdge> edges;
    std::vector<std::pair<PatternID, PatternID>> pairs;

    // Create multiple associations
    for (int i = 0; i < 5; ++i) {
        PatternID source = PatternID::Generate();
        PatternID target = PatternID::Generate();
        edges.push_back(CreateTestAssociation(source, target));
        pairs.emplace_back(source, target);
    }

    // Store batch
    size_t stored = tier_->StoreAssociationsBatch(edges);
    EXPECT_EQ(5u, stored);
    EXPECT_EQ(5u, tier_->GetAssociationCount());

    // Verify all associations exist
    for (const auto& [source, target] : pairs) {
        EXPECT_TRUE(tier_->HasAssociation(source, target));
    }
}

TEST_F(WarmTierTest, StorePatternsBatch) {
    std::vector<PatternNode> patterns;

    // Create multiple patterns
    for (int i = 0; i < 5; ++i) {
        patterns.push_back(CreateTestPattern());
    }

    // Store batch
    size_t stored = tier_->StorePatternsBatch(patterns);
    EXPECT_EQ(5u, stored);
    EXPECT_EQ(5u, tier_->GetPatternCount());
}

TEST_F(WarmTierTest, LoadPatternsBatch) {
    std::vector<PatternID> ids;

    // Create and store patterns
    for (int i = 0; i < 5; ++i) {
        PatternNode pattern = CreateTestPattern();
        ids.push_back(pattern.GetID());
        tier_->StorePattern(pattern);
    }

    // Load batch
    auto loaded = tier_->LoadPatternsBatch(ids);
    EXPECT_EQ(5u, loaded.size());
}

// ============================================================================
// Statistics Tests (3+ tests)
// ============================================================================

TEST_F(ActiveTierTest, GetPatternCount) {
    EXPECT_EQ(0u, tier_->GetPatternCount());

    // Add patterns
    for (int i = 0; i < 10; ++i) {
        tier_->StorePattern(CreateTestPattern());
    }

    EXPECT_EQ(10u, tier_->GetPatternCount());

    // Remove some patterns
    PatternNode pattern = CreateTestPattern();
    PatternID id = pattern.GetID();
    tier_->StorePattern(pattern);
    tier_->RemovePattern(id);

    EXPECT_EQ(10u, tier_->GetPatternCount());
}

TEST_F(ActiveTierTest, GetAssociationCount) {
    EXPECT_EQ(0u, tier_->GetAssociationCount());

    // Add associations
    for (int i = 0; i < 10; ++i) {
        PatternID source = PatternID::Generate();
        PatternID target = PatternID::Generate();
        tier_->StoreAssociation(CreateTestAssociation(source, target));
    }

    EXPECT_EQ(10u, tier_->GetAssociationCount());
}

TEST_F(ActiveTierTest, EstimateMemoryUsage) {
    // Empty tier should have minimal usage
    size_t empty_usage = tier_->EstimateMemoryUsage();
    EXPECT_GE(empty_usage, 0u);

    // Add some patterns
    for (int i = 0; i < 10; ++i) {
        tier_->StorePattern(CreateTestPattern());
    }

    // Usage should increase
    size_t used = tier_->EstimateMemoryUsage();
    EXPECT_GT(used, empty_usage);
}

TEST_F(WarmTierTest, EstimateMemoryUsage) {
    // Empty tier should have minimal usage
    size_t empty_usage = tier_->EstimateMemoryUsage();
    EXPECT_GE(empty_usage, 0u);

    // Add some patterns
    for (int i = 0; i < 10; ++i) {
        tier_->StorePattern(CreateTestPattern());
    }

    // Usage should increase (file-based storage)
    size_t used = tier_->EstimateMemoryUsage();
    EXPECT_GT(used, empty_usage);
}

// ============================================================================
// Tier Information Tests (2+ tests)
// ============================================================================

TEST_F(ActiveTierTest, GetTierLevel) {
    EXPECT_EQ(MemoryTier::ACTIVE, tier_->GetTierLevel());
}

TEST_F(WarmTierTest, GetTierLevel) {
    EXPECT_EQ(MemoryTier::WARM, tier_->GetTierLevel());
}

TEST_F(ColdTierTest, GetTierLevel) {
    EXPECT_EQ(MemoryTier::COLD, tier_->GetTierLevel());
}

TEST_F(ArchiveTierTest, GetTierLevel) {
    EXPECT_EQ(MemoryTier::ARCHIVE, tier_->GetTierLevel());
}

TEST_F(ActiveTierTest, GetTierName) {
    EXPECT_EQ("Active", tier_->GetTierName());
}

TEST_F(WarmTierTest, GetTierName) {
    EXPECT_EQ("Warm", tier_->GetTierName());
}

TEST_F(ColdTierTest, GetTierName) {
    EXPECT_EQ("Cold", tier_->GetTierName());
}

TEST_F(ArchiveTierTest, GetTierName) {
    EXPECT_EQ("Archive", tier_->GetTierName());
}

// ============================================================================
// Maintenance Operation Tests (3+ tests)
// ============================================================================

TEST_F(ActiveTierTest, Clear) {
    // Add some data
    for (int i = 0; i < 5; ++i) {
        tier_->StorePattern(CreateTestPattern());
    }

    for (int i = 0; i < 3; ++i) {
        PatternID source = PatternID::Generate();
        PatternID target = PatternID::Generate();
        tier_->StoreAssociation(CreateTestAssociation(source, target));
    }

    EXPECT_EQ(5u, tier_->GetPatternCount());
    EXPECT_EQ(3u, tier_->GetAssociationCount());

    // Clear
    tier_->Clear();

    EXPECT_EQ(0u, tier_->GetPatternCount());
    EXPECT_EQ(0u, tier_->GetAssociationCount());
}

TEST_F(WarmTierTest, Clear) {
    // Add some data
    for (int i = 0; i < 5; ++i) {
        tier_->StorePattern(CreateTestPattern());
    }

    EXPECT_EQ(5u, tier_->GetPatternCount());

    // Clear
    tier_->Clear();

    EXPECT_EQ(0u, tier_->GetPatternCount());
    EXPECT_EQ(0u, tier_->GetAssociationCount());
}

TEST_F(ActiveTierTest, Flush) {
    // Add some data
    tier_->StorePattern(CreateTestPattern());

    // Flush should not throw or change state for in-memory tier
    EXPECT_NO_THROW(tier_->Flush());
    EXPECT_EQ(1u, tier_->GetPatternCount());
}

TEST_F(WarmTierTest, Flush) {
    // Add some data
    tier_->StorePattern(CreateTestPattern());

    // Flush should complete without error
    EXPECT_NO_THROW(tier_->Flush());
    EXPECT_EQ(1u, tier_->GetPatternCount());
}

TEST_F(ActiveTierTest, Compact) {
    // Add some data
    tier_->StorePattern(CreateTestPattern());

    // Compact should not throw or change state
    EXPECT_NO_THROW(tier_->Compact());
    EXPECT_EQ(1u, tier_->GetPatternCount());
}

TEST_F(WarmTierTest, Compact) {
    // Add some data
    tier_->StorePattern(CreateTestPattern());

    // Compact should complete without error
    EXPECT_NO_THROW(tier_->Compact());
    EXPECT_EQ(1u, tier_->GetPatternCount());
}

// ============================================================================
// Utility Function Tests (2+ tests)
// ============================================================================

TEST(MemoryTierUtilityTest, TierToString) {
    EXPECT_EQ("Active", TierToString(MemoryTier::ACTIVE));
    EXPECT_EQ("Warm", TierToString(MemoryTier::WARM));
    EXPECT_EQ("Cold", TierToString(MemoryTier::COLD));
    EXPECT_EQ("Archive", TierToString(MemoryTier::ARCHIVE));
}

TEST(MemoryTierUtilityTest, StringToTier) {
    // Case-sensitive matching
    EXPECT_EQ(MemoryTier::ACTIVE, StringToTier("Active"));
    EXPECT_EQ(MemoryTier::WARM, StringToTier("Warm"));
    EXPECT_EQ(MemoryTier::COLD, StringToTier("Cold"));
    EXPECT_EQ(MemoryTier::ARCHIVE, StringToTier("Archive"));

    // Case-insensitive matching (uppercase)
    EXPECT_EQ(MemoryTier::ACTIVE, StringToTier("ACTIVE"));
    EXPECT_EQ(MemoryTier::WARM, StringToTier("WARM"));
    EXPECT_EQ(MemoryTier::COLD, StringToTier("COLD"));
    EXPECT_EQ(MemoryTier::ARCHIVE, StringToTier("ARCHIVE"));

    // Invalid strings
    EXPECT_FALSE(StringToTier("invalid").has_value());
    EXPECT_FALSE(StringToTier("").has_value());
    EXPECT_FALSE(StringToTier("active").has_value());  // lowercase not supported
}

TEST(MemoryTierUtilityTest, TierToStringRoundtrip) {
    // Verify round-trip conversion
    auto tiers = {MemoryTier::ACTIVE, MemoryTier::WARM, MemoryTier::COLD, MemoryTier::ARCHIVE};

    for (auto tier : tiers) {
        std::string str = TierToString(tier);
        auto parsed = StringToTier(str);
        ASSERT_TRUE(parsed.has_value());
        EXPECT_EQ(tier, *parsed);
    }
}

// ============================================================================
// Persistence Tests (file-based tiers)
// ============================================================================

// NOTE: Persistence tests are disabled until PatternID::FromString() is implemented
// RebuildIndex() currently cannot restore indices from file names

// TEST_F(WarmTierTest, PersistenceAcrossInstances) {
//     PatternID id = PatternID::Generate();
//     PatternNode pattern = CreateTestPattern(id);
//
//     // Store pattern in first instance
//     EXPECT_TRUE(tier_->StorePattern(pattern));
//     tier_.reset();
//
//     // Create new instance with same storage path
//     tier_ = CreateWarmTier(temp_dir_.string());
//
//     // Pattern should still exist
//     EXPECT_TRUE(tier_->HasPattern(id));
//     auto loaded = tier_->LoadPattern(id);
//     ASSERT_TRUE(loaded.has_value());
//     EXPECT_EQ(id, loaded->GetID());
// }

// TEST_F(ColdTierTest, PersistenceAcrossInstances) {
//     PatternID source = PatternID::Generate();
//     PatternID target = PatternID::Generate();
//     AssociationEdge edge = CreateTestAssociation(source, target);
//
//     // Store association in first instance
//     EXPECT_TRUE(tier_->StoreAssociation(edge));
//     tier_.reset();
//
//     // Create new instance with same storage path
//     tier_ = CreateColdTier(temp_dir_.string());
//
//     // Association should still exist
//     EXPECT_TRUE(tier_->HasAssociation(source, target));
//     auto loaded = tier_->LoadAssociation(source, target);
//     ASSERT_TRUE(loaded.has_value());
//     EXPECT_EQ(source, loaded->GetSource());
//     EXPECT_EQ(target, loaded->GetTarget());
// }

// ============================================================================
// Edge Case Tests
// ============================================================================

TEST_F(ActiveTierTest, EmptyBatchOperations) {
    std::vector<PatternNode> empty_patterns;
    std::vector<PatternID> empty_ids;
    std::vector<AssociationEdge> empty_edges;

    // Empty batch operations should handle gracefully
    EXPECT_EQ(0u, tier_->StorePatternsBatch(empty_patterns));
    EXPECT_EQ(0u, tier_->LoadPatternsBatch(empty_ids).size());
    EXPECT_EQ(0u, tier_->RemovePatternsBatch(empty_ids));
    EXPECT_EQ(0u, tier_->StoreAssociationsBatch(empty_edges));
}

TEST_F(ActiveTierTest, DuplicatePatternsInBatch) {
    PatternID id = PatternID::Generate();

    std::vector<PatternNode> patterns;
    patterns.push_back(CreateTestPattern(id));
    patterns.push_back(CreateTestPattern(id));
    patterns.push_back(CreateTestPattern(id));

    // Store batch with duplicates
    size_t stored = tier_->StorePatternsBatch(patterns);
    EXPECT_EQ(3u, stored);  // All stores should succeed
    EXPECT_EQ(1u, tier_->GetPatternCount());  // But only one unique pattern
}

TEST_F(WarmTierTest, ClearEmptyTier) {
    // Clear on empty tier should not fail
    EXPECT_NO_THROW(tier_->Clear());
    EXPECT_EQ(0u, tier_->GetPatternCount());
    EXPECT_EQ(0u, tier_->GetAssociationCount());
}
