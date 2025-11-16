// File: tests/discovery/pattern_creator_test.cpp
#include "discovery/pattern_creator.hpp"
#include "storage/memory_backend.hpp"
#include <gtest/gtest.h>

namespace dpan {
namespace {

// Helper to create test database
std::shared_ptr<PatternDatabase> CreateTestDatabase() {
    return std::make_shared<MemoryBackend>(MemoryBackend::Config{});
}

// ============================================================================
// PatternCreator Tests
// ============================================================================

TEST(PatternCreatorTest, ConstructorRequiresNonNullDatabase) {
    EXPECT_THROW(PatternCreator(nullptr), std::invalid_argument);
}

TEST(PatternCreatorTest, CreatePatternWorks) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    FeatureVector features({1.0f, 2.0f, 3.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    PatternID id = creator.CreatePattern(data);

    EXPECT_GT(id.value(), 0u);
    EXPECT_TRUE(db->Exists(id));
}

TEST(PatternCreatorTest, CreatePatternSetsCorrectType) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    FeatureVector features({1.0f, 2.0f, 3.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    PatternID id = creator.CreatePattern(data, PatternType::COMPOSITE);

    auto node_opt = db->Retrieve(id);
    ASSERT_TRUE(node_opt.has_value());
    EXPECT_EQ(PatternType::COMPOSITE, node_opt->GetType());
}

TEST(PatternCreatorTest, CreatePatternSetsInitialConfidence) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    FeatureVector features({1.0f, 2.0f, 3.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    PatternID id = creator.CreatePattern(data, PatternType::ATOMIC, 0.75f);

    auto node_opt = db->Retrieve(id);
    ASSERT_TRUE(node_opt.has_value());
    EXPECT_FLOAT_EQ(0.75f, node_opt->GetConfidenceScore());
}

TEST(PatternCreatorTest, CreatePatternRejectsInvalidConfidence) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    FeatureVector features({1.0f, 2.0f, 3.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    EXPECT_THROW(
        creator.CreatePattern(data, PatternType::ATOMIC, 1.5f),
        std::invalid_argument
    );
}

TEST(PatternCreatorTest, CreatePatternAssignsUniqueIDs) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    FeatureVector features({1.0f, 2.0f, 3.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    PatternID id1 = creator.CreatePattern(data);
    PatternID id2 = creator.CreatePattern(data);
    PatternID id3 = creator.CreatePattern(data);

    EXPECT_NE(id1.value(), id2.value());
    EXPECT_NE(id2.value(), id3.value());
    EXPECT_NE(id1.value(), id3.value());
}

TEST(PatternCreatorTest, CreateCompositePatternWorks) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    // Create sub-patterns first
    FeatureVector f1({1.0f, 2.0f});
    FeatureVector f2({3.0f, 4.0f});
    PatternData d1 = PatternData::FromFeatures(f1, DataModality::NUMERIC);
    PatternData d2 = PatternData::FromFeatures(f2, DataModality::NUMERIC);

    PatternID sub1 = creator.CreatePattern(d1);
    PatternID sub2 = creator.CreatePattern(d2);

    // Create composite pattern
    FeatureVector composite_features({2.0f, 3.0f});
    PatternData composite_data = PatternData::FromFeatures(composite_features, DataModality::NUMERIC);

    PatternID composite_id = creator.CreateCompositePattern({sub1, sub2}, composite_data);

    EXPECT_TRUE(db->Exists(composite_id));

    auto composite_node_opt = db->Retrieve(composite_id);
    ASSERT_TRUE(composite_node_opt.has_value());
    EXPECT_EQ(PatternType::COMPOSITE, composite_node_opt->GetType());

    auto sub_patterns = composite_node_opt->GetSubPatterns();
    EXPECT_EQ(2u, sub_patterns.size());

    // Verify sub-patterns are included
    bool has_sub1 = false, has_sub2 = false;
    for (const auto& sub_id : sub_patterns) {
        if (sub_id.value() == sub1.value()) has_sub1 = true;
        if (sub_id.value() == sub2.value()) has_sub2 = true;
    }
    EXPECT_TRUE(has_sub1);
    EXPECT_TRUE(has_sub2);
}

TEST(PatternCreatorTest, CreateCompositePatternRequiresSubPatterns) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    FeatureVector features({1.0f, 2.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    EXPECT_THROW(
        creator.CreateCompositePattern({}, data),
        std::invalid_argument
    );
}

TEST(PatternCreatorTest, CreateCompositePatternRequiresExistingSubPatterns) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    FeatureVector features({1.0f, 2.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    // Non-existent sub-pattern ID
    PatternID non_existent(9999);

    EXPECT_THROW(
        creator.CreateCompositePattern({non_existent}, data),
        std::invalid_argument
    );
}

TEST(PatternCreatorTest, CreateMetaPatternWorks) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    // Create pattern instances
    FeatureVector f1({1.0f, 2.0f});
    FeatureVector f2({3.0f, 4.0f});
    PatternData d1 = PatternData::FromFeatures(f1, DataModality::NUMERIC);
    PatternData d2 = PatternData::FromFeatures(f2, DataModality::NUMERIC);

    PatternID inst1 = creator.CreatePattern(d1);
    PatternID inst2 = creator.CreatePattern(d2);

    // Create meta-pattern
    FeatureVector meta_features({2.5f, 3.5f});
    PatternData meta_data = PatternData::FromFeatures(meta_features, DataModality::NUMERIC);

    PatternID meta_id = creator.CreateMetaPattern({inst1, inst2}, meta_data);

    EXPECT_TRUE(db->Exists(meta_id));

    auto meta_node_opt = db->Retrieve(meta_id);
    ASSERT_TRUE(meta_node_opt.has_value());
    EXPECT_EQ(PatternType::META, meta_node_opt->GetType());

    auto sub_patterns = meta_node_opt->GetSubPatterns();
    EXPECT_EQ(2u, sub_patterns.size());
}

TEST(PatternCreatorTest, CreateMetaPatternRequiresInstances) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    FeatureVector features({1.0f, 2.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    EXPECT_THROW(
        creator.CreateMetaPattern({}, data),
        std::invalid_argument
    );
}

TEST(PatternCreatorTest, CreateMetaPatternRequiresExistingInstances) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    FeatureVector features({1.0f, 2.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    PatternID non_existent(9999);

    EXPECT_THROW(
        creator.CreateMetaPattern({non_existent}, data),
        std::invalid_argument
    );
}

TEST(PatternCreatorTest, SetInitialActivationThresholdWorks) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    creator.SetInitialActivationThreshold(0.8f);
    EXPECT_FLOAT_EQ(0.8f, creator.GetInitialActivationThreshold());

    // Create a pattern and verify it uses the new threshold
    FeatureVector features({1.0f, 2.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    PatternID id = creator.CreatePattern(data);

    auto node_opt = db->Retrieve(id);
    ASSERT_TRUE(node_opt.has_value());
    EXPECT_FLOAT_EQ(0.8f, node_opt->GetActivationThreshold());
}

TEST(PatternCreatorTest, SetInitialActivationThresholdRejectsInvalidValue) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    EXPECT_THROW(
        creator.SetInitialActivationThreshold(1.5f),
        std::invalid_argument
    );
}

TEST(PatternCreatorTest, SetInitialConfidenceWorks) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    creator.SetInitialConfidence(0.9f);
    EXPECT_FLOAT_EQ(0.9f, creator.GetInitialConfidence());

    // Create a pattern and verify it uses the new confidence (when not explicitly specified)
    FeatureVector features({1.0f, 2.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);

    // CreatePattern with explicit confidence should override default
    PatternID id1 = creator.CreatePattern(data, PatternType::ATOMIC, 0.6f);
    auto node1_opt = db->Retrieve(id1);
    ASSERT_TRUE(node1_opt.has_value());
    EXPECT_FLOAT_EQ(0.6f, node1_opt->GetConfidenceScore());

    // CreateCompositePattern should use the default
    PatternID id2 = creator.CreatePattern(data);
    PatternID composite_id = creator.CreateCompositePattern({id2}, data);
    auto composite_opt = db->Retrieve(composite_id);
    ASSERT_TRUE(composite_opt.has_value());
    EXPECT_FLOAT_EQ(0.9f, composite_opt->GetConfidenceScore());
}

TEST(PatternCreatorTest, SetInitialConfidenceRejectsInvalidValue) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    EXPECT_THROW(
        creator.SetInitialConfidence(-0.5f),
        std::invalid_argument
    );
}

TEST(PatternCreatorTest, GetInitialActivationThresholdReturnsDefault) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    EXPECT_FLOAT_EQ(0.5f, creator.GetInitialActivationThreshold());
}

TEST(PatternCreatorTest, GetInitialConfidenceReturnsDefault) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    EXPECT_FLOAT_EQ(0.5f, creator.GetInitialConfidence());
}

TEST(PatternCreatorTest, PatternInitializationSetsBaseActivation) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    FeatureVector features({1.0f, 2.0f});
    PatternData data = PatternData::FromFeatures(features, DataModality::NUMERIC);
    PatternID id = creator.CreatePattern(data);

    auto node_opt = db->Retrieve(id);
    ASSERT_TRUE(node_opt.has_value());
    EXPECT_FLOAT_EQ(0.0f, node_opt->GetBaseActivation());
}

TEST(PatternCreatorTest, MultipleCompositePatternCreation) {
    auto db = CreateTestDatabase();
    PatternCreator creator(db);

    // Create several atomic patterns
    std::vector<PatternID> atomic_ids;
    for (int i = 0; i < 5; ++i) {
        FeatureVector f(std::vector<float>{static_cast<float>(i)});
        PatternData d = PatternData::FromFeatures(f, DataModality::NUMERIC);
        atomic_ids.push_back(creator.CreatePattern(d));
    }

    // Create composite patterns using different subsets
    FeatureVector comp_features(std::vector<float>{10.0f});
    PatternData comp_data = PatternData::FromFeatures(comp_features, DataModality::NUMERIC);

    PatternID comp1 = creator.CreateCompositePattern(
        {atomic_ids[0], atomic_ids[1]},
        comp_data
    );
    PatternID comp2 = creator.CreateCompositePattern(
        {atomic_ids[2], atomic_ids[3], atomic_ids[4]},
        comp_data
    );

    EXPECT_TRUE(db->Exists(comp1));
    EXPECT_TRUE(db->Exists(comp2));
}

} // namespace
} // namespace dpan
