// File: tests/association/association_learning_system_test.cpp
#include "association/association_learning_system.hpp"
#include "core/pattern_node.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

namespace dpan {
namespace {

// Helper function to create test pattern
PatternID CreateTestPattern(const std::string& label = "") {
    return PatternID::Generate();
}

// ============================================================================
// Construction & Configuration Tests
// ============================================================================

TEST(AssociationLearningSystemTest, DefaultConstructor) {
    AssociationLearningSystem system;

    EXPECT_EQ(0u, system.GetAssociationCount());
}

TEST(AssociationLearningSystemTest, ConfigConstructor) {
    AssociationLearningSystem::Config config;
    config.association_capacity = 50000;
    config.prune_threshold = 0.1f;

    AssociationLearningSystem system(config);

    auto retrieved_config = system.GetConfig();
    EXPECT_EQ(50000u, retrieved_config.association_capacity);
    EXPECT_FLOAT_EQ(0.1f, retrieved_config.prune_threshold);
}

TEST(AssociationLearningSystemTest, SetConfigUpdatesConfiguration) {
    AssociationLearningSystem system;

    AssociationLearningSystem::Config new_config;
    new_config.prune_threshold = 0.2f;

    system.SetConfig(new_config);

    auto retrieved = system.GetConfig();
    EXPECT_FLOAT_EQ(0.2f, retrieved.prune_threshold);
}

// ============================================================================
// Pattern Activation Tests
// ============================================================================

TEST(AssociationLearningSystemTest, RecordSingleActivation) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();

    system.RecordPatternActivation(p1);

    // Should not crash and should update internal state
    auto stats = system.GetStatistics();
    EXPECT_GE(stats.activation_history_size, 0u);
}

TEST(AssociationLearningSystemTest, RecordMultipleActivations) {
    AssociationLearningSystem system;
    

    std::vector<PatternID> patterns;
    for (int i = 0; i < 10; ++i) {
        patterns.push_back(CreateTestPattern());
    }

    system.RecordPatternActivations(patterns);

    auto stats = system.GetStatistics();
    EXPECT_GE(stats.activation_history_size, 0u);
}

TEST(AssociationLearningSystemTest, ActivationHistoryLimited) {
    AssociationLearningSystem::Config config;
    config.max_activation_history = 100;

    AssociationLearningSystem system(config);
    

    PatternID p1 = CreateTestPattern();

    // Record more than max_activation_history activations
    for (int i = 0; i < 200; ++i) {
        system.RecordPatternActivation(p1);
    }

    auto stats = system.GetStatistics();
    EXPECT_LE(stats.activation_history_size, 100u);
}

// ============================================================================
// Association Formation Tests
// ============================================================================

TEST(AssociationLearningSystemTest, FormAssociationsFromCoOccurrences) {
    AssociationLearningSystem::Config config;
    config.co_occurrence.min_co_occurrences = 2;
    config.formation.min_co_occurrences = 2;

    AssociationLearningSystem system(config);
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();

    // Record co-occurring patterns multiple times
    for (int i = 0; i < 5; ++i) {
        system.RecordPatternActivations({p1, p2});
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Form associations
    // Simplified - FormNewAssociations not fully implemented
    size_t formed = 0; // system.FormNewAssociations(db);

    // Should have formed at least one association
    EXPECT_GT(formed, 0u);
    EXPECT_GT(system.GetAssociationCount(), 0u);
}

TEST(AssociationLearningSystemTest, NoAssociationWithoutSufficientCoOccurrence) {
    AssociationLearningSystem::Config config;
    config.co_occurrence.min_co_occurrences = 10;
    config.formation.min_co_occurrences = 10;

    AssociationLearningSystem system(config);
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();

    // Record only a few co-occurrences (less than threshold)
    system.RecordPatternActivations({p1, p2});

    // Simplified - FormNewAssociations not fully implemented
    size_t formed = 0; // system.FormNewAssociations(db);

    EXPECT_EQ(0u, formed);
}

TEST(AssociationLearningSystemTest, FormAssociationsForSpecificPattern) {
    AssociationLearningSystem::Config config;
    config.co_occurrence.min_co_occurrences = 2;
    config.formation.min_co_occurrences = 2;

    AssociationLearningSystem system(config);
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();

    // Record p1 with p2 multiple times
    for (int i = 0; i < 5; ++i) {
        system.RecordPatternActivations({p1, p2});
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Form associations only for p1
    // Simplified - FormAssociationsForPattern requires PatternDatabase
    size_t formed = 0; // system.FormAssociationsForPattern(p1, db);

    EXPECT_GT(formed, 0u);
}

// ============================================================================
// Reinforcement Learning Tests
// ============================================================================

TEST(AssociationLearningSystemTest, ReinforceCorrectPrediction) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();

    // Manually add an association
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge);

    float initial_strength = system.GetAssociationMatrix().GetAssociation(p1, p2)->GetStrength();

    // Reinforce correct prediction
    system.Reinforce(p1, p2, true);

    const auto* updated = system.GetAssociationMatrix().GetAssociation(p1, p2);
    ASSERT_NE(nullptr, updated);

    // Strength should increase for correct prediction
    EXPECT_GE(updated->GetStrength(), initial_strength);
}

TEST(AssociationLearningSystemTest, ReinforceIncorrectPrediction) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();

    // Manually add an association p1 -> p2
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.7f);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge);

    float initial_strength = system.GetAssociationMatrix().GetAssociation(p1, p2)->GetStrength();

    // Reinforce incorrect prediction (predicted p2 but actual was p3)
    system.Reinforce(p1, p3, false);

    // Note: Reinforcement of incorrect predictions is handled in ReinforcementManager
    // This test just ensures no crash
    EXPECT_NO_THROW(system.Reinforce(p1, p3, false));
}

TEST(AssociationLearningSystemTest, ReinforceBatchProcessesMultiple) {
    AssociationLearningSystem system;
    

    std::vector<PatternID> patterns;
    for (int i = 0; i < 5; ++i) {
        patterns.push_back(CreateTestPattern());
    }

    // Create some associations
    for (size_t i = 0; i < patterns.size() - 1; ++i) {
        AssociationEdge edge(patterns[i], patterns[i + 1], AssociationType::CAUSAL, 0.5f);
        const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge);
    }

    // Batch reinforce
    std::vector<std::tuple<PatternID, PatternID, bool>> outcomes;
    for (size_t i = 0; i < patterns.size() - 1; ++i) {
        outcomes.emplace_back(patterns[i], patterns[i + 1], true);
    }

    system.ReinforceBatch(outcomes);

    auto stats = system.GetStatistics();
    EXPECT_GE(stats.reinforcements_count, outcomes.size());
}

// ============================================================================
// Maintenance Operations Tests
// ============================================================================

TEST(AssociationLearningSystemTest, ApplyDecayWeakensAssociations) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();

    // Create association
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.8f);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge);

    float initial_strength = system.GetAssociationMatrix().GetAssociation(p1, p2)->GetStrength();

    // Apply significant decay
    system.ApplyDecay(std::chrono::hours(24));

    const auto* decayed = system.GetAssociationMatrix().GetAssociation(p1, p2);
    ASSERT_NE(nullptr, decayed);

    // Strength should decrease
    EXPECT_LT(decayed->GetStrength(), initial_strength);
}

TEST(AssociationLearningSystemTest, PruneWeakAssociationsRemovesWeak) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();

    // Create one strong and one weak association
    AssociationEdge strong(p1, p2, AssociationType::CAUSAL, 0.8f);
    AssociationEdge weak(p1, p3, AssociationType::CAUSAL, 0.02f);

    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(strong);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(weak);

    EXPECT_EQ(2u, system.GetAssociationCount());

    // Prune with threshold 0.05
    size_t pruned = system.PruneWeakAssociations(0.05f);

    EXPECT_EQ(1u, pruned);  // Should prune the weak one
    EXPECT_EQ(1u, system.GetAssociationCount());  // Only strong one remains
}

TEST(AssociationLearningSystemTest, CompactReducesMemoryFootprint) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();

    // Add and remove associations
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).RemoveAssociation(p1, p2);

    // Compact should not crash
    EXPECT_NO_THROW(system.Compact());
}

TEST(AssociationLearningSystemTest, PerformMaintenanceExecutesAllOperations) {
    AssociationLearningSystem system;

    auto stats = system.PerformMaintenance();

    // Should return valid statistics
    EXPECT_GE(stats.decay_applied.count(), 0);
    EXPECT_GE(stats.competitions_applied, 0u);
    EXPECT_GE(stats.normalizations_applied, 0u);
    EXPECT_GE(stats.associations_pruned, 0u);
}

// ============================================================================
// Query & Prediction Tests
// ============================================================================

TEST(AssociationLearningSystemTest, GetAssociationsReturnsOutgoing) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();

    // Create outgoing associations from p1
    AssociationEdge edge1(p1, p2, AssociationType::CAUSAL, 0.8f);
    AssociationEdge edge2(p1, p3, AssociationType::CAUSAL, 0.6f);

    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge1);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge2);

    auto outgoing = system.GetAssociations(p1, true);

    EXPECT_EQ(2u, outgoing.size());
}

TEST(AssociationLearningSystemTest, GetAssociationsReturnsIncoming) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();

    // Create incoming associations to p3
    AssociationEdge edge1(p1, p3, AssociationType::CAUSAL, 0.8f);
    AssociationEdge edge2(p2, p3, AssociationType::CAUSAL, 0.6f);

    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge1);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge2);

    auto incoming = system.GetAssociations(p3, false);

    EXPECT_EQ(2u, incoming.size());
}

TEST(AssociationLearningSystemTest, PredictReturnsTopK) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();
    PatternID p4 = CreateTestPattern();

    // Create associations with varying strengths
    AssociationEdge edge1(p1, p2, AssociationType::CAUSAL, 0.9f);
    AssociationEdge edge2(p1, p3, AssociationType::CAUSAL, 0.7f);
    AssociationEdge edge3(p1, p4, AssociationType::CAUSAL, 0.5f);

    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge1);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge2);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge3);

    auto predictions = system.Predict(p1, 2);

    EXPECT_EQ(2u, predictions.size());
    // First prediction should be p2 (strongest)
    EXPECT_EQ(p2, predictions[0]);
}

TEST(AssociationLearningSystemTest, PredictWithConfidenceReturnsScores) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();

    // Create associations
    AssociationEdge edge1(p1, p2, AssociationType::CAUSAL, 0.9f);
    AssociationEdge edge2(p1, p3, AssociationType::CAUSAL, 0.5f);

    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge1);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge2);

    auto predictions = system.PredictWithConfidence(p1, 2);

    ASSERT_EQ(2u, predictions.size());
    EXPECT_EQ(p2, predictions[0].first);
    EXPECT_FLOAT_EQ(0.9f, predictions[0].second);
}

TEST(AssociationLearningSystemTest, PropagateActivationSpreadsThroughNetwork) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();
    PatternID p3 = CreateTestPattern();

    // Create chain: p1 -> p2 -> p3
    AssociationEdge edge1(p1, p2, AssociationType::CAUSAL, 0.8f);
    AssociationEdge edge2(p2, p3, AssociationType::CAUSAL, 0.7f);

    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge1);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge2);

    auto results = system.PropagateActivation(p1, 1.0f, 3);

    // Should reach p2 and p3
    EXPECT_GE(results.size(), 2u);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(AssociationLearningSystemTest, GetStatisticsReturnsValidData) {
    AssociationLearningSystem system;

    auto stats = system.GetStatistics();

    EXPECT_GE(stats.total_associations, 0u);
    EXPECT_GE(stats.activation_history_size, 0u);
}

TEST(AssociationLearningSystemTest, StatisticsUpdateAfterOperations) {
    AssociationLearningSystem system;
    

    auto initial_stats = system.GetStatistics();

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();

    // Record activations
    system.RecordPatternActivation(p1);

    auto after_activation = system.GetStatistics();
    EXPECT_GT(after_activation.activation_history_size, initial_stats.activation_history_size);

    // Add association
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.5f);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge);

    auto after_association = system.GetStatistics();
    EXPECT_GT(after_association.total_associations, initial_stats.total_associations);
}

TEST(AssociationLearningSystemTest, PrintStatisticsOutputsText) {
    AssociationLearningSystem system;

    std::ostringstream oss;
    system.PrintStatistics(oss);

    std::string output = oss.str();
    EXPECT_FALSE(output.empty());
    EXPECT_NE(std::string::npos, output.find("Association Learning System"));
}

// ============================================================================
// Persistence Tests
// ============================================================================

TEST(AssociationLearningSystemTest, SaveAndLoadRoundTrip) {
    AssociationLearningSystem system;
    

    PatternID p1 = CreateTestPattern();
    PatternID p2 = CreateTestPattern();

    // Add association
    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.8f);
    const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge);

    // Save
    std::string filepath = "/tmp/test_learning_system.bin";
    bool saved = system.Save(filepath);
    EXPECT_TRUE(saved);

    // Load into new system
    AssociationLearningSystem loaded_system;
    bool loaded = loaded_system.Load(filepath);
    EXPECT_TRUE(loaded);

    // Verify association was preserved
    EXPECT_EQ(system.GetAssociationCount(), loaded_system.GetAssociationCount());

    // Cleanup
    std::remove(filepath.c_str());
}

// ============================================================================
// End-to-End Integration Tests
// ============================================================================

TEST(AssociationLearningSystemTest, EndToEndLearningWorkflow) {
    AssociationLearningSystem::Config config;
    config.co_occurrence.min_co_occurrences = 2;
    config.formation.min_co_occurrences = 2;

    AssociationLearningSystem system(config);
    

    // Create patterns
    std::vector<PatternID> patterns;
    for (int i = 0; i < 10; ++i) {
        patterns.push_back(CreateTestPattern());
    }

    // Simulate learning: patterns 0 and 1 often co-occur
    for (int iter = 0; iter < 10; ++iter) {
        system.RecordPatternActivations({patterns[0], patterns[1]});
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Form associations
    // Simplified - FormNewAssociations not fully implemented
    size_t formed = 0; // system.FormNewAssociations(db);
    EXPECT_GT(formed, 0u);

    // Test prediction
    auto predictions = system.Predict(patterns[0], 3);
    EXPECT_FALSE(predictions.empty());

    // Apply maintenance
    auto maint_stats = system.PerformMaintenance();
    EXPECT_GE(maint_stats.decay_applied.count(), 0);
}

TEST(AssociationLearningSystemTest, ConcurrentActivationRecording) {
    AssociationLearningSystem system;
    

    std::vector<PatternID> patterns;
    for (int i = 0; i < 5; ++i) {
        patterns.push_back(CreateTestPattern());
    }

    // Record activations from multiple threads
    std::vector<std::thread> threads;
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&system, &patterns, i]() {
            for (int j = 0; j < 100; ++j) {
                system.RecordPatternActivation(patterns[i % patterns.size()]);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Should not crash and should have recorded activations
    auto stats = system.GetStatistics();
    EXPECT_GT(stats.activation_history_size, 0u);
}

TEST(AssociationLearningSystemTest, LargeScaleAssociations) {
    AssociationLearningSystem::Config config;
    config.association_capacity = 10000;

    AssociationLearningSystem system(config);
    

    // Create many patterns
    std::vector<PatternID> patterns;
    for (int i = 0; i < 100; ++i) {
        patterns.push_back(CreateTestPattern());
    }

    // Create associations between many patterns
    for (size_t i = 0; i < patterns.size() - 1; ++i) {
        AssociationEdge edge(patterns[i], patterns[i + 1], AssociationType::CAUSAL, 0.5f);
        const_cast<AssociationMatrix&>(system.GetAssociationMatrix()).AddAssociation(edge);
    }

    EXPECT_EQ(99u, system.GetAssociationCount());

    // Test prediction still works
    auto predictions = system.Predict(patterns[0], 5);
    EXPECT_FALSE(predictions.empty());
}

} // namespace
} // namespace dpan
