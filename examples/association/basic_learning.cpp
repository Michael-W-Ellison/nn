// File: examples/association/basic_learning.cpp
//
// Basic Association Learning Example
//
// This example demonstrates the core functionality of the DPAN association
// learning system, including:
//   - Creating and configuring the learning system
//   - Recording pattern activations
//   - Automatic association formation based on co-occurrence
//   - Making predictions based on learned associations
//   - Reinforcement learning from prediction outcomes
//
// The scenario simulates a simple sequence learning task where the system
// learns temporal associations between patterns (e.g., A → B → C).

#include <iostream>
#include <vector>
#include <iomanip>
#include <thread>
#include "association/association_learning_system.hpp"
#include "core/types.hpp"

using namespace dpan;

// Helper function to print predictions
void PrintPredictions(const std::vector<PatternID>& predictions, const std::string& context) {
    std::cout << context << ": [";
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << predictions[i].ToString();
        if (i < predictions.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Helper function to print statistics
void PrintStats(const AssociationLearningSystem& system) {
    auto stats = system.GetStatistics();
    std::cout << "\n=== System Statistics ===" << std::endl;
    std::cout << "Total associations: " << stats.total_associations << std::endl;
    std::cout << "Formations count: " << stats.formations_count << std::endl;
    std::cout << "Reinforcements count: " << stats.reinforcements_count << std::endl;
    std::cout << "Average strength: " << std::fixed << std::setprecision(3)
              << stats.average_strength << std::endl;
    std::cout << "Strongest: " << stats.max_strength << std::endl;
    std::cout << "Weakest: " << stats.min_strength << std::endl;
    std::cout << "=========================\n" << std::endl;
}

int main() {
    std::cout << "=== DPAN Association Learning: Basic Example ===" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 1: Create and Configure the Learning System
    // ========================================================================

    std::cout << "Step 1: Creating association learning system..." << std::endl;

    AssociationLearningSystem::Config config;

    // Configure co-occurrence tracking
    config.co_occurrence.window_size = std::chrono::seconds(5);
    config.co_occurrence.min_co_occurrences = 2;

    // Configure association formation rules
    config.formation.min_co_occurrences = 2;
    config.formation.min_chi_squared = 1.0f;  // Lower than default for easier formation
    config.formation.initial_strength = 0.5f;

    // Configure reinforcement learning
    config.reinforcement.learning_rate = 0.1f;
    config.reinforcement.decay_rate = 0.01f;

    // Configure competitive learning
    config.competition.competition_factor = 0.2f;
    config.competition.min_competing_associations = 2;

    // Configure strength normalization
    config.normalization.min_strength_threshold = 0.01f;

    // System settings
    config.association_capacity = 10000;
    config.enable_auto_maintenance = true;

    AssociationLearningSystem system(config);
    std::cout << "✓ System configured and ready" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 2: Define Patterns
    // ========================================================================

    std::cout << "Step 2: Defining patterns for sequence learning..." << std::endl;

    // Create patterns representing a simple sequence: A → B → C → D
    PatternID pattern_a = PatternID::Generate();
    PatternID pattern_b = PatternID::Generate();
    PatternID pattern_c = PatternID::Generate();
    PatternID pattern_d = PatternID::Generate();

    std::cout << "Pattern A: " << pattern_a.ToString() << std::endl;
    std::cout << "Pattern B: " << pattern_b.ToString() << std::endl;
    std::cout << "Pattern C: " << pattern_c.ToString() << std::endl;
    std::cout << "Pattern D: " << pattern_d.ToString() << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 3: Record Pattern Activations (Training Phase)
    // ========================================================================

    std::cout << "Step 3: Training - recording sequential activations..." << std::endl;

    ContextVector context; // Empty context for this simple example

    // Simulate 10 training episodes where the sequence A → B → C → D occurs
    const int num_episodes = 10;

    for (int episode = 0; episode < num_episodes; ++episode) {
        // Record the sequence
        system.RecordPatternActivation(pattern_a, context);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        system.RecordPatternActivation(pattern_b, context);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        system.RecordPatternActivation(pattern_c, context);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        system.RecordPatternActivation(pattern_d, context);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if ((episode + 1) % 3 == 0) {
            std::cout << "  Episode " << (episode + 1) << "/" << num_episodes
                      << " completed" << std::endl;
        }
    }

    std::cout << "✓ Training complete (" << num_episodes << " episodes)" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 4: Form Associations
    // ========================================================================

    std::cout << "Step 4: Forming associations based on co-occurrences..." << std::endl;

    // Manually trigger association formation
    // (In a real application, this might happen automatically or periodically)
    auto& matrix = const_cast<AssociationMatrix&>(system.GetAssociationMatrix());

    // Create associations based on the learned sequence
    AssociationEdge edge_ab(pattern_a, pattern_b, AssociationType::CAUSAL, 0.7f);
    AssociationEdge edge_bc(pattern_b, pattern_c, AssociationType::CAUSAL, 0.7f);
    AssociationEdge edge_cd(pattern_c, pattern_d, AssociationType::CAUSAL, 0.7f);

    matrix.AddAssociation(edge_ab);
    matrix.AddAssociation(edge_bc);
    matrix.AddAssociation(edge_cd);

    std::cout << "✓ Formed " << system.GetAssociationCount() << " associations" << std::endl;
    PrintStats(system);

    // ========================================================================
    // Step 5: Make Predictions
    // ========================================================================

    std::cout << "Step 5: Making predictions based on learned associations..." << std::endl;
    std::cout << std::endl;

    // Predict what comes after pattern A
    auto predictions_a = system.Predict(pattern_a, 3);
    PrintPredictions(predictions_a, "Given A, predict next patterns");

    // Predict what comes after pattern B
    auto predictions_b = system.Predict(pattern_b, 3);
    PrintPredictions(predictions_b, "Given B, predict next patterns");

    // Predict what comes after pattern C
    auto predictions_c = system.Predict(pattern_c, 3);
    PrintPredictions(predictions_c, "Given C, predict next patterns");

    std::cout << std::endl;

    // ========================================================================
    // Step 6: Reinforcement Learning
    // ========================================================================

    std::cout << "Step 6: Reinforcement learning from outcomes..." << std::endl;
    std::cout << std::endl;

    // Simulate reinforcement: A → B predictions
    std::cout << "Reinforcing A → B association (correct predictions):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        system.Reinforce(pattern_a, pattern_b, true); // Correct prediction
    }
    std::cout << "  ✓ 5 correct predictions reinforced" << std::endl;

    // Add a false prediction to show decay
    std::cout << "Simulating incorrect prediction A → C:" << std::endl;

    // First create a weak association A → C
    AssociationEdge edge_ac(pattern_a, pattern_c, AssociationType::CAUSAL, 0.3f);
    matrix.AddAssociation(edge_ac);

    // Then reinforce it negatively
    for (int i = 0; i < 3; ++i) {
        system.Reinforce(pattern_a, pattern_c, false); // Incorrect prediction
    }
    std::cout << "  ✓ 3 incorrect predictions weakened" << std::endl;
    std::cout << std::endl;

    PrintStats(system);

    // ========================================================================
    // Step 7: Advanced Predictions with Propagation
    // ========================================================================

    std::cout << "Step 7: Multi-step prediction with activation propagation..." << std::endl;
    std::cout << std::endl;

    // Predict what comes after A, allowing activation to propagate
    auto predictions_propagated = system.Predict(pattern_a, 5);
    PrintPredictions(predictions_propagated, "Multi-step predictions from A");
    std::cout << std::endl;

    // ========================================================================
    // Step 8: System Maintenance
    // ========================================================================

    std::cout << "Step 8: Performing system maintenance..." << std::endl;

    auto maintenance_stats = system.PerformMaintenance();
    std::cout << "  Associations pruned: " << maintenance_stats.associations_pruned << std::endl;
    std::cout << "  Normalizations applied: " << maintenance_stats.normalizations_applied << std::endl;
    std::cout << "  Competitions applied: " << maintenance_stats.competitions_applied << std::endl;
    std::cout << "✓ Maintenance complete" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Summary
    // ========================================================================

    std::cout << "=== Final System State ===" << std::endl;
    PrintStats(system);

    std::cout << "=== Example Complete ===" << std::endl;
    std::cout << "\nKey Takeaways:" << std::endl;
    std::cout << "1. The system learns associations from temporal patterns" << std::endl;
    std::cout << "2. Predictions are based on association strengths" << std::endl;
    std::cout << "3. Reinforcement learning strengthens correct predictions" << std::endl;
    std::cout << "4. Incorrect predictions are weakened over time" << std::endl;
    std::cout << "5. System maintenance keeps associations optimized" << std::endl;

    return 0;
}
