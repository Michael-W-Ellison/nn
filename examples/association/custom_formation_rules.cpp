// File: examples/association/custom_formation_rules.cpp
//
// Custom Association Formation Rules Example
//
// This example demonstrates advanced association formation techniques:
//   - Different association types (Causal, Categorical, Spatial, Functional)
//   - Custom formation rules and thresholds
//   - Type-specific formation strategies
//   - Spatial and categorical grouping
//
// The scenario simulates a knowledge graph where different types of
// relationships exist between patterns (e.g., cause-effect, category membership).

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "association/association_learning_system.hpp"
#include "association/formation_rules.hpp"
#include "core/types.hpp"

using namespace dpan;

// Helper to print association type
std::string AssociationTypeToString(AssociationType type) {
    switch (type) {
        case AssociationType::CAUSAL: return "CAUSAL";
        case AssociationType::CATEGORICAL: return "CATEGORICAL";
        case AssociationType::SPATIAL: return "SPATIAL";
        case AssociationType::FUNCTIONAL: return "FUNCTIONAL";
        case AssociationType::COMPOSITIONAL: return "COMPOSITIONAL";
        default: return "UNKNOWN";
    }
}

// Helper to print associations by type
void PrintAssociationsByType(const AssociationMatrix& matrix,
                             const std::map<PatternID, std::string>& pattern_names) {
    std::cout << "\n=== Associations by Type ===" << std::endl;

    std::map<AssociationType, size_t> type_counts;

    // Count associations by type
    for (const auto& [pattern_id, name] : pattern_names) {
        auto outgoing = matrix.GetOutgoingAssociations(pattern_id);
        for (const auto* edge : outgoing) {
            type_counts[edge->GetType()]++;
        }
    }

    // Print counts
    for (const auto& [type, count] : type_counts) {
        std::cout << "  " << AssociationTypeToString(type) << ": " << count << std::endl;
    }
    std::cout << "  TOTAL: " << matrix.GetAssociationCount() << std::endl;
}

int main() {
    std::cout << "=== DPAN: Custom Association Formation Rules ===" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 1: Create System with Custom Configuration
    // ========================================================================

    std::cout << "Step 1: Configuring system with custom formation rules..." << std::endl;

    AssociationLearningSystem::Config config;

    // Strict formation rules - require strong evidence
    config.formation.min_co_occurrences = 3;
    config.formation.min_chi_squared = 5.0f;  // Require strong statistical significance
    config.formation.initial_strength = 0.6f;

    // Lower learning rate for more stable associations
    config.reinforcement.learning_rate = 0.05f;

    // Aggressive competition for sparse associations
    config.competition.competition_factor = 0.3f;

    AssociationLearningSystem system(config);
    std::cout << "✓ System configured" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 2: Create Knowledge Graph Patterns
    // ========================================================================

    std::cout << "Step 2: Creating knowledge graph patterns..." << std::endl;

    // Create a knowledge graph:
    // Categories: Animals, Actions, Locations
    // Relationships: is-a (categorical), causes (causal), happens-at (spatial)

    std::map<PatternID, std::string> pattern_names;

    // Animals
    PatternID dog = PatternID::Generate();
    PatternID cat = PatternID::Generate();
    PatternID bird = PatternID::Generate();
    pattern_names[dog] = "dog";
    pattern_names[cat] = "cat";
    pattern_names[bird] = "bird";

    // Actions
    PatternID bark = PatternID::Generate();
    PatternID meow = PatternID::Generate();
    PatternID fly = PatternID::Generate();
    pattern_names[bark] = "bark";
    pattern_names[meow] = "meow";
    pattern_names[fly] = "fly";

    // Locations
    PatternID home = PatternID::Generate();
    PatternID park = PatternID::Generate();
    PatternID sky = PatternID::Generate();
    pattern_names[home] = "home";
    pattern_names[park] = "park";
    pattern_names[sky] = "sky";

    std::cout << "✓ Created " << pattern_names.size() << " patterns" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 3: Create Different Types of Associations
    // ========================================================================

    std::cout << "Step 3: Creating typed associations..." << std::endl;
    std::cout << std::endl;

    auto& matrix = const_cast<AssociationMatrix&>(system.GetAssociationMatrix());

    // CAUSAL associations (X causes Y)
    std::cout << "Creating CAUSAL associations (cause-effect):" << std::endl;
    AssociationEdge dog_bark(dog, bark, AssociationType::CAUSAL, 0.9f);
    AssociationEdge cat_meow(cat, meow, AssociationType::CAUSAL, 0.9f);
    AssociationEdge bird_fly(bird, fly, AssociationType::CAUSAL, 0.8f);

    matrix.AddAssociation(dog_bark);
    matrix.AddAssociation(cat_meow);
    matrix.AddAssociation(bird_fly);
    std::cout << "  dog → bark (0.9)" << std::endl;
    std::cout << "  cat → meow (0.9)" << std::endl;
    std::cout << "  bird → fly (0.8)" << std::endl;
    std::cout << std::endl;

    // CATEGORICAL associations (X is similar to Y)
    std::cout << "Creating CATEGORICAL associations (same category):" << std::endl;
    AssociationEdge dog_cat(dog, cat, AssociationType::CATEGORICAL, 0.7f);
    AssociationEdge cat_bird(cat, bird, AssociationType::CATEGORICAL, 0.6f);
    AssociationEdge bark_meow(bark, meow, AssociationType::CATEGORICAL, 0.7f);

    matrix.AddAssociation(dog_cat);
    matrix.AddAssociation(cat_bird);
    matrix.AddAssociation(bark_meow);
    std::cout << "  dog ↔ cat (0.7)" << std::endl;
    std::cout << "  cat ↔ bird (0.6)" << std::endl;
    std::cout << "  bark ↔ meow (0.7)" << std::endl;
    std::cout << std::endl;

    // SPATIAL associations (X happens at Y)
    std::cout << "Creating SPATIAL associations (location-based):" << std::endl;
    AssociationEdge dog_home(dog, home, AssociationType::SPATIAL, 0.8f);
    AssociationEdge dog_park(dog, park, AssociationType::SPATIAL, 0.6f);
    AssociationEdge bird_sky(bird, sky, AssociationType::SPATIAL, 0.9f);
    AssociationEdge cat_home(cat, home, AssociationType::SPATIAL, 0.7f);

    matrix.AddAssociation(dog_home);
    matrix.AddAssociation(dog_park);
    matrix.AddAssociation(bird_sky);
    matrix.AddAssociation(cat_home);
    std::cout << "  dog @ home (0.8)" << std::endl;
    std::cout << "  dog @ park (0.6)" << std::endl;
    std::cout << "  bird @ sky (0.9)" << std::endl;
    std::cout << "  cat @ home (0.7)" << std::endl;
    std::cout << std::endl;

    // FUNCTIONAL associations (X serves purpose similar to Y)
    std::cout << "Creating FUNCTIONAL associations (similar purpose):" << std::endl;
    AssociationEdge home_park(home, park, AssociationType::FUNCTIONAL, 0.5f);
    // Both are places where animals spend time

    matrix.AddAssociation(home_park);
    std::cout << "  home ≈ park (0.5)" << std::endl;
    std::cout << std::endl;

    PrintAssociationsByType(matrix, pattern_names);

    // ========================================================================
    // Step 4: Type-Specific Queries
    // ========================================================================

    std::cout << "\nStep 4: Querying associations by type..." << std::endl;
    std::cout << std::endl;

    // Query: What does a dog cause?
    std::cout << "Query: What does 'dog' cause?" << std::endl;
    auto dog_outgoing = matrix.GetOutgoingAssociations(dog);
    for (const auto* edge : dog_outgoing) {
        if (edge->GetType() == AssociationType::CAUSAL) {
            std::cout << "  → " << pattern_names[edge->GetTarget()]
                      << " (strength: " << edge->GetStrength() << ")" << std::endl;
        }
    }
    std::cout << std::endl;

    // Query: What is categorically similar to dog?
    std::cout << "Query: What is categorically similar to 'dog'?" << std::endl;
    for (const auto* edge : dog_outgoing) {
        if (edge->GetType() == AssociationType::CATEGORICAL) {
            std::cout << "  ↔ " << pattern_names[edge->GetTarget()]
                      << " (strength: " << edge->GetStrength() << ")" << std::endl;
        }
    }
    std::cout << std::endl;

    // Query: Where does dog occur spatially?
    std::cout << "Query: Where does 'dog' occur spatially?" << std::endl;
    for (const auto* edge : dog_outgoing) {
        if (edge->GetType() == AssociationType::SPATIAL) {
            std::cout << "  @ " << pattern_names[edge->GetTarget()]
                      << " (strength: " << edge->GetStrength() << ")" << std::endl;
        }
    }
    std::cout << std::endl;

    // ========================================================================
    // Step 5: Competitive Learning by Type
    // ========================================================================

    std::cout << "Step 5: Applying competitive learning within types..." << std::endl;
    std::cout << std::endl;

    // Before competition
    std::cout << "Dog's spatial associations BEFORE competition:" << std::endl;
    for (const auto* edge : dog_outgoing) {
        if (edge->GetType() == AssociationType::SPATIAL) {
            std::cout << "  @ " << pattern_names[edge->GetTarget()]
                      << ": " << edge->GetStrength() << std::endl;
        }
    }

    // Apply competition only to spatial associations
    CompetitiveLearner::Config comp_config;
    comp_config.competition_factor = 0.3f;
    CompetitiveLearner::ApplyTypedCompetition(matrix, dog, AssociationType::SPATIAL, comp_config);

    // After competition
    dog_outgoing = matrix.GetOutgoingAssociations(dog); // Refresh
    std::cout << "\nDog's spatial associations AFTER competition:" << std::endl;
    for (const auto* edge : dog_outgoing) {
        if (edge->GetType() == AssociationType::SPATIAL) {
            std::cout << "  @ " << pattern_names[edge->GetTarget()]
                      << ": " << edge->GetStrength() << std::endl;
        }
    }
    std::cout << "✓ Competition strengthened 'home', weakened 'park'" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 6: Multi-Hop Queries
    // ========================================================================

    std::cout << "Step 6: Multi-hop inference using associations..." << std::endl;
    std::cout << std::endl;

    std::cout << "Inference chain: dog → bark → ?" << std::endl;

    // First hop: dog → bark
    std::cout << "  Hop 1: dog → bark" << std::endl;

    // Second hop: What is bark similar to?
    auto bark_outgoing = matrix.GetOutgoingAssociations(bark);
    std::cout << "  Hop 2: bark is categorically similar to:" << std::endl;
    for (const auto* edge : bark_outgoing) {
        if (edge->GetType() == AssociationType::CATEGORICAL) {
            std::cout << "    → " << pattern_names[edge->GetTarget()] << std::endl;

            // Third hop: meow → ?
            auto third_incoming = matrix.GetIncomingAssociations(edge->GetTarget());
            std::cout << "      Hop 3: What causes " << pattern_names[edge->GetTarget()] << "?" << std::endl;
            for (const auto* third_edge : third_incoming) {
                if (third_edge->GetType() == AssociationType::CAUSAL) {
                    std::cout << "        ← " << pattern_names[third_edge->GetSource()] << std::endl;
                }
            }
        }
    }
    std::cout << "\n✓ Inference: dog → bark (similar to) → meow ← cat" << std::endl;
    std::cout << "   Conclusion: dog and cat both produce vocalizations" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 7: Custom Formation Strategy
    // ========================================================================

    std::cout << "Step 7: Demonstrating custom formation thresholds..." << std::endl;
    std::cout << std::endl;

    // Simulate strict vs. lenient formation rules
    AssociationFormationRules::Config strict_config;
    strict_config.min_co_occurrences = 5;
    strict_config.min_chi_squared = 7.0f;  // Very high statistical significance
    strict_config.initial_strength = 0.8f;

    AssociationFormationRules::Config lenient_config;
    lenient_config.min_co_occurrences = 2;
    lenient_config.min_chi_squared = 1.0f;  // Low statistical threshold
    lenient_config.initial_strength = 0.4f;

    std::cout << "Strict formation rules:" << std::endl;
    std::cout << "  Min co-occurrences: " << strict_config.min_co_occurrences << std::endl;
    std::cout << "  Min chi-squared: " << strict_config.min_chi_squared << std::endl;
    std::cout << "  Initial strength: " << strict_config.initial_strength << std::endl;
    std::cout << "  → Produces fewer, higher-quality associations" << std::endl;
    std::cout << std::endl;

    std::cout << "Lenient formation rules:" << std::endl;
    std::cout << "  Min co-occurrences: " << lenient_config.min_co_occurrences << std::endl;
    std::cout << "  Min chi-squared: " << lenient_config.min_chi_squared << std::endl;
    std::cout << "  Initial strength: " << lenient_config.initial_strength << std::endl;
    std::cout << "  → Produces more associations, some weak" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Summary
    // ========================================================================

    std::cout << "=== Final Statistics ===" << std::endl;
    auto stats = system.GetStatistics();
    std::cout << "Total associations: " << stats.total_associations << std::endl;
    PrintAssociationsByType(matrix, pattern_names);

    std::cout << "\n=== Example Complete ===" << std::endl;
    std::cout << "\nKey Takeaways:" << std::endl;
    std::cout << "1. Different association types model different relationships" << std::endl;
    std::cout << "2. Type-specific queries enable precise knowledge retrieval" << std::endl;
    std::cout << "3. Competitive learning can be applied per type" << std::endl;
    std::cout << "4. Multi-hop inference enables complex reasoning" << std::endl;
    std::cout << "5. Formation rules control association quality vs. quantity" << std::endl;

    return 0;
}
