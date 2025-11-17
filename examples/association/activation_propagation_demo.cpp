// File: examples/association/activation_propagation_demo.cpp
//
// Activation Propagation Demonstration
//
// This example demonstrates how activation spreads through an association
// network using breadth-first search with decay:
//   - Creating a network of interconnected patterns
//   - Triggering activation from a source pattern
//   - Observing activation spread with decay
//   - Analyzing propagation depth and activation levels
//   - Bidirectional propagation (forward and backward)
//
// The scenario simulates a semantic network where activating one concept
// spreads activation to related concepts, with decreasing intensity.

#include <iostream>
#include <vector>
#include <iomanip>
#include <map>
#include "association/association_matrix.hpp"
#include "core/types.hpp"

using namespace dpan;

// Helper function to visualize activation propagation
void PrintPropagationResults(const std::vector<AssociationMatrix::ActivationResult>& results,
                             const std::map<PatternID, std::string>& names,
                             const std::string& title) {
    std::cout << "\n=== " << title << " ===" << std::endl;

    // Sort by activation level (descending)
    std::vector<std::pair<float, PatternID>> sorted;
    for (const auto& result : results) {
        sorted.push_back({result.activation, result.pattern});
    }
    std::sort(sorted.begin(), sorted.end(), std::greater<>());

    // Print in bar chart format
    for (const auto& [activation, id] : sorted) {
        auto it = names.find(id);
        std::string name = (it != names.end()) ? it->second : id.ToString();

        std::cout << "  " << std::setw(15) << std::left << name << " ";
        std::cout << std::fixed << std::setprecision(2) << activation << " ";

        // Visual bar
        int bar_length = static_cast<int>(activation * 40);
        std::cout << "[";
        for (int i = 0; i < 40; ++i) {
            std::cout << (i < bar_length ? '█' : '░');
        }
        std::cout << "]" << std::endl;
    }
}

int main() {
    std::cout << "=== DPAN: Activation Propagation Demo ===" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 1: Create Semantic Network
    // ========================================================================

    std::cout << "Step 1: Building semantic network..." << std::endl;
    std::cout << std::endl;

    AssociationMatrix matrix;

    // Create patterns representing a semantic network about transportation
    std::map<PatternID, std::string> pattern_names;

    // Core concepts
    PatternID car = PatternID::Generate();
    PatternID wheel = PatternID::Generate();
    PatternID engine = PatternID::Generate();
    PatternID road = PatternID::Generate();
    PatternID driver = PatternID::Generate();

    // Related concepts
    PatternID bicycle = PatternID::Generate();
    PatternID motorcycle = PatternID::Generate();
    PatternID bus = PatternID::Generate();
    PatternID highway = PatternID::Generate();
    PatternID traffic = PatternID::Generate();

    // Distant concepts
    PatternID fuel = PatternID::Generate();
    PatternID tire = PatternID::Generate();

    pattern_names[car] = "car";
    pattern_names[wheel] = "wheel";
    pattern_names[engine] = "engine";
    pattern_names[road] = "road";
    pattern_names[driver] = "driver";
    pattern_names[bicycle] = "bicycle";
    pattern_names[motorcycle] = "motorcycle";
    pattern_names[bus] = "bus";
    pattern_names[highway] = "highway";
    pattern_names[traffic] = "traffic";
    pattern_names[fuel] = "fuel";
    pattern_names[tire] = "tire";

    std::cout << "Created " << pattern_names.size() << " concepts" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 2: Build Association Network
    // ========================================================================

    std::cout << "Step 2: Creating associations..." << std::endl;

    // Car's components (strong compositional associations)
    matrix.AddAssociation(AssociationEdge(car, wheel, AssociationType::COMPOSITIONAL, 0.9f));
    matrix.AddAssociation(AssociationEdge(car, engine, AssociationType::COMPOSITIONAL, 0.9f));
    std::cout << "  car → wheel (0.9), engine (0.9)" << std::endl;

    // Car's context (spatial/functional associations)
    matrix.AddAssociation(AssociationEdge(car, road, AssociationType::SPATIAL, 0.8f));
    matrix.AddAssociation(AssociationEdge(car, driver, AssociationType::FUNCTIONAL, 0.8f));
    std::cout << "  car → road (0.8), driver (0.8)" << std::endl;

    // Similar vehicles (categorical associations)
    matrix.AddAssociation(AssociationEdge(car, bus, AssociationType::CATEGORICAL, 0.7f));
    matrix.AddAssociation(AssociationEdge(car, motorcycle, AssociationType::CATEGORICAL, 0.6f));
    std::cout << "  car ↔ bus (0.7), motorcycle (0.6)" << std::endl;

    // Shared components
    matrix.AddAssociation(AssociationEdge(bicycle, wheel, AssociationType::COMPOSITIONAL, 0.9f));
    matrix.AddAssociation(AssociationEdge(motorcycle, wheel, AssociationType::COMPOSITIONAL, 0.9f));
    matrix.AddAssociation(AssociationEdge(motorcycle, engine, AssociationType::COMPOSITIONAL, 0.8f));
    std::cout << "  bicycle → wheel (0.9)" << std::endl;
    std::cout << "  motorcycle → wheel (0.9), engine (0.8)" << std::endl;

    // Road network
    matrix.AddAssociation(AssociationEdge(road, highway, AssociationType::CATEGORICAL, 0.8f));
    matrix.AddAssociation(AssociationEdge(road, traffic, AssociationType::CAUSAL, 0.7f));
    std::cout << "  road → highway (0.8), traffic (0.7)" << std::endl;

    // Engine components
    matrix.AddAssociation(AssociationEdge(engine, fuel, AssociationType::FUNCTIONAL, 0.9f));
    std::cout << "  engine → fuel (0.9)" << std::endl;

    // Wheel components
    matrix.AddAssociation(AssociationEdge(wheel, tire, AssociationType::COMPOSITIONAL, 0.9f));
    std::cout << "  wheel → tire (0.9)" << std::endl;

    std::cout << "\n✓ Created " << matrix.GetAssociationCount() << " associations" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 3: Single-Level Propagation
    // ========================================================================

    std::cout << "Step 3: Single-level activation propagation..." << std::endl;

    // Activate "car" with initial strength 1.0, propagate 1 level
    auto results_level1 = matrix.PropagateActivation(car, 1.0f, 1);

    PrintPropagationResults(results_level1, pattern_names,
                           "Activation from 'car' (depth=1)");

    std::cout << "\nObservation: Direct associations receive activation proportional to their strength" << std::endl;
    std::cout << "  wheel & engine (0.9) receive highest activation" << std::endl;
    std::cout << "  road & driver (0.8) receive strong activation" << std::endl;
    std::cout << "  bus (0.7) & motorcycle (0.6) receive moderate activation" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 4: Multi-Level Propagation with Decay
    // ========================================================================

    std::cout << "Step 4: Multi-level activation propagation (depth=3)..." << std::endl;

    // Activate "car", propagate 3 levels deep
    auto results_level3 = matrix.PropagateActivation(car, 1.0f, 3);

    PrintPropagationResults(results_level3, pattern_names,
                           "Activation from 'car' (depth=3)");

    std::cout << "\nObservation: Activation spreads and decays with distance" << std::endl;
    std::cout << "  Level 1: wheel, engine, road, driver (direct)" << std::endl;
    std::cout << "  Level 2: tire, fuel, highway (through intermediates)" << std::endl;
    std::cout << "  Level 3: traffic (3 hops away: car → road → highway → traffic)" << std::endl;
    std::cout << "\n✓ Distant concepts receive weaker activation" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 5: Bidirectional Propagation
    // ========================================================================

    std::cout << "Step 5: Bidirectional propagation..." << std::endl;
    std::cout << std::endl;

    // Forward: What does "car" activate?
    auto forward = matrix.PropagateActivation(car, 1.0f, 2);
    std::cout << "Forward propagation from 'car':" << std::endl;
    std::cout << "  Activates: components, context, similar vehicles" << std::endl;

    // Backward: What activates "wheel"?
    auto backward_wheel = matrix.GetIncomingAssociations(wheel);
    std::cout << "\nBackward: What patterns lead to 'wheel'?" << std::endl;
    for (const auto* edge : backward_wheel) {
        std::cout << "  ← " << pattern_names[edge->GetSource()]
                  << " (strength: " << edge->GetStrength() << ")" << std::endl;
    }
    std::cout << "  (car, bicycle, motorcycle all have wheels)" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 6: Activation Decay Analysis
    // ========================================================================

    std::cout << "Step 6: Analyzing activation decay..." << std::endl;
    std::cout << std::endl;

    std::cout << "Propagating from 'car' with varying depths:" << std::endl;
    std::cout << std::endl;

    for (int depth = 1; depth <= 4; ++depth) {
        auto results = matrix.PropagateActivation(car, 1.0f, depth);
        std::cout << "  Depth " << depth << ": " << results.size() << " patterns activated, ";

        // Find max activation at this depth
        float max_activation = 0.0f;
        for (const auto& result : results) {
            max_activation = std::max(max_activation, result.activation);
        }
        std::cout << "max activation = " << std::fixed << std::setprecision(3)
                  << max_activation << std::endl;
    }

    std::cout << "\nObservation: Activation spreads wider but weaker with depth" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 7: Comparative Propagation
    // ========================================================================

    std::cout << "Step 7: Comparing propagation from different sources..." << std::endl;
    std::cout << std::endl;

    // Propagate from car vs. bicycle
    auto from_car = matrix.PropagateActivation(car, 1.0f, 2);
    auto from_bicycle = matrix.PropagateActivation(bicycle, 1.0f, 2);

    std::cout << "Activation from 'car' reaches " << from_car.size() << " patterns" << std::endl;
    std::cout << "Activation from 'bicycle' reaches " << from_bicycle.size() << " patterns" << std::endl;
    std::cout << std::endl;

    // Find shared activations
    std::cout << "Shared activated patterns:" << std::endl;
    for (const auto& car_result : from_car) {
        for (const auto& bicycle_result : from_bicycle) {
            if (car_result.pattern == bicycle_result.pattern) {
                std::cout << "  " << pattern_names[car_result.pattern] << " (activated by both)" << std::endl;
                break;
            }
        }
    }
    std::cout << std::endl;

    // ========================================================================
    // Step 8: Practical Application - Semantic Priming
    // ========================================================================

    std::cout << "Step 8: Semantic priming simulation..." << std::endl;
    std::cout << std::endl;

    std::cout << "Scenario: User thinks about 'car', what concepts are primed?" << std::endl;
    auto primed = matrix.PropagateActivation(car, 1.0f, 2);

    // Find highly primed concepts (activation > 0.5)
    std::cout << "\nHighly primed concepts (activation > 0.5):" << std::endl;
    std::vector<std::pair<float, std::string>> highly_primed;
    for (const auto& result : primed) {
        if (result.activation > 0.5f && result.pattern != car) {
            highly_primed.push_back({result.activation, pattern_names[result.pattern]});
        }
    }
    std::sort(highly_primed.rbegin(), highly_primed.rend());

    for (const auto& [activation, name] : highly_primed) {
        std::cout << "  " << name << " (" << std::fixed << std::setprecision(2)
                  << activation << ")" << std::endl;
    }

    std::cout << "\nApplication: These concepts are more likely to be retrieved from memory" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Step 9: Activation with Custom Initial Strength
    // ========================================================================

    std::cout << "Step 9: Varying initial activation strength..." << std::endl;
    std::cout << std::endl;

    float strengths[] = {0.3f, 0.6f, 1.0f};
    for (float strength : strengths) {
        auto results = matrix.PropagateActivation(car, strength, 1);

        float total_activation = 0.0f;
        for (const auto& result : results) {
            total_activation += result.activation;
        }

        std::cout << "  Initial strength " << strength << " → total activation = "
                  << std::fixed << std::setprecision(2) << total_activation << std::endl;
    }

    std::cout << "\nObservation: Higher initial activation spreads more total activation" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Summary
    // ========================================================================

    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Network size: " << pattern_names.size() << " patterns, "
              << matrix.GetAssociationCount() << " associations" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Example Complete ===" << std::endl;
    std::cout << "\nKey Takeaways:" << std::endl;
    std::cout << "1. Activation spreads through associations via BFS" << std::endl;
    std::cout << "2. Activation decays with distance (multiple hops)" << std::endl;
    std::cout << "3. Association strength modulates activation transfer" << std::endl;
    std::cout << "4. Bidirectional queries enable backward reasoning" << std::endl;
    std::cout << "5. Propagation depth controls spread vs. focus trade-off" << std::endl;
    std::cout << "6. Applications: semantic priming, spreading activation, memory retrieval" << std::endl;

    return 0;
}
