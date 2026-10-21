// File: examples/consumer/main.cpp
//
// Minimal program using DPAN as a library: stores feature-vector patterns,
// searches them by similarity, learns associations from a stream of
// activations and runs memory maintenance.

#include "association/association_learning_system.hpp"
#include "core/pattern_engine.hpp"
#include "memory/memory_manager.hpp"
#include "similarity/geometric_similarity.hpp"

#include <iostream>
#include <thread>
#include <vector>

using namespace dpan;

int main() {
    // 1. Pattern storage and similarity search
    PatternEngine::Config engine_config;
    engine_config.database_type = "memory";
    engine_config.similarity_metric = "cosine";
    PatternEngine engine(engine_config);

    std::vector<PatternID> ids;
    for (int i = 0; i < 4; ++i) {
        FeatureVector features(std::vector<float>{
            static_cast<float>(i), 1.0f, static_cast<float>(i % 2)});
        ids.push_back(engine.CreatePattern(PatternData::FromFeatures(features, DataModality::NUMERIC)));
    }

    auto query = engine.GetPattern(ids[1])->GetData();
    auto similar = engine.FindSimilarPatterns(query, 3);
    std::cout << "Patterns: " << engine.GetStatistics().total_patterns
              << ", most similar to #1: " << similar.front().pattern_id.ToString() << "\n";

    // 2. Association learning from a stream of activations
    AssociationLearningSystem::Config assoc_config;
    assoc_config.co_occurrence.window_size = std::chrono::milliseconds(20);
    assoc_config.formation.min_co_occurrences = 3;
    AssociationLearningSystem associations(assoc_config);

    for (int episode = 0; episode < 5; ++episode) {
        associations.RecordPatternActivation(ids[0]);
        associations.RecordPatternActivation(ids[1]);
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
    size_t formed = associations.FormNewAssociations(engine.GetDatabase());
    std::cout << "Associations formed: " << formed << "\n";

    for (const auto& [target, confidence] : associations.PredictWithConfidence(ids[1], 1)) {
        std::cout << "Predicted after #1: " << target.ToString() << " (" << confidence << ")\n";
    }

    // 3. Memory management over the same database and associations
    MemoryManager memory;
    memory.Initialize(&engine.GetDatabase(),
                      const_cast<AssociationMatrix*>(&associations.GetAssociationMatrix()),
                      std::make_shared<CosineSimilarity>());
    memory.RecordPatternAccess(ids[0]);
    memory.PerformMaintenance();
    std::cout << "Patterns after maintenance: " << memory.GetStatistics().total_patterns << "\n";

    return formed > 0 ? 0 : 1;
}
