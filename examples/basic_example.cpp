// File: examples/basic_example.cpp
//
// Basic pattern recognition example using the DPAN system.
// Demonstrates:
// - Creating a PatternEngine with memory backend
// - Processing raw input data
// - Creating and retrieving patterns
// - Searching for similar patterns
// - Viewing statistics

#include "core/pattern_engine.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

using namespace dpan;

/// Convert float array to byte vector
std::vector<uint8_t> FloatsToBytes(const std::vector<float>& floats) {
    std::vector<uint8_t> bytes;
    bytes.reserve(floats.size() * sizeof(float));

    for (float value : floats) {
        const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(&value);
        bytes.insert(bytes.end(), byte_ptr, byte_ptr + sizeof(float));
    }

    return bytes;
}

int main() {
    std::cout << "=== DPAN Basic Pattern Recognition Example ===\n\n";

    // Step 1: Configure and create PatternEngine
    std::cout << "Step 1: Creating PatternEngine...\n";

    PatternEngine::Config config;
    config.database_type = "memory";
    config.similarity_metric = "context";
    config.enable_auto_refinement = true;
    config.enable_indexing = true;

    // Configure extraction for numeric data
    config.extraction_config.modality = DataModality::NUMERIC;
    config.extraction_config.min_pattern_size = 10;
    config.extraction_config.max_pattern_size = 1000;

    PatternEngine engine(config);
    std::cout << "  ✓ PatternEngine initialized\n\n";

    // Step 2: Create some initial patterns manually
    std::cout << "Step 2: Creating initial patterns...\n";

    std::vector<PatternID> pattern_ids;

    // Pattern 1: Low values
    FeatureVector fv1(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
    PatternData pd1 = PatternData::FromFeatures(fv1, DataModality::NUMERIC);
    PatternID id1 = engine.CreatePattern(pd1, 0.9f);
    pattern_ids.push_back(id1);
    std::cout << "  Created Pattern " << id1.value() << ": [1.0, 2.0, 3.0, 4.0, 5.0]\n";

    // Pattern 2: Medium values
    FeatureVector fv2(std::vector<float>{10.0f, 11.0f, 12.0f, 13.0f, 14.0f});
    PatternData pd2 = PatternData::FromFeatures(fv2, DataModality::NUMERIC);
    PatternID id2 = engine.CreatePattern(pd2, 0.85f);
    pattern_ids.push_back(id2);
    std::cout << "  Created Pattern " << id2.value() << ": [10.0, 11.0, 12.0, 13.0, 14.0]\n";

    // Pattern 3: High values
    FeatureVector fv3(std::vector<float>{100.0f, 101.0f, 102.0f, 103.0f, 104.0f});
    PatternData pd3 = PatternData::FromFeatures(fv3, DataModality::NUMERIC);
    PatternID id3 = engine.CreatePattern(pd3, 0.8f);
    pattern_ids.push_back(id3);
    std::cout << "  Created Pattern " << id3.value() << ": [100.0, 101.0, 102.0, 103.0, 104.0]\n\n";

    // Step 3: Process new input data
    std::cout << "Step 3: Processing new input data...\n";

    // Input similar to Pattern 1
    std::vector<float> input_floats = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
    auto input_bytes = FloatsToBytes(input_floats);

    std::cout << "  Input: [1.5, 2.5, 3.5, 4.5, 5.5]\n";

    auto result = engine.ProcessInput(input_bytes, DataModality::NUMERIC);

    std::cout << "  Processing completed in " << std::fixed << std::setprecision(2)
              << result.processing_time_ms << " ms\n";
    std::cout << "  Created patterns: " << result.created_patterns.size() << "\n";
    std::cout << "  Activated patterns: " << result.activated_patterns.size() << "\n";
    std::cout << "  Updated patterns: " << result.updated_patterns.size() << "\n\n";

    // Step 4: Search for similar patterns
    std::cout << "Step 4: Searching for similar patterns...\n";

    // Query with values similar to Pattern 2
    FeatureVector query_fv(std::vector<float>{10.5f, 11.5f, 12.5f, 13.5f, 14.5f});
    PatternData query_pd = PatternData::FromFeatures(query_fv, DataModality::NUMERIC);

    std::cout << "  Query: [10.5, 11.5, 12.5, 13.5, 14.5]\n";

    auto similar = engine.FindSimilarPatterns(query_pd, 3, 0.0f);

    std::cout << "  Found " << similar.size() << " similar patterns:\n";
    for (const auto& match : similar) {
        std::cout << "    Pattern " << match.pattern_id.value()
                  << " - Similarity: " << std::fixed << std::setprecision(4)
                  << match.similarity << "\n";
    }
    std::cout << "\n";

    // Step 5: Retrieve and display pattern details
    std::cout << "Step 5: Retrieving pattern details...\n";

    for (const auto& pid : pattern_ids) {
        auto pattern_opt = engine.GetPattern(pid);
        if (pattern_opt.has_value()) {
            const auto& pattern = pattern_opt.value();
            std::cout << "  Pattern " << pid.value() << ":\n";
            std::cout << "    Type: " << (pattern.GetType() == PatternType::ATOMIC ? "ATOMIC" :
                                        pattern.GetType() == PatternType::COMPOSITE ? "COMPOSITE" : "META") << "\n";
            std::cout << "    Confidence: " << std::fixed << std::setprecision(2)
                      << pattern.GetConfidenceScore() << "\n";
            std::cout << "    Features: " << pattern.GetData().GetFeatures().Dimension() << " dims\n";
        }
    }
    std::cout << "\n";

    // Step 6: Display engine statistics
    std::cout << "Step 6: Engine statistics:\n";

    auto stats = engine.GetStatistics();
    std::cout << "  Total patterns: " << stats.total_patterns << "\n";
    std::cout << "  Atomic patterns: " << stats.atomic_patterns << "\n";
    std::cout << "  Composite patterns: " << stats.composite_patterns << "\n";
    std::cout << "  Meta patterns: " << stats.meta_patterns << "\n";
    std::cout << "  Average confidence: " << std::fixed << std::setprecision(2)
              << stats.avg_confidence << "\n";
    std::cout << "  Average pattern size: " << std::fixed << std::setprecision(1)
              << stats.avg_pattern_size_bytes << " bytes\n";
    std::cout << "  Memory usage: " << stats.storage_stats.memory_usage_bytes << " bytes\n\n";

    std::cout << "=== Example completed successfully ===\n";

    return 0;
}
