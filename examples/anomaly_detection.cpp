// File: examples/anomaly_detection.cpp
//
// Anomaly detection example using the DPAN pattern recognition system.
// Demonstrates:
// - Learning normal patterns from training data
// - Detecting anomalies in new data
// - Using similarity thresholds for anomaly detection
// - Adaptive pattern learning

#include "core/pattern_engine.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

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

/// Generate normal data (sine wave with noise)
std::vector<float> GenerateNormalData(size_t samples, float frequency = 1.0f, float noise_level = 0.1f) {
    std::mt19937 gen(42);
    std::normal_distribution<float> noise(0.0f, noise_level);

    std::vector<float> data(samples);
    for (size_t i = 0; i < samples; ++i) {
        float t = static_cast<float>(i) / samples;
        data[i] = std::sin(2.0f * M_PI * frequency * t) + noise(gen);
    }

    return data;
}

/// Generate anomalous data (spike)
std::vector<float> GenerateAnomalyData(size_t samples) {
    std::vector<float> data(samples);
    for (size_t i = 0; i < samples; ++i) {
        float t = static_cast<float>(i) / samples;
        // Sudden spike in the middle
        if (i == samples / 2) {
            data[i] = 5.0f;  // Large spike
        } else {
            data[i] = std::sin(2.0f * M_PI * t) * 0.1f;
        }
    }

    return data;
}

/// Check if data is anomalous based on similarity to known patterns
bool IsAnomaly(PatternEngine& engine, const std::vector<float>& data, float threshold) {
    FeatureVector fv(data);
    PatternData pd = PatternData::FromFeatures(fv, DataModality::NUMERIC);

    auto similar = engine.FindSimilarPatterns(pd, 1, 0.0f);

    if (similar.empty()) {
        return true;  // No similar patterns = anomaly
    }

    return similar[0].similarity < threshold;
}

int main() {
    std::cout << "=== DPAN Anomaly Detection Example ===\n\n";

    // Step 1: Configure PatternEngine
    std::cout << "Step 1: Configuring anomaly detection system...\n";

    PatternEngine::Config config;
    config.database_type = "memory";
    config.similarity_metric = "context";
    config.enable_auto_refinement = true;
    config.enable_indexing = true;

    config.extraction_config.modality = DataModality::NUMERIC;
    config.extraction_config.min_pattern_size = 10;
    config.extraction_config.feature_dimension = 32;

    // Higher threshold for strong matches (normal patterns)
    config.matching_config.similarity_threshold = 0.75f;
    config.matching_config.strong_match_threshold = 0.85f;

    PatternEngine engine(config);
    std::cout << "  ✓ System initialized\n\n";

    // Step 2: Train on normal data
    std::cout << "Step 2: Learning normal patterns...\n";

    const size_t num_training_samples = 20;
    const size_t window_size = 10;

    for (size_t i = 0; i < num_training_samples; ++i) {
        auto normal_data = GenerateNormalData(window_size);
        auto bytes = FloatsToBytes(normal_data);

        engine.ProcessInput(bytes, DataModality::NUMERIC);
    }

    auto stats = engine.GetStatistics();
    std::cout << "  Learned " << stats.total_patterns << " normal patterns\n";
    std::cout << "  Average confidence: " << std::fixed << std::setprecision(2)
              << stats.avg_confidence << "\n\n";

    // Step 3: Test anomaly detection
    std::cout << "Step 3: Testing anomaly detection...\n\n";

    const float anomaly_threshold = 0.6f;  // Similarity below this = anomaly

    // Test 1: Normal data
    std::cout << "  Test 1: Normal data\n";
    auto normal_test = GenerateNormalData(window_size);
    bool is_normal_anomaly = IsAnomaly(engine, normal_test, anomaly_threshold);

    FeatureVector normal_fv(normal_test);
    PatternData normal_pd = PatternData::FromFeatures(normal_fv, DataModality::NUMERIC);
    auto normal_similar = engine.FindSimilarPatterns(normal_pd, 1, 0.0f);

    std::cout << "    Max similarity: " << std::fixed << std::setprecision(4)
              << (normal_similar.empty() ? 0.0f : normal_similar[0].similarity) << "\n";
    std::cout << "    Classification: " << (is_normal_anomaly ? "ANOMALY" : "NORMAL") << "\n";
    std::cout << "    ✓ " << (is_normal_anomaly ? "False positive!" : "Correctly identified") << "\n\n";

    // Test 2: Anomalous data (spike)
    std::cout << "  Test 2: Anomalous data (spike)\n";
    auto anomaly_test = GenerateAnomalyData(window_size);
    bool is_spike_anomaly = IsAnomaly(engine, anomaly_test, anomaly_threshold);

    FeatureVector anomaly_fv(anomaly_test);
    PatternData anomaly_pd = PatternData::FromFeatures(anomaly_fv, DataModality::NUMERIC);
    auto anomaly_similar = engine.FindSimilarPatterns(anomaly_pd, 1, 0.0f);

    std::cout << "    Max similarity: " << std::fixed << std::setprecision(4)
              << (anomaly_similar.empty() ? 0.0f : anomaly_similar[0].similarity) << "\n";
    std::cout << "    Classification: " << (is_spike_anomaly ? "ANOMALY" : "NORMAL") << "\n";
    std::cout << "    ✓ " << (is_spike_anomaly ? "Correctly detected!" : "Missed anomaly!") << "\n\n";

    // Test 3: Very different data
    std::cout << "  Test 3: Completely different data\n";
    std::vector<float> different_data(window_size, 100.0f);  // Constant high values
    bool is_different_anomaly = IsAnomaly(engine, different_data, anomaly_threshold);

    FeatureVector different_fv(different_data);
    PatternData different_pd = PatternData::FromFeatures(different_fv, DataModality::NUMERIC);
    auto different_similar = engine.FindSimilarPatterns(different_pd, 1, 0.0f);

    std::cout << "    Max similarity: " << std::fixed << std::setprecision(4)
              << (different_similar.empty() ? 0.0f : different_similar[0].similarity) << "\n";
    std::cout << "    Classification: " << (is_different_anomaly ? "ANOMALY" : "NORMAL") << "\n";
    std::cout << "    ✓ " << (is_different_anomaly ? "Correctly detected!" : "Missed anomaly!") << "\n\n";

    // Step 4: Continuous learning
    std::cout << "Step 4: Demonstrating adaptive learning...\n";

    // Add the normal test data to training set
    auto normal_bytes = FloatsToBytes(normal_test);
    engine.ProcessInput(normal_bytes, DataModality::NUMERIC);

    auto updated_stats = engine.GetStatistics();
    std::cout << "  Patterns after update: " << updated_stats.total_patterns << "\n";
    std::cout << "  System adapted to new normal pattern\n\n";

    // Step 5: Summary
    std::cout << "Step 5: Summary\n";
    std::cout << "  Detection threshold: " << std::fixed << std::setprecision(2)
              << anomaly_threshold << "\n";
    std::cout << "  Total patterns learned: " << updated_stats.total_patterns << "\n";
    std::cout << "  Normal data correctly classified: "
              << (!is_normal_anomaly ? "YES" : "NO") << "\n";
    std::cout << "  Spike anomaly detected: "
              << (is_spike_anomaly ? "YES" : "NO") << "\n";
    std::cout << "  Different data detected: "
              << (is_different_anomaly ? "YES" : "NO") << "\n\n";

    std::cout << "=== Anomaly detection example completed ===\n";

    return 0;
}
