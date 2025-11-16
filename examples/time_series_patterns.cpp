// File: examples/time_series_patterns.cpp
//
// Time series pattern recognition example using DPAN.
// Demonstrates:
// - Finding recurring patterns in time series data
// - Pattern discovery and matching
// - Similarity search across temporal data
// - Using temporal similarity metrics

#include "core/pattern_engine.hpp"
#include <iostream>
#include <vector>
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

/// Generate time series with repeating patterns
std::vector<float> GenerateTimeSeriesWithPatterns(size_t total_length) {
    std::vector<float> series;
    series.reserve(total_length);

    // Pattern 1: Rising trend (repeats 3 times)
    std::vector<float> pattern1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // Pattern 2: Falling trend (repeats 3 times)
    std::vector<float> pattern2 = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

    // Pattern 3: Oscillation (repeats 2 times)
    std::vector<float> pattern3 = {3.0f, 5.0f, 3.0f, 1.0f, 3.0f};

    // Combine patterns
    for (int i = 0; i < 3; ++i) {
        series.insert(series.end(), pattern1.begin(), pattern1.end());
    }
    for (int i = 0; i < 3; ++i) {
        series.insert(series.end(), pattern2.begin(), pattern2.end());
    }
    for (int i = 0; i < 2; ++i) {
        series.insert(series.end(), pattern3.begin(), pattern3.end());
    }

    return series;
}

/// Extract sliding windows from time series
std::vector<std::vector<float>> ExtractWindows(const std::vector<float>& series, size_t window_size, size_t stride = 1) {
    std::vector<std::vector<float>> windows;

    for (size_t i = 0; i + window_size <= series.size(); i += stride) {
        std::vector<float> window(series.begin() + i, series.begin() + i + window_size);
        windows.push_back(window);
    }

    return windows;
}

int main() {
    std::cout << "=== DPAN Time Series Pattern Recognition Example ===\n\n";

    // Step 1: Generate time series data
    std::cout << "Step 1: Generating time series with recurring patterns...\n";

    auto time_series = GenerateTimeSeriesWithPatterns(100);

    std::cout << "  Generated time series with " << time_series.size() << " data points\n";
    std::cout << "  Contains 3 types of patterns:\n";
    std::cout << "    - Rising trend (3 occurrences)\n";
    std::cout << "    - Falling trend (3 occurrences)\n";
    std::cout << "    - Oscillation (2 occurrences)\n\n";

    // Step 2: Configure PatternEngine for time series
    std::cout << "Step 2: Configuring pattern recognition engine...\n";

    PatternEngine::Config config;
    config.database_type = "memory";
    config.similarity_metric = "context";  // Good for temporal patterns
    config.enable_auto_refinement = true;
    config.enable_indexing = true;

    config.extraction_config.modality = DataModality::NUMERIC;
    config.extraction_config.min_pattern_size = 5;
    config.extraction_config.feature_dimension = 16;

    // Lower threshold to catch pattern variations
    config.matching_config.similarity_threshold = 0.65f;
    config.matching_config.strong_match_threshold = 0.80f;

    PatternEngine engine(config);
    std::cout << "  ✓ Engine initialized with temporal configuration\n\n";

    // Step 3: Discover patterns in time series
    std::cout << "Step 3: Discovering patterns in time series...\n";

    const size_t window_size = 5;  // Match our pattern length
    const size_t stride = 5;       // Non-overlapping windows to find distinct patterns

    auto windows = ExtractWindows(time_series, window_size, stride);

    std::cout << "  Extracted " << windows.size() << " windows of size " << window_size << "\n";

    // Process each window
    size_t patterns_discovered = 0;
    for (size_t i = 0; i < windows.size(); ++i) {
        auto bytes = FloatsToBytes(windows[i]);
        auto result = engine.ProcessInput(bytes, DataModality::NUMERIC);

        if (!result.created_patterns.empty()) {
            patterns_discovered += result.created_patterns.size();
            std::cout << "  Window " << std::setw(2) << i << ": ";

            // Display window values
            std::cout << "[";
            for (size_t j = 0; j < windows[i].size(); ++j) {
                std::cout << std::fixed << std::setprecision(1) << windows[i][j];
                if (j < windows[i].size() - 1) std::cout << ", ";
            }
            std::cout << "] -> Created " << result.created_patterns.size() << " pattern(s)\n";
        }
    }

    std::cout << "\n  Total patterns discovered: " << patterns_discovered << "\n\n";

    // Step 4: Analyze discovered patterns
    std::cout << "Step 4: Analyzing discovered patterns...\n";

    auto stats = engine.GetStatistics();
    std::cout << "  Total unique patterns: " << stats.total_patterns << "\n";
    std::cout << "  Average confidence: " << std::fixed << std::setprecision(2)
              << stats.avg_confidence << "\n\n";

    // Step 5: Search for specific pattern types
    std::cout << "Step 5: Searching for similar patterns...\n\n";

    // Query 1: Rising trend pattern
    std::cout << "  Query 1: Rising trend [1.0, 2.0, 3.0, 4.0, 5.0]\n";
    std::vector<float> rising_query = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    FeatureVector rising_fv(rising_query);
    PatternData rising_pd = PatternData::FromFeatures(rising_fv, DataModality::NUMERIC);

    auto rising_matches = engine.FindSimilarPatterns(rising_pd, 5, 0.5f);
    std::cout << "    Found " << rising_matches.size() << " similar patterns:\n";
    for (const auto& match : rising_matches) {
        std::cout << "      Pattern " << match.pattern_id.value()
                  << " - Similarity: " << std::fixed << std::setprecision(4)
                  << match.similarity << "\n";
    }
    std::cout << "\n";

    // Query 2: Falling trend pattern
    std::cout << "  Query 2: Falling trend [5.0, 4.0, 3.0, 2.0, 1.0]\n";
    std::vector<float> falling_query = {5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    FeatureVector falling_fv(falling_query);
    PatternData falling_pd = PatternData::FromFeatures(falling_fv, DataModality::NUMERIC);

    auto falling_matches = engine.FindSimilarPatterns(falling_pd, 5, 0.5f);
    std::cout << "    Found " << falling_matches.size() << " similar patterns:\n";
    for (const auto& match : falling_matches) {
        std::cout << "      Pattern " << match.pattern_id.value()
                  << " - Similarity: " << std::fixed << std::setprecision(4)
                  << match.similarity << "\n";
    }
    std::cout << "\n";

    // Query 3: Oscillation pattern
    std::cout << "  Query 3: Oscillation [3.0, 5.0, 3.0, 1.0, 3.0]\n";
    std::vector<float> osc_query = {3.0f, 5.0f, 3.0f, 1.0f, 3.0f};
    FeatureVector osc_fv(osc_query);
    PatternData osc_pd = PatternData::FromFeatures(osc_fv, DataModality::NUMERIC);

    auto osc_matches = engine.FindSimilarPatterns(osc_pd, 5, 0.5f);
    std::cout << "    Found " << osc_matches.size() << " similar patterns:\n";
    for (const auto& match : osc_matches) {
        std::cout << "      Pattern " << match.pattern_id.value()
                  << " - Similarity: " << std::fixed << std::setprecision(4)
                  << match.similarity << "\n";
    }
    std::cout << "\n";

    // Step 6: Test with new unseen data
    std::cout << "Step 6: Testing with new unseen data...\n";

    // Similar to rising trend but with slight variation
    std::vector<float> new_rising = {1.2f, 2.1f, 3.3f, 4.2f, 4.8f};
    FeatureVector new_rising_fv(new_rising);
    PatternData new_rising_pd = PatternData::FromFeatures(new_rising_fv, DataModality::NUMERIC);

    auto new_matches = engine.FindSimilarPatterns(new_rising_pd, 3, 0.4f);

    std::cout << "  New data: [1.2, 2.1, 3.3, 4.2, 4.8]\n";
    std::cout << "  Found " << new_matches.size() << " matching patterns:\n";
    for (const auto& match : new_matches) {
        std::cout << "    Pattern " << match.pattern_id.value()
                  << " - Similarity: " << std::fixed << std::setprecision(4)
                  << match.similarity;

        if (match.similarity > 0.7f) {
            std::cout << " (Strong match - likely rising trend)";
        } else if (match.similarity > 0.5f) {
            std::cout << " (Moderate match)";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Step 7: Summary
    std::cout << "Step 7: Pattern Recognition Summary\n";
    std::cout << "  Time series length: " << time_series.size() << " points\n";
    std::cout << "  Window size: " << window_size << "\n";
    std::cout << "  Unique patterns found: " << stats.total_patterns << "\n";
    std::cout << "  Rising trend matches: " << rising_matches.size() << "\n";
    std::cout << "  Falling trend matches: " << falling_matches.size() << "\n";
    std::cout << "  Oscillation matches: " << osc_matches.size() << "\n";
    std::cout << "  ✓ Successfully identified recurring patterns in time series\n\n";

    std::cout << "=== Time series example completed ===\n";

    return 0;
}
