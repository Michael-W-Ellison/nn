// File: src/discovery/pattern_extractor.hpp
#pragma once

#include "core/pattern_data.hpp"
#include <vector>
#include <memory>
#include <cstdint>

namespace dpan {

/// PatternExtractor - Autonomous pattern discovery from raw input data
///
/// Extracts patterns from raw byte data across multiple modalities:
/// - Numeric: Statistical features, time series patterns
/// - Image: Visual features, texture descriptors
/// - Audio: Spectral features, rhythm patterns
/// - Text: N-grams, semantic features
class PatternExtractor {
public:
    /// Configuration for pattern extraction
    struct Config {
        /// Data modality for extraction
        DataModality modality{DataModality::NUMERIC};

        /// Minimum pattern size in bytes
        size_t min_pattern_size{10};

        /// Maximum pattern size in bytes
        size_t max_pattern_size{10000};

        /// Noise threshold for filtering (0.0 to 1.0)
        float noise_threshold{0.1f};

        /// Enable feature normalization
        bool enable_normalization{true};

        /// Feature dimension for extracted patterns
        size_t feature_dimension{128};
    };

    /// Constructor
    /// @param config Extraction configuration
    explicit PatternExtractor(const Config& config);

    /// Extract patterns from raw input data
    /// @param raw_input Raw byte data
    /// @return Vector of extracted patterns
    std::vector<PatternData> Extract(const std::vector<uint8_t>& raw_input) const;

    /// Extract feature vector from pattern data
    /// @param pattern Pattern to extract features from
    /// @return Feature vector representation
    FeatureVector ExtractFeatures(const PatternData& pattern) const;

    /// Filter noise from pattern data
    /// @param pattern Pattern to filter
    /// @return Filtered pattern
    PatternData FilterNoise(const PatternData& pattern) const;

    /// Abstract/compress pattern while preserving essential characteristics
    /// @param pattern Pattern to abstract
    /// @return Abstracted pattern
    PatternData Abstract(const PatternData& pattern) const;

    /// Get current configuration
    const Config& GetConfig() const { return config_; }

    /// Update configuration
    void SetConfig(const Config& config) { config_ = config; }

private:
    Config config_;

    /// Modality-specific extraction methods
    std::vector<PatternData> ExtractNumeric(const std::vector<uint8_t>& raw_input) const;
    std::vector<PatternData> ExtractImage(const std::vector<uint8_t>& raw_input) const;
    std::vector<PatternData> ExtractAudio(const std::vector<uint8_t>& raw_input) const;
    std::vector<PatternData> ExtractText(const std::vector<uint8_t>& raw_input) const;

    /// Normalize features to [0, 1] range
    FeatureVector NormalizeFeatures(const FeatureVector& features) const;

    /// Compute statistical features from numeric data
    FeatureVector ComputeStatisticalFeatures(const std::vector<float>& data) const;

    /// Detect patterns using sliding window
    std::vector<std::vector<uint8_t>> SlidingWindowExtract(
        const std::vector<uint8_t>& raw_input,
        size_t window_size,
        size_t stride) const;

    /// Convert raw bytes to float values (for numeric processing)
    std::vector<float> BytesToFloats(const std::vector<uint8_t>& bytes) const;

    /// Compute signal energy (for noise detection)
    float ComputeEnergy(const std::vector<float>& signal) const;
};

} // namespace dpan
