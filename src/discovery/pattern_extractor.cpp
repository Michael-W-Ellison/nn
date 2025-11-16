// File: src/discovery/pattern_extractor.cpp
#include "pattern_extractor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace dpan {

// ============================================================================
// PatternExtractor Implementation
// ============================================================================

PatternExtractor::PatternExtractor(const Config& config) : config_(config) {
    if (config_.min_pattern_size > config_.max_pattern_size) {
        throw std::invalid_argument("min_pattern_size cannot exceed max_pattern_size");
    }
    if (config_.noise_threshold < 0.0f || config_.noise_threshold > 1.0f) {
        throw std::invalid_argument("noise_threshold must be in range [0.0, 1.0]");
    }
    if (config_.feature_dimension == 0) {
        throw std::invalid_argument("feature_dimension must be greater than 0");
    }
}

std::vector<PatternData> PatternExtractor::Extract(const std::vector<uint8_t>& raw_input) const {
    if (raw_input.empty()) {
        return {};
    }

    if (raw_input.size() < config_.min_pattern_size) {
        return {};
    }

    // Route to modality-specific extraction
    switch (config_.modality) {
        case DataModality::NUMERIC:
            return ExtractNumeric(raw_input);
        case DataModality::IMAGE:
            return ExtractImage(raw_input);
        case DataModality::AUDIO:
            return ExtractAudio(raw_input);
        case DataModality::TEXT:
            return ExtractText(raw_input);
        default:
            throw std::runtime_error("Unsupported modality");
    }
}

FeatureVector PatternExtractor::ExtractFeatures(const PatternData& pattern) const {
    // Always extract statistical features from raw data
    const auto& raw_data = pattern.GetRawData();
    if (raw_data.empty()) {
        return FeatureVector(std::vector<float>(config_.feature_dimension, 0.0f));
    }

    // Convert to floats and compute statistical features
    std::vector<float> float_data = BytesToFloats(raw_data);
    FeatureVector features = ComputeStatisticalFeatures(float_data);

    return config_.enable_normalization ?
        NormalizeFeatures(features) :
        features;
}

PatternData PatternExtractor::FilterNoise(const PatternData& pattern) const {
    const auto& raw_data = pattern.GetRawData();
    if (raw_data.empty()) {
        return pattern;
    }

    // Convert to float for processing
    std::vector<float> signal = BytesToFloats(raw_data);

    // Compute signal energy
    float energy = ComputeEnergy(signal);

    // If energy is below noise threshold, return empty pattern
    if (energy < config_.noise_threshold) {
        return PatternData::FromBytes({}, pattern.GetModality());
    }

    // Simple noise filtering: remove low-amplitude components
    std::vector<float> filtered;
    filtered.reserve(signal.size());

    float threshold = config_.noise_threshold * energy;
    for (float value : signal) {
        if (std::abs(value) > threshold) {
            filtered.push_back(value);
        }
    }

    // Convert back to bytes
    std::vector<uint8_t> filtered_bytes;
    filtered_bytes.reserve(filtered.size() * sizeof(float));
    for (float value : filtered) {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&value);
        filtered_bytes.insert(filtered_bytes.end(), bytes, bytes + sizeof(float));
    }

    return PatternData::FromBytes(filtered_bytes, pattern.GetModality());
}

PatternData PatternExtractor::Abstract(const PatternData& pattern) const {
    // Abstraction via dimensionality reduction
    FeatureVector features = ExtractFeatures(pattern);

    // Downsample features if they exceed target dimension
    if (features.Dimension() > config_.feature_dimension) {
        std::vector<float> abstracted;
        abstracted.reserve(config_.feature_dimension);

        size_t stride = features.Dimension() / config_.feature_dimension;
        for (size_t i = 0; i < config_.feature_dimension; ++i) {
            size_t idx = i * stride;
            if (idx < features.Dimension()) {
                abstracted.push_back(features[idx]);
            }
        }

        return PatternData::FromFeatures(FeatureVector(abstracted), pattern.GetModality());
    }

    return PatternData::FromFeatures(features, pattern.GetModality());
}

// ============================================================================
// Modality-Specific Extraction
// ============================================================================

std::vector<PatternData> PatternExtractor::ExtractNumeric(const std::vector<uint8_t>& raw_input) const {
    std::vector<PatternData> patterns;

    // Convert bytes to floats
    std::vector<float> numeric_data = BytesToFloats(raw_input);

    if (numeric_data.size() < config_.min_pattern_size / sizeof(float)) {
        return patterns;
    }

    // Sliding window extraction
    size_t window_size = std::min(
        config_.max_pattern_size / sizeof(float),
        numeric_data.size()
    );
    size_t stride = window_size / 2;  // 50% overlap

    for (size_t i = 0; i + window_size <= numeric_data.size(); i += stride) {
        std::vector<float> window(
            numeric_data.begin() + i,
            numeric_data.begin() + i + window_size
        );

        // Compute features for this window
        FeatureVector features = ComputeStatisticalFeatures(window);

        // Filter by energy (noise detection)
        float energy = ComputeEnergy(window);
        if (energy > config_.noise_threshold) {
            patterns.push_back(PatternData::FromFeatures(features, DataModality::NUMERIC));
        }
    }

    return patterns;
}

std::vector<PatternData> PatternExtractor::ExtractImage(const std::vector<uint8_t>& raw_input) const {
    std::vector<PatternData> patterns;

    // Simple image feature extraction
    // Assume raw_input contains pixel values (0-255)

    if (raw_input.size() < config_.min_pattern_size) {
        return patterns;
    }

    // Extract patches using sliding window
    size_t patch_size = std::min(
        config_.max_pattern_size,
        raw_input.size()
    );
    size_t stride = patch_size / 2;

    for (size_t i = 0; i + patch_size <= raw_input.size(); i += stride) {
        std::vector<uint8_t> patch(
            raw_input.begin() + i,
            raw_input.begin() + i + patch_size
        );

        // Compute image features (simple statistical approach)
        std::vector<float> float_patch = BytesToFloats(patch);
        FeatureVector features = ComputeStatisticalFeatures(float_patch);

        patterns.push_back(PatternData::FromFeatures(features, DataModality::IMAGE));
    }

    return patterns;
}

std::vector<PatternData> PatternExtractor::ExtractAudio(const std::vector<uint8_t>& raw_input) const {
    std::vector<PatternData> patterns;

    // Convert to audio samples (assuming float encoding)
    std::vector<float> samples = BytesToFloats(raw_input);

    if (samples.size() < config_.min_pattern_size / sizeof(float)) {
        return patterns;
    }

    // Extract audio frames using sliding window
    size_t frame_size = std::min(
        config_.max_pattern_size / sizeof(float),
        samples.size()
    );
    size_t hop_size = frame_size / 4;  // 75% overlap for audio

    for (size_t i = 0; i + frame_size <= samples.size(); i += hop_size) {
        std::vector<float> frame(
            samples.begin() + i,
            samples.begin() + i + frame_size
        );

        // Compute spectral features
        FeatureVector features = ComputeStatisticalFeatures(frame);

        // Check energy
        float energy = ComputeEnergy(frame);
        if (energy > config_.noise_threshold) {
            patterns.push_back(PatternData::FromFeatures(features, DataModality::AUDIO));
        }
    }

    return patterns;
}

std::vector<PatternData> PatternExtractor::ExtractText(const std::vector<uint8_t>& raw_input) const {
    std::vector<PatternData> patterns;

    if (raw_input.size() < config_.min_pattern_size) {
        return patterns;
    }

    // Extract text n-grams/chunks
    size_t chunk_size = std::min(config_.max_pattern_size, raw_input.size());
    size_t stride = chunk_size / 2;

    for (size_t i = 0; i + chunk_size <= raw_input.size(); i += stride) {
        std::vector<uint8_t> chunk(
            raw_input.begin() + i,
            raw_input.begin() + i + chunk_size
        );

        // Compute text features (character frequency, etc.)
        std::vector<float> char_freq(256, 0.0f);
        for (uint8_t byte : chunk) {
            char_freq[byte] += 1.0f;
        }

        // Normalize frequencies
        float total = static_cast<float>(chunk.size());
        for (float& freq : char_freq) {
            freq /= total;
        }

        // Downsample to target feature dimension
        std::vector<float> features;
        features.reserve(config_.feature_dimension);
        size_t bin_size = 256 / config_.feature_dimension;
        for (size_t j = 0; j < config_.feature_dimension; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < bin_size && j * bin_size + k < 256; ++k) {
                sum += char_freq[j * bin_size + k];
            }
            features.push_back(sum);
        }

        patterns.push_back(PatternData::FromFeatures(
            FeatureVector(features),
            DataModality::TEXT
        ));
    }

    return patterns;
}

// ============================================================================
// Helper Methods
// ============================================================================

FeatureVector PatternExtractor::NormalizeFeatures(const FeatureVector& features) const {
    if (features.Dimension() == 0) {
        return features;
    }

    // Find min and max
    float min_val = features[0];
    float max_val = features[0];

    for (size_t i = 1; i < features.Dimension(); ++i) {
        min_val = std::min(min_val, features[i]);
        max_val = std::max(max_val, features[i]);
    }

    // Normalize to [0, 1]
    if (max_val - min_val < 1e-10f) {
        // All values are the same
        return FeatureVector(std::vector<float>(features.Dimension(), 0.5f));
    }

    std::vector<float> normalized;
    normalized.reserve(features.Dimension());

    for (size_t i = 0; i < features.Dimension(); ++i) {
        float norm_val = (features[i] - min_val) / (max_val - min_val);
        normalized.push_back(norm_val);
    }

    return FeatureVector(normalized);
}

FeatureVector PatternExtractor::ComputeStatisticalFeatures(const std::vector<float>& data) const {
    if (data.empty()) {
        return FeatureVector(std::vector<float>(config_.feature_dimension, 0.0f));
    }

    std::vector<float> features;
    features.reserve(config_.feature_dimension);

    // Basic statistics
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    float mean = sum / data.size();
    features.push_back(mean);

    // Variance and standard deviation
    float variance = 0.0f;
    for (float value : data) {
        float diff = value - mean;
        variance += diff * diff;
    }
    variance /= data.size();
    features.push_back(std::sqrt(variance));

    // Min and max
    features.push_back(*std::min_element(data.begin(), data.end()));
    features.push_back(*std::max_element(data.begin(), data.end()));

    // Skewness approximation
    float skewness = 0.0f;
    float std_dev = std::sqrt(variance);
    if (std_dev > 1e-10f) {
        for (float value : data) {
            float z = (value - mean) / std_dev;
            skewness += z * z * z;
        }
        skewness /= data.size();
    }
    features.push_back(skewness);

    // Energy
    features.push_back(ComputeEnergy(data));

    // Zero-crossing rate
    size_t zero_crossings = 0;
    for (size_t i = 1; i < data.size(); ++i) {
        if ((data[i-1] >= 0 && data[i] < 0) || (data[i-1] < 0 && data[i] >= 0)) {
            zero_crossings++;
        }
    }
    features.push_back(static_cast<float>(zero_crossings) / data.size());

    // Percentiles (quartiles)
    std::vector<float> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());

    size_t q1_idx = sorted_data.size() / 4;
    size_t q2_idx = sorted_data.size() / 2;
    size_t q3_idx = 3 * sorted_data.size() / 4;

    features.push_back(sorted_data[q1_idx]);
    features.push_back(sorted_data[q2_idx]);
    features.push_back(sorted_data[q3_idx]);

    // Pad or truncate to target dimension
    while (features.size() < config_.feature_dimension) {
        // Add autocorrelation-like features if we need more
        size_t lag = features.size() - 10;
        if (lag < data.size() / 2) {
            float autocorr = 0.0f;
            for (size_t i = 0; i + lag < data.size(); ++i) {
                autocorr += data[i] * data[i + lag];
            }
            features.push_back(autocorr / (data.size() - lag));
        } else {
            features.push_back(0.0f);
        }
    }

    // Truncate if too many features
    if (features.size() > config_.feature_dimension) {
        features.resize(config_.feature_dimension);
    }

    return FeatureVector(features);
}

std::vector<std::vector<uint8_t>> PatternExtractor::SlidingWindowExtract(
    const std::vector<uint8_t>& raw_input,
    size_t window_size,
    size_t stride) const {

    std::vector<std::vector<uint8_t>> windows;

    if (raw_input.size() < window_size) {
        return windows;
    }

    for (size_t i = 0; i + window_size <= raw_input.size(); i += stride) {
        windows.emplace_back(
            raw_input.begin() + i,
            raw_input.begin() + i + window_size
        );
    }

    return windows;
}

std::vector<float> PatternExtractor::BytesToFloats(const std::vector<uint8_t>& bytes) const {
    std::vector<float> floats;

    // If size is multiple of 4, interpret as float array
    if (bytes.size() % sizeof(float) == 0) {
        floats.reserve(bytes.size() / sizeof(float));
        for (size_t i = 0; i + sizeof(float) <= bytes.size(); i += sizeof(float)) {
            float value;
            std::memcpy(&value, &bytes[i], sizeof(float));
            floats.push_back(value);
        }
    } else {
        // Otherwise, normalize bytes to [0, 1] range
        floats.reserve(bytes.size());
        for (uint8_t byte : bytes) {
            floats.push_back(static_cast<float>(byte) / 255.0f);
        }
    }

    return floats;
}

float PatternExtractor::ComputeEnergy(const std::vector<float>& signal) const {
    if (signal.empty()) {
        return 0.0f;
    }

    float energy = 0.0f;
    for (float value : signal) {
        energy += value * value;
    }

    return energy / signal.size();
}

} // namespace dpan
