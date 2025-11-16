// File: tests/discovery/pattern_extractor_test.cpp
#include "discovery/pattern_extractor.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

namespace dpan {
namespace {

// Helper to create test raw data
std::vector<uint8_t> CreateNumericData(const std::vector<float>& values) {
    std::vector<uint8_t> bytes;
    bytes.reserve(values.size() * sizeof(float));
    for (float value : values) {
        const uint8_t* byte_ptr = reinterpret_cast<const uint8_t*>(&value);
        bytes.insert(bytes.end(), byte_ptr, byte_ptr + sizeof(float));
    }
    return bytes;
}

// ============================================================================
// PatternExtractor Tests
// ============================================================================

TEST(PatternExtractorTest, ConstructorWithValidConfig) {
    PatternExtractor::Config config;
    config.modality = DataModality::NUMERIC;
    config.min_pattern_size = 10;
    config.max_pattern_size = 1000;

    EXPECT_NO_THROW(PatternExtractor extractor(config));
}

TEST(PatternExtractorTest, ConstructorRejectsInvalidSizeRange) {
    PatternExtractor::Config config;
    config.min_pattern_size = 1000;
    config.max_pattern_size = 10;

    EXPECT_THROW(PatternExtractor extractor(config), std::invalid_argument);
}

TEST(PatternExtractorTest, ConstructorRejectsInvalidNoiseThreshold) {
    PatternExtractor::Config config;
    config.noise_threshold = 1.5f;

    EXPECT_THROW(PatternExtractor extractor(config), std::invalid_argument);
}

TEST(PatternExtractorTest, ConstructorRejectsZeroFeatureDimension) {
    PatternExtractor::Config config;
    config.feature_dimension = 0;

    EXPECT_THROW(PatternExtractor extractor(config), std::invalid_argument);
}

TEST(PatternExtractorTest, ExtractFromEmptyDataReturnsEmpty) {
    PatternExtractor::Config config;
    PatternExtractor extractor(config);

    std::vector<uint8_t> empty_data;
    auto patterns = extractor.Extract(empty_data);

    EXPECT_TRUE(patterns.empty());
}

TEST(PatternExtractorTest, ExtractFromTooSmallDataReturnsEmpty) {
    PatternExtractor::Config config;
    config.min_pattern_size = 100;
    PatternExtractor extractor(config);

    std::vector<uint8_t> small_data(50, 0);
    auto patterns = extractor.Extract(small_data);

    EXPECT_TRUE(patterns.empty());
}

TEST(PatternExtractorTest, ExtractNumericPatternsWorks) {
    PatternExtractor::Config config;
    config.modality = DataModality::NUMERIC;
    config.min_pattern_size = 10;
    config.feature_dimension = 32;
    PatternExtractor extractor(config);

    // Create numeric data (100 floats)
    std::vector<float> numeric_values;
    for (int i = 0; i < 100; ++i) {
        numeric_values.push_back(std::sin(i * 0.1f));
    }
    auto raw_data = CreateNumericData(numeric_values);

    auto patterns = extractor.Extract(raw_data);

    EXPECT_FALSE(patterns.empty());
    for (const auto& pattern : patterns) {
        EXPECT_EQ(DataModality::NUMERIC, pattern.GetModality());
        EXPECT_GT(pattern.GetFeatures().Dimension(), 0u);
    }
}

TEST(PatternExtractorTest, ExtractImagePatternsWorks) {
    PatternExtractor::Config config;
    config.modality = DataModality::IMAGE;
    config.min_pattern_size = 64;
    config.max_pattern_size = 256;
    PatternExtractor extractor(config);

    // Create image data (pixel values 0-255)
    std::vector<uint8_t> image_data;
    for (int i = 0; i < 1024; ++i) {
        image_data.push_back(static_cast<uint8_t>(i % 256));
    }

    auto patterns = extractor.Extract(image_data);

    EXPECT_FALSE(patterns.empty());
    for (const auto& pattern : patterns) {
        EXPECT_EQ(DataModality::IMAGE, pattern.GetModality());
    }
}

TEST(PatternExtractorTest, ExtractAudioPatternsWorks) {
    PatternExtractor::Config config;
    config.modality = DataModality::AUDIO;
    config.min_pattern_size = 10;
    config.noise_threshold = 0.01f;
    PatternExtractor extractor(config);

    // Create audio data (sine wave)
    std::vector<float> audio_samples;
    for (int i = 0; i < 1000; ++i) {
        audio_samples.push_back(std::sin(i * 0.05f));
    }
    auto raw_data = CreateNumericData(audio_samples);

    auto patterns = extractor.Extract(raw_data);

    EXPECT_FALSE(patterns.empty());
    for (const auto& pattern : patterns) {
        EXPECT_EQ(DataModality::AUDIO, pattern.GetModality());
    }
}

TEST(PatternExtractorTest, ExtractTextPatternsWorks) {
    PatternExtractor::Config config;
    config.modality = DataModality::TEXT;
    config.min_pattern_size = 10;
    config.max_pattern_size = 100;
    PatternExtractor extractor(config);

    // Create text data
    std::string text = "The quick brown fox jumps over the lazy dog. "
                      "Pack my box with five dozen liquor jugs.";
    std::vector<uint8_t> text_data(text.begin(), text.end());

    auto patterns = extractor.Extract(text_data);

    EXPECT_FALSE(patterns.empty());
    for (const auto& pattern : patterns) {
        EXPECT_EQ(DataModality::TEXT, pattern.GetModality());
    }
}

TEST(PatternExtractorTest, ExtractFeaturesFromPatternWithFeatures) {
    PatternExtractor::Config config;
    config.enable_normalization = false;
    config.feature_dimension = 32;
    PatternExtractor extractor(config);

    FeatureVector original_features({1.0f, 2.0f, 3.0f});
    PatternData pattern = PatternData::FromFeatures(original_features, DataModality::NUMERIC);

    FeatureVector extracted = extractor.ExtractFeatures(pattern);

    // ExtractFeatures always computes statistical features from raw data
    EXPECT_EQ(config.feature_dimension, extracted.Dimension());
}

TEST(PatternExtractorTest, ExtractFeaturesNormalizesWhenEnabled) {
    PatternExtractor::Config config;
    config.enable_normalization = true;
    PatternExtractor extractor(config);

    FeatureVector features({0.0f, 5.0f, 10.0f});
    PatternData pattern = PatternData::FromFeatures(features, DataModality::NUMERIC);

    FeatureVector extracted = extractor.ExtractFeatures(pattern);

    // Should be normalized to [0, 1]
    for (size_t i = 0; i < extracted.Dimension(); ++i) {
        EXPECT_GE(extracted[i], 0.0f);
        EXPECT_LE(extracted[i], 1.0f);
    }
}

TEST(PatternExtractorTest, ExtractFeaturesFromRawData) {
    PatternExtractor::Config config;
    config.feature_dimension = 32;
    PatternExtractor extractor(config);

    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto raw_data = CreateNumericData(values);
    PatternData pattern = PatternData::FromBytes(raw_data, DataModality::NUMERIC);

    FeatureVector extracted = extractor.ExtractFeatures(pattern);

    EXPECT_EQ(config.feature_dimension, extracted.Dimension());
}

TEST(PatternExtractorTest, FilterNoiseRemovesLowEnergyPatterns) {
    PatternExtractor::Config config;
    config.noise_threshold = 0.5f;
    PatternExtractor extractor(config);

    // Create low-energy data (close to zero)
    std::vector<float> low_energy = {0.01f, 0.02f, 0.01f, 0.015f, 0.012f};
    auto raw_data = CreateNumericData(low_energy);
    PatternData pattern = PatternData::FromBytes(raw_data, DataModality::NUMERIC);

    PatternData filtered = extractor.FilterNoise(pattern);

    // Should be filtered out (empty)
    EXPECT_TRUE(filtered.GetRawData().empty());
}

TEST(PatternExtractorTest, FilterNoiseKeepsHighEnergyPatterns) {
    PatternExtractor::Config config;
    config.noise_threshold = 0.01f;
    PatternExtractor extractor(config);

    // Create high-energy data
    std::vector<float> high_energy = {1.0f, 2.0f, 3.0f, 2.5f, 1.5f};
    auto raw_data = CreateNumericData(high_energy);
    PatternData pattern = PatternData::FromBytes(raw_data, DataModality::NUMERIC);

    PatternData filtered = extractor.FilterNoise(pattern);

    // Should not be empty
    EXPECT_FALSE(filtered.GetRawData().empty());
}

TEST(PatternExtractorTest, AbstractReducesDimensionality) {
    PatternExtractor::Config config;
    config.feature_dimension = 16;
    PatternExtractor extractor(config);

    // Create pattern with many features
    std::vector<float> many_features;
    for (int i = 0; i < 128; ++i) {
        many_features.push_back(static_cast<float>(i));
    }
    PatternData pattern = PatternData::FromFeatures(
        FeatureVector(many_features),
        DataModality::NUMERIC
    );

    PatternData abstracted = extractor.Abstract(pattern);

    EXPECT_LE(abstracted.GetFeatures().Dimension(), config.feature_dimension);
}

TEST(PatternExtractorTest, GetAndSetConfigWorks) {
    PatternExtractor::Config config;
    config.modality = DataModality::IMAGE;
    config.feature_dimension = 64;
    PatternExtractor extractor(config);

    const auto& retrieved_config = extractor.GetConfig();
    EXPECT_EQ(DataModality::IMAGE, retrieved_config.modality);
    EXPECT_EQ(64u, retrieved_config.feature_dimension);

    PatternExtractor::Config new_config;
    new_config.modality = DataModality::AUDIO;
    new_config.feature_dimension = 32;
    extractor.SetConfig(new_config);

    const auto& updated_config = extractor.GetConfig();
    EXPECT_EQ(DataModality::AUDIO, updated_config.modality);
    EXPECT_EQ(32u, updated_config.feature_dimension);
}

TEST(PatternExtractorTest, MultipleExtractionsConsistent) {
    PatternExtractor::Config config;
    config.modality = DataModality::NUMERIC;
    config.min_pattern_size = 10;
    PatternExtractor extractor(config);

    std::vector<float> values;
    for (int i = 0; i < 100; ++i) {
        values.push_back(std::cos(i * 0.1f));
    }
    auto raw_data = CreateNumericData(values);

    auto patterns1 = extractor.Extract(raw_data);
    auto patterns2 = extractor.Extract(raw_data);

    EXPECT_EQ(patterns1.size(), patterns2.size());
}

TEST(PatternExtractorTest, ExtractHandlesVariousDataSizes) {
    PatternExtractor::Config config;
    config.modality = DataModality::NUMERIC;
    config.min_pattern_size = 10;
    config.max_pattern_size = 500;
    PatternExtractor extractor(config);

    // Test with different sizes
    for (size_t size : {50, 100, 200, 500, 1000}) {
        std::vector<float> values(size, 1.0f);
        auto raw_data = CreateNumericData(values);

        auto patterns = extractor.Extract(raw_data);
        EXPECT_FALSE(patterns.empty()) << "Failed for size: " << size;
    }
}

TEST(PatternExtractorTest, NoiseThresholdAffectsExtraction) {
    PatternExtractor::Config low_threshold_config;
    low_threshold_config.modality = DataModality::NUMERIC;
    low_threshold_config.noise_threshold = 0.001f;
    PatternExtractor low_threshold_extractor(low_threshold_config);

    PatternExtractor::Config high_threshold_config;
    high_threshold_config.modality = DataModality::NUMERIC;
    high_threshold_config.noise_threshold = 0.5f;
    PatternExtractor high_threshold_extractor(high_threshold_config);

    // Create data with some noise
    std::vector<float> values;
    for (int i = 0; i < 100; ++i) {
        values.push_back(0.1f * std::sin(i * 0.1f));  // Low amplitude
    }
    auto raw_data = CreateNumericData(values);

    auto patterns_low = low_threshold_extractor.Extract(raw_data);
    auto patterns_high = high_threshold_extractor.Extract(raw_data);

    // Low threshold should extract more patterns
    EXPECT_GE(patterns_low.size(), patterns_high.size());
}

TEST(PatternExtractorTest, FeatureDimensionIsRespected) {
    PatternExtractor::Config config;
    config.feature_dimension = 64;
    PatternExtractor extractor(config);

    std::vector<float> values(100, 1.0f);
    auto raw_data = CreateNumericData(values);
    PatternData pattern = PatternData::FromBytes(raw_data, DataModality::NUMERIC);

    FeatureVector features = extractor.ExtractFeatures(pattern);

    EXPECT_EQ(64u, features.Dimension());
}

} // namespace
} // namespace dpan
