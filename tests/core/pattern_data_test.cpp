// File: tests/core/pattern_data_test.cpp
#include "core/pattern_data.hpp"
#include <gtest/gtest.h>
#include <sstream>

namespace dpan {
namespace {

// ============================================================================
// FeatureVector Tests
// ============================================================================

TEST(FeatureVectorTest, DefaultConstructorCreatesEmpty) {
    FeatureVector fv;
    EXPECT_EQ(0u, fv.Dimension());
}

TEST(FeatureVectorTest, DimensionConstructorInitializesZero) {
    FeatureVector fv(5);
    EXPECT_EQ(5u, fv.Dimension());
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(0.0f, fv[i]);
    }
}

TEST(FeatureVectorTest, DataConstructorCopiesData) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f};
    FeatureVector fv(data);

    EXPECT_EQ(3u, fv.Dimension());
    EXPECT_FLOAT_EQ(1.0f, fv[0]);
    EXPECT_FLOAT_EQ(2.0f, fv[1]);
    EXPECT_FLOAT_EQ(3.0f, fv[2]);
}

TEST(FeatureVectorTest, NormComputation) {
    FeatureVector fv(3);
    fv[0] = 3.0f;
    fv[1] = 4.0f;
    fv[2] = 0.0f;

    // sqrt(3^2 + 4^2) = 5.0
    EXPECT_FLOAT_EQ(5.0f, fv.Norm());
}

TEST(FeatureVectorTest, Normalization) {
    FeatureVector fv(2);
    fv[0] = 3.0f;
    fv[1] = 4.0f;

    FeatureVector normalized = fv.Normalized();

    EXPECT_FLOAT_EQ(1.0f, normalized.Norm());
    EXPECT_FLOAT_EQ(0.6f, normalized[0]);  // 3/5
    EXPECT_FLOAT_EQ(0.8f, normalized[1]);  // 4/5
}

TEST(FeatureVectorTest, DotProduct) {
    FeatureVector fv1(3);
    fv1[0] = 1.0f;
    fv1[1] = 2.0f;
    fv1[2] = 3.0f;

    FeatureVector fv2(3);
    fv2[0] = 4.0f;
    fv2[1] = 5.0f;
    fv2[2] = 6.0f;

    // 1*4 + 2*5 + 3*6 = 32
    EXPECT_FLOAT_EQ(32.0f, fv1.DotProduct(fv2));
}

TEST(FeatureVectorTest, DotProductThrowsOnDimensionMismatch) {
    FeatureVector fv1(3);
    FeatureVector fv2(4);

    EXPECT_THROW(fv1.DotProduct(fv2), std::invalid_argument);
}

TEST(FeatureVectorTest, EuclideanDistance) {
    FeatureVector fv1(2);
    fv1[0] = 0.0f;
    fv1[1] = 0.0f;

    FeatureVector fv2(2);
    fv2[0] = 3.0f;
    fv2[1] = 4.0f;

    // sqrt(3^2 + 4^2) = 5.0
    EXPECT_FLOAT_EQ(5.0f, fv1.EuclideanDistance(fv2));
}

TEST(FeatureVectorTest, CosineSimilarity) {
    FeatureVector fv1(2);
    fv1[0] = 1.0f;
    fv1[1] = 0.0f;

    FeatureVector fv2(2);
    fv2[0] = 1.0f;
    fv2[1] = 0.0f;

    // Identical vectors
    EXPECT_FLOAT_EQ(1.0f, fv1.CosineSimilarity(fv2));

    // Perpendicular vectors
    FeatureVector fv3(2);
    fv3[0] = 0.0f;
    fv3[1] = 1.0f;
    EXPECT_FLOAT_EQ(0.0f, fv1.CosineSimilarity(fv3));
}

TEST(FeatureVectorTest, VectorAddition) {
    FeatureVector fv1(3);
    fv1[0] = 1.0f;
    fv1[1] = 2.0f;
    fv1[2] = 3.0f;

    FeatureVector fv2(3);
    fv2[0] = 4.0f;
    fv2[1] = 5.0f;
    fv2[2] = 6.0f;

    FeatureVector result = fv1 + fv2;

    EXPECT_FLOAT_EQ(5.0f, result[0]);
    EXPECT_FLOAT_EQ(7.0f, result[1]);
    EXPECT_FLOAT_EQ(9.0f, result[2]);
}

TEST(FeatureVectorTest, VectorSubtraction) {
    FeatureVector fv1(3);
    fv1[0] = 5.0f;
    fv1[1] = 7.0f;
    fv1[2] = 9.0f;

    FeatureVector fv2(3);
    fv2[0] = 1.0f;
    fv2[1] = 2.0f;
    fv2[2] = 3.0f;

    FeatureVector result = fv1 - fv2;

    EXPECT_FLOAT_EQ(4.0f, result[0]);
    EXPECT_FLOAT_EQ(5.0f, result[1]);
    EXPECT_FLOAT_EQ(6.0f, result[2]);
}

TEST(FeatureVectorTest, ScalarMultiplication) {
    FeatureVector fv(3);
    fv[0] = 1.0f;
    fv[1] = 2.0f;
    fv[2] = 3.0f;

    FeatureVector result = fv * 2.0f;

    EXPECT_FLOAT_EQ(2.0f, result[0]);
    EXPECT_FLOAT_EQ(4.0f, result[1]);
    EXPECT_FLOAT_EQ(6.0f, result[2]);
}

TEST(FeatureVectorTest, EqualityComparison) {
    FeatureVector fv1(3);
    fv1[0] = 1.0f;
    fv1[1] = 2.0f;
    fv1[2] = 3.0f;

    FeatureVector fv2(3);
    fv2[0] = 1.0f;
    fv2[1] = 2.0f;
    fv2[2] = 3.0f;

    EXPECT_EQ(fv1, fv2);

    fv2[0] = 1.1f;
    EXPECT_NE(fv1, fv2);
}

TEST(FeatureVectorTest, SerializationRoundTrip) {
    FeatureVector original(5);
    original[0] = 1.5f;
    original[1] = 2.5f;
    original[2] = 3.5f;
    original[3] = 4.5f;
    original[4] = 5.5f;

    std::stringstream ss;
    original.Serialize(ss);
    FeatureVector deserialized = FeatureVector::Deserialize(ss);

    EXPECT_EQ(original, deserialized);
}

TEST(FeatureVectorTest, ToStringProducesReadableOutput) {
    FeatureVector fv(3);
    fv[0] = 1.5f;
    fv[1] = 2.5f;
    fv[2] = 3.5f;

    std::string str = fv.ToString();
    EXPECT_NE(std::string::npos, str.find("FeatureVector"));
    EXPECT_NE(std::string::npos, str.find("3"));
}

// ============================================================================
// PatternData Tests
// ============================================================================

TEST(PatternDataTest, DefaultConstructorCreatesEmpty) {
    PatternData pd;
    EXPECT_TRUE(pd.IsEmpty());
    EXPECT_EQ(DataModality::UNKNOWN, pd.GetModality());
}

TEST(PatternDataTest, ModalityConstructorSetsModality) {
    PatternData pd(DataModality::NUMERIC);
    EXPECT_EQ(DataModality::NUMERIC, pd.GetModality());
}

TEST(PatternDataTest, FromBytesCreatesPatternData) {
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    PatternData pd = PatternData::FromBytes(data, DataModality::NUMERIC);

    EXPECT_FALSE(pd.IsEmpty());
    EXPECT_EQ(DataModality::NUMERIC, pd.GetModality());
    EXPECT_EQ(5u, pd.GetOriginalSize());
}

TEST(PatternDataTest, FromBytesThrowsOnOversizedData) {
    std::vector<uint8_t> data(PatternData::kMaxRawDataSize + 1);
    EXPECT_THROW(PatternData::FromBytes(data, DataModality::NUMERIC), std::invalid_argument);
}

TEST(PatternDataTest, FromFeaturesCreatesPatternData) {
    FeatureVector features(3);
    features[0] = 1.0f;
    features[1] = 2.0f;
    features[2] = 3.0f;

    PatternData pd = PatternData::FromFeatures(features, DataModality::NUMERIC);

    EXPECT_FALSE(pd.IsEmpty());
    EXPECT_EQ(DataModality::NUMERIC, pd.GetModality());
}

TEST(PatternDataTest, GetFeaturesRoundTrip) {
    FeatureVector original(3);
    original[0] = 1.5f;
    original[1] = 2.5f;
    original[2] = 3.5f;

    PatternData pd = PatternData::FromFeatures(original, DataModality::NUMERIC);
    FeatureVector retrieved = pd.GetFeatures();

    EXPECT_EQ(original, retrieved);
}

TEST(PatternDataTest, GetRawDataRoundTrip) {
    std::vector<uint8_t> original = {10, 20, 30, 40, 50};
    PatternData pd = PatternData::FromBytes(original, DataModality::IMAGE);

    std::vector<uint8_t> retrieved = pd.GetRawData();

    EXPECT_EQ(original, retrieved);
}

TEST(PatternDataTest, CompressionRatioCalculation) {
    // Create data with lots of repetition (should compress well with RLE)
    std::vector<uint8_t> data(100, 42);  // 100 bytes of value 42
    PatternData pd = PatternData::FromBytes(data, DataModality::NUMERIC);

    // RLE should compress this to 2 bytes (count, value)
    EXPECT_LT(pd.GetCompressionRatio(), 1.0f);
    EXPECT_LT(pd.GetCompressedSize(), pd.GetOriginalSize());
}

TEST(PatternDataTest, CompressionHandlesVariedData) {
    // Create data with varied values
    std::vector<uint8_t> data;
    for (int i = 0; i < 100; ++i) {
        data.push_back(static_cast<uint8_t>(i));
    }

    PatternData pd = PatternData::FromBytes(data, DataModality::NUMERIC);
    std::vector<uint8_t> retrieved = pd.GetRawData();

    EXPECT_EQ(data, retrieved);
}

TEST(PatternDataTest, SerializationRoundTrip) {
    FeatureVector features(5);
    for (size_t i = 0; i < 5; ++i) {
        features[i] = static_cast<float>(i) + 0.5f;
    }

    PatternData original = PatternData::FromFeatures(features, DataModality::AUDIO);

    std::stringstream ss;
    original.Serialize(ss);
    PatternData deserialized = PatternData::Deserialize(ss);

    EXPECT_EQ(original, deserialized);
    EXPECT_EQ(original.GetModality(), deserialized.GetModality());
    EXPECT_EQ(original.GetOriginalSize(), deserialized.GetOriginalSize());
}

TEST(PatternDataTest, ToStringProducesReadableOutput) {
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    PatternData pd = PatternData::FromBytes(data, DataModality::TEXT);

    std::string str = pd.ToString();
    EXPECT_NE(std::string::npos, str.find("PatternData"));
    EXPECT_NE(std::string::npos, str.find("TEXT"));
}

TEST(PatternDataTest, EqualityComparison) {
    std::vector<uint8_t> data1 = {1, 2, 3};
    std::vector<uint8_t> data2 = {1, 2, 3};
    std::vector<uint8_t> data3 = {4, 5, 6};

    PatternData pd1 = PatternData::FromBytes(data1, DataModality::NUMERIC);
    PatternData pd2 = PatternData::FromBytes(data2, DataModality::NUMERIC);
    PatternData pd3 = PatternData::FromBytes(data3, DataModality::NUMERIC);

    EXPECT_EQ(pd1, pd2);
    EXPECT_NE(pd1, pd3);
}

TEST(PatternDataTest, EmptyPatternDataOperations) {
    PatternData pd;

    EXPECT_TRUE(pd.IsEmpty());
    EXPECT_EQ(0u, pd.GetOriginalSize());
    EXPECT_EQ(0u, pd.GetCompressedSize());
    EXPECT_FLOAT_EQ(0.0f, pd.GetCompressionRatio());

    std::vector<uint8_t> raw = pd.GetRawData();
    EXPECT_TRUE(raw.empty());

    FeatureVector features = pd.GetFeatures();
    EXPECT_EQ(0u, features.Dimension());
}

TEST(DataModalityTest, ToStringConvertsCorrectly) {
    EXPECT_STREQ("UNKNOWN", ToString(DataModality::UNKNOWN));
    EXPECT_STREQ("NUMERIC", ToString(DataModality::NUMERIC));
    EXPECT_STREQ("IMAGE", ToString(DataModality::IMAGE));
    EXPECT_STREQ("AUDIO", ToString(DataModality::AUDIO));
    EXPECT_STREQ("TEXT", ToString(DataModality::TEXT));
    EXPECT_STREQ("COMPOSITE", ToString(DataModality::COMPOSITE));
}

} // namespace
} // namespace dpan
