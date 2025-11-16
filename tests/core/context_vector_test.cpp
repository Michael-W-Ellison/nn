// File: tests/core/context_vector_test.cpp
#include "core/types.hpp"
#include <gtest/gtest.h>

namespace dpan {
namespace {

TEST(ContextVectorTest, DefaultConstructorCreatesEmpty) {
    ContextVector cv;
    EXPECT_TRUE(cv.IsEmpty());
    EXPECT_EQ(0u, cv.Size());
}

TEST(ContextVectorTest, SetAndGet) {
    ContextVector cv;
    cv.Set("dim1", 1.5f);
    cv.Set("dim2", 2.5f);

    EXPECT_EQ(1.5f, cv.Get("dim1"));
    EXPECT_EQ(2.5f, cv.Get("dim2"));
    EXPECT_EQ(0.0f, cv.Get("nonexistent"));
    EXPECT_EQ(2u, cv.Size());
}

TEST(ContextVectorTest, SetZeroRemovesDimension) {
    ContextVector cv;
    cv.Set("dim1", 1.5f);
    EXPECT_EQ(1u, cv.Size());

    cv.Set("dim1", 0.0f);
    EXPECT_EQ(0u, cv.Size());
    EXPECT_FALSE(cv.Has("dim1"));
}

TEST(ContextVectorTest, RemoveDimension) {
    ContextVector cv;
    cv.Set("dim1", 1.5f);
    cv.Set("dim2", 2.5f);

    cv.Remove("dim1");
    EXPECT_FALSE(cv.Has("dim1"));
    EXPECT_TRUE(cv.Has("dim2"));
    EXPECT_EQ(1u, cv.Size());
}

TEST(ContextVectorTest, DotProduct) {
    ContextVector cv1;
    cv1.Set("x", 1.0f);
    cv1.Set("y", 2.0f);
    cv1.Set("z", 3.0f);

    ContextVector cv2;
    cv2.Set("x", 4.0f);
    cv2.Set("y", 5.0f);
    cv2.Set("z", 6.0f);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_FLOAT_EQ(32.0f, cv1.DotProduct(cv2));
}

TEST(ContextVectorTest, Norm) {
    ContextVector cv;
    cv.Set("x", 3.0f);
    cv.Set("y", 4.0f);

    // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5
    EXPECT_FLOAT_EQ(5.0f, cv.Norm());
}

TEST(ContextVectorTest, Normalized) {
    ContextVector cv;
    cv.Set("x", 3.0f);
    cv.Set("y", 4.0f);

    ContextVector normalized = cv.Normalized();

    EXPECT_FLOAT_EQ(1.0f, normalized.Norm());
    EXPECT_FLOAT_EQ(0.6f, normalized.Get("x"));  // 3/5
    EXPECT_FLOAT_EQ(0.8f, normalized.Get("y"));  // 4/5
}

TEST(ContextVectorTest, CosineSimilarity) {
    ContextVector cv1;
    cv1.Set("x", 1.0f);
    cv1.Set("y", 0.0f);

    ContextVector cv2;
    cv2.Set("x", 1.0f);
    cv2.Set("y", 0.0f);

    // Identical vectors
    EXPECT_FLOAT_EQ(1.0f, cv1.CosineSimilarity(cv2));

    // Perpendicular vectors
    ContextVector cv3;
    cv3.Set("x", 0.0f);
    cv3.Set("y", 1.0f);
    EXPECT_FLOAT_EQ(0.0f, cv1.CosineSimilarity(cv3));

    // Opposite vectors
    ContextVector cv4;
    cv4.Set("x", -1.0f);
    cv4.Set("y", 0.0f);
    EXPECT_FLOAT_EQ(-1.0f, cv1.CosineSimilarity(cv4));
}

TEST(ContextVectorTest, EuclideanDistance) {
    ContextVector cv1;
    cv1.Set("x", 0.0f);
    cv1.Set("y", 0.0f);

    ContextVector cv2;
    cv2.Set("x", 3.0f);
    cv2.Set("y", 4.0f);

    // Distance = sqrt(3^2 + 4^2) = 5
    EXPECT_FLOAT_EQ(5.0f, cv1.EuclideanDistance(cv2));
}

TEST(ContextVectorTest, VectorAddition) {
    ContextVector cv1;
    cv1.Set("x", 1.0f);
    cv1.Set("y", 2.0f);

    ContextVector cv2;
    cv2.Set("x", 3.0f);
    cv2.Set("z", 4.0f);

    ContextVector result = cv1 + cv2;

    EXPECT_FLOAT_EQ(4.0f, result.Get("x"));  // 1 + 3
    EXPECT_FLOAT_EQ(2.0f, result.Get("y"));  // 2 + 0
    EXPECT_FLOAT_EQ(4.0f, result.Get("z"));  // 0 + 4
}

TEST(ContextVectorTest, ScalarMultiplication) {
    ContextVector cv;
    cv.Set("x", 2.0f);
    cv.Set("y", 3.0f);

    ContextVector result = cv * 2.0f;

    EXPECT_FLOAT_EQ(4.0f, result.Get("x"));
    EXPECT_FLOAT_EQ(6.0f, result.Get("y"));
}

TEST(ContextVectorTest, SerializationRoundTrip) {
    ContextVector original;
    original.Set("dim1", 1.5f);
    original.Set("dim2", 2.5f);
    original.Set("dim3", 3.5f);

    std::stringstream ss;
    original.Serialize(ss);
    ContextVector deserialized = ContextVector::Deserialize(ss);

    EXPECT_EQ(original, deserialized);
}

TEST(ContextVectorTest, ToStringProducesReadableOutput) {
    ContextVector cv;
    cv.Set("temperature", 25.5f);
    cv.Set("humidity", 60.0f);

    std::string str = cv.ToString();
    EXPECT_NE(std::string::npos, str.find("temperature"));
    EXPECT_NE(std::string::npos, str.find("humidity"));
}

TEST(ContextVectorTest, SparseVectorEfficiency) {
    // Test that sparse vectors with few overlapping dimensions are efficient
    ContextVector cv1;
    for (int i = 0; i < 1000; ++i) {
        cv1.Set("dim" + std::to_string(i), static_cast<float>(i));
    }

    ContextVector cv2;
    for (int i = 500; i < 1500; ++i) {
        cv2.Set("dim" + std::to_string(i), static_cast<float>(i));
    }

    // This should be fast despite large vectors
    float dot = cv1.DotProduct(cv2);
    EXPECT_GT(dot, 0.0f);
}

} // namespace
} // namespace dpan
