// File: tests/core/types_test.cpp
#include "core/types.hpp"
#include <gtest/gtest.h>
#include <unordered_set>
#include <thread>
#include <vector>

namespace dpan {
namespace {

TEST(PatternIDTest, DefaultConstructorCreatesInvalid) {
    PatternID id;
    EXPECT_FALSE(id.IsValid());
    EXPECT_EQ(0u, id.value());
}

TEST(PatternIDTest, GenerateCreatesUniqueIDs) {
    PatternID id1 = PatternID::Generate();
    PatternID id2 = PatternID::Generate();

    EXPECT_TRUE(id1.IsValid());
    EXPECT_TRUE(id2.IsValid());
    EXPECT_NE(id1, id2);
}

TEST(PatternIDTest, GenerateIsThreadSafe) {
    constexpr int kNumThreads = 10;
    constexpr int kIDsPerThread = 1000;

    std::vector<std::thread> threads;
    std::vector<std::vector<PatternID>> thread_ids(kNumThreads);

    // Generate IDs in parallel
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&thread_ids, i]() {
            for (int j = 0; j < kIDsPerThread; ++j) {
                thread_ids[i].push_back(PatternID::Generate());
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all IDs are unique
    std::unordered_set<PatternID> unique_ids;
    for (const auto& ids : thread_ids) {
        for (const auto& id : ids) {
            EXPECT_TRUE(id.IsValid());
            EXPECT_TRUE(unique_ids.insert(id).second) << "Duplicate ID: " << id.ToString();
        }
    }

    EXPECT_EQ(kNumThreads * kIDsPerThread, unique_ids.size());
}

TEST(PatternIDTest, ComparisonOperators) {
    PatternID id1(100);
    PatternID id2(200);
    PatternID id3(100);

    EXPECT_EQ(id1, id3);
    EXPECT_NE(id1, id2);
    EXPECT_LT(id1, id2);
    EXPECT_GT(id2, id1);
    EXPECT_LE(id1, id2);
    EXPECT_LE(id1, id3);
    EXPECT_GE(id2, id1);
    EXPECT_GE(id1, id3);
}

TEST(PatternIDTest, HashingWorks) {
    std::unordered_set<PatternID> id_set;

    for (int i = 0; i < 100; ++i) {
        PatternID id = PatternID::Generate();
        id_set.insert(id);
    }

    EXPECT_EQ(100u, id_set.size());
}

TEST(PatternIDTest, SerializationRoundTrip) {
    PatternID original = PatternID::Generate();

    std::stringstream ss;
    original.Serialize(ss);

    PatternID deserialized = PatternID::Deserialize(ss);

    EXPECT_EQ(original, deserialized);
}

TEST(PatternIDTest, ToStringProducesReadableOutput) {
    PatternID invalid;
    EXPECT_NE(std::string::npos, invalid.ToString().find("INVALID"));

    PatternID valid = PatternID::Generate();
    std::string str = valid.ToString();
    EXPECT_NE(std::string::npos, str.find("PatternID"));
    EXPECT_GT(str.length(), 0u);
}

// Enum Tests

TEST(EnumTest, PatternTypeToString) {
    EXPECT_STREQ("ATOMIC", ToString(PatternType::ATOMIC));
    EXPECT_STREQ("COMPOSITE", ToString(PatternType::COMPOSITE));
    EXPECT_STREQ("META", ToString(PatternType::META));
}

TEST(EnumTest, ParsePatternType) {
    EXPECT_EQ(PatternType::ATOMIC, ParsePatternType("ATOMIC"));
    EXPECT_EQ(PatternType::COMPOSITE, ParsePatternType("COMPOSITE"));
    EXPECT_EQ(PatternType::META, ParsePatternType("META"));
    EXPECT_THROW(ParsePatternType("INVALID"), std::invalid_argument);
}

TEST(EnumTest, AssociationTypeToString) {
    EXPECT_STREQ("CAUSAL", ToString(AssociationType::CAUSAL));
    EXPECT_STREQ("CATEGORICAL", ToString(AssociationType::CATEGORICAL));
    EXPECT_STREQ("SPATIAL", ToString(AssociationType::SPATIAL));
    EXPECT_STREQ("FUNCTIONAL", ToString(AssociationType::FUNCTIONAL));
    EXPECT_STREQ("COMPOSITIONAL", ToString(AssociationType::COMPOSITIONAL));
}

TEST(EnumTest, ParseAssociationType) {
    EXPECT_EQ(AssociationType::CAUSAL, ParseAssociationType("CAUSAL"));
    EXPECT_THROW(ParseAssociationType("INVALID"), std::invalid_argument);
}

// Timestamp Tests

TEST(TimestampTest, NowCreatesValidTimestamp) {
    Timestamp t1 = Timestamp::Now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Timestamp t2 = Timestamp::Now();

    EXPECT_LT(t1, t2);
}

TEST(TimestampTest, DurationCalculation) {
    Timestamp t1 = Timestamp::Now();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    Timestamp t2 = Timestamp::Now();

    auto duration = t2 - t1;
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    EXPECT_GE(millis, 100);
    EXPECT_LT(millis, 150); // Allow some overhead
}

TEST(TimestampTest, SerializationRoundTrip) {
    Timestamp original = Timestamp::Now();

    std::stringstream ss;
    original.Serialize(ss);
    Timestamp deserialized = Timestamp::Deserialize(ss);

    EXPECT_EQ(original.ToMicros(), deserialized.ToMicros());
}

TEST(TimestampTest, FromMicrosRoundTrip) {
    int64_t micros = 1234567890123456LL;
    Timestamp ts = Timestamp::FromMicros(micros);

    EXPECT_EQ(micros, ts.ToMicros());
}

} // namespace
} // namespace dpan
