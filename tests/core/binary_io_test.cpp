// File: tests/core/binary_io_test.cpp
//
// Portable serialization: exact little-endian byte layouts (fixed fixtures)
// for the primitive helpers and the persisted DPAN formats. The fixtures are
// written out byte by byte, so these tests fail on any host whose output
// differs, including big-endian hosts if the helpers stop swapping.
#include "core/binary_io.hpp"
#include "core/clock.hpp"
#include "core/types.hpp"
#include "core/pattern_data.hpp"
#include "association/association_edge.hpp"
#include "nn/sequential.hpp"

#include <gtest/gtest.h>

#include <sstream>

namespace dpan {
namespace {

using Bytes = std::vector<uint8_t>;

Bytes ToBytes(const std::string& s) { return Bytes(s.begin(), s.end()); }
std::string FromBytes(const Bytes& b) { return std::string(b.begin(), b.end()); }

void Append(Bytes& out, const Bytes& more) { out.insert(out.end(), more.begin(), more.end()); }

// Little-endian encodings written independently of the code under test
Bytes U32(uint32_t v) { return {uint8_t(v), uint8_t(v >> 8), uint8_t(v >> 16), uint8_t(v >> 24)}; }
Bytes U64(uint64_t v) {
    Bytes b;
    for (int i = 0; i < 8; ++i) b.push_back(uint8_t(v >> (8 * i)));
    return b;
}
// IEEE-754 bit patterns
Bytes F32Bits(uint32_t bits) { return U32(bits); }
const Bytes kOne = F32Bits(0x3F800000);       // 1.0f
const Bytes kHalf = F32Bits(0x3F000000);      // 0.5f
const Bytes kMinusTwo = F32Bits(0xC0000000);  // -2.0f
const Bytes kZero = F32Bits(0x00000000);      // 0.0f

template <typename T>
Bytes Serialized(const T& value) {
    std::ostringstream out(std::ios::binary);
    value.Serialize(out);
    return ToBytes(out.str());
}

// ============================================================================
// Primitives
// ============================================================================

TEST(BinaryIoTest, WritesLittleEndian) {
    std::ostringstream out(std::ios::binary);
    io::Write<uint32_t>(out, 0x01020304u);
    io::Write<int64_t>(out, -2);
    io::Write<float>(out, 1.0f);
    io::Write<double>(out, 1.0);
    io::Write<uint8_t>(out, 0xAB);
    io::Write<int16_t>(out, -1);

    Bytes expected{0x04, 0x03, 0x02, 0x01};
    Append(expected, {0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF});
    Append(expected, kOne);
    Append(expected, {0, 0, 0, 0, 0, 0, 0xF0, 0x3F});
    Append(expected, {0xAB});
    Append(expected, {0xFF, 0xFF});
    EXPECT_EQ(expected, ToBytes(out.str()));
}

TEST(BinaryIoTest, ReadsWhatItWrites) {
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    io::Write<int32_t>(ss, -123456);
    io::Write<float>(ss, -2.5f);
    io::Write<double>(ss, 3.141592653589793);
    io::WriteSize(ss, 42);
    io::WriteString(ss, "dpan");
    std::vector<float> values{1.5f, -0.25f, 1e-30f};
    io::WriteArray<float>(ss, values.data(), values.size());

    EXPECT_EQ(-123456, io::Read<int32_t>(ss));
    EXPECT_EQ(-2.5f, io::Read<float>(ss));
    EXPECT_EQ(3.141592653589793, io::Read<double>(ss));
    EXPECT_EQ(42u, io::ReadSize(ss));
    EXPECT_EQ("dpan", io::ReadString(ss));
    std::vector<float> read(3);
    io::ReadArray<float>(ss, read.data(), read.size());
    EXPECT_EQ(values, read);
}

TEST(BinaryIoTest, TruncatedOrOversizedInputThrows) {
    std::istringstream short_input(std::string("\x01\x02", 2));
    EXPECT_THROW(io::Read<uint32_t>(short_input), std::runtime_error);

    std::istringstream huge_size(FromBytes(U64(uint64_t{1} << 50)));
    EXPECT_THROW(io::ReadSize(huge_size), std::runtime_error);

    std::istringstream truncated_string(FromBytes(U64(10)) + "abc");
    EXPECT_THROW(io::ReadString(truncated_string), std::runtime_error);
}

// ============================================================================
// Format fixtures
// ============================================================================

TEST(SerializationFixtureTest, PatternIdAndTimestamp) {
    Bytes id_bytes = U64(0x0102030405060708ULL);
    EXPECT_EQ(id_bytes, Serialized(PatternID(0x0102030405060708ULL)));
    std::istringstream id_in(FromBytes(id_bytes));
    EXPECT_EQ(PatternID(0x0102030405060708ULL), PatternID::Deserialize(id_in));

    const int64_t kTime = 1700000000123456LL;  // 2023, microseconds since the Unix epoch
    Bytes time_bytes = U64(static_cast<uint64_t>(kTime));
    EXPECT_EQ(time_bytes, Serialized(Timestamp::FromMicros(kTime)));
    std::istringstream time_in(FromBytes(time_bytes));
    EXPECT_EQ(kTime, Timestamp::Deserialize(time_in).ToMicros());
}

TEST(SerializationFixtureTest, LegacyBootRelativeTimestampsBecomeLoadTime) {
    auto clock = std::make_shared<ManualClock>(Timestamp::FromMicros(1700000000000000LL));
    ScopedClock scoped(clock);

    // Older versions stored time since boot, e.g. 3 hours of uptime
    std::istringstream legacy(FromBytes(U64(3ULL * 3600 * 1000000)));
    EXPECT_EQ(1700000000000000LL, Timestamp::Deserialize(legacy).ToMicros());

    // Zero means "never" and is kept
    std::istringstream never(FromBytes(U64(0)));
    EXPECT_EQ(0, Timestamp::Deserialize(never).ToMicros());
}

TEST(SerializationFixtureTest, FeatureVector) {
    Bytes expected = U64(3);
    Append(expected, kOne);
    Append(expected, kHalf);
    Append(expected, kMinusTwo);

    FeatureVector vector(std::vector<float>{1.0f, 0.5f, -2.0f});
    EXPECT_EQ(expected, Serialized(vector));

    std::istringstream in(FromBytes(expected));
    EXPECT_EQ(vector.Data(), FeatureVector::Deserialize(in).Data());
}

TEST(SerializationFixtureTest, ContextVector) {
    // Count, then (name length, name, value) per dimension in name order
    Bytes expected = U64(2);
    Append(expected, U64(1));
    Append(expected, {'a'});
    Append(expected, kHalf);
    Append(expected, U64(2));
    Append(expected, {'b', 'c'});
    Append(expected, kOne);

    ContextVector context;
    context.Set("bc", 1.0f);
    context.Set("a", 0.5f);
    EXPECT_EQ(expected, Serialized(context));

    std::istringstream in(FromBytes(expected));
    ContextVector read = ContextVector::Deserialize(in);
    EXPECT_EQ(0.5f, read.Get("a"));
    EXPECT_EQ(1.0f, read.Get("bc"));
}

TEST(SerializationFixtureTest, AssociationEdge) {
    const uint64_t kTime = 1700000000000000ULL;
    auto clock = std::make_shared<ManualClock>(Timestamp::FromMicros(static_cast<int64_t>(kTime)));
    ScopedClock scoped(clock);
    AssociationEdge edge(PatternID(7), PatternID(9), AssociationType::SPATIAL, 0.5f);
    edge.SetDecayRate(1.0f);

    Bytes expected = U64(7);
    Append(expected, U64(9));
    Append(expected, {2});           // SPATIAL
    Append(expected, kHalf);         // strength
    Append(expected, U32(0));        // co-occurrence count
    Append(expected, kZero);         // temporal correlation
    Append(expected, kOne);          // decay rate
    Append(expected, U64(kTime));    // last reinforcement (micros)
    Append(expected, U64(kTime));    // creation time (micros)
    Append(expected, U64(0));        // empty context profile
    EXPECT_EQ(expected, Serialized(edge));

    std::istringstream in(FromBytes(expected));
    auto read = AssociationEdge::Deserialize(in);
    EXPECT_EQ(PatternID(7), read->GetSource());
    EXPECT_EQ(PatternID(9), read->GetTarget());
    EXPECT_EQ(AssociationType::SPATIAL, read->GetType());
    EXPECT_EQ(0.5f, read->GetStrength());
    EXPECT_EQ(1.0f, read->GetDecayRate());
    EXPECT_EQ(static_cast<int64_t>(kTime), read->GetLastReinforcement().ToMicros());
}

TEST(SerializationFixtureTest, NeuralNetworkModel) {
    nn::Sequential model;
    auto dense = std::make_unique<nn::Dense>(2, 1, nn::Initialization::ZEROS);
    dense->Weights() = nn::Matrix::FromRows({{1.0f}, {0.5f}});
    dense->Bias() = nn::Matrix::RowVector({-2.0f});
    model.Add(std::move(dense)).Emplace<nn::Sigmoid>();

    Bytes expected{'D', 'P', 'N', 'N'};
    Append(expected, U32(1));            // format version
    Append(expected, U32(2));            // layer count
    Append(expected, U32(5));
    Append(expected, {'D', 'e', 'n', 's', 'e'});
    Append(expected, U64(2));            // weights rows
    Append(expected, U64(1));            // weights cols
    Append(expected, kOne);
    Append(expected, kHalf);
    Append(expected, U64(1));            // bias rows
    Append(expected, U64(1));            // bias cols
    Append(expected, kMinusTwo);
    Append(expected, U32(7));
    Append(expected, {'S', 'i', 'g', 'm', 'o', 'i', 'd'});
    EXPECT_EQ(expected, Serialized(model));

    std::istringstream in(FromBytes(expected));
    nn::Sequential read = nn::Sequential::Deserialize(in);
    EXPECT_EQ(model.Predict({2.0f, 4.0f}), read.Predict({2.0f, 4.0f}));
}

}  // namespace
}  // namespace dpan
