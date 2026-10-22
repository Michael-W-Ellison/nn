// File: src/core/binary_io.hpp
//
// Portable binary serialization: fixed-width little-endian integers and
// IEEE-754 floats, independent of the host's byte order. Sizes and counts
// are always written as 64-bit values.
//
// On little-endian hosts the bytes are identical to writing the values from
// memory, so files written before these helpers existed (on x86-64 or ARM64)
// remain readable.
//
// Read functions throw std::runtime_error when the stream ends early.
#pragma once

#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace dpan {
namespace io {

namespace detail {

template <typename T>
using UnsignedOfSize = std::conditional_t<sizeof(T) == 1, uint8_t,
                       std::conditional_t<sizeof(T) == 2, uint16_t,
                       std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;

inline void Fail(const char* what) {
    throw std::runtime_error(std::string("binary read: ") + what);
}

}  // namespace detail

/// Write an integer, enum-underlying value or float (1, 2, 4 or 8 bytes)
template <typename T>
void Write(std::ostream& out, T value) {
    static_assert(std::is_arithmetic_v<T>, "io::Write needs an arithmetic type");
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                  "io::Write supports 1, 2, 4 and 8 byte types");
    using Bits = detail::UnsignedOfSize<T>;
    Bits bits;
    std::memcpy(&bits, &value, sizeof(T));
    unsigned char bytes[sizeof(T)];
    for (size_t i = 0; i < sizeof(T); ++i) {
        bytes[i] = static_cast<unsigned char>((static_cast<uint64_t>(bits) >> (8 * i)) & 0xFF);
    }
    out.write(reinterpret_cast<const char*>(bytes), sizeof(T));
}

/// Read a value written by Write<T>
template <typename T>
T Read(std::istream& in) {
    static_assert(std::is_arithmetic_v<T>, "io::Read needs an arithmetic type");
    using Bits = detail::UnsignedOfSize<T>;
    unsigned char bytes[sizeof(T)];
    in.read(reinterpret_cast<char*>(bytes), sizeof(T));
    if (!in) {
        detail::Fail("unexpected end of data");
    }
    uint64_t bits = 0;
    for (size_t i = 0; i < sizeof(T); ++i) {
        bits |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    Bits narrowed = static_cast<Bits>(bits);
    T value;
    std::memcpy(&value, &narrowed, sizeof(T));
    return value;
}

/// Sizes and counts: always 64-bit on disk
inline void WriteSize(std::ostream& out, size_t value) {
    Write<uint64_t>(out, static_cast<uint64_t>(value));
}

/// @param max Largest accepted value (guards allocations on corrupt input)
inline size_t ReadSize(std::istream& in, uint64_t max = (uint64_t{1} << 40)) {
    uint64_t value = Read<uint64_t>(in);
    if (value > max) {
        detail::Fail("size out of range");
    }
    return static_cast<size_t>(value);
}

/// Raw bytes (no length prefix)
inline void WriteBytes(std::ostream& out, const void* data, size_t size) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}

inline void ReadBytes(std::istream& in, void* data, size_t size) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!in) {
        detail::Fail("unexpected end of data");
    }
}

/// 64-bit length followed by the characters
inline void WriteString(std::ostream& out, const std::string& value) {
    WriteSize(out, value.size());
    WriteBytes(out, value.data(), value.size());
}

inline std::string ReadString(std::istream& in, uint64_t max_length = (uint64_t{1} << 30)) {
    std::string value(ReadSize(in, max_length), '\0');
    ReadBytes(in, value.data(), value.size());
    return value;
}

/// Array of arithmetic values without a length prefix
template <typename T>
void WriteArray(std::ostream& out, const T* values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        Write<T>(out, values[i]);
    }
}

template <typename T>
void ReadArray(std::istream& in, T* values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        values[i] = Read<T>(in);
    }
}

}  // namespace io
}  // namespace dpan
