// File: src/core/types.hpp
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <atomic>

namespace dpan {

// PatternID: Unique identifier for patterns
// Uses 64-bit integer for efficiency and range
class PatternID {
public:
    // Type alias for underlying storage
    using ValueType = uint64_t;

    // Default constructor creates invalid ID
    PatternID() : value_(kInvalidID) {}

    // Explicit constructor from value
    explicit PatternID(ValueType value) : value_(value) {}

    // Generate new unique ID (thread-safe)
    static PatternID Generate();

    // Check if ID is valid
    bool IsValid() const { return value_ != kInvalidID; }

    // Get underlying value
    ValueType value() const { return value_; }

    // Comparison operators
    bool operator==(const PatternID& other) const { return value_ == other.value_; }
    bool operator!=(const PatternID& other) const { return value_ != other.value_; }
    bool operator<(const PatternID& other) const { return value_ < other.value_; }
    bool operator>(const PatternID& other) const { return value_ > other.value_; }
    bool operator<=(const PatternID& other) const { return value_ <= other.value_; }
    bool operator>=(const PatternID& other) const { return value_ >= other.value_; }

    // String conversion for debugging
    std::string ToString() const;

    // Serialization
    void Serialize(std::ostream& out) const;
    static PatternID Deserialize(std::istream& in);

    // Hash support for std::unordered_map
    struct Hash {
        size_t operator()(const PatternID& id) const {
            return std::hash<ValueType>()(id.value_);
        }
    };

private:
    static constexpr ValueType kInvalidID = 0;
    static std::atomic<ValueType> next_id_;

    ValueType value_;
};

// PatternType: Classification of pattern complexity
enum class PatternType : uint8_t {
    ATOMIC = 0,      // Indivisible, basic pattern
    COMPOSITE = 1,   // Composed of multiple sub-patterns
    META = 2,        // Pattern of patterns (highest abstraction)
};

// Convert PatternType to string
const char* ToString(PatternType type);

// Parse PatternType from string
PatternType ParsePatternType(const std::string& str);

// AssociationType: Type of relationship between patterns
enum class AssociationType : uint8_t {
    CAUSAL = 0,         // A typically precedes B
    CATEGORICAL = 1,    // A and B belong to same category
    SPATIAL = 2,        // A and B appear in similar spatial config
    FUNCTIONAL = 3,     // A and B serve similar purposes
    COMPOSITIONAL = 4,  // A contains B or vice versa
};

// Convert AssociationType to string
const char* ToString(AssociationType type);

// Parse AssociationType from string
AssociationType ParseAssociationType(const std::string& str);

} // namespace dpan

// Hash specialization for std::unordered_map
namespace std {
    template<>
    struct hash<dpan::PatternID> {
        size_t operator()(const dpan::PatternID& id) const {
            return dpan::PatternID::Hash()(id);
        }
    };
}
