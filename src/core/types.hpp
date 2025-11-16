// File: src/core/types.hpp
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <atomic>
#include <chrono>
#include <vector>
#include <map>

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

// Timestamp: Microsecond-precision time point
class Timestamp {
public:
    using ClockType = std::chrono::steady_clock;
    using TimePoint = ClockType::time_point;
    using Duration = std::chrono::microseconds;

    // Create timestamp for current time
    static Timestamp Now();

    // Create timestamp from microseconds since epoch
    static Timestamp FromMicros(int64_t micros);

    // Default constructor creates zero timestamp
    Timestamp() : time_point_(TimePoint{}) {}

    // Get microseconds since epoch
    int64_t ToMicros() const;

    // Get duration since another timestamp
    Duration operator-(const Timestamp& other) const {
        return std::chrono::duration_cast<Duration>(time_point_ - other.time_point_);
    }

    // Comparison operators
    bool operator<(const Timestamp& other) const { return time_point_ < other.time_point_; }
    bool operator>(const Timestamp& other) const { return time_point_ > other.time_point_; }
    bool operator<=(const Timestamp& other) const { return time_point_ <= other.time_point_; }
    bool operator>=(const Timestamp& other) const { return time_point_ >= other.time_point_; }
    bool operator==(const Timestamp& other) const { return time_point_ == other.time_point_; }
    bool operator!=(const Timestamp& other) const { return time_point_ != other.time_point_; }

    // String conversion
    std::string ToString() const;

    // Serialization
    void Serialize(std::ostream& out) const;
    static Timestamp Deserialize(std::istream& in);

private:
    explicit Timestamp(TimePoint tp) : time_point_(tp) {}
    TimePoint time_point_;
};

// ContextVector: Sparse representation of contextual information
// Used to describe the conditions under which patterns/associations are relevant
class ContextVector {
public:
    using DimensionType = std::string;
    using ValueType = float;
    using StorageType = std::map<DimensionType, ValueType>;

    // Constructors
    ContextVector() = default;
    explicit ContextVector(const StorageType& data) : data_(data) {}

    // Set a dimension value
    void Set(const DimensionType& dimension, ValueType value);

    // Get a dimension value (returns 0.0 if not present)
    ValueType Get(const DimensionType& dimension) const;

    // Check if dimension exists
    bool Has(const DimensionType& dimension) const;

    // Remove a dimension
    void Remove(const DimensionType& dimension);

    // Clear all dimensions
    void Clear();

    // Get number of dimensions
    size_t Size() const { return data_.size(); }

    // Check if empty
    bool IsEmpty() const { return data_.empty(); }

    // Get all dimensions
    std::vector<DimensionType> GetDimensions() const;

    // Compute cosine similarity with another context vector
    float CosineSimilarity(const ContextVector& other) const;

    // Compute Euclidean distance
    float EuclideanDistance(const ContextVector& other) const;

    // Compute dot product
    float DotProduct(const ContextVector& other) const;

    // Get L2 norm (magnitude)
    float Norm() const;

    // Normalize to unit length
    ContextVector Normalized() const;

    // Vector addition
    ContextVector operator+(const ContextVector& other) const;

    // Scalar multiplication
    ContextVector operator*(float scalar) const;

    // Equality comparison
    bool operator==(const ContextVector& other) const;

    // Serialization
    void Serialize(std::ostream& out) const;
    static ContextVector Deserialize(std::istream& in);

    // String representation
    std::string ToString() const;

    // Iterator support
    using const_iterator = StorageType::const_iterator;
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

private:
    StorageType data_;
};

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
