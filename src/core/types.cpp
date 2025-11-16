// File: src/core/types.cpp
#include "core/types.hpp"
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <set>

namespace dpan {

// Static member initialization
std::atomic<PatternID::ValueType> PatternID::next_id_{1};

PatternID PatternID::Generate() {
    // Thread-safe atomic increment
    ValueType new_id = next_id_.fetch_add(1, std::memory_order_relaxed);
    return PatternID(new_id);
}

std::string PatternID::ToString() const {
    if (!IsValid()) {
        return "PatternID(INVALID)";
    }
    std::ostringstream oss;
    oss << "PatternID(" << std::hex << std::setw(16) << std::setfill('0') << value_ << ")";
    return oss.str();
}

void PatternID::Serialize(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&value_), sizeof(value_));
}

PatternID PatternID::Deserialize(std::istream& in) {
    ValueType value;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    return PatternID(value);
}

// Enum implementations

const char* ToString(PatternType type) {
    switch (type) {
        case PatternType::ATOMIC: return "ATOMIC";
        case PatternType::COMPOSITE: return "COMPOSITE";
        case PatternType::META: return "META";
        default: return "UNKNOWN";
    }
}

PatternType ParsePatternType(const std::string& str) {
    if (str == "ATOMIC") return PatternType::ATOMIC;
    if (str == "COMPOSITE") return PatternType::COMPOSITE;
    if (str == "META") return PatternType::META;
    throw std::invalid_argument("Unknown PatternType: " + str);
}

const char* ToString(AssociationType type) {
    switch (type) {
        case AssociationType::CAUSAL: return "CAUSAL";
        case AssociationType::CATEGORICAL: return "CATEGORICAL";
        case AssociationType::SPATIAL: return "SPATIAL";
        case AssociationType::FUNCTIONAL: return "FUNCTIONAL";
        case AssociationType::COMPOSITIONAL: return "COMPOSITIONAL";
        default: return "UNKNOWN";
    }
}

AssociationType ParseAssociationType(const std::string& str) {
    if (str == "CAUSAL") return AssociationType::CAUSAL;
    if (str == "CATEGORICAL") return AssociationType::CATEGORICAL;
    if (str == "SPATIAL") return AssociationType::SPATIAL;
    if (str == "FUNCTIONAL") return AssociationType::FUNCTIONAL;
    if (str == "COMPOSITIONAL") return AssociationType::COMPOSITIONAL;
    throw std::invalid_argument("Unknown AssociationType: " + str);
}

// Timestamp implementations

Timestamp Timestamp::Now() {
    return Timestamp(ClockType::now());
}

Timestamp Timestamp::FromMicros(int64_t micros) {
    TimePoint tp{Duration(micros)};
    return Timestamp(tp);
}

int64_t Timestamp::ToMicros() const {
    auto duration = time_point_.time_since_epoch();
    return std::chrono::duration_cast<Duration>(duration).count();
}

std::string Timestamp::ToString() const {
    auto micros = ToMicros();
    auto seconds = micros / 1000000;
    auto remaining_micros = micros % 1000000;

    std::ostringstream oss;
    oss << "Timestamp(" << seconds << "."
        << std::setw(6) << std::setfill('0') << remaining_micros << "s)";
    return oss.str();
}

void Timestamp::Serialize(std::ostream& out) const {
    int64_t micros = ToMicros();
    out.write(reinterpret_cast<const char*>(&micros), sizeof(micros));
}

Timestamp Timestamp::Deserialize(std::istream& in) {
    int64_t micros;
    in.read(reinterpret_cast<char*>(&micros), sizeof(micros));
    return FromMicros(micros);
}

// ContextVector implementations

void ContextVector::Set(const DimensionType& dimension, ValueType value) {
    if (value != 0.0f) {
        data_[dimension] = value;
    } else {
        data_.erase(dimension);
    }
}

ContextVector::ValueType ContextVector::Get(const DimensionType& dimension) const {
    auto it = data_.find(dimension);
    return (it != data_.end()) ? it->second : 0.0f;
}

bool ContextVector::Has(const DimensionType& dimension) const {
    return data_.find(dimension) != data_.end();
}

void ContextVector::Remove(const DimensionType& dimension) {
    data_.erase(dimension);
}

void ContextVector::Clear() {
    data_.clear();
}

std::vector<ContextVector::DimensionType> ContextVector::GetDimensions() const {
    std::vector<DimensionType> dimensions;
    dimensions.reserve(data_.size());
    for (const auto& pair : data_) {
        dimensions.push_back(pair.first);
    }
    return dimensions;
}

float ContextVector::CosineSimilarity(const ContextVector& other) const {
    float dot = DotProduct(other);
    float norm_product = Norm() * other.Norm();

    if (norm_product == 0.0f) {
        return 0.0f;
    }

    return dot / norm_product;
}

float ContextVector::EuclideanDistance(const ContextVector& other) const {
    float sum_sq_diff = 0.0f;

    // Get all unique dimensions
    std::set<DimensionType> all_dims;
    for (const auto& pair : data_) {
        all_dims.insert(pair.first);
    }
    for (const auto& pair : other.data_) {
        all_dims.insert(pair.first);
    }

    // Compute sum of squared differences
    for (const auto& dim : all_dims) {
        float diff = Get(dim) - other.Get(dim);
        sum_sq_diff += diff * diff;
    }

    return std::sqrt(sum_sq_diff);
}

float ContextVector::DotProduct(const ContextVector& other) const {
    float dot = 0.0f;

    // Iterate over the smaller vector for efficiency
    const ContextVector* smaller = (Size() <= other.Size()) ? this : &other;
    const ContextVector* larger = (Size() <= other.Size()) ? &other : this;

    for (const auto& pair : *smaller) {
        dot += pair.second * larger->Get(pair.first);
    }

    return dot;
}

float ContextVector::Norm() const {
    float sum_sq = 0.0f;
    for (const auto& pair : data_) {
        sum_sq += pair.second * pair.second;
    }
    return std::sqrt(sum_sq);
}

ContextVector ContextVector::Normalized() const {
    float norm = Norm();
    if (norm == 0.0f) {
        return ContextVector();
    }
    return (*this) * (1.0f / norm);
}

ContextVector ContextVector::operator+(const ContextVector& other) const {
    ContextVector result = *this;
    for (const auto& pair : other.data_) {
        result.Set(pair.first, result.Get(pair.first) + pair.second);
    }
    return result;
}

ContextVector ContextVector::operator*(float scalar) const {
    ContextVector result;
    for (const auto& pair : data_) {
        result.Set(pair.first, pair.second * scalar);
    }
    return result;
}

bool ContextVector::operator==(const ContextVector& other) const {
    if (Size() != other.Size()) {
        return false;
    }

    for (const auto& pair : data_) {
        if (std::abs(pair.second - other.Get(pair.first)) > 1e-6f) {
            return false;
        }
    }

    return true;
}

void ContextVector::Serialize(std::ostream& out) const {
    // Write size
    size_t size = data_.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));

    // Write each dimension-value pair
    for (const auto& pair : data_) {
        // Write dimension string length and data
        size_t dim_len = pair.first.length();
        out.write(reinterpret_cast<const char*>(&dim_len), sizeof(dim_len));
        out.write(pair.first.data(), dim_len);

        // Write value
        out.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }
}

ContextVector ContextVector::Deserialize(std::istream& in) {
    // Read size
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));

    ContextVector result;
    for (size_t i = 0; i < size; ++i) {
        // Read dimension
        size_t dim_len;
        in.read(reinterpret_cast<char*>(&dim_len), sizeof(dim_len));
        std::string dimension(dim_len, '\0');
        in.read(&dimension[0], dim_len);

        // Read value
        ValueType value;
        in.read(reinterpret_cast<char*>(&value), sizeof(value));

        result.Set(dimension, value);
    }

    return result;
}

std::string ContextVector::ToString() const {
    if (IsEmpty()) {
        return "ContextVector{}";
    }

    std::ostringstream oss;
    oss << "ContextVector{";

    bool first = true;
    for (const auto& pair : data_) {
        if (!first) oss << ", ";
        oss << pair.first << ":" << pair.second;
        first = false;
    }

    oss << "}";
    return oss.str();
}

} // namespace dpan
