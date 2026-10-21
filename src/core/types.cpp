// File: src/core/types.cpp
#include "core/types.hpp"
#include "core/binary_io.hpp"
#include "core/clock.hpp"
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
    io::Write<uint64_t>(out, value_);
}

PatternID PatternID::Deserialize(std::istream& in) {
    ValueType value = kInvalidID;
    try {
        value = io::Read<uint64_t>(in);
    } catch (const std::runtime_error&) {
        return PatternID();
    }
    ReserveThrough(value);
    return PatternID(value);
}

void PatternID::ReserveThrough(ValueType id) {
    if (id == kInvalidID) {
        return;
    }
    ValueType next = next_id_.load(std::memory_order_relaxed);
    while (next <= id &&
           !next_id_.compare_exchange_weak(next, id + 1, std::memory_order_relaxed)) {
    }
}

std::optional<PatternID> PatternID::FromString(const std::string& text) {
    std::string hex = text;
    const std::string prefix = "PatternID(";
    if (hex.compare(0, prefix.size(), prefix) == 0) {
        if (hex.size() <= prefix.size() + 1 || hex.back() != ')') {
            return std::nullopt;
        }
        hex = hex.substr(prefix.size(), hex.size() - prefix.size() - 1);
    }
    if (hex.empty() || hex.size() > 16) {
        return std::nullopt;
    }

    ValueType value = 0;
    for (char c : hex) {
        int digit;
        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            digit = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            digit = c - 'A' + 10;
        } else {
            return std::nullopt;
        }
        value = (value << 4) | static_cast<ValueType>(digit);
    }

    if (value == kInvalidID) {
        return std::nullopt;
    }
    ReserveThrough(value);
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
    return CurrentClock().Now();
}

Timestamp Timestamp::FromPersistedMicros(int64_t micros) {
    constexpr int64_t kYear2000Micros = 946684800LL * 1000000LL;
    if (micros > 0 && micros < kYear2000Micros) {
        return Now();
    }
    return FromMicros(micros);
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
    io::Write<int64_t>(out, ToMicros());
}

Timestamp Timestamp::Deserialize(std::istream& in) {
    return FromPersistedMicros(io::Read<int64_t>(in));
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
    io::WriteSize(out, data_.size());

    // Each dimension: name (length-prefixed), then value
    for (const auto& pair : data_) {
        io::WriteString(out, pair.first);
        io::Write<float>(out, pair.second);
    }
}

ContextVector ContextVector::Deserialize(std::istream& in) {
    size_t size = io::ReadSize(in);

    ContextVector result;
    for (size_t i = 0; i < size; ++i) {
        std::string dimension = io::ReadString(in);
        result.Set(dimension, io::Read<float>(in));
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
