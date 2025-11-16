// File: src/core/types.cpp
#include "core/types.hpp"
#include <sstream>
#include <iomanip>
#include <stdexcept>

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

} // namespace dpan
