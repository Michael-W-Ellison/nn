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

} // namespace dpan
