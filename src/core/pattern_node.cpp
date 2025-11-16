// File: src/core/pattern_node.cpp
#include "core/pattern_node.hpp"
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace dpan {

// Constructor
PatternNode::PatternNode(PatternID id, const PatternData& data, PatternType type)
    : id_(id), data_(data), type_(type), creation_timestamp_(Timestamp::Now()) {
    last_accessed_.store(creation_timestamp_.ToMicros(), std::memory_order_relaxed);
}

// Move constructor
PatternNode::PatternNode(PatternNode&& other) noexcept
    : id_(other.id_),
      data_(std::move(other.data_)),
      type_(other.type_),
      creation_timestamp_(other.creation_timestamp_),
      sub_patterns_(std::move(other.sub_patterns_)) {
    // Move atomic values by loading and storing
    activation_threshold_.store(other.activation_threshold_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    base_activation_.store(other.base_activation_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    last_accessed_.store(other.last_accessed_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    access_count_.store(other.access_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    confidence_score_.store(other.confidence_score_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    // Note: mutex is not moved, a new one is default-constructed
}

// Getters
Timestamp PatternNode::GetLastAccessed() const {
    uint64_t micros = last_accessed_.load(std::memory_order_relaxed);
    return Timestamp::FromMicros(static_cast<int64_t>(micros));
}

// Setters
void PatternNode::SetActivationThreshold(float threshold) {
    activation_threshold_.store(threshold, std::memory_order_relaxed);
}

void PatternNode::SetBaseActivation(float activation) {
    base_activation_.store(activation, std::memory_order_relaxed);
}

void PatternNode::SetConfidenceScore(float score) {
    // Clamp score to [0.0, 1.0]
    float clamped_score = std::max(0.0f, std::min(1.0f, score));
    confidence_score_.store(clamped_score, std::memory_order_relaxed);
}

// Update statistics
void PatternNode::RecordAccess() {
    Timestamp now = Timestamp::Now();
    last_accessed_.store(static_cast<uint64_t>(now.ToMicros()), std::memory_order_relaxed);
    access_count_.fetch_add(1, std::memory_order_relaxed);
}

void PatternNode::IncrementAccessCount(uint32_t count) {
    access_count_.fetch_add(count, std::memory_order_relaxed);
}

void PatternNode::UpdateConfidence(float delta) {
    float old_confidence = confidence_score_.load(std::memory_order_relaxed);
    float new_confidence = std::max(0.0f, std::min(1.0f, old_confidence + delta));
    confidence_score_.store(new_confidence, std::memory_order_relaxed);
}

// Sub-patterns
std::vector<PatternID> PatternNode::GetSubPatterns() const {
    std::lock_guard<std::mutex> lock(sub_patterns_mutex_);
    return sub_patterns_;
}

void PatternNode::AddSubPattern(PatternID sub_pattern_id) {
    std::lock_guard<std::mutex> lock(sub_patterns_mutex_);

    // Check if already exists
    auto it = std::find(sub_patterns_.begin(), sub_patterns_.end(), sub_pattern_id);
    if (it == sub_patterns_.end()) {
        sub_patterns_.push_back(sub_pattern_id);
    }
}

void PatternNode::RemoveSubPattern(PatternID sub_pattern_id) {
    std::lock_guard<std::mutex> lock(sub_patterns_mutex_);

    auto it = std::find(sub_patterns_.begin(), sub_patterns_.end(), sub_pattern_id);
    if (it != sub_patterns_.end()) {
        sub_patterns_.erase(it);
    }
}

bool PatternNode::HasSubPatterns() const {
    std::lock_guard<std::mutex> lock(sub_patterns_mutex_);
    return !sub_patterns_.empty();
}

// Activation computation
float PatternNode::ComputeActivation(const FeatureVector& input_features) const {
    if (data_.IsEmpty()) {
        return base_activation_.load(std::memory_order_relaxed);
    }

    // Get pattern's features
    FeatureVector pattern_features = data_.GetFeatures();

    // Compute similarity (cosine similarity)
    float similarity = 0.0f;
    try {
        similarity = pattern_features.CosineSimilarity(input_features);
    } catch (const std::invalid_argument&) {
        // Dimension mismatch - return base activation
        return base_activation_.load(std::memory_order_relaxed);
    }

    // Activation is combination of similarity and base activation
    float base = base_activation_.load(std::memory_order_relaxed);
    return (similarity + base) / 2.0f;
}

bool PatternNode::IsActivated(const FeatureVector& input_features) const {
    float activation = ComputeActivation(input_features);
    float threshold = activation_threshold_.load(std::memory_order_relaxed);
    return activation >= threshold;
}

// Serialization
void PatternNode::Serialize(std::ostream& out) const {
    // Serialize PatternID
    id_.Serialize(out);

    // Serialize PatternData
    data_.Serialize(out);

    // Serialize PatternType
    uint8_t type_byte = static_cast<uint8_t>(type_);
    out.write(reinterpret_cast<const char*>(&type_byte), sizeof(type_byte));

    // Serialize activation parameters
    float threshold = activation_threshold_.load(std::memory_order_relaxed);
    float base_activation = base_activation_.load(std::memory_order_relaxed);
    out.write(reinterpret_cast<const char*>(&threshold), sizeof(threshold));
    out.write(reinterpret_cast<const char*>(&base_activation), sizeof(base_activation));

    // Serialize statistics
    int64_t creation_micros = creation_timestamp_.ToMicros();
    uint64_t last_accessed = last_accessed_.load(std::memory_order_relaxed);
    uint32_t access_count = access_count_.load(std::memory_order_relaxed);
    float confidence = confidence_score_.load(std::memory_order_relaxed);

    out.write(reinterpret_cast<const char*>(&creation_micros), sizeof(creation_micros));
    out.write(reinterpret_cast<const char*>(&last_accessed), sizeof(last_accessed));
    out.write(reinterpret_cast<const char*>(&access_count), sizeof(access_count));
    out.write(reinterpret_cast<const char*>(&confidence), sizeof(confidence));

    // Serialize sub-patterns
    {
        std::lock_guard<std::mutex> lock(sub_patterns_mutex_);
        size_t sub_count = sub_patterns_.size();
        out.write(reinterpret_cast<const char*>(&sub_count), sizeof(sub_count));

        for (const auto& sub_id : sub_patterns_) {
            sub_id.Serialize(out);
        }
    }
}

PatternNode PatternNode::Deserialize(std::istream& in) {
    // Deserialize PatternID
    PatternID id = PatternID::Deserialize(in);

    // Deserialize PatternData
    PatternData data = PatternData::Deserialize(in);

    // Deserialize PatternType
    uint8_t type_byte;
    in.read(reinterpret_cast<char*>(&type_byte), sizeof(type_byte));
    PatternType type = static_cast<PatternType>(type_byte);

    // Create node
    PatternNode node(id, data, type);

    // Deserialize activation parameters
    float threshold, base_activation;
    in.read(reinterpret_cast<char*>(&threshold), sizeof(threshold));
    in.read(reinterpret_cast<char*>(&base_activation), sizeof(base_activation));
    node.activation_threshold_.store(threshold, std::memory_order_relaxed);
    node.base_activation_.store(base_activation, std::memory_order_relaxed);

    // Deserialize statistics
    int64_t creation_micros;
    uint64_t last_accessed;
    uint32_t access_count;
    float confidence;

    in.read(reinterpret_cast<char*>(&creation_micros), sizeof(creation_micros));
    in.read(reinterpret_cast<char*>(&last_accessed), sizeof(last_accessed));
    in.read(reinterpret_cast<char*>(&access_count), sizeof(access_count));
    in.read(reinterpret_cast<char*>(&confidence), sizeof(confidence));

    node.creation_timestamp_ = Timestamp::FromMicros(creation_micros);
    node.last_accessed_.store(last_accessed, std::memory_order_relaxed);
    node.access_count_.store(access_count, std::memory_order_relaxed);
    node.confidence_score_.store(confidence, std::memory_order_relaxed);

    // Deserialize sub-patterns
    size_t sub_count;
    in.read(reinterpret_cast<char*>(&sub_count), sizeof(sub_count));

    for (size_t i = 0; i < sub_count; ++i) {
        PatternID sub_id = PatternID::Deserialize(in);
        node.sub_patterns_.push_back(sub_id);
    }

    return node;
}

// String representation
std::string PatternNode::ToString() const {
    std::ostringstream oss;
    oss << "PatternNode{"
        << "id=" << id_.ToString()
        << ", type=" << dpan::ToString(type_)
        << ", threshold=" << std::fixed << std::setprecision(2) << activation_threshold_.load(std::memory_order_relaxed)
        << ", base_activation=" << base_activation_.load(std::memory_order_relaxed)
        << ", confidence=" << confidence_score_.load(std::memory_order_relaxed)
        << ", access_count=" << access_count_.load(std::memory_order_relaxed)
        << ", sub_patterns=" << sub_patterns_.size()
        << "}";
    return oss.str();
}

// Memory footprint estimation
size_t PatternNode::EstimateMemoryUsage() const {
    size_t total = sizeof(PatternNode);

    // Add data size
    total += data_.GetCompressedSize();

    // Add sub-patterns vector capacity
    std::lock_guard<std::mutex> lock(sub_patterns_mutex_);
    total += sub_patterns_.capacity() * sizeof(PatternID);

    return total;
}

} // namespace dpan
