// File: src/core/pattern_data.cpp
#include "core/pattern_data.hpp"
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace dpan {

// DataModality string conversion
const char* ToString(DataModality modality) {
    switch (modality) {
        case DataModality::UNKNOWN: return "UNKNOWN";
        case DataModality::NUMERIC: return "NUMERIC";
        case DataModality::IMAGE: return "IMAGE";
        case DataModality::AUDIO: return "AUDIO";
        case DataModality::TEXT: return "TEXT";
        case DataModality::COMPOSITE: return "COMPOSITE";
        default: return "INVALID";
    }
}

// ============================================================================
// FeatureVector Implementation
// ============================================================================

FeatureVector::FeatureVector(size_t dimension) : data_(dimension, 0.0f) {}

FeatureVector::FeatureVector(const StorageType& data) : data_(data) {}

FeatureVector::FeatureVector(StorageType&& data) : data_(std::move(data)) {}

float FeatureVector::Norm() const {
    float sum_sq = 0.0f;
    for (float val : data_) {
        sum_sq += val * val;
    }
    return std::sqrt(sum_sq);
}

FeatureVector FeatureVector::Normalized() const {
    float norm = Norm();
    if (norm == 0.0f) {
        return FeatureVector(data_.size());
    }

    FeatureVector result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] / norm;
    }
    return result;
}

float FeatureVector::DotProduct(const FeatureVector& other) const {
    if (Dimension() != other.Dimension()) {
        throw std::invalid_argument("FeatureVector dimensions must match for dot product");
    }

    float dot = 0.0f;
    for (size_t i = 0; i < data_.size(); ++i) {
        dot += data_[i] * other.data_[i];
    }
    return dot;
}

float FeatureVector::EuclideanDistance(const FeatureVector& other) const {
    if (Dimension() != other.Dimension()) {
        throw std::invalid_argument("FeatureVector dimensions must match for distance");
    }

    float sum_sq_diff = 0.0f;
    for (size_t i = 0; i < data_.size(); ++i) {
        float diff = data_[i] - other.data_[i];
        sum_sq_diff += diff * diff;
    }
    return std::sqrt(sum_sq_diff);
}

float FeatureVector::CosineSimilarity(const FeatureVector& other) const {
    if (Dimension() != other.Dimension()) {
        throw std::invalid_argument("FeatureVector dimensions must match for cosine similarity");
    }

    float dot = DotProduct(other);
    float norm_product = Norm() * other.Norm();

    if (norm_product == 0.0f) {
        return 0.0f;
    }

    return dot / norm_product;
}

FeatureVector FeatureVector::operator+(const FeatureVector& other) const {
    if (Dimension() != other.Dimension()) {
        throw std::invalid_argument("FeatureVector dimensions must match for addition");
    }

    FeatureVector result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] + other.data_[i];
    }
    return result;
}

FeatureVector FeatureVector::operator-(const FeatureVector& other) const {
    if (Dimension() != other.Dimension()) {
        throw std::invalid_argument("FeatureVector dimensions must match for subtraction");
    }

    FeatureVector result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] - other.data_[i];
    }
    return result;
}

FeatureVector FeatureVector::operator*(float scalar) const {
    FeatureVector result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result[i] = data_[i] * scalar;
    }
    return result;
}

bool FeatureVector::operator==(const FeatureVector& other) const {
    if (Dimension() != other.Dimension()) {
        return false;
    }

    for (size_t i = 0; i < data_.size(); ++i) {
        if (std::abs(data_[i] - other.data_[i]) > 1e-6f) {
            return false;
        }
    }
    return true;
}

void FeatureVector::Serialize(std::ostream& out) const {
    size_t dim = data_.size();
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(data_.data()), dim * sizeof(ValueType));
}

FeatureVector FeatureVector::Deserialize(std::istream& in) {
    size_t dim;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));

    FeatureVector result(dim);
    in.read(reinterpret_cast<char*>(result.data_.data()), dim * sizeof(ValueType));
    return result;
}

std::string FeatureVector::ToString(size_t max_elements) const {
    if (data_.empty()) {
        return "FeatureVector[]";
    }

    std::ostringstream oss;
    oss << "FeatureVector[" << data_.size() << "](";

    size_t count = std::min(max_elements, data_.size());
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) oss << ", ";
        oss << std::fixed << std::setprecision(4) << data_[i];
    }

    if (data_.size() > max_elements) {
        oss << ", ...";
    }
    oss << ")";

    return oss.str();
}

// ============================================================================
// PatternData Implementation
// ============================================================================

PatternData::PatternData(DataModality modality) : modality_(modality) {}

PatternData PatternData::FromBytes(const std::vector<uint8_t>& data, DataModality modality) {
    if (data.size() > kMaxRawDataSize) {
        throw std::invalid_argument("Data size exceeds maximum allowed size");
    }

    PatternData pattern(modality);
    pattern.original_size_ = data.size();
    pattern.compressed_data_ = Compress(data);
    return pattern;
}

PatternData PatternData::FromFeatures(const FeatureVector& features, DataModality modality) {
    PatternData pattern(modality);

    // Convert feature vector to bytes
    const auto& feature_data = features.Data();
    size_t byte_size = feature_data.size() * sizeof(float);
    std::vector<uint8_t> raw_data(byte_size);
    std::memcpy(raw_data.data(), feature_data.data(), byte_size);

    pattern.original_size_ = raw_data.size();
    pattern.compressed_data_ = Compress(raw_data);
    return pattern;
}

FeatureVector PatternData::GetFeatures() const {
    if (IsEmpty()) {
        return FeatureVector();
    }

    // Decompress data
    std::vector<uint8_t> raw_data = Decompress(compressed_data_, original_size_);

    // Convert bytes to feature vector
    size_t num_features = raw_data.size() / sizeof(float);
    FeatureVector features(num_features);
    std::memcpy(features.Data().data(), raw_data.data(), raw_data.size());

    return features;
}

std::vector<uint8_t> PatternData::GetRawData() const {
    if (IsEmpty()) {
        return std::vector<uint8_t>();
    }

    return Decompress(compressed_data_, original_size_);
}

void PatternData::Serialize(std::ostream& out) const {
    // Write modality
    uint8_t modality_byte = static_cast<uint8_t>(modality_);
    out.write(reinterpret_cast<const char*>(&modality_byte), sizeof(modality_byte));

    // Write original size
    out.write(reinterpret_cast<const char*>(&original_size_), sizeof(original_size_));

    // Write compressed data size
    size_t compressed_size = compressed_data_.size();
    out.write(reinterpret_cast<const char*>(&compressed_size), sizeof(compressed_size));

    // Write compressed data
    if (compressed_size > 0) {
        out.write(reinterpret_cast<const char*>(compressed_data_.data()), compressed_size);
    }
}

PatternData PatternData::Deserialize(std::istream& in) {
    // Read modality
    uint8_t modality_byte;
    in.read(reinterpret_cast<char*>(&modality_byte), sizeof(modality_byte));

    PatternData pattern(static_cast<DataModality>(modality_byte));

    // Read original size
    in.read(reinterpret_cast<char*>(&pattern.original_size_), sizeof(pattern.original_size_));

    // Read compressed data size
    size_t compressed_size;
    in.read(reinterpret_cast<char*>(&compressed_size), sizeof(compressed_size));

    // Read compressed data
    if (compressed_size > 0) {
        pattern.compressed_data_.resize(compressed_size);
        in.read(reinterpret_cast<char*>(pattern.compressed_data_.data()), compressed_size);
    }

    return pattern;
}

std::string PatternData::ToString() const {
    std::ostringstream oss;
    oss << "PatternData{modality=" << dpan::ToString(modality_)
        << ", original_size=" << original_size_
        << ", compressed_size=" << compressed_data_.size()
        << ", ratio=" << std::fixed << std::setprecision(2) << GetCompressionRatio()
        << "}";
    return oss.str();
}

bool PatternData::operator==(const PatternData& other) const {
    return modality_ == other.modality_ &&
           original_size_ == other.original_size_ &&
           compressed_data_ == other.compressed_data_;
}

// Simple RLE (Run-Length Encoding) compression
std::vector<uint8_t> PatternData::Compress(const std::vector<uint8_t>& data) {
    if (data.empty()) {
        return std::vector<uint8_t>();
    }

    std::vector<uint8_t> compressed;
    compressed.reserve(data.size()); // Reserve at least original size

    size_t i = 0;
    while (i < data.size()) {
        uint8_t value = data[i];
        uint8_t count = 1;

        // Count consecutive identical bytes (max 255)
        while (i + count < data.size() && data[i + count] == value && count < 255) {
            count++;
        }

        // Store count and value
        compressed.push_back(count);
        compressed.push_back(value);

        i += count;
    }

    return compressed;
}

std::vector<uint8_t> PatternData::Decompress(const std::vector<uint8_t>& data, size_t original_size) {
    if (data.empty()) {
        return std::vector<uint8_t>();
    }

    std::vector<uint8_t> decompressed;
    decompressed.reserve(original_size);

    for (size_t i = 0; i + 1 < data.size(); i += 2) {
        uint8_t count = data[i];
        uint8_t value = data[i + 1];

        for (uint8_t j = 0; j < count; ++j) {
            decompressed.push_back(value);
        }
    }

    return decompressed;
}

} // namespace dpan
