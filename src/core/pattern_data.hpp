// File: src/core/pattern_data.hpp
#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstdint>

namespace dpan {

// Forward declarations
class FeatureVector;

// DataModality: Type of data this pattern represents
enum class DataModality : uint8_t {
    UNKNOWN = 0,
    NUMERIC = 1,      // Numerical vector data
    IMAGE = 2,        // Image/visual data
    AUDIO = 3,        // Audio/sound data
    TEXT = 4,         // Text/language data
    COMPOSITE = 5,    // Mix of multiple modalities
};

const char* ToString(DataModality modality);

// PatternData: Stores abstracted pattern representation
class PatternData {
public:
    // Maximum raw data size (10MB)
    static constexpr size_t kMaxRawDataSize = 10 * 1024 * 1024;

    // Constructors
    PatternData() = default;
    explicit PatternData(DataModality modality);

    // Create from raw bytes
    static PatternData FromBytes(const std::vector<uint8_t>& data, DataModality modality);

    // Create from feature vector
    static PatternData FromFeatures(const FeatureVector& features, DataModality modality);

    // Get modality
    DataModality GetModality() const { return modality_; }

    // Get feature vector representation
    FeatureVector GetFeatures() const;

    // Get raw data (may decompress)
    std::vector<uint8_t> GetRawData() const;

    // Get compressed size
    size_t GetCompressedSize() const { return compressed_data_.size(); }

    // Get original size (before compression)
    size_t GetOriginalSize() const { return original_size_; }

    // Compression ratio
    float GetCompressionRatio() const {
        return original_size_ > 0
            ? static_cast<float>(compressed_data_.size()) / original_size_
            : 0.0f;
    }

    // Check if empty
    bool IsEmpty() const { return compressed_data_.empty(); }

    // Serialization
    void Serialize(std::ostream& out) const;
    static PatternData Deserialize(std::istream& in);

    // String representation (for debugging)
    std::string ToString() const;

    // Equality comparison
    bool operator==(const PatternData& other) const;
    bool operator!=(const PatternData& other) const { return !(*this == other); }

private:
    DataModality modality_{DataModality::UNKNOWN};
    std::vector<uint8_t> compressed_data_;
    size_t original_size_{0};

    // Compression/decompression helpers
    static std::vector<uint8_t> Compress(const std::vector<uint8_t>& data);
    static std::vector<uint8_t> Decompress(const std::vector<uint8_t>& data, size_t original_size);
};

// FeatureVector: Standard numerical representation for any pattern
class FeatureVector {
public:
    using ValueType = float;
    using StorageType = std::vector<ValueType>;

    // Constructors
    FeatureVector() = default;
    explicit FeatureVector(size_t dimension);
    explicit FeatureVector(const StorageType& data);
    explicit FeatureVector(StorageType&& data);

    // Get dimension
    size_t Dimension() const { return data_.size(); }

    // Element access
    ValueType operator[](size_t index) const { return data_[index]; }
    ValueType& operator[](size_t index) { return data_[index]; }

    // Get raw data
    const StorageType& Data() const { return data_; }
    StorageType& Data() { return data_; }

    // Compute L2 norm
    float Norm() const;

    // Normalize to unit length
    FeatureVector Normalized() const;

    // Dot product
    float DotProduct(const FeatureVector& other) const;

    // Euclidean distance
    float EuclideanDistance(const FeatureVector& other) const;

    // Cosine similarity
    float CosineSimilarity(const FeatureVector& other) const;

    // Vector operations
    FeatureVector operator+(const FeatureVector& other) const;
    FeatureVector operator-(const FeatureVector& other) const;
    FeatureVector operator*(float scalar) const;

    // Equality comparison
    bool operator==(const FeatureVector& other) const;
    bool operator!=(const FeatureVector& other) const { return !(*this == other); }

    // Serialization
    void Serialize(std::ostream& out) const;
    static FeatureVector Deserialize(std::istream& in);

    // String representation
    std::string ToString(size_t max_elements = 10) const;

private:
    StorageType data_;
};

} // namespace dpan
