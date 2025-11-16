// File: src/core/pattern_node.hpp
#pragma once

#include "core/types.hpp"
#include "core/pattern_data.hpp"
#include <mutex>
#include <atomic>
#include <vector>

namespace dpan {

// PatternNode: Complete pattern representation with statistics and metadata
class PatternNode {
public:
    // Constructors
    PatternNode() = default;
    explicit PatternNode(PatternID id, const PatternData& data, PatternType type);

    // Move constructor (needed for deserialization)
    PatternNode(PatternNode&& other) noexcept;

    // Delete copy constructor and assignment (due to atomics and mutex)
    PatternNode(const PatternNode&) = delete;
    PatternNode& operator=(const PatternNode&) = delete;
    PatternNode& operator=(PatternNode&&) = delete;

    // Getters (thread-safe for read-only access)
    PatternID GetID() const { return id_; }
    const PatternData& GetData() const { return data_; }
    PatternType GetType() const { return type_; }
    float GetActivationThreshold() const { return activation_threshold_.load(std::memory_order_relaxed); }
    float GetBaseActivation() const { return base_activation_.load(std::memory_order_relaxed); }
    Timestamp GetCreationTime() const { return creation_timestamp_; }
    Timestamp GetLastAccessed() const;
    uint32_t GetAccessCount() const { return access_count_.load(std::memory_order_relaxed); }
    float GetConfidenceScore() const { return confidence_score_.load(std::memory_order_relaxed); }

    // Setters (thread-safe)
    void SetActivationThreshold(float threshold);
    void SetBaseActivation(float activation);
    void SetConfidenceScore(float score);

    // Update statistics (thread-safe)
    void RecordAccess();
    void IncrementAccessCount(uint32_t count = 1);
    void UpdateConfidence(float delta);

    // Sub-patterns (thread-safe)
    std::vector<PatternID> GetSubPatterns() const;
    void AddSubPattern(PatternID sub_pattern_id);
    void RemoveSubPattern(PatternID sub_pattern_id);
    bool HasSubPatterns() const;

    // Activation computation
    float ComputeActivation(const FeatureVector& input_features) const;
    bool IsActivated(const FeatureVector& input_features) const;

    // Age calculation
    Timestamp::Duration GetAge() const {
        return Timestamp::Now() - creation_timestamp_;
    }

    // Serialization
    void Serialize(std::ostream& out) const;
    static PatternNode Deserialize(std::istream& in);

    // String representation
    std::string ToString() const;

    // Memory footprint estimation
    size_t EstimateMemoryUsage() const;

private:
    // Core identity and data
    PatternID id_;
    PatternData data_;
    PatternType type_{PatternType::ATOMIC};

    // Activation parameters (atomic for thread-safety)
    std::atomic<float> activation_threshold_{0.5f};
    std::atomic<float> base_activation_{0.0f};

    // Statistics (atomic for thread-safety)
    Timestamp creation_timestamp_;
    mutable std::atomic<uint64_t> last_accessed_{0};  // Stored as micros
    std::atomic<uint32_t> access_count_{0};
    std::atomic<float> confidence_score_{0.5f};

    // Hierarchical structure
    mutable std::mutex sub_patterns_mutex_;
    std::vector<PatternID> sub_patterns_;
};

} // namespace dpan
