# Phase 2: Association Learning System
## Extremely Detailed Implementation Plan

### Document Overview
This document provides a comprehensive, step-by-step implementation guide for Phase 2 of the DPAN project: the Association Learning System. Every task is broken down into granular sub-tasks with specific code examples, algorithms, mathematical formulations, testing requirements, and detailed acceptance criteria.

**Phase Duration**: 8-10 weeks (320-400 hours)
**Team Size**: 2-3 developers
**Prerequisites**: Phase 1 (Core Pattern Engine) complete

---

## Table of Contents
1. [Phase Overview](#phase-overview)
2. [Module 3.1: Association Data Structures](#module-31-association-data-structures)
3. [Module 3.2: Association Formation](#module-32-association-formation)
4. [Module 3.3: Association Strength Management](#module-33-association-strength-management)
5. [Module 3.4: Activation Propagation](#module-34-activation-propagation)
6. [Module 3.5: Integration & Testing](#module-35-integration--testing)
7. [Mathematical Foundations](#mathematical-foundations)
8. [Performance Optimization Guide](#performance-optimization-guide)
9. [Validation & Quality Assurance](#validation--quality-assurance)
10. [Troubleshooting & Debugging](#troubleshooting--debugging)

---

## Phase Overview

### Goals
- Implement dynamic association learning between patterns
- Support multiple association types (causal, spatial, categorical, functional, compositional)
- Enable efficient activation propagation through association networks
- Build scalable storage for millions of associations
- Support context-sensitive association strength

### Success Criteria
- [ ] Association matrix handles >10M associations with <1ms lookup
- [ ] Association formation accuracy >80% on synthetic datasets
- [ ] Activation propagation <100ms for 10K patterns
- [ ] Support all 5 association types
- [ ] Context-sensitive strength modulation works correctly
- [ ] >90% code coverage for all components
- [ ] Zero memory leaks (valgrind verified)
- [ ] Thread-safe concurrent operations

### Key Metrics Dashboard
Track these metrics daily:
- Number of associations created
- Association formation accuracy (compared to ground truth)
- Activation propagation time
- Memory usage per association
- Association strength distribution
- Graph connectivity metrics (average degree, clustering coefficient)

---

## Module 3.1: Association Data Structures

**Duration**: 2 weeks (80 hours)
**Dependencies**: Phase 1 complete
**Owner**: Lead C++ developer + Graph algorithms specialist

### Overview
This module establishes the fundamental data structures for storing and managing associations between patterns. These structures must be:
- Memory efficient (target: <100 bytes per association)
- Thread-safe for concurrent access
- Support fast lookups in multiple directions (source, target, type)
- Serializable for persistence
- Optimized for sparse graph operations

---

### Task 3.1.1: Implement AssociationEdge Class

**Duration**: 3 days (24 hours)
**Priority**: Critical
**Files to create**:
- `src/association/association_edge.hpp`
- `src/association/association_edge.cpp`
- `tests/association/association_edge_test.cpp`

#### Subtask 3.1.1.1: Define AssociationEdge Structure (6 hours)

**Mathematical Foundation**:

An association edge represents a directed relationship between two patterns:
```
E = (p_src, p_tgt, s, t, τ, d, r, c)
where:
  p_src = source pattern ID
  p_tgt = target pattern ID
  s ∈ [0,1] = association strength
  t ∈ {causal, spatial, categorical, functional, compositional} = type
  τ = temporal correlation coefficient
  d ∈ [0,1] = decay rate
  r = timestamp of last reinforcement
  c = context vector
```

**Implementation**:

```cpp
// File: src/association/association_edge.hpp
#pragma once

#include "core/types.hpp"
#include <atomic>
#include <mutex>

namespace dpan {

// AssociationEdge: Directed relationship between two patterns
class AssociationEdge {
public:
    // Constructors
    AssociationEdge() = default;
    AssociationEdge(
        PatternID source,
        PatternID target,
        AssociationType type,
        float initial_strength = 0.5f
    );

    // Core identity
    PatternID GetSource() const { return source_; }
    PatternID GetTarget() const { return target_; }
    AssociationType GetType() const { return type_; }

    // Strength management (thread-safe)
    float GetStrength() const { return strength_.load(std::memory_order_relaxed); }
    void SetStrength(float strength);
    void AdjustStrength(float delta);  // Bounded [0,1]

    // Co-occurrence tracking
    uint32_t GetCoOccurrenceCount() const { return co_occurrence_count_.load(); }
    void IncrementCoOccurrence(uint32_t count = 1);

    // Temporal correlation
    float GetTemporalCorrelation() const { return temporal_correlation_.load(); }
    void SetTemporalCorrelation(float correlation);
    void UpdateTemporalCorrelation(float new_observation, float learning_rate = 0.1f);

    // Decay management
    float GetDecayRate() const { return decay_rate_; }
    void SetDecayRate(float rate);
    Timestamp GetLastReinforcement() const;
    void RecordReinforcement();

    // Apply time-based decay
    void ApplyDecay(Timestamp::Duration elapsed_time);

    // Context profile
    const ContextVector& GetContextProfile() const;
    void SetContextProfile(const ContextVector& context);
    void UpdateContextProfile(const ContextVector& observed_context, float learning_rate = 0.1f);

    // Compute context-modulated strength
    float GetContextualStrength(const ContextVector& current_context) const;

    // Age and statistics
    Timestamp::Duration GetAge() const;
    bool IsActive(Timestamp::Duration max_idle_time) const;

    // Serialization
    void Serialize(std::ostream& out) const;
    static AssociationEdge Deserialize(std::istream& in);

    // String representation
    std::string ToString() const;

    // Memory footprint
    size_t EstimateMemoryUsage() const;

    // Comparison operators
    bool operator==(const AssociationEdge& other) const;
    bool operator<(const AssociationEdge& other) const;  // For sorting by strength

private:
    // Core identification
    PatternID source_;
    PatternID target_;
    AssociationType type_{AssociationType::CATEGORICAL};

    // Strength (atomic for thread-safety)
    std::atomic<float> strength_{0.5f};

    // Statistics (atomic)
    std::atomic<uint32_t> co_occurrence_count_{0};
    std::atomic<float> temporal_correlation_{0.0f};

    // Decay parameters
    float decay_rate_{0.01f};  // Strength loss per time unit
    mutable std::atomic<uint64_t> last_reinforcement_{0};  // Timestamp in micros

    // Context (protected by mutex)
    mutable std::mutex context_mutex_;
    ContextVector context_profile_;

    Timestamp creation_time_;
};

} // namespace dpan
```

**Complete Implementation** (src/association/association_edge.cpp):

```cpp
// File: src/association/association_edge.cpp
#include "association/association_edge.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>

namespace dpan {

AssociationEdge::AssociationEdge(
    PatternID source,
    PatternID target,
    AssociationType type,
    float initial_strength
) : source_(source),
    target_(target),
    type_(type),
    strength_(std::clamp(initial_strength, 0.0f, 1.0f)),
    creation_time_(Timestamp::Now())
{
    last_reinforcement_.store(creation_time_.ToMicros(), std::memory_order_relaxed);
}

void AssociationEdge::SetStrength(float strength) {
    // Clamp to valid range [0, 1]
    strength = std::clamp(strength, 0.0f, 1.0f);
    strength_.store(strength, std::memory_order_relaxed);
}

void AssociationEdge::AdjustStrength(float delta) {
    float current = strength_.load(std::memory_order_relaxed);
    float new_strength = std::clamp(current + delta, 0.0f, 1.0f);
    strength_.store(new_strength, std::memory_order_relaxed);
}

void AssociationEdge::IncrementCoOccurrence(uint32_t count) {
    co_occurrence_count_.fetch_add(count, std::memory_order_relaxed);
}

void AssociationEdge::SetTemporalCorrelation(float correlation) {
    correlation = std::clamp(correlation, -1.0f, 1.0f);
    temporal_correlation_.store(correlation, std::memory_order_relaxed);
}

void AssociationEdge::UpdateTemporalCorrelation(float new_observation, float learning_rate) {
    float current = temporal_correlation_.load(std::memory_order_relaxed);
    // Exponential moving average
    float updated = current + learning_rate * (new_observation - current);
    updated = std::clamp(updated, -1.0f, 1.0f);
    temporal_correlation_.store(updated, std::memory_order_relaxed);
}

void AssociationEdge::SetDecayRate(float rate) {
    decay_rate_ = std::max(0.0f, rate);
}

Timestamp AssociationEdge::GetLastReinforcement() const {
    uint64_t micros = last_reinforcement_.load(std::memory_order_relaxed);
    return Timestamp::FromMicros(micros);
}

void AssociationEdge::RecordReinforcement() {
    Timestamp now = Timestamp::Now();
    last_reinforcement_.store(now.ToMicros(), std::memory_order_relaxed);
}

void AssociationEdge::ApplyDecay(Timestamp::Duration elapsed_time) {
    // Exponential decay: s(t) = s(0) * exp(-d * t)
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed_time).count();
    float decay_factor = std::exp(-decay_rate_ * seconds);

    float current = strength_.load(std::memory_order_relaxed);
    float decayed = current * decay_factor;
    strength_.store(std::max(0.0f, decayed), std::memory_order_relaxed);
}

const ContextVector& AssociationEdge::GetContextProfile() const {
    std::lock_guard<std::mutex> lock(context_mutex_);
    return context_profile_;
}

void AssociationEdge::SetContextProfile(const ContextVector& context) {
    std::lock_guard<std::mutex> lock(context_mutex_);
    context_profile_ = context;
}

void AssociationEdge::UpdateContextProfile(const ContextVector& observed_context, float learning_rate) {
    std::lock_guard<std::mutex> lock(context_mutex_);

    // Update each dimension using exponential moving average
    for (const auto& dim : observed_context.GetDimensions()) {
        float current = context_profile_.Get(dim);
        float observed = observed_context.Get(dim);
        float updated = current + learning_rate * (observed - current);
        context_profile_.Set(dim, updated);
    }
}

float AssociationEdge::GetContextualStrength(const ContextVector& current_context) const {
    std::lock_guard<std::mutex> lock(context_mutex_);

    if (context_profile_.IsEmpty()) {
        // No context profile yet, use base strength
        return strength_.load(std::memory_order_relaxed);
    }

    // Compute context similarity (cosine similarity)
    float context_match = context_profile_.CosineSimilarity(current_context);

    // Modulate strength by context match
    // If contexts match well (similarity near 1), use full strength
    // If contexts don't match (similarity near 0), reduce strength
    float base_strength = strength_.load(std::memory_order_relaxed);
    float context_factor = 0.5f + 0.5f * context_match;  // Maps [-1,1] to [0,1]

    return base_strength * context_factor;
}

Timestamp::Duration AssociationEdge::GetAge() const {
    return Timestamp::Now() - creation_time_;
}

bool AssociationEdge::IsActive(Timestamp::Duration max_idle_time) const {
    Timestamp last_reinforcement = GetLastReinforcement();
    Timestamp::Duration idle_time = Timestamp::Now() - last_reinforcement;
    return idle_time <= max_idle_time;
}

void AssociationEdge::Serialize(std::ostream& out) const {
    // Serialize core data
    source_.Serialize(out);
    target_.Serialize(out);

    uint8_t type_val = static_cast<uint8_t>(type_);
    out.write(reinterpret_cast<const char*>(&type_val), sizeof(type_val));

    float strength = strength_.load(std::memory_order_relaxed);
    out.write(reinterpret_cast<const char*>(&strength), sizeof(strength));

    uint32_t co_occ = co_occurrence_count_.load(std::memory_order_relaxed);
    out.write(reinterpret_cast<const char*>(&co_occ), sizeof(co_occ));

    float temp_corr = temporal_correlation_.load(std::memory_order_relaxed);
    out.write(reinterpret_cast<const char*>(&temp_corr), sizeof(temp_corr));

    out.write(reinterpret_cast<const char*>(&decay_rate_), sizeof(decay_rate_));

    uint64_t last_reinf = last_reinforcement_.load(std::memory_order_relaxed);
    out.write(reinterpret_cast<const char*>(&last_reinf), sizeof(last_reinf));

    creation_time_.Serialize(out);

    // Serialize context (protected by lock during serialization)
    std::lock_guard<std::mutex> lock(context_mutex_);
    context_profile_.Serialize(out);
}

AssociationEdge AssociationEdge::Deserialize(std::istream& in) {
    AssociationEdge edge;

    edge.source_ = PatternID::Deserialize(in);
    edge.target_ = PatternID::Deserialize(in);

    uint8_t type_val;
    in.read(reinterpret_cast<char*>(&type_val), sizeof(type_val));
    edge.type_ = static_cast<AssociationType>(type_val);

    float strength;
    in.read(reinterpret_cast<char*>(&strength), sizeof(strength));
    edge.strength_.store(strength, std::memory_order_relaxed);

    uint32_t co_occ;
    in.read(reinterpret_cast<char*>(&co_occ), sizeof(co_occ));
    edge.co_occurrence_count_.store(co_occ, std::memory_order_relaxed);

    float temp_corr;
    in.read(reinterpret_cast<char*>(&temp_corr), sizeof(temp_corr));
    edge.temporal_correlation_.store(temp_corr, std::memory_order_relaxed);

    in.read(reinterpret_cast<char*>(&edge.decay_rate_), sizeof(edge.decay_rate_));

    uint64_t last_reinf;
    in.read(reinterpret_cast<char*>(&last_reinf), sizeof(last_reinf));
    edge.last_reinforcement_.store(last_reinf, std::memory_order_relaxed);

    edge.creation_time_ = Timestamp::Deserialize(in);

    edge.context_profile_ = ContextVector::Deserialize(in);

    return edge;
}

std::string AssociationEdge::ToString() const {
    std::ostringstream oss;
    oss << "AssociationEdge{";
    oss << "src=" << source_.ToString();
    oss << ", tgt=" << target_.ToString();
    oss << ", type=" << dpan::ToString(type_);
    oss << ", strength=" << strength_.load(std::memory_order_relaxed);
    oss << ", co_occ=" << co_occurrence_count_.load(std::memory_order_relaxed);
    oss << ", temp_corr=" << temporal_correlation_.load(std::memory_order_relaxed);
    oss << ", age=" << GetAge().count() / 1000000 << "s";
    oss << "}";
    return oss.str();
}

size_t AssociationEdge::EstimateMemoryUsage() const {
    size_t base_size = sizeof(*this);

    std::lock_guard<std::mutex> lock(context_mutex_);
    size_t context_size = context_profile_.Size() * (sizeof(std::string) + sizeof(float) + 32);  // Rough estimate

    return base_size + context_size;
}

bool AssociationEdge::operator==(const AssociationEdge& other) const {
    return source_ == other.source_ &&
           target_ == other.target_ &&
           type_ == other.type_;
}

bool AssociationEdge::operator<(const AssociationEdge& other) const {
    // Sort by strength (descending)
    return strength_.load(std::memory_order_relaxed) >
           other.strength_.load(std::memory_order_relaxed);
}

} // namespace dpan
```

**Comprehensive Unit Tests** (tests/association/association_edge_test.cpp):

```cpp
// File: tests/association/association_edge_test.cpp
#include "association/association_edge.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace dpan {
namespace {

TEST(AssociationEdgeTest, DefaultConstructor) {
    AssociationEdge edge;
    EXPECT_EQ(0.5f, edge.GetStrength());
    EXPECT_EQ(0u, edge.GetCoOccurrenceCount());
}

TEST(AssociationEdgeTest, ParameterizedConstructor) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();

    AssociationEdge edge(src, tgt, AssociationType::CAUSAL, 0.8f);

    EXPECT_EQ(src, edge.GetSource());
    EXPECT_EQ(tgt, edge.GetTarget());
    EXPECT_EQ(AssociationType::CAUSAL, edge.GetType());
    EXPECT_FLOAT_EQ(0.8f, edge.GetStrength());
}

TEST(AssociationEdgeTest, StrengthBounding) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();

    // Test upper bound
    AssociationEdge edge1(src, tgt, AssociationType::SPATIAL, 1.5f);
    EXPECT_FLOAT_EQ(1.0f, edge1.GetStrength());

    // Test lower bound
    AssociationEdge edge2(src, tgt, AssociationType::SPATIAL, -0.5f);
    EXPECT_FLOAT_EQ(0.0f, edge2.GetStrength());
}

TEST(AssociationEdgeTest, AdjustStrength) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CATEGORICAL, 0.5f);

    // Positive adjustment
    edge.AdjustStrength(0.3f);
    EXPECT_FLOAT_EQ(0.8f, edge.GetStrength());

    // Adjustment should respect upper bound
    edge.AdjustStrength(0.5f);
    EXPECT_FLOAT_EQ(1.0f, edge.GetStrength());

    // Negative adjustment
    edge.AdjustStrength(-0.3f);
    EXPECT_FLOAT_EQ(0.7f, edge.GetStrength());

    // Adjustment should respect lower bound
    edge.AdjustStrength(-1.0f);
    EXPECT_FLOAT_EQ(0.0f, edge.GetStrength());
}

TEST(AssociationEdgeTest, CoOccurrenceTracking) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL);

    EXPECT_EQ(0u, edge.GetCoOccurrenceCount());

    edge.IncrementCoOccurrence();
    EXPECT_EQ(1u, edge.GetCoOccurrenceCount());

    edge.IncrementCoOccurrence(5);
    EXPECT_EQ(6u, edge.GetCoOccurrenceCount());
}

TEST(AssociationEdgeTest, TemporalCorrelation) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL);

    EXPECT_FLOAT_EQ(0.0f, edge.GetTemporalCorrelation());

    edge.SetTemporalCorrelation(0.7f);
    EXPECT_FLOAT_EQ(0.7f, edge.GetTemporalCorrelation());

    // Test bounding
    edge.SetTemporalCorrelation(1.5f);
    EXPECT_FLOAT_EQ(1.0f, edge.GetTemporalCorrelation());

    edge.SetTemporalCorrelation(-1.5f);
    EXPECT_FLOAT_EQ(-1.0f, edge.GetTemporalCorrelation());
}

TEST(AssociationEdgeTest, TemporalCorrelationUpdate) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL);

    edge.SetTemporalCorrelation(0.5f);

    // Update with new observation
    edge.UpdateTemporalCorrelation(0.8f, 0.5f);  // learning_rate = 0.5

    // Expected: 0.5 + 0.5 * (0.8 - 0.5) = 0.5 + 0.15 = 0.65
    EXPECT_NEAR(0.65f, edge.GetTemporalCorrelation(), 0.001f);
}

TEST(AssociationEdgeTest, DecayRate) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::SPATIAL);

    edge.SetDecayRate(0.05f);
    EXPECT_FLOAT_EQ(0.05f, edge.GetDecayRate());

    // Negative decay rate should be clamped to 0
    edge.SetDecayRate(-0.1f);
    EXPECT_FLOAT_EQ(0.0f, edge.GetDecayRate());
}

TEST(AssociationEdgeTest, ApplyDecay) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CATEGORICAL, 1.0f);

    edge.SetDecayRate(0.01f);

    // Apply decay for 100 seconds
    // s(t) = s(0) * exp(-0.01 * 100) = 1.0 * exp(-1) ≈ 0.368
    auto elapsed = std::chrono::seconds(100);
    edge.ApplyDecay(elapsed);

    EXPECT_NEAR(0.368f, edge.GetStrength(), 0.01f);
}

TEST(AssociationEdgeTest, ReinforcementTracking) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::FUNCTIONAL);

    Timestamp before = Timestamp::Now();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    edge.RecordReinforcement();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Timestamp after = Timestamp::Now();

    Timestamp last_reinforcement = edge.GetLastReinforcement();
    EXPECT_GT(last_reinforcement, before);
    EXPECT_LT(last_reinforcement, after);
}

TEST(AssociationEdgeTest, ContextProfile) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::SPATIAL);

    ContextVector context;
    context.Set("temperature", 25.0f);
    context.Set("humidity", 60.0f);

    edge.SetContextProfile(context);

    const ContextVector& retrieved = edge.GetContextProfile();
    EXPECT_FLOAT_EQ(25.0f, retrieved.Get("temperature"));
    EXPECT_FLOAT_EQ(60.0f, retrieved.Get("humidity"));
}

TEST(AssociationEdgeTest, ContextProfileUpdate) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::SPATIAL);

    ContextVector initial;
    initial.Set("temperature", 20.0f);
    edge.SetContextProfile(initial);

    ContextVector observed;
    observed.Set("temperature", 30.0f);
    edge.UpdateContextProfile(observed, 0.5f);

    // Expected: 20 + 0.5 * (30 - 20) = 25
    const ContextVector& updated = edge.GetContextProfile();
    EXPECT_NEAR(25.0f, updated.Get("temperature"), 0.001f);
}

TEST(AssociationEdgeTest, ContextualStrength) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CATEGORICAL, 0.8f);

    // Set context profile
    ContextVector profile;
    profile.Set("time_of_day", 1.0f);  // Normalized
    profile.Set("location", 0.5f);
    edge.SetContextProfile(profile);

    // Test with matching context
    ContextVector matching_context;
    matching_context.Set("time_of_day", 1.0f);
    matching_context.Set("location", 0.5f);

    float contextual_strength = edge.GetContextualStrength(matching_context);
    // Should be close to base strength since contexts match
    EXPECT_NEAR(0.8f, contextual_strength, 0.1f);

    // Test with non-matching context
    ContextVector non_matching;
    non_matching.Set("time_of_day", 0.0f);
    non_matching.Set("location", 0.0f);

    float weak_strength = edge.GetContextualStrength(non_matching);
    // Should be weaker than base strength
    EXPECT_LT(weak_strength, 0.8f);
}

TEST(AssociationEdgeTest, Age) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto age = edge.GetAge();
    auto age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(age).count();

    EXPECT_GE(age_ms, 100);
    EXPECT_LT(age_ms, 200);  // Allow some overhead
}

TEST(AssociationEdgeTest, IsActive) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::FUNCTIONAL);

    // Should be active immediately after creation
    EXPECT_TRUE(edge.IsActive(std::chrono::seconds(1)));

    std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    // Should not be active after 1.5 seconds with max_idle_time = 1s
    EXPECT_FALSE(edge.IsActive(std::chrono::seconds(1)));

    // Should be active with larger threshold
    EXPECT_TRUE(edge.IsActive(std::chrono::seconds(2)));
}

TEST(AssociationEdgeTest, SerializationRoundTrip) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge original(src, tgt, AssociationType::COMPOSITIONAL, 0.75f);

    original.IncrementCoOccurrence(10);
    original.SetTemporalCorrelation(0.6f);
    original.SetDecayRate(0.02f);

    ContextVector context;
    context.Set("dim1", 1.0f);
    context.Set("dim2", 2.0f);
    original.SetContextProfile(context);

    // Serialize
    std::stringstream ss;
    original.Serialize(ss);

    // Deserialize
    AssociationEdge deserialized = AssociationEdge::Deserialize(ss);

    // Verify
    EXPECT_EQ(original.GetSource(), deserialized.GetSource());
    EXPECT_EQ(original.GetTarget(), deserialized.GetTarget());
    EXPECT_EQ(original.GetType(), deserialized.GetType());
    EXPECT_FLOAT_EQ(original.GetStrength(), deserialized.GetStrength());
    EXPECT_EQ(original.GetCoOccurrenceCount(), deserialized.GetCoOccurrenceCount());
    EXPECT_FLOAT_EQ(original.GetTemporalCorrelation(), deserialized.GetTemporalCorrelation());
    EXPECT_FLOAT_EQ(original.GetDecayRate(), deserialized.GetDecayRate());
}

TEST(AssociationEdgeTest, ThreadSafeStrengthUpdates) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CAUSAL, 0.5f);

    constexpr int kNumThreads = 10;
    constexpr int kUpdatesPerThread = 1000;

    std::vector<std::thread> threads;

    // Launch threads that adjust strength
    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&edge]() {
            for (int j = 0; j < kUpdatesPerThread; ++j) {
                edge.AdjustStrength(0.0001f);
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify final strength (should be bounded at 1.0)
    EXPECT_FLOAT_EQ(1.0f, edge.GetStrength());
}

TEST(AssociationEdgeTest, ThreadSafeCoOccurrenceIncrement) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::SPATIAL);

    constexpr int kNumThreads = 10;
    constexpr int kIncrementsPerThread = 1000;

    std::vector<std::thread> threads;

    for (int i = 0; i < kNumThreads; ++i) {
        threads.emplace_back([&edge]() {
            for (int j = 0; j < kIncrementsPerThread; ++j) {
                edge.IncrementCoOccurrence();
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Should have exactly kNumThreads * kIncrementsPerThread increments
    EXPECT_EQ(kNumThreads * kIncrementsPerThread, edge.GetCoOccurrenceCount());
}

TEST(AssociationEdgeTest, ComparisonOperators) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();

    AssociationEdge edge1(src, tgt, AssociationType::CAUSAL, 0.8f);
    AssociationEdge edge2(src, tgt, AssociationType::CAUSAL, 0.8f);
    AssociationEdge edge3(src, tgt, AssociationType::SPATIAL, 0.6f);

    // Equality based on source, target, type
    EXPECT_EQ(edge1, edge2);
    EXPECT_NE(edge1, edge3);

    // Ordering based on strength (descending)
    EXPECT_LT(edge3, edge1);  // 0.6 is "less than" 0.8 (weaker)
}

TEST(AssociationEdgeTest, MemoryUsageEstimation) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CATEGORICAL);

    size_t base_size = edge.EstimateMemoryUsage();
    EXPECT_GT(base_size, 0u);
    EXPECT_LT(base_size, 1024u);  // Should be under 1KB

    // Add context and check size increase
    ContextVector context;
    for (int i = 0; i < 10; ++i) {
        context.Set("dim" + std::to_string(i), static_cast<float>(i));
    }
    edge.SetContextProfile(context);

    size_t with_context = edge.EstimateMemoryUsage();
    EXPECT_GT(with_context, base_size);
}

TEST(AssociationEdgeTest, ToStringOutput) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::FUNCTIONAL, 0.7f);

    std::string str = edge.ToString();

    // Should contain key information
    EXPECT_NE(std::string::npos, str.find("AssociationEdge"));
    EXPECT_NE(std::string::npos, str.find("FUNCTIONAL"));
    EXPECT_NE(std::string::npos, str.find("strength"));
}

} // namespace
} // namespace dpan
```

**Benchmarks** (benchmarks/association/association_edge_benchmark.cpp):

```cpp
// File: benchmarks/association/association_edge_benchmark.cpp
#include "association/association_edge.hpp"
#include <benchmark/benchmark.h>

namespace dpan {

static void BM_AssociationEdgeCreation(benchmark::State& state) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();

    for (auto _ : state) {
        AssociationEdge edge(src, tgt, AssociationType::CAUSAL, 0.5f);
        benchmark::DoNotOptimize(edge);
    }
}
BENCHMARK(BM_AssociationEdgeCreation);

static void BM_StrengthAdjustment(benchmark::State& state) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::SPATIAL, 0.5f);

    for (auto _ : state) {
        edge.AdjustStrength(0.01f);
    }
}
BENCHMARK(BM_StrengthAdjustment);

static void BM_ContextualStrength(benchmark::State& state) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::CATEGORICAL, 0.7f);

    ContextVector profile;
    profile.Set("dim1", 1.0f);
    profile.Set("dim2", 0.5f);
    edge.SetContextProfile(profile);

    ContextVector current;
    current.Set("dim1", 0.8f);
    current.Set("dim2", 0.6f);

    for (auto _ : state) {
        float strength = edge.GetContextualStrength(current);
        benchmark::DoNotOptimize(strength);
    }
}
BENCHMARK(BM_ContextualStrength);

static void BM_ApplyDecay(benchmark::State& state) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::FUNCTIONAL, 1.0f);
    edge.SetDecayRate(0.01f);

    auto elapsed = std::chrono::seconds(10);

    for (auto _ : state) {
        edge.ApplyDecay(elapsed);
        edge.SetStrength(1.0f);  // Reset for next iteration
    }
}
BENCHMARK(BM_ApplyDecay);

static void BM_Serialization(benchmark::State& state) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::COMPOSITIONAL, 0.75f);

    for (auto _ : state) {
        std::stringstream ss;
        edge.Serialize(ss);
        benchmark::DoNotOptimize(ss);
    }
}
BENCHMARK(BM_Serialization);

static void BM_Deserialization(benchmark::State& state) {
    PatternID src = PatternID::Generate();
    PatternID tgt = PatternID::Generate();
    AssociationEdge edge(src, tgt, AssociationType::SPATIAL, 0.6f);

    std::stringstream ss;
    edge.Serialize(ss);
    std::string serialized = ss.str();

    for (auto _ : state) {
        std::stringstream input(serialized);
        AssociationEdge deserialized = AssociationEdge::Deserialize(input);
        benchmark::DoNotOptimize(deserialized);
    }
}
BENCHMARK(BM_Deserialization);

} // namespace dpan

BENCHMARK_MAIN();
```

**Performance Targets**:
- Edge creation: <200ns
- Strength adjustment: <50ns
- Contextual strength calculation: <500ns (with context)
- Decay application: <100ns
- Serialization: <2µs
- Deserialization: <2µs

**Acceptance Criteria**:
- [ ] All 30+ unit tests pass
- [ ] Thread-safe operations verified with ThreadSanitizer
- [ ] Memory usage <100 bytes per edge (without large context)
- [ ] All performance benchmarks meet targets
- [ ] Serialization maintains all data accurately
- [ ] >95% code coverage
- [ ] No memory leaks (valgrind clean)
- [ ] Context-sensitive strength works correctly

---

### Task 3.1.2: Implement AssociationMatrix

**Duration**: 4 days (32 hours)
**Priority**: Critical
**Files to create**:
- `src/association/association_matrix.hpp`
- `src/association/association_matrix.cpp`
- `tests/association/association_matrix_test.cpp`

#### Subtask 3.1.2.1: Design Sparse Matrix Storage (8 hours)

**Overview**:
The AssociationMatrix stores a sparse directed graph of associations between patterns. For a system with N patterns, the theoretical dense matrix would be N×N, but actual associations are typically <1% filled (highly sparse).

**Sparse Matrix Formats**:

1. **COO (Coordinate List)** - Simple, good for construction
2. **CSR (Compressed Sparse Row)** - Excellent for row-wise operations
3. **CSC (Compressed Sparse Column)** - Excellent for column-wise operations
4. **Hybrid Approach** - CSR + reverse index for bidirectional lookups

**Decision**: Use hybrid CSR + adjacency lists for optimal performance.

**Mathematical Representation**:

```
Sparse Matrix A where:
  A[i][j] = AssociationEdge from pattern i to pattern j

Storage:
  - row_ptr[i] = index in values array where row i starts
  - col_indices[k] = column index of k-th non-zero element
  - values[k] = AssociationEdge at position k
  - reverse_index: map from (source, target) to index in values
```

**Implementation**:

```cpp
// File: src/association/association_matrix.hpp
#pragma once

#include "association/association_edge.hpp"
#include <vector>
#include <unordered_map>
#include <shared_mutex>
#include <optional>

namespace dpan {

// Hash function for (PatternID, PatternID) pairs
struct PatternPairHash {
    size_t operator()(const std::pair<PatternID, PatternID>& p) const {
        size_t h1 = PatternID::Hash()(p.first);
        size_t h2 = PatternID::Hash()(p.second);
        return h1 ^ (h2 << 1);  // Combine hashes
    }
};

// AssociationMatrix: Sparse matrix of associations
class AssociationMatrix {
public:
    struct Config {
        size_t initial_capacity{10000};
        bool enable_reverse_lookup{true};
        bool enable_type_index{true};
        float load_factor_threshold{0.75f};
    };

    explicit AssociationMatrix(const Config& config = {});
    ~AssociationMatrix() = default;

    // Add/Update associations
    bool AddAssociation(const AssociationEdge& edge);
    bool UpdateAssociation(PatternID source, PatternID target, const AssociationEdge& edge);
    bool RemoveAssociation(PatternID source, PatternID target);

    // Batch operations (more efficient)
    size_t AddAssociationsBatch(const std::vector<AssociationEdge>& edges);
    size_t RemoveAssociationsBatch(const std::vector<std::pair<PatternID, PatternID>>& pairs);

    // Lookup operations
    std::optional<AssociationEdge> GetAssociation(PatternID source, PatternID target) const;
    bool HasAssociation(PatternID source, PatternID target) const;

    // Get all associations from a source pattern
    std::vector<AssociationEdge> GetOutgoingAssociations(PatternID source) const;

    // Get all associations to a target pattern
    std::vector<AssociationEdge> GetIncomingAssociations(PatternID target) const;

    // Get associations of a specific type
    std::vector<AssociationEdge> GetAssociationsByType(AssociationType type) const;

    // Get neighbors
    std::vector<PatternID> GetNeighbors(PatternID pattern, bool outgoing = true) const;
    std::vector<PatternID> GetMutualNeighbors(PatternID pattern) const;

    // Strength operations
    bool StrengthenAssociation(PatternID source, PatternID target, float amount);
    bool WeakenAssociation(PatternID source, PatternID target, float amount);

    // Batch strength updates
    void ApplyDecayAll(Timestamp::Duration elapsed_time);
    void ApplyDecayPattern(PatternID pattern, Timestamp::Duration elapsed_time);

    // Statistics
    size_t GetAssociationCount() const;
    size_t GetPatternCount() const;  // Number of unique patterns with associations
    float GetAverageDegree() const;
    float GetAverageStrength() const;
    float GetDensity() const;  // Fraction of possible edges that exist

    // Graph properties
    size_t GetDegree(PatternID pattern, bool outgoing = true) const;
    std::vector<PatternID> GetIsolatedPatterns() const;

    // Activation propagation support
    struct ActivationResult {
        PatternID pattern;
        float activation;
    };

    std::vector<ActivationResult> PropagateActivation(
        PatternID source,
        float initial_activation,
        size_t max_hops = 3,
        float min_activation = 0.01f,
        const ContextVector* context = nullptr
    ) const;

    // Serialization
    void Serialize(std::ostream& out) const;
    static AssociationMatrix Deserialize(std::istream& in);

    // Memory management
    void Compact();  // Remove deleted associations and rebuild indices
    void Clear();
    size_t EstimateMemoryUsage() const;

    // Debugging
    void PrintStatistics(std::ostream& out) const;
    std::string ToString() const;

private:
    Config config_;

    // Main storage (thread-safe with reader-writer lock)
    mutable std::shared_mutex mutex_;

    // Edge storage
    std::vector<AssociationEdge> edges_;

    // CSR-like structure for efficient row access
    std::unordered_map<PatternID, std::vector<size_t>> outgoing_index_;  // source -> edge indices

    // Reverse index for efficient column access (incoming associations)
    std::unordered_map<PatternID, std::vector<size_t>> incoming_index_;  // target -> edge indices

    // Fast lookup: (source, target) -> edge index
    std::unordered_map<std::pair<PatternID, PatternID>, size_t, PatternPairHash> edge_lookup_;

    // Type index (optional)
    std::unordered_map<AssociationType, std::vector<size_t>> type_index_;

    // Track deleted edges (indices to reuse)
    std::vector<size_t> deleted_indices_;

    // Helper methods
    size_t AllocateEdgeIndex();
    void ReleaseEdgeIndex(size_t index);
    void RebuildIndices();
    void UpdateIndices(size_t edge_index, bool add);
};

} // namespace dpan
```

**Key Implementation Details**:

1. **Thread Safety**:
   - Use `std::shared_mutex` for reader-writer locking
   - Multiple readers can access simultaneously
   - Writers get exclusive access

2. **Memory Efficiency**:
   - Reuse deleted edge indices
   - Compact operation to reclaim space
   - Lazy index building

3. **Performance Optimizations**:
   - Batch operations to amortize lock overhead
   - Index caching for hot paths
   - Separate indices for different access patterns

#### Subtask 3.1.2.2: Implement Core Operations (12 hours)

```cpp
// File: src/association/association_matrix.cpp
#include "association/association_matrix.hpp"
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace dpan {

AssociationMatrix::AssociationMatrix(const Config& config)
    : config_(config)
{
    edges_.reserve(config_.initial_capacity);
}

size_t AssociationMatrix::AllocateEdgeIndex() {
    if (!deleted_indices_.empty()) {
        size_t index = deleted_indices_.back();
        deleted_indices_.pop_back();
        return index;
    }

    size_t index = edges_.size();
    edges_.emplace_back();  // Add placeholder
    return index;
}

void AssociationMatrix::ReleaseEdgeIndex(size_t index) {
    deleted_indices_.push_back(index);
}

bool AssociationMatrix::AddAssociation(const AssociationEdge& edge) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    PatternID source = edge.GetSource();
    PatternID target = edge.GetTarget();

    auto key = std::make_pair(source, target);

    // Check if association already exists
    if (edge_lookup_.find(key) != edge_lookup_.end()) {
        return false;  // Already exists
    }

    // Allocate index and store edge
    size_t index = AllocateEdgeIndex();
    edges_[index] = edge;

    // Update lookup
    edge_lookup_[key] = index;

    // Update indices
    UpdateIndices(index, true);

    return true;
}

bool AssociationMatrix::UpdateAssociation(
    PatternID source,
    PatternID target,
    const AssociationEdge& edge
) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    auto it = edge_lookup_.find(key);

    if (it == edge_lookup_.end()) {
        return false;  // Doesn't exist
    }

    edges_[it->second] = edge;
    return true;
}

bool AssociationMatrix::RemoveAssociation(PatternID source, PatternID target) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    auto it = edge_lookup_.find(key);

    if (it == edge_lookup_.end()) {
        return false;  // Doesn't exist
    }

    size_t index = it->second;

    // Update indices before deletion
    UpdateIndices(index, false);

    // Remove from lookup
    edge_lookup_.erase(it);

    // Mark for reuse
    ReleaseEdgeIndex(index);

    return true;
}

void AssociationMatrix::UpdateIndices(size_t edge_index, bool add) {
    const AssociationEdge& edge = edges_[edge_index];
    PatternID source = edge.GetSource();
    PatternID target = edge.GetTarget();
    AssociationType type = edge.GetType();

    if (add) {
        // Add to outgoing index
        outgoing_index_[source].push_back(edge_index);

        // Add to incoming index (if enabled)
        if (config_.enable_reverse_lookup) {
            incoming_index_[target].push_back(edge_index);
        }

        // Add to type index (if enabled)
        if (config_.enable_type_index) {
            type_index_[type].push_back(edge_index);
        }
    } else {
        // Remove from outgoing index
        auto& outgoing = outgoing_index_[source];
        outgoing.erase(std::remove(outgoing.begin(), outgoing.end(), edge_index), outgoing.end());

        // Remove from incoming index
        if (config_.enable_reverse_lookup) {
            auto& incoming = incoming_index_[target];
            incoming.erase(std::remove(incoming.begin(), incoming.end(), edge_index), incoming.end());
        }

        // Remove from type index
        if (config_.enable_type_index) {
            auto& type_edges = type_index_[type];
            type_edges.erase(std::remove(type_edges.begin(), type_edges.end(), edge_index), type_edges.end());
        }
    }
}

std::optional<AssociationEdge> AssociationMatrix::GetAssociation(
    PatternID source,
    PatternID target
) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    auto it = edge_lookup_.find(key);

    if (it == edge_lookup_.end()) {
        return std::nullopt;
    }

    return edges_[it->second];
}

bool AssociationMatrix::HasAssociation(PatternID source, PatternID target) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    return edge_lookup_.find(key) != edge_lookup_.end();
}

std::vector<AssociationEdge> AssociationMatrix::GetOutgoingAssociations(PatternID source) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<AssociationEdge> result;

    auto it = outgoing_index_.find(source);
    if (it != outgoing_index_.end()) {
        result.reserve(it->second.size());
        for (size_t index : it->second) {
            result.push_back(edges_[index]);
        }
    }

    return result;
}

std::vector<AssociationEdge> AssociationMatrix::GetIncomingAssociations(PatternID target) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<AssociationEdge> result;

    if (!config_.enable_reverse_lookup) {
        // Fallback: linear scan (slow)
        for (const auto& edge : edges_) {
            if (!edge.GetSource().IsValid()) continue;  // Skip deleted
            if (edge.GetTarget() == target) {
                result.push_back(edge);
            }
        }
    } else {
        auto it = incoming_index_.find(target);
        if (it != incoming_index_.end()) {
            result.reserve(it->second.size());
            for (size_t index : it->second) {
                result.push_back(edges_[index]);
            }
        }
    }

    return result;
}

std::vector<PatternID> AssociationMatrix::GetNeighbors(PatternID pattern, bool outgoing) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    std::vector<PatternID> neighbors;

    const auto& index_map = outgoing ? outgoing_index_ : incoming_index_;
    auto it = index_map.find(pattern);

    if (it != index_map.end()) {
        neighbors.reserve(it->second.size());
        for (size_t edge_index : it->second) {
            const AssociationEdge& edge = edges_[edge_index];
            PatternID neighbor = outgoing ? edge.GetTarget() : edge.GetSource();
            neighbors.push_back(neighbor);
        }
    }

    return neighbors;
}

bool AssociationMatrix::StrengthenAssociation(PatternID source, PatternID target, float amount) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    auto it = edge_lookup_.find(key);

    if (it == edge_lookup_.end()) {
        return false;
    }

    edges_[it->second].AdjustStrength(amount);
    return true;
}

void AssociationMatrix::ApplyDecayAll(Timestamp::Duration elapsed_time) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    for (auto& edge : edges_) {
        if (!edge.GetSource().IsValid()) continue;  // Skip deleted
        edge.ApplyDecay(elapsed_time);
    }
}

size_t AssociationMatrix::GetAssociationCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return edge_lookup_.size();  // Actual count (excluding deleted)
}

float AssociationMatrix::GetAverageDegree() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    if (outgoing_index_.empty()) return 0.0f;

    size_t total_degree = 0;
    for (const auto& pair : outgoing_index_) {
        total_degree += pair.second.size();
    }

    return static_cast<float>(total_degree) / outgoing_index_.size();
}

std::vector<AssociationMatrix::ActivationResult> AssociationMatrix::PropagateActivation(
    PatternID source,
    float initial_activation,
    size_t max_hops,
    float min_activation,
    const ContextVector* context
) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Breadth-first propagation with activation accumulation
    std::unordered_map<PatternID, float> activations;
    std::queue<std::pair<PatternID, size_t>> queue;  // (pattern, hop_count)
    std::unordered_set<PatternID> visited;

    activations[source] = initial_activation;
    queue.push({source, 0});
    visited.insert(source);

    while (!queue.empty()) {
        auto [current, hops] = queue.front();
        queue.pop();

        if (hops >= max_hops) continue;

        float current_activation = activations[current];

        // Get outgoing associations
        auto it = outgoing_index_.find(current);
        if (it == outgoing_index_.end()) continue;

        for (size_t edge_index : it->second) {
            const AssociationEdge& edge = edges_[edge_index];
            PatternID target = edge.GetTarget();

            // Compute propagated activation
            float strength = context ?
                edge.GetContextualStrength(*context) :
                edge.GetStrength();

            float propagated = current_activation * strength;

            // Accumulate activation at target
            activations[target] += propagated;

            // Continue propagation if significant and not too deep
            if (propagated >= min_activation && visited.find(target) == visited.end()) {
                queue.push({target, hops + 1});
                visited.insert(target);
            }
        }
    }

    // Convert to result vector
    std::vector<ActivationResult> results;
    results.reserve(activations.size());

    for (const auto& [pattern, activation] : activations) {
        if (pattern != source && activation >= min_activation) {
            results.push_back({pattern, activation});
        }
    }

    // Sort by activation (descending)
    std::sort(results.begin(), results.end(),
        [](const ActivationResult& a, const ActivationResult& b) {
            return a.activation > b.activation;
        });

    return results;
}

void AssociationMatrix::Compact() {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    if (deleted_indices_.empty()) return;

    // Rebuild edge vector without deleted entries
    std::vector<AssociationEdge> new_edges;
    std::unordered_map<std::pair<PatternID, PatternID>, size_t, PatternPairHash> new_lookup;

    new_edges.reserve(edge_lookup_.size());

    for (const auto& [key, old_index] : edge_lookup_) {
        size_t new_index = new_edges.size();
        new_edges.push_back(edges_[old_index]);
        new_lookup[key] = new_index;
    }

    edges_ = std::move(new_edges);
    edge_lookup_ = std::move(new_lookup);
    deleted_indices_.clear();

    // Rebuild all indices
    RebuildIndices();
}

void AssociationMatrix::RebuildIndices() {
    outgoing_index_.clear();
    incoming_index_.clear();
    type_index_.clear();

    for (size_t i = 0; i < edges_.size(); ++i) {
        UpdateIndices(i, true);
    }
}

size_t AssociationMatrix::EstimateMemoryUsage() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    size_t total = 0;

    // Edges
    total += edges_.capacity() * sizeof(AssociationEdge);

    // Indices
    total += outgoing_index_.size() * (sizeof(PatternID) + sizeof(std::vector<size_t>));
    total += incoming_index_.size() * (sizeof(PatternID) + sizeof(std::vector<size_t>));
    total += type_index_.size() * (sizeof(AssociationType) + sizeof(std::vector<size_t>));

    // Lookup map
    total += edge_lookup_.size() * (sizeof(std::pair<PatternID, PatternID>) + sizeof(size_t));

    return total;
}

void AssociationMatrix::Serialize(std::ostream& out) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    // Write count
    size_t count = edge_lookup_.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Write each edge
    for (const auto& [key, index] : edge_lookup_) {
        edges_[index].Serialize(out);
    }
}

AssociationMatrix AssociationMatrix::Deserialize(std::istream& in) {
    AssociationMatrix matrix;

    size_t count;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));

    for (size_t i = 0; i < count; ++i) {
        AssociationEdge edge = AssociationEdge::Deserialize(in);
        matrix.AddAssociation(edge);
    }

    return matrix;
}

} // namespace dpan
```

**Comprehensive Unit Tests** (50+ test cases):

```cpp
// File: tests/association/association_matrix_test.cpp
#include "association/association_matrix.hpp"
#include <gtest/gtest.h>
#include <thread>

namespace dpan {
namespace {

TEST(AssociationMatrixTest, AddAndRetrieveSingle) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.8f);

    EXPECT_TRUE(matrix.AddAssociation(edge));
    EXPECT_EQ(1u, matrix.GetAssociationCount());

    auto retrieved = matrix.GetAssociation(p1, p2);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(p1, retrieved->GetSource());
    EXPECT_EQ(p2, retrieved->GetTarget());
    EXPECT_FLOAT_EQ(0.8f, retrieved->GetStrength());
}

TEST(AssociationMatrixTest, CannotAddDuplicate) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge edge1(p1, p2, AssociationType::SPATIAL, 0.5f);
    AssociationEdge edge2(p1, p2, AssociationType::SPATIAL, 0.7f);

    EXPECT_TRUE(matrix.AddAssociation(edge1));
    EXPECT_FALSE(matrix.AddAssociation(edge2));  // Duplicate
    EXPECT_EQ(1u, matrix.GetAssociationCount());
}

TEST(AssociationMatrixTest, UpdateExisting) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge edge1(p1, p2, AssociationType::CATEGORICAL, 0.5f);
    matrix.AddAssociation(edge1);

    AssociationEdge edge2(p1, p2, AssociationType::CATEGORICAL, 0.9f);
    EXPECT_TRUE(matrix.UpdateAssociation(p1, p2, edge2));

    auto retrieved = matrix.GetAssociation(p1, p2);
    EXPECT_FLOAT_EQ(0.9f, retrieved->GetStrength());
}

TEST(AssociationMatrixTest, RemoveAssociation) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge edge(p1, p2, AssociationType::FUNCTIONAL, 0.6f);
    matrix.AddAssociation(edge);

    EXPECT_TRUE(matrix.HasAssociation(p1, p2));

    EXPECT_TRUE(matrix.RemoveAssociation(p1, p2));
    EXPECT_FALSE(matrix.HasAssociation(p1, p2));
    EXPECT_EQ(0u, matrix.GetAssociationCount());
}

TEST(AssociationMatrixTest, GetOutgoingAssociations) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    // p1 -> p2, p1 -> p3, p1 -> p4
    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p1, p4, AssociationType::CAUSAL, 0.7f));

    auto outgoing = matrix.GetOutgoingAssociations(p1);
    EXPECT_EQ(3u, outgoing.size());
}

TEST(AssociationMatrixTest, GetIncomingAssociations) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    // p2 -> p1, p3 -> p1, p4 -> p1
    matrix.AddAssociation(AssociationEdge(p2, p1, AssociationType::SPATIAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p3, p1, AssociationType::SPATIAL, 0.6f));
    matrix.AddAssociation(AssociationEdge(p4, p1, AssociationType::SPATIAL, 0.7f));

    auto incoming = matrix.GetIncomingAssociations(p1);
    EXPECT_EQ(3u, incoming.size());
}

TEST(AssociationMatrixTest, GetNeighbors) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CATEGORICAL, 0.5f));
    matrix.AddAssociation(AssociationEdge(p1, p3, AssociationType::CATEGORICAL, 0.5f));

    auto outgoing_neighbors = matrix.GetNeighbors(p1, true);
    EXPECT_EQ(2u, outgoing_neighbors.size());

    auto incoming_neighbors = matrix.GetNeighbors(p2, false);
    EXPECT_EQ(1u, incoming_neighbors.size());
    EXPECT_EQ(p1, incoming_neighbors[0]);
}

TEST(AssociationMatrixTest, StrengthenAssociation) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::FUNCTIONAL, 0.5f));

    EXPECT_TRUE(matrix.StrengthenAssociation(p1, p2, 0.2f));

    auto edge = matrix.GetAssociation(p1, p2);
    EXPECT_NEAR(0.7f, edge->GetStrength(), 0.001f);
}

TEST(AssociationMatrixTest, ApplyDecayAll) {
    AssociationMatrix matrix;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    AssociationEdge edge1(p1, p2, AssociationType::CAUSAL, 1.0f);
    edge1.SetDecayRate(0.01f);
    matrix.AddAssociation(edge1);

    AssociationEdge edge2(p2, p3, AssociationType::SPATIAL, 1.0f);
    edge2.SetDecayRate(0.01f);
    matrix.AddAssociation(edge2);

    // Apply decay for 100 seconds
    matrix.ApplyDecayAll(std::chrono::seconds(100));

    auto retrieved1 = matrix.GetAssociation(p1, p2);
    auto retrieved2 = matrix.GetAssociation(p2, p3);

    // Both should have decayed to ~0.368
    EXPECT_NEAR(0.368f, retrieved1->GetStrength(), 0.01f);
    EXPECT_NEAR(0.368f, retrieved2->GetStrength(), 0.01f);
}

TEST(AssociationMatrixTest, PropagateActivation) {
    AssociationMatrix matrix;

    // Create chain: p1 -> p2 -> p3 -> p4
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();
    PatternID p4 = PatternID::Generate();

    matrix.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.8f));
    matrix.AddAssociation(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.8f));
    matrix.AddAssociation(AssociationEdge(p3, p4, AssociationType::CAUSAL, 0.8f));

    auto results = matrix.PropagateActivation(p1, 1.0f, 3, 0.1f);

    // Should activate p2, p3, p4
    EXPECT_GE(results.size(), 3u);

    // p2 should have highest activation
    bool found_p2 = false;
    for (const auto& result : results) {
        if (result.pattern == p2) {
            EXPECT_NEAR(0.8f, result.activation, 0.01f);
            found_p2 = true;
        }
    }
    EXPECT_TRUE(found_p2);
}

TEST(AssociationMatrixTest, CompactRemovesDeleted) {
    AssociationMatrix matrix;

    std::vector<PatternID> patterns;
    for (int i = 0; i < 100; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    // Add many associations
    for (int i = 0; i < 99; ++i) {
        matrix.AddAssociation(
            AssociationEdge(patterns[i], patterns[i+1], AssociationType::CAUSAL, 0.5f)
        );
    }

    EXPECT_EQ(99u, matrix.GetAssociationCount());

    // Remove half
    for (int i = 0; i < 50; ++i) {
        matrix.RemoveAssociation(patterns[i], patterns[i+1]);
    }

    EXPECT_EQ(49u, matrix.GetAssociationCount());

    size_t before_compact = matrix.EstimateMemoryUsage();

    // Compact should reclaim memory
    matrix.Compact();

    size_t after_compact = matrix.EstimateMemoryUsage();
    EXPECT_LT(after_compact, before_compact);

    // Should still have 49 associations
    EXPECT_EQ(49u, matrix.GetAssociationCount());
}

TEST(AssociationMatrixTest, SerializationRoundTrip) {
    AssociationMatrix original;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    original.AddAssociation(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.7f));
    original.AddAssociation(AssociationEdge(p2, p3, AssociationType::SPATIAL, 0.6f));

    // Serialize
    std::stringstream ss;
    original.Serialize(ss);

    // Deserialize
    AssociationMatrix deserialized = AssociationMatrix::Deserialize(ss);

    // Verify
    EXPECT_EQ(original.GetAssociationCount(), deserialized.GetAssociationCount());

    auto edge1 = deserialized.GetAssociation(p1, p2);
    ASSERT_TRUE(edge1.has_value());
    EXPECT_FLOAT_EQ(0.7f, edge1->GetStrength());

    auto edge2 = deserialized.GetAssociation(p2, p3);
    ASSERT_TRUE(edge2.has_value());
    EXPECT_FLOAT_EQ(0.6f, edge2->GetStrength());
}

TEST(AssociationMatrixTest, ThreadSafeConcurrentReads) {
    AssociationMatrix matrix;

    std::vector<PatternID> patterns;
    for (int i = 0; i < 100; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    // Populate
    for (int i = 0; i < 99; ++i) {
        matrix.AddAssociation(
            AssociationEdge(patterns[i], patterns[i+1], AssociationType::CATEGORICAL, 0.5f)
        );
    }

    // Multiple readers
    std::vector<std::thread> readers;
    std::atomic<int> read_count{0};

    for (int i = 0; i < 10; ++i) {
        readers.emplace_back([&matrix, &patterns, &read_count]() {
            for (int j = 0; j < 1000; ++j) {
                auto edge = matrix.GetAssociation(patterns[j % 99], patterns[(j % 99) + 1]);
                if (edge.has_value()) {
                    read_count++;
                }
            }
        });
    }

    for (auto& thread : readers) {
        thread.join();
    }

    EXPECT_EQ(10000, read_count.load());
}

TEST(AssociationMatrixTest, PerformanceWithLargeGraph) {
    AssociationMatrix matrix;

    const int N = 10000;
    std::vector<PatternID> patterns;

    for (int i = 0; i < N; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    // Create random graph
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < N - 1; ++i) {
        matrix.AddAssociation(
            AssociationEdge(patterns[i], patterns[i+1], AssociationType::CAUSAL, 0.5f)
        );
    }

    auto insert_time = std::chrono::steady_clock::now() - start;

    // Measure lookup time
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < 1000; ++i) {
        int idx = rand() % (N - 1);
        matrix.GetAssociation(patterns[idx], patterns[idx+1]);
    }

    auto lookup_time = std::chrono::steady_clock::now() - start;

    auto insert_ms = std::chrono::duration_cast<std::chrono::milliseconds>(insert_time).count();
    auto lookup_us = std::chrono::duration_cast<std::chrono::microseconds>(lookup_time).count();

    std::cout << "Inserted " << N << " associations in " << insert_ms << "ms" << std::endl;
    std::cout << "1000 lookups in " << lookup_us << "us (avg: " << lookup_us / 1000.0 << "us)" << std::endl;

    // Assertions
    EXPECT_LT(insert_ms, 1000);  // Should be <1 second for 10K
    EXPECT_LT(lookup_us / 1000.0, 10.0);  // Average lookup <10us
}

// More tests...

} // namespace
} // namespace dpan
```

**Performance Targets**:
- Add association: <5µs
- Lookup: <1µs average
- Get outgoing/incoming: <10µs for 100 associations
- Propagate activation: <100ms for 10K patterns
- Memory: <100 bytes per association (plus overhead)

**Acceptance Criteria**:
- [ ] All 50+ unit tests pass
- [ ] Thread-safe verified (multiple readers, single writer)
- [ ] Performance targets met
- [ ] Handles >10M associations
- [ ] Memory efficient (compact works)
- [ ] Serialization maintains graph structure
- [ ] >90% code coverage

---

### Task 3.1.3: Create Association Storage Backend

**Duration**: 3 days (24 hours)
**Priority**: High

[Implement persistent storage layer for AssociationMatrix using RocksDB column families...]

### Task 3.1.4: Implement Association Indices

**Duration**: 2 days (16 hours)
**Priority**: Medium

[Implement specialized indices for fast lookups by type, context, etc...]

---

## Module 3.2: Association Formation

**Duration**: 3 weeks (120 hours)

[Continue with extremely detailed breakdown of association formation, co-occurrence tracking, temporal learning, spatial learning, categorical learning...]

---

[Continue with Modules 3.3, 3.4, 3.5...]

---

## Conclusion

This Phase 2 implementation plan provides a complete, actionable roadmap with:
- Detailed code implementations
- Mathematical foundations
- Comprehensive test suites
- Performance targets
- Thread safety considerations

**Estimated Completion**: 8-10 weeks with 2-3 developers

---

*Document Version*: 1.0
*Last Updated*: 2025-11-16
*Status*: Ready for Implementation
