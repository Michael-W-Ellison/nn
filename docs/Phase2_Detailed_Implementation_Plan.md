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
**Files to create**:
- `src/association/association_storage.hpp`
- `src/association/association_storage.cpp`
- `src/association/rocksdb_association_storage.hpp`
- `src/association/rocksdb_association_storage.cpp`
- `tests/association/association_storage_test.cpp`

#### Subtask 3.1.3.1: Design Storage Interface (4 hours)

**Overview**:
Provide persistent storage for AssociationMatrix to survive process restarts. The storage backend must support:
- Atomic batch writes
- Range queries by pattern ID
- Fast point lookups
- Column families for different indices
- Snapshot consistency

**Interface Design**:

```cpp
// File: src/association/association_storage.hpp
#pragma once

#include "association/association_edge.hpp"
#include <vector>
#include <memory>
#include <functional>

namespace dpan {

// Abstract storage interface for associations
class AssociationStorage {
public:
    virtual ~AssociationStorage() = default;

    // Write operations
    virtual bool Put(const AssociationEdge& edge) = 0;
    virtual bool Delete(PatternID source, PatternID target) = 0;

    // Batch operations
    virtual bool WriteBatch(const std::vector<AssociationEdge>& edges) = 0;

    // Read operations
    virtual std::optional<AssociationEdge> Get(PatternID source, PatternID target) const = 0;

    // Range queries
    virtual std::vector<AssociationEdge> GetBySource(PatternID source) const = 0;
    virtual std::vector<AssociationEdge> GetByTarget(PatternID target) const = 0;
    virtual std::vector<AssociationEdge> GetByType(AssociationType type) const = 0;

    // Iteration
    using EdgeCallback = std::function<bool(const AssociationEdge&)>;
    virtual void ForEach(EdgeCallback callback) const = 0;

    // Statistics
    virtual size_t Count() const = 0;
    virtual size_t EstimateSize() const = 0;

    // Maintenance
    virtual void Compact() = 0;
    virtual void Clear() = 0;

    // Snapshot
    virtual std::unique_ptr<AssociationStorage> CreateSnapshot() const = 0;
};

// Factory
class StorageConfig {
public:
    std::string db_path;
    bool enable_compression{true};
    bool enable_bloom_filters{true};
    size_t write_buffer_size{64 * 1024 * 1024};  // 64MB
    int max_open_files{1000};
};

std::unique_ptr<AssociationStorage> CreateRocksDBStorage(const StorageConfig& config);

} // namespace dpan
```

#### Subtask 3.1.3.2: Implement RocksDB Backend (16 hours)

**Column Family Design**:
- `edges`: Main storage (source:target -> AssociationEdge)
- `by_source`: Index (source -> list of targets)
- `by_target`: Index (target -> list of sources)
- `by_type`: Index (type -> list of edges)

**Key Encoding**:
```
Edge key: <source_id:8 bytes><target_id:8 bytes>
Source index key: <source_id:8 bytes><target_id:8 bytes>
Target index key: <target_id:8 bytes><source_id:8 bytes>
Type index key: <type:1 byte><source_id:8 bytes><target_id:8 bytes>
```

**Implementation**:

```cpp
// File: src/association/rocksdb_association_storage.hpp
#pragma once

#include "association/association_storage.hpp"
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <rocksdb/utilities/transaction_db.h>

namespace dpan {

class RocksDBAssociationStorage : public AssociationStorage {
public:
    explicit RocksDBAssociationStorage(const StorageConfig& config);
    ~RocksDBAssociationStorage() override;

    // AssociationStorage interface
    bool Put(const AssociationEdge& edge) override;
    bool Delete(PatternID source, PatternID target) override;
    bool WriteBatch(const std::vector<AssociationEdge>& edges) override;

    std::optional<AssociationEdge> Get(PatternID source, PatternID target) const override;
    std::vector<AssociationEdge> GetBySource(PatternID source) const override;
    std::vector<AssociationEdge> GetByTarget(PatternID target) const override;
    std::vector<AssociationEdge> GetByType(AssociationType type) const override;

    void ForEach(EdgeCallback callback) const override;
    size_t Count() const override;
    size_t EstimateSize() const override;

    void Compact() override;
    void Clear() override;

    std::unique_ptr<AssociationStorage> CreateSnapshot() const override;

private:
    std::unique_ptr<rocksdb::DB> db_;

    // Column family handles
    rocksdb::ColumnFamilyHandle* edges_cf_;
    rocksdb::ColumnFamilyHandle* by_source_cf_;
    rocksdb::ColumnFamilyHandle* by_target_cf_;
    rocksdb::ColumnFamilyHandle* by_type_cf_;

    // Key encoding helpers
    std::string EncodeEdgeKey(PatternID source, PatternID target) const;
    std::string EncodeSourceKey(PatternID source, PatternID target) const;
    std::string EncodeTargetKey(PatternID target, PatternID source) const;
    std::string EncodeTypeKey(AssociationType type, PatternID source, PatternID target) const;

    // Value encoding
    std::string EncodeEdge(const AssociationEdge& edge) const;
    AssociationEdge DecodeEdge(const std::string& value) const;

    // Helper methods
    void UpdateIndices(const AssociationEdge& edge, bool is_delete);
};

} // namespace dpan
```

**Complete Implementation**:

```cpp
// File: src/association/rocksdb_association_storage.cpp
#include "association/rocksdb_association_storage.hpp"
#include <rocksdb/write_batch.h>
#include <rocksdb/slice.h>
#include <sstream>

namespace dpan {

RocksDBAssociationStorage::RocksDBAssociationStorage(const StorageConfig& config) {
    rocksdb::Options options;
    options.create_if_missing = true;
    options.compression = config.enable_compression ?
        rocksdb::kSnappyCompression : rocksdb::kNoCompression;
    options.write_buffer_size = config.write_buffer_size;
    options.max_open_files = config.max_open_files;

    // Column family descriptors
    std::vector<rocksdb::ColumnFamilyDescriptor> column_families;
    column_families.push_back(rocksdb::ColumnFamilyDescriptor(
        rocksdb::kDefaultColumnFamilyName, rocksdb::ColumnFamilyOptions()));
    column_families.push_back(rocksdb::ColumnFamilyDescriptor(
        "edges", rocksdb::ColumnFamilyOptions()));
    column_families.push_back(rocksdb::ColumnFamilyDescriptor(
        "by_source", rocksdb::ColumnFamilyOptions()));
    column_families.push_back(rocksdb::ColumnFamilyDescriptor(
        "by_target", rocksdb::ColumnFamilyOptions()));
    column_families.push_back(rocksdb::ColumnFamilyDescriptor(
        "by_type", rocksdb::ColumnFamilyOptions()));

    // Open database with column families
    std::vector<rocksdb::ColumnFamilyHandle*> handles;
    rocksdb::DB* db_ptr;
    rocksdb::Status status = rocksdb::DB::Open(options, config.db_path, column_families, &handles, &db_ptr);

    if (!status.ok()) {
        // Try creating new database
        status = rocksdb::DB::Open(options, config.db_path, &db_ptr);
        if (!status.ok()) {
            throw std::runtime_error("Failed to open RocksDB: " + status.ToString());
        }

        // Create column families
        rocksdb::ColumnFamilyHandle* handle;
        db_ptr->CreateColumnFamily(rocksdb::ColumnFamilyOptions(), "edges", &handle);
        handles.push_back(handle);
        db_ptr->CreateColumnFamily(rocksdb::ColumnFamilyOptions(), "by_source", &handle);
        handles.push_back(handle);
        db_ptr->CreateColumnFamily(rocksdb::ColumnFamilyOptions(), "by_target", &handle);
        handles.push_back(handle);
        db_ptr->CreateColumnFamily(rocksdb::ColumnFamilyOptions(), "by_type", &handle);
        handles.push_back(handle);
    }

    db_.reset(db_ptr);
    edges_cf_ = handles[1];
    by_source_cf_ = handles[2];
    by_target_cf_ = handles[3];
    by_type_cf_ = handles[4];
}

RocksDBAssociationStorage::~RocksDBAssociationStorage() {
    if (db_) {
        delete edges_cf_;
        delete by_source_cf_;
        delete by_target_cf_;
        delete by_type_cf_;
    }
}

std::string RocksDBAssociationStorage::EncodeEdgeKey(PatternID source, PatternID target) const {
    std::string key;
    key.resize(16);
    uint64_t src_val = source.value();
    uint64_t tgt_val = target.value();
    std::memcpy(&key[0], &src_val, 8);
    std::memcpy(&key[8], &tgt_val, 8);
    return key;
}

std::string RocksDBAssociationStorage::EncodeEdge(const AssociationEdge& edge) const {
    std::ostringstream oss;
    edge.Serialize(oss);
    return oss.str();
}

AssociationEdge RocksDBAssociationStorage::DecodeEdge(const std::string& value) const {
    std::istringstream iss(value);
    return AssociationEdge::Deserialize(iss);
}

bool RocksDBAssociationStorage::Put(const AssociationEdge& edge) {
    rocksdb::WriteBatch batch;

    std::string edge_key = EncodeEdgeKey(edge.GetSource(), edge.GetTarget());
    std::string edge_value = EncodeEdge(edge);

    batch.Put(edges_cf_, edge_key, edge_value);

    // Update indices
    std::string source_key = EncodeSourceKey(edge.GetSource(), edge.GetTarget());
    batch.Put(by_source_cf_, source_key, "");

    std::string target_key = EncodeTargetKey(edge.GetTarget(), edge.GetSource());
    batch.Put(by_target_cf_, target_key, "");

    std::string type_key = EncodeTypeKey(edge.GetType(), edge.GetSource(), edge.GetTarget());
    batch.Put(by_type_cf_, type_key, "");

    rocksdb::WriteOptions write_options;
    rocksdb::Status status = db_->Write(write_options, &batch);

    return status.ok();
}

bool RocksDBAssociationStorage::Delete(PatternID source, PatternID target) {
    // First get the edge to know its type
    auto edge_opt = Get(source, target);
    if (!edge_opt.has_value()) {
        return false;
    }

    rocksdb::WriteBatch batch;

    std::string edge_key = EncodeEdgeKey(source, target);
    batch.Delete(edges_cf_, edge_key);

    std::string source_key = EncodeSourceKey(source, target);
    batch.Delete(by_source_cf_, source_key);

    std::string target_key = EncodeTargetKey(target, source);
    batch.Delete(by_target_cf_, target_key);

    std::string type_key = EncodeTypeKey(edge_opt->GetType(), source, target);
    batch.Delete(by_type_cf_, type_key);

    rocksdb::WriteOptions write_options;
    rocksdb::Status status = db_->Write(write_options, &batch);

    return status.ok();
}

std::optional<AssociationEdge> RocksDBAssociationStorage::Get(PatternID source, PatternID target) const {
    std::string key = EncodeEdgeKey(source, target);
    std::string value;

    rocksdb::ReadOptions read_options;
    rocksdb::Status status = db_->Get(read_options, edges_cf_, key, &value);

    if (!status.ok()) {
        return std::nullopt;
    }

    return DecodeEdge(value);
}

std::vector<AssociationEdge> RocksDBAssociationStorage::GetBySource(PatternID source) const {
    std::vector<AssociationEdge> results;

    std::string prefix;
    prefix.resize(8);
    uint64_t src_val = source.value();
    std::memcpy(&prefix[0], &src_val, 8);

    rocksdb::ReadOptions read_options;
    read_options.iterate_upper_bound = new rocksdb::Slice(prefix);

    rocksdb::Iterator* it = db_->NewIterator(read_options, by_source_cf_);

    for (it->Seek(prefix); it->Valid(); it->Next()) {
        // Extract target from key
        std::string key = it->key().ToString();
        if (key.size() != 16) continue;

        uint64_t tgt_val;
        std::memcpy(&tgt_val, &key[8], 8);
        PatternID target(tgt_val);

        auto edge = Get(source, target);
        if (edge.has_value()) {
            results.push_back(*edge);
        }
    }

    delete it;
    delete read_options.iterate_upper_bound;

    return results;
}

void RocksDBAssociationStorage::Compact() {
    rocksdb::CompactRangeOptions options;
    db_->CompactRange(options, edges_cf_, nullptr, nullptr);
    db_->CompactRange(options, by_source_cf_, nullptr, nullptr);
    db_->CompactRange(options, by_target_cf_, nullptr, nullptr);
    db_->CompactRange(options, by_type_cf_, nullptr, nullptr);
}

size_t RocksDBAssociationStorage::Count() const {
    size_t count = 0;
    rocksdb::Iterator* it = db_->NewIterator(rocksdb::ReadOptions(), edges_cf_);
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        count++;
    }
    delete it;
    return count;
}

// Remaining methods...

} // namespace dpan
```

**Unit Tests** (15+ tests):

```cpp
// File: tests/association/association_storage_test.cpp
#include "association/rocksdb_association_storage.hpp"
#include <gtest/gtest.h>
#include <filesystem>

namespace dpan {
namespace {

class AssociationStorageTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/dpan_storage_test_" + std::to_string(time(nullptr));
        std::filesystem::create_directories(test_dir_);

        StorageConfig config;
        config.db_path = test_dir_;
        storage_ = CreateRocksDBStorage(config);
    }

    void TearDown() override {
        storage_.reset();
        std::filesystem::remove_all(test_dir_);
    }

    std::string test_dir_;
    std::unique_ptr<AssociationStorage> storage_;
};

TEST_F(AssociationStorageTest, PutAndGet) {
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge edge(p1, p2, AssociationType::CAUSAL, 0.7f);

    EXPECT_TRUE(storage_->Put(edge));

    auto retrieved = storage_->Get(p1, p2);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(p1, retrieved->GetSource());
    EXPECT_EQ(p2, retrieved->GetTarget());
    EXPECT_FLOAT_EQ(0.7f, retrieved->GetStrength());
}

TEST_F(AssociationStorageTest, PersistenceAcrossRestarts) {
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    AssociationEdge edge(p1, p2, AssociationType::SPATIAL, 0.9f);
    storage_->Put(edge);

    // Close and reopen
    storage_.reset();
    StorageConfig config;
    config.db_path = test_dir_;
    storage_ = CreateRocksDBStorage(config);

    auto retrieved = storage_->Get(p1, p2);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_FLOAT_EQ(0.9f, retrieved->GetStrength());
}

TEST_F(AssociationStorageTest, GetBySource) {
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    storage_->Put(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    storage_->Put(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.6f));

    auto edges = storage_->GetBySource(p1);
    EXPECT_EQ(2u, edges.size());
}

// More tests...

} // namespace
} // namespace dpan
```

**Acceptance Criteria**:
- [ ] RocksDB integration works correctly
- [ ] Persistence across restarts verified
- [ ] All indices updated atomically
- [ ] >15 unit tests pass
- [ ] Performance: <100µs writes, <50µs reads
- [ ] Handles millions of associations
- [ ] Compaction reduces disk usage

---

### Task 3.1.4: Implement Association Indices

**Duration**: 2 days (16 hours)
**Priority**: Medium
**Files to create**:
- `src/association/association_index.hpp`
- `src/association/association_index.cpp`
- `src/association/type_index.hpp`
- `src/association/context_index.hpp`
- `tests/association/association_index_test.cpp`

#### Subtask 3.1.4.1: Design Index System (4 hours)

**Specialized Indices**:

1. **Type Index**: Fast lookup by AssociationType
2. **Context Index**: Find associations matching context profiles
3. **Strength Index**: Range queries by strength threshold
4. **Temporal Index**: Associations by age or last reinforcement

**Interface**:

```cpp
// File: src/association/association_index.hpp
#pragma once

#include "association/association_edge.hpp"
#include <vector>
#include <set>

namespace dpan {

// Base index interface
class AssociationIndex {
public:
    virtual ~AssociationIndex() = default;

    virtual void Insert(const AssociationEdge& edge) = 0;
    virtual void Remove(PatternID source, PatternID target) = 0;
    virtual void Update(const AssociationEdge& edge) = 0;

    virtual std::vector<AssociationEdge> Query(const void* query_params) const = 0;

    virtual void Clear() = 0;
    virtual size_t Size() const = 0;
};

// Type-based index
class TypeIndex : public AssociationIndex {
public:
    void Insert(const AssociationEdge& edge) override;
    void Remove(PatternID source, PatternID target) override;
    void Update(const AssociationEdge& edge) override;

    std::vector<AssociationEdge> GetByType(AssociationType type) const;
    std::vector<AssociationEdge> Query(const void* query_params) const override;

    void Clear() override;
    size_t Size() const override;

private:
    std::unordered_map<AssociationType, std::set<std::pair<PatternID, PatternID>>> type_map_;
    std::unordered_map<std::pair<PatternID, PatternID>, AssociationEdge, PatternPairHash> edge_map_;
};

// Strength-based index (range queries)
class StrengthIndex : public AssociationIndex {
public:
    void Insert(const AssociationEdge& edge) override;
    void Remove(PatternID source, PatternID target) override;
    void Update(const AssociationEdge& edge) override;

    // Get associations with strength >= threshold
    std::vector<AssociationEdge> GetAboveThreshold(float threshold) const;

    // Get top-k strongest associations
    std::vector<AssociationEdge> GetTopK(size_t k) const;

    std::vector<AssociationEdge> Query(const void* query_params) const override;

    void Clear() override;
    size_t Size() const override;

private:
    // Multimap: strength -> (source, target)
    std::multimap<float, std::pair<PatternID, PatternID>, std::greater<float>> strength_map_;
    std::unordered_map<std::pair<PatternID, PatternID>, AssociationEdge, PatternPairHash> edge_map_;
};

// Context-aware index
class ContextIndex : public AssociationIndex {
public:
    void Insert(const AssociationEdge& edge) override;
    void Remove(PatternID source, PatternID target) override;
    void Update(const AssociationEdge& edge) override;

    // Find associations matching context (similarity >= threshold)
    std::vector<AssociationEdge> FindByContext(
        const ContextVector& context,
        float similarity_threshold = 0.7f
    ) const;

    std::vector<AssociationEdge> Query(const void* query_params) const override;

    void Clear() override;
    size_t Size() const override;

private:
    // Simple implementation: linear scan with caching
    // Future: LSH or other approximate nearest neighbor structure
    std::vector<AssociationEdge> edges_;
    mutable std::unordered_map<ContextVector, std::vector<AssociationEdge>> cache_;
};

} // namespace dpan
```

#### Subtask 3.1.4.2: Implement Indices (12 hours)

**Type Index Implementation**:

```cpp
// File: src/association/type_index.cpp
#include "association/association_index.hpp"

namespace dpan {

void TypeIndex::Insert(const AssociationEdge& edge) {
    auto key = std::make_pair(edge.GetSource(), edge.GetTarget());
    type_map_[edge.GetType()].insert(key);
    edge_map_[key] = edge;
}

void TypeIndex::Remove(PatternID source, PatternID target) {
    auto key = std::make_pair(source, target);
    auto it = edge_map_.find(key);
    if (it != edge_map_.end()) {
        type_map_[it->second.GetType()].erase(key);
        edge_map_.erase(it);
    }
}

void TypeIndex::Update(const AssociationEdge& edge) {
    Remove(edge.GetSource(), edge.GetTarget());
    Insert(edge);
}

std::vector<AssociationEdge> TypeIndex::GetByType(AssociationType type) const {
    std::vector<AssociationEdge> results;

    auto it = type_map_.find(type);
    if (it != type_map_.end()) {
        results.reserve(it->second.size());
        for (const auto& key : it->second) {
            auto edge_it = edge_map_.find(key);
            if (edge_it != edge_map_.end()) {
                results.push_back(edge_it->second);
            }
        }
    }

    return results;
}

std::vector<AssociationEdge> TypeIndex::Query(const void* query_params) const {
    auto type = *static_cast<const AssociationType*>(query_params);
    return GetByType(type);
}

void TypeIndex::Clear() {
    type_map_.clear();
    edge_map_.clear();
}

size_t TypeIndex::Size() const {
    return edge_map_.size();
}

} // namespace dpan
```

**Strength Index Implementation**:

```cpp
// File: src/association/strength_index.cpp
#include "association/association_index.hpp"

namespace dpan {

void StrengthIndex::Insert(const AssociationEdge& edge) {
    auto key = std::make_pair(edge.GetSource(), edge.GetTarget());
    strength_map_.insert({edge.GetStrength(), key});
    edge_map_[key] = edge;
}

void StrengthIndex::Remove(PatternID source, PatternID target) {
    auto key = std::make_pair(source, target);
    auto it = edge_map_.find(key);
    if (it != edge_map_.end()) {
        float strength = it->second.GetStrength();

        // Find and remove from multimap
        auto range = strength_map_.equal_range(strength);
        for (auto map_it = range.first; map_it != range.second; ++map_it) {
            if (map_it->second == key) {
                strength_map_.erase(map_it);
                break;
            }
        }

        edge_map_.erase(it);
    }
}

void StrengthIndex::Update(const AssociationEdge& edge) {
    Remove(edge.GetSource(), edge.GetTarget());
    Insert(edge);
}

std::vector<AssociationEdge> StrengthIndex::GetAboveThreshold(float threshold) const {
    std::vector<AssociationEdge> results;

    for (auto it = strength_map_.begin(); it != strength_map_.end(); ++it) {
        if (it->first < threshold) break;  // Map is sorted descending

        auto edge_it = edge_map_.find(it->second);
        if (edge_it != edge_map_.end()) {
            results.push_back(edge_it->second);
        }
    }

    return results;
}

std::vector<AssociationEdge> StrengthIndex::GetTopK(size_t k) const {
    std::vector<AssociationEdge> results;
    results.reserve(std::min(k, strength_map_.size()));

    auto it = strength_map_.begin();
    for (size_t i = 0; i < k && it != strength_map_.end(); ++i, ++it) {
        auto edge_it = edge_map_.find(it->second);
        if (edge_it != edge_map_.end()) {
            results.push_back(edge_it->second);
        }
    }

    return results;
}

} // namespace dpan
```

**Unit Tests**:

```cpp
// File: tests/association/association_index_test.cpp
#include "association/association_index.hpp"
#include <gtest/gtest.h>

namespace dpan {
namespace {

TEST(TypeIndexTest, InsertAndQuery) {
    TypeIndex index;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    index.Insert(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.5f));
    index.Insert(AssociationEdge(p2, p3, AssociationType::CAUSAL, 0.6f));
    index.Insert(AssociationEdge(p1, p3, AssociationType::SPATIAL, 0.7f));

    auto causal = index.GetByType(AssociationType::CAUSAL);
    EXPECT_EQ(2u, causal.size());

    auto spatial = index.GetByType(AssociationType::SPATIAL);
    EXPECT_EQ(1u, spatial.size());
}

TEST(StrengthIndexTest, TopKQuery) {
    StrengthIndex index;

    std::vector<PatternID> patterns;
    for (int i = 0; i < 10; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    // Insert with varying strengths
    for (int i = 0; i < 9; ++i) {
        float strength = 0.1f * (i + 1);
        index.Insert(AssociationEdge(patterns[i], patterns[i+1],
                                     AssociationType::CATEGORICAL, strength));
    }

    auto top3 = index.GetTopK(3);
    EXPECT_EQ(3u, top3.size());

    // Should be in descending order
    EXPECT_GE(top3[0].GetStrength(), top3[1].GetStrength());
    EXPECT_GE(top3[1].GetStrength(), top3[2].GetStrength());
}

TEST(StrengthIndexTest, ThresholdQuery) {
    StrengthIndex index;

    std::vector<PatternID> patterns;
    for (int i = 0; i < 10; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    for (int i = 0; i < 9; ++i) {
        float strength = 0.1f * (i + 1);
        index.Insert(AssociationEdge(patterns[i], patterns[i+1],
                                     AssociationType::FUNCTIONAL, strength));
    }

    auto strong = index.GetAboveThreshold(0.7f);
    EXPECT_EQ(3u, strong.size());  // 0.7, 0.8, 0.9
}

// More tests...

} // namespace
} // namespace dpan
```

**Acceptance Criteria**:
- [ ] Type index: O(1) lookup by type
- [ ] Strength index: Top-k in O(k) time
- [ ] Context index: Similarity search works
- [ ] >20 unit tests pass
- [ ] Indices stay synchronized with matrix
- [ ] Memory overhead <20% of edge data

---

## Module 3.2: Association Formation

**Duration**: 3 weeks (120 hours)
**Dependencies**: Module 3.1 complete
**Owner**: Lead ML engineer + C++ developer

### Overview
This module implements the algorithms that automatically discover and create associations between patterns based on co-occurrence, temporal relationships, spatial proximity, and categorical similarity.

### Key Concepts:
- **Co-occurrence**: Patterns appearing together within a time window
- **Temporal Association**: Causal relationships (A precedes B)
- **Spatial Association**: Patterns appearing in similar locations/contexts
- **Categorical Association**: Patterns that cluster together
- **Functional Association**: Patterns serving similar roles

---

### Task 3.2.1: Implement Co-occurrence Tracker

**Duration**: 3 days (24 hours)
**Priority**: Critical
**Files to create**:
- `src/association/co_occurrence_tracker.hpp`
- `src/association/co_occurrence_tracker.cpp`
- `tests/association/co_occurrence_tracker_test.cpp`

#### Subtask 3.2.1.1: Design Temporal Window System (6 hours)

**Mathematical Foundation**:

Co-occurrence within temporal window `W`:
```
Let P = {p1, p2, ..., pn} be pattern activations
Let t(pi) be activation timestamp of pattern pi

Co-occur(pi, pj, W) = 1 if |t(pi) - t(pj)| <= W
                       0 otherwise

Co-occurrence count:
C(pi, pj) = Σ Co-occur(pi, pj, W) over all time
```

**Sliding Window Implementation**:

```cpp
// File: src/association/co_occurrence_tracker.hpp
#pragma once

#include "core/types.hpp"
#include <deque>
#include <unordered_map>

namespace dpan {

// Tracks pattern co-occurrences within temporal windows
class CoOccurrenceTracker {
public:
    struct Config {
        Timestamp::Duration window_size{std::chrono::seconds(10)};
        uint32_t min_co_occurrences{3};  // Minimum to form association
        float significance_threshold{0.05f};  // Chi-squared p-value
    };

    explicit CoOccurrenceTracker(const Config& config = {});

    // Record pattern activation
    void RecordActivation(PatternID pattern, Timestamp timestamp = Timestamp::Now());

    // Batch record
    void RecordActivations(const std::vector<PatternID>& patterns, Timestamp timestamp = Timestamp::Now());

    // Query co-occurrence statistics
    uint32_t GetCoOccurrenceCount(PatternID p1, PatternID p2) const;
    float GetCoOccurrenceProbability(PatternID p1, PatternID p2) const;

    // Statistical significance test
    bool IsSignificant(PatternID p1, PatternID p2) const;
    float GetChiSquared(PatternID p1, PatternID p2) const;

    // Get all significant co-occurrences for a pattern
    std::vector<std::pair<PatternID, uint32_t>> GetCoOccurringPatterns(
        PatternID pattern,
        uint32_t min_count = 0
    ) const;

    // Maintenance
    void PruneOldActivations(Timestamp cutoff_time);
    void Clear();

    // Statistics
    size_t GetActivationCount() const { return activations_.size(); }
    size_t GetUniquePatternCount() const;

private:
    Config config_;

    // Activation history (timestamp, pattern) sorted by time
    std::deque<std::pair<Timestamp, PatternID>> activations_;

    // Co-occurrence matrix
    std::unordered_map<std::pair<PatternID, PatternID>, uint32_t, PatternPairHash> co_occurrence_counts_;

    // Pattern activation counts (for probability calculation)
    std::unordered_map<PatternID, uint32_t> pattern_counts_;

    // Total number of windows processed
    uint64_t total_windows_{0};

    // Helper methods
    void UpdateCoOccurrences(const std::vector<PatternID>& patterns_in_window);
    std::vector<PatternID> GetPatternsInWindow(Timestamp start, Timestamp end) const;
};

} // namespace dpan
```

#### Subtask 3.2.1.2: Implement Core Tracking Logic (12 hours)

```cpp
// File: src/association/co_occurrence_tracker.cpp
#include "association/co_occurrence_tracker.hpp"
#include <algorithm>
#include <cmath>

namespace dpan {

CoOccurrenceTracker::CoOccurrenceTracker(const Config& config)
    : config_(config)
{
}

void CoOccurrenceTracker::RecordActivation(PatternID pattern, Timestamp timestamp) {
    // Add to activation history
    activations_.push_back({timestamp, pattern});
    pattern_counts_[pattern]++;

    // Get all patterns in the current window
    Timestamp window_start = timestamp - config_.window_size;
    auto patterns_in_window = GetPatternsInWindow(window_start, timestamp);

    // Update co-occurrences
    UpdateCoOccurrences(patterns_in_window);

    // Prune old activations (older than 2x window size to save memory)
    Timestamp cutoff = timestamp - (config_.window_size * 2);
    PruneOldActivations(cutoff);
}

void CoOccurrenceTracker::RecordActivations(
    const std::vector<PatternID>& patterns,
    Timestamp timestamp
) {
    for (const auto& pattern : patterns) {
        activations_.push_back({timestamp, pattern});
        pattern_counts_[pattern]++;
    }

    // Update co-occurrences for this window
    UpdateCoOccurrences(patterns);
    total_windows_++;
}

std::vector<PatternID> CoOccurrenceTracker::GetPatternsInWindow(
    Timestamp start,
    Timestamp end
) const {
    std::vector<PatternID> result;

    // Binary search for start position
    auto it_start = std::lower_bound(
        activations_.begin(),
        activations_.end(),
        start,
        [](const auto& activation, Timestamp time) {
            return activation.first < time;
        }
    );

    // Collect patterns in window
    for (auto it = it_start; it != activations_.end() && it->first <= end; ++it) {
        result.push_back(it->second);
    }

    return result;
}

void CoOccurrenceTracker::UpdateCoOccurrences(const std::vector<PatternID>& patterns_in_window) {
    // Create unique set
    std::unordered_set<PatternID> unique_patterns(patterns_in_window.begin(), patterns_in_window.end());

    // Update co-occurrence for all pairs
    std::vector<PatternID> pattern_vec(unique_patterns.begin(), unique_patterns.end());

    for (size_t i = 0; i < pattern_vec.size(); ++i) {
        for (size_t j = i + 1; j < pattern_vec.size(); ++j) {
            PatternID p1 = pattern_vec[i];
            PatternID p2 = pattern_vec[j];

            // Always store in consistent order (smaller ID first)
            if (p2 < p1) std::swap(p1, p2);

            auto key = std::make_pair(p1, p2);
            co_occurrence_counts_[key]++;
        }
    }
}

uint32_t CoOccurrenceTracker::GetCoOccurrenceCount(PatternID p1, PatternID p2) const {
    if (p2 < p1) std::swap(p1, p2);

    auto key = std::make_pair(p1, p2);
    auto it = co_occurrence_counts_.find(key);

    return it != co_occurrence_counts_.end() ? it->second : 0;
}

float CoOccurrenceTracker::GetCoOccurrenceProbability(PatternID p1, PatternID p2) const {
    uint32_t co_count = GetCoOccurrenceCount(p1, p2);

    if (co_count == 0) return 0.0f;

    // P(p1, p2) = count(p1 AND p2) / total_windows
    if (total_windows_ == 0) return 0.0f;

    return static_cast<float>(co_count) / total_windows_;
}

bool CoOccurrenceTracker::IsSignificant(PatternID p1, PatternID p2) const {
    float chi_squared = GetChiSquared(p1, p2);

    // Chi-squared critical value for df=1, alpha=0.05 is 3.841
    return chi_squared > 3.841f;
}

float CoOccurrenceTracker::GetChiSquared(PatternID p1, PatternID p2) const {
    if (total_windows_ == 0) return 0.0f;

    // Contingency table:
    // |  p2  | !p2  |
    // | p1   | a    | b    |
    // | !p1  | c    | d    |

    uint32_t a = GetCoOccurrenceCount(p1, p2);  // Both present

    auto it1 = pattern_counts_.find(p1);
    auto it2 = pattern_counts_.find(p2);

    if (it1 == pattern_counts_.end() || it2 == pattern_counts_.end()) {
        return 0.0f;
    }

    uint32_t p1_count = it1->second;
    uint32_t p2_count = it2->second;

    uint32_t b = p1_count - a;  // p1 without p2
    uint32_t c = p2_count - a;  // p2 without p1
    uint32_t d = total_windows_ - a - b - c;  // Neither

    uint32_t n = total_windows_;

    // Chi-squared formula
    float numerator = n * (a * d - b * c) * (a * d - b * c);
    float denominator = (a + b) * (c + d) * (a + c) * (b + d);

    if (denominator == 0) return 0.0f;

    return numerator / denominator;
}

std::vector<std::pair<PatternID, uint32_t>> CoOccurrenceTracker::GetCoOccurringPatterns(
    PatternID pattern,
    uint32_t min_count
) const {
    std::vector<std::pair<PatternID, uint32_t>> results;

    for (const auto& [key, count] : co_occurrence_counts_) {
        if (count < min_count) continue;

        if (key.first == pattern) {
            results.push_back({key.second, count});
        } else if (key.second == pattern) {
            results.push_back({key.first, count});
        }
    }

    // Sort by count (descending)
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

    return results;
}

void CoOccurrenceTracker::PruneOldActivations(Timestamp cutoff_time) {
    while (!activations_.empty() && activations_.front().first < cutoff_time) {
        activations_.pop_front();
    }
}

void CoOccurrenceTracker::Clear() {
    activations_.clear();
    co_occurrence_counts_.clear();
    pattern_counts_.clear();
    total_windows_ = 0;
}

size_t CoOccurrenceTracker::GetUniquePatternCount() const {
    return pattern_counts_.size();
}

} // namespace dpan
```

**Unit Tests** (20+ tests):

```cpp
// File: tests/association/co_occurrence_tracker_test.cpp
#include "association/co_occurrence_tracker.hpp"
#include <gtest/gtest.h>

namespace dpan {
namespace {

TEST(CoOccurrenceTrackerTest, BasicCoOccurrence) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t1 = Timestamp::Now();
    Timestamp t2 = t1 + std::chrono::seconds(1);

    tracker.RecordActivation(p1, t1);
    tracker.RecordActivation(p2, t2);

    // Should co-occur within default window
    EXPECT_GT(tracker.GetCoOccurrenceCount(p1, p2), 0u);
}

TEST(CoOccurrenceTrackerTest, OutsideWindow) {
    CoOccurrenceTracker::Config config;
    config.window_size = std::chrono::seconds(5);
    CoOccurrenceTracker tracker(config);

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp t1 = Timestamp::Now();
    Timestamp t2 = t1 + std::chrono::seconds(10);  // Outside window

    tracker.RecordActivation(p1, t1);
    tracker.RecordActivation(p2, t2);

    // Should NOT co-occur
    EXPECT_EQ(0u, tracker.GetCoOccurrenceCount(p1, p2));
}

TEST(CoOccurrenceTrackerTest, BatchRecording) {
    CoOccurrenceTracker tracker;

    std::vector<PatternID> patterns;
    for (int i = 0; i < 5; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    tracker.RecordActivations(patterns);

    // All pairs should co-occur
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = i + 1; j < patterns.size(); ++j) {
            EXPECT_EQ(1u, tracker.GetCoOccurrenceCount(patterns[i], patterns[j]));
        }
    }
}

TEST(CoOccurrenceTrackerTest, ChiSquaredSignificance) {
    CoOccurrenceTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    // Create strong association between p1 and p2
    for (int i = 0; i < 100; ++i) {
        tracker.RecordActivations({p1, p2});
    }

    // p3 appears alone
    for (int i = 0; i < 100; ++i) {
        tracker.RecordActivations({p3});
    }

    // p1-p2 should be significant
    EXPECT_TRUE(tracker.IsSignificant(p1, p2));

    // p1-p3 should not be significant
    EXPECT_FALSE(tracker.IsSignificant(p1, p3));
}

// More tests...

} // namespace
} // namespace dpan
```

**Acceptance Criteria**:
- [ ] Temporal window tracking works correctly
- [ ] Co-occurrence counts accurate
- [ ] Chi-squared test identifies significant co-occurrences
- [ ] >20 unit tests pass
- [ ] Performance: <1µs per activation
- [ ] Memory: O(W × P) where W=window size, P=patterns
- [ ] Old activations pruned automatically

---

### Task 3.2.2: Implement Association Formation Rules

**Duration**: 3 days (24 hours)
**Priority**: Critical
**Files to create**:
- `src/association/formation_rules.hpp`
- `src/association/formation_rules.cpp`
- `tests/association/formation_rules_test.cpp`

#### Subtask 3.2.2.1: Define Formation Criteria (6 hours)

**Formation Rules**:

1. **Threshold-Based Formation**:
   - Co-occurrence count >= `min_co_occurrences`
   - Statistical significance (chi-squared test)
   - Temporal correlation >= `min_correlation`

2. **Type Classification**:
   - **Causal**: p1 consistently precedes p2
   - **Spatial**: p1 and p2 appear in similar contexts
   - **Categorical**: p1 and p2 cluster together
   - **Functional**: p1 and p2 serve similar roles
   - **Compositional**: p1 contains p2 as component

**Implementation**:

```cpp
// File: src/association/formation_rules.hpp
#pragma once

#include "association/association_edge.hpp"
#include "association/co_occurrence_tracker.hpp"
#include "core/pattern_node.hpp"

namespace dpan {

class AssociationFormationRules {
public:
    struct Config {
        uint32_t min_co_occurrences{5};
        float min_chi_squared{3.841f};  // p < 0.05
        float min_temporal_correlation{0.3f};
        float min_spatial_similarity{0.7f};
        float min_categorical_similarity{0.6f};
        float initial_strength{0.5f};
    };

    explicit AssociationFormationRules(const Config& config = {});

    // Evaluate if association should be formed
    bool ShouldFormAssociation(
        const CoOccurrenceTracker& tracker,
        PatternID p1,
        PatternID p2
    ) const;

    // Determine association type
    AssociationType ClassifyAssociationType(
        const PatternNode& pattern1,
        const PatternNode& pattern2,
        const std::vector<std::pair<Timestamp, PatternID>>& activation_sequence
    ) const;

    // Calculate initial strength
    float CalculateInitialStrength(
        const CoOccurrenceTracker& tracker,
        PatternID p1,
        PatternID p2,
        AssociationType type
    ) const;

    // Create association from co-occurrence data
    std::optional<AssociationEdge> CreateAssociation(
        const CoOccurrenceTracker& tracker,
        const PatternNode& pattern1,
        const PatternNode& pattern2,
        const std::vector<std::pair<Timestamp, PatternID>>& activation_sequence
    ) const;

private:
    Config config_;

    // Type classification helpers
    bool IsCausal(
        PatternID p1,
        PatternID p2,
        const std::vector<std::pair<Timestamp, PatternID>>& sequence
    ) const;

    bool IsSpatial(const PatternNode& p1, const PatternNode& p2) const;
    bool IsCategorical(const PatternNode& p1, const PatternNode& p2) const;
    bool IsFunctional(const PatternNode& p1, const PatternNode& p2) const;
    bool IsCompositional(const PatternNode& p1, const PatternNode& p2) const;
};

} // namespace dpan
```

#### Subtask 3.2.2.2: Implement Classification Logic (18 hours)

```cpp
// File: src/association/formation_rules.cpp
#include "association/formation_rules.hpp"
#include <algorithm>

namespace dpan {

AssociationFormationRules::AssociationFormationRules(const Config& config)
    : config_(config)
{
}

bool AssociationFormationRules::ShouldFormAssociation(
    const CoOccurrenceTracker& tracker,
    PatternID p1,
    PatternID p2
) const {
    // Check minimum co-occurrence count
    uint32_t co_count = tracker.GetCoOccurrenceCount(p1, p2);
    if (co_count < config_.min_co_occurrences) {
        return false;
    }

    // Check statistical significance
    if (!tracker.IsSignificant(p1, p2)) {
        return false;
    }

    float chi_squared = tracker.GetChiSquared(p1, p2);
    if (chi_squared < config_.min_chi_squared) {
        return false;
    }

    return true;
}

AssociationType AssociationFormationRules::ClassifyAssociationType(
    const PatternNode& pattern1,
    const PatternNode& pattern2,
    const std::vector<std::pair<Timestamp, PatternID>>& activation_sequence
) const {
    // Try to classify in order of specificity

    // 1. Compositional (most specific)
    if (IsCompositional(pattern1, pattern2)) {
        return AssociationType::COMPOSITIONAL;
    }

    // 2. Causal (requires temporal data)
    if (IsCausal(pattern1.GetID(), pattern2.GetID(), activation_sequence)) {
        return AssociationType::CAUSAL;
    }

    // 3. Functional (patterns serve similar role)
    if (IsFunctional(pattern1, pattern2)) {
        return AssociationType::FUNCTIONAL;
    }

    // 4. Spatial (appear in similar contexts)
    if (IsSpatial(pattern1, pattern2)) {
        return AssociationType::SPATIAL;
    }

    // 5. Categorical (default fallback)
    return AssociationType::CATEGORICAL;
}

bool AssociationFormationRules::IsCausal(
    PatternID p1,
    PatternID p2,
    const std::vector<std::pair<Timestamp, PatternID>>& sequence
) const {
    // Count how often p1 precedes p2
    int p1_before_p2 = 0;
    int p2_before_p1 = 0;

    for (size_t i = 0; i < sequence.size(); ++i) {
        if (sequence[i].second == p1) {
            // Look ahead for p2
            for (size_t j = i + 1; j < sequence.size(); ++j) {
                if (sequence[j].second == p2) {
                    p1_before_p2++;
                    break;
                }
                if (sequence[j].second == p1) break;  // Another p1
            }
        } else if (sequence[i].second == p2) {
            // Look ahead for p1
            for (size_t j = i + 1; j < sequence.size(); ++j) {
                if (sequence[j].second == p1) {
                    p2_before_p1++;
                    break;
                }
                if (sequence[j].second == p2) break;  // Another p2
            }
        }
    }

    // Causal if one direction is significantly more common
    int total = p1_before_p2 + p2_before_p1;
    if (total == 0) return false;

    float ratio = static_cast<float>(std::max(p1_before_p2, p2_before_p1)) / total;
    return ratio >= 0.7f;  // 70% in one direction
}

bool AssociationFormationRules::IsSpatial(
    const PatternNode& p1,
    const PatternNode& p2
) const {
    // Check if patterns have similar context profiles
    // (This assumes PatternNode has context information)

    // Placeholder: Check if patterns appear in similar spatial contexts
    // In reality, would compare spatial features from pattern data
    return false;  // Implement based on pattern structure
}

bool AssociationFormationRules::IsCategorical(
    const PatternNode& p1,
    const PatternNode& p2
) const {
    // Check if patterns belong to same cluster/category
    // Could use similarity metrics from Phase 1

    // Placeholder implementation
    return true;  // Default type
}

bool AssociationFormationRules::IsFunctional(
    const PatternNode& p1,
    const PatternNode& p2
) const {
    // Patterns are functional if they serve similar purposes
    // in different contexts

    // Check if patterns have similar association profiles
    // (similar outgoing/incoming associations)

    return false;  // Implement based on association profiles
}

bool AssociationFormationRules::IsCompositional(
    const PatternNode& p1,
    const PatternNode& p2
) const {
    // Check if one pattern contains the other as sub-pattern

    const auto& p1_subs = p1.GetSubPatterns();
    const auto& p2_subs = p2.GetSubPatterns();

    // p1 contains p2?
    if (std::find(p1_subs.begin(), p1_subs.end(), p2.GetID()) != p1_subs.end()) {
        return true;
    }

    // p2 contains p1?
    if (std::find(p2_subs.begin(), p2_subs.end(), p1.GetID()) != p2_subs.end()) {
        return true;
    }

    return false;
}

float AssociationFormationRules::CalculateInitialStrength(
    const CoOccurrenceTracker& tracker,
    PatternID p1,
    PatternID p2,
    AssociationType type
) const {
    // Base strength from co-occurrence probability
    float prob = tracker.GetCoOccurrenceProbability(p1, p2);

    // Normalize to [0, 1] range
    float base_strength = std::min(1.0f, prob * 10.0f);

    // Boost based on statistical significance
    float chi_squared = tracker.GetChiSquared(p1, p2);
    float significance_boost = std::min(0.3f, chi_squared / 100.0f);

    // Type-specific adjustments
    float type_factor = 1.0f;
    if (type == AssociationType::CAUSAL || type == AssociationType::COMPOSITIONAL) {
        type_factor = 1.2f;  // Boost stronger association types
    }

    float strength = (base_strength + significance_boost) * type_factor;
    return std::clamp(strength, 0.0f, 1.0f);
}

std::optional<AssociationEdge> AssociationFormationRules::CreateAssociation(
    const CoOccurrenceTracker& tracker,
    const PatternNode& pattern1,
    const PatternNode& pattern2,
    const std::vector<std::pair<Timestamp, PatternID>>& activation_sequence
) const {
    PatternID p1 = pattern1.GetID();
    PatternID p2 = pattern2.GetID();

    // Check formation criteria
    if (!ShouldFormAssociation(tracker, p1, p2)) {
        return std::nullopt;
    }

    // Classify type
    AssociationType type = ClassifyAssociationType(pattern1, pattern2, activation_sequence);

    // Calculate strength
    float strength = CalculateInitialStrength(tracker, p1, p2, type);

    // Create edge
    AssociationEdge edge(p1, p2, type, strength);

    // Set co-occurrence count
    edge.IncrementCoOccurrence(tracker.GetCoOccurrenceCount(p1, p2));

    return edge;
}

} // namespace dpan
```

**Unit Tests**:

```cpp
// File: tests/association/formation_rules_test.cpp
#include "association/formation_rules.hpp"
#include <gtest/gtest.h>

namespace dpan {
namespace {

TEST(FormationRulesTest, ShouldFormWithSufficientCoOccurrence) {
    AssociationFormationRules::Config config;
    config.min_co_occurrences = 5;
    AssociationFormationRules rules(config);

    CoOccurrenceTracker tracker;
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Record enough co-occurrences
    for (int i = 0; i < 10; ++i) {
        tracker.RecordActivations({p1, p2});
    }

    EXPECT_TRUE(rules.ShouldFormAssociation(tracker, p1, p2));
}

TEST(FormationRulesTest, ShouldNotFormWithInsufficientData) {
    AssociationFormationRules::Config config;
    config.min_co_occurrences = 10;
    AssociationFormationRules rules(config);

    CoOccurrenceTracker tracker;
    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Only a few co-occurrences
    for (int i = 0; i < 3; ++i) {
        tracker.RecordActivations({p1, p2});
    }

    EXPECT_FALSE(rules.ShouldFormAssociation(tracker, p1, p2));
}

TEST(FormationRulesTest, CausalClassification) {
    AssociationFormationRules rules;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    // Create sequence where p1 always precedes p2
    std::vector<std::pair<Timestamp, PatternID>> sequence;
    Timestamp t = Timestamp::Now();

    for (int i = 0; i < 10; ++i) {
        sequence.push_back({t, p1});
        t = t + std::chrono::seconds(1);
        sequence.push_back({t, p2});
        t = t + std::chrono::seconds(5);
    }

    // Should classify as causal
    PatternNode node1(p1);
    PatternNode node2(p2);

    AssociationType type = rules.ClassifyAssociationType(node1, node2, sequence);
    EXPECT_EQ(AssociationType::CAUSAL, type);
}

// More tests...

} // namespace
} // namespace dpan
```

**Acceptance Criteria**:
- [ ] Formation rules correctly identify valid associations
- [ ] Type classification works for all 5 types
- [ ] Initial strength calculation is reasonable
- [ ] Statistical significance prevents spurious associations
- [ ] >15 unit tests pass
- [ ] Integration with CoOccurrenceTracker works

### Task 3.2.3: Implement Temporal Association Learning

**Duration**: 3 days (24 hours)
**Priority**: High
**Files to create**:
- `src/association/temporal_learner.hpp`
- `src/association/temporal_learner.cpp`
- `tests/association/temporal_learner_test.cpp`

**Overview**:
Learn causal relationships by analyzing temporal sequences of pattern activations. Detect consistent ordering patterns and compute temporal correlation coefficients.

**Key Algorithm - Temporal Correlation**:

```
For patterns p1, p2, compute temporal correlation:
  - Track time delays: Δt = t(p2) - t(p1)
  - Compute mean delay: μ = mean(Δt)
  - Compute std deviation: σ = std(Δt)
  - Correlation strength: τ = 1 / (1 + σ/μ)  (consistency measure)
```

**Implementation Summary** (see files):
- `TemporalLearner` class with sequence analysis
- Sliding window for temporal pattern detection
- Mean/variance computation for delay distribution
- Temporal correlation coefficient calculation
- Integration with AssociationFormationRules

**Acceptance Criteria**:
- [ ] Detects causal sequences correctly
- [ ] Temporal correlation values in [0,1]
- [ ] Handles noisy data (variable delays)
- [ ] >15 unit tests pass
- [ ] Performance: <10µs per sequence update

---

### Task 3.2.4: Implement Spatial Association Learning

**Duration**: 3 days (24 hours)
**Priority**: High
**Files to create**:
- `src/association/spatial_learner.hpp`
- `src/association/spatial_learner.cpp`
- `tests/association/spatial_learner_test.cpp`

**Overview**:
Learn spatial relationships by analyzing context similarity between pattern activations.

**Key Algorithm - Spatial Clustering**:

```cpp
class SpatialLearner {
public:
    struct SpatialContext {
        ContextVector context;
        Timestamp timestamp;
        std::vector<PatternID> co_occurring_patterns;
    };

    // Learn spatial associations
    void RecordSpatialContext(PatternID pattern, const ContextVector& context);

    // Check if two patterns have similar spatial profiles
    bool AreSpatiallyRelated(PatternID p1, PatternID p2, float threshold = 0.7f);

    // Get average context for a pattern
    ContextVector GetAverageContext(PatternID pattern) const;
};
```

**Acceptance Criteria**:
- [ ] Context similarity accurately computed
- [ ] Spatial associations identified correctly
- [ ] Context averaging works properly
- [ ] >12 unit tests pass

---

### Task 3.2.5: Implement Categorical Association Learning

**Duration**: 3 days (24 hours)
**Priority**: Medium
**Files to create**:
- `src/association/categorical_learner.hpp`
- `src/association/categorical_learner.cpp`
- `tests/association/categorical_learner_test.cpp`

**Overview**:
Learn categorical relationships by clustering patterns with similar features and activation patterns.

**Key Algorithm - Pattern Clustering**:

```cpp
class CategoricalLearner {
public:
    // Cluster patterns based on feature similarity
    void ComputeClusters(const std::vector<PatternNode>& patterns, size_t k_clusters);

    // Check if patterns belong to same category
    bool AreCategoricallyRelated(PatternID p1, PatternID p2) const;

    // Get cluster ID for a pattern
    std::optional<size_t> GetClusterID(PatternID pattern) const;

private:
    // K-means clustering
    std::vector<std::vector<PatternID>> clusters_;
    std::unordered_map<PatternID, size_t> pattern_to_cluster_;
};
```

**Acceptance Criteria**:
- [ ] Clustering identifies similar patterns
- [ ] Categorical associations form within clusters
- [ ] >10 unit tests pass
- [ ] Integration with formation rules works

---

## Module 3.3: Association Strength Management

**Duration**: 2 weeks (80 hours)
**Dependencies**: Module 3.2 complete
**Owner**: C++ developer + ML engineer

### Overview
Manages dynamic strengthening, weakening, and decay of associations based on usage, reinforcement, and time.

---

### Task 3.3.1: Implement Reinforcement Learning

**Duration**: 3 days (24 hours)
**Priority**: Critical
**Files to create**:
- `src/association/reinforcement_manager.hpp`
- `src/association/reinforcement_manager.cpp`

**Overview**:
Strengthen associations when patterns successfully predict each other's activation.

**Key Algorithm - Hebbian Learning**:

```
When p1 predicts p2 correctly:
  Δs = η × (1 - s) × prediction_accuracy

Where:
  s = current strength
  η = learning rate (0.1)
  prediction_accuracy ∈ [0,1]
```

**Implementation**:

```cpp
class ReinforcementManager {
public:
    struct Config {
        float learning_rate{0.1f};
        float decay_rate{0.01f};
        float min_strength{0.1f};
        float max_strength{1.0f};
    };

    // Reinforce association (strengthen)
    void Reinforce(AssociationEdge& edge, float reward);

    // Weaken association (punishment)
    void Weaken(AssociationEdge& edge, float penalty);

    // Apply time-based decay
    void ApplyDecay(AssociationEdge& edge, Timestamp::Duration elapsed);

    // Batch reinforcement
    void ReinforceBatch(AssociationMatrix& matrix,
                       const std::vector<std::pair<PatternID, PatternID>>& pairs,
                       float reward);
};
```

**Acceptance Criteria**:
- [ ] Reinforcement correctly strengthens edges
- [ ] Weakening correctly reduces strength
- [ ] Strength bounded in [min, max]
- [ ] >20 unit tests pass

---

### Task 3.3.2: Implement Decay Mechanisms

**Duration**: 2 days (16 hours)
**Priority**: High

**Overview**:
Apply exponential decay to unused associations to prevent stale data.

**Decay Function**:
```
s(t) = s(0) × exp(-d × t)

Where:
  s(t) = strength at time t
  s(0) = initial strength
  d = decay rate
  t = time since last reinforcement
```

**Implementation in AssociationMatrix**:

```cpp
void AssociationMatrix::ApplyDecayAll(Timestamp::Duration elapsed) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    for (auto& edge : edges_) {
        if (!edge.GetSource().IsValid()) continue;

        // Apply decay
        edge.ApplyDecay(elapsed);

        // Remove if strength too low
        if (edge.GetStrength() < 0.05f) {
            MarkForPruning(edge);
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Decay applied correctly over time
- [ ] Weak associations pruned automatically
- [ ] >10 unit tests pass

---

### Task 3.3.3: Implement Strength Normalization

**Duration**: 2 days (16 hours)
**Priority**: Medium

**Overview**:
Normalize outgoing association strengths to prevent strength inflation.

**Normalization Algorithm**:

```
For pattern p with outgoing associations {a1, a2, ..., an}:

s'_i = s_i / Σ(s_j) for all j

This ensures Σ(s'_i) = 1.0
```

**Implementation**:

```cpp
void NormalizeOutgoingStrengths(AssociationMatrix& matrix, PatternID pattern) {
    auto outgoing = matrix.GetOutgoingAssociations(pattern);

    float total = 0.0f;
    for (const auto& edge : outgoing) {
        total += edge.GetStrength();
    }

    if (total > 0.0f) {
        for (const auto& edge : outgoing) {
            float normalized = edge.GetStrength() / total;
            matrix.UpdateStrength(edge.GetSource(), edge.GetTarget(), normalized);
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Normalization preserves relative strengths
- [ ] Sum of outgoing strengths = 1.0
- [ ] >8 unit tests pass

---

### Task 3.3.4: Implement Competitive Learning

**Duration**: 2 days (16 hours)
**Priority**: Medium

**Overview**:
Implement winner-take-all competition where strong associations suppress weaker ones.

**Competition Algorithm**:

```
For competing associations from p1 to {p2, p3, ..., pn}:

1. Find strongest: s_max = max(s_i)
2. Boost winner: s_max = s_max + β × (1 - s_max)
3. Suppress others: s_i = s_i × (1 - β) for i ≠ max
```

**Acceptance Criteria**:
- [ ] Strongest associations become stronger
- [ ] Weaker associations suppressed
- [ ] Competition parameter tunable
- [ ] >10 unit tests pass

---

## Module 3.4: Activation Propagation

**Duration**: 2 weeks (80 hours)
**Dependencies**: Modules 3.1-3.3 complete
**Owner**: Algorithms specialist + C++ developer

### Overview
Implements efficient algorithms for spreading activation through the association network.

---

### Task 3.4.1: Implement BFS Propagation

**Duration**: 3 days (24 hours)
**Priority**: Critical
**Files to create**:
- `src/association/activation_propagator.hpp`
- `src/association/activation_propagator.cpp`
- `tests/association/activation_propagator_test.cpp`

**Algorithm** (already implemented in AssociationMatrix::PropagateActivation):

```cpp
class ActivationPropagator {
public:
    struct PropagationResult {
        std::unordered_map<PatternID, float> activations;
        std::vector<std::pair<PatternID, PatternID>> traversed_edges;
        size_t hops_used;
        std::chrono::microseconds elapsed_time;
    };

    PropagationResult Propagate(
        const AssociationMatrix& matrix,
        PatternID source,
        float initial_activation,
        size_t max_hops = 3,
        float min_activation = 0.01f,
        const ContextVector* context = nullptr
    );

private:
    // BFS implementation with activation accumulation
    void BreadthFirstPropagate(/* ... */);
};
```

**Optimizations**:
- Early termination when activation < threshold
- Context-aware strength modulation
- Visited set to prevent cycles
- Activation accumulation at targets

**Acceptance Criteria**:
- [ ] BFS correctly explores graph
- [ ] Activation accumulates properly
- [ ] Max hops limit respected
- [ ] >20 unit tests pass
- [ ] Performance: <100ms for 10K patterns

---

### Task 3.4.2: Implement Parallel Propagation

**Duration**: 3 days (24 hours)
**Priority**: High

**Overview**:
Parallelize activation propagation using thread pools for large graphs.

**Implementation**:

```cpp
class ParallelPropagator {
public:
    struct Config {
        size_t num_threads{std::thread::hardware_concurrency()};
        size_t batch_size{1000};
    };

    // Parallel BFS with thread pool
    PropagationResult PropagateParallel(
        const AssociationMatrix& matrix,
        const std::vector<PatternID>& sources,
        float initial_activation
    );

private:
    std::vector<std::thread> thread_pool_;
    std::queue<PropagationTask> task_queue_;
};
```

**Acceptance Criteria**:
- [ ] Speedup >2x for large graphs
- [ ] Thread-safe operation
- [ ] No race conditions
- [ ] >15 unit tests pass

---

### Task 3.4.3: Implement Spreading Activation with Decay

**Duration**: 2 days (16 hours)
**Priority**: Medium

**Overview**:
Implement activation decay as it spreads through the network.

**Decay Formula**:

```
a_target = a_source × s × (1 - decay_factor)^hop_distance

Where:
  a = activation level
  s = association strength
  decay_factor = 0.2 (configurable)
```

**Acceptance Criteria**:
- [ ] Activation decays with distance
- [ ] Configurable decay rate
- [ ] >10 unit tests pass

---

### Task 3.4.4: Implement Bidirectional Propagation

**Duration**: 2 days (16 hours)
**Priority**: Low

**Overview**:
Propagate activation both forward (outgoing) and backward (incoming) associations.

**Acceptance Criteria**:
- [ ] Both directions explored
- [ ] Separate activation levels tracked
- [ ] >8 unit tests pass

---

## Module 3.5: Integration & Testing

**Duration**: 1 week (40 hours)
**Dependencies**: All previous modules complete
**Owner**: Integration lead + QA engineer

### Overview
Integrate all components into a unified AssociationLearningSystem and perform comprehensive testing.

---

### Task 3.5.1: Create Unified Association Learning System

**Duration**: 3 days (24 hours)
**Priority**: Critical
**Files to create**:
- `src/association/association_learning_system.hpp`
- `src/association/association_learning_system.cpp`
- `tests/association/integration_test.cpp`

**System Architecture**:

```cpp
class AssociationLearningSystem {
public:
    struct Config {
        CoOccurrenceTracker::Config co_occurrence;
        AssociationFormationRules::Config formation;
        ReinforcementManager::Config reinforcement;
        size_t association_capacity{1000000};
    };

    explicit AssociationLearningSystem(const Config& config = {});

    // Record pattern activation and update associations
    void RecordPatternActivation(PatternID pattern, const ContextVector& context);

    // Batch record
    void RecordPatternActivations(const std::vector<PatternID>& patterns,
                                   const ContextVector& context);

    // Trigger association formation
    void FormNewAssociations(const PatternDatabase& pattern_db);

    // Apply reinforcement learning
    void Reinforce(PatternID predicted, PatternID actual, bool correct);

    // Maintenance operations
    void ApplyDecay(Timestamp::Duration elapsed);
    void PruneWeakAssociations(float min_strength = 0.05f);
    void Compact();

    // Query operations
    const AssociationMatrix& GetAssociationMatrix() const { return matrix_; }
    std::vector<AssociationEdge> GetAssociations(PatternID pattern) const;
    std::vector<PatternID> Predict(PatternID pattern, size_t k = 5) const;

    // Statistics
    size_t GetAssociationCount() const;
    float GetAverageStrength() const;
    void PrintStatistics(std::ostream& out) const;

private:
    Config config_;

    AssociationMatrix matrix_;
    CoOccurrenceTracker tracker_;
    AssociationFormationRules formation_rules_;
    ReinforcementManager reinforcement_mgr_;
    ActivationPropagator propagator_;

    // Activation history for temporal learning
    std::deque<std::pair<Timestamp, PatternID>> activation_history_;
};
```

**Acceptance Criteria**:
- [ ] All components integrated correctly
- [ ] End-to-end workflow functional
- [ ] >30 integration tests pass
- [ ] Performance targets met

---

### Task 3.5.2: Comprehensive Testing Suite

**Duration**: 2 days (16 hours)
**Priority**: Critical

**Test Categories**:

1. **Unit Tests**: Component-level (already implemented in each task)
2. **Integration Tests**: Multi-component interactions
3. **Performance Tests**: Benchmarks for all operations
4. **Stress Tests**: Large-scale data (millions of associations)
5. **Concurrency Tests**: Thread safety verification

**Example Integration Test**:

```cpp
TEST(AssociationLearningSystemTest, EndToEndWorkflow) {
    AssociationLearningSystem system;
    PatternDatabase pattern_db;

    // Create test patterns
    std::vector<PatternID> patterns;
    for (int i = 0; i < 100; ++i) {
        patterns.push_back(pattern_db.CreatePattern(/* ... */));
    }

    // Simulate pattern activations
    for (int iter = 0; iter < 1000; ++iter) {
        // Activate correlated patterns together
        std::vector<PatternID> batch = {patterns[iter % 10], patterns[(iter % 10) + 1]};
        system.RecordPatternActivations(batch, ContextVector());
    }

    // Form associations
    system.FormNewAssociations(pattern_db);

    // Verify associations formed
    EXPECT_GT(system.GetAssociationCount(), 0u);

    // Test prediction
    auto predictions = system.Predict(patterns[0], 5);
    EXPECT_FALSE(predictions.empty());

    // Apply decay
    system.ApplyDecay(std::chrono::hours(24));

    // Prune weak associations
    system.PruneWeakAssociations(0.1f);
}
```

**Acceptance Criteria**:
- [ ] >200 total unit tests pass
- [ ] >30 integration tests pass
- [ ] >20 performance benchmarks pass
- [ ] >95% code coverage
- [ ] Zero memory leaks (valgrind verified)
- [ ] Thread-safe verified (ThreadSanitizer)

---

### Task 3.5.3: Documentation & Examples

**Duration**: 1 day (8 hours)
**Priority**: Medium

**Deliverables**:

1. **API Documentation**: Doxygen-style comments for all public APIs
2. **Usage Examples**:
   - `examples/association/basic_learning.cpp`
   - `examples/association/custom_formation_rules.cpp`
   - `examples/association/activation_propagation_demo.cpp`
3. **Performance Guide**: Optimization tips and tuning parameters
4. **Integration Guide**: How to integrate with Phase 1 Pattern Engine

**Acceptance Criteria**:
- [ ] All public APIs documented
- [ ] 3+ working examples provided
- [ ] Documentation builds without warnings
- [ ] Examples compile and run successfully

---

## Mathematical Foundations

### Association Strength Dynamics

**Reinforcement Learning Update**:
```
s(t+1) = s(t) + η × δ

Where:
  δ = r - s(t)  (prediction error)
  r = reward (1 if prediction correct, 0 otherwise)
  η = learning rate
```

**Exponential Decay**:
```
s(t) = s(0) × e^(-λt)

Where:
  λ = decay constant
  t = time since last reinforcement
```

**Competitive Learning**:
```
Winner: s_winner = s_winner + α × (1 - s_winner)
Losers: s_i = s_i × (1 - α) for i ≠ winner

Where α = competition factor (0.1-0.3)
```

---

## Performance Optimization Guide

### Memory Optimization

1. **Sparse Storage**: Use CSR format for association matrix
2. **Index Reuse**: Reuse deleted edge indices
3. **Compaction**: Regularly compact to reclaim memory
4. **Batch Operations**: Amortize lock overhead

### CPU Optimization

1. **Cache Locality**: Store edges contiguously
2. **SIMD**: Use vectorized operations for batch updates
3. **Parallel Processing**: Thread pool for large-scale operations
4. **Lock-Free Reads**: Use shared_mutex for concurrent reads

### I/O Optimization

1. **Batch Writes**: Group RocksDB writes
2. **Bloom Filters**: Reduce disk seeks
3. **Compression**: Enable Snappy compression
4. **Write Buffering**: Large write buffers (64MB+)

---

## Validation & Quality Assurance

### Code Quality Metrics

- **Code Coverage**: Target >90%
- **Cyclomatic Complexity**: Max 15 per function
- **Function Length**: Max 50 lines (guidance, not strict)
- **Memory Safety**: Zero leaks, zero invalid accesses

### Performance Targets

| Operation | Target | Measured |
|-----------|--------|----------|
| Add Association | <5µs | TBD |
| Lookup Association | <1µs | TBD |
| Propagate (10K patterns) | <100ms | TBD |
| Form Associations (100 patterns) | <10ms | TBD |
| Apply Decay (1M associations) | <500ms | TBD |
| Memory per Association | <100 bytes | TBD |

### Testing Strategy

1. **Development**: Unit tests for each component
2. **Integration**: End-to-end workflows
3. **Performance**: Benchmark suite
4. **Stress**: Large-scale data (10M+ associations)
5. **Regression**: Automated test suite on every commit

---

## Troubleshooting & Debugging

### Common Issues

**Issue**: Associations not forming
- **Check**: Co-occurrence counts >= threshold
- **Check**: Statistical significance test passing
- **Solution**: Lower `min_co_occurrences` or `min_chi_squared`

**Issue**: Memory usage growing unbounded
- **Check**: Decay being applied regularly
- **Check**: Pruning weak associations
- **Solution**: Enable automatic compaction, lower prune threshold

**Issue**: Slow activation propagation
- **Check**: Graph size and connectivity
- **Check**: Max hops parameter
- **Solution**: Reduce max_hops, use parallel propagator, prune weak edges

**Issue**: Thread safety violations
- **Check**: Proper use of shared_mutex
- **Check**: Atomic operations for counters
- **Solution**: Run with ThreadSanitizer, add missing locks

### Debugging Tools

- **Valgrind**: Memory leak detection
- **ThreadSanitizer**: Data race detection
- **Perf**: CPU profiling
- **Heaptrack**: Memory allocation profiling
- **GDB**: General debugging

### Logging

Use structured logging for key events:

```cpp
LOG(INFO) << "Formed association: " << edge.ToString();
LOG(WARNING) << "Association strength near zero: " << edge.GetStrength();
LOG(DEBUG) << "Propagation activated " << results.size() << " patterns";
```

---

## Conclusion

This Phase 2 implementation plan provides a complete, actionable roadmap with:

- **Detailed code implementations** for all core components
- **Mathematical foundations** for association learning algorithms
- **Comprehensive test suites** (200+ tests total)
- **Performance targets** with optimization guidance
- **Thread safety** with proper synchronization primitives
- **Scalability** to handle millions of associations
- **Integration guide** with Phase 1 Pattern Engine

### Estimated Completion

**Duration**: 8-10 weeks with 2-3 developers

**Weekly Milestones**:
- **Weeks 1-2**: Module 3.1 (Association Data Structures)
- **Weeks 3-5**: Module 3.2 (Association Formation)
- **Weeks 6-7**: Module 3.3 (Strength Management)
- **Week 8**: Module 3.4 (Activation Propagation)
- **Week 9-10**: Module 3.5 (Integration & Testing)

### Success Criteria

- [ ] All 200+ unit tests pass
- [ ] All performance targets met
- [ ] >90% code coverage
- [ ] Zero memory leaks
- [ ] Thread-safe verified
- [ ] Documentation complete
- [ ] Integration with Phase 1 working

---

**Document Version**: 1.0
**Last Updated**: 2025-11-16
**Status**: Ready for Implementation
**Total Lines of Code (estimated)**: ~8,000 C++ (~4,000 implementation + ~4,000 tests)

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
