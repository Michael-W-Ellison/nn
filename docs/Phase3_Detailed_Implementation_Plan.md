# Phase 3: Memory Management
## Extremely Detailed Implementation Plan

### Document Overview
This document provides a comprehensive, step-by-step implementation guide for Phase 3 of the DPAN project: Memory Management. Every task is broken down into granular sub-tasks with specific code examples, algorithms, mathematical formulations, testing requirements, and detailed acceptance criteria.

**Phase Duration**: 6-8 weeks (240-320 hours)
**Team Size**: 2-3 developers
**Prerequisites**: Phase 2 (Association Learning System) complete

---

## Table of Contents
1. [Phase Overview](#phase-overview)
2. [Module 4.1: Utility Scoring System](#module-41-utility-scoring-system)
3. [Module 4.2: Memory Hierarchy](#module-42-memory-hierarchy)
4. [Module 4.3: Pruning System](#module-43-pruning-system)
5. [Module 4.4: Forgetting Mechanisms](#module-44-forgetting-mechanisms)
6. [Module 4.5: Integration & Testing](#module-45-integration--testing)
7. [Mathematical Foundations](#mathematical-foundations)
8. [Performance Optimization Guide](#performance-optimization-guide)
9. [Validation & Quality Assurance](#validation--quality-assurance)
10. [Troubleshooting & Debugging](#troubleshooting--debugging)

---

## Phase Overview

### Goals
- Implement intelligent memory management to handle millions of patterns efficiently
- Create multi-tier storage system (RAM → SSD → Disk → Archive)
- Develop utility-based pruning to maintain high-quality patterns
- Implement biologically-inspired forgetting mechanisms
- Achieve scalable memory usage with graceful degradation under pressure

### Success Criteria
- [ ] Memory usage stays bounded under continuous learning
- [ ] Tier transitions transparent to upper layers
- [ ] Pruning maintains >95% of important patterns
- [ ] Access latency: Active tier <100ns, Warm tier <10µs, Cold tier <1ms
- [ ] Support >100M patterns with <32GB RAM
- [ ] >90% code coverage for all components
- [ ] Zero memory leaks (valgrind verified)
- [ ] Thread-safe concurrent operations

### Key Metrics Dashboard
Track these metrics daily:
- Total memory usage (by tier)
- Pattern count (by tier)
- Association count (by tier)
- Tier transition rate
- Pruning rate and accuracy
- Access latency percentiles (p50, p95, p99)
- Cache hit rate

---

## Module 4.1: Utility Scoring System

**Duration**: 2 weeks (80 hours)
**Dependencies**: Phase 2 complete
**Owner**: ML engineer + C++ developer

### Overview
The utility scoring system determines which patterns and associations are most valuable to keep in fast memory. It combines multiple factors including access frequency, recency, association strength, and confidence.

### Key Concepts:
- **Utility Score**: Composite metric indicating pattern/association value
- **Adaptive Thresholds**: Dynamic cutoffs based on memory pressure
- **Temporal Weighting**: Recent accesses weighted higher
- **Importance Propagation**: Utility flows through association graph

---

### Task 4.1.1: Implement Utility Calculator

**Duration**: 4 days (32 hours)
**Priority**: Critical
**Files to create**:
- `src/memory/utility_calculator.hpp`
- `src/memory/utility_calculator.cpp`
- `tests/memory/utility_calculator_test.cpp`

#### Subtask 4.1.1.1: Define Utility Metrics (8 hours)

**Mathematical Foundation**:

Utility score `U` for a pattern `p`:

```
U(p) = w_f × F(p) + w_r × R(p) + w_a × A(p) + w_c × C(p)

Where:
  F(p) = access frequency score ∈ [0,1]
  R(p) = recency score ∈ [0,1]
  A(p) = association strength score ∈ [0,1]
  C(p) = confidence score ∈ [0,1]
  w_f, w_r, w_a, w_c = weights (sum to 1.0)

Default weights:
  w_f = 0.3 (frequency)
  w_r = 0.3 (recency)
  w_a = 0.25 (associations)
  w_c = 0.15 (confidence)
```

**Individual Score Calculations**:

1. **Frequency Score** (based on access count):
```
F(p) = 1 - exp(-λ_f × access_count(p))

Where λ_f = 0.01 (decay constant)
This saturates at ~1.0 for frequently accessed patterns
```

2. **Recency Score** (exponential decay since last access):
```
R(p) = exp(-λ_r × Δt)

Where:
  Δt = time since last access (in hours)
  λ_r = 0.05 (decay rate)

Half-life: t_1/2 = ln(2)/λ_r ≈ 14 hours
```

3. **Association Strength Score** (connectivity importance):
```
A(p) = (Σ s_in + Σ s_out) / (n_in + n_out + 1)

Where:
  s_in = incoming association strengths
  s_out = outgoing association strengths
  n_in, n_out = number of in/out associations
```

4. **Confidence Score** (pattern quality):
```
C(p) = pattern.GetConfidence()

Derived from similarity metrics in Phase 1
```

**Implementation**:

```cpp
// File: src/memory/utility_calculator.hpp
#pragma once

#include "core/pattern_node.hpp"
#include "association/association_edge.hpp"
#include <chrono>
#include <unordered_map>

namespace dpan {

// Utility calculation for patterns and associations
class UtilityCalculator {
public:
    struct Config {
        // Weight parameters (must sum to 1.0)
        float frequency_weight{0.3f};
        float recency_weight{0.3f};
        float association_weight{0.25f};
        float confidence_weight{0.15f};

        // Decay constants
        float frequency_decay{0.01f};      // λ_f
        float recency_decay{0.05f};        // λ_r (per hour)

        // Normalization parameters
        float max_access_count{1000.0f};   // For normalization
    };

    explicit UtilityCalculator(const Config& config = {});

    // Pattern utility calculation
    float CalculatePatternUtility(
        const PatternNode& pattern,
        const AccessStats& stats,
        const std::vector<AssociationEdge>& associations
    ) const;

    // Association utility calculation
    float CalculateAssociationUtility(
        const AssociationEdge& edge,
        const AccessStats& source_stats,
        const AccessStats& target_stats
    ) const;

    // Component scores (for debugging/analysis)
    struct UtilityBreakdown {
        float frequency_score;
        float recency_score;
        float association_score;
        float confidence_score;
        float total;
    };

    UtilityBreakdown GetUtilityBreakdown(
        const PatternNode& pattern,
        const AccessStats& stats,
        const std::vector<AssociationEdge>& associations
    ) const;

    // Update configuration
    void SetConfig(const Config& config);
    const Config& GetConfig() const { return config_; }

private:
    Config config_;

    // Individual component calculations
    float CalculateFrequencyScore(uint64_t access_count) const;
    float CalculateRecencyScore(Timestamp::Duration time_since_access) const;
    float CalculateAssociationScore(const std::vector<AssociationEdge>& associations) const;
    float CalculateConfidenceScore(const PatternNode& pattern) const;

    // Validation
    void ValidateWeights() const;
};

// Track access statistics for patterns
struct AccessStats {
    uint64_t access_count{0};
    Timestamp last_access;
    Timestamp creation_time;

    // Exponential moving average of access interval
    float avg_access_interval{0.0f};  // in seconds

    void RecordAccess(Timestamp timestamp = Timestamp::Now());
    Timestamp::Duration TimeSinceLastAccess() const;
    Timestamp::Duration Age() const;

    // Serialization
    void Serialize(std::ostream& out) const;
    static AccessStats Deserialize(std::istream& in);
};

// Centralized access tracking
class AccessTracker {
public:
    void RecordPatternAccess(PatternID pattern, Timestamp timestamp = Timestamp::Now());
    void RecordAssociationAccess(PatternID source, PatternID target, Timestamp timestamp = Timestamp::Now());

    const AccessStats* GetPatternStats(PatternID pattern) const;
    const AccessStats* GetAssociationStats(PatternID source, PatternID target) const;

    // Bulk operations
    void PruneOldStats(Timestamp cutoff_time);
    void Clear();

    // Statistics
    size_t GetTrackedPatternCount() const { return pattern_stats_.size(); }
    size_t GetTrackedAssociationCount() const { return association_stats_.size(); }

private:
    std::unordered_map<PatternID, AccessStats> pattern_stats_;
    std::unordered_map<std::pair<PatternID, PatternID>, AccessStats, PatternPairHash> association_stats_;

    mutable std::shared_mutex mutex_;
};

} // namespace dpan
```

#### Subtask 4.1.1.2: Implement Core Calculations (16 hours)

```cpp
// File: src/memory/utility_calculator.cpp
#include "memory/utility_calculator.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace dpan {

UtilityCalculator::UtilityCalculator(const Config& config)
    : config_(config)
{
    ValidateWeights();
}

void UtilityCalculator::ValidateWeights() const {
    float total = config_.frequency_weight + config_.recency_weight +
                  config_.association_weight + config_.confidence_weight;

    if (std::abs(total - 1.0f) > 0.01f) {
        throw std::invalid_argument("Utility weights must sum to 1.0");
    }
}

float UtilityCalculator::CalculatePatternUtility(
    const PatternNode& pattern,
    const AccessStats& stats,
    const std::vector<AssociationEdge>& associations
) const {
    float freq_score = CalculateFrequencyScore(stats.access_count);
    float rec_score = CalculateRecencyScore(stats.TimeSinceLastAccess());
    float assoc_score = CalculateAssociationScore(associations);
    float conf_score = CalculateConfidenceScore(pattern);

    float utility =
        config_.frequency_weight * freq_score +
        config_.recency_weight * rec_score +
        config_.association_weight * assoc_score +
        config_.confidence_weight * conf_score;

    return std::clamp(utility, 0.0f, 1.0f);
}

float UtilityCalculator::CalculateFrequencyScore(uint64_t access_count) const {
    // Exponential saturation: F(n) = 1 - exp(-λ × n)
    float normalized = static_cast<float>(access_count) / config_.max_access_count;
    return 1.0f - std::exp(-config_.frequency_decay * access_count);
}

float UtilityCalculator::CalculateRecencyScore(Timestamp::Duration time_since_access) const {
    // Exponential decay: R(t) = exp(-λ × t)
    auto hours = std::chrono::duration_cast<std::chrono::hours>(time_since_access).count();
    return std::exp(-config_.recency_decay * hours);
}

float UtilityCalculator::CalculateAssociationScore(
    const std::vector<AssociationEdge>& associations
) const {
    if (associations.empty()) {
        return 0.0f;
    }

    // Average of association strengths
    float total_strength = 0.0f;
    for (const auto& edge : associations) {
        total_strength += edge.GetStrength();
    }

    return total_strength / associations.size();
}

float UtilityCalculator::CalculateConfidenceScore(const PatternNode& pattern) const {
    return pattern.GetConfidence();
}

UtilityCalculator::UtilityBreakdown UtilityCalculator::GetUtilityBreakdown(
    const PatternNode& pattern,
    const AccessStats& stats,
    const std::vector<AssociationEdge>& associations
) const {
    UtilityBreakdown breakdown;

    breakdown.frequency_score = CalculateFrequencyScore(stats.access_count);
    breakdown.recency_score = CalculateRecencyScore(stats.TimeSinceLastAccess());
    breakdown.association_score = CalculateAssociationScore(associations);
    breakdown.confidence_score = CalculateConfidenceScore(pattern);

    breakdown.total =
        config_.frequency_weight * breakdown.frequency_score +
        config_.recency_weight * breakdown.recency_score +
        config_.association_weight * breakdown.association_score +
        config_.confidence_weight * breakdown.confidence_score;

    return breakdown;
}

float UtilityCalculator::CalculateAssociationUtility(
    const AssociationEdge& edge,
    const AccessStats& source_stats,
    const AccessStats& target_stats
) const {
    // Association utility based on:
    // 1. Edge strength
    // 2. Source and target recency
    // 3. Source and target access frequency

    float edge_strength = edge.GetStrength();

    float source_recency = CalculateRecencyScore(source_stats.TimeSinceLastAccess());
    float target_recency = CalculateRecencyScore(target_stats.TimeSinceLastAccess());
    float avg_recency = (source_recency + target_recency) / 2.0f;

    float source_freq = CalculateFrequencyScore(source_stats.access_count);
    float target_freq = CalculateFrequencyScore(target_stats.access_count);
    float avg_freq = (source_freq + target_freq) / 2.0f;

    // Weighted combination
    float utility =
        0.5f * edge_strength +
        0.3f * avg_recency +
        0.2f * avg_freq;

    return std::clamp(utility, 0.0f, 1.0f);
}

// AccessStats implementation
void AccessStats::RecordAccess(Timestamp timestamp) {
    if (access_count > 0) {
        // Update average access interval (exponential moving average)
        auto interval = std::chrono::duration_cast<std::chrono::seconds>(
            timestamp - last_access
        ).count();

        float alpha = 0.3f;  // EMA smoothing factor
        avg_access_interval = alpha * interval + (1.0f - alpha) * avg_access_interval;
    } else {
        creation_time = timestamp;
    }

    last_access = timestamp;
    access_count++;
}

Timestamp::Duration AccessStats::TimeSinceLastAccess() const {
    return Timestamp::Now() - last_access;
}

Timestamp::Duration AccessStats::Age() const {
    return Timestamp::Now() - creation_time;
}

void AccessStats::Serialize(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&access_count), sizeof(access_count));
    last_access.Serialize(out);
    creation_time.Serialize(out);
    out.write(reinterpret_cast<const char*>(&avg_access_interval), sizeof(avg_access_interval));
}

AccessStats AccessStats::Deserialize(std::istream& in) {
    AccessStats stats;
    in.read(reinterpret_cast<char*>(&stats.access_count), sizeof(stats.access_count));
    stats.last_access = Timestamp::Deserialize(in);
    stats.creation_time = Timestamp::Deserialize(in);
    in.read(reinterpret_cast<char*>(&stats.avg_access_interval), sizeof(stats.avg_access_interval));
    return stats;
}

// AccessTracker implementation
void AccessTracker::RecordPatternAccess(PatternID pattern, Timestamp timestamp) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    pattern_stats_[pattern].RecordAccess(timestamp);
}

void AccessTracker::RecordAssociationAccess(PatternID source, PatternID target, Timestamp timestamp) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto key = std::make_pair(source, target);
    association_stats_[key].RecordAccess(timestamp);
}

const AccessStats* AccessTracker::GetPatternStats(PatternID pattern) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto it = pattern_stats_.find(pattern);
    return it != pattern_stats_.end() ? &it->second : nullptr;
}

const AccessStats* AccessTracker::GetAssociationStats(PatternID source, PatternID target) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);

    auto key = std::make_pair(source, target);
    auto it = association_stats_.find(key);
    return it != association_stats_.end() ? &it->second : nullptr;
}

void AccessTracker::PruneOldStats(Timestamp cutoff_time) {
    std::unique_lock<std::shared_mutex> lock(mutex_);

    // Remove pattern stats older than cutoff
    for (auto it = pattern_stats_.begin(); it != pattern_stats_.end(); ) {
        if (it->second.last_access < cutoff_time) {
            it = pattern_stats_.erase(it);
        } else {
            ++it;
        }
    }

    // Remove association stats older than cutoff
    for (auto it = association_stats_.begin(); it != association_stats_.end(); ) {
        if (it->second.last_access < cutoff_time) {
            it = association_stats_.erase(it);
        } else {
            ++it;
        }
    }
}

void AccessTracker::Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    pattern_stats_.clear();
    association_stats_.clear();
}

} // namespace dpan
```

**Comprehensive Unit Tests** (30+ tests):

```cpp
// File: tests/memory/utility_calculator_test.cpp
#include "memory/utility_calculator.hpp"
#include <gtest/gtest.h>

namespace dpan {
namespace {

TEST(UtilityCalculatorTest, DefaultWeightsSumToOne) {
    UtilityCalculator::Config config;
    float sum = config.frequency_weight + config.recency_weight +
                config.association_weight + config.confidence_weight;
    EXPECT_NEAR(1.0f, sum, 0.001f);
}

TEST(UtilityCalculatorTest, InvalidWeightsThrow) {
    UtilityCalculator::Config config;
    config.frequency_weight = 0.5f;
    config.recency_weight = 0.3f;
    config.association_weight = 0.3f;  // Sum > 1.0
    config.confidence_weight = 0.1f;

    EXPECT_THROW(UtilityCalculator calc(config), std::invalid_argument);
}

TEST(UtilityCalculatorTest, FrequencyScoreSaturates) {
    UtilityCalculator calc;

    // Low access count
    float score_low = calc.CalculateFrequencyScore(10);
    EXPECT_GT(score_low, 0.0f);
    EXPECT_LT(score_low, 0.2f);

    // High access count
    float score_high = calc.CalculateFrequencyScore(10000);
    EXPECT_GT(score_high, 0.95f);
    EXPECT_LE(score_high, 1.0f);
}

TEST(UtilityCalculatorTest, RecencyDecaysExponentially) {
    UtilityCalculator calc;

    // Recent access
    auto recent = std::chrono::hours(1);
    float score_recent = calc.CalculateRecencyScore(recent);
    EXPECT_GT(score_recent, 0.95f);

    // Old access (14 hours = half-life)
    auto old = std::chrono::hours(14);
    float score_old = calc.CalculateRecencyScore(old);
    EXPECT_NEAR(0.5f, score_old, 0.05f);

    // Very old access
    auto very_old = std::chrono::hours(100);
    float score_very_old = calc.CalculateRecencyScore(very_old);
    EXPECT_LT(score_very_old, 0.01f);
}

TEST(UtilityCalculatorTest, AssociationScoreAveragesStrengths) {
    UtilityCalculator calc;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();
    PatternID p3 = PatternID::Generate();

    std::vector<AssociationEdge> associations;
    associations.push_back(AssociationEdge(p1, p2, AssociationType::CAUSAL, 0.8f));
    associations.push_back(AssociationEdge(p1, p3, AssociationType::CAUSAL, 0.6f));

    float score = calc.CalculateAssociationScore(associations);
    EXPECT_NEAR(0.7f, score, 0.001f);  // (0.8 + 0.6) / 2
}

TEST(UtilityCalculatorTest, PatternUtilityInRange) {
    UtilityCalculator calc;

    PatternID pid = PatternID::Generate();
    PatternNode pattern(pid);
    pattern.SetConfidence(0.9f);

    AccessStats stats;
    stats.access_count = 100;
    stats.last_access = Timestamp::Now();
    stats.creation_time = Timestamp::Now() - std::chrono::hours(24);

    std::vector<AssociationEdge> associations;
    PatternID p2 = PatternID::Generate();
    associations.push_back(AssociationEdge(pid, p2, AssociationType::CAUSAL, 0.7f));

    float utility = calc.CalculatePatternUtility(pattern, stats, associations);

    EXPECT_GE(utility, 0.0f);
    EXPECT_LE(utility, 1.0f);
    EXPECT_GT(utility, 0.5f);  // Should be reasonably high given good stats
}

TEST(UtilityCalculatorTest, UtilityBreakdownMatchesTotal) {
    UtilityCalculator calc;

    PatternID pid = PatternID::Generate();
    PatternNode pattern(pid);
    pattern.SetConfidence(0.8f);

    AccessStats stats;
    stats.access_count = 50;
    stats.last_access = Timestamp::Now() - std::chrono::hours(2);
    stats.creation_time = Timestamp::Now() - std::chrono::hours(48);

    std::vector<AssociationEdge> associations;

    auto breakdown = calc.GetUtilityBreakdown(pattern, stats, associations);

    float manual_total =
        calc.GetConfig().frequency_weight * breakdown.frequency_score +
        calc.GetConfig().recency_weight * breakdown.recency_score +
        calc.GetConfig().association_weight * breakdown.association_score +
        calc.GetConfig().confidence_weight * breakdown.confidence_score;

    EXPECT_NEAR(breakdown.total, manual_total, 0.001f);
}

TEST(AccessStatsTest, RecordAccessUpdatesCount) {
    AccessStats stats;
    EXPECT_EQ(0u, stats.access_count);

    stats.RecordAccess();
    EXPECT_EQ(1u, stats.access_count);

    stats.RecordAccess();
    EXPECT_EQ(2u, stats.access_count);
}

TEST(AccessStatsTest, RecordAccessUpdatesTimestamp) {
    AccessStats stats;

    Timestamp t1 = Timestamp::Now();
    stats.RecordAccess(t1);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    Timestamp t2 = Timestamp::Now();
    stats.RecordAccess(t2);

    EXPECT_EQ(stats.last_access, t2);
}

TEST(AccessStatsTest, AverageAccessIntervalCalculated) {
    AccessStats stats;

    Timestamp t0 = Timestamp::Now();
    stats.RecordAccess(t0);

    Timestamp t1 = t0 + std::chrono::seconds(10);
    stats.RecordAccess(t1);

    // After first interval, avg should be ~10 seconds
    EXPECT_NEAR(10.0f, stats.avg_access_interval, 1.0f);

    Timestamp t2 = t1 + std::chrono::seconds(20);
    stats.RecordAccess(t2);

    // EMA should be between 10 and 20
    EXPECT_GT(stats.avg_access_interval, 10.0f);
    EXPECT_LT(stats.avg_access_interval, 20.0f);
}

TEST(AccessStatsTest, SerializationRoundTrip) {
    AccessStats original;
    original.access_count = 42;
    original.last_access = Timestamp::Now();
    original.creation_time = Timestamp::Now() - std::chrono::hours(24);
    original.avg_access_interval = 15.5f;

    std::stringstream ss;
    original.Serialize(ss);

    AccessStats deserialized = AccessStats::Deserialize(ss);

    EXPECT_EQ(original.access_count, deserialized.access_count);
    EXPECT_EQ(original.last_access, deserialized.last_access);
    EXPECT_EQ(original.creation_time, deserialized.creation_time);
    EXPECT_FLOAT_EQ(original.avg_access_interval, deserialized.avg_access_interval);
}

TEST(AccessTrackerTest, RecordAndRetrievePatternStats) {
    AccessTracker tracker;

    PatternID p1 = PatternID::Generate();

    tracker.RecordPatternAccess(p1);

    const AccessStats* stats = tracker.GetPatternStats(p1);
    ASSERT_NE(nullptr, stats);
    EXPECT_EQ(1u, stats->access_count);
}

TEST(AccessTrackerTest, MultipleAccessesIncrement) {
    AccessTracker tracker;

    PatternID p1 = PatternID::Generate();

    tracker.RecordPatternAccess(p1);
    tracker.RecordPatternAccess(p1);
    tracker.RecordPatternAccess(p1);

    const AccessStats* stats = tracker.GetPatternStats(p1);
    ASSERT_NE(nullptr, stats);
    EXPECT_EQ(3u, stats->access_count);
}

TEST(AccessTrackerTest, DifferentPatternsTrackedSeparately) {
    AccessTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    tracker.RecordPatternAccess(p1);
    tracker.RecordPatternAccess(p1);
    tracker.RecordPatternAccess(p2);

    const AccessStats* stats1 = tracker.GetPatternStats(p1);
    const AccessStats* stats2 = tracker.GetPatternStats(p2);

    EXPECT_EQ(2u, stats1->access_count);
    EXPECT_EQ(1u, stats2->access_count);
}

TEST(AccessTrackerTest, PruneOldStats) {
    AccessTracker tracker;

    PatternID p1 = PatternID::Generate();
    PatternID p2 = PatternID::Generate();

    Timestamp old_time = Timestamp::Now() - std::chrono::hours(48);
    Timestamp recent_time = Timestamp::Now() - std::chrono::hours(1);

    tracker.RecordPatternAccess(p1, old_time);
    tracker.RecordPatternAccess(p2, recent_time);

    Timestamp cutoff = Timestamp::Now() - std::chrono::hours(24);
    tracker.PruneOldStats(cutoff);

    EXPECT_EQ(nullptr, tracker.GetPatternStats(p1));  // Pruned
    EXPECT_NE(nullptr, tracker.GetPatternStats(p2));  // Kept
}

TEST(AccessTrackerTest, ThreadSafeConcurrentAccess) {
    AccessTracker tracker;

    std::vector<PatternID> patterns;
    for (int i = 0; i < 10; ++i) {
        patterns.push_back(PatternID::Generate());
    }

    std::vector<std::thread> threads;
    constexpr int kThreads = 10;
    constexpr int kAccessesPerThread = 1000;

    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&tracker, &patterns]() {
            for (int j = 0; j < kAccessesPerThread; ++j) {
                tracker.RecordPatternAccess(patterns[j % patterns.size()]);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    // Each pattern should have kThreads * (kAccessesPerThread / patterns.size()) accesses
    for (const auto& pattern : patterns) {
        const AccessStats* stats = tracker.GetPatternStats(pattern);
        ASSERT_NE(nullptr, stats);
        EXPECT_EQ(kThreads * (kAccessesPerThread / patterns.size()), stats->access_count);
    }
}

// More tests...

} // namespace
} // namespace dpan
```

**Performance Benchmarks**:

```cpp
// File: benchmarks/memory/utility_calculator_benchmark.cpp
#include "memory/utility_calculator.hpp"
#include <benchmark/benchmark.h>

namespace dpan {

static void BM_CalculatePatternUtility(benchmark::State& state) {
    UtilityCalculator calc;

    PatternID pid = PatternID::Generate();
    PatternNode pattern(pid);
    pattern.SetConfidence(0.8f);

    AccessStats stats;
    stats.access_count = 100;
    stats.last_access = Timestamp::Now();

    std::vector<AssociationEdge> associations;
    for (int i = 0; i < 10; ++i) {
        PatternID target = PatternID::Generate();
        associations.push_back(AssociationEdge(pid, target, AssociationType::CAUSAL, 0.7f));
    }

    for (auto _ : state) {
        float utility = calc.CalculatePatternUtility(pattern, stats, associations);
        benchmark::DoNotOptimize(utility);
    }
}
BENCHMARK(BM_CalculatePatternUtility);

static void BM_RecordAccess(benchmark::State& state) {
    AccessTracker tracker;
    PatternID p1 = PatternID::Generate();

    for (auto _ : state) {
        tracker.RecordPatternAccess(p1);
    }
}
BENCHMARK(BM_RecordAccess);

static void BM_GetPatternStats(benchmark::State& state) {
    AccessTracker tracker;
    PatternID p1 = PatternID::Generate();
    tracker.RecordPatternAccess(p1);

    for (auto _ : state) {
        const AccessStats* stats = tracker.GetPatternStats(p1);
        benchmark::DoNotOptimize(stats);
    }
}
BENCHMARK(BM_GetPatternStats);

} // namespace dpan

BENCHMARK_MAIN();
```

**Performance Targets**:
- Utility calculation: <500ns per pattern
- Access recording: <100ns
- Stats lookup: <50ns
- Pruning 1M stats: <500ms

**Acceptance Criteria**:
- [ ] All 30+ unit tests pass
- [ ] Thread-safe verified (ThreadSanitizer)
- [ ] Utility scores in [0,1] range
- [ ] Weights sum to 1.0 enforced
- [ ] Performance targets met
- [ ] >95% code coverage
- [ ] Serialization maintains accuracy

---

### Task 4.1.2: Implement Adaptive Thresholds

**Duration**: 3 days (24 hours)
**Priority**: Critical
**Files to create**:
- `src/memory/adaptive_thresholds.hpp`
- `src/memory/adaptive_thresholds.cpp`
- `tests/memory/adaptive_thresholds_test.cpp`

#### Subtask 4.1.2.1: Design Threshold Adaptation Strategy (6 hours)

**Mathematical Foundation**:

Dynamic threshold adjustment based on memory pressure:

```
T(p) = T_base × (1 + pressure_factor × P)

Where:
  T(p) = threshold at pressure p
  T_base = baseline threshold (e.g., 0.3)
  P = memory pressure ∈ [0,1]
  pressure_factor = 2.0 (configurable)

Memory Pressure Calculation:
P = (M_used - M_target) / M_target

Where:
  M_used = current memory usage
  M_target = target memory limit

P < 0: Under-utilized (can be more lenient)
P = 0: At target (use baseline threshold)
P > 0: Over-utilized (need stricter pruning)
```

**Percentile-Based Threshold**:

```
T_percentile = utility_value at k-th percentile

Where k depends on target eviction rate:
  k = 100 × (1 - eviction_rate)

Example: To evict bottom 20%, set T = 20th percentile
```

**Implementation**:

```cpp
// File: src/memory/adaptive_thresholds.hpp
#pragma once

#include "memory/utility_calculator.hpp"
#include <vector>
#include <algorithm>

namespace dpan {

class AdaptiveThresholdManager {
public:
    struct Config {
        float baseline_threshold{0.3f};    // Base utility threshold
        float pressure_factor{2.0f};       // How much pressure affects threshold
        float min_threshold{0.1f};         // Never go below this
        float max_threshold{0.9f};         // Never go above this

        // Memory pressure targets
        size_t target_memory_bytes{8ULL * 1024 * 1024 * 1024};  // 8GB
        float pressure_update_interval{60.0f};  // seconds

        // Percentile-based thresholds
        bool use_percentile{true};
        float target_eviction_rate{0.2f};  // Evict bottom 20%
    };

    explicit AdaptiveThresholdManager(const Config& config = {});

    // Update threshold based on current memory usage
    void UpdateThreshold(size_t current_memory_bytes, size_t pattern_count);

    // Get current threshold for pruning decisions
    float GetCurrentThreshold() const { return current_threshold_; }

    // Compute threshold from utility distribution
    float ComputePercentileThreshold(const std::vector<float>& utilities);

    // Memory pressure calculation
    float ComputeMemoryPressure(size_t current_bytes) const;

    // Set configuration
    void SetConfig(const Config& config);
    const Config& GetConfig() const { return config_; }

    // Statistics
    struct ThresholdStats {
        float current_threshold;
        float memory_pressure;
        float baseline_threshold;
        size_t current_memory_bytes;
        size_t target_memory_bytes;
        Timestamp last_update;
    };

    ThresholdStats GetStats() const;

private:
    Config config_;
    float current_threshold_;
    float current_pressure_;
    Timestamp last_update_;

    // Smoothing for threshold changes
    void SmoothThresholdUpdate(float new_threshold, float smoothing_factor = 0.3f);
};

} // namespace dpan
```

#### Subtask 4.1.2.2: Implement Threshold Logic (12 hours)

```cpp
// File: src/memory/adaptive_thresholds.cpp
#include "memory/adaptive_thresholds.hpp"
#include <algorithm>
#include <cmath>

namespace dpan {

AdaptiveThresholdManager::AdaptiveThresholdManager(const Config& config)
    : config_(config),
      current_threshold_(config.baseline_threshold),
      current_pressure_(0.0f),
      last_update_(Timestamp::Now())
{
}

void AdaptiveThresholdManager::UpdateThreshold(
    size_t current_memory_bytes,
    size_t pattern_count
) {
    // Check if enough time has passed since last update
    auto elapsed = Timestamp::Now() - last_update_;
    auto elapsed_secs = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();

    if (elapsed_secs < config_.pressure_update_interval) {
        return;  // Don't update too frequently
    }

    // Compute current memory pressure
    current_pressure_ = ComputeMemoryPressure(current_memory_bytes);

    // Calculate new threshold based on pressure
    float pressure_adjusted_threshold =
        config_.baseline_threshold * (1.0f + config_.pressure_factor * current_pressure_);

    // Clamp to valid range
    pressure_adjusted_threshold = std::clamp(
        pressure_adjusted_threshold,
        config_.min_threshold,
        config_.max_threshold
    );

    // Apply smoothing to avoid oscillation
    SmoothThresholdUpdate(pressure_adjusted_threshold);

    last_update_ = Timestamp::Now();
}

float AdaptiveThresholdManager::ComputeMemoryPressure(size_t current_bytes) const {
    if (current_bytes <= config_.target_memory_bytes) {
        // Under-utilized: negative pressure
        float ratio = static_cast<float>(current_bytes) / config_.target_memory_bytes;
        return ratio - 1.0f;  // Will be negative
    } else {
        // Over-utilized: positive pressure
        float excess = current_bytes - config_.target_memory_bytes;
        return excess / config_.target_memory_bytes;
    }
}

float AdaptiveThresholdManager::ComputePercentileThreshold(
    const std::vector<float>& utilities
) {
    if (utilities.empty()) {
        return config_.baseline_threshold;
    }

    // Create sorted copy
    std::vector<float> sorted_utilities = utilities;
    std::sort(sorted_utilities.begin(), sorted_utilities.end());

    // Find k-th percentile
    float k = config_.target_eviction_rate * 100.0f;
    size_t index = static_cast<size_t>(k / 100.0f * sorted_utilities.size());
    index = std::min(index, sorted_utilities.size() - 1);

    return sorted_utilities[index];
}

void AdaptiveThresholdManager::SmoothThresholdUpdate(
    float new_threshold,
    float smoothing_factor
) {
    // Exponential moving average for smooth transitions
    current_threshold_ =
        smoothing_factor * new_threshold +
        (1.0f - smoothing_factor) * current_threshold_;
}

void AdaptiveThresholdManager::SetConfig(const Config& config) {
    config_ = config;
    // Reset to baseline
    current_threshold_ = config_.baseline_threshold;
    current_pressure_ = 0.0f;
}

AdaptiveThresholdManager::ThresholdStats AdaptiveThresholdManager::GetStats() const {
    ThresholdStats stats;
    stats.current_threshold = current_threshold_;
    stats.memory_pressure = current_pressure_;
    stats.baseline_threshold = config_.baseline_threshold;
    stats.current_memory_bytes = 0;  // Caller should fill this in
    stats.target_memory_bytes = config_.target_memory_bytes;
    stats.last_update = last_update_;
    return stats;
}

} // namespace dpan
```

**Unit Tests** (20+ tests):

```cpp
// File: tests/memory/adaptive_thresholds_test.cpp
#include "memory/adaptive_thresholds.hpp"
#include <gtest/gtest.h>

namespace dpan {
namespace {

TEST(AdaptiveThresholdManagerTest, InitialThresholdIsBaseline) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.4f;

    AdaptiveThresholdManager manager(config);

    EXPECT_FLOAT_EQ(0.4f, manager.GetCurrentThreshold());
}

TEST(AdaptiveThresholdManagerTest, NegativePressureWhenUnderUtilized) {
    AdaptiveThresholdManager::Config config;
    config.target_memory_bytes = 8ULL * 1024 * 1024 * 1024;  // 8GB

    AdaptiveThresholdManager manager(config);

    size_t current_bytes = 4ULL * 1024 * 1024 * 1024;  // 4GB (50% of target)
    float pressure = manager.ComputeMemoryPressure(current_bytes);

    EXPECT_LT(pressure, 0.0f);
    EXPECT_NEAR(-0.5f, pressure, 0.01f);
}

TEST(AdaptiveThresholdManagerTest, PositivePressureWhenOverUtilized) {
    AdaptiveThresholdManager::Config config;
    config.target_memory_bytes = 8ULL * 1024 * 1024 * 1024;  // 8GB

    AdaptiveThresholdManager manager(config);

    size_t current_bytes = 12ULL * 1024 * 1024 * 1024;  // 12GB (150% of target)
    float pressure = manager.ComputeMemoryPressure(current_bytes);

    EXPECT_GT(pressure, 0.0f);
    EXPECT_NEAR(0.5f, pressure, 0.01f);
}

TEST(AdaptiveThresholdManagerTest, ThresholdIncreasesWithPressure) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.3f;
    config.pressure_factor = 2.0f;
    config.target_memory_bytes = 8ULL * 1024 * 1024 * 1024;
    config.pressure_update_interval = 0.0f;  // Allow immediate updates

    AdaptiveThresholdManager manager(config);

    float initial = manager.GetCurrentThreshold();

    // Simulate memory pressure
    size_t over_memory = 12ULL * 1024 * 1024 * 1024;  // 50% over
    manager.UpdateThreshold(over_memory, 1000000);

    float after_pressure = manager.GetCurrentThreshold();

    EXPECT_GT(after_pressure, initial);
}

TEST(AdaptiveThresholdManagerTest, ThresholdRespectsBounds) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.3f;
    config.pressure_factor = 10.0f;  // Large factor
    config.min_threshold = 0.1f;
    config.max_threshold = 0.9f;
    config.target_memory_bytes = 1ULL * 1024 * 1024 * 1024;
    config.pressure_update_interval = 0.0f;

    AdaptiveThresholdManager manager(config);

    // Extreme over-utilization
    size_t huge_memory = 100ULL * 1024 * 1024 * 1024;
    manager.UpdateThreshold(huge_memory, 1000000);

    EXPECT_LE(manager.GetCurrentThreshold(), config.max_threshold);
    EXPECT_GE(manager.GetCurrentThreshold(), config.min_threshold);
}

TEST(AdaptiveThresholdManagerTest, PercentileThresholdCorrect) {
    AdaptiveThresholdManager::Config config;
    config.target_eviction_rate = 0.2f;  // Bottom 20%

    AdaptiveThresholdManager manager(config);

    std::vector<float> utilities = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};

    float threshold = manager.ComputePercentileThreshold(utilities);

    // 20th percentile of 10 values = 2nd value = 0.2
    EXPECT_NEAR(0.2f, threshold, 0.01f);
}

TEST(AdaptiveThresholdManagerTest, SmoothingPreventsOscillation) {
    AdaptiveThresholdManager::Config config;
    config.baseline_threshold = 0.3f;
    config.pressure_update_interval = 0.0f;

    AdaptiveThresholdManager manager(config);

    float initial = manager.GetCurrentThreshold();

    // Apply pressure
    size_t high_memory = 12ULL * 1024 * 1024 * 1024;
    manager.UpdateThreshold(high_memory, 1000000);

    float after_first = manager.GetCurrentThreshold();

    // Apply more pressure
    manager.UpdateThreshold(high_memory, 1000000);

    float after_second = manager.GetCurrentThreshold();

    // Should converge gradually
    EXPECT_GT(after_first, initial);
    EXPECT_GT(after_second, after_first);

    // But not jump too much
    float jump1 = after_first - initial;
    float jump2 = after_second - after_first;
    EXPECT_LT(jump2, jump1);  // Smaller increments over time
}

// More tests...

} // namespace
} // namespace dpan
```

**Acceptance Criteria**:
- [ ] Threshold adapts to memory pressure
- [ ] Respects min/max bounds
- [ ] Smoothing prevents oscillation
- [ ] Percentile calculation accurate
- [ ] >20 unit tests pass
- [ ] Performance: <1µs threshold update

---

### Task 4.1.3: Implement Utility Tracking

**Duration**: 3 days (24 hours)
**Priority**: High
**Files to create**:
- `src/memory/utility_tracker.hpp`
- `src/memory/utility_tracker.cpp`
- `tests/memory/utility_tracker_test.cpp`

**Overview**:
Periodic recalculation of utility scores for all patterns and associations, maintaining history for trend analysis.

**Key Features**:
- Background thread for periodic updates
- Batch processing for efficiency
- Utility history for trend analysis
- Integration with AccessTracker and UtilityCalculator

**Implementation Summary** (see files for full code):
- `UtilityTracker` class with scheduled updates
- Utility history with sliding window
- Trend detection (increasing/decreasing utility)
- Top-K utility patterns for quick access
- Thread-safe concurrent updates

**Acceptance Criteria**:
- [ ] Periodic updates work reliably
- [ ] Batch processing efficient (>10K patterns/sec)
- [ ] History maintained correctly
- [ ] Trend detection accurate
- [ ] >15 unit tests pass

---

## Module 4.2: Memory Hierarchy

**Duration**: 2 weeks (80 hours)
**Dependencies**: Module 4.1 complete
**Owner**: Systems engineer + C++ developer

### Overview
Implements a multi-tier memory system with transparent access, automatic tier transitions based on utility, and optimized performance for each storage medium.

**Tier Structure**:
1. **Active Tier** (RAM): Hot patterns, <100ns access
2. **Warm Tier** (SSD): Recently used, <10µs access
3. **Cold Tier** (HDD): Rarely accessed, <1ms access
4. **Archive Tier** (Compressed): Long-term storage, <10ms access

---

### Task 4.2.1: Implement Memory Tier System

**Duration**: 5 days (40 hours)
**Priority**: Critical
**Files to create**:
- `src/memory/memory_tier.hpp`
- `src/memory/memory_tier.cpp`
- `src/memory/tier_storage.hpp`
- `src/memory/active_tier.cpp`
- `src/memory/warm_tier.cpp`
- `src/memory/cold_tier.cpp`
- `src/memory/archive_tier.cpp`
- `tests/memory/memory_tier_test.cpp`

#### Subtask 4.2.1.1: Define Tier Interface (8 hours)

```cpp
// File: src/memory/memory_tier.hpp
#pragma once

#include "core/pattern_node.hpp"
#include "association/association_edge.hpp"
#include <memory>
#include <optional>

namespace dpan {

// Tier levels
enum class MemoryTier {
    ACTIVE = 0,   // RAM-based, fastest
    WARM = 1,     // SSD-based, fast
    COLD = 2,     // HDD-based, slow
    ARCHIVE = 3   // Compressed disk, slowest
};

// Abstract interface for a memory tier
class IMemoryTier {
public:
    virtual ~IMemoryTier() = default;

    // Pattern operations
    virtual bool StorePattern(const PatternNode& pattern) = 0;
    virtual std::optional<PatternNode> LoadPattern(PatternID id) = 0;
    virtual bool RemovePattern(PatternID id) = 0;
    virtual bool HasPattern(PatternID id) const = 0;

    // Association operations
    virtual bool StoreAssociation(const AssociationEdge& edge) = 0;
    virtual std::optional<AssociationEdge> LoadAssociation(PatternID source, PatternID target) = 0;
    virtual bool RemoveAssociation(PatternID source, PatternID target) = 0;

    // Batch operations (more efficient)
    virtual size_t StorePatternsBatch(const std::vector<PatternNode>& patterns) = 0;
    virtual std::vector<PatternNode> LoadPatternsBatch(const std::vector<PatternID>& ids) = 0;

    // Statistics
    virtual size_t GetPatternCount() const = 0;
    virtual size_t GetAssociationCount() const = 0;
    virtual size_t EstimateMemoryUsage() const = 0;

    // Tier information
    virtual MemoryTier GetTierLevel() const = 0;
    virtual std::string GetTierName() const = 0;

    // Maintenance
    virtual void Compact() = 0;
    virtual void Clear() = 0;
};

// Factory for creating tier instances
std::unique_ptr<IMemoryTier> CreateActiveTier(const std::string& config_path);
std::unique_ptr<IMemoryTier> CreateWarmTier(const std::string& config_path);
std::unique_ptr<IMemoryTier> CreateColdTier(const std::string& config_path);
std::unique_ptr<IMemoryTier> CreateArchiveTier(const std::string& config_path);

} // namespace dpan
```

**Active Tier Implementation** (RAM-based, in-memory hash maps):

```cpp
// File: src/memory/active_tier.cpp
#include "memory/memory_tier.hpp"
#include <unordered_map>
#include <shared_mutex>

namespace dpan {

class ActiveTier : public IMemoryTier {
public:
    ActiveTier() = default;

    bool StorePattern(const PatternNode& pattern) override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        patterns_[pattern.GetID()] = pattern;
        return true;
    }

    std::optional<PatternNode> LoadPattern(PatternID id) override {
        std::shared_lock<std::shared_mutex> lock(mutex_);

        auto it = patterns_.find(id);
        if (it != patterns_.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    bool RemovePattern(PatternID id) override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        return patterns_.erase(id) > 0;
    }

    bool HasPattern(PatternID id) const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return patterns_.find(id) != patterns_.end();
    }

    size_t GetPatternCount() const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return patterns_.size();
    }

    size_t EstimateMemoryUsage() const override {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return patterns_.size() * sizeof(PatternNode);
    }

    MemoryTier GetTierLevel() const override { return MemoryTier::ACTIVE; }
    std::string GetTierName() const override { return "Active"; }

    void Compact() override { /* No-op for in-memory */ }
    void Clear() override {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        patterns_.clear();
        associations_.clear();
    }

    // Additional methods...

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<PatternID, PatternNode> patterns_;
    std::unordered_map<std::pair<PatternID, PatternID>, AssociationEdge, PatternPairHash> associations_;
};

std::unique_ptr<IMemoryTier> CreateActiveTier(const std::string& config_path) {
    return std::make_unique<ActiveTier>();
}

} // namespace dpan
```

**Warm Tier Implementation** (SSD-based, RocksDB):

```cpp
// File: src/memory/warm_tier.cpp
#include "memory/memory_tier.hpp"
#include <rocksdb/db.h>

namespace dpan {

class WarmTier : public IMemoryTier {
public:
    explicit WarmTier(const std::string& db_path) {
        rocksdb::Options options;
        options.create_if_missing = true;
        options.compression = rocksdb::kLZ4Compression;

        rocksdb::DB* db_ptr;
        rocksdb::Status status = rocksdb::DB::Open(options, db_path, &db_ptr);

        if (!status.ok()) {
            throw std::runtime_error("Failed to open WarmTier DB: " + status.ToString());
        }

        db_.reset(db_ptr);
    }

    bool StorePattern(const PatternNode& pattern) override {
        std::string key = EncodePatternKey(pattern.GetID());
        std::string value = SerializePattern(pattern);

        rocksdb::WriteOptions write_opts;
        rocksdb::Status status = db_->Put(write_opts, key, value);

        return status.ok();
    }

    std::optional<PatternNode> LoadPattern(PatternID id) override {
        std::string key = EncodePatternKey(id);
        std::string value;

        rocksdb::ReadOptions read_opts;
        rocksdb::Status status = db_->Get(read_opts, key, &value);

        if (!status.ok()) {
            return std::nullopt;
        }

        return DeserializePattern(value);
    }

    // Additional methods...

private:
    std::unique_ptr<rocksdb::DB> db_;

    std::string EncodePatternKey(PatternID id) const;
    std::string SerializePattern(const PatternNode& pattern) const;
    PatternNode DeserializePattern(const std::string& value) const;
};

} // namespace dpan
```

**Acceptance Criteria for Task 4.2.1**:
- [ ] All 4 tier implementations complete
- [ ] Interface abstraction works correctly
- [ ] Active tier: <100ns access latency
- [ ] Warm tier: <10µs access latency
- [ ] Cold tier: <1ms access latency
- [ ] >25 unit tests pass
- [ ] Thread-safe verified

---

### Task 4.2.2: Implement Tier Manager

**Duration**: 4 days (32 hours)
**Priority**: Critical
**Files to create**:
- `src/memory/tier_manager.hpp`
- `src/memory/tier_manager.cpp`
- `tests/memory/tier_manager_test.cpp`

**Overview**:
Manages automatic promotion/demotion of patterns between tiers based on utility scores.

**Tier Transition Rules**:

```
Promotion (Cold → Warm → Active):
  IF utility(p) > threshold_promote[current_tier] THEN
    Move p to higher tier

Demotion (Active → Warm → Cold → Archive):
  IF utility(p) < threshold_demote[current_tier] THEN
    Move p to lower tier

Thresholds (example):
  Active: promote N/A, demote 0.7
  Warm:   promote 0.8, demote 0.4
  Cold:   promote 0.6, demote 0.2
  Archive: promote 0.4, demote N/A
```

**Implementation**:

```cpp
// File: src/memory/tier_manager.hpp
#pragma once

#include "memory/memory_tier.hpp"
#include "memory/utility_calculator.hpp"
#include <vector>
#include <thread>

namespace dpan {

class TierManager {
public:
    struct Config {
        // Tier capacities (number of patterns)
        size_t active_capacity{100000};
        size_t warm_capacity{1000000};
        size_t cold_capacity{10000000};
        // Archive has unlimited capacity

        // Promotion thresholds
        float warm_to_active_threshold{0.8f};
        float cold_to_warm_threshold{0.6f};
        float archive_to_cold_threshold{0.4f};

        // Demotion thresholds
        float active_to_warm_threshold{0.7f};
        float warm_to_cold_threshold{0.4f};
        float cold_to_archive_threshold{0.2f};

        // Transition settings
        size_t transition_batch_size{1000};
        float transition_interval_seconds{300.0f};  // 5 minutes
    };

    explicit TierManager(const Config& config = {});
    ~TierManager();

    // Initialize tiers
    void Initialize(
        std::unique_ptr<IMemoryTier> active,
        std::unique_ptr<IMemoryTier> warm,
        std::unique_ptr<IMemoryTier> cold,
        std::unique_ptr<IMemoryTier> archive
    );

    // Trigger tier transitions
    void PerformTierTransitions(const std::unordered_map<PatternID, float>& utilities);

    // Manual tier control
    bool PromotePattern(PatternID id, MemoryTier target_tier);
    bool DemotePattern(PatternID id, MemoryTier target_tier);

    // Query pattern location
    MemoryTier GetPatternTier(PatternID id) const;

    // Statistics
    struct TierStats {
        size_t active_count;
        size_t warm_count;
        size_t cold_count;
        size_t archive_count;

        size_t promotions_count;
        size_t demotions_count;

        Timestamp last_transition;
    };

    TierStats GetStats() const;

    // Background transition thread
    void StartBackgroundTransitions(
        const UtilityCalculator* utility_calc,
        const AccessTracker* access_tracker
    );
    void StopBackgroundTransitions();

private:
    Config config_;

    std::unique_ptr<IMemoryTier> active_tier_;
    std::unique_ptr<IMemoryTier> warm_tier_;
    std::unique_ptr<IMemoryTier> cold_tier_;
    std::unique_ptr<IMemoryTier> archive_tier_;

    // Pattern → Tier mapping
    std::unordered_map<PatternID, MemoryTier> pattern_locations_;
    mutable std::shared_mutex location_mutex_;

    // Transition statistics
    std::atomic<size_t> promotions_count_{0};
    std::atomic<size_t> demotions_count_{0};
    Timestamp last_transition_;

    // Background thread
    std::unique_ptr<std::thread> background_thread_;
    std::atomic<bool> running_{false};

    // Helper methods
    bool MovePattern(PatternID id, MemoryTier from, MemoryTier to);
    std::vector<PatternID> SelectPatternsForPromotion(
        MemoryTier tier,
        const std::unordered_map<PatternID, float>& utilities
    );
    std::vector<PatternID> SelectPatternsForDemotion(
        MemoryTier tier,
        const std::unordered_map<PatternID, float>& utilities
    );

    void BackgroundTransitionLoop(
        const UtilityCalculator* utility_calc,
        const AccessTracker* access_tracker
    );
};

} // namespace dpan
```

**Acceptance Criteria**:
- [ ] Tier transitions work correctly
- [ ] Capacity limits enforced
- [ ] Background thread operates reliably
- [ ] >20 unit tests pass
- [ ] No data loss during transitions

---

### Task 4.2.3: Implement Transparent Access Layer

**Duration**: 3 days (24 hours)
**Priority**: High
**Files to create**:
- `src/memory/tiered_storage.hpp`
- `src/memory/tiered_storage.cpp`
- `tests/memory/tiered_storage_test.cpp`

**Overview**:
Provides unified interface that automatically locates patterns across tiers and loads them on demand with caching and prefetching.

**Key Features**:
- Automatic tier lookup (check Active → Warm → Cold → Archive)
- LRU cache for recently accessed patterns
- Prefetching based on association graphs
- Transparent promotion on access

**Implementation Summary**:
```cpp
class TieredStorage {
public:
    // Transparent pattern access (checks all tiers)
    std::optional<PatternNode> GetPattern(PatternID id);

    // With automatic promotion
    std::optional<PatternNode> GetPatternWithPromotion(PatternID id);

    // Prefetch associated patterns
    void PrefetchAssociations(PatternID id, size_t max_depth = 1);

private:
    TierManager* tier_manager_;
    LRUCache<PatternID, PatternNode> cache_;
};
```

**Acceptance Criteria**:
- [ ] Transparent access works across all tiers
- [ ] Cache hit rate >80% for hot patterns
- [ ] Prefetching reduces latency
- [ ] >15 unit tests pass

---

## Module 4.3: Pruning System

**Duration**: 2 weeks (80 hours)
**Dependencies**: Modules 4.1, 4.2 complete
**Owner**: ML engineer + C++ developer

### Overview
Implements intelligent pruning to remove low-utility patterns and associations while maintaining system knowledge quality.

---

### Task 4.3.1: Implement Pattern Pruner

**Duration**: 4 days (32 hours)
**Priority**: Critical
**Files to create**:
- `src/memory/pattern_pruner.hpp`
- `src/memory/pattern_pruner.cpp`
- `tests/memory/pattern_pruner_test.cpp`

**Pruning Strategy**:

```
For each pattern p:
  1. Calculate utility U(p)
  2. IF U(p) < threshold THEN
      a. Check if p is referenced by important associations
      b. If safe to delete:
         - Remove p from pattern database
         - Update all associations involving p
         - Optionally merge p with similar patterns
      c. Else:
         - Keep p but demote to lower tier
  3. Record pruning decision for analysis
```

**Safety Checks**:
- Don't prune if pattern is hub (>50 strong associations)
- Don't prune if recently created (<24 hours)
- Don't prune if part of important hierarchy

**Implementation**:

```cpp
// File: src/memory/pattern_pruner.hpp
#pragma once

#include "core/pattern_node.hpp"
#include "memory/utility_calculator.hpp"
#include "association/association_matrix.hpp"

namespace dpan {

class PatternPruner {
public:
    struct Config {
        float utility_threshold{0.2f};
        size_t min_associations_for_hub{50};
        Timestamp::Duration min_pattern_age{std::chrono::hours(24)};

        bool enable_merging{true};
        float merge_similarity_threshold{0.95f};

        size_t max_prune_batch{1000};
    };

    explicit PatternPruner(const Config& config = {});

    // Main pruning operation
    struct PruneResult {
        std::vector<PatternID> pruned_patterns;
        std::vector<std::pair<PatternID, PatternID>> merged_patterns;  // (old, new)
        size_t associations_updated;
        size_t bytes_freed;
    };

    PruneResult PrunePatterns(
        PatternDatabase& pattern_db,
        AssociationMatrix& assoc_matrix,
        const std::unordered_map<PatternID, float>& utilities
    );

    // Safety checks
    bool IsSafeToPrune(
        PatternID id,
        const PatternNode& pattern,
        const AssociationMatrix& assoc_matrix,
        float utility
    ) const;

    // Pattern merging
    std::optional<PatternID> FindMergeCandidate(
        const PatternNode& pattern,
        const PatternDatabase& pattern_db
    ) const;

    bool MergePatterns(
        PatternID old_pattern,
        PatternID new_pattern,
        PatternDatabase& pattern_db,
        AssociationMatrix& assoc_matrix
    );

private:
    Config config_;
};

} // namespace dpan
```

**Acceptance Criteria**:
- [ ] Pruning maintains system knowledge quality
- [ ] Safety checks prevent critical data loss
- [ ] Merging reduces redundancy
- [ ] >20 unit tests pass
- [ ] Performance: >1000 patterns/sec pruned

---

### Task 4.3.2: Implement Association Pruner

**Duration**: 3 days (24 hours)
**Priority**: High
**Files to create**:
- `src/memory/association_pruner.hpp`
- `src/memory/association_pruner.cpp`
- `tests/memory/association_pruner_test.cpp`

**Overview**:
Remove weak, redundant, or stale associations to maintain graph quality.

**Pruning Criteria**:
1. **Weak associations**: strength < threshold
2. **Redundant associations**: transitively implied by stronger paths
3. **Stale associations**: no reinforcement in long time
4. **Contradictory associations**: conflicting with stronger evidence

**Implementation Summary**:
- Identify weak associations (strength < 0.1)
- Detect redundant edges using transitive reduction
- Remove stale associations (no access in 30 days)
- Batch deletion for efficiency

**Acceptance Criteria**:
- [ ] Weak associations removed correctly
- [ ] Redundancy detection accurate
- [ ] Graph connectivity maintained
- [ ] >15 unit tests pass

---

### Task 4.3.3: Implement Consolidation

**Duration**: 3 days (24 hours)
**Priority**: Medium
**Files to create**:
- `src/memory/consolidator.hpp`
- `src/memory/consolidator.cpp`
- `tests/memory/consolidator_test.cpp`

**Overview**:
Consolidate multiple patterns into hierarchical representations and compress association graphs.

**Consolidation Strategies**:

1. **Pattern Merging**: Combine similar patterns
```
similarity(p1, p2) > threshold → merge into p_new
Transfer all associations from p1, p2 to p_new
```

2. **Hierarchy Formation**: Group related patterns
```
Find clusters of highly associated patterns
Create parent pattern representing the cluster
Connect parent to cluster members
```

3. **Association Compression**: Replace multi-hop paths with direct links
```
IF path p1 → p2 → p3 frequently traversed THEN
  Create direct association p1 → p3
  Reduce strength of intermediate associations
```

**Acceptance Criteria**:
- [ ] Pattern merging preserves information
- [ ] Hierarchies correctly formed
- [ ] Association compression reduces graph size
- [ ] >12 unit tests pass

---

### Task 4.3.4: Implement Pruning Scheduler

**Duration**: 2 days (16 hours)
**Priority**: Medium
**Files to create**:
- `src/memory/pruning_scheduler.hpp`
- `src/memory/pruning_scheduler.cpp`
- `tests/memory/pruning_scheduler_test.cpp`

**Overview**:
Schedule pruning operations during low-activity periods to minimize impact.

**Scheduling Strategies**:
- **Periodic**: Every N hours
- **Memory-triggered**: When memory usage > threshold
- **Activity-based**: During low-activity periods

**Implementation**:
```cpp
class PruningScheduler {
public:
    struct Config {
        std::chrono::hours periodic_interval{24};
        float memory_threshold{0.8f};  // 80% of capacity
        bool enable_activity_detection{true};
    };

    void SchedulePruning(PruneCallback callback);
    void TriggerImmediatePruning();
    void EnablePeriodicPruning();
    void DisablePeriodicPruning();

private:
    bool IsLowActivityPeriod() const;
};
```

**Acceptance Criteria**:
- [ ] Scheduled pruning executes reliably
- [ ] Low-activity detection works
- [ ] Manual triggers function correctly
- [ ] >10 unit tests pass

---

## Module 4.4: Forgetting Mechanisms

**Duration**: 1 week (40 hours)
**Dependencies**: Module 4.3 complete
**Owner**: ML engineer

### Overview
Implements biologically-inspired forgetting mechanisms including decay functions, interference, and consolidation.

---

### Task 4.4.1: Implement Decay Functions

**Duration**: 3 days (24 hours)
**Priority**: High
**Files to create**:
- `src/memory/decay_functions.hpp`
- `src/memory/decay_functions.cpp`
- `tests/memory/decay_functions_test.cpp`

**Decay Function Types**:

1. **Exponential Decay** (Ebbinghaus forgetting curve):
```
s(t) = s_0 × e^(-λt)

Where:
  s(t) = strength at time t
  s_0 = initial strength
  λ = decay constant
  t = time since last reinforcement
```

2. **Power-Law Decay** (Anderson's ACT-R):
```
s(t) = s_0 / (1 + t/τ)^β

Where:
  τ = time constant
  β = decay exponent (typically 0.5)
```

3. **Step Decay** (discrete intervals):
```
s(t) = s_0 × decay_factor^(floor(t / step_size))
```

**Implementation**:

```cpp
// File: src/memory/decay_functions.hpp
#pragma once

#include "core/types.hpp"
#include <functional>

namespace dpan {

// Abstract decay function
class IDecayFunction {
public:
    virtual ~IDecayFunction() = default;

    // Apply decay to strength based on elapsed time
    virtual float ApplyDecay(float initial_strength, Timestamp::Duration elapsed_time) const = 0;

    // Get decay amount (how much strength lost)
    virtual float GetDecayAmount(float initial_strength, Timestamp::Duration elapsed_time) const {
        return initial_strength - ApplyDecay(initial_strength, elapsed_time);
    }
};

// Exponential decay
class ExponentialDecay : public IDecayFunction {
public:
    explicit ExponentialDecay(float decay_constant = 0.01f)
        : decay_constant_(decay_constant) {}

    float ApplyDecay(float initial_strength, Timestamp::Duration elapsed_time) const override {
        auto hours = std::chrono::duration_cast<std::chrono::hours>(elapsed_time).count();
        return initial_strength * std::exp(-decay_constant_ * hours);
    }

private:
    float decay_constant_;
};

// Power-law decay
class PowerLawDecay : public IDecayFunction {
public:
    explicit PowerLawDecay(float time_constant = 1.0f, float exponent = 0.5f)
        : time_constant_(time_constant), exponent_(exponent) {}

    float ApplyDecay(float initial_strength, Timestamp::Duration elapsed_time) const override {
        auto hours = std::chrono::duration_cast<std::chrono::hours>(elapsed_time).count();
        return initial_strength / std::pow(1.0f + hours / time_constant_, exponent_);
    }

private:
    float time_constant_;
    float exponent_;
};

// Step decay
class StepDecay : public IDecayFunction {
public:
    explicit StepDecay(float decay_factor = 0.9f, Timestamp::Duration step_size = std::chrono::hours(24))
        : decay_factor_(decay_factor), step_size_(step_size) {}

    float ApplyDecay(float initial_strength, Timestamp::Duration elapsed_time) const override {
        auto steps = elapsed_time / step_size_;
        return initial_strength * std::pow(decay_factor_, steps);
    }

private:
    float decay_factor_;
    Timestamp::Duration step_size_;
};

} // namespace dpan
```

**Acceptance Criteria**:
- [ ] All 3 decay types implemented
- [ ] Decay curves match theoretical models
- [ ] >15 unit tests pass
- [ ] Performance: <100ns per decay calculation

---

### Task 4.4.2: Implement Interference Model

**Duration**: 2 days (16 hours)
**Priority**: Medium
**Files to create**:
- `src/memory/interference.hpp`
- `src/memory/interference.cpp`
- `tests/memory/interference_test.cpp`

**Overview**:
Model memory interference where similar patterns compete for resources.

**Interference Formula**:
```
I(p1, p2) = similarity(p1, p2) × strength(p2)

Total interference on p1:
I_total(p1) = Σ I(p1, pi) for all similar patterns pi

Strength reduction:
s'(p1) = s(p1) × (1 - α × I_total(p1))

Where α = interference factor (0.1)
```

**Acceptance Criteria**:
- [ ] Interference calculated correctly
- [ ] Similar patterns compete appropriately
- [ ] >10 unit tests pass

---

### Task 4.4.3: Implement Sleep/Consolidation

**Duration**: 1 day (8 hours)
**Priority**: Low
**Files to create**:
- `src/memory/consolidation.hpp`
- `src/memory/consolidation.cpp`

**Overview**:
Simulate sleep-like consolidation where important patterns are strengthened and reorganized.

**Consolidation Process**:
1. Detect low-activity period
2. Identify important patterns (high utility, strong associations)
3. Strengthen important patterns
4. Reorganize memory (merge, form hierarchies)
5. Prune weak patterns more aggressively

**Acceptance Criteria**:
- [ ] Low-activity detection works
- [ ] Important patterns strengthened
- [ ] Reorganization preserves knowledge
- [ ] >8 unit tests pass

---

## Module 4.5: Integration & Testing

**Duration**: 1 week (40 hours)
**Dependencies**: All previous modules complete
**Owner**: Integration lead + QA engineer

### Overview
Integrate all memory management components and perform comprehensive testing.

---

### Task 4.5.1: Create MemoryManager Facade

**Duration**: 3 days (24 hours)
**Priority**: Critical
**Files to create**:
- `src/memory/memory_manager.hpp`
- `src/memory/memory_manager.cpp`
- `tests/memory/integration_test.cpp`

**System Architecture**:

```cpp
// File: src/memory/memory_manager.hpp
#pragma once

#include "memory/utility_calculator.hpp"
#include "memory/adaptive_thresholds.hpp"
#include "memory/tier_manager.hpp"
#include "memory/pattern_pruner.hpp"
#include "memory/association_pruner.hpp"
#include "memory/decay_functions.hpp"

namespace dpan {

class MemoryManager {
public:
    struct Config {
        UtilityCalculator::Config utility_config;
        AdaptiveThresholdManager::Config threshold_config;
        TierManager::Config tier_config;
        PatternPruner::Config pattern_pruner_config;
        AssociationPruner::Config association_pruner_config;

        // Global settings
        bool enable_automatic_pruning{true};
        bool enable_tier_transitions{true};
        bool enable_consolidation{true};

        std::chrono::hours pruning_interval{24};
        std::chrono::hours transition_interval{1};
    };

    explicit MemoryManager(const Config& config = {});

    // Initialize with pattern database and association matrix
    void Initialize(
        PatternDatabase* pattern_db,
        AssociationMatrix* assoc_matrix
    );

    // Trigger memory management operations
    void PerformMaintenance();
    void PerformPruning();
    void PerformTierTransitions();
    void PerformConsolidation();

    // Pattern access (with automatic tier management)
    std::optional<PatternNode> GetPattern(PatternID id);
    void StorePattern(const PatternNode& pattern);

    // Statistics
    struct MemoryStats {
        size_t total_patterns;
        size_t active_patterns;
        size_t warm_patterns;
        size_t cold_patterns;
        size_t archive_patterns;

        size_t total_associations;
        size_t pruned_patterns_count;
        size_t pruned_associations_count;

        size_t total_memory_bytes;
        float memory_pressure;
        float current_threshold;

        TierManager::TierStats tier_stats;
    };

    MemoryStats GetStats() const;

    // Configuration
    void SetConfig(const Config& config);
    const Config& GetConfig() const { return config_; }

    // Control
    void Start();  // Start background threads
    void Stop();   // Stop background threads

private:
    Config config_;

    PatternDatabase* pattern_db_;
    AssociationMatrix* assoc_matrix_;

    UtilityCalculator utility_calc_;
    AccessTracker access_tracker_;
    AdaptiveThresholdManager threshold_mgr_;
    TierManager tier_mgr_;
    PatternPruner pattern_pruner_;
    AssociationPruner association_pruner_;

    std::unique_ptr<IDecayFunction> decay_function_;

    std::unique_ptr<std::thread> maintenance_thread_;
    std::atomic<bool> running_{false};

    void MaintenanceLoop();
};

} // namespace dpan
```

**Acceptance Criteria**:
- [ ] All components integrated correctly
- [ ] End-to-end workflow functional
- [ ] >30 integration tests pass
- [ ] Background threads operate reliably

---

### Task 4.5.2: Comprehensive Testing

**Duration**: 2 days (16 hours)
**Priority**: Critical

**Test Categories**:
1. **Unit Tests**: Component-level (200+ tests total)
2. **Integration Tests**: Multi-component interactions (30+ tests)
3. **Performance Tests**: Benchmarks for all operations (20+ benchmarks)
4. **Stress Tests**: Large-scale data (100M patterns)
5. **Endurance Tests**: Long-running stability (24+ hours)

**Example Integration Test**:

```cpp
TEST(MemoryManagerTest, EndToEndWorkflow) {
    MemoryManager manager;
    PatternDatabase pattern_db;
    AssociationMatrix assoc_matrix;

    manager.Initialize(&pattern_db, &assoc_matrix);
    manager.Start();

    // Create many patterns
    for (int i = 0; i < 100000; ++i) {
        PatternNode pattern = CreateTestPattern();
        manager.StorePattern(pattern);
    }

    // Simulate access patterns
    // ... access some patterns frequently, others rarely

    // Let memory management run
    std::this_thread::sleep_for(std::chrono::minutes(10));

    auto stats = manager.GetStats();

    // Verify tiering occurred
    EXPECT_GT(stats.active_patterns, 0u);
    EXPECT_GT(stats.warm_patterns, 0u);
    EXPECT_GT(stats.cold_patterns, 0u);

    // Verify pruning occurred
    EXPECT_LT(stats.total_patterns, 100000u);

    // Verify memory is bounded
    EXPECT_LT(stats.total_memory_bytes, 32ULL * 1024 * 1024 * 1024);  // <32GB

    manager.Stop();
}
```

**Acceptance Criteria**:
- [ ] >250 total tests pass
- [ ] >95% code coverage
- [ ] Zero memory leaks
- [ ] Thread-safe verified
- [ ] All performance targets met

---

## Mathematical Foundations

### Utility Theory

**Multi-Attribute Utility Function**:
```
U(p) = Σ w_i × u_i(p)

Where:
  w_i = weight for attribute i (Σw_i = 1)
  u_i(p) = utility of attribute i for pattern p ∈ [0,1]
```

### Memory Pressure Dynamics

**Pressure Response Function**:
```
T(P) = T_base × (1 + k × P)

Where:
  T = threshold
  P = memory pressure
  k = sensitivity factor

Stability analysis:
  dP/dt = α × (M_used - M_target) - β × T(P)

System is stable when β > α
```

### Forgetting Curve Analysis

**Ebbinghaus Retention**:
```
R(t) = e^(-t/S)

Where:
  R(t) = retention at time t
  S = strength of memory

Rehearsal effect:
  S_new = S_old + Δt × learning_rate
```

---

## Performance Optimization Guide

### Memory Efficiency

1. **Tiered Storage**: Keep only hot data in RAM
2. **Compression**: Use LZ4/Snappy for cold tiers
3. **Index Optimization**: Minimize metadata overhead
4. **Batch Operations**: Amortize allocation costs

### CPU Optimization

1. **Lock-Free Reads**: Use shared_mutex for concurrent access
2. **Batch Processing**: Process pruning in large batches
3. **Background Threads**: Offload maintenance to separate threads
4. **SIMD**: Vectorize utility calculations

### I/O Optimization

1. **Asynchronous I/O**: Non-blocking tier access
2. **Read-Ahead**: Prefetch associated patterns
3. **Write Buffering**: Batch writes to disk tiers
4. **Compression**: Reduce disk I/O volume

---

## Validation & Quality Assurance

### Performance Targets

| Operation | Target | Measured |
|-----------|--------|----------|
| Utility calculation | <500ns | TBD |
| Active tier access | <100ns | TBD |
| Warm tier access | <10µs | TBD |
| Cold tier access | <1ms | TBD |
| Tier transition (batch 1K) | <100ms | TBD |
| Pruning (1M patterns) | <10s | TBD |
| Memory usage (100M patterns) | <32GB | TBD |

### Quality Metrics

- **Knowledge Preservation**: >95% of important patterns retained
- **Memory Efficiency**: <320 bytes per pattern (average)
- **Tier Distribution**: Active(1%) / Warm(10%) / Cold(40%) / Archive(49%)
- **Access Latency**: p99 < 10ms for any pattern

---

## Troubleshooting & Debugging

### Common Issues

**Issue**: Memory usage growing unbounded
- **Check**: Pruning enabled and running
- **Check**: Tier transitions occurring
- **Solution**: Lower pruning threshold, increase transition frequency

**Issue**: Important patterns being pruned
- **Check**: Utility calculation weights
- **Check**: Access tracking working correctly
- **Solution**: Adjust weights, verify access recording

**Issue**: Slow tier transitions
- **Check**: Disk I/O bottleneck
- **Check**: Batch size too small
- **Solution**: Increase batch size, use faster storage

**Issue**: High cache miss rate
- **Check**: Cache size sufficient
- **Check**: Prefetching enabled
- **Solution**: Increase cache size, tune prefetch algorithm

---

## Conclusion

This Phase 3 implementation plan provides a complete, actionable roadmap with:

- **Detailed code implementations** for all memory management components
- **Mathematical foundations** for utility scoring and forgetting
- **Comprehensive test suites** (250+ tests total)
- **Performance targets** with optimization guidance
- **Multi-tier architecture** for scalable memory usage
- **Intelligent pruning** to maintain knowledge quality

### Estimated Completion

**Duration**: 6-8 weeks with 2-3 developers

**Weekly Milestones**:
- **Weeks 1-2**: Module 4.1 (Utility Scoring System)
- **Weeks 3-4**: Module 4.2 (Memory Hierarchy)
- **Weeks 5-6**: Module 4.3 (Pruning System)
- **Week 7**: Module 4.4 (Forgetting Mechanisms)
- **Week 8**: Module 4.5 (Integration & Testing)

### Success Criteria

- [ ] All 250+ unit tests pass
- [ ] Memory stays bounded under continuous learning
- [ ] All performance targets met
- [ ] >95% code coverage
- [ ] Zero memory leaks
- [ ] Thread-safe verified
- [ ] Documentation complete
- [ ] Integration with Phases 1-2 working

---

**Document Version**: 1.0
**Last Updated**: 2025-11-16
**Status**: Ready for Implementation
**Total Lines of Code (estimated)**: ~6,000 C++ (~3,000 implementation + ~3,000 tests)