# DPAN Association Learning Integration Guide

## Overview

This guide explains how to integrate the DPAN Association Learning system (Phase 2) with the Pattern Engine (Phase 1) and prepare for Discovery modules (Phase 3).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
┌────────▼────────┐  ┌────────▼────────┐  ┌───────▼──────────┐
│   Discovery     │  │   Association    │  │  Similarity      │
│   (Phase 3)     │  │   Learning       │  │  (Phase 1)       │
│                 │  │   (Phase 2)      │  │                  │
└────────┬────────┘  └────────┬─────────┘  └───────┬──────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Pattern Engine    │
                    │  (Phase 1 Core)    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Storage Backend   │
                    │  (Phase 1)         │
                    └────────────────────┘
```

## Quick Start Integration

### Basic Integration

```cpp
#include "core/pattern_engine.hpp"
#include "storage/memory_backend.hpp"
#include "association/association_learning_system.hpp"

// 1. Create storage backend
MemoryBackend::Config storage_config;
auto storage = std::make_shared<MemoryBackend>(storage_config);

// 2. Create pattern engine
PatternEngine::Config engine_config;
PatternEngine engine(storage, engine_config);

// 3. Create association learning system
AssociationLearningSystem::Config assoc_config;
AssociationLearningSystem associations(assoc_config);

// 4. Use together
ContextVector context;
std::vector<float> data = {1.0, 2.0, 3.0};
FeatureVector features(data);
PatternData pattern_data = PatternData::FromFeatures(features, DataModality::NUMERIC);

// Store pattern
PatternNode pattern(PatternID::Generate(), pattern_data, PatternType::ATOMIC);
storage->Store(pattern);

// Record activation in association system
associations.RecordPatternActivation(pattern.GetID(), context);
```

## Integration Patterns

### Pattern 1: Shared Pattern Storage

Use a common storage backend for both pattern data and associations:

```cpp
class IntegratedDPAN {
public:
    IntegratedDPAN() {
        // Shared storage
        storage_ = std::make_shared<MemoryBackend>(MemoryBackend::Config{});

        // Pattern engine for pattern operations
        engine_ = std::make_unique<PatternEngine>(storage_, PatternEngine::Config{});

        // Association system for learning
        assoc_system_ = std::make_unique<AssociationLearningSystem>(
            AssociationLearningSystem::Config{});
    }

    void LearnPattern(const std::vector<float>& data) {
        // 1. Create pattern
        FeatureVector features(data);
        PatternData pattern_data = PatternData::FromFeatures(features, DataModality::NUMERIC);
        PatternNode pattern(PatternID::Generate(), pattern_data, PatternType::ATOMIC);

        // 2. Store in shared backend
        storage_->Store(pattern);

        // 3. Record in association system
        ContextVector context;
        assoc_system_->RecordPatternActivation(pattern.GetID(), context);

        // 4. Track for prediction
        recent_patterns_.push_back(pattern.GetID());
    }

    std::vector<PatternID> PredictNext(size_t k) {
        if (recent_patterns_.empty()) return {};
        return assoc_system_->Predict(recent_patterns_.back(), k);
    }

private:
    std::shared_ptr<MemoryBackend> storage_;
    std::unique_ptr<PatternEngine> engine_;
    std::unique_ptr<AssociationLearningSystem> assoc_system_;
    std::vector<PatternID> recent_patterns_;
};
```

### Pattern 2: Event-Driven Learning

Integrate with event-driven architectures:

```cpp
class EventDrivenLearning {
public:
    using EventCallback = std::function<void(PatternID)>;

    EventDrivenLearning(AssociationLearningSystem& assoc_system)
        : assoc_system_(assoc_system) {

        // Subscribe to pattern events
        on_pattern_created_ = [this](PatternID id) {
            ContextVector context;
            assoc_system_.RecordPatternActivation(id, context);
        };

        on_pattern_matched_ = [this](PatternID predicted, PatternID actual) {
            bool correct = (predicted == actual);
            assoc_system_.Reinforce(predicted, actual, correct);
        };
    }

    void OnPatternCreated(PatternID id) {
        on_pattern_created_(id);

        // Make prediction for next pattern
        auto predictions = assoc_system_.Predict(id, 3);
        pending_predictions_[id] = predictions;
    }

    void OnPatternMatched(PatternID prev, PatternID current) {
        // Check if we predicted this
        if (pending_predictions_.count(prev)) {
            auto& predictions = pending_predictions_[prev];
            bool predicted = std::find(predictions.begin(),
                                      predictions.end(),
                                      current) != predictions.end();

            if (predicted && !predictions.empty()) {
                on_pattern_matched_(predictions[0], current);
            }
        }
    }

private:
    AssociationLearningSystem& assoc_system_;
    EventCallback on_pattern_created_;
    EventCallback on_pattern_matched_;
    std::map<PatternID, std::vector<PatternID>> pending_predictions_;
};
```

### Pattern 3: Similarity-Guided Association

Combine similarity metrics with association learning:

```cpp
#include "similarity/similarity_metric.hpp"
#include "similarity/geometric_similarity.hpp"

class SimilarityGuidedAssociations {
public:
    SimilarityGuidedAssociations(
        PatternDatabase& db,
        AssociationLearningSystem& assoc_system)
        : db_(db), assoc_system_(assoc_system) {

        // Use geometric similarity
        similarity_ = std::make_unique<GeometricSimilarity>();
    }

    void FormSimilarityAssociations(PatternID pattern, float threshold = 0.7f) {
        // Get pattern data
        auto pattern_node = db_.Retrieve(pattern);
        if (!pattern_node.has_value()) return;

        // Find similar patterns
        auto similar = FindSimilarPatterns(*pattern_node, threshold);

        // Create categorical associations
        auto& matrix = const_cast<AssociationMatrix&>(
            assoc_system_.GetAssociationMatrix());

        for (const auto& [similar_id, similarity] : similar) {
            AssociationEdge edge(pattern, similar_id,
                               AssociationType::CATEGORICAL,
                               similarity);
            matrix.AddAssociation(edge);
        }
    }

private:
    std::vector<std::pair<PatternID, float>>
    FindSimilarPatterns(const PatternNode& pattern, float threshold) {
        std::vector<std::pair<PatternID, float>> results;

        // Query all patterns (in practice, use spatial index)
        // Compare and keep those above threshold
        // ... implementation details ...

        return results;
    }

    PatternDatabase& db_;
    AssociationLearningSystem& assoc_system_;
    std::unique_ptr<SimilarityMetric> similarity_;
};
```

## Common Integration Scenarios

### Scenario 1: Time Series Prediction

```cpp
class TimeSeriesPredictor {
public:
    TimeSeriesPredictor() {
        AssociationLearningSystem::Config config;
        config.co_occurrence.window_size = std::chrono::seconds(5);
        config.formation.min_co_occurrences = 3;
        assoc_system_ = std::make_unique<AssociationLearningSystem>(config);
    }

    void ObserveValue(float value, Timestamp time) {
        // Discretize value into pattern
        PatternID pattern = DiscretizeValue(value);

        // Record with timestamp
        ContextVector context;
        assoc_system_->RecordPatternActivation(pattern, context);

        // Store observation
        observations_.push_back({pattern, time, value});
    }

    std::vector<float> PredictNext(size_t horizon) {
        if (observations_.empty()) return {};

        // Get last pattern
        PatternID last = observations_.back().pattern;

        // Predict next patterns
        auto predicted_patterns = assoc_system_->Predict(last, horizon);

        // Convert back to values
        std::vector<float> predictions;
        for (auto p : predicted_patterns) {
            predictions.push_back(PatternToValue(p));
        }

        return predictions;
    }

private:
    struct Observation {
        PatternID pattern;
        Timestamp time;
        float value;
    };

    PatternID DiscretizeValue(float value) {
        // Simple quantization (in practice, use proper discretization)
        int bucket = static_cast<int>(value * 10);

        if (value_to_pattern_.count(bucket) == 0) {
            value_to_pattern_[bucket] = PatternID::Generate();
        }

        return value_to_pattern_[bucket];
    }

    float PatternToValue(PatternID pattern) {
        for (const auto& [bucket, id] : value_to_pattern_) {
            if (id == pattern) {
                return bucket / 10.0f;
            }
        }
        return 0.0f;
    }

    std::unique_ptr<AssociationLearningSystem> assoc_system_;
    std::vector<Observation> observations_;
    std::map<int, PatternID> value_to_pattern_;
};
```

### Scenario 2: Sequence Learning

```cpp
class SequenceLearner {
public:
    void LearnSequence(const std::vector<PatternID>& sequence) {
        ContextVector context;

        // Record all activations
        for (const auto& pattern : sequence) {
            assoc_system_.RecordPatternActivation(pattern, context);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Form causal associations
        auto& matrix = const_cast<AssociationMatrix&>(
            assoc_system_.GetAssociationMatrix());

        for (size_t i = 0; i < sequence.size() - 1; ++i) {
            AssociationEdge edge(sequence[i], sequence[i + 1],
                               AssociationType::CAUSAL, 0.7f);
            matrix.AddAssociation(edge);
        }
    }

    std::vector<PatternID> CompleteSequence(PatternID start, size_t max_length) {
        std::vector<PatternID> sequence = {start};
        PatternID current = start;

        for (size_t i = 0; i < max_length; ++i) {
            auto next = assoc_system_.Predict(current, 1);
            if (next.empty()) break;

            sequence.push_back(next[0]);
            current = next[0];
        }

        return sequence;
    }

private:
    AssociationLearningSystem assoc_system_;
};
```

### Scenario 3: Anomaly Detection with Associations

```cpp
class AssociationBasedAnomalyDetector {
public:
    float DetectAnomaly(PatternID pattern) {
        // Get expected associations
        auto predictions = assoc_system_.Predict(pattern, 5);

        if (predictions.empty()) {
            return 1.0f; // Novel pattern = high anomaly score
        }

        // Calculate average association strength
        const auto& matrix = assoc_system_.GetAssociationMatrix();
        float total_strength = 0.0f;
        size_t count = 0;

        for (auto pred : predictions) {
            auto* edge = matrix.GetAssociation(pattern, pred);
            if (edge) {
                total_strength += edge->GetStrength();
                count++;
            }
        }

        if (count == 0) return 1.0f;

        float avg_strength = total_strength / count;

        // Low average strength = anomalous
        return 1.0f - avg_strength;
    }

private:
    AssociationLearningSystem assoc_system_;
};
```

## Thread Safety Considerations

The association system is thread-safe for concurrent operations:

```cpp
// Multiple threads can safely:
std::thread t1([&]() {
    assoc_system.RecordPatternActivation(p1, ctx);  // Write
});

std::thread t2([&]() {
    auto predictions = assoc_system.Predict(p2, 5);  // Read
});

std::thread t3([&]() {
    assoc_system.Reinforce(p3, p4, true);  // Write
});

t1.join();
t2.join();
t3.join();
```

**Guidelines:**
- Reads use shared locks (concurrent reads OK)
- Writes use exclusive locks (serialized)
- Batch operations when possible to reduce contention
- See `tests/benchmarks/concurrency_tests.cpp` for examples

## Migration from Phase 1

If you have existing Phase 1 code, migrate incrementally:

### Step 1: Add Association System

```cpp
// Before (Phase 1 only):
PatternEngine engine(storage, config);

// After (Phase 1 + Phase 2):
PatternEngine engine(storage, config);
AssociationLearningSystem associations;  // Add this
```

### Step 2: Record Activations

```cpp
// Whenever you create/match a pattern:
PatternID id = /* ... */;
ContextVector context;
associations.RecordPatternActivation(id, context);
```

### Step 3: Use Predictions

```cpp
// Use predictions to enhance discovery:
auto predictions = associations.Predict(current_pattern, 3);
// Use predictions to guide search, pre-fetch, etc.
```

## Preparing for Phase 3 (Discovery)

The association system prepares for discovery modules:

```cpp
// Discovery will use associations for:
// 1. Pattern extraction guided by strong associations
// 2. Matching optimization using predictions
// 3. Refinement based on association feedback

class DiscoveryIntegration {
public:
    void ExtractPatternsGuided(const AssociationLearningSystem& assoc) {
        // Use strong associations to guide extraction
        const auto& matrix = assoc.GetAssociationMatrix();

        // Find patterns with strong outgoing associations
        // These are likely important for pattern extraction
    }

    void MatchWithPrediction(PatternID query,
                            const AssociationLearningSystem& assoc) {
        // Get predicted patterns
        auto predictions = assoc.Predict(query, 10);

        // Prioritize matching against predicted patterns
        // This optimizes search performance
    }
};
```

## Best Practices

1. **Use Shared Storage**: One PatternDatabase for all components
2. **Consistent IDs**: Use PatternID for all pattern references
3. **Context Awareness**: Pass ContextVector when available
4. **Batch Operations**: Group activations/predictions when possible
5. **Monitor Statistics**: Track association growth and quality
6. **Enable Maintenance**: Use auto-maintenance in production
7. **Thread-Safe by Default**: Trust the internal synchronization
8. **Profile First**: Measure before optimizing integration

## Troubleshooting

### Issue: Associations Not Forming

**Check:**
- Co-occurrence window is appropriate for your data
- Patterns are being recorded with RecordPatternActivation
- Formation thresholds are not too strict
- Sufficient time has passed for co-occurrence

### Issue: Poor Prediction Quality

**Check:**
- Associations have been reinforced with Reinforce()
- Competitive learning is enabled
- Association strengths are reasonable (0.3-0.8 range)
- Enough training data has been provided

### Issue: Memory Growth

**Check:**
- Auto-maintenance is enabled
- Pruning threshold is appropriate
- Formation rules are not too lenient
- Decay is enabled

## Example Applications

See complete examples in `examples/association/`:
- `basic_learning.cpp`: Simple sequence learning
- `custom_formation_rules.cpp`: Knowledge graph with typed associations
- `activation_propagation_demo.cpp`: Semantic network activation

## API Reference

Key classes for integration:
- `AssociationLearningSystem`: Main integration point
- `AssociationMatrix`: Low-level association graph
- `PatternID`: Pattern identifier (from Phase 1)
- `ContextVector`: Contextual information
- `AssociationType`: Relationship types

See Doxygen-generated documentation for complete API details.

## References

- Phase 2 Implementation Plan: `docs/Phase2_Detailed_Implementation_Plan.md`
- Performance Guide: `docs/Association_Performance_Guide.md`
- Phase 1 Documentation: `docs/DPAN_Design_Document.md`
- Examples: `examples/association/`
