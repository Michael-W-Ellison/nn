# Attention Mechanisms Implementation Plan

## Overview

This document outlines the implementation plan for adding **Attention Mechanisms** to DPAN, enabling the system to focus on important patterns and improve prediction quality through context-aware weighting.

**Current State**: Equal weight to all patterns during prediction
**Target State**: Dynamic attention-based weighting for improved predictions
**Priority**: Medium (from capability evaluation)
**Estimated Effort**: 4 weeks
**Impact**: Improved prediction quality and contextual relevance

---

## Problem Statement

### Current Limitations

1. **Equal Treatment**: All patterns are given equal consideration during prediction
2. **No Context Sensitivity**: Pattern importance doesn't vary based on current context
3. **Missing Relevance Scoring**: No mechanism to identify which patterns are most relevant
4. **Suboptimal Predictions**: Less relevant patterns can dilute prediction quality

### Example Current Behavior

```cpp
// Current prediction (from AssociationLearningSystem)
auto predictions = assoc_system_->Predict(input_pattern, k=5);

// All predictions weighted equally by association strength only
// No consideration of:
// - Pattern recency
// - Context similarity
// - Pattern importance
// - Historical accuracy
```

---

## Proposed Solution: Attention Mechanisms

### Core Concept

Implement an **attention layer** that computes dynamic importance weights for patterns based on:

1. **Query-Key Similarity**: How relevant is this pattern to the current input?
2. **Pattern Importance**: Intrinsic importance based on usage, confidence, associations
3. **Context Alignment**: How well does this pattern fit the current context?
4. **Multi-Head Attention**: Multiple attention perspectives for robustness

### Architecture

```
Input Pattern (Query)
         ↓
    [Attention Layer]
         ↓
   Compute Attention Weights
   - Context similarity
   - Pattern importance
   - Association strength
   - Historical performance
         ↓
   Apply Attention to Predictions
         ↓
   Weighted Predictions (Better Quality)
```

---

## Implementation Tasks

### Phase 1: Foundation (Week 1)

#### Task 1: Design Architecture
**Status**: Pending
**Effort**: 2 days

**Deliverables**:
- Architecture diagram
- Interface definitions
- Integration points with existing code

**Design Decisions**:
```cpp
// File: include/learning/attention_mechanism.hpp

namespace dpan {

// Core attention interface
class AttentionMechanism {
public:
    struct AttentionConfig {
        size_t num_heads = 4;              // Multi-head attention
        float temperature = 1.0f;          // Softmax temperature
        bool use_context = true;           // Context-aware attention
        bool use_importance = true;        // Pattern importance weighting
        std::string attention_type = "dot_product"; // or "additive"
    };

    // Compute attention weights for candidate patterns
    virtual std::map<PatternID, float> ComputeAttention(
        PatternID query,
        std::vector<PatternID> candidates,
        const ContextVector& context
    ) = 0;

    // Apply attention to predictions
    virtual std::vector<std::pair<PatternID, float>>
    ApplyAttention(
        PatternID query,
        std::vector<PatternID> predictions,
        const ContextVector& context
    ) = 0;
};

} // namespace dpan
```

#### Task 2: Create Base Class
**Status**: Pending
**Effort**: 1 day

**Implementation**:
```cpp
// File: src/learning/attention_mechanism.cpp

class BasicAttentionMechanism : public AttentionMechanism {
private:
    AttentionConfig config_;
    std::shared_ptr<PatternDatabase> pattern_db_;

public:
    std::map<PatternID, float> ComputeAttention(
        PatternID query,
        std::vector<PatternID> candidates,
        const ContextVector& context
    ) override {

        std::map<PatternID, float> attention_weights;

        // Get query pattern features
        auto query_pattern = pattern_db_->Retrieve(query);
        if (!query_pattern) return attention_weights;

        // Compute attention for each candidate
        std::vector<float> scores;
        for (auto candidate : candidates) {
            float score = ComputeAttentionScore(
                query_pattern.value(),
                candidate,
                context
            );
            scores.push_back(score);
        }

        // Apply softmax to get probabilities
        auto weights = Softmax(scores, config_.temperature);

        // Map weights to pattern IDs
        for (size_t i = 0; i < candidates.size(); ++i) {
            attention_weights[candidates[i]] = weights[i];
        }

        return attention_weights;
    }

private:
    float ComputeAttentionScore(
        const PatternNode& query,
        PatternID candidate,
        const ContextVector& context
    );

    std::vector<float> Softmax(
        const std::vector<float>& scores,
        float temperature
    );
};
```

### Phase 2: Core Mechanisms (Week 2)

#### Task 3: Pattern Importance Weighting
**Status**: Pending
**Effort**: 2 days

**Purpose**: Assign intrinsic importance scores to patterns

**Factors**:
1. **Access Frequency**: How often is this pattern activated?
2. **Confidence Score**: How confident are we in this pattern?
3. **Association Richness**: How many strong associations does it have?
4. **Prediction Success**: Historical accuracy when used for prediction

**Implementation**:
```cpp
class PatternImportanceCalculator {
public:
    struct ImportanceFactors {
        float frequency_score;      // 0-1 based on access count
        float confidence_score;     // Pattern's intrinsic confidence
        float association_score;    // Richness of associations
        float success_rate;         // Historical prediction accuracy
    };

    float ComputeImportance(PatternID pattern) {
        auto factors = GetImportanceFactors(pattern);

        // Weighted combination
        float importance =
            0.3f * factors.frequency_score +
            0.3f * factors.confidence_score +
            0.2f * factors.association_score +
            0.2f * factors.success_rate;

        return importance;
    }

private:
    ImportanceFactors GetImportanceFactors(PatternID pattern);
};
```

#### Task 4: Context-Aware Attention
**Status**: Pending
**Effort**: 3 days

**Purpose**: Weight patterns based on context similarity

**Approach**:
```cpp
class ContextAwareAttention {
public:
    float ComputeContextSimilarity(
        const ContextVector& query_context,
        PatternID candidate
    ) {
        // Get candidate's typical context
        auto candidate_contexts = GetHistoricalContexts(candidate);

        if (candidate_contexts.empty()) return 0.0f;

        // Compute similarity with query context
        float max_similarity = 0.0f;
        for (const auto& hist_context : candidate_contexts) {
            float sim = CosineSimilarity(query_context, hist_context);
            max_similarity = std::max(max_similarity, sim);
        }

        return max_similarity;
    }

private:
    std::vector<ContextVector> GetHistoricalContexts(PatternID pattern);

    float CosineSimilarity(
        const ContextVector& a,
        const ContextVector& b
    );
};
```

#### Task 5: Attention Weight Computation
**Status**: Pending
**Effort**: 2 days

**Dot-Product Attention**:
```cpp
float DotProductAttention::ComputeScore(
    const PatternNode& query,
    const PatternNode& key
) {
    // Extract feature vectors
    auto query_features = ExtractFeatures(query);
    auto key_features = ExtractFeatures(key);

    // Compute dot product
    float score = DotProduct(query_features, key_features);

    // Scale by square root of dimension (as in Transformer)
    float d_k = std::sqrt(query_features.size());
    return score / d_k;
}
```

**Additive Attention** (alternative):
```cpp
float AdditiveAttention::ComputeScore(
    const PatternNode& query,
    const PatternNode& key
) {
    // W_q * query + W_k * key
    auto query_proj = LinearProjection(query_weights_, query);
    auto key_proj = LinearProjection(key_weights_, key);

    // tanh(combined)
    auto combined = Add(query_proj, key_proj);
    auto activated = Tanh(combined);

    // v^T * activated
    return DotProduct(v_weights_, activated);
}
```

### Phase 3: Advanced Features (Week 3)

#### Task 6: Multi-Head Attention
**Status**: Pending
**Effort**: 3 days

**Purpose**: Multiple attention perspectives for robustness

**Architecture**:
```cpp
class MultiHeadAttention {
public:
    struct Head {
        std::string name;
        AttentionMechanism* mechanism;
        float weight;  // Importance of this head
    };

    std::map<PatternID, float> ComputeAttention(
        PatternID query,
        std::vector<PatternID> candidates,
        const ContextVector& context
    ) {
        std::map<PatternID, std::vector<float>> head_outputs;

        // Compute attention from each head
        for (const auto& head : heads_) {
            auto weights = head.mechanism->ComputeAttention(
                query, candidates, context
            );

            // Store weighted output
            for (const auto& [pattern, weight] : weights) {
                head_outputs[pattern].push_back(weight * head.weight);
            }
        }

        // Combine head outputs
        std::map<PatternID, float> combined;
        for (const auto& [pattern, weights] : head_outputs) {
            combined[pattern] = Average(weights);
        }

        return combined;
    }

private:
    std::vector<Head> heads_;
};
```

**Example Heads**:
1. **Semantic Head**: Focus on content similarity
2. **Temporal Head**: Focus on recency and temporal patterns
3. **Structural Head**: Focus on pattern structure and composition
4. **Association Head**: Focus on strong associations

#### Task 7: Self-Attention
**Status**: Pending
**Effort**: 2 days

**Purpose**: Discover relationships between patterns

**Use Case**: Find related patterns even without explicit associations

```cpp
class SelfAttention {
public:
    // Compute attention between all patterns in a set
    std::map<std::pair<PatternID, PatternID>, float>
    ComputeSelfAttention(std::vector<PatternID> patterns) {

        std::map<std::pair<PatternID, PatternID>, float> attention_matrix;

        // For each pattern pair
        for (auto query : patterns) {
            for (auto key : patterns) {
                if (query == key) continue;

                // Compute attention score
                float score = ComputeAttentionScore(query, key);
                attention_matrix[{query, key}] = score;
            }
        }

        // Normalize per query (softmax over keys)
        NormalizeAttentionMatrix(attention_matrix);

        return attention_matrix;
    }

    // Use self-attention to discover implicit relationships
    std::vector<PatternID> DiscoverRelatedPatterns(
        PatternID query,
        size_t top_k = 5
    ) {
        auto all_patterns = pattern_db_->FindAll();
        auto attention = ComputeSelfAttention(all_patterns);

        // Find patterns with highest attention from query
        std::vector<std::pair<PatternID, float>> scored;
        for (auto pattern : all_patterns) {
            if (pattern == query) continue;
            float score = attention[{query, pattern}];
            scored.push_back({pattern, score});
        }

        // Sort by score and return top-k
        std::sort(scored.begin(), scored.end(),
            [](auto& a, auto& b) { return a.second > b.second; });

        std::vector<PatternID> related;
        for (size_t i = 0; i < std::min(top_k, scored.size()); ++i) {
            related.push_back(scored[i].first);
        }

        return related;
    }
};
```

### Phase 4: Integration (Week 3-4)

#### Task 8: Integrate with Prediction Pipeline
**Status**: Pending
**Effort**: 2 days

**Current Prediction Flow**:
```cpp
// AssociationLearningSystem::Predict (current)
std::vector<PatternID> Predict(PatternID source, size_t k) {
    auto candidates = matrix_.GetOutgoingAssociations(source);

    // Sort by association strength
    std::sort(candidates, [](auto& a, auto& b) {
        return a->GetStrength() > b->GetStrength();
    });

    // Return top-k
    return TopK(candidates, k);
}
```

**Enhanced with Attention**:
```cpp
// AssociationLearningSystem::PredictWithAttention (new)
std::vector<PatternID> PredictWithAttention(
    PatternID source,
    size_t k,
    const ContextVector& context
) {
    // Get association-based candidates
    auto candidates = matrix_.GetOutgoingAssociations(source);

    std::vector<PatternID> candidate_ids;
    std::map<PatternID, float> assoc_strengths;
    for (auto* edge : candidates) {
        candidate_ids.push_back(edge->GetTarget());
        assoc_strengths[edge->GetTarget()] = edge->GetStrength();
    }

    // Compute attention weights
    auto attention_weights = attention_->ComputeAttention(
        source, candidate_ids, context
    );

    // Combine association strength and attention
    std::vector<std::pair<PatternID, float>> scored;
    for (auto id : candidate_ids) {
        float assoc = assoc_strengths[id];
        float attention = attention_weights[id];

        // Weighted combination
        float combined_score =
            0.6f * assoc +      // Association strength
            0.4f * attention;   // Attention weight

        scored.push_back({id, combined_score});
    }

    // Sort by combined score
    std::sort(scored.begin(), scored.end(),
        [](auto& a, auto& b) { return a.second > b.second; });

    // Return top-k
    std::vector<PatternID> result;
    for (size_t i = 0; i < std::min(k, scored.size()); ++i) {
        result.push_back(scored[i].first);
    }

    return result;
}
```

#### Task 9: Attention Visualization
**Status**: Pending
**Effort**: 2 days

**CLI Commands**:
```cpp
// New CLI command: /attention <pattern>
void DPANCli::ShowAttention(const std::string& text) {
    auto pattern_id = GetPatternForText(text);
    if (!pattern_id) {
        std::cout << "Unknown pattern\n";
        return;
    }

    // Get predictions with attention
    auto predictions = assoc_system_->PredictWithAttention(
        pattern_id.value(), 10, current_context_
    );

    // Get attention weights
    auto attention_weights = attention_->ComputeAttention(
        pattern_id.value(), predictions, current_context_
    );

    // Display
    std::cout << "\nAttention Analysis for \"" << text << "\":\n";
    std::cout << "=========================================\n\n";

    for (size_t i = 0; i < predictions.size(); ++i) {
        auto pred = predictions[i];
        auto pred_text = GetTextForPattern(pred);
        float attention = attention_weights[pred];

        // Visual bar
        std::string bar(static_cast<int>(attention * 50), '█');

        std::cout << i+1 << ". \"" << pred_text.value_or("<unknown>")
                 << "\"\n";
        std::cout << "   Attention: [" << bar << "] "
                 << std::fixed << std::setprecision(3) << attention << "\n";
    }
}
```

### Phase 5: Testing & Optimization (Week 4)

#### Task 10-11: Testing
**Status**: Pending
**Effort**: 3 days

**Unit Tests** (`tests/learning/attention_test.cpp`):
```cpp
TEST(AttentionTest, ComputesNormalizedWeights) {
    BasicAttentionMechanism attention;

    PatternID query(1);
    std::vector<PatternID> candidates = {
        PatternID(2), PatternID(3), PatternID(4)
    };

    auto weights = attention.ComputeAttention(query, candidates, {});

    // Weights should sum to 1.0
    float sum = 0.0f;
    for (auto [id, weight] : weights) {
        sum += weight;
        EXPECT_GE(weight, 0.0f);
        EXPECT_LE(weight, 1.0f);
    }
    EXPECT_NEAR(sum, 1.0f, 0.001f);
}

TEST(AttentionTest, ContextAwareWeighting) {
    ContextAwareAttention attention;

    // Create patterns with different contexts
    PatternID query(1);
    PatternID similar_context(2);   // High context similarity
    PatternID diff_context(3);      // Low context similarity

    ContextVector query_ctx = CreateContext({{"topic", 1.0f}});

    auto weights = attention.ComputeAttention(
        query, {similar_context, diff_context}, query_ctx
    );

    // Pattern with similar context should have higher weight
    EXPECT_GT(weights[similar_context], weights[diff_context]);
}
```

**Integration Tests**:
```cpp
TEST(AttentionIntegrationTest, ImprovesPredictionQuality) {
    // Create test scenario
    AssociationLearningSystem system;
    AttentionMechanism* attention = new BasicAttentionMechanism();

    // Train on sequences
    // ...

    // Compare predictions with and without attention
    auto predictions_no_attention = system.Predict(query, 5);
    auto predictions_with_attention = system.PredictWithAttention(
        query, 5, context
    );

    // Measure prediction quality
    float quality_no_attention = MeasureQuality(predictions_no_attention);
    float quality_with_attention = MeasureQuality(predictions_with_attention);

    // Attention should improve quality
    EXPECT_GT(quality_with_attention, quality_no_attention);
}
```

#### Task 12: Performance Benchmarking
**Status**: Pending
**Effort**: 1 day

**Metrics to Measure**:
1. **Latency Impact**: Overhead of attention computation
2. **Memory Usage**: Additional memory for attention weights
3. **Prediction Quality**: Accuracy improvement
4. **Scalability**: Performance with large candidate sets

```cpp
void BenchmarkAttention() {
    // Measure latency
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; ++i) {
        attention.ComputeAttention(query, candidates, context);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end - start
    ).count();

    std::cout << "Average latency: " << duration / 1000.0 << " μs\n";

    // Measure quality improvement
    // ...
}
```

**Optimization Targets**:
- Attention computation: < 100μs per prediction
- Memory overhead: < 10% increase
- Prediction quality: > 15% improvement

#### Task 13: Configuration
**Status**: Pending
**Effort**: 1 day

**Config File** (`config/attention.yaml`):
```yaml
attention:
  enabled: true

  mechanism:
    type: "multi_head"  # or "basic", "context_aware"
    num_heads: 4
    temperature: 1.0

  importance_weighting:
    enabled: true
    weights:
      frequency: 0.3
      confidence: 0.3
      associations: 0.2
      success_rate: 0.2

  context_awareness:
    enabled: true
    similarity_threshold: 0.5

  multi_head:
    heads:
      - name: "semantic"
        type: "dot_product"
        weight: 0.4
      - name: "temporal"
        type: "recency_based"
        weight: 0.3
      - name: "structural"
        type: "composition_aware"
        weight: 0.2
      - name: "association"
        type: "strength_based"
        weight: 0.1

  combination:
    association_weight: 0.6  # Weight of association strength
    attention_weight: 0.4     # Weight of attention scores
```

#### Task 14: Documentation
**Status**: Pending
**Effort**: 2 days

**Documentation Sections**:
1. **Architecture Overview**
2. **API Reference**
3. **Configuration Guide**
4. **Usage Examples**
5. **Performance Characteristics**
6. **Comparison with Baseline**

#### Task 15: CLI Integration
**Status**: Pending
**Effort**: 1 day

**New Commands**:
```
/attention <text>       - Show attention weights for predictions
/attention config       - Show attention configuration
/attention enable       - Enable attention mechanism
/attention disable      - Disable attention mechanism
/attention heads        - Show multi-head attention breakdown
```

---

## Expected Benefits

### 1. Improved Prediction Quality

**Before** (Equal weighting):
```
Input: "Hello"
Predictions:
  1. "Hi" [0.85]
  2. "Hey there" [0.72]
  3. "Goodbye" [0.68]  ← Less relevant but high association
  4. "How are you?" [0.65]
```

**After** (Attention-weighted):
```
Input: "Hello"
Context: {social: 1.0, greeting: 1.0}

Predictions (with attention):
  1. "Hi" [0.91]              ← Boosted by context
  2. "How are you?" [0.88]    ← Boosted by relevance
  3. "Hey there" [0.76]
  4. "Good morning" [0.71]    ← New, context-relevant

"Goodbye" dropped due to low context relevance
```

### 2. Context Sensitivity

Attention enables different predictions based on context:

**Context 1: Greeting**
```
"Hello" → "How are you?"  [high attention]
```

**Context 2: Farewell**
```
"Hello" → "Goodbye"  [high attention]
```

### 3. Dynamic Importance

Patterns gain/lose importance based on:
- Usage frequency
- Prediction success
- Context relevance

### 4. Robustness via Multi-Head

Multiple perspectives reduce over-fitting to single aspect:
- Semantic similarity
- Temporal patterns
- Structural composition
- Association strength

---

## Success Metrics

### Quantitative

1. **Prediction Accuracy**: +15-20% improvement
2. **Context Relevance**: +25% in contextual scenarios
3. **Latency**: < 100μs overhead
4. **Memory**: < 10% increase

### Qualitative

1. **More relevant predictions** in context
2. **Better handling of ambiguity**
3. **Improved conversation flow**
4. **Reduced irrelevant suggestions**

---

## Risks & Mitigation

### Risk 1: Performance Overhead
**Mitigation**:
- Cache attention computations
- Use approximate methods for large candidate sets
- Make attention optional (can be disabled)

### Risk 2: Over-Fitting
**Mitigation**:
- Use multi-head attention for diversity
- Regularize attention weights
- Maintain baseline association strength

### Risk 3: Complexity
**Mitigation**:
- Start with simple dot-product attention
- Add complexity incrementally
- Comprehensive testing at each stage

---

## Dependencies

### Existing Systems
- PatternEngine (pattern retrieval)
- AssociationMatrix (candidate generation)
- ContextVector (context representation)

### New Components
- AttentionMechanism (core implementation)
- PatternImportanceCalculator
- ContextAwareAttention
- MultiHeadAttention
- SelfAttention

---

## Timeline

**Week 1**: Foundation
- Architecture design
- Base class implementation
- Initial testing

**Week 2**: Core Mechanisms
- Pattern importance weighting
- Context-aware attention
- Attention computation

**Week 3**: Advanced Features
- Multi-head attention
- Self-attention
- Integration

**Week 4**: Testing & Refinement
- Unit tests
- Integration tests
- Performance optimization
- Documentation

**Total**: 4 weeks (as estimated)

---

## References

### Papers
1. "Attention Is All You Need" (Vaswani et al., 2017)
2. "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2014)
3. "Effective Approaches to Attention-based Neural Machine Translation" (Luong et al., 2015)

### Similar Implementations
- Transformer architecture (NLP)
- Graph Attention Networks (GNNs)
- Memory Networks with Attention

---

## Next Steps

1. ✅ Create TODO list (DONE)
2. ✅ Write implementation plan (DONE)
3. ⏳ Review and approve architecture
4. ⏳ Begin Phase 1: Foundation
5. ⏳ Iterative development following plan

---

## Conclusion

Attention mechanisms will significantly enhance DPAN's prediction quality by enabling context-aware, importance-weighted pattern selection. The phased implementation approach ensures incremental progress with testing at each stage.

**Start Date**: TBD
**Target Completion**: 4 weeks from start
**Owner**: TBD
