# DPAN Association Learning Performance Guide

## Overview

This guide provides optimization strategies and performance tuning recommendations for the DPAN Association Learning system. Follow these guidelines to achieve optimal performance for your specific use case.

## Configuration Tuning

### Co-Occurrence Tracking

```cpp
CoOccurrenceTracker::Config config;
config.window_size = std::chrono::seconds(10);  // Temporal window
config.min_co_occurrences = 3;                  // Minimum count
```

**Tuning Guidelines:**
- **Smaller window** (1-5s): Real-time, fast-changing patterns
- **Medium window** (10-30s): General-purpose learning
- **Larger window** (60s+): Long-term relationship discovery
- **Higher min_co_occurrences**: Reduces noise, increases quality

### Association Formation Rules

```cpp
AssociationFormationRules::Config config;
config.min_co_occurrences = 3;          // Minimum observations
config.min_chi_squared = 3.84f;         // Statistical significance (p<0.05)
config.initial_strength = 0.5f;         // Starting strength
```

**Tuning Guidelines:**
- **Strict rules** (min_co_occurrences=5, chi²=6.63): High-quality, fewer associations
- **Balanced** (min_co_occurrences=3, chi²=3.84): Recommended default
- **Lenient** (min_co_occurrences=2, chi²=2.71): More associations, some weak

### Reinforcement Learning

```cpp
ReinforcementManager::Config config;
config.learning_rate = 0.1f;            // Rate of strength updates
config.decay_rate = 0.01f;              // Decay per time unit
config.min_strength = 0.01f;            // Pruning threshold
```

**Tuning Guidelines:**
- **High learning_rate** (0.2-0.5): Fast adaptation, less stable
- **Medium learning_rate** (0.05-0.15): Balanced (recommended)
- **Low learning_rate** (0.01-0.05): Slow learning, very stable
- **Decay_rate**: 0.001 (slow decay) to 0.1 (fast decay)

### Competitive Learning

```cpp
CompetitiveLearner::Config config;
config.competition_factor = 0.3f;       // Winner boost/loser suppression
config.min_competing_associations = 2;  // Minimum for competition
```

**Tuning Guidelines:**
- **Low factor** (0.1-0.2): Gentle competition, preserves alternatives
- **Medium factor** (0.2-0.3): Recommended for most cases
- **High factor** (0.4-0.5): Aggressive competition, sparse associations

### Strength Normalization

```cpp
StrengthNormalizer::Config config;
config.min_strength_threshold = 0.01f;  // Pruning threshold
config.l1_penalty = 0.0f;               // L1 regularization (optional)
```

**Tuning Guidelines:**
- **Higher threshold** (0.05-0.1): Aggressive pruning, fewer associations
- **Lower threshold** (0.001-0.01): Keeps weak associations
- **L1 penalty**: 0.001-0.01 for sparsity regularization

## Performance Benchmarks

Based on comprehensive benchmark suite (see `tests/benchmarks/`):

### Association Matrix Operations

| Operation | Scale | Expected Performance |
|-----------|-------|---------------------|
| AddAssociation | 10k | < 100ms (>100k ops/sec) |
| GetAssociation | 10k | < 50ms (>200k ops/sec) |
| UpdateAssociation | 10k | < 100ms (>100k ops/sec) |
| PropagateActivation | 1k | < 500ms depth=3 |
| Large-scale | 100k | < 10s for creation |

### Co-Occurrence Tracking

| Operation | Scale | Expected Performance |
|-----------|-------|---------------------|
| RecordActivation | 10k | < 100ms (>100k ops/sec) |
| GetCoOccurrenceCount | 10k | < 50ms (>200k ops/sec) |
| Large-scale tracking | 100k | < 5s |

### Learning System

| Operation | Scale | Expected Performance |
|-----------|-------|---------------------|
| RecordPatternActivation | 10k | < 200ms (>50k ops/sec) |
| Predict | 10k | < 100ms (>100k ops/sec) |
| Reinforce | 10k | < 500ms (>20k ops/sec) |
| PerformMaintenance | 1k assoc | < 100ms |

## Memory Optimization

### Capacity Planning

```cpp
AssociationLearningSystem::Config config;
config.association_capacity = 1000000;  // Reserve space
```

**Guidelines:**
- Reserve capacity upfront to avoid reallocations
- Estimate: `associations ≈ patterns² × sparsity_factor`
- Typical sparsity: 0.001-0.01 (0.1%-1% connectivity)
- Memory per association: ~100-200 bytes

### Pruning Strategies

1. **Automatic Maintenance**:
```cpp
config.enable_auto_maintenance = true;
config.auto_decay_interval = std::chrono::minutes(5);
```

2. **Manual Pruning**:
```cpp
auto stats = system.PerformMaintenance();
// Prunes weak associations below min_strength threshold
```

3. **Periodic Compaction**:
- Run maintenance every N operations or time interval
- Remove associations with strength < threshold
- Apply competitive learning to reduce redundancy

## CPU Optimization

### Thread Safety

The association system uses fine-grained locking for concurrency:

```cpp
// Thread-safe operations
std::thread t1([&]() { system.RecordPatternActivation(p1, ctx); });
std::thread t2([&]() { system.Predict(p2, 5); });
```

**Concurrency Guidelines:**
- Reads can proceed concurrently (shared_mutex)
- Writes use exclusive locks
- Batch operations when possible to reduce lock contention
- Expected throughput: 10k+ concurrent ops/sec (see `concurrency_tests`)

### Batch Processing

```cpp
// Instead of individual activations:
for (auto pattern : patterns) {
    system.RecordPatternActivation(pattern, ctx);
}

// Use batch:
system.RecordPatternActivations(patterns, ctx);
```

**Benefits:**
- Amortizes lock overhead
- Better cache locality
- 2-3x throughput improvement

## Scalability Guidelines

### Small-Scale (<10k patterns, <100k associations)
- Default configuration works well
- Enable auto-maintenance
- No special tuning needed

### Medium-Scale (10k-100k patterns, 100k-1M associations)
- Increase `association_capacity` reservation
- Use stricter formation rules to control growth
- Enable periodic maintenance (every 1000 operations)
- Consider lower `min_strength_threshold` (0.05)

### Large-Scale (>100k patterns, >1M associations)
- **Critical**: Use strict formation rules (chi²>5, min_co>5)
- Aggressive pruning (min_strength>0.1)
- Higher competition_factor (0.3-0.4)
- Periodic maintenance (every 100 operations)
- Consider distributed architecture (future work)

### Stress Test Results

From `tests/benchmarks/stress_tests.cpp`:

| Test | Scale | Time | Result |
|------|-------|------|--------|
| Million associations | 1M | ~60s | ✓ Handles large networks |
| Dense connectivity | 1000×100 | ~15s | ✓ High-density graphs |
| High-frequency updates | 100k | ~3s | ✓ Rapid modifications |
| Continuous learning | 100k activations | ~25s | ✓ Long-running sessions |

## Profiling and Monitoring

### Statistics Monitoring

```cpp
auto stats = system.GetStatistics();
std::cout << "Total associations: " << stats.total_associations << std::endl;
std::cout << "Average strength: " << stats.average_strength << std::endl;
std::cout << "Max strength: " << stats.max_strength << std::endl;
```

**Key Metrics:**
- `total_associations`: Monitor growth rate
- `average_strength`: Should stay in 0.3-0.7 range
- `formations_count`: New associations formed
- `reinforcements_count`: Learning activity

### Performance Monitoring

Track these metrics in production:
1. **Throughput**: Operations per second
2. **Latency**: P50, P95, P99 response times
3. **Memory**: RSS, association count
4. **Quality**: Average strength, prediction accuracy

## Common Performance Issues

### Issue: Slow Predictions

**Symptoms:** `Predict()` takes >100ms

**Solutions:**
- Reduce max depth in PropagateActivation
- Prune weak associations
- Apply competitive learning
- Check association fan-out (avg outgoing per pattern)

### Issue: High Memory Usage

**Symptoms:** RSS growing unbounded

**Solutions:**
- Enable auto-maintenance
- Lower `min_strength_threshold`
- Use stricter formation rules
- Increase `decay_rate`

### Issue: Slow Learning

**Symptoms:** Associations not strengthening

**Solutions:**
- Increase `learning_rate`
- Lower `min_chi_squared` threshold
- Reduce `min_co_occurrences`
- Check co-occurrence window size

### Issue: Too Many Weak Associations

**Symptoms:** Most associations < 0.2 strength

**Solutions:**
- Apply competitive learning
- Increase pruning threshold
- Use stricter formation rules
- Enable normalization

## Best Practices

1. **Start Conservative**: Use strict rules, relax as needed
2. **Monitor Metrics**: Track statistics regularly
3. **Tune Incrementally**: Change one parameter at a time
4. **Profile First**: Measure before optimizing
5. **Test at Scale**: Use benchmark suite for validation
6. **Enable Maintenance**: Always use auto-maintenance in production
7. **Batch Operations**: Group activations when possible
8. **Reserve Capacity**: Pre-allocate for known workloads

## Example Configurations

### Real-Time Learning (Fast-Changing Data)
```cpp
config.co_occurrence.window_size = std::chrono::seconds(2);
config.formation.min_co_occurrences = 2;
config.reinforcement.learning_rate = 0.2f;
config.competition.competition_factor = 0.3f;
config.enable_auto_maintenance = true;
```

### Knowledge Graph (Stable, High-Quality)
```cpp
config.co_occurrence.window_size = std::chrono::seconds(60);
config.formation.min_co_occurrences = 5;
config.formation.min_chi_squared = 6.63f;
config.reinforcement.learning_rate = 0.05f;
config.competition.competition_factor = 0.2f;
```

### Memory-Constrained (Mobile/Embedded)
```cpp
config.association_capacity = 10000;
config.formation.min_co_occurrences = 5;
config.normalization.min_strength_threshold = 0.1f;
config.competition.competition_factor = 0.4f;
config.enable_auto_maintenance = true;
config.auto_decay_interval = std::chrono::minutes(1);
```

## References

- Benchmark suite: `tests/benchmarks/`
- API documentation: See Doxygen-generated docs
- Examples: `examples/association/`
- Phase 2 Plan: `docs/Phase2_Detailed_Implementation_Plan.md`
