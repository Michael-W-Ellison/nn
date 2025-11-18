#!/bin/bash
# Quick test runner - tests individual modules

echo "===== DPAN Test-Driven Development Analysis ====="
echo ""

declare -A TESTS
declare -A RESULTS

# Association tests
TESTS[association_edge]="build/tests/association/association_edge_test"
TESTS[association_matrix]="build/tests/association/association_matrix_test"
TESTS[formation_rules]="build/tests/association/formation_rules_test"
TESTS[temporal_learner]="build/tests/association/temporal_learner_test"
TESTS[spatial_learner]="build/tests/association/spatial_learner_test"
TESTS[categorical_learner]="build/tests/association/categorical_learner_test"
TESTS[competitive_learner]="build/tests/association/competitive_learner_test"
TESTS[reinforcement]="build/tests/association/reinforcement_manager_test"
TESTS[co_occurrence]="build/tests/association/co_occurrence_tracker_test"
TESTS[strength_normalizer]="build/tests/association/strength_normalizer_test"
TESTS[learning_system]="build/tests/association/association_learning_system_test"

# Memory tests
TESTS[utility_calc]="build/tests/memory/utility_calculator_test"
TESTS[adaptive_thresh]="build/tests/memory/adaptive_thresholds_test"
TESTS[utility_track]="build/tests/memory/utility_tracker_test"
TESTS[memory_tier]="build/tests/memory/memory_tier_test"
TESTS[tier_manager]="build/tests/memory/tier_manager_test"
TESTS[tiered_storage]="build/tests/memory/tiered_storage_test"
TESTS[pattern_pruner]="build/tests/memory/pattern_pruner_test"
TESTS[assoc_pruner]="build/tests/memory/association_pruner_test"
TESTS[consolidator]="build/tests/memory/consolidator_test"
TESTS[decay]="build/tests/memory/decay_functions_test"
TESTS[interference]="build/tests/memory/interference_test"
TESTS[sleep_consol]="build/tests/memory/sleep_consolidator_test"
TESTS[memory_mgr]="build/tests/memory/memory_manager_test"

# Discovery tests
TESTS[anomaly]="build/tests/discovery/anomaly_detector_test"
TESTS[novelty]="build/tests/discovery/novelty_detector_test"
TESTS[pattern_disc]="build/tests/discovery/pattern_discovery_test"

# Similarity tests
TESTS[cosine]="build/tests/similarity/cosine_similarity_test"
TESTS[euclidean]="build/tests/similarity/euclidean_similarity_test"
TESTS[jaccard]="build/tests/similarity/jaccard_similarity_test"

total_passed=0
total_failed=0

for name in "${!TESTS[@]}"; do
  test_path="${TESTS[$name]}"
  if [ -x "$test_path" ]; then
    result=$(timeout 20 "$test_path" 2>&1 | grep "PASSED\|FAILED" | tail -1)
    if echo "$result" | grep -q "PASSED"; then
      count=$(echo "$result" | grep -o "[0-9]*" | head -1)
      echo "✓ $name: $count tests"
      total_passed=$((total_passed + count))
    elif echo "$result" | grep -q "FAILED"; then
      echo "✗ $name: FAILED"
      total_failed=$((total_failed + 1))
    else
      echo "? $name: TIMEOUT or ERROR"
      total_failed=$((total_failed + 1))
    fi
  fi
done

echo ""
echo "========================================="
echo "RESULTS: $total_passed tests passed"
if [ $total_failed -gt 0 ]; then
  echo "WARNING: $total_failed test suites failed/timed out"
fi
echo "========================================="
