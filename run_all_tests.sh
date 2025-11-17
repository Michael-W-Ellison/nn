#!/bin/bash

echo "========================================="
echo "DPAN Neural Network - Comprehensive Test Suite"
echo "========================================="
echo ""

TOTAL_PASSED=0
TOTAL_FAILED=0
FAILED_TESTS=""

# Test categories
CATEGORIES=(
  "build/tests/core"
  "build/tests/storage"
  "build/tests/storage/indices"
  "build/tests/similarity"
  "build/tests/discovery"
  "build/tests/association"
  "build/tests/memory"
)

for category in "${CATEGORIES[@]}"; do
  if [ -d "$category" ]; then
    echo "=== Category: $(basename $(dirname $category))/$(basename $category) ==="
    for test in "$category"/*test; do
      if [ -x "$test" ]; then
        test_name=$(basename "$test")
        echo -n "  Running $test_name... "

        # Run test and capture output
        output=$("$test" 2>&1)
        result=$(echo "$output" | grep -E "\[  PASSED  \]|\[  FAILED  \]" | tail -1)

        if echo "$output" | grep -q "FAILED"; then
          echo "❌ FAILED"
          TOTAL_FAILED=$((TOTAL_FAILED + 1))
          FAILED_TESTS="$FAILED_TESTS\n  - $test_name"
          # Show failure details
          echo "$output" | grep -A 5 "FAILED"
        elif echo "$output" | grep -q "PASSED"; then
          count=$(echo "$result" | grep -o "[0-9]*" | head -1)
          echo "✓ $count tests"
          TOTAL_PASSED=$((TOTAL_PASSED + count))
        else
          echo "? Unknown result"
        fi
      fi
    done
    echo ""
  fi
done

echo "========================================="
echo "SUMMARY"
echo "========================================="
echo "Total Passed: $TOTAL_PASSED tests"
echo "Total Failed: $TOTAL_FAILED test suites"
if [ $TOTAL_FAILED -gt 0 ]; then
  echo ""
  echo "Failed Tests:$FAILED_TESTS"
fi
echo ""
