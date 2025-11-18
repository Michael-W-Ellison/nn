#!/bin/bash

# Run test and capture where it hangs
./build/tests/storage/persistent_backend_test --gtest_print_time=1 2>&1 > /tmp/test_output.txt &
TEST_PID=$!

# Wait 15 seconds
sleep 15

if ps -p $TEST_PID > /dev/null 2>&1; then
    echo "Test still running after 15 seconds - likely hung"
    echo "Last output:"
    tail -30 /tmp/test_output.txt
    kill -9 $TEST_PID 2>/dev/null
    exit 124
else
    echo "Test completed"
    tail -20 /tmp/test_output.txt
    wait $TEST_PID
    exit_code=$?
    echo "Exit code: $exit_code"
    exit $exit_code
fi
