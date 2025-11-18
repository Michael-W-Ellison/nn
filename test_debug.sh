#!/bin/bash

echo "Running persistent_backend_test with monitoring..."

stdbuf -oL ./build/tests/storage/persistent_backend_test 2>&1 &
TEST_PID=$!

# Monitor for 20 seconds
for i in {1..20}; do
    sleep 1
    if ! ps -p $TEST_PID > /dev/null 2>&1; then
        echo "Test completed after $i seconds"
        wait $TEST_PID
        exit_code=$?
        echo "Exit code: $exit_code"
        exit $exit_code
    fi
    echo "Still running... ($i seconds)"
done

echo "=== TEST HANGING AFTER 20 SECONDS ==="
echo "Attempting to get stack trace..."
gdb -batch -ex "attach $TEST_PID" -ex "thread apply all bt" -ex "detach" -ex "quit" 2>&1 | head -50
kill -9 $TEST_PID 2>/dev/null
echo "Test killed"
exit 124
