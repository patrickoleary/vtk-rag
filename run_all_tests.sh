#!/bin/bash
# Run all tests in the VTK RAG project

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    VTK RAG - Complete Test Suite                          ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
FAILED_FILES=()

# Test directories
TEST_DIRS=(
    "tests/llm-generation"
    "tests/grounding-prompting"
    "tests/post-processing"
    "tests/evaluation"
    "tests/integration"
    "tests/api_mcp"
    "tests/visual_testing"
)

echo "Running tests from tests/ directory..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Function to run tests in a directory
run_tests_in_dir() {
    local dir=$1
    echo "📂 Testing: $dir"
    echo "─────────────────────────────────────────────────────────────────────────────"
    
    # Check if this is visual_testing directory
    if [[ "$dir" == *"visual_testing"* ]]; then
        if [ -z "$RUN_VISUAL_TESTS" ]; then
            echo "  ⏭️  Skipping (requires Docker - set RUN_VISUAL_TESTS=1 to enable)"
            echo ""
            return
        fi
    fi
    
    for test_file in $dir/test_*.py; do
        if [ -f "$test_file" ]; then
            filename=$(basename "$test_file")
            echo -n "  ▶ $filename ... "
            
            # Run test and capture output
            if output=$(RUN_VISUAL_TESTS=$RUN_VISUAL_TESTS python "$test_file" 2>&1); then
                # Extract test count from output
                if [[ $output =~ Ran\ ([0-9]+)\ test ]]; then
                    count="${BASH_REMATCH[1]}"
                    TOTAL_TESTS=$((TOTAL_TESTS + count))
                    PASSED_TESTS=$((PASSED_TESTS + count))
                    echo "✅ PASSED ($count tests)"
                else
                    echo "✅ PASSED"
                fi
            else
                echo "❌ FAILED"
                FAILED_TESTS=$((FAILED_TESTS + 1))
                FAILED_FILES+=("$test_file")
                echo ""
                echo "  Error output:"
                echo "$output" | head -20 | sed 's/^/    /'
                echo ""
            fi
        fi
    done
    echo ""
}

# Run tests in each directory
for dir in "${TEST_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        run_tests_in_dir "$dir"
    fi
done

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                              TEST SUMMARY                                   "
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Total Tests Run:    $TOTAL_TESTS"
echo "  Tests Passed:       $PASSED_TESTS"
echo "  Tests Failed:       $FAILED_TESTS"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "  ✅ SUCCESS - All tests passed!"
    exit 0
else
    echo "  ❌ FAILURE - Some tests failed"
    echo ""
    echo "  Failed files:"
    for file in "${FAILED_FILES[@]}"; do
        echo "    • $file"
    done
    exit 1
fi
