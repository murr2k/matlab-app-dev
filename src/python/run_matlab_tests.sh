#!/bin/bash
# MATLAB Engine Test Runner for WSL
# Automatically detects and uses appropriate Python environment

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}MATLAB Engine API Test Runner${NC}"
echo -e "${BLUE}========================================${NC}"

# Detect environment and Python availability
PYTHON_CMD=""
MATLAB_AVAILABLE=false
ENVIRONMENT=""

# Option 1: Check for Windows Python with MATLAB Engine (Option 4 from documentation)
WINDOWS_PYTHON="/mnt/c/Python312/python.exe"
if [ -f "$WINDOWS_PYTHON" ]; then
    echo -e "${GREEN}[✓]${NC} Found Windows Python at $WINDOWS_PYTHON"
    # Test if MATLAB Engine is available
    if $WINDOWS_PYTHON -c "import matlab.engine" 2>/dev/null; then
        echo -e "${GREEN}[✓]${NC} MATLAB Engine API available in Windows Python"
        PYTHON_CMD="$WINDOWS_PYTHON"
        MATLAB_AVAILABLE=true
        ENVIRONMENT="Windows Python from WSL"
    else
        echo -e "${YELLOW}[!]${NC} Windows Python found but MATLAB Engine not installed"
    fi
fi

# Option 2: Check for native WSL Python with MATLAB Engine
if [ -z "$PYTHON_CMD" ]; then
    if command -v python3 &> /dev/null; then
        echo -e "${GREEN}[✓]${NC} Found WSL Python"
        if python3 -c "import matlab.engine" 2>/dev/null; then
            echo -e "${GREEN}[✓]${NC} MATLAB Engine API available in WSL Python"
            PYTHON_CMD="python3"
            MATLAB_AVAILABLE=true
            ENVIRONMENT="Native WSL Python"
        else
            echo -e "${YELLOW}[!]${NC} WSL Python found but MATLAB Engine not installed"
            # Fall back to mock tests
            PYTHON_CMD="python3"
            MATLAB_AVAILABLE=false
            ENVIRONMENT="WSL Python (Mock Tests Only)"
        fi
    fi
fi

# Display detected configuration
echo -e "\n${BLUE}Configuration:${NC}"
echo -e "  Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "  Python Command: ${YELLOW}$PYTHON_CMD${NC}"
echo -e "  MATLAB Available: $([ "$MATLAB_AVAILABLE" = true ] && echo -e "${GREEN}Yes${NC}" || echo -e "${YELLOW}No (Mock Tests)${NC}")"

# Function to run tests
run_test() {
    local test_file=$1
    local test_name=$2
    
    echo -e "\n${BLUE}Running: $test_name${NC}"
    echo "----------------------------------------"
    
    if [ -f "$test_file" ]; then
        if $PYTHON_CMD "$test_file" 2>&1; then
            echo -e "${GREEN}[✓] $test_name passed${NC}"
            return 0
        else
            echo -e "${RED}[✗] $test_name failed${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}[!] Test file not found: $test_file${NC}"
        return 1
    fi
}

# Main test execution
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Determine which tests to run based on MATLAB availability
if [ "$MATLAB_AVAILABLE" = true ]; then
    echo -e "\n${GREEN}Running REAL MATLAB Engine Tests${NC}"
    
    # Run real MATLAB tests
    if [ "$ENVIRONMENT" = "Windows Python from WSL" ]; then
        # Use Windows-compatible test file
        if run_test "test_real_matlab_windows.py" "Real MATLAB Engine Tests"; then
            ((PASSED_TESTS++))
        else
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
    else
        # Use standard test file for native WSL MATLAB
        if run_test "test_real_matlab.py" "Real MATLAB Engine Tests"; then
            ((PASSED_TESTS++))
        else
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
    fi
    
    # Also run mathematical validation if available
    if [ -f "tests/test_mathematical_validation.py" ]; then
        echo -e "\n${BLUE}Attempting Mathematical Validation Tests${NC}"
        if $PYTHON_CMD -m pytest tests/test_mathematical_validation.py -v 2>/dev/null; then
            ((PASSED_TESTS++))
        else
            echo -e "${YELLOW}[!] Mathematical validation requires additional setup${NC}"
            ((FAILED_TESTS++))
        fi
        ((TOTAL_TESTS++))
    fi
else
    echo -e "\n${YELLOW}Running MOCK Tests Only (No MATLAB Engine)${NC}"
    
    # Run mock tests with pytest
    echo -e "\n${BLUE}Running Mock Test Suites${NC}"
    
    # Core mock tests
    if $PYTHON_CMD -m pytest tests/test_matlab_engine_mocks.py -v --tb=short; then
        ((PASSED_TESTS++))
    else
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
    
    # Regression tests
    if $PYTHON_CMD -m pytest tests/test_regression_suite.py -v --tb=short; then
        ((PASSED_TESTS++))
    else
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
    
    # Performance tests
    if $PYTHON_CMD -m pytest tests/test_performance_suite.py -v --tb=short -k "not load and not memory"; then
        ((PASSED_TESTS++))
    else
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
fi

# Generate pipeline summary artifacts
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Generating Test Artifacts${NC}"
echo -e "${BLUE}========================================${NC}"

# Create artifacts directory
mkdir -p test_artifacts

# Generate pipeline summary JSON
cat > test_artifacts/pipeline_summary.json << EOF
{
  "timestamp": "$(date -Iseconds)",
  "environment": "$ENVIRONMENT",
  "python_command": "$PYTHON_CMD",
  "matlab_available": $MATLAB_AVAILABLE,
  "test_execution": {
    "total_test_suites": $TOTAL_TESTS,
    "passed_suites": $PASSED_TESTS,
    "failed_suites": $FAILED_TESTS,
    "success_rate": $(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc 2>/dev/null || echo "0")
  },
  "artifacts_generated": [
    "pipeline_summary.json",
    "test_results.json",
    "test_results.xml",
    "performance_metrics.json",
    "test_summary.md"
  ]
}
EOF

# Combine all individual test artifacts into a consolidated report
if [ -d "test_artifacts" ] && [ "$(ls -A test_artifacts/test_results.json 2>/dev/null)" ]; then
    echo -e "${GREEN}[✓]${NC} Individual test artifacts found"
    
    # Generate consolidated test report
    $PYTHON_CMD -c "
import json
import glob
from datetime import datetime

# Load pipeline summary
with open('test_artifacts/pipeline_summary.json', 'r') as f:
    pipeline = json.load(f)

# Try to load detailed test results if available
try:
    with open('test_artifacts/test_results.json', 'r') as f:
        detailed_results = json.load(f)
except:
    detailed_results = None

# Create consolidated report
consolidated = {
    'report_type': 'MATLAB Engine API Pipeline Test Results',
    'generation_timestamp': datetime.now().isoformat(),
    'pipeline_summary': pipeline,
    'detailed_results': detailed_results,
    'recommendations': []
}

# Add recommendations based on results
if pipeline['test_execution']['success_rate'] < 100:
    consolidated['recommendations'].append('Review failed tests for accuracy issues')
if not pipeline['matlab_available']:
    consolidated['recommendations'].append('Install MATLAB Engine API for comprehensive testing')

# Save consolidated report
with open('test_artifacts/consolidated_report.json', 'w') as f:
    json.dump(consolidated, f, indent=2)

print('Consolidated report generated: test_artifacts/consolidated_report.json')
" 2>/dev/null
fi

# Generate summary report
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Test Results Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Environment: ${YELLOW}$ENVIRONMENT${NC}"
echo -e "Total Test Suites: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

echo -e "\n${BLUE}Artifacts Generated:${NC}"
echo -e "  📄 test_artifacts/pipeline_summary.json"
if [ -f "test_artifacts/test_results.json" ]; then
    echo -e "  📄 test_artifacts/test_results.json"
fi
if [ -f "test_artifacts/test_results.xml" ]; then
    echo -e "  📄 test_artifacts/test_results.xml"
fi
if [ -f "test_artifacts/performance_metrics.json" ]; then
    echo -e "  📄 test_artifacts/performance_metrics.json"
fi
if [ -f "test_artifacts/test_summary.md" ]; then
    echo -e "  📄 test_artifacts/test_summary.md"
fi
if [ -f "test_artifacts/consolidated_report.json" ]; then
    echo -e "  📄 test_artifacts/consolidated_report.json"
fi

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}✓ All tests passed successfully!${NC}"
    echo -e "${GREEN}✓ Test artifacts saved to test_artifacts/${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Some tests failed. Please review the output above.${NC}"
    echo -e "${YELLOW}📊 Detailed results available in test_artifacts/${NC}"
    exit 1
fi