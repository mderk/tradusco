#!/bin/bash
# Script to run integration tests

# Make script executable with: chmod +x tests/run_integration_tests.sh

# Set up colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Running integration tests that make real API calls${NC}"
echo -e "${YELLOW}These tests will use GEMINI_API_KEY and OPENROUTER_API_KEY (loaded from .env if present)${NC}"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed. Please install it with 'pip install pytest pytest-asyncio'.${NC}"
    exit 1
fi

# Determine the base directory of the project
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to the project directory
cd "$PROJECT_DIR" || { echo -e "${RED}Error: Could not change to project directory.${NC}"; exit 1; }

# Run the integration tests
echo -e "${GREEN}Starting integration tests...${NC}"
# Override the default configuration to run integration tests specifically
python - <<'PY'
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # If python-dotenv isn't installed, environment variables must be set manually.
    pass

import pytest

sys.exit(pytest.main(["tests/test_integration_translation_methods.py", "-vv", "-k", "integration"]))
PY

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All integration tests passed!${NC}"
else
    echo -e "${RED}Some integration tests failed. Check the output above for details.${NC}"
    exit 1
fi