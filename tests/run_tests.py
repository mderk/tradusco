#!/usr/bin/env python3
import sys
import pytest
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_tests():
    """Run all tests using pytest"""
    print("\n=== Running all tests with pytest ===")

    # Run pytest on all test_*.py files
    pytest_args = ["-v", "tests/"]

    # Run pytest and capture its return code
    pytest_result = pytest.main(pytest_args)

    # pytest.ExitCode.OK equals 0, meaning all tests passed
    return 0 if pytest_result == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
