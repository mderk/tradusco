#!/usr/bin/env python3
import unittest
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import all test modules
from tests.test_translation_project import TestTranslationProject
from tests.test_batch_response_parsing import TestBatchResponseParsing
from tests.test_prompt_handling import TestPromptHandling


def run_tests():
    """Run all tests and return the result"""
    # Create a test loader
    loader = unittest.TestLoader()

    # Create a test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTest(loader.loadTestsFromTestCase(TestTranslationProject))
    test_suite.addTest(loader.loadTestsFromTestCase(TestBatchResponseParsing))
    test_suite.addTest(loader.loadTestsFromTestCase(TestPromptHandling))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return success/failure
    return len(result.errors) == 0 and len(result.failures) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
