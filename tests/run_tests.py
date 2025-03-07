#!/usr/bin/env python3
import unittest
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import test modules
from tests.test_translation_project import TestTranslationProject
from tests.test_batch_response_parsing import TestBatchResponseParsing
from tests.test_prompt_handling import TestPromptHandling
from tests.test_context_handling import TestContextHandling


def run_tests():
    # Create a test suite
    test_suite = unittest.TestSuite()

    # Add test cases
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestTranslationProject)
    )
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestBatchResponseParsing)
    )
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPromptHandling))
    test_suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TestContextHandling)
    )

    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
