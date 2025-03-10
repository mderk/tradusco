# AI Translator Tests

This directory contains tests for the AI Translator application.

## Test Structure

-   **conftest.py**: Contains pytest fixtures used across multiple test files
-   **test_translation_project.py**: Tests for the `TranslationProject` class
-   **test_translation_tool.py**: Tests for the `TranslationTool` class
-   **test_utils.py**: Tests for utility functions

## Running Tests

To run the tests, execute the following command from the project root:

```bash
pytest
```

To run specific test files:

```bash
pytest tests/test_translation_project.py
pytest tests/test_translation_tool.py
pytest tests/test_utils.py
```

To run with verbose output:

```bash
pytest -v
```

## Test Coverage

To generate test coverage reports, run:

```bash
pytest --cov=lib tests/
```

## Writing New Tests

When writing new tests:

1. Place them in the appropriate test file based on the module they test
2. Use the fixtures defined in `conftest.py` where applicable
3. Follow the naming convention: `test_<function_or_method_name>_<scenario>`
4. Use meaningful assertions to verify expected behavior

## Mock Data

The tests use mock data to simulate the translation process without making actual API calls.
This approach ensures tests are deterministic and do not depend on external services.
