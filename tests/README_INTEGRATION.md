# Integration Tests

This directory contains integration tests that make real API calls to test the translator against actual LLM services.

## Running Integration Tests

Integration tests are **excluded from normal test runs** (when using `pytest` command) to avoid unnecessary API usage and costs. There are several ways to run the integration tests:

```bash
# Use the provided script (recommended)
./tests/run_integration_tests.sh

# Alternative methods with pytest directly:
# Run all integration tests
pytest -k "integration" tests/test_integration_translation_methods.py -v

# Run a specific integration test
pytest -k "integration" tests/test_integration_translation_methods.py::TestIntegrationTranslationMethods::test_standard_method -v

# Run with extra verbosity to see all output
pytest -k "integration" tests/test_integration_translation_methods.py -vv
```

## Test Description

The integration tests verify that all three translation methods work with the Gemini API:

1. **Standard Method**: Uses basic prompt formatting and parses the JSON response
2. **Structured Method**: Uses the structured output API for more reliable JSON responses
3. **Function Method**: Uses function calling to guide the response format

The tests also include a comparison test that runs all three methods on the same inputs and displays a side-by-side comparison of the results.

## Requirements

To run these tests, you need:

1. An active Gemini API key in your `.env` file
2. An active OpenRouter API key in your `.env` file
3. Internet connectivity to make API calls

## API Key Setup

Ensure your `.env` file contains the API key:

```
GEMINI_API_KEY=your_api_key_here
OPENROUTER_API_KEY=your_api_key_here
```

## Adding New Integration Tests

When adding new integration tests:

1. Mark the test class or method with `@pytest.mark.integration`
2. Keep the number of API calls minimal to reduce costs
3. Use short input texts to avoid large token usage
4. Consider using the test fixture patterns established in the existing tests
