# Tradusco Tests

This directory contains tests for the Tradusco application.

## Test Structure

- `test_translation_*`, `test_envelope.py` and `test_prompt_*` cover the engine;
- `test_glossary_tool.py` and `test_context_tool.py` cover preparation;
- `test_review_tool.py`, `test_delivery_tools.py` and `test_run_tool.py` cover the workflow tools;
- `test_acceptance_small_shop.py` covers the independent round trip;
- `test_init_scaffold.py` and `test_reference_host.py` cover new integrations;
- `test_e2e_*` and `test_integration_translation_methods.py` make real API calls.

## Running Tests

To run the tests, execute the following command from the project root:

```bash
uv run pytest
```

To run a specific area:

```bash
uv run pytest tests/test_run_tool.py tests/test_delivery_tools.py
uv run pytest tests/test_glossary_tool.py tests/test_context_tool.py
uv run pytest tests/test_acceptance_small_shop.py -m "not integration"
```

To run with verbose output:

```bash
uv run pytest -v
```

## Integration Tests (real API calls)

Integration tests are marked with `@pytest.mark.integration` and are excluded from normal runs.
See `tests/README_INTEGRATION.md` for details.

## Test Coverage

To generate test coverage reports, run:

```bash
uv run pytest --cov=lib tests/
```

## Writing New Tests

When writing new tests:

1. Place them in the appropriate test file based on the module they test
2. Use the fixtures defined in `conftest.py` where applicable
3. Follow the naming convention: `test_<function_or_method_name>_<scenario>`
4. Use meaningful assertions to verify expected behavior

## Mock Data

Normal tests use controlled model responses and temporary host projects. Tests
marked `integration` are the only tests allowed to call external providers.
