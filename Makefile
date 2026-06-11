.PHONY: help sync lock lint fmt format test test-integration typecheck check clean

help:
	@echo "Common targets:"
	@echo "  make sync             Create/update .venv from uv.lock"
	@echo "  make lock             Re-resolve and write uv.lock"
	@echo "  make lint             Run ruff lints"
	@echo "  make fmt              Format code with ruff"
	@echo "  make test             Run pytest (unit tests; integration skipped)"
	@echo "  make test-integration Run integration tests (real API calls)"
	@echo "  make typecheck        Run mypy type checks"
	@echo "  make check            Run lint + unit tests"
	@echo "  make clean            Remove caches"

sync:
	uv sync

lock:
	uv lock

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

format: fmt

test:
	uv run pytest

test-integration:
	./tests/run_integration_tests.sh

typecheck:
	uv run mypy lib

check: lint test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache __pycache__
