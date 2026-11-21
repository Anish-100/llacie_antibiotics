.PHONY: install test-install test test-fast test-integration test-unit test-cov lint format clean help

help:
	@echo "LLaCIE Development Commands (uses uv: https://docs.astral.sh/uv/):"
	@echo "  make install           Create .venv and install package + dev/test dependencies"
	@echo "  make test              Run all tests"
	@echo "  make test-fast         Run only fast tests (skip slow LLM-dependent tests)"
	@echo "  make test-integration  Run integration tests only"
	@echo "  make test-unit         Run unit tests only"
	@echo "  make test-cov          Run tests with coverage report"
	@echo "  make lint              Lint with ruff"
	@echo "  make format            Format with ruff"
	@echo "  make clean             Remove test artifacts and cache"

install:
	uv sync

# Kept for backwards compatibility
test-install: install

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

test-integration:
	uv run pytest tests/integration/

test-unit:
	uv run pytest tests/unit/

test-cov:
	# Note that this doesn't work with the integration tests since llacie is run in a subprocess.
	uv run pytest --cov=llacie --cov-report=term-missing --cov-report=html

lint:
	uv run ruff check llacie/

format:
	uv run ruff format llacie/

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
