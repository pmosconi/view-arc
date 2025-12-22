.PHONY: help install test test-cov test-tracking test-visual mypy lint format profile clean

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install project with dev dependencies
	uv pip install -e ".[dev]"

test:  ## Run all tests
	uv run pytest

test-cov:  ## Run tests with coverage report
	uv run pytest --cov=view_arc --cov-report=html --cov-report=term

test-tracking:  ## Run only tracking tests
	uv run pytest tests/test_tracking_*.py

test-visual:  ## Run visual tests (generates output images)
	uv run pytest tests/visual/

mypy:  ## Run type checking with mypy
	uv run mypy .

lint:  ## Check code with ruff linter
	uv run ruff check .

lint-fix:  ## Fix auto-fixable linting issues
	uv run ruff check --fix .

format:  ## Format code with ruff
	uv run ruff format .

profile:  ## Run performance profiling baseline
	uv run python profile_workload.py --scenario tracking_baseline

profile-all:  ## Run all performance profiling scenarios
	uv run python profile_workload.py --scenario all

profile-save:  ## Run profiling and save to CSV
	uv run python profile_workload.py --scenario tracking_baseline --save-csv

check:  ## Run full validation pipeline (type check + test + profile)
	@echo "==> Running type check..."
	@uv run mypy .
	@echo "==> Running tests..."
	@uv run pytest
	@echo "==> Running performance baseline..."
	@uv run python profile_workload.py --scenario tracking_baseline
	@echo "==> All checks passed!"

example-basic:  ## Run basic attention tracking example
	uv run python examples/attention_tracking_basic.py

example-simulation:  ## Run simulated store session example
	uv run python examples/simulated_store_session.py

clean:  ## Clean up generated files
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf examples/output/*.png
	rm -rf tests/visual/output/*.png
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
