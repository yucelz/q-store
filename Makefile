.PHONY: help install install-dev clean test lint format type-check build docs

help:
	@echo "Q-Store Development Commands"
	@echo "============================"
	@echo "install        - Install package in production mode"
	@echo "install-dev    - Install package with development dependencies"
	@echo "clean          - Remove build artifacts and cache files"
	@echo "test           - Run tests with pytest"
	@echo "test-cov       - Run tests with coverage report"
	@echo "lint           - Run linters (flake8, pylint)"
	@echo "format         - Format code with black and isort"
	@echo "format-check   - Check code formatting without changes"
	@echo "type-check     - Run type checking with mypy"
	@echo "build          - Build distribution packages"
	@echo "docs           - Generate documentation"
	@echo "verify         - Run all checks (format, lint, type, test)"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,backends]"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/q_store --cov-report=html --cov-report=term

lint:
	flake8 src/q_store tests/
	pylint src/q_store

format:
	black src/q_store tests/
	isort src/q_store tests/

format-check:
	black --check src/q_store tests/
	isort --check-only src/q_store tests/

type-check:
	mypy src/q_store

build: clean
	python -m build

docs:
	@echo "Documentation is in docs/ directory"
	@echo "For API docs, use: pdoc src/q_store -o docs/api"

verify: format-check lint type-check test
	@echo "✓ All checks passed!"

.DEFAULT_GOAL := help
