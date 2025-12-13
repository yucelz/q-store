# Makefile for Q-Store Examples

.PHONY: help install install-minimal install-dev install-full verify clean test format lint

# Default target
help:
	@echo "Q-Store Examples - Available Commands"
	@echo "======================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install examples with core dependencies"
	@echo "  make install-minimal  Install minimal dependencies (no ML)"
	@echo "  make install-full     Install all dependencies including ML"
	@echo "  make install-dev      Install with development dependencies"
	@echo ""
	@echo "Verification:"
	@echo "  make verify           Verify installation"
	@echo "  make test             Run tests"
	@echo ""
	@echo "Examples:"
	@echo "  make run-basic        Run basic example"
	@echo "  make run-financial    Run financial example"
	@echo "  make run-quickstart   Run quantum DB quickstart"
	@echo "  make run-react        Run React training workflow"
	@echo ""
	@echo "Development:"
	@echo "  make format           Format code with black and isort"
	@echo "  make lint             Run linting checks"
	@echo "  make clean            Clean generated files"
	@echo ""
	@echo "Environment:"
	@echo "  make env-setup        Create .env from template"
	@echo "  make conda-create     Create conda environment"
	@echo ""

# Installation targets
install:
	@echo "Installing Q-Store examples..."
	cd .. && pip install -e .
	pip install -r requirements.txt
	@echo "✅ Installation complete!"

install-minimal:
	@echo "Installing minimal dependencies..."
	cd .. && pip install -e .
	pip install -r requirements-minimal.txt
	@echo "✅ Minimal installation complete!"

install-full:
	@echo "Installing all dependencies..."
	cd .. && pip install -e .
	pip install -r requirements.txt
	pip install -e ".[all]"
	@echo "✅ Full installation complete!"

install-dev:
	@echo "Installing development dependencies..."
	cd .. && pip install -e .
	pip install -r requirements.txt
	pip install -e ".[dev]"
	@echo "✅ Development installation complete!"

# Verification
verify:
	@echo "Verifying installation..."
	python scripts/verify_installation.py

verify-react:
	@echo "Verifying React integration..."
	python scripts/verify_react_integration.py

# Testing
test:
	@echo "Running tests..."
	pytest -v

test-cov:
	@echo "Running tests with coverage..."
	pytest --cov=. --cov-report=html --cov-report=term

# Examples
run-basic:
	@echo "Running basic example..."
	python -m q_store_examples.basic_example

run-financial:
	@echo "Running financial example..."
	python -m q_store_examples.financial_example

run-quickstart:
	@echo "Running quantum DB quickstart..."
	python -m q_store_examples.quantum_db_quickstart

run-ml:
	@echo "Running ML training example..."
	python -m q_store_examples.ml_training_example

run-react:
	@echo "Running React training workflow..."
	./scripts/run_react_training.sh

generate-dataset:
	@echo "Generating React dataset..."
	python -m q_store_examples.react_dataset_generator

# Development
format:
	@echo "Formatting code..."
	black src/ tests/ scripts/
	isort src/ tests/ scripts/
	@echo "✅ Code formatted!"

lint:
	@echo "Running linters..."
	flake8 src/ tests/ scripts/
	mypy src/ --ignore-missing-imports

# Cleaning
clean:
	@echo "Cleaning generated files..."
	rm -rf __pycache__/
	rm -rf *.pyc
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info/
	rm -rf tinyllama-react-quantum/
	rm -f react_train.jsonl
	@echo "✅ Cleaned!"

clean-all: clean
	@echo "Cleaning all including models..."
	rm -rf models/
	rm -rf checkpoints/
	rm -rf .cache/
	@echo "✅ Deep clean complete!"

# Environment setup
env-setup:
	@echo "Setting up .env file..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env from template"; \
		echo "⚠️  Please edit .env and add your API keys"; \
	else \
		echo "⚠️  .env already exists"; \
	fi

conda-create:
	@echo "Creating conda environment..."
	conda env create -f environment.yml
	@echo "✅ Conda environment created!"
	@echo "Activate with: conda activate q-store-examples"

conda-update:
	@echo "Updating conda environment..."
	conda env update -f environment.yml --prune
	@echo "✅ Conda environment updated!"

# Documentation
docs:
	@echo "Available documentation:"
	@echo "  README.md - Main documentation"
	@echo "  REACT_QUICK_REFERENCE.md - React training quick start"
	@echo "  REACT_TRAINING_WORKFLOW.md - Detailed React workflow"
	@echo "  TINYLLAMA_TRAINING_README.md - TinyLlama guide"

# Git helpers
git-status:
	@git status

git-diff:
	@git diff

# All-in-one setup
setup: env-setup install verify
	@echo ""
	@echo "🎉 Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env and add your API keys"
	@echo "  2. Run: make verify"
	@echo "  3. Try: make run-basic"
	@echo ""
