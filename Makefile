.PHONY: setup env format lint test clean

ENV_NAME = rag-benchmark-lab

setup:
	@echo "Creating micromamba environment..."
	micromamba env create -f environment.yml
	@echo "Installing package in editable mode..."
	micromamba run -n $(ENV_NAME) pip install -e ".[dev]"

env:
	@echo "Updating micromamba environment..."
	micromamba env update -f environment.yml --prune

format:
	micromamba run -n $(ENV_NAME) black src/ tests/
	micromamba run -n $(ENV_NAME) ruff check --fix src/ tests/

lint:
	micromamba run -n $(ENV_NAME) ruff check src/ tests/
	micromamba run -n $(ENV_NAME) mypy src/

test:
	micromamba run -n $(ENV_NAME) pytest tests/ -v

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
