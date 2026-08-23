.PHONY: help install install-dev test lint validate-inference docker-build

PYTHON ?= python

help:
	@echo "install              Install runtime dependencies and CLI commands"
	@echo "install-dev          Install development and notebook dependencies"
	@echo "test                 Run the complete offline test suite"
	@echo "lint                 Run Ruff checks"
	@echo "validate-inference   Run offline direct/inverse inference validation"
	@echo "docker-build         Build the CUDA runtime image"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev,notebooks]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

validate-inference:
	$(PYTHON) tests/run_inference_pipeline_validation.py

docker-build:
	docker compose build
