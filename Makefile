# Makefile for DeepSeek AZE project

# Variables
PYTHON = python3
PIP = pip
PYTEST = pytest
BLACK = black
ISORT = isort
FLAKE8 = flake8
MYPY = mypy

# Directories
SRC_DIR = src
TESTS_DIR = tests
CONFIGS_DIR = configs
DATA_DIR = data

# Default target
.PHONY: help
help:
	@echo "DeepSeek AZE - Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  install          Install package and dependencies"
	@echo "  install-dev      Install package with development dependencies"
	@echo "  install-all      Install package with all optional dependencies"
	@echo ""
	@echo "Development:"
	@echo "  format           Format code with black and isort"
	@echo "  lint             Run linting checks"
	@echo "  test             Run tests"
	@echo "  test-cov         Run tests with coverage"
	@echo "  type-check       Run type checking with mypy"
	@echo "  check            Run all checks (format, lint, type-check, test)"
	@echo ""
	@echo "Training:"
	@echo "  train-quick      Quick training with default settings"
	@echo "  train-qlora      Training with QLoRA configuration"
	@echo "  train-lora       Training with LoRA configuration"
	@echo ""
	@echo "Evaluation:"
	@echo "  evaluate         Evaluate trained model"
	@echo "  benchmark        Run comprehensive benchmark"
	@echo ""
	@echo "Data:"
	@echo "  setup-data       Setup data directories"
	@echo "  download-sample  Download sample data"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            Clean build artifacts"
	@echo "  clean-all        Clean all generated files"

# Installation targets
.PHONY: install
install:
	$(PIP) install -e .

.PHONY: install-dev
install-dev:
	$(PIP) install -e ".[dev]"

.PHONY: install-all
install-all:
	$(PIP) install -e ".[dev,docs,evaluation,api]"

# Development targets
.PHONY: format
format:
	$(BLACK) $(SRC_DIR) $(TESTS_DIR) *.py
	$(ISORT) $(SRC_DIR) $(TESTS_DIR) *.py

.PHONY: lint
lint:
	$(FLAKE8) $(SRC_DIR) $(TESTS_DIR) *.py
	$(BLACK) --check $(SRC_DIR) $(TESTS_DIR) *.py
	$(ISORT) --check-only $(SRC_DIR) $(TESTS_DIR) *.py

.PHONY: type-check
type-check:
	$(MYPY) $(SRC_DIR)

.PHONY: test
test:
	$(PYTEST) $(TESTS_DIR) -v

.PHONY: test-cov
test-cov:
	$(PYTEST) $(TESTS_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

.PHONY: check
check: lint type-check test

# Training targets
.PHONY: train-quick
train-quick:
	$(PYTHON) train.py \
		--model deepseek-llm-7b-base \
		--dataset data/processed/sample_instructions.json \
		--method qlora \
		--epochs 1 \
		--batch-size 2 \
		--output-dir ./checkpoints/quick_test

.PHONY: train-qlora
train-qlora:
	$(PYTHON) train.py \
		--model deepseek-ai/deepseek-llm-7b-base \
		--dataset data/processed/azerbaijani_instructions.json \
		--method qlora \
		--epochs 3 \
		--batch-size 4 \
		--learning-rate 2e-4 \
		--output-dir ./checkpoints/qlora_model

.PHONY: train-lora
train-lora:
	$(PYTHON) train.py \
		--model deepseek-ai/deepseek-llm-7b-base \
		--dataset data/processed/azerbaijani_instructions.json \
		--method lora \
		--epochs 3 \
		--batch-size 2 \
		--learning-rate 1e-4 \
		--output-dir ./checkpoints/lora_model

.PHONY: train-config
train-config:
	$(PYTHON) train.py --config $(CONFIGS_DIR)/training_configs/qlora_config.yaml

# Evaluation targets
.PHONY: evaluate
evaluate:
	$(PYTHON) evaluate.py \
		--model ./checkpoints/best_model \
		--benchmark comprehensive \
		--output-dir ./evaluation_results

.PHONY: benchmark
benchmark:
	$(PYTHON) evaluate.py \
		--model ./checkpoints/best_model \
		--benchmark comprehensive \
		--tasks text_generation question_answering perplexity \
		--output-dir ./benchmark_results

.PHONY: compare-models
compare-models:
	$(PYTHON) evaluate.py \
		--compare ./checkpoints/model1 ./checkpoints/model2 \
		--benchmark comprehensive \
		--output-dir ./comparison_results

# Data targets
.PHONY: setup-data
setup-data:
	mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(DATA_DIR)/synthetic
	mkdir -p checkpoints outputs logs

.PHONY: download-sample
download-sample:
	@echo "Creating sample data..."
	@mkdir -p $(DATA_DIR)/processed
	@$(PYTHON) -c "import json; \
	sample_data = [ \
		{'instruction': 'Azərbaycan haqqında maraqlı fakt de', 'input': '', 'output': 'Azərbaycan dünyada ən çox vulkan palçığına malik ölkədir.'}, \
		{'instruction': 'Bakının paytaxt olduğu ili de', 'input': '', 'output': 'Bakı 1920-ci ildən Azərbaycanın paytaxtıdır.'}, \
		{'instruction': 'Azərbaycan dilinə tərcümə et', 'input': 'Good morning', 'output': 'Sabahınız xeyir'} \
	]; \
	json.dump(sample_data, open('$(DATA_DIR)/processed/sample_instructions.json', 'w'), indent=2, ensure_ascii=False)"
	@echo "Sample data created at $(DATA_DIR)/processed/sample_instructions.json"

# Docker targets
.PHONY: docker-build
docker-build:
	docker build -t deepseek-aze:latest .

.PHONY: docker-run
docker-run:
	docker run -it --rm \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		-v $(PWD)/outputs:/app/outputs \
		deepseek-aze:latest

.PHONY: docker-train
docker-train:
	docker run -it --rm \
		--gpus all \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/checkpoints:/app/checkpoints \
		deepseek-aze:latest \
		python train.py --model deepseek-llm-7b-base --dataset data/processed/sample_instructions.json

# Documentation targets
.PHONY: docs
docs:
	@echo "Building documentation..."
	@echo "Documentation will be available soon"

# Cleanup targets
.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

.PHONY: clean-all
clean-all: clean
	rm -rf logs/
	rm -rf outputs/
	rm -rf model_cache/
	rm -rf .wandb/

# Development utilities
.PHONY: jupyter
jupyter:
	jupyter notebook notebooks/

.PHONY: tensorboard
tensorboard:
	tensorboard --logdir=logs

.PHONY: wandb-login
wandb-login:
	wandb login

# Pre-commit hooks
.PHONY: pre-commit-install
pre-commit-install:
	pre-commit install

.PHONY: pre-commit-run
pre-commit-run:
	pre-commit run --all-files

# Version management
.PHONY: version
version:
	@$(PYTHON) -c "import src; print('Current version: 0.1.0')"

# Environment setup
.PHONY: env-create
env-create:
	conda create -n deepseek_aze python=3.9 -y
	@echo "Created conda environment 'deepseek_aze'"
	@echo "Activate with: conda activate deepseek_aze"

.PHONY: env-export
env-export:
	conda env export > environment.yml

# Quick setup for new users
.PHONY: quick-start
quick-start: setup-data download-sample install-dev
	@echo ""
	@echo "✅ Quick start setup completed!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run a quick test: make train-quick"
	@echo "2. Evaluate the model: make evaluate"
	@echo "3. Check the results in ./checkpoints/ and ./evaluation_results/"
	@echo ""

# Production setup
.PHONY: prod-setup
prod-setup: install-all setup-data
	@echo "Production environment setup completed"