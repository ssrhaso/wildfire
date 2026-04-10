# Wildfire Classification Makefile
# Quick start: make setup-linux (or setup-windows), then make experiments-vit

PYTHON ?= python
PIP ?= pip
VENV_DIR ?= venv

# Full setup (clone -> run in one go)

setup-linux: venv install download unzip-linux preprocess
	@echo ""
	@echo "  Setup complete. Activate the venv and run experiments:"
	@echo "    source $(VENV_DIR)/bin/activate"
	@echo "    make experiments-vit"
	@echo ""

setup-windows: venv install download unzip-windows preprocess
	@echo ""
	@echo "  Setup complete. Activate the venv and run experiments:"
	@echo "    $(VENV_DIR)\Scripts\activate"
	@echo "    make experiments-vit"
	@echo ""

# Environment

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "  Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
	else \
		echo "  Virtual environment already exists, skipping."; \
	fi

install: venv
	@echo "  Installing dependencies..."
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

install-windows: venv
	@echo "  Installing dependencies..."
	$(VENV_DIR)/Scripts/pip install --upgrade pip
	$(VENV_DIR)/Scripts/pip install -r requirements.txt

# Data download & extraction

download:
	mkdir -p data/raw
	mkdir -p data/processed
	kaggle datasets download warcoder/flamevision-dataset-for-wildfire-classification
	kaggle datasets download dani215/fire-dataset
	kaggle datasets download amerzishminha/forest-fire-smoke-and-non-fire-image-dataset
	cp flamevision-dataset-for-wildfire-classification.zip data/raw/
	cp fire-dataset.zip data/raw/
	cp forest-fire-smoke-and-non-fire-image-dataset.zip data/raw/

unzip-linux:
	mkdir -p data/raw/flamevision
	unzip data/raw/flamevision-dataset-for-wildfire-classification.zip -d data/raw/flamevision
	rm data/raw/flamevision-dataset-for-wildfire-classification.zip
	mkdir -p data/raw/dani215
	unzip data/raw/fire-dataset.zip -d data/raw/dani215
	rm data/raw/fire-dataset.zip
	mkdir -p data/raw/minha
	unzip data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip -d data/raw/minha
	rm data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip

unzip-windows:
	mkdir -p data/raw/flamevision
	powershell -Command "New-Item -ItemType Directory -Force -Path 'data/raw/flamevision'; tar -xf 'data/raw/flamevision-dataset-for-wildfire-classification.zip' -C 'data/raw/flamevision'"
	powershell -Command "Remove-Item -Force -Path 'data/raw/flamevision-dataset-for-wildfire-classification.zip'"
	mkdir -p data/raw/dani215
	powershell -Command "New-Item -ItemType Directory -Force -Path 'data/raw/dani215'; tar -xf 'data/raw/fire-dataset.zip' -C 'data/raw/dani215'"
	powershell -Command "Remove-Item -Force -Path 'data/raw/fire-dataset.zip'"
	mkdir -p data/raw/minha
	powershell -Command "New-Item -ItemType Directory -Force -Path 'data/raw/minha'; tar -xf 'data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip' -C 'data/raw/minha'"
	powershell -Command "Remove-Item -Force -Path 'data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip'"

# Preprocessing

preprocess:
	$(PYTHON) src/preprocess.py

# Experiments

experiments-vit:
	bash scripts/run_vit.sh

experiments-resnet:
	bash scripts/run_resnet.sh

experiments-hybrid:
	bash scripts/run_hybrid.sh

experiments-all:
	bash scripts/run_all.sh

experiments-vit-win:
	powershell -ExecutionPolicy Bypass -File scripts/run_vit.ps1

experiments-resnet-win:
	powershell -ExecutionPolicy Bypass -File scripts/run_resnet.ps1

experiments-hybrid-win:
	powershell -ExecutionPolicy Bypass -File scripts/run_hybrid.ps1

experiments-all-win:
	powershell -ExecutionPolicy Bypass -File scripts/run_all.ps1

# Analysis

analyse-vit:
	$(PYTHON) src/analyse_results.py --model vit

analyse-resnet:
	$(PYTHON) src/analyse_results.py --model resnet

analyse-hybrid:
	$(PYTHON) src/analyse_results.py --model hybrid

analyse-all: analyse-vit analyse-resnet analyse-hybrid

# Quick test (single epoch, verifies setup)

test-vit:
	$(PYTHON) src/run_experiment.py --model vit --freeze-config freeze_none --seed 0 --epochs 1 --no-wandb

test-resnet:
	$(PYTHON) src/run_experiment.py --model resnet --freeze-config freeze_none --seed 0 --epochs 1 --no-wandb

test-hybrid:
	$(PYTHON) src/run_experiment.py --model hybrid --freeze-config freeze_none --seed 0 --epochs 1 --no-wandb

# Cleanup

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true

clean-results:
	@echo "  This will delete all experiment results. Press Ctrl+C to cancel."
	@sleep 3
	rm -rf results/vit results/resnet results/hybrid results/checkpoints results/analysis

.PHONY: setup-linux setup-windows venv install install-windows \
	download unzip-linux unzip-windows preprocess \
	experiments-vit experiments-resnet experiments-hybrid experiments-all \
	experiments-vit-win experiments-resnet-win experiments-hybrid-win experiments-all-win \
	analyse-vit analyse-resnet analyse-hybrid analyse-all \
	test-vit test-resnet test-hybrid \
	clean clean-results
