# Wildfire Classification Makefile
# All commands are designed to be run from the project root directory.
# Quick start: make setup-linux (or setup-windows), then make experiments-vit

PYTHON ?= python
PIP ?= pip
VENV_DIR ?= venv

help:
	@echo "Wildfire Classification - make targets"
	@echo ""
	@echo "  Setup:"
	@echo "    setup-linux                  venv + install + download + unzip + preprocess (Linux/macOS)"
	@echo "    setup-windows                venv + install + download + unzip + preprocess (Windows)"
	@echo "    venv                         create Python virtual environment"
	@echo "    install                      install requirements into venv (Linux/macOS)"
	@echo "    install-windows              install requirements into venv (Windows)"
	@echo ""
	@echo "  Data:"
	@echo "    download                     download Kaggle datasets into data/raw/"
	@echo "    unzip-linux                  extract dataset zips (Linux/macOS)"
	@echo "    unzip-windows                extract dataset zips (Windows)"
	@echo "    preprocess                   build processed dataset + labels.csv"
	@echo ""
	@echo "  Experiments (Linux/macOS):"
	@echo "    experiments-vit              ViT-B/16 freezing ablation"
	@echo "    experiments-resnet           ResNet-50 freezing ablation"
	@echo "    experiments-hybrid           Hybrid CNN-ViT freezing ablation"
	@echo "    experiments-hybrid-bnfrozen  Hybrid BN-frozen variants"
	@echo "    experiments-all              all three models (excludes BN frozen)"
	@echo ""
	@echo "  Experiments (Windows):"
	@echo "    experiments-vit-win          ViT-B/16 freezing ablation"
	@echo "    experiments-resnet-win       ResNet-50 freezing ablation"
	@echo "    experiments-hybrid-win       Hybrid CNN-ViT (includes BN frozen)"
	@echo "    experiments-all-win          all three models"
	@echo ""
	@echo "  Analysis:"
	@echo "    analyse-vit                  statistics + plots for ViT results"
	@echo "    analyse-resnet               statistics + plots for ResNet results"
	@echo "    analyse-hybrid               statistics + plots for Hybrid results"
	@echo "    analyse-all                  run all three analyses"
	@echo ""
	@echo "  Quick tests (1 epoch, no wandb):"
	@echo "    test-vit / test-resnet / test-hybrid"
	@echo ""
	@echo "  Cleanup:"
	@echo "    clean                        remove __pycache__ and checkpoint caches"
	@echo "    clean-results                delete all experiment results (results/)"

# Full setup (clone -> run in one go)

setup-linux: venv install download unzip-linux preprocess
	@echo ""
	@echo "  Setup complete. Activate the venv and run experiments:"
	@echo "    source $(VENV_DIR)/bin/activate"
	@echo "    make experiments-vit"
	@echo ""

setup-windows: venv install-windows download unzip-windows preprocess
	@echo ""
	@echo "  Setup complete. Activate the venv and run experiments:"
	@echo "    $(VENV_DIR)\\Scripts\\activate"
	@echo "    make experiments-vit-win"
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
	mkdir -p data/raw data/processed
	kaggle datasets download -p data/raw warcoder/flamevision-dataset-for-wildfire-classification
	kaggle datasets download -p data/raw dani215/fire-dataset
	kaggle datasets download -p data/raw amerzishminha/forest-fire-smoke-and-non-fire-image-dataset

unzip-linux:
	mkdir -p data/raw/flamevision data/raw/dani215 data/raw/minha
	unzip -o data/raw/flamevision-dataset-for-wildfire-classification.zip -d data/raw/flamevision
	unzip -o data/raw/fire-dataset.zip -d data/raw/dani215
	unzip -o data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip -d data/raw/minha
	rm -f data/raw/flamevision-dataset-for-wildfire-classification.zip
	rm -f data/raw/fire-dataset.zip
	rm -f data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip

unzip-windows:
	powershell -Command "New-Item -ItemType Directory -Force -Path 'data/raw/flamevision','data/raw/dani215','data/raw/minha' | Out-Null; tar -xf 'data/raw/flamevision-dataset-for-wildfire-classification.zip' -C 'data/raw/flamevision'; tar -xf 'data/raw/fire-dataset.zip' -C 'data/raw/dani215'; tar -xf 'data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip' -C 'data/raw/minha'; Remove-Item -Force 'data/raw/flamevision-dataset-for-wildfire-classification.zip','data/raw/fire-dataset.zip','data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip'"

# Preprocessing

preprocess:
	$(PYTHON) src/preprocess.py

# Experiments (Linux/macOS)

experiments-vit:
	bash scripts/run_vit.sh

experiments-resnet:
	bash scripts/run_resnet.sh

experiments-hybrid:
	bash scripts/run_hybrid.sh

experiments-hybrid-bnfrozen:
	bash scripts/run_hybrid_bnfrozen.sh

experiments-all:
	bash scripts/run_all.sh

# Experiments (Windows)

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

.PHONY: help setup-linux setup-windows venv install install-windows \
	download unzip-linux unzip-windows preprocess \
	experiments-vit experiments-resnet experiments-hybrid experiments-hybrid-bnfrozen experiments-all \
	experiments-vit-win experiments-resnet-win experiments-hybrid-win experiments-all-win \
	analyse-vit analyse-resnet analyse-hybrid analyse-all \
	test-vit test-resnet test-hybrid \
	clean clean-results
