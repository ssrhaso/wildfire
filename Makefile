# Wildfire Classification Makefile

# Data Download & Extraction

windows: download unzip-windows
linux: download unzip-linux

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
	python src/preprocess.py

# Experiments (freezing ablation)

experiments-vit:
	bash scripts/run_vit.sh

experiments-resnet:
	bash scripts/run_resnet.sh

experiments-hybrid:
	bash scripts/run_hybrid.sh

experiments-all:
	bash scripts/run_all.sh

# Analysis only (after experiments are done)

analyse-vit:
	python src/analyse_results.py --model vit

analyse-resnet:
	python src/analyse_results.py --model resnet

analyse-hybrid:
	python src/analyse_results.py --model hybrid

analyse-all: analyse-vit analyse-resnet analyse-hybrid

# Quick single-run test (verifies setup works)

test-vit:
	python src/run_experiment.py --model vit --freeze-config freeze_none --seed 0 --epochs 1 --no-wandb

test-resnet:
	python src/run_experiment.py --model resnet --freeze-config freeze_none --seed 0 --epochs 1 --no-wandb

test-hybrid:
	python src/run_experiment.py --model hybrid --freeze-config freeze_none --seed 0 --epochs 1 --no-wandb

.PHONY: windows linux download unzip-linux unzip-windows preprocess \
	experiments-vit experiments-resnet experiments-hybrid experiments-all \
	analyse-vit analyse-resnet analyse-hybrid analyse-all \
	test-vit test-resnet test-hybrid
