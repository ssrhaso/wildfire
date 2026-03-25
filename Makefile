windows: download unzip-windows
linux: download unzip-linux

donwload:
	kaggle datasets download warcoder/flamevision-dataset-for-wildfire-classification
	kaggle datasets download dani215/fire-dataset
	kaggle datasets download amerzishminha/forest-fire-smoke-and-non-fire-image-dataset
	cp flamevision-dataset-for-wildfire-classification.zip src/data/raw/
	cp fire-dataset.zip data/raw/
	cp forest-fire-smoke-and-non-fire-image-dataset.zip data/raw/

unzip-linux:
	unzip data/raw/flamevision-dataset-for-wildfire-classification.zip -d data/raw/flamevision
	rm data/raw/flamevision-dataset-for-wildfire-classification.zip
	unzip data/raw/fire-dataset.zip -d data/raw/dani215
	rm data/raw/fire-dataset.zip
	unzip data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip -d data/raw/minha
	rm data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip

unzip-windows:
	powershell -Command "New-Item -ItemType Directory -Force -Path 'data/raw/flamevision'; tar -xf 'data/raw/flamevision-dataset-for-wildfire-classification.zip' -C 'data/raw/flamevision'"
	powershell -Command "Remove-Item -Force -Path 'data/raw/flamevision-dataset-for-wildfire-classification.zip'"
	powershell -Command "New-Item -ItemType Directory -Force -Path 'data/raw/dani215'; tar -xf 'data/raw/fire-dataset.zip' -C 'data/raw/dani215'"
	powershell -Command "Remove-Item -Force -Path 'data/raw/fire-dataset.zip'"
	powershell -Command "New-Item -ItemType Directory -Force -Path 'data/raw/minha'; tar -xf 'data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip' -C 'data/raw/minha'"
	powershell -Command "Remove-Item -Force -Path 'data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip'"

.PHONY: windows linux download unzip-linux unzip-windows