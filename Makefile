windows: download unzip-windows
linux: download unzip-linux

donwload:
	kaggle datasets download warcoder/flamevision-dataset-for-wildfire-classification
	kaggle datasets download dani215/fire-dataset
	kaggle datasets download amerzishminha/forest-fire-smoke-and-non-fire-image-dataset
	cp flamevision-dataset-for-wildfire-classification.zip src/data/raw/
	cp fire-dataset.zip src/data/raw/
	cp forest-fire-smoke-and-non-fire-image-dataset.zip src/data/raw/

unzip-linux:
	unzip src/data/raw/flamevision-dataset-for-wildfire-classification.zip -d src/data/raw/flamevision
	rm src/data/raw/flamevision-dataset-for-wildfire-classification.zip
	unzip src/data/raw/fire-dataset.zip -d src/data/raw/dani215
	rm src/data/raw/fire-dataset.zip
	unzip src/data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip -d src/data/raw/minha
	rm src/data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip

unzip-windows:
	powershell -Command "New-Item -ItemType Directory -Force -Path 'src/data/raw/flamevision'; tar -xf 'src/data/raw/flamevision-dataset-for-wildfire-classification.zip' -C 'src/data/raw/flamevision'"
	powershell -Command "Remove-Item -Force -Path 'src/data/raw/flamevision-dataset-for-wildfire-classification.zip'"
	powershell -Command "New-Item -ItemType Directory -Force -Path 'src/data/raw/dani215'; tar -xf 'src/data/raw/fire-dataset.zip' -C 'src/data/raw/dani215'"
	powershell -Command "Remove-Item -Force -Path 'src/data/raw/fire-dataset.zip'"
	powershell -Command "New-Item -ItemType Directory -Force -Path 'src/data/raw/minha'; tar -xf 'src/data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip' -C 'src/data/raw/minha'"
	powershell -Command "Remove-Item -Force -Path 'src/data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip'"

.PHONY: windows linux download unzip-linux unzip-windows