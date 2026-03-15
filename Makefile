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
	unzip src/data/raw/fire-dataset.zip -d src/data/raw/dani215
	unzip src/data/raw/forest-fire-smoke-and-non-fire-image-dataset.zip -d src/data/raw/minha

unzip-windows:
	Expand-Archive -Path "flamevision-dataset-for-wildfire-classification.zip" -DestinationPath "flamevision"
	Expand-Archive -Path "fire-dataset.zip" -DestinationPath "dani215"
	Expand-Archive -Path "forest-fire-smoke-and-non-fire-image-dataset.zip" -DestinationPath "minha"

.PHONY: windows linux download unzip-linux unzip-windows