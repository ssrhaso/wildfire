# Wildfire Classification

## Prerequisites

* Python 3.10+
* Kaggle account
* ~15GB free disk space

---

## Step 1 — Virtual Environment & Dependencies

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

## Step 2 — Download Datasets

Zips are large (~10GB total). This will take approximately 10 minutes depending on your connection.
Run each command one at a time in your terminal:

```bash
kaggle datasets download warcoder/flamevision-dataset-for-wildfire-classification
```

```bash
kaggle datasets download dani215/fire-dataset
```

```bash
kaggle datasets download amerzishminha/forest-fire-smoke-and-non-fire-image-dataset
```

---

## Step 3 - Unzip Datasets

Move all zips into `data/raw/`, then run:

```powershell
cd data/raw

Expand-Archive -Path "flamevision-dataset-for-wildfire-classification.zip" -DestinationPath "flamevision"
Expand-Archive -Path "fire-dataset.zip" -DestinationPath "dani215"
Expand-Archive -Path "forest-fire-smoke-and-non-fire-image-dataset.zip" -DestinationPath "minha"
```

Extraction takes approximately 10 minutes.

---

## Alternative Way - Using Makefile

If you want an automated approach, you can use the provided `Makefile` to download and extract everything in one command from the project root. (Takes ~10 minutes to download and ~10 minutes to extract).

**For Linux / Mac:**
```bash
make linux
```
*(If prompted to replace files during unzip, type `A` for All).*

**For Windows:**
```powershell
make windows
```
*(If `make` isn't recognized on Windows, use `mingw32-make windows` instead).*

---

## Step 4 - Verify

```bash
python inspect.py
```

Each dataset should show image counts per class folder. If any show 0 images, check the zip extracted into the correct directory.


## Step 5 - Run Preprocessing

```bash
python preprocess.py
```

Run the preprocessing script on the 3 datasets
