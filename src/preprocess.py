from pathlib import Path
from typing import Dict, List, Tuple

import imagehash
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42
RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "labels.csv"
LABEL_INT = {"fire": 1, "nofire": 0}

DATASET_CONFIG: Dict[str, Dict] = {
    "flamevision": {
        "root": RAW_DIR / "flamevision" / "Classification",
        "label_map": {"fire": "fire", "nofire": "nofire"},
    },
    "dani215": {
        "root": RAW_DIR / "dani215" / "fire_dataset",
        "label_map": {"fire": "fire", "not_fire": "nofire"},
    },
    "minha": {
        "root": RAW_DIR / "minha" / "FOREST_FIRE_SMOKE_AND_NON_FIRE_DATASET",
        "label_map": {"fire": "fire"},
    },
}



def collect_images() -> pd.DataFrame:
    """Walk all source directories and build a dataframe of image paths with labels."""
    records: List[Dict] = []
    for source, cfg in DATASET_CONFIG.items():
        root: Path = cfg["root"]
        label_map: Dict[str, str] = cfg["label_map"]
        count = 0
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            folder = path.parent.name.lower().replace(" ", "_")
            if folder not in label_map:
                continue
            records.append({
                "src_path": str(path),
                "label": label_map[folder],
                "source": source,
            })
            count += 1
        print(f"  {source}: {count} images")
    return pd.DataFrame(records)



def verify_and_hash(df: pd.DataFrame) -> pd.DataFrame:
    """Open every image to verify it, compute its pHash, drop corrupt/duplicate files."""
    corrupt_counts: Dict[str, int] = {}
    valid_indices: List[int] = []
    hashes: List[str] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying & hashing"):
        try:
            with Image.open(row["src_path"]) as img:
                img.verify()
            hashes.append(str(imagehash.phash(Image.open(row["src_path"]))))
            valid_indices.append(idx)
        except Exception:
            corrupt_counts[row["source"]] = corrupt_counts.get(row["source"], 0) + 1

    print("\nIntegrity Check")
    if corrupt_counts:
        for source, count in corrupt_counts.items():
            print(f"  {source}: {count} corrupt file(s) removed")
    else:
        print("  No corrupt files found.")

    df = df.loc[valid_indices].copy().reset_index(drop=True)
    df["phash"] = hashes
    before = len(df)
    df = df.drop_duplicates(subset="phash", keep="first").reset_index(drop=True)
    df = df.drop(columns=["phash"])
    print(f"\nDeduplication")
    print(f"  Duplicates removed: {before - len(df)}")

    return df



def stratified_split(df: pd.DataFrame) -> pd.DataFrame:
    """80/10/10 stratified split using two-stage train_test_split."""
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_STATE, stratify=temp_df["label"],
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)



def copy_rename_resize(df: pd.DataFrame) -> pd.DataFrame:
    """Copy images into data/processed/{split}/{label}/, rename sequentially, resize to 224x224."""
    dest_paths: List[str] = [""] * len(df)

    for split in ["train", "val", "test"]:
        split_df = df[df["split"] == split]
        counters: Dict[str, int] = {"fire": 0, "nofire": 0}

        for idx, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Copying {split}"):
            label: str = row["label"]
            counters[label] += 1
            out_dir = OUTPUT_DIR / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{label}_{counters[label]:05d}.jpg"
            dest = out_dir / filename

            try:
                with Image.open(row["src_path"]) as img:
                    img = img.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS)
                    img.save(dest, "JPEG")
            except Exception as e:
                print(f"  Failed to process {row['src_path']}: {e}")
                continue

            dest_paths[idx] = str(dest)

    df = df.copy()
    df["path"] = dest_paths
    df = df[df["path"] != ""].reset_index(drop=True)
    return df



def print_summary(df: pd.DataFrame) -> None:
    """Print a formatted table of image counts and class ratios per split."""
    print(f"\n{'Split':<8} {'Total':>7} {'Fire':>7} {'Nofire':>8} {'Fire %':>8}")
    print("-" * 42)
    for split in ["train", "val", "test"]:
        s = df[df["split"] == split]
        total = len(s)
        fire = (s["label"] == "fire").sum()
        nofire = (s["label"] == "nofire").sum()
        pct = fire / total * 100 if total else 0
        print(f"  {split:<6} {total:>7} {fire:>7} {nofire:>8} {pct:>7.1f}%")
    print()



def main() -> None:
    """Run the full preprocessing pipeline."""
    print("Step 1 — Collecting images...")
    df = collect_images()
    print(f"  Total: {len(df)}\n")

    print("Step 2/3 — Verifying integrity & deduplicating...")
    df = verify_and_hash(df)

    print("\nStep 4 — Stratified split (80/10/10)...")
    df = stratified_split(df)
    print("  Done.\n")

    print("Step 5 — Copying, renaming & resizing to 224×224...")
    df = copy_rename_resize(df)

    csv_df = df[["path", "label", "split"]].copy()
    csv_df["label"] = csv_df["label"].map(LABEL_INT)
    split_order = {"train": 0, "val": 1, "test": 2}
    label_order = {1: 0, 0: 1}
    csv_df["_s"] = csv_df["split"].map(split_order)
    csv_df["_l"] = csv_df["label"].map(label_order)
    csv_df = csv_df.sort_values(["_s", "_l"]).drop(columns=["_s", "_l"]).reset_index(drop=True)
    csv_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV}")

    print("\nStep 6 — Summary")
    print_summary(df)


if __name__ == "__main__":
    main()
