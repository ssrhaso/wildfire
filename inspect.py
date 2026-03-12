import os
from pathlib import Path

RAW = Path("data/raw")

for dataset in ['flamevision', 'dani215', 'minha']:
    print(f"\n=== {dataset} ===")
    root = RAW / dataset
    for dirpath, dirs, files in os.walk(root):
        imgs = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            rel = os.path.relpath(dirpath, root)
            print(f"  {rel}: {len(imgs)} images")