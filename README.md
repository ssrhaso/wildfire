# Wildfire Classification

Comparative layer-freezing ablation study across three architectures for binary wildfire detection.

| Model | Pretrained | Freeze Configs | Runs per model |
|-------|-----------|----------------|----------------|
| ViT-B/16 | ImageNet-1K | 6 (patch -> blocks 0-11) | 30 (6 x 5 seeds) |
| ResNet-50 | ImageNet-1K | 6 (conv1 -> layer1-4) | 30 (6 x 5 seeds) |
| Hybrid CNN-ViT | ImageNet-1K (backbone) | 5 (backbone / transformer / both) | 25 (5 x 5 seeds) |

---

## Prerequisites

* Python 3.10+
* Kaggle account
* ~15GB free disk space
* GPU recommended (experiments run on CPU but are slow)

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

```bash
kaggle datasets download warcoder/flamevision-dataset-for-wildfire-classification
kaggle datasets download dani215/fire-dataset
kaggle datasets download amerzishminha/forest-fire-smoke-and-non-fire-image-dataset
```

---

## Step 3 — Unzip Datasets

Move all zips into `data/raw/`, then run:

**Linux/Mac:**
```bash
make linux
```

**Windows:**
```powershell
make windows
```

---

## Step 4 — Preprocess

```bash
python src/preprocess.py
```

---

## Step 5 — Run Experiments

Each team member runs their assigned model. All scripts are in `scripts/`.

### ViT-B/16 (6 freeze configs x 5 seeds = 30 runs)

```powershell
# Windows
.\scripts\run_vit.ps1

# Linux/Mac
bash scripts/run_vit.sh
```

### ResNet-50 (6 freeze configs x 5 seeds = 30 runs)

```powershell
# Windows
.\scripts\run_resnet.ps1

# Linux/Mac
bash scripts/run_resnet.sh
```

### Hybrid CNN-ViT (5 freeze configs x 5 seeds = 25 runs)

```powershell
# Windows
.\scripts\run_hybrid.ps1

# Linux/Mac
bash scripts/run_hybrid.sh
```

### Run all models sequentially

```powershell
# Windows
.\scripts\run_all.ps1

# Linux/Mac
bash scripts/run_all.sh
```

---

## Quick Test (verify setup before full run)

Run a single 1-epoch dry run to make sure everything works:

```bash
make test-vit
make test-resnet
make test-hybrid
```

---

## Analysis

Analysis runs automatically at the end of each experiment script. To re-run manually:

```bash
python src/analyse_results.py --model vit
python src/analyse_results.py --model resnet
python src/analyse_results.py --model hybrid
```

### Outputs

```
results/
  vit/                        # per-seed JSON results
  resnet/
  hybrid/
  checkpoints/                # best model weights
  analysis/
    vit/                      # plots + CSVs
      summary.csv             # mean acc, F1, std per config
      statistics.csv          # pairwise t-tests + Cohen's d
      boxplot_accuracy.png
      val_curves.png
      val_f1_fire_curves.png
      lr_schedule.png
      train_val_curves.png
      confusion_matrices.png
    resnet/
    hybrid/
```

---

## Project Structure

```
wildfire/
  configs/
    config.yaml               # hybrid model architecture config
  scripts/
    run_vit.ps1 / .sh         # ViT experiment runner
    run_resnet.ps1 / .sh       # ResNet experiment runner
    run_hybrid.ps1 / .sh       # Hybrid experiment runner
    run_all.ps1 / .sh          # all models sequentially
  src/
    dataset.py                 # WildfireDataset + DataLoaders
    preprocess.py              # raw data preprocessing
    freeze.py                  # freezing configs for all 3 models
    run_experiment.py          # single-run training script
    analyse_results.py         # statistical analysis + plots
    train.py                   # legacy two-phase training script
    evaluate.py
    gradcam.py
    models/
      vit.py                   # ViT-B/16 classifier
      resnet.py                # ResNet-50 classifier
      hybrid.py                # Hybrid CNN-ViT classifier
  Makefile
  requirements.txt
```

---

## Experiment Details

**Shared across all models:**
- Seeds: `0, 5, 10, 15, 20`
- Epochs: 20 (with early stopping, patience=5)
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-2)
- Scheduler: Cosine annealing
- Loss: Weighted CrossEntropyLoss (inverse class frequency)
- Metrics logged per epoch: loss, accuracy, F1 (fire/nofire/macro)
- Gradient accumulation: 2 steps for ViT/ResNet (eff. batch=32), 4 steps for Hybrid (eff. batch=32)

### Freeze Configurations

**ViT-B/16** — progressive transformer block freezing:
| Config | What's frozen | Trainable % |
|--------|--------------|-------------|
| `freeze_none` | nothing | 100% |
| `freeze_patch` | patch embed + pos embed + CLS | ~99% |
| `freeze_patch_blocks0-3` | above + blocks 0-3 | ~66% |
| `freeze_patch_blocks0-5` | above + blocks 0-5 | ~50% |
| `freeze_patch_blocks0-8` | above + blocks 0-8 | ~25% |
| `freeze_patch_blocks0-11` | above + all blocks | ~0.5% |

**ResNet-50** — progressive layer freezing:
| Config | What's frozen | Trainable % |
|--------|--------------|-------------|
| `freeze_none` | nothing | 100% |
| `freeze_conv1` | conv1 + bn1 | ~99.7% |
| `freeze_conv1_layer1` | above + layer1 (3 blocks) | ~93% |
| `freeze_conv1_layer1-2` | above + layer2 (4 blocks) | ~78% |
| `freeze_conv1_layer1-3` | above + layer3 (6 blocks) | ~45% |
| `freeze_conv1_layer1-4` | above + layer4 (3 blocks) | ~0.1% |

**Hybrid CNN-ViT** — component-level freezing:
| Config | What's frozen | What's trainable | Trainable % |
|--------|--------------|-----------------|-------------|
| `freeze_none` | nothing | everything | 100% |
| `freeze_backbone` | ResNet backbone | transformer + proj + head | ~50% |
| `freeze_backbone_proj` | backbone + conv projection | transformer + head | ~50% |
| `freeze_transformer_only` | transformer + CLS token | backbone + proj + head | ~50% |
| `freeze_backbone_proj_transformer` | everything except head | head only | ~0.01% |
