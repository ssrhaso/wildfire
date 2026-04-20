# Wildfire Detection: A Layer-Freezing Ablation Study

This project investigates how progressive layer freezing during transfer learning affects classification performance for binary wildfire detection (fire vs. no fire). Three pretrained architectures (ViT-B/16, ResNet-50, and a Hybrid CNN-ViT) are systematically evaluated across multiple freezing configurations, each repeated over five random seeds for statistical rigour. The goal is to determine how much fine-tuning is actually necessary when adapting large vision models to a domain-specific task.

## Results

### ViT-B/16

Five of six freezing configurations are complete (only `freeze_patch_blocks0-3` is still pending). All values are mean ± std across seeds.

| Freeze Config               | Trainable (%) | Test Accuracy  | Test F1 (Fire) | Seeds |
| --------------------------- | ------------- | -------------- | -------------- | ----- |
| `freeze_patch_blocks0-8`  | 24.79%        | 99.32 ± 0.16% | 0.993 ± 0.002 | 5     |
| `freeze_patch_blocks0-5`  | 49.57%        | 99.07 ± 0.11% | 0.991 ± 0.001 | 4     |
| `freeze_patch_blocks0-11` | 0.00%         | 98.33 ± 0.08% | 0.984 ± 0.001 | 5     |
| `freeze_patch`            | 99.13%        | 96.41 ± 0.27% | 0.965 ± 0.003 | 5     |
| `freeze_none`             | 100.00%       | 93.61 ± 5.45% | 0.939 ± 0.052 | 5     |

The best performance comes from freezing the patch embedding and blocks 0--8 (24.79% trainable), reaching 99.32% accuracy. Full fine-tuning (`freeze_none`) performs worst with the highest variance, indicating overfitting. Even training only the classification head (`freeze_patch_blocks0-11`, 3,074 parameters) achieves 98.33%, confirming that pretrained ImageNet features generalise well to wildfire imagery. Overall, there is a clear sweet spot: moderate freezing outperforms both extremes.

## Dataset

The dataset is assembled from three public Kaggle sources:

1. [FlameVision](https://www.kaggle.com/datasets/warcoder/flamevision-dataset-for-wildfire-classification)
2. [Dani215 Fire Dataset](https://www.kaggle.com/datasets/dani215/fire-dataset)
3. [Forest Fire/Smoke (Minha)](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset)

After deduplication via perceptual hashing and removal of corrupt files, the processed dataset contains **23,559 images** with a stratified 80/10/10 split:

| Split | Fire  | No Fire | Total  |
| ----- | ----- | ------- | ------ |
| Train | 9,763 | 9,084   | 18,847 |
| Val   | 1,220 | 1,136   | 2,356  |
| Test  | 1,221 | 1,135   | 2,356  |

Training augmentations include random resized crop, horizontal/vertical flips, rotation, colour jitter, grayscale, Gaussian blur, and random erasing. All images are normalised using ImageNet statistics.

## Architectures

**ViT-B/16** uses the standard Vision Transformer with 12 encoder blocks operating on 16x16 patch embeddings (768-d). A dropout layer (p=0.1) precedes the binary classification head. Total parameters: ~85.8M.

**ResNet-50** follows the standard bottleneck architecture (conv1, layers 1-4) with global average pooling. The fully connected layer is replaced with a dropout (p=0.1) and linear head. Total parameters: ~25.5M.

**Hybrid CNN-ViT** uses ResNet-50 (truncated to layer3) as a feature extractor, projects CNN outputs from 1024-d to 768-d via a 1x1 convolution, then passes the resulting token sequence through 12 ViT transformer blocks. Total parameters: ~105M.

All models are initialised with ImageNet-1K pretrained weights.

## Freezing Configurations

### ViT-B/16 (progressive transformer block freezing)

| Config                      | What is Frozen                             | Trainable % |
| --------------------------- | ------------------------------------------ | ----------- |
| `freeze_none`             | Nothing                                    | 100%        |
| `freeze_patch`            | Patch embed + positional embed + CLS token | ~99%        |
| `freeze_patch_blocks0-3`  | Above + blocks 0-3                         | ~66%        |
| `freeze_patch_blocks0-5`  | Above + blocks 0-5                         | ~50%        |
| `freeze_patch_blocks0-8`  | Above + blocks 0-8                         | ~25%        |
| `freeze_patch_blocks0-11` | Above + all blocks                         | ~0.5%       |

### ResNet-50 (progressive layer freezing)

| Config                    | What is Frozen            | Trainable % |
| ------------------------- | ------------------------- | ----------- |
| `freeze_none`           | Nothing                   | 100%        |
| `freeze_conv1`          | conv1 + bn1               | ~99.7%      |
| `freeze_conv1_layer1`   | Above + layer1 (3 blocks) | ~93%        |
| `freeze_conv1_layer1-2` | Above + layer2 (4 blocks) | ~78%        |
| `freeze_conv1_layer1-3` | Above + layer3 (6 blocks) | ~45%        |
| `freeze_conv1_layer1-4` | Above + layer4 (3 blocks) | ~0.1%       |

### Hybrid CNN-ViT (component-level freezing)

| Config                               | What is Frozen             | Trainable % |
| ------------------------------------ | -------------------------- | ----------- |
| `freeze_none`                      | Nothing                    | 100%        |
| `freeze_backbone`                  | ResNet backbone            | ~50%        |
| `freeze_backbone_proj`             | Backbone + conv projection | ~50%        |
| `freeze_transformer_only`          | Transformer + CLS token    | ~50%        |
| `freeze_backbone_proj_transformer` | Everything except head     | ~0.01%      |

## Training Setup

All models share the following configuration:

- **Optimiser:** AdamW (lr=1e-3, weight decay=1e-2)
- **Scheduler:** Cosine annealing (T_max=20)
- **Loss:** Weighted cross-entropy (inverse class frequency)
- **Epochs:** 20 (early stopping with patience=5, min delta=1e-4)
- **Gradient accumulation:** 2 steps for ViT/ResNet, 4 steps for Hybrid (effective batch size=32)
- **Seeds:** 0, 5, 10, 15, 20
- **Deterministic mode:** Enabled (cudnn.deterministic=True, cudnn.benchmark=True)
- **Experiment tracking:** Weights & Biases (project: `wildfire-freezing`)

## Getting Started

### Prerequisites

- Python 3.10+
- Kaggle account and API key (see [Kaggle API docs](https://www.kaggle.com/docs/api))
- ~15 GB free disk space

### One-Command Setup

This creates a virtual environment, installs all dependencies, downloads the datasets, extracts them, and runs preprocessing:

```bash
# Linux/Mac
make setup-linux

# Windows (via Git Bash or similar)
make setup-windows
```

Then activate the environment:

```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Running Experiments

Each model can be run independently. The scripts automatically skip completed runs, so they are safe to re-run.

```bash
# Linux/Mac
make experiments-vit              # ViT-B/16 (6 configs x 5 seeds = 30 runs)
make experiments-resnet           # ResNet-50 (6 configs x 5 seeds = 30 runs)
make experiments-hybrid           # Hybrid CNN-ViT (5 configs x 5 seeds = 25 runs)
make experiments-all              # All models sequentially

# Windows
make experiments-vit-win
make experiments-resnet-win
make experiments-hybrid-win
make experiments-all-win
```

To verify the setup with a single-epoch dry run:

```bash
make test-vit
make test-resnet
make test-hybrid
```

## Analysis

Analysis runs automatically at the end of each experiment script. To regenerate manually:

```bash
python src/analyse_results.py --model vit
python src/analyse_results.py --model resnet
python src/analyse_results.py --model hybrid
```

Outputs are written to `results/analysis/<model>/` and include summary statistics (CSV), pairwise t-tests with Cohen's d, box plots, validation curves, learning rate schedules, and confusion matrices.

## Project Structure

```
wildfire/
  configs/
    config.yaml                # hybrid model architecture config
  scripts/
    run_vit.ps1 / .sh          # ViT experiment runner
    run_resnet.ps1 / .sh       # ResNet experiment runner
    run_hybrid.ps1 / .sh       # Hybrid experiment runner
    run_all.ps1 / .sh          # all models sequentially
  src/
    dataset.py                 # WildfireDataset + DataLoaders
    preprocess.py              # raw data preprocessing + deduplication
    freeze.py                  # freezing configs for all 3 models
    run_experiment.py          # single-run training script
    analyse_results.py         # statistical analysis + plots
    evaluate.py                # evaluation utilities
    gradcam.py                 # Grad-CAM visualisation
    models/
      vit.py                   # ViT-B/16 classifier
      resnet.py                # ResNet-50 classifier
      hybrid.py                # Hybrid CNN-ViT classifier
  results/
    vit/                       # per-seed JSON results
    resnet/
    hybrid/
    checkpoints/               # best model weights per config
    analysis/                  # plots + summary CSVs
  Makefile
  requirements.txt
```

## Current Status

- **Hybrid CNN-ViT:** complete (21 configs x 5 seeds = 105 runs, including BatchNorm-frozen variants). Best config: `freeze_backbone` at 98.82% accuracy.
- **ViT-B/16:** nearly complete (5/6 configs done; `freeze_patch_blocks0-8` leads at 99.32%). Remaining: `freeze_patch_blocks0-3` (5 seeds)
- **ResNet-50:** (2/6 done;`freeze_conv1_layer1` (4 seeds done))
- **Analysis pipeline:** statistical tests, box plots, validation curves, confusion matrices, and Grad-CAM visualisations all operational
- **BatchNorm investigation:** complete -- freezing BN while the backbone is unfrozen severely degrades performance (51-68% accuracy)

## Future Work

- Complete remaining ViT-B/16 config (`freeze_patch_blocks0-3`) and all ResNet-50 ablation runs
- Add ROC curves and AUC scores to the evaluation pipeline
- Generate cross-model comparison figures (accuracy vs trainable parameter %)
- Produce t-SNE/UMAP feature space visualisations at different freezing levels
- Explore progressive unfreezing schedules as an alternative to static freezing
