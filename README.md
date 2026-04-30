# Wildfire Detection: A Layer-Freezing Ablation Study

This project investigates how progressive layer freezing during transfer learning affects classification performance for binary wildfire detection (fire vs. no fire). Three pretrained architectures (ViT-B/16, ResNet-50, and a Hybrid CNN-ViT) are systematically evaluated across multiple freezing configurations, each repeated over five random seeds for statistical rigour. The goal is to determine how much fine-tuning is actually necessary when adapting large vision models to a domain-specific task.

## Paper

Ahmad, H., Modassir Mushtaq, I., and Machin, O. *Fire-Freezing: Layer-Freezing Ablations of ViT-B/16, ResNet-50, and a Hybrid CNN-ViT for Wildfire Classification*. UWE Bristol, 2026.

This repository is the companion artefact to the paper. All numbers, tables, figures, and statistical claims in the paper are reproducible from the seed JSONs under `results/` via `src/analyse_results.py` and `scripts/paper_extract.py`.

```bibtex
@misc{ahmad2026firefreezing,
  title  = {Fire-Freezing: Layer-Freezing Ablations of ViT-B/16, ResNet-50, and a Hybrid CNN-ViT for Wildfire Classification},
  author = {Ahmad, Hasaan and Modassir Mushtaq, Ishaq and Machin, Orion},
  year   = {2026},
  note   = {School of Computing and Creative Technologies, UWE Bristol}
}
```

## Quick start

Reproduce the headline ViT result from a fresh checkout (~5 minutes on a single GPU):

```bash
git clone https://github.com/ssrhaso/wildfire.git && cd wildfire
make setup-linux                                                     # venv + deps + Kaggle download + preprocess
source venv/bin/activate
make run-one MODEL=vit CONFIG=freeze_patch_blocks0-8 SEED=0          # one cell of paper Table 4
cat results/vit/freeze_patch_blocks0-8/seed_0.json                   # per-run metrics, hyperparameters, env
```

Other entry points:

| Command                  | What it does                                                                  |
| ------------------------ | ----------------------------------------------------------------------------- |
| `make help`              | List all targets with one-line descriptions.                                  |
| `make test-vit`          | Single-epoch smoke test; verifies the dataset and model load correctly.       |
| `make experiments-vit`   | Run all 6 ViT freezing configs across 5 seeds (30 runs).                      |
| `make reproduce-paper`   | Run the full 165-run sweep and regenerate all analysis (Linux/macOS).         |
| `make analyse-all`       | Re-aggregate `results/<model>/` JSONs into `results/analysis/` CSVs and plots.|

Replace `setup-linux` with `setup-windows`, and `experiments-*` with `experiments-*-win`, on Windows.

## Results at a glance

| Model          | Best Config                | Best Test Accuracy | Linear Probe Accuracy |
| -------------- | -------------------------- | ------------------ | --------------------- |
| ViT-B/16       | `freeze_patch_blocks0-8` | 99.32 ± 0.16%     | 98.33% (head only)    |
| ResNet-50      | `freeze_conv1_layer1-3`  | 98.73 ± 0.27%     | 96.63% (head only)    |
| Hybrid CNN-ViT | `freeze_backbone`        | 98.78 ± 0.15%     | 68.46% (head only)    |

Each architecture exhibits a distinct accuracy-vs-freeze profile (near-monotonic for ViT-B/16, inverted-U for ResNet-50, wide plateau for the Hybrid), and full fine-tuning is the highest-variance configuration in every case. The full per-config tables, statistical tests, and Grad-CAM analysis are in §5 of the paper. The raw per-seed JSONs that those tables were aggregated from live under [results/vit/](results/vit/), [results/resnet/](results/resnet/), and [results/hybrid/](results/hybrid/); rerun [src/analyse_results.py](src/analyse_results.py) to regenerate the aggregated CSVs and plots from them.

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

### Preprocessing Pipeline

End-to-end steps applied by [src/preprocess.py](src/preprocess.py) to go from raw Kaggle downloads to `data/processed/labels.csv`:

| Step                | Details                                                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Image collection    | Recursive walk across the three source roots; accepted extensions .jpg, .jpeg, .png, .tif, .tiff                                                              |
| Label harmonisation | `flamevision`: fire/nofire kept; `dani215`: fire/not_fire renamed to fire/nofire; `minha`: fire-only subset used                                        |
| Integrity check     | `PIL.Image.verify()` on every file; corrupt files dropped per source                                                                                        |
| Deduplication       | Perceptual hash (`imagehash.phash`) computed per image; `drop_duplicates(subset='phash', keep='first')` removes near-duplicates within and across sources |
| Split               | Two-stage `sklearn.model_selection.train_test_split`, stratified on label, `random_state=42`, 80 / 10 / 10 for train / val / test                         |
| Resize and save     | Each image opened, converted to RGB, resized to 224 x 224 with LANCZOS resampling, re-encoded as JPEG into `data/processed/{split}/{label}/`                |
| Label CSV           | Emitted as `path,label,split` with label encoded as integer (fire=1, nofire=0)                                                                              |

The stratified split is deterministic given `RANDOM_STATE=42`, so rerunning preprocessing produces byte-identical `labels.csv` contents (path ordering aside).

## Architectures

**ViT-B/16** uses the standard Vision Transformer with 12 encoder blocks operating on 16x16 patch embeddings (768-d). A dropout layer (p=0.1) precedes the binary classification head. Total parameters: ~85.8M.

**ResNet-50** follows the standard bottleneck architecture (conv1, layers 1-4) with global average pooling. The fully connected layer is replaced with a dropout (p=0.1) and linear head. Total parameters: ~23.5M.

**Hybrid CNN-ViT** uses ResNet-50 (truncated to layer3) as a feature extractor, projects CNN outputs from 1024-d to 768-d via a 1x1 convolution, then passes the resulting token sequence through 12 ViT transformer blocks. Total parameters: ~94.5M.

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
| `freeze_patch_blocks0-11` | Above + all blocks (linear probe)          | ~0.004%     |

### ResNet-50 (progressive layer freezing)

| Config                    | What is Frozen                | Trainable % |
| ------------------------- | ----------------------------- | ----------- |
| `freeze_none`           | Nothing                       | 100%        |
| `freeze_conv1`          | conv1 + bn1                   | ~99.9%      |
| `freeze_conv1_layer1`   | Above + layer1 (3 blocks)     | ~99%        |
| `freeze_conv1_layer1-2` | Above + layer2 (4 blocks)     | ~93.9%      |
| `freeze_conv1_layer1-3` | Above + layer3 (6 blocks)     | ~63.7%      |
| `freeze_conv1_layer1-4` | Above + layer4 (linear probe) | ~0.02%      |

### Hybrid CNN-ViT (component-level freezing)

Standard configurations (prefix `freeze_`):

| Config               | What is Frozen                 | Trainable % |
| -------------------- | ------------------------------ | ----------- |
| `none`             | Nothing                        | ~100%       |
| `backbone`         | ResNet backbone (layers 1-3)   | ~91.0%      |
| `backbone_proj`    | Backbone + 1x1 conv projection | ~90.1%      |
| `blocks0-3`        | ViT blocks 0-3                 | ~70.0%      |
| `blocks0-5`        | ViT blocks 0-5                 | ~55.0%      |
| `blocks0-8`        | ViT blocks 0-8                 | ~32.5%      |
| `blocks0-11`       | ViT blocks 0-11                | ~10.0%      |
| `transformer_only` | ViT transformer + CLS token    | ~9.9%       |
| `transformer_proj` | Transformer + conv projection  | ~9.0%       |

Progressive backbone+projection (prefix `freeze_backbone_proj_`):

| Config          | What is Frozen                      | Trainable % |
| --------------- | ----------------------------------- | ----------- |
| `blocks0-3`   | Backbone + proj + ViT blocks 0-3    | ~60.0%      |
| `blocks0-5`   | Backbone + proj + ViT blocks 0-5    | ~45.0%      |
| `blocks0-8`   | Backbone + proj + ViT blocks 0-8    | ~22.5%      |
| `blocks0-11`  | Backbone + proj + all ViT blocks    | ~0.0%       |
| `transformer` | Backbone + proj + transformer + CLS | ~0.0%       |

BatchNorm-frozen variants (suffix `_bnfrozen`) lock the ResNet backbone's BN running statistics to ImageNet values. They are run for the seven standard configs above and excluded from the main analysis after universal convergence failure (52-69% accuracy); see the Hybrid results table.

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

### Full Training Recipe

Additional details not covered above, grounded in [src/run_experiment.py](src/run_experiment.py) and [src/dataset.py](src/dataset.py):

| Component              | Value                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------- |
| Input resolution       | 224 x 224                                                                                   |
| Normalisation          | ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]                              |
| Train augmentation     | RandomResizedCrop(224, scale=(0.7, 1.0)), HFlip(p=0.5), VFlip(p=0.5), Rotation(15 deg)      |
|                        | ColorJitter(b=0.3, c=0.3, s=0.2, h=0.05), RandomGrayscale(p=0.05), GaussianBlur(k=3, p=0.2) |
|                        | RandomErasing(p=0.1, scale=(0.02, 0.1))                                                     |
| Eval transform         | Resize(256), CenterCrop(224)                                                                |
| AdamW betas / eps      | (0.9, 0.999) / 1e-8 (PyTorch defaults)                                                      |
| Gradient clipping      | L2 norm clipped at max_norm=1.0                                                             |
| Mixed precision        | Optional via `--amp` (torch.amp.autocast on CUDA)                                         |
| Early stopping monitor | Validation loss (patience=5, min_delta=1e-4)                                                |
| Model selection        | Best-val-loss checkpoint restored before test evaluation                                    |
| DataLoader             | num_workers=4, pin_memory=True, drop_last=True (train only)                                 |
| Class weighting        | Inverse frequency computed on train split, passed to CrossEntropyLoss                       |
| Reported test metrics  | Accuracy, loss, precision/recall/F1 (per class), confusion matrix                           |

## Getting Started

### Prerequisites

- Python 3.10+
- Kaggle account and API key (see [Kaggle API docs](https://www.kaggle.com/docs/api))
- ~15 GB free disk space
- Nvidia GPU

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
make experiments-hybrid           # Hybrid CNN-ViT (21 configs x 5 seeds = 105 runs)
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

### Environment and Reproducibility

Runtime dependencies are pinned to minimums in [requirements.txt](requirements.txt):

| Component    | Constraint | Role                                        |
| ------------ | ---------- | ------------------------------------------- |
| Python       | 3.10+      | Interpreter                                 |
| torch        | >= 2.2.0   | Training loop, AMP, determinism flags       |
| torchvision  | >= 0.17.0  | Pretrained weights and `transforms.v2`    |
| timm         | >= 0.9.12  | Hybrid architecture building blocks         |
| scikit-learn | >= 1.3.0   | F1, classification report, stratified split |
| scipy        | >= 1.11.0  | Welch's t-test                              |
| imagehash    | >= 4.3.1   | Perceptual-hash deduplication               |
| grad-cam     | >= 1.4.8   | Grad-CAM visualisations                     |
| wandb        | >= 0.15.0  | Experiment tracking                         |

Determinism is established inside `set_seed` in [src/run_experiment.py](src/run_experiment.py):

```python
random.seed(seed); np.random.seed(seed)
torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
```

Per-run reproducibility notes:

- Every invocation logs GPU name, total GPU memory, and per-epoch wall-clock time, all captured in the `seed_<n>.json` artefact under `results/<model>/<freeze_config>/`.
- `run_config` in the JSON records the full argument set (optimiser, lr, weight decay, dropout, batch size, gradient accumulation steps, effective batch size, AMP flag, dataset counts, class weights, and parameter counts) so a single file reproduces the exact setup.
- Re-running an already-completed `(model, freeze_config, seed)` triple is skipped by the shell runners under [scripts/](scripts/); delete the JSON to force a rerun.

Minimal single-run reproduction:

```bash
python src/run_experiment.py --config configs/vit.yaml --freeze-config freeze_patch_blocks0-8 --seed 0
```

## Analysis

Analysis runs automatically at the end of each experiment script. To regenerate manually:

```bash
python src/analyse_results.py --model vit
python src/analyse_results.py --model resnet
python src/analyse_results.py --model hybrid
```

Outputs are written to `results/analysis/<model>/` and include summary statistics (CSV), pairwise t-tests with Cohen's d, box plots, validation curves, learning rate schedules, and confusion matrices.

### Metric Definitions

Reported metrics use the `fire` class as the positive label (`label=1` in [data/processed/labels.csv](data/processed/labels.csv)). Test-set evaluation occurs on the checkpoint with the lowest validation loss.

| Metric                 | Definition                                                                  |
| ---------------------- | --------------------------------------------------------------------------- |
| Accuracy               | (TP + TN) / (TP + TN + FP + FN)                                             |
| Precision (fire)       | TP / (TP + FP)                                                              |
| Recall (fire)          | TP / (TP + FN)                                                              |
| F1 (fire)              | 2 * P * R / (P + R), computed via `sklearn.metrics.f1_score(pos_label=1)` |
| F1 (nofire)            | Same formula with `pos_label=0`                                           |
| Macro F1               | Mean of F1 (fire) and F1 (nofire)                                           |
| Cross-seed aggregation | Per-config mean and std over five seeds (0, 5, 10, 15, 20)                  |
| Significance test      | Welch's t-test via `scipy.stats.ttest_ind(a, b, equal_var=False)`         |
| Effect size            | Cohen's d with pooled standard deviation (ddof=1 per group, pooled n-2)     |
| Significance threshold | p < 0.05 flags a pair as significant in `statistics.csv`                  |

### Figure Manifest

Every artefact produced by the analysis pipeline, with its intended purpose. All plots are written as both `.pdf` (vector) and `.png` (raster) except the CSVs.

| Artefact                                          | Contents                                                              |
| ------------------------------------------------- | --------------------------------------------------------------------- |
| `results/analysis/<model>/summary.csv`          | Per-config aggregated metrics (mean and std across seeds)             |
| `results/analysis/<model>/statistics.csv`       | Pairwise Welch's t-tests and Cohen's d between every pair of configs  |
| `results/analysis/<model>/boxplot_accuracy.*`   | Test accuracy distribution across seeds, one box per freeze config    |
| `results/analysis/<model>/confusion_matrices.*` | Grid of test-set confusion matrices, one per freeze config            |
| `results/analysis/<model>/train_val_curves.*`   | Train and validation loss / accuracy per epoch, overlaid across seeds |
| `results/analysis/<model>/val_curves.*`         | Validation loss / accuracy only, per epoch, across seeds              |
| `results/analysis/<model>/val_f1_fire_curves.*` | Per-epoch validation F1 on the fire class, across seeds               |
| `results/analysis/<model>/lr_schedule.*`        | Cosine annealing learning-rate schedule actually applied              |
| `results/analysis/cross_model_comparison.*`     | Best-config test accuracy compared across ViT, ResNet, and Hybrid     |

## Project Structure

```
wildfire/
  configs/
    config.yaml                # hybrid model architecture config
    resnet.yaml                # ResNet-50 training config
    vit.yaml                   # ViT-B/16 training config
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

All 165 runs complete (33 configs x 5 seeds).

- **ViT-B/16:** 6/6 configs. Best: `freeze_patch_blocks0-8` at 99.32%.
- **ResNet-50:** 6/6 configs. Best: `freeze_conv1_layer1-3` at 98.73%.
- **Hybrid CNN-ViT:** 21/21 configs (includes BatchNorm-frozen variants). Best: `freeze_backbone` at 98.78%.
- **Analysis pipeline:** statistical tests, box plots, validation curves, confusion matrices, cross-model comparison, Grad-CAM. All operational.
- **BatchNorm-frozen variants:** excluded from main analysis. Freezing BN with a trainable backbone collapses accuracy to 52-69%.

## Future Work

- Per-layer CNN freezing within the Hybrid (currently the ResNet backbone is treated as a single unit).
- Self-supervised pretraining (DINOv2, MAE, CLIP) and other backbones (Swin, ConvNeXt, EVA).
- Differential learning rates per unfrozen segment.
- Multi-class fire-severity prediction; cross-source generalisation tests.
- ROC/AUC, t-SNE/UMAP feature visualisations, and Grad-CAM across all freezing depths.

## References

Architectures and methods:

- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- Loshchilov and Hutter, *Decoupled Weight Decay Regularization (AdamW)*, ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
- Loshchilov and Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts (cosine annealing)*, ICLR 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983)
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Ioffe and Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

Software and pretrained weights:

- PyTorch: [pytorch.org](https://pytorch.org)
- torchvision models: [pytorch.org/vision/stable/models.html](https://pytorch.org/vision/stable/models.html)
- timm (PyTorch Image Models): [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- pytorch-grad-cam: [github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
- imagehash: [github.com/JohannesBuchner/imagehash](https://github.com/JohannesBuchner/imagehash)
- Weights & Biases: [wandb.ai](https://wandb.ai)

Datasets:

- FlameVision: [kaggle.com/datasets/warcoder/flamevision-dataset-for-wildfire-classification](https://www.kaggle.com/datasets/warcoder/flamevision-dataset-for-wildfire-classification)
- Dani215 Fire Dataset: [kaggle.com/datasets/dani215/fire-dataset](https://www.kaggle.com/datasets/dani215/fire-dataset)
- Forest Fire / Smoke (Minha): [kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset](https://www.kaggle.com/datasets/amerzishminha/forest-fire-smoke-and-non-fire-image-dataset)
