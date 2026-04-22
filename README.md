# Wildfire Detection: A Layer-Freezing Ablation Study

This project investigates how progressive layer freezing during transfer learning affects classification performance for binary wildfire detection (fire vs. no fire). Three pretrained architectures (ViT-B/16, ResNet-50, and a Hybrid CNN-ViT) are systematically evaluated across multiple freezing configurations, each repeated over five random seeds for statistical rigour. The goal is to determine how much fine-tuning is actually necessary when adapting large vision models to a domain-specific task.

## Results

All values are mean ± std across seeds; configs sorted by test accuracy.

### ViT-B/16 (5/6 configs complete, `freeze_patch_blocks0-3` pending)

| Freeze Config               | Trainable (%) | Test Accuracy  | Test F1 (Fire) | Seeds |
| --------------------------- | ------------- | -------------- | -------------- | ----- |
| `freeze_patch_blocks0-8`  | 24.79%        | 99.32 ± 0.16% | 0.993 ± 0.002 | 5     |
| `freeze_patch_blocks0-5`  | 49.57%        | 99.07 ± 0.11% | 0.991 ± 0.001 | 4     |
| `freeze_patch_blocks0-11` | 0.00%         | 98.33 ± 0.08% | 0.984 ± 0.001 | 5     |
| `freeze_patch`            | 99.13%        | 96.41 ± 0.27% | 0.965 ± 0.003 | 5     |
| `freeze_none`             | 100.00%       | 93.61 ± 5.45% | 0.939 ± 0.052 | 5     |

Best: `freeze_patch_blocks0-8` (24.79% trainable) at 99.32%. Full fine-tuning performs worst with the highest variance, indicating overfitting. Head-only training (`freeze_patch_blocks0-11`, 3,074 parameters) still reaches 98.33%, showing ImageNet features transfer strongly. Moderate freezing beats both extremes.

### ResNet-50 (6/6 configs complete)

| Freeze Config             | Trainable (%) | Test Accuracy  | Test F1 (Fire) | Seeds |
| ------------------------- | ------------- | -------------- | -------------- | ----- |
| `freeze_conv1_layer1-3` | 63.66%        | 98.73 ± 0.27% | 0.988 ± 0.003 | 5     |
| `freeze_conv1_layer1-2` | 93.85%        | 98.70 ± 0.12% | 0.987 ± 0.001 | 5     |
| `freeze_conv1_layer1`   | 99.04%        | 98.57 ± 0.23% | 0.986 ± 0.002 | 5     |
| `freeze_conv1`          | 99.96%        | 98.54 ± 0.18% | 0.986 ± 0.002 | 5     |
| `freeze_conv1_layer1-4` | 0.02%         | 96.63 ± 0.38% | 0.967 ± 0.004 | 5     |
| `freeze_none`           | 100.00%       | 96.50 ± 4.95% | 0.968 ± 0.044 | 5     |

Best: `freeze_conv1_layer1-3` (63.66% trainable) at 98.73%. The top four configs cluster within 0.2 percentage points, so ResNet is largely insensitive to how much of the early stack is frozen. Full fine-tuning again shows the highest variance, and linear probing (`freeze_conv1_layer1-4`) drops only ~2 points despite training just 0.02% of parameters.

### Hybrid CNN-ViT (21/21 configs complete)

| Freeze Config                             | Trainable (%) | Test Accuracy   | Test F1 (Fire) | Seeds |
| ----------------------------------------- | ------------- | --------------- | -------------- | ----- |
| `freeze_backbone`                       | 90.96%        | 98.82 ± 0.23%  | 0.989 ± 0.002 | 5     |
| `freeze_blocks0-11`                     | 10.03%        | 98.79 ± 0.13%  | 0.988 ± 0.001 | 5     |
| `freeze_blocks0-8`                      | 32.53%        | 98.76 ± 0.19%  | 0.988 ± 0.002 | 5     |
| `freeze_backbone_proj`                  | 90.13%        | 98.75 ± 0.17%  | 0.988 ± 0.002 | 5     |
| `freeze_transformer_proj`               | 9.04%         | 98.52 ± 0.29%  | 0.986 ± 0.003 | 5     |
| `freeze_transformer_only`               | 9.87%         | 98.28 ± 1.16%  | 0.983 ± 0.011 | 5     |
| `freeze_blocks0-5`                      | 55.02%        | 98.24 ± 0.59%  | 0.983 ± 0.006 | 5     |
| `freeze_blocks0-3`                      | 70.01%        | 98.23 ± 0.97%  | 0.983 ± 0.010 | 5     |
| `freeze_none`                           | 100.00%       | 97.61 ± 1.23%  | 0.977 ± 0.012 | 5     |
| `freeze_backbone_proj_blocks0-3`        | 59.98%        | 97.48 ± 0.34%  | 0.976 ± 0.003 | 5     |
| `freeze_backbone_proj_blocks0-5`        | 44.99%        | 96.42 ± 0.55%  | 0.965 ± 0.006 | 5     |
| `freeze_backbone_proj_blocks0-8`        | 22.49%        | 92.47 ± 0.73%  | 0.926 ± 0.007 | 5     |
| `freeze_backbone_proj_blocks0-11`       | 0.00%         | 69.02 ± 2.37%  | 0.703 ± 0.021 | 5     |
| `freeze_transformer_proj_bnfrozen`      | 9.01%         | 68.84 ± 16.78% | 0.769 ± 0.118 | 5     |
| `freeze_backbone_proj_transformer`      | 0.00%         | 68.46 ± 2.54%  | 0.699 ± 0.022 | 5     |
| `freeze_transformer_only_bnfrozen`      | 9.84%         | 61.03 ± 20.58% | 0.742 ± 0.133 | 5     |
| `freeze_blocks0-11_bnfrozen`            | 10.00%        | 59.94 ± 12.81% | 0.718 ± 0.060 | 5     |
| `freeze_blocks0-5_bnfrozen`             | 54.98%        | 56.52 ± 10.50% | 0.704 ± 0.048 | 5     |
| `freeze_blocks0-3_bnfrozen`             | 69.98%        | 52.87 ± 2.33%  | 0.682 ± 0.002 | 5     |
| `freeze_none_bnfrozen`                  | 99.97%        | 51.83 ± 0.00%  | 0.683 ± 0.000 | 5     |
| `freeze_blocks0-8_bnfrozen`             | 32.49%        | 51.71 ± 0.25%  | 0.586 ± 0.217 | 5     |

Best: `freeze_backbone` at 98.82%. The top four configs (all ≥98.75%) span a wide trainable-parameter range (10--91%), showing multiple paths to near-optimal performance. Freezing backbone+proj and progressively adding transformer blocks collapses sharply once blocks 0--8 are frozen (92.47% -> 69.02%). All seven BN-frozen variants degrade to 52--69%, confirming that freezing BatchNorm while the backbone is trainable breaks the network -- the batch statistics and learnable affine parameters must evolve together.

### Cross-Model Summary

| Model        | Best Config                  | Best Test Accuracy | Linear Probe Accuracy |
| ------------ | ---------------------------- | ------------------ | --------------------- |
| ViT-B/16     | `freeze_patch_blocks0-8`   | 99.32 ± 0.16%     | 98.33% (head only)    |
| ResNet-50    | `freeze_conv1_layer1-3`    | 98.73 ± 0.27%     | 96.63% (head only)    |
| Hybrid CNN-ViT | `freeze_backbone`        | 98.82 ± 0.23%     | 68.46% (head only)    |

ViT-B/16 is the strongest architecture for this task. The Hybrid model matches ResNet but cannot be linear-probed effectively: its randomly-initialised conv projection and transformer stack need training to produce useful representations for the head.

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

| Step                 | Details                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------- |
| Image collection     | Recursive walk across the three source roots; accepted extensions .jpg, .jpeg, .png, .tif, .tiff |
| Label harmonisation  | `flamevision`: fire/nofire kept; `dani215`: fire/not_fire renamed to fire/nofire; `minha`: fire-only subset used |
| Integrity check      | `PIL.Image.verify()` on every file; corrupt files dropped per source                           |
| Deduplication        | Perceptual hash (`imagehash.phash`) computed per image; `drop_duplicates(subset='phash', keep='first')` removes near-duplicates within and across sources |
| Split                | Two-stage `sklearn.model_selection.train_test_split`, stratified on label, `random_state=42`, 80 / 10 / 10 for train / val / test |
| Resize and save      | Each image opened, converted to RGB, resized to 224 x 224 with LANCZOS resampling, re-encoded as JPEG into `data/processed/{split}/{label}/` |
| Label CSV            | Emitted as `path,label,split` with label encoded as integer (fire=1, nofire=0)                 |

The stratified split is deterministic given `RANDOM_STATE=42`, so rerunning preprocessing produces byte-identical `labels.csv` contents (path ordering aside).

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

### Full Training Recipe

Additional details not covered above, grounded in [src/run_experiment.py](src/run_experiment.py) and [src/dataset.py](src/dataset.py):

| Component                  | Value                                                                                          |
| -------------------------- | ---------------------------------------------------------------------------------------------- |
| Input resolution           | 224 x 224                                                                                      |
| Normalisation              | ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]                                 |
| Train augmentation         | RandomResizedCrop(224, scale=(0.7, 1.0)), HFlip(p=0.5), VFlip(p=0.5), Rotation(15 deg)         |
|                            | ColorJitter(b=0.3, c=0.3, s=0.2, h=0.05), RandomGrayscale(p=0.05), GaussianBlur(k=3, p=0.2)    |
|                            | RandomErasing(p=0.1, scale=(0.02, 0.1))                                                        |
| Eval transform             | Resize(256), CenterCrop(224)                                                                   |
| AdamW betas / eps          | (0.9, 0.999) / 1e-8 (PyTorch defaults)                                                         |
| Gradient clipping          | L2 norm clipped at max_norm=1.0                                                                |
| Mixed precision            | Optional via `--amp` (torch.amp.autocast on CUDA)                                              |
| Early stopping monitor     | Validation loss (patience=5, min_delta=1e-4)                                                   |
| Model selection            | Best-val-loss checkpoint restored before test evaluation                                       |
| DataLoader                 | num_workers=4, pin_memory=True, drop_last=True (train only)                                    |
| Class weighting            | Inverse frequency computed on train split, passed to CrossEntropyLoss                          |
| Reported test metrics      | Accuracy, loss, precision/recall/F1 (per class), confusion matrix                              |

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

### Environment and Reproducibility

Runtime dependencies are pinned to minimums in [requirements.txt](requirements.txt):

| Component   | Constraint      | Role                                                |
| ----------- | --------------- | --------------------------------------------------- |
| Python      | 3.10+           | Interpreter                                         |
| torch       | >= 2.2.0        | Training loop, AMP, determinism flags               |
| torchvision | >= 0.17.0       | Pretrained weights and `transforms.v2`              |
| timm        | >= 0.9.12       | Hybrid architecture building blocks                 |
| scikit-learn| >= 1.3.0        | F1, classification report, stratified split        |
| scipy       | >= 1.11.0       | Welch's t-test                                      |
| imagehash   | >= 4.3.1        | Perceptual-hash deduplication                       |
| grad-cam    | >= 1.4.8        | Grad-CAM visualisations                             |
| wandb       | >= 0.15.0       | Experiment tracking                                 |

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

| Metric                 | Definition                                                                 |
| ---------------------- | -------------------------------------------------------------------------- |
| Accuracy               | (TP + TN) / (TP + TN + FP + FN)                                            |
| Precision (fire)       | TP / (TP + FP)                                                             |
| Recall (fire)          | TP / (TP + FN)                                                             |
| F1 (fire)              | 2 * P * R / (P + R), computed via `sklearn.metrics.f1_score(pos_label=1)`  |
| F1 (nofire)            | Same formula with `pos_label=0`                                            |
| Macro F1               | Mean of F1 (fire) and F1 (nofire)                                          |
| Cross-seed aggregation | Per-config mean and std over five seeds (0, 5, 10, 15, 20)                 |
| Significance test      | Welch's t-test via `scipy.stats.ttest_ind(a, b, equal_var=False)`          |
| Effect size            | Cohen's d with pooled standard deviation (ddof=1 per group, pooled n-2)    |
| Significance threshold | p < 0.05 flags a pair as significant in `statistics.csv`                   |

### Figure Manifest

Every artefact produced by the analysis pipeline, with its intended purpose. All plots are written as both `.pdf` (vector) and `.png` (raster) except the CSVs.

| Artefact                                          | Contents                                                                  |
| ------------------------------------------------- | ------------------------------------------------------------------------- |
| `results/analysis/<model>/summary.csv`            | Per-config aggregated metrics (mean and std across seeds)                 |
| `results/analysis/<model>/statistics.csv`         | Pairwise Welch's t-tests and Cohen's d between every pair of configs      |
| `results/analysis/<model>/boxplot_accuracy.*`     | Test accuracy distribution across seeds, one box per freeze config        |
| `results/analysis/<model>/confusion_matrices.*`   | Grid of test-set confusion matrices, one per freeze config                |
| `results/analysis/<model>/train_val_curves.*`     | Train and validation loss / accuracy per epoch, overlaid across seeds     |
| `results/analysis/<model>/val_curves.*`           | Validation loss / accuracy only, per epoch, across seeds                  |
| `results/analysis/<model>/val_f1_fire_curves.*`   | Per-epoch validation F1 on the fire class, across seeds                   |
| `results/analysis/<model>/lr_schedule.*`          | Cosine annealing learning-rate schedule actually applied                  |
| `results/analysis/cross_model_comparison.*`       | Best-config test accuracy compared across ViT, ResNet, and Hybrid         |

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

- **Hybrid CNN-ViT:** complete (21 configs x 5 seeds = 105 runs, including BatchNorm-frozen variants). Best: `freeze_backbone` at 98.82%.
- **ResNet-50:** complete (6 configs x 5 seeds = 30 runs). Best: `freeze_conv1_layer1-3` at 98.73%.
- **ViT-B/16:** 5/6 configs complete. Best so far: `freeze_patch_blocks0-8` at 99.32%. Remaining: `freeze_patch_blocks0-3` (5 seeds) and 1 missing seed for `freeze_patch_blocks0-5`.
- **Analysis pipeline:** statistical tests, box plots, validation curves, confusion matrices, cross-model comparison, and Grad-CAM visualisations all operational.
- **BatchNorm investigation:** complete -- freezing BN while the backbone is unfrozen severely degrades performance (52--69% accuracy).

## Future Work

- Complete remaining ViT-B/16 config (`freeze_patch_blocks0-3`) and all ResNet-50 ablation runs
- Add ROC curves and AUC scores to the evaluation pipeline
- Generate cross-model comparison figures (accuracy vs trainable parameter %)
- Produce t-SNE/UMAP feature space visualisations at different freezing levels
- Explore progressive unfreezing schedules as an alternative to static freezing

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
