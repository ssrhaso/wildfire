"""Evaluation and visualisation utilities for wildfire classification models.

Loads trained checkpoints, runs inference on the test set, and produces:
    - ROC curves with AUC scores per freeze configuration
    - t-SNE feature space visualisations at different freezing levels

Intermediate results (probabilities, features) are cached to .npz files
so that plots can be regenerated without re-running inference.

Usage:
    # ROC curves for a model (all available configs)
    python src/evaluate.py --model vit --mode roc

    # t-SNE visualisation (default subset of configs)
    python src/evaluate.py --model vit --mode tsne

    # Both analyses
    python src/evaluate.py --model vit --mode all

    # Specific configs only
    python src/evaluate.py --model vit --mode roc \
        --configs freeze_none freeze_patch_blocks0-11

    # Custom seed for t-SNE
    python src/evaluate.py --model hybrid --mode tsne --seed 5
"""

import argparse
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from analyse_results import CONFIG_LABELS
from dataset import WildfireDataset, get_eval_transform
from freeze import apply_freeze
from models.hybrid import HybridCNNViT
from models.resnet import ResNetClassifier
from models.vit import ViTClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Publication-quality matplotlib configuration (serif, sized for single-column)
PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.5,
}

# Default configs for t-SNE (sparse subset spanning the freezing spectrum)
TSNE_DEFAULTS = {
    "vit": [
        "freeze_none",
        "freeze_patch_blocks0-5",
        "freeze_patch_blocks0-11",
    ],
    "resnet": [
        "freeze_none",
        "freeze_conv1_layer1-2",
        "freeze_conv1_layer1-4",
    ],
    "hybrid": [
        "freeze_none",
        "freeze_backbone",
        "freeze_backbone_proj_transformer",
    ],
}

# Default configs for ROC curves
ROC_DEFAULTS = {
    "vit": [
        "freeze_none",
        "freeze_patch",
        "freeze_patch_blocks0-3",
        "freeze_patch_blocks0-5",
        "freeze_patch_blocks0-8",
        "freeze_patch_blocks0-11",
    ],
    "resnet": [
        "freeze_none",
        "freeze_conv1",
        "freeze_conv1_layer1",
        "freeze_conv1_layer1-2",
        "freeze_conv1_layer1-3",
        "freeze_conv1_layer1-4",
    ],
    "hybrid": [
        "freeze_none",
        "freeze_backbone",
        "freeze_backbone_proj",
        "freeze_transformer_only",
        "freeze_backbone_proj_transformer",
        "freeze_blocks0-11",
    ],
}

# Colourblind-safe palette (Okabe-Ito inspired)
CURVE_PALETTE = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#999999",
]
CLASS_COLOURS = {"nofire": "#3B7DD8", "fire": "#D94A4A"}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Captures penultimate-layer features via a forward hook."""

    def __init__(self, model: nn.Module, model_name: str) -> None:
        self.features: Optional[torch.Tensor] = None
        self.model_name = model_name

        if model_name == "vit":
            target = model.encoder.encoder.ln
        elif model_name == "resnet":
            target = model.encoder.avgpool
        elif model_name == "hybrid":
            target = model.transformer.ln
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self._hook = target.register_forward_hook(self._hook_fn)

    def _hook_fn(
        self, module: nn.Module, input: tuple, output: torch.Tensor
    ) -> None:
        if self.model_name in ("vit", "hybrid"):
            # CLS token from sequence output: (B, seq_len, D) -> (B, D)
            self.features = output[:, 0, :].detach()
        else:
            # ResNet avgpool: (B, 2048, 1, 1) -> (B, 2048)
            self.features = output.flatten(1).detach()

    def remove(self) -> None:
        self._hook.remove()


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def build_model(
    model_name: str, freeze_config: str, checkpoint_path: Path
) -> nn.Module:
    """Build model, apply freeze config, and load checkpoint weights."""
    if model_name == "vit":
        model = ViTClassifier(num_classes=2, dropout=0.0, freeze_encoder=False)
    elif model_name == "resnet":
        model = ResNetClassifier(num_classes=2, dropout=0.0, freeze_encoder=False)
    elif model_name == "hybrid":
        model = HybridCNNViT(num_classes=2, dropout_rate=0.0)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    apply_freeze(model, model_name, freeze_config)

    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return softmax probabilities, features, and labels."""
    extractor = FeatureExtractor(model, model_name)
    all_probs, all_features, all_labels = [], [], []

    for images, labels in tqdm(loader, desc="  Inference"):
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_features.append(extractor.features.cpu().numpy())
        all_labels.append(labels.numpy())

    extractor.remove()

    return (
        np.concatenate(all_probs),
        np.concatenate(all_features),
        np.concatenate(all_labels),
    )


def get_cached_or_compute(
    model_name: str,
    config: str,
    seed: int,
    cache_dir: Path,
    loader: DataLoader,
    results_dir: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load cached inference results or compute from checkpoint."""
    cache_path = cache_dir / f"{config}_seed{seed}.npz"

    if cache_path.exists():
        print(f"  Loading cached: {cache_path.name}")
        data = np.load(cache_path)
        return data["probs"], data["features"], data["labels"]

    ckpt_path = (
        results_dir / "checkpoints" / model_name / config / f"seed_{seed}_best.pt"
    )
    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        return None

    print(f"  Running inference: {config} (seed {seed})")
    model = build_model(model_name, config, ckpt_path)
    probs, features, labels = run_inference(model, loader, model_name)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, probs=probs, features=features, labels=labels)
    print(f"  Cached: {cache_path.name}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return probs, features, labels


def find_available_seeds(
    results_dir: Path, model_name: str, config: str
) -> List[int]:
    """Return seed numbers that have both JSON results and a checkpoint."""
    json_dir = results_dir / model_name / config
    ckpt_dir = results_dir / "checkpoints" / model_name / config

    seeds = []
    for json_file in sorted(json_dir.glob("seed_*.json")):
        seed = int(json_file.stem.split("_")[1])
        ckpt = ckpt_dir / f"seed_{seed}_best.pt"
        if ckpt.exists():
            seeds.append(seed)
    return seeds


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def compute_mean_roc(
    seed_results: List[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute mean ROC curve with standard deviation band across seeds.

    Each seed's TPR is interpolated onto a shared FPR grid so that
    point-wise mean and standard deviation are well-defined.
    """
    base_fpr = np.linspace(0, 1, 200)
    tprs, aucs = [], []

    for probs, labels in seed_results:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        interp_tpr = np.interp(base_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0) if len(tprs) > 1 else np.zeros_like(mean_tpr)

    return base_fpr, mean_tpr, std_tpr, float(np.mean(aucs)), float(np.std(aucs))


def plot_roc_curves(
    roc_data: Dict[str, Tuple],
    output_dir: Path,
    model_name: str,
) -> None:
    """Plot mean ROC curves with +/-1 std shading for multiple configs."""
    fig, ax = plt.subplots(figsize=(4.5, 4.0))

    for i, (config, (mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc)) in enumerate(
        roc_data.items()
    ):
        colour = CURVE_PALETTE[i % len(CURVE_PALETTE)]
        label = CONFIG_LABELS.get(config, config)

        if std_auc > 0:
            legend_str = f"{label} ({mean_auc:.3f} \u00b1 {std_auc:.3f})"
        else:
            legend_str = f"{label} ({mean_auc:.3f})"

        ax.plot(mean_fpr, mean_tpr, color=colour, label=legend_str)

        if std_tpr.max() > 0:
            ax.fill_between(
                mean_fpr,
                np.clip(mean_tpr - std_tpr, 0, 1),
                np.clip(mean_tpr + std_tpr, 0, 1),
                color=colour,
                alpha=0.12,
            )

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Chance (0.500)")

    # Zoomed inset when all AUCs are high (common in this study)
    all_aucs = [d[3] for d in roc_data.values()]
    if all_aucs and min(all_aucs) > 0.95:
        axins = ax.inset_axes([0.35, 0.05, 0.58, 0.50])
        for i, (config, (mean_fpr, mean_tpr, std_tpr, _, _)) in enumerate(
            roc_data.items()
        ):
            colour = CURVE_PALETTE[i % len(CURVE_PALETTE)]
            axins.plot(mean_fpr, mean_tpr, color=colour, linewidth=1.2)
            if std_tpr.max() > 0:
                axins.fill_between(
                    mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    color=colour,
                    alpha=0.12,
                )
        axins.set_xlim(-0.005, 0.12)
        axins.set_ylim(0.88, 1.005)
        axins.grid(True, alpha=0.3)
        axins.tick_params(labelsize=7)
        ax.indicate_inset_zoom(axins, edgecolor="grey", alpha=0.6)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves - {model_name.upper()}")
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(output_dir / f"roc_curves.{fmt}")
    plt.close(fig)
    print(f"  Saved: roc_curves.png/pdf")


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def plot_tsne(
    tsne_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
    model_name: str,
) -> None:
    """Plot t-SNE embeddings for selected freeze configurations.

    Each subplot shows the test-set feature space for one config, coloured
    by true class label.  PCA pre-reduction to 50 dims is applied for
    stability and speed when the feature dimension exceeds 50.
    """
    n = len(tsne_data)
    fig, axes = plt.subplots(1, n, figsize=(3.3 * n, 3.0))
    if n == 1:
        axes = [axes]

    for ax, (config, (features, labels)) in zip(axes, tsne_data.items()):
        # PCA pre-reduction (standard practice for high-dim t-SNE input)
        if features.shape[1] > 50:
            features = PCA(n_components=50, random_state=42).fit_transform(features)

        embedding = TSNE(
            n_components=2,
            random_state=42,
            perplexity=min(30, len(features) // 4),
            n_iter=1000,
            init="pca",
            learning_rate="auto",
        ).fit_transform(features)

        for cls_idx, cls_name in enumerate(["nofire", "fire"]):
            mask = labels == cls_idx
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=CLASS_COLOURS[cls_name],
                label=cls_name.capitalize(),
                s=4,
                alpha=0.45,
                edgecolors="none",
                rasterized=True,
            )

        label = CONFIG_LABELS.get(config, config)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    # Shared legend below all subplots
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        f"t-SNE Feature Embeddings - {model_name.upper()}",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    for fmt in ("png", "pdf"):
        fig.savefig(output_dir / f"tsne.{fmt}")
    plt.close(fig)
    print(f"  Saved: tsne.png/pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluation: ROC curves and t-SNE visualisations"
    )
    p.add_argument(
        "--model", required=True, choices=["vit", "resnet", "hybrid"],
        help="Model architecture to evaluate.",
    )
    p.add_argument(
        "--mode", default="all", choices=["roc", "tsne", "all"],
        help="Which analysis to run (default: all).",
    )
    p.add_argument(
        "--configs", nargs="+", default=None,
        help="Freeze configs to evaluate. Defaults to a curated subset per mode.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Seed for t-SNE (single seed). ROC uses all available seeds.",
    )
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--batch-size", type=int, default=64)
    default_workers = 0 if platform.system() == "Windows" else 4
    p.add_argument("--num-workers", type=int, default=default_workers)
    args = p.parse_args()

    plt.rcParams.update(PAPER_RC)

    results_dir = Path(args.results_dir)
    output_dir = results_dir / "analysis" / args.model
    cache_dir = output_dir / "eval_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Device: {DEVICE}")
    print(f"  Model:  {args.model}\n")

    test_ds = WildfireDataset(split="test", transform=get_eval_transform())
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    do_roc = args.mode in ("roc", "all")
    do_tsne = args.mode in ("tsne", "all")

    # ---- ROC curves ----
    if do_roc:
        roc_configs = args.configs or ROC_DEFAULTS.get(args.model, [])
        print(f"  ROC analysis: {len(roc_configs)} configs\n")

        roc_data = {}
        for config in roc_configs:
            seeds = find_available_seeds(results_dir, args.model, config)
            if not seeds:
                print(f"  Skipping {config}: no checkpoints found")
                continue

            seed_results = []
            for seed in seeds:
                result = get_cached_or_compute(
                    args.model, config, seed, cache_dir, test_loader, results_dir,
                )
                if result is not None:
                    probs, _, labels = result
                    seed_results.append((probs, labels))

            if seed_results:
                roc_data[config] = compute_mean_roc(seed_results)

        if roc_data:
            plot_roc_curves(roc_data, output_dir, args.model)

            # Print AUC summary table
            print("\n  AUC Summary")
            print(f"  {'Config':<35} {'AUC':>14}  Seeds")
            print(f"  {'-' * 35} {'-' * 14}  -----")
            for config, (_, _, _, m_auc, s_auc) in roc_data.items():
                label = CONFIG_LABELS.get(config, config)
                seeds = find_available_seeds(results_dir, args.model, config)
                if s_auc > 0:
                    print(f"  {label:<35} {m_auc:.4f} +/- {s_auc:.4f}  {len(seeds)}")
                else:
                    print(f"  {label:<35} {m_auc:.4f}          {len(seeds)}")
        else:
            print("  No data available for ROC curves.")

    # ---- t-SNE ----
    if do_tsne:
        tsne_configs = args.configs or TSNE_DEFAULTS.get(args.model, [])
        print(f"\n  t-SNE analysis: {len(tsne_configs)} configs (seed {args.seed})\n")

        tsne_data = {}
        for config in tsne_configs:
            result = get_cached_or_compute(
                args.model, config, args.seed, cache_dir, test_loader, results_dir,
            )
            if result is not None:
                _, features, labels = result
                tsne_data[config] = (features, labels)
            else:
                print(
                    f"  Skipping {config}: checkpoint not found for seed {args.seed}"
                )

        if tsne_data:
            plot_tsne(tsne_data, output_dir, args.model)
        else:
            print("  No data available for t-SNE.")

    print(f"\n  All outputs in: {output_dir}\n")


if __name__ == "__main__":
    main()
