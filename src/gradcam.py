"""Grad-CAM visualisation for wildfire classification models.

Generates activation heatmaps showing which image regions drive the model's
predictions. Supports ViT-B/16, ResNet-50, and Hybrid CNN-ViT with any
freeze configuration.

Usage:
    # Single config
    python src/gradcam.py --model vit --freeze-config freeze_patch_blocks0-11 --seed 0

    # Compare two configs side-by-side
    python src/gradcam.py --model vit \
        --freeze-config freeze_none --seed 0 \
        --compare freeze_patch_blocks0-11 --compare-seed 0

    # Custom image count and output
    python src/gradcam.py --model hybrid --freeze-config freeze_backbone --seed 0 \
        --num-images 20 --output-dir results/analysis/hybrid/gradcam
"""

import argparse
import platform
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

sys.path.insert(0, str(Path(__file__).parent))

from dataset import WildfireDataset, get_eval_transform, IMAGENET_MEAN, IMAGENET_STD
from freeze import apply_freeze
from models.hybrid import HybridCNNViT
from models.resnet import ResNetClassifier
from models.vit import ViTClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["nofire", "fire"]


def build_model(model_name: str, freeze_config: str, checkpoint_path: Path) -> nn.Module:
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


def get_target_layers(model: nn.Module, model_name: str) -> List[nn.Module]:
    """Return the target layer(s) for Grad-CAM activation extraction.

    ViT: last transformer encoder block (requires reshape_transform).
    ResNet: layer4 (final residual stage, 2048-channel feature maps).
    Hybrid: last transformer encoder block (requires reshape_transform).
    """
    if model_name == "vit":
        return [model.encoder.encoder.layers[-1].ln_1]
    elif model_name == "resnet":
        return [model.encoder.layer4[-1]]
    elif model_name == "hybrid":
        return [model.transformer.layers[-1].ln_1]
    raise ValueError(f"Unsupported model: {model_name}")


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape ViT/Hybrid encoder output from (B, 197, 768) to (B, 768, 14, 14).

    Drops the [CLS] token (index 0) and reshapes the 196 patch tokens
    back to a 14x14 spatial grid for Grad-CAM overlay.
    """
    result = tensor[:, 1:, :]  # drop CLS token
    h = w = int(result.shape[1] ** 0.5)  # 196 -> 14x14
    result = result.reshape(result.shape[0], h, w, result.shape[2])
    result = result.permute(0, 3, 1, 2)  # (B, C, H, W)
    return result


def get_reshape_transform(model_name: str):
    """Return reshape_transform for transformer-based models, None for CNNs."""
    if model_name in ("vit", "hybrid"):
        return vit_reshape_transform
    return None


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor back to a [0, 1] RGB numpy array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def find_checkpoint(model_name: str, freeze_config: str, seed: int,
                    results_dir: Path) -> Path:
    """Locate the best checkpoint file for a given run."""
    ckpt_path = results_dir / "checkpoints" / model_name / freeze_config / f"seed_{seed}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run the training first, or check that checkpoints were not deleted.\n"
            f"  python src/run_experiment.py --model {model_name} "
            f"--freeze-config {freeze_config} --seed {seed}"
        )
    return ckpt_path


def select_images(dataset: WildfireDataset, num_images: int,
                  balanced: bool = True) -> List[int]:
    """Select image indices from the dataset, optionally balanced by class."""
    if not balanced:
        return list(range(min(num_images, len(dataset))))

    fire_indices = dataset.df.index[dataset.df["label"] == 1].tolist()
    nofire_indices = dataset.df.index[dataset.df["label"] == 0].tolist()
    per_class = num_images // 2

    selected = fire_indices[:per_class] + nofire_indices[:per_class]
    if num_images % 2 == 1 and len(fire_indices) > per_class:
        selected.append(fire_indices[per_class])
    return selected[:num_images]


def generate_single(
    model: nn.Module,
    model_name: str,
    freeze_config: str,
    dataset: WildfireDataset,
    indices: List[int],
    output_dir: Path,
) -> None:
    """Generate Grad-CAM overlays for a single model configuration."""
    target_layers = get_target_layers(model, model_name)
    reshape_transform = get_reshape_transform(model_name)

    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform)

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        image_tensor, label = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        rgb_img = denormalise(image_tensor)

        # Get prediction
        with torch.no_grad():
            logits = model(input_tensor)
            pred = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0, pred].item()

        # Generate heatmap for the predicted class
        targets = [ClassifierOutputTarget(pred)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Original (GT: {CLASS_NAMES[label]})")
        axes[0].axis("off")

        axes[1].imshow(grayscale_cam, cmap="jet")
        axes[1].set_title("Activation Map")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Pred: {CLASS_NAMES[pred]} ({confidence:.1%})")
        axes[2].axis("off")

        fig.suptitle(f"{model_name} / {freeze_config}", fontsize=11)
        fig.tight_layout()

        filename = f"gradcam_{idx:04d}_gt{CLASS_NAMES[label]}_pred{CLASS_NAMES[pred]}.png"
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / filename}")

    cam.__del__()


def generate_comparison(
    model_a: nn.Module,
    model_b: nn.Module,
    model_name: str,
    config_a: str,
    config_b: str,
    dataset: WildfireDataset,
    indices: List[int],
    output_dir: Path,
) -> None:
    """Generate side-by-side Grad-CAM comparison between two freeze configs."""
    target_layers_a = get_target_layers(model_a, model_name)
    target_layers_b = get_target_layers(model_b, model_name)
    reshape_transform = get_reshape_transform(model_name)

    cam_a = GradCAM(model=model_a, target_layers=target_layers_a,
                    reshape_transform=reshape_transform)
    cam_b = GradCAM(model=model_b, target_layers=target_layers_b,
                    reshape_transform=reshape_transform)

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in indices:
        image_tensor, label = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        rgb_img = denormalise(image_tensor)

        # Predictions from both models
        with torch.no_grad():
            logits_a = model_a(input_tensor)
            pred_a = logits_a.argmax(dim=1).item()
            conf_a = torch.softmax(logits_a, dim=1)[0, pred_a].item()

            logits_b = model_b(input_tensor)
            pred_b = logits_b.argmax(dim=1).item()
            conf_b = torch.softmax(logits_b, dim=1)[0, pred_b].item()

        # Heatmaps for predicted class
        heatmap_a = cam_a(input_tensor=input_tensor,
                          targets=[ClassifierOutputTarget(pred_a)])[0]
        heatmap_b = cam_b(input_tensor=input_tensor,
                          targets=[ClassifierOutputTarget(pred_b)])[0]

        overlay_a = show_cam_on_image(rgb_img, heatmap_a, use_rgb=True)
        overlay_b = show_cam_on_image(rgb_img, heatmap_b, use_rgb=True)

        # Plot: original | config_a overlay | config_b overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(rgb_img)
        axes[0].set_title(f"Original (GT: {CLASS_NAMES[label]})")
        axes[0].axis("off")

        axes[1].imshow(overlay_a)
        axes[1].set_title(f"{config_a}\n{CLASS_NAMES[pred_a]} ({conf_a:.1%})")
        axes[1].axis("off")

        axes[2].imshow(overlay_b)
        axes[2].set_title(f"{config_b}\n{CLASS_NAMES[pred_b]} ({conf_b:.1%})")
        axes[2].axis("off")

        fig.suptitle(f"{model_name}: {config_a} vs {config_b}", fontsize=11)
        fig.tight_layout()

        filename = f"compare_{idx:04d}_gt{CLASS_NAMES[label]}.png"
        fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {output_dir / filename}")

    cam_a.__del__()
    cam_b.__del__()


def generate_grid(
    model: nn.Module,
    model_name: str,
    freeze_config: str,
    dataset: WildfireDataset,
    indices: List[int],
    output_dir: Path,
) -> None:
    """Generate a single grid image with all Grad-CAM overlays for quick overview."""
    target_layers = get_target_layers(model, model_name)
    reshape_transform = get_reshape_transform(model_name)

    cam = GradCAM(model=model, target_layers=target_layers,
                  reshape_transform=reshape_transform)

    n = len(indices)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, idx in enumerate(indices):
        image_tensor, label = dataset[idx]
        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
        rgb_img = denormalise(image_tensor)

        with torch.no_grad():
            logits = model(input_tensor)
            pred = logits.argmax(dim=1).item()
            conf = torch.softmax(logits, dim=1)[0, pred].item()

        heatmap = cam(input_tensor=input_tensor,
                      targets=[ClassifierOutputTarget(pred)])[0]
        overlay = show_cam_on_image(rgb_img, heatmap, use_rgb=True)

        correct = "correct" if pred == label else "WRONG"
        axes[i].imshow(overlay)
        axes[i].set_title(
            f"GT: {CLASS_NAMES[label]} | Pred: {CLASS_NAMES[pred]}\n"
            f"{conf:.1%} ({correct})",
            fontsize=9,
        )
        axes[i].axis("off")

    # Hide unused axes
    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{model_name} / {freeze_config}", fontsize=13)
    fig.tight_layout()

    grid_path = output_dir / f"gradcam_grid_{freeze_config}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid: {grid_path}")

    cam.__del__()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grad-CAM visualisation for wildfire models")

    p.add_argument("--model", type=str, required=True,
                    choices=["vit", "resnet", "hybrid"])
    p.add_argument("--freeze-config", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-images", type=int, default=10,
                    help="Number of test images to visualise (default: 10)")
    p.add_argument("--no-grid", action="store_true",
                    help="Skip grid overview image")

    # Comparison mode
    p.add_argument("--compare", type=str, default=None,
                    help="Second freeze config for side-by-side comparison")
    p.add_argument("--compare-seed", type=int, default=None,
                    help="Seed for the comparison model (defaults to --seed)")

    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--output-dir", type=str, default=None,
                    help="Output directory (default: results/analysis/<model>/gradcam)")

    default_workers = 0 if platform.system() == "Windows" else 4
    p.add_argument("--num-workers", type=int, default=default_workers)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = results_dir / "analysis" / args.model / "gradcam"

    if args.compare_seed is None:
        args.compare_seed = args.seed

    # Load test dataset with deterministic eval transforms
    dataset = WildfireDataset(split="test", transform=get_eval_transform())
    indices = select_images(dataset, args.num_images, balanced=True)

    print(f"\n  Model: {args.model}")
    print(f"  Config: {args.freeze_config} (seed {args.seed})")
    if args.compare:
        print(f"  Compare: {args.compare} (seed {args.compare_seed})")
    print(f"  Images: {len(indices)}")
    print(f"  Output: {output_dir}\n")

    # Load primary model
    ckpt_path = find_checkpoint(args.model, args.freeze_config, args.seed, results_dir)
    model = build_model(args.model, args.freeze_config, ckpt_path)
    print(f"  Loaded: {ckpt_path}\n")

    if args.compare:
        # Comparison mode
        ckpt_path_b = find_checkpoint(args.model, args.compare, args.compare_seed, results_dir)
        model_b = build_model(args.model, args.compare, ckpt_path_b)
        print(f"  Loaded: {ckpt_path_b}\n")

        compare_dir = output_dir / f"{args.freeze_config}_vs_{args.compare}"
        generate_comparison(
            model, model_b, args.model,
            args.freeze_config, args.compare,
            dataset, indices, compare_dir,
        )
    else:
        # Single config mode
        config_dir = output_dir / args.freeze_config
        generate_single(model, args.model, args.freeze_config,
                        dataset, indices, config_dir)

        if not args.no_grid:
            generate_grid(model, args.model, args.freeze_config,
                          dataset, indices, output_dir)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
