"""Single-run experiment script for ViT-B/16 layer freezing ablation.

Each invocation trains one freeze configuration with one seed.
Logs everything to wandb and saves results as local JSON.

Usage:
    python src/run_experiment.py --model vit --freeze-config freeze_patch_blocks0-3 --seed 42
    python src/run_experiment.py --model vit --freeze-config freeze_none --seed 42 --no-wandb
"""

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset import get_dataloaders, compute_class_weights
from freeze import apply_vit_freeze, count_parameters, get_freeze_configs
from models.vit import ViTClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "vit":
        return ViTClassifier(num_classes=2, dropout=args.dropout, freeze_encoder=False)
    raise ValueError(f"Unsupported model: {args.model}")


def build_optimizer(
    model: nn.Module, args: argparse.Namespace
) -> torch.optim.Optimizer:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters after freezing.")
    if args.optimizer == "adamw":
        return AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return SGD(params, lr=args.lr, momentum=0.9)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"  Epoch {epoch}")):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx + 1}/{len(loader)} — loss: {loss.item():.4f}")

    return {
        "train_loss": running_loss / total,
        "train_acc": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return {
        "val_loss": running_loss / total,
        "val_acc": correct / total,
    }


@torch.no_grad()
def get_test_predictions(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[List[int], List[int]]:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        logits = model(images)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def get_dataset_info(loader: DataLoader) -> Dict[str, int]:
    ds = loader.dataset
    labels = ds.df["label"].values
    fire = int((labels == 1).sum())
    nofire = int((labels == 0).sum())
    return {"total": len(labels), "fire": fire, "nofire": nofire}


def run(args: argparse.Namespace) -> None:
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

    set_seed(args.seed)

    print(f"\n  Device: {DEVICE}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        gpu_name = "cpu"

    print(f"  Model: {args.model}")
    print(f"  Freeze config: {args.freeze_config}")
    print(f"  Seed: {args.seed}")
    print()

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    weights = compute_class_weights(train_loader.dataset).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = build_model(args).to(DEVICE)
    apply_vit_freeze(model, args.freeze_config)
    param_info = count_parameters(model)

    optimizer = build_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stop = EarlyStopping(patience=args.patience)

    dataset_info = {
        "train": get_dataset_info(train_loader),
        "val": get_dataset_info(val_loader),
        "test": get_dataset_info(test_loader),
    }

    run_config = {
        "model": args.model,
        "freeze_config": args.freeze_config,
        "seed": args.seed,
        "device": str(DEVICE),
        "gpu_name": gpu_name,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "epochs_configured": args.epochs,
        "patience": args.patience,
        "num_trainable_params": param_info["trainable"],
        "num_frozen_params": param_info["frozen"],
        "num_total_params": param_info["total"],
        "trainable_pct": round(param_info["trainable_pct"], 2),
        "dataset": dataset_info,
        "class_weights": [round(w, 4) for w in weights.cpu().tolist()],
    }

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            name=f"{args.model}_{args.freeze_config}_seed{args.seed}",
            tags=[args.model, args.freeze_config, "freezing-ablation"],
            config=run_config,
        )

    ckpt_dir = Path(args.output_dir) / "checkpoints" / args.model / args.freeze_config
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"seed_{args.seed}_best.pt"

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    train_history = []
    total_start = time.time()

    print(f"\n  Training — {args.epochs} epochs, patience {args.patience}\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:>2}/{args.epochs} — "
            f"train_loss: {train_metrics['train_loss']:.4f}  "
            f"train_acc: {train_metrics['train_acc']:.4f}  "
            f"val_loss: {val_metrics['val_loss']:.4f}  "
            f"val_acc: {val_metrics['val_acc']:.4f}  "
            f"lr: {lr_current:.2e}  "
            f"({elapsed:.1f}s)"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["train_loss"], 6),
            "train_acc": round(train_metrics["train_acc"], 6),
            "val_loss": round(val_metrics["val_loss"], 6),
            "val_acc": round(val_metrics["val_acc"], 6),
            "lr": lr_current,
            "epoch_time_seconds": round(elapsed, 2),
        }
        train_history.append(epoch_record)

        if use_wandb:
            wandb.log({
                "train/loss": train_metrics["train_loss"],
                "train/acc": train_metrics["train_acc"],
                "val/loss": val_metrics["val_loss"],
                "val/acc": val_metrics["val_acc"],
                "lr": lr_current,
                "epoch_time": elapsed,
                "epoch": epoch,
            })

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_val_acc = val_metrics["val_acc"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "freeze_config": args.freeze_config,
                "seed": args.seed,
            }, ckpt_path)
            print(f"    -> saved checkpoint: {ckpt_path}")

        if early_stop.step(val_metrics["val_loss"]):
            print(f"    -> early stopping after {epoch} epochs")
            break

    total_time = time.time() - total_start
    epochs_completed = len(train_history)

    print(f"\n  Loading best checkpoint (epoch {best_epoch}, val_loss: {best_val_loss:.4f})")
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion)
    all_preds, all_labels = get_test_predictions(model, test_loader)

    from sklearn.metrics import classification_report, confusion_matrix

    report = classification_report(
        all_labels, all_preds,
        target_names=["nofire", "fire"],
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n  TEST — loss: {test_metrics['val_loss']:.4f}  acc: {test_metrics['val_acc']:.4f}")
    print(f"  Fire     — P: {report['fire']['precision']:.3f}  R: {report['fire']['recall']:.3f}  F1: {report['fire']['f1-score']:.3f}")
    print(f"  Nofire   — P: {report['nofire']['precision']:.3f}  R: {report['nofire']['recall']:.3f}  F1: {report['nofire']['f1-score']:.3f}")
    print(f"  Total training time: {total_time:.1f}s\n")

    results = {
        **run_config,
        "epochs_completed": epochs_completed,
        "early_stopped": epochs_completed < args.epochs,
        "test_acc": round(test_metrics["val_acc"], 6),
        "test_loss": round(test_metrics["val_loss"], 6),
        "test_precision_fire": round(report["fire"]["precision"], 4),
        "test_recall_fire": round(report["fire"]["recall"], 4),
        "test_f1_fire": round(report["fire"]["f1-score"], 4),
        "test_precision_nofire": round(report["nofire"]["precision"], 4),
        "test_recall_nofire": round(report["nofire"]["recall"], 4),
        "test_f1_nofire": round(report["nofire"]["f1-score"], 4),
        "test_confusion_matrix": cm.tolist(),
        "best_val_acc": round(best_val_acc, 6),
        "best_val_loss": round(best_val_loss, 6),
        "best_epoch": best_epoch,
        "total_train_time_seconds": round(total_time, 2),
        "train_history": train_history,
    }

    json_dir = Path(args.output_dir) / args.model / args.freeze_config
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"seed_{args.seed}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {json_path}")

    if use_wandb:
        wandb.summary["test/acc"] = test_metrics["val_acc"]
        wandb.summary["test/loss"] = test_metrics["val_loss"]
        wandb.summary["test/f1_fire"] = report["fire"]["f1-score"]
        wandb.summary["test/f1_nofire"] = report["nofire"]["f1-score"]
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["total_train_time"] = total_time

        wandb.log({
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=all_labels,
                preds=all_preds,
                class_names=["nofire", "fire"],
            )
        })
        wandb.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Freezing ablation experiment runner")

    p.add_argument("--model", type=str, default="vit", choices=["vit"])
    p.add_argument("--freeze-config", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)

    default_workers = 0 if platform.system() == "Windows" else 4
    p.add_argument("--num-workers", type=int, default=default_workers)

    p.add_argument("--output-dir", type=str, default="results")

    p.add_argument("--wandb-project", type=str, default="wildfire-freezing")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    valid_configs = get_freeze_configs(args.model)
    if args.freeze_config not in valid_configs:
        print(f"  Invalid freeze config '{args.freeze_config}' for {args.model}.")
        print(f"  Valid configs: {valid_configs}")
        sys.exit(1)
    run(args)

