"""
Training script for model 
1. CNN
2. ViT
3. Hybrid CNN-ViT
"""

import argparse
import time
from pathlib import Path
from typing import Dict
from tqdm import tqdm
import yaml

 
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
 
from dataset import get_dataloaders, compute_class_weights
from models.vit import ViTClassifier

from models.hybrid import HybridCNNViT
# from models.resnet import ResNetClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


def load_yaml_config(config_path: str) -> Dict:
    """Load config YAML as dictionary. Returns empty dict if path is missing."""
    path = Path(config_path)
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def get_hybrid_settings(args: argparse.Namespace) -> Dict:
    """Resolve hybrid settings from config file with optional hybrid-only CLI overrides."""
    cfg = load_yaml_config(args.hybrid_config)
    hybrid_model_cfg = cfg.get("hybrid_model", {}) if isinstance(cfg, dict) else {}
    hybrid_training_cfg = cfg.get("hybrid_training", {}) if isinstance(cfg, dict) else {}

    settings = {
        "num_classes": args.hybrid_num_classes if args.hybrid_num_classes is not None else hybrid_model_cfg.get("num_classes", args.num_classes),
        "embed_dim": args.hybrid_embed_dim if args.hybrid_embed_dim is not None else hybrid_model_cfg.get("embed_dim"),
        "num_heads": args.hybrid_num_heads if args.hybrid_num_heads is not None else hybrid_model_cfg.get("num_heads"),
        "depth": args.hybrid_depth if args.hybrid_depth is not None else hybrid_model_cfg.get("depth"),
        "dropout_rate": args.hybrid_dropout_rate if args.hybrid_dropout_rate is not None else hybrid_model_cfg.get("dropout_rate", args.dropout),
        "batch_size": args.hybrid_batch_size if args.hybrid_batch_size is not None else hybrid_training_cfg.get("batch_size", args.batch_size),
        "learning_rate": args.hybrid_learning_rate if args.hybrid_learning_rate is not None else hybrid_training_cfg.get("learning_rate", args.lr_head),
        "epochs": args.hybrid_epochs if args.hybrid_epochs is not None else hybrid_training_cfg.get("epochs", args.epochs_phase1),
    }

    required = ["embed_dim", "num_heads", "depth"]
    missing = [key for key in required if settings.get(key) is None]
    if missing:
        raise ValueError(
            f"Missing hybrid settings in {args.hybrid_config}: {missing}. "
            "Provide them in config or via --hybrid-* flags."
        )

    return settings



def build_model(
    args: argparse.Namespace
) -> nn.Module:
        
    if args.model == "vit":
        return ViTClassifier(
            num_classes = 2,
            dropout = 0.1,
            freeze_encoder = True,
        )
    
    elif args.model == "hybrid":
        return HybridCNNViT(
            num_classes = 2,
            embed_dim = args.embed_dim,
            num_heads = args.num_heads,
            depth = args.depth,
            dropout_rate = args.dropout,
        )
    # elif args.model == "resnet":
    # else:
    raise ValueError(f"Unsupported model: {args.model}")
    
def train_one_epoch(
    model : nn.Module,
    loader : torch.utils.data.DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    epoch : int
) -> Dict[str, float]:
    """Train the model for one epoch and return average loss and accuracy."""
    
    # TRAINING mode, initialise params
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # ITERATE OVER BATCHES
    for batch_index, (images, labels) in enumerate(tqdm(loader, desc=f"  Epoch {epoch}")):

        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        # RUNNING LOSS, ACCURACY
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim = 1) == labels).sum().item()
        total += images.size(0)
    
        # LOGGING EVERY 50 BATCHES
        if (batch_index + 1) % 50 == 0:
            print(f"    batch {batch_index + 1}/{len(loader)} — "
                  f"loss: {loss.item():.4f}")
            
        
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
    """Evaluate on val/test set. Returns dict with loss and accuracy."""
    
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # ITERATE OVER BATCHES
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits = model(images)
        loss = criterion(logits, labels)
 
        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim = 1) == labels).sum().item()
        total += images.size(0)
 
    return {
        "val_loss": running_loss / total,
        "val_acc": correct / total,
    }
    
""" EARLY STOPPING (OPTIONAL) """
class EarlyStopping:
    """Stop training when val_loss stops improving."""
 
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


def run_phase(
    tag: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    patience: int,
    model_name: str,
) -> float:
    """Run one training phase. Returns best val_loss."""
    print()
    print(f"  {tag}")
    print()

    early_stop = EarlyStopping(patience = patience)
    best_val_loss = float("inf")
    
    # ITERATE OVER EPOCHS
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
 
        # METRIC LOGGING
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step()
 
        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]
 
        # OUTPUT LOGGING
        print(
            f"  [{tag}] Epoch {epoch:>2}/{num_epochs} — "
            f"train_loss: {train_metrics['train_loss']:.4f}  "
            f"train_acc: {train_metrics['train_acc']:.4f}  "
            f"val_loss: {val_metrics['val_loss']:.4f}  "
            f"val_acc: {val_metrics['val_acc']:.4f}  "
            f"lr: {lr_current:.2e}  "
            f"({elapsed:.1f}s)"
        )
        
        # CHECKPOINT on val_loss improvement
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            ckpt_name = f"{model_name}_{tag.lower().replace(' ', '_')}_best.pt"
            ckpt_path = CHECKPOINT_DIR / ckpt_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": val_metrics["val_acc"],
            }, ckpt_path)
            print(f"    -> saved checkpoint: {ckpt_path}")
        
        if early_stop.step(val_metrics["val_loss"]):
            print(f"    -> early stopping after {epoch} epochs")
            break
    
    return best_val_loss


def train_vit(
    args : argparse.Namespace
) -> None:
    """ TWO PHASE (FROZEN + FINE TUNING) TRAINING FOR VIT """
    print()
    print(f"\n  Device: {DEVICE}")
    print(f"  Model:  ViT-B/16\n")
    
    # DATA
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size = args.batch_size,
        num_workers = args.num_workers,
    )
    
    # CLASS WEIGHTS 
    weights = compute_class_weights(train_loader.dataset).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight = weights)
 
    # MODEL 
    model = build_model(args).to(DEVICE)
    
    """ PHASE 1: FROZEN ENCODER """
    optimizer_p1 = AdamW(
        model.head_params(),
        lr = args.lr_head,
        weight_decay = args.weight_decay,
    )
    
    scheduler_p1 = CosineAnnealingLR(
        optimizer_p1,
        T_max = args.epochs_phase1,
    )
    
    run_phase(
        tag="Phase 1 (frozen)",
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer_p1,
        scheduler = scheduler_p1,
        num_epochs = args.epochs_phase1,
        patience = args.patience,
        model_name="vit",
    )
 
    
    """ PHASE 2: UNFROZEN ENCODER """
    
    model.unfreeze_encoder()
    
    optimizer_p2 = AdamW([
        {"params": model.encoder_params(), "lr": args.lr_backbone},
        {"params": model.head_params(),    "lr": args.lr_head_phase2},
    ], weight_decay = args.weight_decay)
    
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=args.epochs_phase2)

    run_phase(
        tag="Phase 2 (unfrozen)",
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer_p2,
        scheduler = scheduler_p2,
        num_epochs = args.epochs_phase2,
        patience = args.patience,
        model_name="vit",
    )
    
    best_ckpt = CHECKPOINT_DIR / "vit_phase_2_(unfrozen)_best.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        print(f"\n  Loaded best checkpoint (val_loss: {state['val_loss']:.4f})")
 
    test_metrics = evaluate(model, test_loader, criterion)
    print()
    print(f"  TEST — loss: {test_metrics['val_loss']:.4f}  "
          f"acc: {test_metrics['val_acc']:.4f}")
    print()


def train_one_epoch_hybrid(
    model : nn.Module,
    loader : torch.utils.data.DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    epoch : int
) -> Dict[str, float]:
    """Hybrid-specific training epoch wrapper."""
    return train_one_epoch(model, loader, criterion, optimizer, epoch)


@torch.no_grad()
def evaluate_hybrid(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Hybrid-specific evaluation wrapper."""
    return evaluate(model, loader, criterion)


def run_phase_hybrid(
    tag: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    patience: int,
    model_name: str,
) -> float:
    """Run one hybrid training phase. Returns best val_loss."""
    print()
    print(f"  {tag}")
    print()

    early_stop = EarlyStopping(patience = patience)
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch_hybrid(model, train_loader, criterion, optimizer, epoch)
        val_metrics = evaluate_hybrid(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        lr_current = optimizer.param_groups[0]["lr"]

        print(
            f"  [{tag}] Epoch {epoch:>2}/{num_epochs} — "
            f"train_loss: {train_metrics['train_loss']:.4f}  "
            f"train_acc: {train_metrics['train_acc']:.4f}  "
            f"val_loss: {val_metrics['val_loss']:.4f}  "
            f"val_acc: {val_metrics['val_acc']:.4f}  "
            f"lr: {lr_current:.2e}  "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            ckpt_name = f"{model_name}_{tag.lower().replace(' ', '_')}_best.pt"
            ckpt_path = CHECKPOINT_DIR / ckpt_name
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": val_metrics["val_acc"],
            }, ckpt_path)
            print(f"    -> saved checkpoint: {ckpt_path}")

        if early_stop.step(val_metrics["val_loss"]):
            print(f"    -> early stopping after {epoch} epochs")
            break

    return best_val_loss


def _hybrid_encoder_params(model: HybridCNNViT):
    """Return parameters treated as encoder/backbone for hybrid model."""
    return [
        *list(model.backbone.parameters()),
        *list(model.conv_proj.parameters()),
        *list(model.transformer.parameters()),
        model.cls_token,
    ]


def _hybrid_head_params(model: HybridCNNViT):
    """Return parameters treated as head for hybrid model."""
    return model.classifier.parameters()


def freeze_hybrid_encoder(model: HybridCNNViT) -> None:
    """Freeze hybrid encoder/backbone parameters for phase-1 training."""
    for param in _hybrid_encoder_params(model):
        param.requires_grad = False


def unfreeze_hybrid_encoder(model: HybridCNNViT) -> None:
    """Unfreeze hybrid encoder/backbone parameters for fine-tuning."""
    for param in _hybrid_encoder_params(model):
        param.requires_grad = True


def train_hybrid(
    args : argparse.Namespace
) -> None:
    """Two-phase (frozen + fine-tuning) training flow for Hybrid CNN-ViT."""
    hybrid = get_hybrid_settings(args)

    print()
    print(f"\n  Device: {DEVICE}")
    print(f"  Model:  Hybrid CNN-ViT\n")
    print(f"  Hybrid config: {args.hybrid_config}\n")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size = hybrid["batch_size"],
        num_workers = args.num_workers,
    )

    weights = compute_class_weights(train_loader.dataset).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight = weights)

    model = HybridCNNViT(
        num_classes = hybrid["num_classes"],
        embed_dim = hybrid["embed_dim"],
        num_heads = hybrid["num_heads"],
        depth = hybrid["depth"],
        dropout_rate = hybrid["dropout_rate"],
    ).to(DEVICE)

    """ PHASE 1: FROZEN ENCODER """
    freeze_hybrid_encoder(model)

    optimizer_p1 = AdamW(
        _hybrid_head_params(model),
        lr = hybrid["learning_rate"],
        weight_decay = args.weight_decay,
    )

    scheduler_p1 = CosineAnnealingLR(
        optimizer_p1,
        T_max = hybrid["epochs"],
    )

    run_phase_hybrid(
        tag = "Phase 1 (frozen)",
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer_p1,
        scheduler = scheduler_p1,
        num_epochs = hybrid["epochs"],
        patience = args.patience,
        model_name = "hybrid",
    )

    """ PHASE 2: UNFROZEN ENCODER """
    unfreeze_hybrid_encoder(model)

    optimizer_p2 = AdamW([
        {"params": _hybrid_encoder_params(model), "lr": args.lr_backbone},
        {"params": _hybrid_head_params(model),    "lr": args.lr_head_phase2},
    ], weight_decay = args.weight_decay)

    scheduler_p2 = CosineAnnealingLR(
        optimizer_p2,
        T_max = hybrid["epochs"],
    )

    run_phase_hybrid(
        tag = "Phase 2 (unfrozen)",
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer_p2,
        scheduler = scheduler_p2,
        num_epochs = hybrid["epochs"],
        patience = args.patience,
        model_name = "hybrid",
    )

    best_ckpt = CHECKPOINT_DIR / "hybrid_phase_2_(unfrozen)_best.pt"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state["model_state_dict"])
        print(f"\n  Loaded best checkpoint (val_loss: {state['val_loss']:.4f})")

    test_metrics = evaluate_hybrid(model, test_loader, criterion)
    print()
    print(f"  TEST — loss: {test_metrics['val_loss']:.4f}  "
          f"acc: {test_metrics['val_acc']:.4f}")
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wildfire Classification Training")
 
    # MODEL SELECTION
    p.add_argument("--model", type=str, default="vit",
                    choices=["vit", "resnet", "hybrid"])
    p.add_argument("--num-classes", type=int, default=2)
 
    # TRAINING
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-phase1", type=int, default=5,
                    help="Epochs with encoder frozen")
    p.add_argument("--epochs-phase2", type=int, default=20,
                    help="Epochs with full fine-tuning")
    p.add_argument("--patience", type=int, default=5,
                    help="Early stopping patience per phase")
 
    # LEARNING RATES
    p.add_argument("--lr-head", type=float, default=1e-3,
                    help="Head LR — Phase 1")
    p.add_argument("--lr-backbone", type=float, default=5e-6,
                    help="Backbone LR — Phase 2 (ViT is LR-sensitive)")
    p.add_argument("--lr-head-phase2", type=float, default=1e-4,
                    help="Head LR — Phase 2")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--dropout", type=float, default=0.1)
 
    # DATA
    p.add_argument("--num-workers", type=int, default=4)
 

    # HYBRID CONFIG-DRIVEN OPTIONS
    p.add_argument("--hybrid-config", type=str, default="configs/config.yaml",
                    help="Path to hybrid YAML config file")
    p.add_argument("--hybrid-num-classes", type=int, default=None)
    p.add_argument("--hybrid-embed-dim", type=int, default=None)
    p.add_argument("--hybrid-num-heads", type=int, default=None)
    p.add_argument("--hybrid-depth", type=int, default=None)
    p.add_argument("--hybrid-dropout-rate", type=float, default=None)
    p.add_argument("--hybrid-batch-size", type=int, default=None)
    p.add_argument("--hybrid-learning-rate", type=float, default=None)
    p.add_argument("--hybrid-epochs", type=int, default=None)
 
    return p.parse_args()



TRAIN_DISPATCH = {
    "vit": train_vit,
    # "resnet": train_resnet,
    "hybrid": train_hybrid,
}
 
def main() -> None:
    args = parse_args()
    print(f"\n  Config: {vars(args)}\n")
 
    trainer = TRAIN_DISPATCH.get(args.model)
    if trainer is None:
        raise NotImplementedError(f"'{args.model}' training not yet implemented")
    trainer(args)
 
 
if __name__ == "__main__":
    main()