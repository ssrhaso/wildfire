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
 
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders, compute_class_weights, CSV_PATH
from models.vit import ViTClassifier

# from models.hybrid import HybridCNNViT
# from models.resnet import ResNetClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)



""" BUILD MODELS """
def build_model(
    args: argparse.Namespace
) -> nn.Module:
    """ INITIATE MODELS with CLI Usage"""
    
    if args.model == "vit":
        return ViTClassifier(
            num_classes = 2,
            dropout = 0.1,
            freeze_encoder = True,
        )
    
    # elif args.model == "hybrid":
    # elif args.model == "resnet":
    # else:
    
""" TRAINING LOOP """
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
    for batch_index, (images,labels) in enumerate(loader):
        
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
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
    return {
        "loss": epoch_loss, 
        "accuracy": epoch_acc
    }