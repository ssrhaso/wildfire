"""Freezing utilities for systematic layer freezing experiments.

Implements progressive freezing configurations for ViT-B/16.
ResNet-50 and Hybrid freezing to be added by other team members.
"""

from typing import Dict, List

import torch.nn as nn

from models.vit import ViTClassifier


VIT_CONFIGS = [
    "freeze_none",
    "freeze_patch",
    "freeze_patch_blocks0-3",
    "freeze_patch_blocks0-5",
    "freeze_patch_blocks0-8",
    "freeze_patch_blocks0-11",
]

FREEZE_CONFIGS = {
    "vit": VIT_CONFIGS,
}

_VIT_BLOCK_MAP = {
    "freeze_none": -1,
    "freeze_patch": -1,
    "freeze_patch_blocks0-3": 3,
    "freeze_patch_blocks0-5": 5,
    "freeze_patch_blocks0-8": 8,
    "freeze_patch_blocks0-11": 11,
}


def _freeze_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _freeze_vit_patch_embedding(model: ViTClassifier) -> None:
    """Freeze patch projection, class token, and positional embedding."""
    _freeze_params(model.encoder.conv_proj)
    model.encoder.class_token.requires_grad = False
    model.encoder.encoder.pos_embedding.requires_grad = False


def apply_vit_freeze(model: ViTClassifier, config: str) -> None:
    """Apply a named freezing configuration to ViT-B/16.

    First unfreezes everything, then selectively freezes based on config.
    The classification head (model.encoder.heads) is never frozen.
    """
    if config not in VIT_CONFIGS:
        raise ValueError(f"Unknown ViT freeze config: {config}. Valid: {VIT_CONFIGS}")

    _unfreeze_all(model)

    if config == "freeze_none":
        pass

    elif config == "freeze_patch":
        _freeze_vit_patch_embedding(model)

    else:
        _freeze_vit_patch_embedding(model)
        max_block = _VIT_BLOCK_MAP[config]
        for i in range(max_block + 1):
            block = getattr(model.encoder.encoder.layers, f"encoder_layer_{i}")
            _freeze_params(block)

    info = count_parameters(model)
    print(
        f"  [ViT freeze] {config} — "
        f"trainable: {info['trainable']:,} / {info['total']:,} "
        f"({info['trainable_pct']:.1f}%)"
    )


def count_parameters(model: nn.Module) -> Dict[str, float]:
    """Return trainable/frozen/total parameter counts and trainable percentage."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    return {
        "trainable": trainable,
        "frozen": frozen,
        "total": total,
        "trainable_pct": (trainable / total * 100) if total > 0 else 0.0,
    }


def get_freeze_configs(model_name: str) -> List[str]:
    """Return valid freeze config names for a given model."""
    if model_name not in FREEZE_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Valid: {list(FREEZE_CONFIGS.keys())}")
    return FREEZE_CONFIGS[model_name]

