"""Freezing utilities for systematic layer freezing experiments.

Implements progressive freezing configurations for:
    - ViT-B/16:  patch embedding → transformer blocks 0-11
    - ResNet-50: conv1 → layer1 → layer2 → layer3 → layer4
    - Hybrid CNN-ViT: ResNet backbone → conv_proj → transformer blocks
"""

from typing import Dict, List

import torch.nn as nn

from models.vit import ViTClassifier
from models.resnet import ResNetClassifier
from models.hybrid import HybridCNNViT


# ViT-B/16 configs

VIT_CONFIGS = [
    "freeze_none",
    "freeze_patch",
    "freeze_patch_blocks0-3",
    "freeze_patch_blocks0-5",
    "freeze_patch_blocks0-8",
    "freeze_patch_blocks0-11",
]

_VIT_BLOCK_MAP = {
    "freeze_none": -1,
    "freeze_patch": -1,
    "freeze_patch_blocks0-3": 3,
    "freeze_patch_blocks0-5": 5,
    "freeze_patch_blocks0-8": 8,
    "freeze_patch_blocks0-11": 11,
}

# ResNet-50 configs

RESNET_CONFIGS = [
    "freeze_none",
    "freeze_conv1",
    "freeze_conv1_layer1",
    "freeze_conv1_layer1-2",
    "freeze_conv1_layer1-3",
    "freeze_conv1_layer1-4",
]

# Maps config name to which layers to freeze (cumulative)
_RESNET_LAYER_MAP = {
    "freeze_none": [],
    "freeze_conv1": ["conv1", "bn1"],
    "freeze_conv1_layer1": ["conv1", "bn1", "layer1"],
    "freeze_conv1_layer1-2": ["conv1", "bn1", "layer1", "layer2"],
    "freeze_conv1_layer1-3": ["conv1", "bn1", "layer1", "layer2", "layer3"],
    "freeze_conv1_layer1-4": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
}

# Hybrid CNN-ViT configs

HYBRID_CONFIGS = [
    "freeze_none",
    "freeze_backbone",
    "freeze_backbone_proj",
    "freeze_transformer_only",
    "freeze_transformer_proj",
    "freeze_backbone_proj_transformer",
    # Progressive: freeze backbone + proj + first N transformer blocks
    "freeze_backbone_proj_blocks0-3",
    "freeze_backbone_proj_blocks0-5",
    "freeze_backbone_proj_blocks0-8",
    "freeze_backbone_proj_blocks0-11",
    # Progressive: freeze only first N transformer blocks
    "freeze_blocks0-3",
    "freeze_blocks0-5",
    "freeze_blocks0-8",
    "freeze_blocks0-11",
    # BN frozen variants (backbone unfrozen, but BatchNorm layers stay in eval mode)
    "freeze_none_bnfrozen",
    "freeze_transformer_only_bnfrozen",
    "freeze_transformer_proj_bnfrozen",
    "freeze_blocks0-3_bnfrozen",
    "freeze_blocks0-5_bnfrozen",
    "freeze_blocks0-8_bnfrozen",
    "freeze_blocks0-11_bnfrozen",
]

def _get_max_block(config: str) -> int:
    """Extract max block index from config name like 'freeze_blocks0-8' or 'freeze_backbone_proj_blocks0-8'."""
    return int(config.split("blocks0-")[1])

# Registry

FREEZE_CONFIGS = {
    "vit": VIT_CONFIGS,
    "resnet": RESNET_CONFIGS,
    "hybrid": HYBRID_CONFIGS,
}


# Helpers

def _freeze_params(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def _unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


# ViT freezing

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
        f"  [ViT freeze] {config} - "
        f"trainable: {info['trainable']:,} / {info['total']:,} "
        f"({info['trainable_pct']:.1f}%)"
    )


# ResNet freezing

def apply_resnet_freeze(model: ResNetClassifier, config: str) -> None:
    """Apply a named freezing configuration to ResNet-50.

    Progressively freezes from conv1 through layer4.
    The classification head (model.encoder.fc) is never frozen.
    """
    if config not in RESNET_CONFIGS:
        raise ValueError(f"Unknown ResNet freeze config: {config}. Valid: {RESNET_CONFIGS}")

    _unfreeze_all(model)

    layers_to_freeze = _RESNET_LAYER_MAP[config]
    for layer_name in layers_to_freeze:
        module = getattr(model.encoder, layer_name)
        _freeze_params(module)

    info = count_parameters(model)
    print(
        f"  [ResNet freeze] {config} - "
        f"trainable: {info['trainable']:,} / {info['total']:,} "
        f"({info['trainable_pct']:.1f}%)"
    )


# Hybrid freezing

def _freeze_batchnorm(model: HybridCNNViT) -> None:
    """Freeze BatchNorm layers in CNN backbone and override train() to keep them in eval mode."""
    for module in model.backbone.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    # Override train() so model.train() in the training loop doesn't reset BN to train mode
    original_train = model.train
    def train_with_frozen_bn(mode=True):
        original_train(mode)
        if mode:
            for m in model.backbone.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.eval()
        return model
    model.train = train_with_frozen_bn


def _freeze_hybrid_base(model: HybridCNNViT) -> None:
    """Freeze CNN backbone, conv projection, cls token, and positional embedding."""
    _freeze_params(model.backbone)
    _freeze_params(model.conv_proj)
    model.cls_token.requires_grad = False
    model.transformer.pos_embedding.requires_grad = False


def apply_hybrid_freeze(model: HybridCNNViT, config: str) -> None:
    """Apply a named freezing configuration to Hybrid CNN-ViT.

    Freezing configs:
        freeze_none                         - all trainable
        freeze_backbone                          - freeze CNN, train transformer+head
        freeze_backbone_proj                     - above + freeze conv projection
        freeze_transformer_only             - freeze transformer+CLS, train CNN+proj+head
        freeze_transformer_proj             - freeze transformer+CLS+proj, train CNN+head
        freeze_backbone_proj_transformer         - freeze everything except head (linear probe)
        freeze_backbone_proj_blocks0-N           - freeze CNN + proj + first N+1 transformer blocks
    The classifier head (model.classifier) is never frozen.
    """
    if config not in HYBRID_CONFIGS:
        raise ValueError(f"Unknown Hybrid freeze config: {config}. Valid: {HYBRID_CONFIGS}")

    _unfreeze_all(model)

    # Strip _bnfrozen suffix to get the base config
    apply_bn = config.endswith("_bnfrozen")
    base = config.removesuffix("_bnfrozen")

    if base == "freeze_none":
        pass

    elif base == "freeze_backbone":
        _freeze_params(model.backbone)

    elif base == "freeze_backbone_proj":
        _freeze_params(model.backbone)
        _freeze_params(model.conv_proj)

    elif base == "freeze_transformer_only":
        _freeze_params(model.transformer)
        model.cls_token.requires_grad = False

    elif base == "freeze_transformer_proj":
        _freeze_params(model.transformer)
        _freeze_params(model.conv_proj)
        model.cls_token.requires_grad = False

    elif base == "freeze_backbone_proj_transformer":
        _freeze_hybrid_base(model)
        _freeze_params(model.transformer)

    elif "blocks0-" in base:
        if base.startswith("freeze_backbone_proj_"):
            _freeze_hybrid_base(model)
        max_block = _get_max_block(base)
        for i in range(min(max_block + 1, len(model.transformer.layers))):
            _freeze_params(model.transformer.layers[i])

    if apply_bn:
        _freeze_batchnorm(model)

    info = count_parameters(model)
    print(
        f"  [Hybrid freeze] {config} - "
        f"trainable: {info['trainable']:,} / {info['total']:,} "
        f"({info['trainable_pct']:.1f}%)"
    )


# Dispatch

_FREEZE_DISPATCH = {
    "vit": apply_vit_freeze,
    "resnet": apply_resnet_freeze,
    "hybrid": apply_hybrid_freeze,
}


def apply_freeze(model: nn.Module, model_name: str, config: str) -> None:
    """Apply freeze config to any supported model type."""
    fn = _FREEZE_DISPATCH.get(model_name)
    if fn is None:
        raise ValueError(f"Unknown model: {model_name}. Valid: {list(_FREEZE_DISPATCH.keys())}")
    fn(model, config)


# Utilities

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
