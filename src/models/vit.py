import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights


class ViTClassifier(nn.Module):
    """
    ViT-B/16 fine-tuned for binary wildfire classification.

    Divides 224x224 input into 196 non-overlapping 16x16 patches, projects each
    to 768-d embeddings, prepends a learnable [CLS] token, and passes the full
    sequence through 12 Transformer encoder layers. The final [CLS] representation
    is fed to the classification head.

    Compared to ResNet-50, self-attention computes global pairwise interactions
    from the first layer - useful for relating spatially separated fire/smoke regions
    that local convolutions would only capture deep in the network.

    Two-phase training: freeze encoder and warm up the head for ~5 epochs, then
    unfreeze with differential LRs (backbone ~5e-6, head ~1e-4). ViT is more
    LR-sensitive than CNNs due to the absence of spatial downsampling.
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.0,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # 768 for ViT-B; read before replacing the head
        hidden_dim: int = self.encoder.heads.head.in_features

        # minimal head - [CLS] from a pre-trained encoder is already structured,
        # a deep MLP risks overfitting on fine-tuning data
        self.encoder.heads = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze all parameters except the classification head (Phase 1)."""
        for name, param in self.encoder.named_parameters():
            if "heads" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  [ViT] Encoder frozen - trainable: {trainable:,} / {total:,}")

    def unfreeze_encoder(self) -> None:
        """Unfreeze all parameters for end-to-end fine-tuning (Phase 2).

        Use differential LRs when constructing the optimiser:
            optimizer = AdamW([
                {"params": model.encoder_params(), "lr": 5e-6},
                {"params": model.head_params(),    "lr": 1e-4},
            ], weight_decay=1e-2)
        """
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  [ViT] Encoder unfrozen - trainable: {trainable:,} / {total:,}")

    def encoder_params(self):
        return [p for n, p in self.encoder.named_parameters() if "heads" not in n]

    def head_params(self):
        return list(self.encoder.heads.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


if __name__ == "__main__":
    model = ViTClassifier(num_classes=2, dropout=0.1, freeze_encoder=True)
    dummy = torch.randn(4, 3, 224, 224)

    logits = model(dummy)
    print(f"  Output shape (Phase 1): {logits.shape}")
    assert logits.shape == (4, 2)

    model.unfreeze_encoder()
    logits = model(dummy)
    print(f"  Output shape (Phase 2): {logits.shape}")
    assert logits.shape == (4, 2)

    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
