import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class ResNetClassifier(nn.Module):
    """
    ResNet-50 fine-tuned for binary wildfire classification.

    Uses ImageNet-pretrained weights. The original 1000-class head is replaced
    with a lightweight dropout + linear head for binary classification.

    ResNet-50 architecture layers (for freezing reference):
        conv1 + bn1       - initial 7x7 convolution
        layer1             - 3 bottleneck blocks  (stride 1, 256-d)
        layer2             - 4 bottleneck blocks  (stride 2, 512-d)
        layer3             - 6 bottleneck blocks  (stride 2, 1024-d)
        layer4             - 3 bottleneck blocks  (stride 2, 2048-d)
        avgpool + fc       - global pool + classification head
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.0,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        hidden_dim: int = self.encoder.fc.in_features  # 2048

        self.encoder.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        if freeze_encoder:
            self.freeze_encoder()

    def freeze_encoder(self) -> None:
        """Freeze all parameters except the classification head."""
        for name, param in self.encoder.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  [ResNet] Encoder frozen - trainable: {trainable:,} / {total:,}")

    def unfreeze_encoder(self) -> None:
        """Unfreeze all parameters for end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  [ResNet] Encoder unfrozen - trainable: {trainable:,} / {total:,}")

    def encoder_params(self):
        return [p for n, p in self.encoder.named_parameters() if "fc" not in n]

    def head_params(self):
        return list(self.encoder.fc.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


if __name__ == "__main__":
    model = ResNetClassifier(num_classes=2, dropout=0.1, freeze_encoder=True)
    dummy = torch.randn(4, 3, 224, 224)

    logits = model(dummy)
    print(f"  Output shape (frozen): {logits.shape}")
    assert logits.shape == (4, 2)

    model.unfreeze_encoder()
    logits = model(dummy)
    print(f"  Output shape (unfrozen): {logits.shape}")
    assert logits.shape == (4, 2)

    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
